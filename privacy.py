"""Decide which columns hold personal data and keep their VALUES away from the AI.

Everything the app sends to Gemini goes through these helpers. The real data stays on the
server and is still used locally for filtering, listing and calculating; the AI only ever
sees column names plus masked or withheld values for the columns detected here.
"""
import re

import pandas as pd

_TOKEN = re.compile(r"[A-Z]+(?=[A-Z][a-z])|[A-Z]?[a-z]+|[A-Z]+|\d+")

# "Student" + one of these (StudentFirstName, StudentID, StudentImportIdentifier, ...)
PERSON = {"student", "pupil", "learner", "child", "guardian", "parent", "carer", "family", "person"}
IDENT = {"name", "first", "last", "given", "sur", "surname", "firstname", "lastname", "middle",
         "preferred", "legal", "id", "identifier", "number", "no", "email", "phone", "mobile",
         "address", "dob", "birth", "birthdate", "username", "login"}
# Personal on their own, whatever the prefix (Surname, Email, DateOfBirth, ...)
ALWAYS = {"surname", "firstname", "lastname", "givenname", "fullname", "email", "phone", "mobile",
          "dob", "birth", "birthdate", "birthday", "address", "username", "medicare", "passport"}
BIGRAMS = {("first", "name"), ("last", "name"), ("given", "name"), ("full", "name"),
           ("middle", "name"), ("preferred", "name"), ("legal", "name"), ("family", "name")}
# Free-text columns routinely contain names
FREE_TEXT = {"comment", "comments", "feedback", "remark", "remarks", "note", "notes"}
EXACT = {("name",), ("id",), ("fullname",)}

_EMAIL = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


def tokens(col) -> list[str]:
    return [t.lower() for t in _TOKEN.findall(str(col))]


def _name_sensitive(col) -> bool:
    t = tokens(col)
    ts = set(t)
    if tuple(t) in EXACT or ts & ALWAYS or ts & FREE_TEXT:
        return True
    if any(b in set(zip(t, t[1:])) for b in BIGRAMS):
        return True
    return bool(ts & PERSON and ts & IDENT)


def _content_sensitive(s: pd.Series) -> bool:
    """Catches personal data hiding under an innocent column name."""
    if pd.api.types.is_numeric_dtype(s):
        return False
    v = s.dropna().astype(str).head(300)
    if v.empty:
        return False
    if v.map(lambda x: bool(_EMAIL.match(x))).mean() >= 0.5:
        return True
    return v.str.len().mean() >= 60  # long text = comments


def detect_sensitive(df: pd.DataFrame, hide=(), allow=()) -> set:
    """Columns whose values must never be sent to the AI.
    hide: extra column names to always hide. allow: names to never hide (overrides detection)."""
    hide_l = {str(h).strip().lower() for h in hide}
    allow_l = {str(a).strip().lower() for a in allow}
    out = set()
    for c in df.columns:
        key = str(c).strip().lower()
        if key in allow_l:
            continue
        if key in hide_l or _name_sensitive(c) or _content_sensitive(df[c]):
            out.add(c)
    return out


def safe_profile(prof: list[dict], hidden: set) -> list[dict]:
    """Schema for the AI: hidden columns keep their name and type but lose every value and statistic."""
    out = []
    for p in prof:
        if p["name"] in hidden:
            out.append({"name": p["name"], "dtype": p["dtype"], "hidden": True,
                        "note": "personal data; values withheld. You can still filter on it using "
                                "values from the question."})
        else:
            out.append(p)
    return out


def safe_sample(df: pd.DataFrame, hidden: set, n: int = 3) -> str:
    """A few example rows for the AI with the hidden columns removed entirely."""
    return df.drop(columns=[c for c in df.columns if c in hidden]).head(n).to_csv(index=False)


def mask_frame(df: pd.DataFrame, hidden: set) -> pd.DataFrame:
    """Copy with hidden columns replaced by stable pseudonyms (Column-001, Column-002, ...)."""
    out = df.copy()
    for c in df.columns:
        if c in hidden:
            codes, _ = pd.factorize(out[c])
            out[c] = [f"{c}-{k + 1:03d}" if k >= 0 else None for k in codes]
    return out


# ---------------------------------------------------------------------------
# Name scrubbing: names/IDs typed into a question are swapped for placeholders
# BEFORE the question goes to the AI, then swapped back for local filtering.
# ---------------------------------------------------------------------------
_WORD = re.compile(r"[\w'’\-]+(?:[@.][\w'’\-]+)*")
# Common words that are also first names; never treated as a name on their own.
COMMON_WORDS = {"mark", "marks", "will", "may", "hope", "grace", "rose", "faith", "april", "june", "august",
                "summer", "autumn", "joy", "max", "ray", "art", "result", "results", "average", "class",
                "year", "task", "term", "student", "students", "teacher", "grade", "score", "scores",
                "test", "exam", "show", "list", "compare", "all", "the", "and", "for", "with"}


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s.replace("’", "'").strip().lower())


def collect_values(df: pd.DataFrame, hidden: set) -> set:
    """Lower-cased values of the personal columns (names, IDs...) used to spot names in questions."""
    out = set()
    for c in hidden:
        if c not in df.columns:
            continue
        for v in df[c].dropna().astype(str).unique():
            k = _norm(v)
            if k.endswith(".0") and k[:-2].isdigit():
                k = k[:-2]
            if not k or len(k) > 40:
                continue  # empty or free text, not a name
            if k.isdigit():
                if len(k) >= 4:
                    out.add(k)
            elif len(k) >= 3 or " " in k:
                out.add(k)
    return out


def vocab_words(entries: list[dict]) -> set:
    """Words from file names and column names; these are never treated as student names."""
    out = set()
    for e in entries:
        for text in [e.get("label", "")] + list(e.get("columns") or []):
            out |= set(tokens(text)) | set(re.findall(r"[a-z]+", str(text).lower()))
    return out


def scrub(text: str, index: set, skip=frozenset()) -> tuple[str, dict]:
    """Replace known names/IDs in `text` with [STUDENT_n]. Returns (scrubbed, {placeholder: original})."""
    toks = [(m.start(), m.end()) for m in _WORD.finditer(text)]
    out, mapping, seen, pos, i = [], {}, {}, 0, 0
    skip = set(skip) | COMMON_WORDS
    while i < len(toks):
        hit = None
        for n in (4, 3, 2, 1):
            if i + n > len(toks):
                continue
            end = toks[i + n - 1][1]
            span = text[toks[i][0]:end]
            key = _norm(span)
            if key not in index and key.endswith("'s") and key[:-2] in index:  # possessive: "Jane's"
                key, end, span = key[:-2], end - 2, span[:-2]
            if key in index and not (n == 1 and key in skip):
                hit = (n, key, span, end)
                break
        if not hit:
            i += 1
            continue
        n, key, span, end = hit
        ph = seen.setdefault(key, f"[STUDENT_{len(seen) + 1}]")
        mapping[ph] = span
        out += [text[pos:toks[i][0]], ph]
        pos = end
        i += n
    out.append(text[pos:])
    return "".join(out), mapping


def unscrub(text: str, mapping: dict, json_safe: bool = False) -> str:
    """Put real values back (for local use / display only). json_safe escapes them for JSON strings."""
    import json as _json
    for ph, orig in mapping.items():
        rep = _json.dumps(orig)[1:-1] if json_safe else orig
        text = re.sub(re.escape(ph), lambda _m, r=rep: r, text, flags=re.I)
    return text


# ---------------------------------------------------------------------------
# "Those students": remember who a previous answer listed, locally, so follow-ups
# can be restricted to them without the AI ever seeing who they are.
# ---------------------------------------------------------------------------
ID_TOKENS = {"id", "identifier", "number", "no"}
NAME_TOKENS = {"name", "first", "last", "given", "surname", "firstname", "lastname", "middle", "preferred"}


def cohort_from(df: pd.DataFrame, result: pd.DataFrame, hidden: set):
    """Identity columns of the students in a plain listing (rows still carry their source-row index)."""
    cols = [c for c in df.columns if c in hidden]
    if not cols or result is None or len(result) == 0 or not result.index.isin(df.index).all():
        return None
    return df.loc[result.index, cols].drop_duplicates()


def _match_columns(cohort_cols, df_cols) -> list:
    shared = [c for c in cohort_cols if c in df_cols]
    ids = [c for c in shared if set(tokens(c)) & ID_TOKENS]
    return ids or [c for c in shared if set(tokens(c)) & NAME_TOKENS]


def restrict_to_cohort(df: pd.DataFrame, cohort: pd.DataFrame):
    """Rows of df belonging to the remembered students. Matches on an ID column if both files have one,
    otherwise on name columns. Returns (rows, columns_used)."""
    keys = _match_columns(list(cohort.columns), list(df.columns))
    if not keys:
        raise ValueError("I can't match those students to this data: it has no student ID or name column "
                         "in common with the earlier list.")

    def key(frame):
        f = frame[keys].dropna()
        k = f.astype(str).apply(lambda s: s.str.strip().str.lower().str.replace(r"\.0$", "", regex=True))
        return k.agg("|".join, axis=1)

    return df.loc[key(df).isin(set(key(cohort))).reindex(df.index, fill_value=False)], keys

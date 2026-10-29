import hmac
import json
import re
import time
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import pandas as pd
import streamlit as st
from google import genai
from google.genai import errors, types

import analysis
import drive_sources as ds
import freshness
import privacy

# Catch out-of-date files up front (a common slip when updating files one at a time on GitHub).
REQUIRED_VERSION = 6
_stale = [n for n, m in [("analysis.py", analysis), ("privacy.py", privacy),
                         ("drive_sources.py", ds), ("freshness.py", freshness)]
          if getattr(m, "VERSION", 0) != REQUIRED_VERSION]
if _stale:
    st.error("These files in your repo are out of date: **" + ", ".join(_stale) + "**. Replace each with the "
             "latest copy (same folder as app.py, exact filenames), commit, then Reboot app.")
    st.stop()

from analysis import clean_frame, profile, run_plan

# ==========================================
# 1. SECURITY & AUTHENTICATION
# ==========================================
def check_password() -> bool:
    if st.session_state.get("password_correct"):
        return True

    st.title("🔒 Staff Data Portal")
    pw = st.text_input("Enter Staff Password", type="password")
    if st.button("Log In"):
        if hmac.compare_digest(pw, st.secrets["passwords"]["staff_password"]):
            st.session_state["password_correct"] = True
            st.rerun()
        else:
            st.error("Incorrect password.")
    return False


if not check_password():
    st.stop()

# ==========================================
# 2. CONFIG & AI SETUP
# ==========================================
st.set_page_config(page_title="Student Insights Assistant", layout="wide")

client = genai.Client(api_key=st.secrets["api_keys"]["gemini_key"],
                      http_options=types.HttpOptions(timeout=25000))  # ms: never hang the page
MODEL_NAME = st.secrets.get("ai", {}).get("model", "gemini-flash-latest")
FALLBACK_MODEL = st.secrets.get("ai", {}).get("fallback_model", "gemini-flash-lite-latest")
_priv = dict(st.secrets.get("privacy", {}))
HIDE_COLS = list(_priv.get("hide_columns", []))    # always keep these away from the AI
ALLOW_COLS = list(_priv.get("allow_columns", []))  # never hide these (overrides detection)
TZ = ZoneInfo(st.secrets.get("app", {}).get("timezone", "Australia/Melbourne"))


def call_model(prompt: str, as_json: bool = True) -> str:
    """Call Gemini, retrying temporary overloads, then falling back to a lighter model."""
    config = types.GenerateContentConfig(
        temperature=0 if as_json else 0.3,
        **({"response_mime_type": "application/json"} if as_json else {}),
    )
    transient = {429, 500, 502, 503, 504}
    last_err = None
    for model_name in dict.fromkeys([MODEL_NAME, FALLBACK_MODEL]):
        for attempt in range(2):
            try:
                return client.models.generate_content(
                    model=model_name, contents=prompt, config=config).text
            except errors.APIError as e:
                last_err = e
                if getattr(e, "code", None) not in transient:
                    raise
                time.sleep(2 ** attempt)
            except Exception as e:  # network timeouts etc.
                if "timeout" not in type(e).__name__.lower():
                    raise
                last_err = e
    raise last_err


def extract_json(raw: str) -> dict:
    start, end = raw.find("{"), raw.rfind("}")
    if start == -1 or end == -1:
        raise ValueError("The AI didn't return valid JSON.")
    return json.loads(raw[start:end + 1])


# ==========================================
# 3. DATA CATALOG & LOADING (links and/or private Drive folders)
# ==========================================
link_sheets = dict(st.secrets.get("sheets", {}))
folders = dict(st.secrets.get("folders", {}))
MAX_HEADER_PEEKS = 60   # how many Google Sheets get their column names read for routing
MAX_COMBINED = 100      # most files combined for one question


def to_csv_url(url: str) -> str:
    m = re.search(r"/spreadsheets/d/([a-zA-Z0-9-_]+)", url)
    if not m:
        raise ValueError(f"Not a valid Google Sheets URL: {url}")
    gid = re.search(r"[#&?]gid=(\d+)", url)
    out = f"https://docs.google.com/spreadsheets/d/{m.group(1)}/export?format=csv"
    return out + (f"&gid={gid.group(1)}" if gid else "")


@st.cache_resource
def get_services():
    if "gcp_service_account" not in st.secrets:
        raise RuntimeError("Add a [gcp_service_account] section to your Streamlit secrets to use Drive folders.")
    return ds.build_services(dict(st.secrets["gcp_service_account"]))


@st.cache_data(ttl=300, show_spinner="Checking what data is available...")
def build_catalog() -> tuple[list[dict], list[str]]:
    """Lightweight list of every available file (name, edit time, column names). No data is loaded."""
    entries, problems = [], []
    for name, url in link_sheets.items():
        try:
            cols = [str(c) for c in pd.read_csv(to_csv_url(url), nrows=0).columns]
        except Exception as e:
            cols = None
            problems.append(f"{name}: {type(e).__name__}")
        entries.append(dict(id=f"link:{name}", label=name, source="Sheet link", kind="link",
                            key=url, mime="", modified=None, columns=cols))
    if folders:
        try:
            drive, sheets = get_services()
        except Exception as e:
            problems.append(f"Drive folders unavailable: {e}")
        else:
            peeks = 0
            for fname, ref in folders.items():
                try:
                    files = ds.list_files(drive, ds.folder_id_from(ref))
                except Exception as e:
                    problems.append(f"Folder '{fname}': {type(e).__name__}: {e}")
                    continue
                for f in files:
                    cols = None
                    if f["mimeType"] == ds.GSHEET and peeks < MAX_HEADER_PEEKS:
                        peeks += 1
                        try:
                            cols = ds.peek_columns(sheets, f["id"])
                        except Exception:
                            pass
                    entries.append(dict(id=f"drive:{f['id']}", label=f["path"], source=fname, kind="drive",
                                        key=f["id"], mime=f["mimeType"], modified=f["modifiedTime"], columns=cols))
    return entries, problems


# cache_resource (not cache_data) so big DataFrames aren't copied on every question.
# They are treated as read-only everywhere.
@st.cache_resource(ttl=600, show_spinner="Loading data...")
def load_one(kind: str, key: str, mime: str, modified: str):
    if kind == "link":
        raw = pd.read_csv(to_csv_url(key))
    else:
        drive, sheets = get_services()
        raw = ds.read_file(drive, sheets, {"id": key, "mimeType": mime})
    df = clean_frame(raw)
    return df, datetime.now(timezone.utc).isoformat(), profile(df)


@st.cache_resource(ttl=600, show_spinner="Combining files...")
def combine(parts: tuple):
    """parts: tuple of (kind, key, mime, modified, label). Adds a SourceFile column."""
    frames = []
    for kind, key, mime, modified, label in parts:
        d, _, _ = load_one(kind, key, mime, modified)
        d = d.copy()
        d.insert(0, "SourceFile", label)
        frames.append(d)
    df = clean_frame(pd.concat(frames, ignore_index=True, sort=False))
    return df, profile(df)


def load_selection(chosen: list[dict]):
    """Load the chosen catalog entries. Returns (df, profile, metas, skipped)."""
    ok, metas, skipped = [], [], []
    for e in chosen[:MAX_COMBINED]:
        try:
            d, fetched, _ = load_one(e["kind"], e["key"], e["mime"], e["modified"] or "")
        except Exception as ex:
            skipped.append(f"{e['label']}: {type(ex).__name__}")
            continue
        if d.empty:
            skipped.append(f"{e['label']}: empty")
            continue
        ok.append(e)
        metas.append({"label": e["label"], "modified": e["modified"], "fetched_at": fetched})
    if len(chosen) > MAX_COMBINED:
        skipped.append(f"{len(chosen) - MAX_COMBINED} more files not loaded (limit {MAX_COMBINED}).")
    if not ok:
        raise ValueError("None of the matching files could be read.")
    if len(ok) == 1:
        e = ok[0]
        df, _, prof = load_one(e["kind"], e["key"], e["mime"], e["modified"] or "")
    else:
        df, prof = combine(tuple((e["kind"], e["key"], e["mime"], e["modified"] or "", e["label"]) for e in ok))
    return df, prof, metas, skipped


entries, catalog_problems = build_catalog()
if not entries:
    st.error("No data found. Add a [sheets] and/or [folders] section to your Streamlit secrets, "
             "and check the folder is shared with the service account.")
    for p in catalog_problems:
        st.warning(p)
    st.stop()


@st.cache_resource(ttl=600, show_spinner="Preparing privacy protections (first question only)...")
def get_name_index(sig: tuple):
    """Every student name/ID across all files, so names in a question can be masked before routing."""
    idx, failed = set(), []
    for e in entries[:MAX_COMBINED]:
        try:
            d, _, _ = load_one(e["kind"], e["key"], e["mime"], e["modified"] or "")
        except Exception:
            failed.append(e["label"])
            continue
        idx |= privacy.collect_values(d, privacy.detect_sensitive(d, HIDE_COLS, ALLOW_COLS))
    return idx, failed, max(0, len(entries) - MAX_COMBINED)


def catalog_text(items: list[dict]) -> str:
    limit = 30 if len(items) <= 60 else 12
    lines = []
    for e in items:
        cols = ", ".join(c[:25] for c in (e["columns"] or [])[:limit]) or "(columns unknown)"
        mod = freshness.parse(e["modified"])
        edited = freshness.when(mod, TZ) if mod else "unknown"
        lines.append(f"{e['id']} | {e['label']} | {e['source']} | edited {edited} | {cols}")
    return "\n".join(lines)


# ==========================================
# 4. PROMPTS
# ==========================================
ROUTER_PROMPT = """You choose which data files are needed to answer a staff question about school data.

Files (id | name | source | last edited | columns):
{catalog}

Files used for the previous question in this conversation: {previous}

Question: {question}

Respond with JSON only: {{"dataset_ids": ["id", ...] or "all", "reason": "short phrase"}}
Rules:
- Choose the fewest files that can answer the question (maximum 10).
- Use "all" only if the question is clearly about every file or the whole school across files.
- For a follow-up question, keep using the previous files unless the question points elsewhere.
- If it asks for different information (for example assessments or behaviours rather than attendance),
  choose the file(s) whose columns contain that information, even if the previous question used another file.
- File names often contain the year level, subject or term; match on those and on column names.
- If it's ambiguous, pick the most plausible file(s) and say so in "reason".
"""

PLAN_PROMPT = """You turn a staff question about a school data sheet into a JSON analysis plan.
Only use columns from the schema. Columns marked "hidden" hold personal data whose values you cannot
see; you can still filter or group by them (use values from the question), and you must never ask for
those values. If several files were combined, a "SourceFile" column says which
file each row came from. Respond with JSON only:
{{
  "intent": "table" | "aggregate" | "distribution",
  "filters": [{{"column": str, "op": "==|!=|>|>=|<|<=|contains|in|isnull|notnull", "value": any}}],
  "focus": [same filter shape],
  "focus_label": "short name of the group of interest, e.g. a teacher code",
  "measure": column holding the marks/results/scores, or null,
  "by": column to break the measure down by (e.g. task/assessment name), or null,
  "group_by": [str],
  "aggregate": {{"column": str, "func": "mean|median|sum|count|min|max"}} or null,
  "select_columns": [str],
  "sort_by": str or null,
  "ascending": bool,
  "limit": int or null,
  "use_previous_students": true | false,
  "note": "one line on how you interpreted the question"
}}

Choose the intent:
- "distribution": questions about spread, distribution, consistency, outliers, fairness, patterns,
  or comparing a teacher/class/group with others, or anything asking for a plot. Put the group of
  interest in "focus" and any context restriction shared by everyone (e.g. a subject or year) in
  "filters". Set "measure" to the mark/result column. Set "by" to the task/assessment column so
  the group is compared with peers on the same tasks. The analysis computes histograms, per-task
  comparisons and consistency flags itself.
- "aggregate": counts, averages or totals per group.
- "table": listing rows.
For "aggregate" and "table", put all conditions in "filters" (leave "focus" empty).
Use the conversation so far to resolve pronouns such as "his", "that class", "them".
Set "use_previous_students" to true when the question refers to the students from the previous answer
("those students", "them", "their", "those 5"). You are never shown who they are; the app restricts the
data to them for you, so do not add filters to identify them.
When listing students, include their name/ID columns in "select_columns" so staff can see who they are
(you can't see the values, but the columns exist).
Never answer with an error because of ambiguity: choose the most plausible reading and say so in "note".
Only return {{"error": "reason"}} if the schema genuinely cannot answer the question.

Conversation so far:
{history}

Schema:
{schema}

Sample rows:
{sample}

Question: {question}
"""

WRAPUP_PROMPT = """You are a data analyst assistant for school staff. Use only numbers that appear below; never invent figures.

Question: {question}
Analysis type: {intent}
Columns available: {columns}

Computed results:
{summary}

Respond with JSON only: {{"commentary": "...", "followups": ["...", "...", "..."]}}

commentary: {commentary_rule}
followups: exactly 3 short follow-up questions staff could ask next. Write each as a natural chat message
that makes sense on its own (name the group, file or column; never say "it" or "them"), answerable from
the available columns, and varied: one to drill down, one to compare, one to check data quality or consistency.
"""

NARRATE_RULE = ("3 to 6 short markdown bullet points: what the results show, and anything inconsistent or "
                "worth a closer look, each backed by a specific number. Then one final line of caveats "
                "(for example small samples, marks possibly on different scales across tasks, peers possibly "
                "teaching different cohorts). Frame findings as 'worth a look', never as accusations about a "
                "person. Plain English, no preamble, no headings.")

def get_starters(items: list[dict]) -> list[str]:
    """Example questions built locally from file names. No AI call, so the page never waits on Gemini."""
    labels = [e["label"].split("/")[-1] for e in items[:2]]
    a_, b_ = labels[0], labels[-1]
    return [
        f"How are the results distributed in {a_}, and is anything inconsistent?",
        f"What's the average result per task in {b_}?",
        f"Which students have a missing result in {a_}?",
        "Compare results across teachers.",
    ]


def history_text(turns: list[dict]) -> str:
    """Previous questions for follow-ups. Uses the scrubbed versions, never the names as typed."""
    lines = [f"Q: {t.get('question_ai') or t['question']} -> {t.get('note_ai') or t['plan'].get('note', '')}"
             for t in turns[-3:] if t.get("plan")]
    return "\n".join(lines) or "(none yet)"


def choose_entries(question: str, history: list[dict]) -> tuple[list[dict], str]:
    if len(entries) == 1:
        return entries, ""
    by_id = {e["id"]: e for e in entries}
    prev_ids = next((t["entry_ids"] for t in reversed(history) if t.get("entry_ids")), [])
    prev = ", ".join(prev_ids) or "(none)"
    r = extract_json(call_model(ROUTER_PROMPT.format(
        catalog=catalog_text(entries), previous=prev, question=question)))
    ids = r.get("dataset_ids")
    if ids == "all" or (isinstance(ids, list) and "all" in ids):
        chosen = list(entries)
    else:
        chosen = [by_id[i] for i in (ids if isinstance(ids, list) else [ids]) if i in by_id][:10]
    if not chosen:  # never guess: analysing the wrong file is worse than asking
        names = ", ".join(e["label"] for e in entries[:20])
        raise ValueError("I couldn't tell which data you meant. Try mentioning a file, year level or "
                         f"subject. Available: {names}")
    return chosen, str(r.get("reason", ""))


def wrapup(question: str, plan: dict, out: dict, columns: list[str]) -> tuple[str, list[str]]:
    """One call returns commentary (for analyses) and 3 follow-up ideas. Never fails the answer."""
    intent = plan.get("intent", "table")
    narrate = intent in ("distribution", "aggregate") and out["summary"]
    if narrate:
        summary, rule = out["summary"], NARRATE_RULE
    else:  # listings: send only the shape, never student rows
        summary = f"A table of {len(out['result'])} rows with columns: {', '.join(map(str, out['result'].columns))}."
        rule = 'an empty string ("").'
    try:
        r = extract_json(call_model(WRAPUP_PROMPT.format(
            question=question, intent=intent, columns=", ".join(columns[:60]),
            summary=summary, commentary_rule=rule)))
        follow = [x for x in r.get("followups", []) if isinstance(x, str)][:3]
        return (r.get("commentary") or "") if narrate else "", follow
    except Exception:
        return "", []


# ==========================================
# 5. SIDEBAR (minimal)
# ==========================================
def ask(q: str):
    st.session_state["pending_q"] = q


with st.sidebar:
    st.header("Data")
    st.caption(f"{len(entries)} file(s) available. Just ask; the assistant picks the right one.")
    st.caption("🔒 Student names, IDs and similar personal columns are detected automatically and "
               "never sent to the AI. Names typed into questions are swapped for placeholders first "
               "(misspelled names may not be caught).")
    with st.expander("What's available"):
        for e in entries:
            mod = freshness.parse(e["modified"])
            now_ = datetime.now(timezone.utc)
            when_ = f"edited {freshness.ago(mod, now_)}" if mod else "edit time n/a"
            st.markdown(f"- **{e['label']}** · {when_}")
    for p in catalog_problems:
        st.warning(p)
    if st.button("🔄 Refresh data"):
        st.cache_data.clear()
        st.cache_resource.clear()
        st.rerun()
    if st.button("🧹 Clear conversation"):
        st.session_state["history"] = []
        st.session_state.pop("cohort", None)
        st.rerun()

# ==========================================
# 6. CHAT
# ==========================================
st.title("💬 Student Insights Assistant")

history: list[dict] = st.session_state.setdefault("history", [])

if not history:
    st.write("Ask anything about your school data. Try one of these:")
    for k, s in enumerate(get_starters(entries)):
        st.button(s, key=f"starter_{k}", on_click=ask, args=(s,))

for i, turn in enumerate(history):
    with st.chat_message("user"):
        st.write(turn["question"])
    with st.chat_message("assistant"):
        if turn.get("error"):
            st.error(turn["error"])
            if turn.get("raw"):
                with st.expander("Raw AI response"):
                    st.code(turn["raw"])
        else:
            out, plan, fresh = turn["out"], turn["plan"], turn["fresh"]
            used = ", ".join(m["label"] for m in turn["metas"][:3])
            if len(turn["metas"]) > 3:
                used += f" and {len(turn['metas']) - 3} more"
            st.caption(f"📂 Using: {used}" + (f" ({turn['reason']})" if turn.get("reason") else ""))
            st.caption(fresh["headline"])
            if turn.get("hidden"):
                st.caption("🔒 Kept from the AI (values never sent): " + ", ".join(turn["hidden"][:8])
                           + (f" +{len(turn['hidden']) - 8} more" if len(turn["hidden"]) > 8 else ""))
            if turn.get("masked"):
                st.caption(f"🔒 {turn['masked']} student name/ID in your question was replaced with a "
                           "placeholder before reaching the AI.")
            if turn.get("privacy_note"):
                st.caption("⚠️ " + turn["privacy_note"])
            if turn.get("cohort_note"):
                st.caption("👥 " + turn["cohort_note"])
            st.caption(f"Interpreted as: {plan.get('note', '')}")
            if turn.get("narrative"):
                st.markdown(turn["narrative"])
            for n in out["notes"]:
                st.caption(f"ℹ️ {n}")
            for j, fig in enumerate(out["figs"]):
                st.plotly_chart(fig, use_container_width=True, key=f"fig_{i}_{j}")

            if plan.get("intent", "table") == "table" or not out["figs"]:
                st.write(f"**{len(out['result'])} result(s)**")
                if len(out["result"]) == 0:
                    st.info("No matching rows. Open 'Details' to see the query plan; the file may not hold this "
                            "information, or those students may not appear in it.")
                st.dataframe(out["result"], use_container_width=True)
            with st.expander("Details: data sources, tables, flags and query plan"):
                st.markdown("**Data sources & timeliness**")
                st.dataframe(pd.DataFrame(fresh["rows"]), use_container_width=True)
                if turn.get("skipped"):
                    st.markdown("**Skipped files**\n\n" + "\n".join(f"- {x}" for x in turn["skipped"]))
                if out["flags"]:
                    st.markdown("**Automatic flags**\n\n" + "\n".join(f"- {f}" for f in out["flags"]))
                if out["stats"] is not None:
                    st.dataframe(out["stats"], use_container_width=True)
                if out["levels"] is not None:
                    st.dataframe(out["levels"], use_container_width=True)
                if plan.get("intent", "table") != "table" and out["figs"]:
                    st.dataframe(out["result"], use_container_width=True)
                st.json(plan)
            if len(out["result"]):
                st.download_button("Download table (CSV)", out["result"].to_csv(index=False),
                                   f"result_{i}.csv", key=f"dl_{i}")

            if i == len(history) - 1 and turn.get("followups"):
                st.markdown("**Ideas to explore next**")
                for k, f in enumerate(turn["followups"]):
                    st.button(f, key=f"fu_{i}_{k}", on_click=ask, args=(f,))

question = st.chat_input("Ask about your school data...") or st.session_state.pop("pending_q", None)

if question:
    turn, raw = {"question": question}, None
    with st.spinner("Working on it..."):
        try:
            # Step 0: mask names/IDs in the question BEFORE any AI call (routing included).
            index, failed, over = get_name_index(tuple((e["id"], e["modified"] or "") for e in entries))
            q_ai, name_map = privacy.scrub(question, index, privacy.vocab_words(entries))
            turn.update(question_ai=q_ai, masked=len(name_map))
            if failed or over:
                n = len(failed) + over
                turn["privacy_note"] = (f"{n} file(s) couldn't be checked for student names, so names that "
                                        "only appear in those files weren't masked.")

            chosen, reason = choose_entries(q_ai, history)
            df, prof, metas, skipped = load_selection(chosen)
            hidden = privacy.detect_sensitive(df, HIDE_COLS, ALLOW_COLS)
            turn.update(hidden=sorted(map(str, hidden)), entry_ids=[e["id"] for e in chosen], reason=reason,
                        metas=metas, skipped=skipped, fresh=freshness.summarize(metas, TZ))

            raw = call_model(PLAN_PROMPT.format(
                history=history_text(history),
                schema=json.dumps(privacy.safe_profile(prof, hidden), default=str),
                sample=privacy.safe_sample(df, hidden),
                question=q_ai,
            ))
            plan_ai = extract_json(raw)                      # what the AI said (placeholders)
            turn["note_ai"] = str(plan_ai.get("note", ""))
            plan = extract_json(privacy.unscrub(raw, name_map, json_safe=True))  # real values, local use only
            if "error" in plan:
                turn["error"] = privacy.unscrub(str(plan["error"]), name_map)
            else:
                turn["plan"] = plan
                run_df = df
                if plan.get("use_previous_students"):
                    cohort = st.session_state.get("cohort")
                    if cohort is None:
                        raise ValueError("I don't have a list of students from earlier in this chat. Ask for the "
                                         "students first (for example 'list 5 students with ...'), then ask about them.")
                    run_df, keys = privacy.restrict_to_cohort(df, cohort)
                    turn["cohort_note"] = (f"Limited to the {len(cohort)} student(s) from your earlier list "
                                           f"(matched on {', '.join(keys)}); {len(run_df):,} matching row(s) in this data.")
                turn["out"] = run_plan(run_df, plan, hidden)
                # Remember who a plain listing showed, so "those students" works next time.
                if (not plan.get("use_previous_students") and plan.get("intent", "table") == "table"
                        and not plan.get("aggregate")):
                    c = privacy.cohort_from(df, turn["out"]["result"], hidden)
                    if c is not None and len(c):
                        st.session_state["cohort"] = c
                narr, follow = wrapup(q_ai, plan_ai, turn["out"], list(df.columns))
                turn["narrative"] = privacy.unscrub(narr, name_map)
                turn["followups"] = [privacy.unscrub(f, name_map) for f in follow]
        except Exception as e:
            turn["error"] = f"{type(e).__name__}: {e}"
            turn["raw"] = None  # raw AI text may hold placeholders only, but never needed for display
    history.append(turn)
    st.rerun()

import hmac
import json
import re

import pandas as pd
import streamlit as st
from google import genai
from google.genai import types

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

client = genai.Client(api_key=st.secrets["api_keys"]["gemini_key"])
# "gemini-flash-latest" always points at the current Flash model, so the app
# doesn't break each time Google retires one. Pin a specific model (e.g.
# "gemini-3.8-flash") by setting [ai] model in secrets if you want stability.
MODEL_NAME = st.secrets.get("ai", {}).get("model", "gemini-flash-latest")

# ==========================================
# 3. DATA INGESTION (multiple sheets, any schema)
# ==========================================
def to_csv_url(url: str) -> str:
    m = re.search(r"/spreadsheets/d/([a-zA-Z0-9-_]+)", url)
    if not m:
        raise ValueError(f"Not a valid Google Sheets URL: {url}")
    gid = re.search(r"[#&?]gid=(\d+)", url)
    out = f"https://docs.google.com/spreadsheets/d/{m.group(1)}/export?format=csv"
    return out + (f"&gid={gid.group(1)}" if gid else "")


@st.cache_data(ttl=600, show_spinner="Loading sheet...")
def load_sheet(url: str) -> pd.DataFrame:
    df = pd.read_csv(to_csv_url(url))
    df.columns = [str(c).strip() for c in df.columns]
    df = df.dropna(how="all").dropna(axis=1, how="all")
    for c in df.select_dtypes(include="object").columns:
        cleaned = df[c].astype(str).str.replace(r"[%,$]", "", regex=True).str.strip()
        num = pd.to_numeric(cleaned, errors="coerce")
        if df[c].notna().sum() and num.notna().sum() / df[c].notna().sum() > 0.9:
            df[c] = num
    return df


def profile(df: pd.DataFrame) -> list[dict]:
    cols = []
    for c in df.columns:
        s = df[c]
        info = {"name": c, "dtype": str(s.dtype), "missing": int(s.isna().sum())}
        if pd.api.types.is_numeric_dtype(s):
            info.update(min=float(s.min()), max=float(s.max()), mean=round(float(s.mean()), 2))
        else:
            u = s.dropna().unique()
            info["unique"] = int(len(u))
            info["examples"] = [str(x) for x in u[:8]]
        cols.append(info)
    return cols


sheets = dict(st.secrets["sheets"])
if not sheets:
    st.error("No sheets configured in secrets.")
    st.stop()

with st.sidebar:
    st.header("Data source")
    sheet_name = st.selectbox("Sheet", list(sheets))
    if st.button("🔄 Refresh data"):
        st.cache_data.clear()
        st.rerun()
    if st.button("🧹 Clear conversation"):
        st.session_state["history"] = {}
        st.rerun()

try:
    df = load_sheet(sheets[sheet_name])
except Exception as e:
    st.error(f"Error loading '{sheet_name}': {e}")
    st.stop()

# ==========================================
# 4. AI ASSISTANT (first thing on the page)
# ==========================================
PLAN_PROMPT = """You translate a staff question into a JSON query plan for a pandas DataFrame.
Only use columns from the schema. Respond with JSON only, no prose, in this shape:
{{
  "filters": [{{"column": str, "op": "==|!=|>|>=|<|<=|contains|in|isnull|notnull", "value": any}}],
  "group_by": [str],
  "aggregate": {{"column": str, "func": "mean|median|sum|count|min|max"}} or null,
  "select_columns": [str],
  "sort_by": str or null,
  "ascending": bool,
  "limit": int or null,
  "note": "one line on how you interpreted the question, including any ambiguity"
}}
If the question can't be answered from the schema, return {{"error": "reason"}}.

Schema:
{schema}

Sample rows:
{sample}

Question: {question}
"""


def extract_json(raw: str) -> dict:
    start, end = raw.find("{"), raw.rfind("}")
    if start == -1 or end == -1:
        raise ValueError("The AI didn't return a query plan.")
    return json.loads(raw[start:end + 1])


def run_plan(df: pd.DataFrame, plan: dict) -> pd.DataFrame:
    out = df
    valid = set(df.columns)
    for f in plan.get("filters") or []:
        c, o, v = f.get("column"), f.get("op"), f.get("value")
        if c not in valid:
            raise ValueError(f"Unknown column '{c}'")
        s = out[c]
        if o in (">", ">=", "<", "<="):
            s = pd.to_numeric(s, errors="coerce")
            v = float(v)
        is_text = not pd.api.types.is_numeric_dtype(s)
        if o in ("in",) and not isinstance(v, list):
            v = [v]
        masks = {
            "==": lambda: s.astype(str).str.lower() == str(v).lower() if is_text else s == float(v),
            "!=": lambda: s.astype(str).str.lower() != str(v).lower() if is_text else s != float(v),
            ">": lambda: s > v, ">=": lambda: s >= v, "<": lambda: s < v, "<=": lambda: s <= v,
            "contains": lambda: s.astype(str).str.contains(str(v), case=False, na=False),
            "in": lambda: s.astype(str).str.lower().isin([str(x).lower() for x in v]),
            "isnull": lambda: s.isna(), "notnull": lambda: s.notna(),
        }
        if o not in masks:
            raise ValueError(f"Unsupported operator '{o}'")
        out = out[masks[o]()]

    agg = plan.get("aggregate")
    if agg:
        if agg.get("column") not in valid:
            raise ValueError(f"Unknown column '{agg.get('column')}'")
        keys = [g for g in plan.get("group_by") or [] if g in valid]
        func = agg.get("func", "mean")
        if func not in {"mean", "median", "sum", "count", "min", "max"}:
            raise ValueError(f"Unsupported aggregation '{func}'")
        out = (out.groupby(keys)[agg["column"]].agg(func).reset_index() if keys
               else pd.DataFrame({agg["column"]: [out[agg["column"]].agg(func)]}))
    else:
        cols = [c for c in plan.get("select_columns") or [] if c in valid]
        if cols:
            out = out[cols]

    sb = plan.get("sort_by")
    if sb in out.columns:
        out = out.sort_values(sb, ascending=bool(plan.get("ascending", True)))
    if plan.get("limit"):
        out = out.head(int(plan["limit"]))
    return out


st.title("💬 Student Insights Assistant")
st.caption(f"{sheet_name} · {len(df):,} rows × {len(df.columns)} columns")

history = st.session_state.setdefault("history", {}).setdefault(sheet_name, [])

if not history:
    st.write("Ask a question about this sheet, for example:")
    st.markdown(
        "- *Which Year 9 students have attendance below 80%?*\n"
        "- *What's the average score by year level?*\n"
        "- *List students with no recorded result.*"
    )

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
            st.caption(f"Interpreted as: {turn['plan'].get('note', '')}")
            st.write(f"**{len(turn['result'])} result(s)**")
            st.dataframe(turn["result"], use_container_width=True)
            if len(turn["result"]):
                st.download_button("Download CSV", turn["result"].to_csv(index=False),
                                   f"result_{i}.csv", key=f"dl_{sheet_name}_{i}")
            with st.expander("Query plan (for checking)"):
                st.json(turn["plan"])

question = st.chat_input("Ask a question about this sheet...")

if question:
    turn = {"question": question}
    raw = None
    with st.spinner("Working it out..."):
        try:
            prompt = PLAN_PROMPT.format(
                schema=json.dumps(profile(df), default=str),
                sample=df.head(3).to_csv(index=False),
                question=question,
            )
            raw = client.models.generate_content(
                model=MODEL_NAME,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    temperature=0,
                ),
            ).text
            plan = extract_json(raw)
            if "error" in plan:
                turn["error"] = plan["error"]
            else:
                turn["plan"] = plan
                turn["result"] = run_plan(df, plan)
        except Exception as e:
            turn["error"] = f"{type(e).__name__}: {e}"
            turn["raw"] = raw
    history.append(turn)
    st.rerun()

# ==========================================
# 5. DATA PREVIEW (collapsed, out of the way)
# ==========================================
st.divider()
with st.expander("Preview sheet & column summary"):
    st.dataframe(df.head(50), use_container_width=True)
    st.dataframe(pd.DataFrame(profile(df)), use_container_width=True)

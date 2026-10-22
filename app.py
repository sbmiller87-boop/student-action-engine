import hmac
import json
import re
import time

import pandas as pd
import streamlit as st
from google import genai
from google.genai import errors, types

from analysis import numeric_view, profile, run_plan

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
MODEL_NAME = st.secrets.get("ai", {}).get("model", "gemini-flash-latest")
FALLBACK_MODEL = st.secrets.get("ai", {}).get("fallback_model", "gemini-flash-lite-latest")


def call_model(prompt: str, as_json: bool = True) -> str:
    """Call Gemini, retrying temporary overloads, then falling back to a lighter model."""
    config = types.GenerateContentConfig(
        temperature=0 if as_json else 0.3,
        **({"response_mime_type": "application/json"} if as_json else {}),
    )
    transient = {429, 500, 502, 503, 504}
    last_err = None
    for model_name in dict.fromkeys([MODEL_NAME, FALLBACK_MODEL]):
        for attempt in range(3):
            try:
                return client.models.generate_content(
                    model=model_name, contents=prompt, config=config).text
            except errors.APIError as e:
                last_err = e
                if getattr(e, "code", None) not in transient:
                    raise
                time.sleep(2 ** attempt)
    raise last_err


def extract_json(raw: str) -> dict:
    start, end = raw.find("{"), raw.rfind("}")
    if start == -1 or end == -1:
        raise ValueError("The AI didn't return a query plan.")
    return json.loads(raw[start:end + 1])


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
    # Only convert a text column to numbers if EVERY value parses. Mixed columns
    # (e.g. marks plus 'Abs') stay as text so no entries are lost; the analysis
    # code handles the numeric view itself.
    for c in [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])]:
        nonnull = df[c].notna().sum()
        if nonnull and numeric_view(df[c]).notna().sum() == nonnull:
            df[c] = numeric_view(df[c])
    return df


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
# 4. PROMPTS
# ==========================================
PLAN_PROMPT = """You turn a staff question about a school data sheet into a JSON analysis plan.
Only use columns from the schema. Respond with JSON only:
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

NARRATE_PROMPT = """You are a data analyst briefing a school leader. Below are computed results.
Use only numbers that appear in them; never invent figures.

Question: {question}

{summary}

Write 3 to 6 short bullet points covering: what the distribution looks like, and anything
inconsistent or worth a closer look, each backed by a specific number. Then one final line of
caveats (for example small samples, marks possibly on different scales across tasks, peers possibly
teaching different cohorts). Frame findings as "worth a look" rather than accusations about a person.
Plain English, no preamble, no headings."""


def history_text(turns: list[dict]) -> str:
    lines = [f"Q: {t['question']} -> {t['plan'].get('note', '')}" for t in turns[-3:] if t.get("plan")]
    return "\n".join(lines) or "(none yet)"


# ==========================================
# 5. AI ASSISTANT
# ==========================================
st.title("💬 Student Insights Assistant")
st.caption(f"{sheet_name} · {len(df):,} rows × {len(df.columns)} columns")

history = st.session_state.setdefault("history", {}).setdefault(sheet_name, [])

if not history:
    st.write("Ask a question about this sheet, for example:")
    st.markdown(
        "- *How are teacher ABC's marks distributed, and is anything inconsistent?*\n"
        "- *Compare Year 9 maths marks with the rest of the school.*\n"
        "- *What's the average result per task?*\n"
        "- *List students with a missing result in Science.*"
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
            continue

        out, plan = turn["out"], turn["plan"]
        st.caption(f"Interpreted as: {plan.get('note', '')}")
        if turn.get("narrative"):
            st.markdown(turn["narrative"])
        for n in out["notes"]:
            st.caption(f"ℹ️ {n}")
        for j, fig in enumerate(out["figs"]):
            st.plotly_chart(fig, use_container_width=True, key=f"fig_{sheet_name}_{i}_{j}")

        if plan.get("intent", "table") == "table" or not out["figs"]:
            st.write(f"**{len(out['result'])} result(s)**")
            st.dataframe(out["result"], use_container_width=True)
        with st.expander("Details: tables, flags and query plan"):
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
                               f"result_{i}.csv", key=f"dl_{sheet_name}_{i}")

question = st.chat_input("Ask a question about this sheet...")

if question:
    turn, raw = {"question": question}, None
    with st.spinner("Analysing..."):
        try:
            prompt = PLAN_PROMPT.format(
                history=history_text(history),
                schema=json.dumps(profile(df), default=str),
                sample=df.head(3).to_csv(index=False),
                question=question,
            )
            raw = call_model(prompt)
            plan = extract_json(raw)
            if "error" in plan:
                turn["error"] = plan["error"]
            else:
                turn["plan"] = plan
                turn["out"] = run_plan(df, plan)
                if turn["out"]["summary"] and plan.get("intent") in ("distribution", "aggregate"):
                    try:
                        turn["narrative"] = call_model(
                            NARRATE_PROMPT.format(question=question, summary=turn["out"]["summary"]),
                            as_json=False)
                    except Exception as e:  # commentary is optional; keep charts/tables
                        turn["narrative"] = f"*(Couldn't generate commentary: {type(e).__name__})*"
        except Exception as e:
            turn["error"] = f"{type(e).__name__}: {e}"
            turn["raw"] = raw
    history.append(turn)
    st.rerun()

# ==========================================
# 6. DATA PREVIEW (collapsed, out of the way)
# ==========================================
st.divider()
with st.expander("Preview sheet & column summary"):
    st.dataframe(df.head(50), use_container_width=True)
    st.dataframe(pd.DataFrame(profile(df)), use_container_width=True)

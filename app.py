import hmac
import json
import re

import pandas as pd
import plotly.express as px
import streamlit as st
import google.generativeai as genai

# ==========================================
# 1. SECURITY & AUTHENTICATION
# ==========================================
def check_password() -> bool:
    if st.session_state.get("password_correct"):
        return True

    st.title("🔒 Staff Data Portal")
    pw = st.text_input("Enter Staff Password", type="password")
    if st.button("Log In"):
        # constant-time comparison
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
st.set_page_config(page_title="Student Action Engine", layout="wide")

genai.configure(api_key=st.secrets["api_keys"]["gemini_key"])
MODEL_NAME = st.secrets.get("ai", {}).get("model", "gemini-2.5-flash")
model = genai.GenerativeModel(MODEL_NAME)

# ==========================================
# 3. DATA INGESTION (multiple sheets, any schema)
# ==========================================
def to_csv_url(url: str) -> str:
    """Turn any Google Sheets URL into a CSV export URL (keeps the tab's gid)."""
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
    # Convert columns that are mostly numeric (handles "85%", "1,200" etc.)
    for c in df.select_dtypes(include="object").columns:
        cleaned = df[c].astype(str).str.replace(r"[%,$]", "", regex=True).str.strip()
        num = pd.to_numeric(cleaned, errors="coerce")
        if df[c].notna().sum() and num.notna().sum() / df[c].notna().sum() > 0.9:
            df[c] = num
    return df


def profile(df: pd.DataFrame) -> list[dict]:
    """Compact schema description - this is what we send to the AI instead of the whole sheet."""
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


sheets = dict(st.secrets["sheets"])  # {"Display name": "https://docs.google.com/..."}
if not sheets:
    st.error("No sheets configured in secrets.")
    st.stop()

with st.sidebar:
    st.header("Data source")
    sheet_name = st.selectbox("Sheet", list(sheets))
    if st.button("🔄 Refresh data"):
        st.cache_data.clear()
        st.rerun()

try:
    df = load_sheet(sheets[sheet_name])
except Exception as e:
    st.error(f"Error loading '{sheet_name}': {e}")
    st.stop()

num_cols = df.select_dtypes(include="number").columns.tolist()
cat_cols = [c for c in df.columns if c not in num_cols and df[c].nunique() <= 30]

st.title("📊 Student Action & Insights Engine")
st.caption(f"{sheet_name}: {len(df):,} rows × {len(df.columns)} columns")

with st.expander("Preview & column summary"):
    st.dataframe(df.head(50), use_container_width=True)
    st.dataframe(pd.DataFrame(profile(df)), use_container_width=True)

# ==========================================
# 4. QUICK ACTION LISTS (user picks the columns)
# ==========================================
st.subheader("⚡ Quick Action Lists")
OPS = {
    "equals": lambda s, v: s.astype(str) == str(v),
    "not equals": lambda s, v: s.astype(str) != str(v),
    "contains": lambda s, v: s.astype(str).str.contains(str(v), case=False, na=False),
    ">": lambda s, v: pd.to_numeric(s, errors="coerce") > float(v),
    ">=": lambda s, v: pd.to_numeric(s, errors="coerce") >= float(v),
    "<": lambda s, v: pd.to_numeric(s, errors="coerce") < float(v),
    "<=": lambda s, v: pd.to_numeric(s, errors="coerce") <= float(v),
}

c1, c2, c3, c4 = st.columns([2, 1.2, 2, 1])
col = c1.selectbox("Column", df.columns)
op = c2.selectbox("Condition", list(OPS))
if col in cat_cols and op in ("equals", "not equals"):
    val = c3.selectbox("Value", sorted(df[col].dropna().astype(str).unique()))
else:
    val = c3.text_input("Value")
list_name = c4.text_input("File name", "action_list")

if st.button("Generate list") and val != "":
    try:
        result = df[OPS[op](df[col], val)]
        st.write(f"**{len(result)} students match**")
        st.dataframe(result, use_container_width=True)
        st.download_button("Download CSV", result.to_csv(index=False), f"{list_name}.csv")
    except Exception as e:
        st.error(f"Couldn't apply that filter: {e}")

st.divider()

# ==========================================
# 5. FLEXIBLE CHARTS
# ==========================================
st.subheader("📈 Explore")
if num_cols:
    g1, g2, g3 = st.columns(3)
    y = g1.selectbox("Measure (Y)", num_cols)
    x = g2.selectbox("Group by (X)", ["(none)"] + cat_cols)
    kind = g3.selectbox("Chart", ["Box", "Histogram", "Bar (mean)"])
    if kind == "Box":
        fig = px.box(df, x=None if x == "(none)" else x, y=y,
                     color=None if x == "(none)" else x, points="all")
    elif kind == "Histogram":
        fig = px.histogram(df, x=y, color=None if x == "(none)" else x)
    else:
        if x == "(none)":
            st.info("Pick a 'Group by' column for a bar chart.")
            fig = None
        else:
            fig = px.bar(df.groupby(x, as_index=False)[y].mean(), x=x, y=y)
    if fig:
        st.plotly_chart(fig, use_container_width=True)
else:
    st.info("No numeric columns detected in this sheet.")

st.divider()

# ==========================================
# 6. AI ASSISTANT (schema + query plan, not the whole dataset)
# ==========================================
# The model never receives the full sheet. It sees column names, types and a few
# example values, returns a JSON query plan, and pandas executes it. That means:
#   - works for any column set and any sheet size
#   - no student rows leave the app beyond a small preview of the *result*
#   - results are real rows, so the model can't invent students
st.subheader("💬 Staff AI Assistant")
st.write("Ask about the selected sheet, e.g. *'Which Year 9 students have attendance below 80%?'*")

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
        masks = {
            "==": lambda: s.astype(str).str.lower() == str(v).lower() if s.dtype == object else s == v,
            "!=": lambda: s.astype(str).str.lower() != str(v).lower() if s.dtype == object else s != v,
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


question = st.chat_input("Ask a question about this sheet...")

if question:
    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):
        with st.spinner("Working it out..."):
            try:
                prompt = PLAN_PROMPT.format(
                    schema=json.dumps(profile(df), default=str),
                    sample=df.head(3).to_csv(index=False),
                    question=question,
                )
                raw = model.generate_content(prompt).text
                plan = json.loads(re.sub(r"^```(?:json)?|```$", "", raw.strip(), flags=re.M).strip())

                if "error" in plan:
                    st.warning(plan["error"])
                else:
                    result = run_plan(df, plan)
                    st.caption(f"Interpreted as: {plan.get('note', '')}")
                    st.write(f"**{len(result)} result(s)**")
                    st.dataframe(result, use_container_width=True)
                    if len(result):
                        st.download_button("Download CSV", result.to_csv(index=False),
                                           "ai_result.csv", key=f"dl_{hash(question)}")
                    with st.expander("Query plan (for checking)"):
                        st.json(plan)
            except Exception as e:
                st.error(f"Couldn't answer that: {e}. Try rephrasing, or use the filters above.")

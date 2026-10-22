import streamlit as st
import pandas as pd
import plotly.express as px
import google.generativeai as genai

# ==========================================
# 1. SECURITY & AUTHENTICATION
# ==========================================
def check_password():
    """Validates the staff password before rendering the app."""
    if "password_correct" not in st.session_state:
        st.session_state["password_correct"] = False

    if st.session_state["password_correct"]:
        return True

    st.title("🔒 Staff Data Portal")
    password_input = st.text_input("Enter Staff Password", type="password")
    
    if st.button("Log In"):
        if password_input == st.secrets["passwords"]["staff_password"]:
            st.session_state["password_correct"] = True
            st.rerun()
        else:
            st.error("Incorrect password.")
    return False

if not check_password():
    st.stop() # Stops the script here if not logged in

# ==========================================
# 2. APP CONFIGURATION & AI SETUP
# ==========================================
st.set_page_config(page_title="Student Action Engine", layout="wide")

# Configure Gemini API (Zero Data Retention)
genai.configure(api_key=st.secrets["api_keys"]["gemini_key"])
model = genai.GenerativeModel('gemini-1.5-flash')

# ==========================================
# 3. DATA INGESTION (Google Sheets)
# ==========================================
@st.cache_data(ttl=600) # Caches data for 10 mins to save API calls
def load_sheet_data(sheet_url):
    """Converts a standard Google Sheets sharing link to a CSV export link and loads it."""
    csv_url = sheet_url.replace('/edit?usp=sharing', '/export?format=csv')
    return pd.read_csv(csv_url)

# Replace this with your actual Google Sheet link (must be set to 'Anyone with the link can view' 
# or use st.connection for strict private service accounts)
SHEET_URL = "https://docs.google.com/spreadsheets/d/YOUR_SHEET_ID/edit?usp=sharing"

try:
    df = load_sheet_data(SHEET_URL)
except Exception as e:
    st.error(f"Error loading data: {e}")
    st.stop()

# ==========================================
# 4. FRONT-END LAYOUT & VISUALIZATIONS
# ==========================================
st.title("📊 Student Action & Insights Engine")

# Top Row: Actionable Quick Filters
st.subheader("⚡ Quick Action Lists")
col1, col2 = st.columns(2)

with col1:
    if st.button("🚀 Generate Celebration List (High Growth)"):
        # Example logic: Filter for top growth
        if 'Relative Growth' in df.columns:
            celebration_df = df[df['Relative Growth'] == 'High']
            st.dataframe(celebration_df, use_container_width=True)
            st.download_button("Download CSV", celebration_df.to_csv(index=False), "celebration_list.csv")

with col2:
    if st.button("⚠️ Generate Intervention List (Low Attendance)"):
        # Example logic: Filter for < 90% attendance
        if 'ClassPercentage' in df.columns:
            risk_df = df[df['ClassPercentage'] < 90]
            st.dataframe(risk_df, use_container_width=True)
            st.download_button("Download CSV", risk_df.to_csv(index=False), "intervention_list.csv")

st.divider()

# Middle Row: Automated Box Plots (e.g., PAT Scores)
st.subheader("📈 Cohort Distributions")
if 'Year_Level' in df.columns and 'PAT_Score' in df.columns: # Adjust column names to match your sheet
    fig = px.box(df, x="Year_Level", y="PAT_Score", color="Year_Level",
                 title="PAT Score Distribution by Year Level",
                 points="all") # Shows individual student dots next to the box
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("Ensure your sheet has 'Year_Level' and 'PAT_Score' columns to render the box plot.")

st.divider()

# ==========================================
# 5. THE AI CHAT ASSISTANT
# ==========================================
st.subheader("💬 Staff AI Assistant")
st.write("Ask the AI to filter, summarize, or identify specific student groups based on the live sheet data.")

user_query = st.chat_input("E.g., 'Which Year 9 students have stalled growth and attendance below 80%?'")

if user_query:
    # Display user query
    with st.chat_message("user"):
        st.write(user_query)
    
    # Generate AI response ephemerally using the dataframe context
    with st.chat_message("assistant"):
        with st.spinner("Analyzing cohort data..."):
            # We convert a snippet of the dataframe to markdown for the AI context
            data_context = df.to_markdown()
            
            prompt = f"""
            You are a data analyst assistant for a school. 
            Analyze the following student dataset to answer the staff member's query.
            Do not hallucinate. If the data isn't there, say so. Keep formatting clean using markdown tables if listing students.
            
            User Query: {user_query}
            
            Dataset:
            {data_context}
            """
            
            response = model.generate_content(prompt)
            st.write(response.text)

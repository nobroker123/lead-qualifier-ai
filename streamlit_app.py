import streamlit as st
from pinecone import Pinecone
import google.generativeai as genai
from sentence_transformers import SentenceTransformer
from datetime import datetime

# 1. Set up the Web Page
st.set_page_config(page_title="Lead Qualifier AI", layout="centered")

# 2. Access Secrets (API Keys)
try:
    PINECONE_KEY = st.secrets["PINECONE_KEY"]
    GEMINI_KEY = st.secrets["GEMINI_KEY"]
except KeyError:
    st.error("API Keys not found in Streamlit Secrets! Please check your settings.")
    st.stop()

# 3. Initialize Models & Database
@st.cache_resource
def load_tools():
    # Initialize Pinecone
    pc = Pinecone(api_key=PINECONE_KEY)
    idx = pc.Index("lead-index")
    # Initialize Embedding Model (turns text into numbers)
    model = SentenceTransformer('all-MiniLM-L6-v2')
    # Initialize Gemini AI (using the stable 'gemini-pro' version)
    genai.configure(api_key=GEMINI_KEY)
    ai_model = genai.GenerativeModel('gemini-pro')
    return idx, model, ai_model

index, embed_model, llm = load_tools()

# UI Layout
st.title("🚀 Lead Qualifier AI")
st.markdown("Deduplication | Hierarchy Priority | Recency Check")

# 4. Load Rules from rules.txt
try:
    with open("rules.txt", "r") as f:
        company_rules = f.read()
except FileNotFoundError:
    company_rules = "No rules.txt file found. Please check GitHub."

# 5. User Input
new_lead = st.text_input("Enter New Lead Name (Society / Address):", placeholder="e.g. Sai Apartment Bldg 4")

if st.button("Check & Qualify"):
    if new_lead:
        with st.spinner("Checking database and applying rules..."):
            # A. Convert text to vector
            vector = embed_model.encode(new_lead).tolist()
            
            # B. Search Pinecone for Top 3 matches (needed for Rule 2 & 3)
            results = index.query(vector=vector, top_k=3, include_metadata=True)
            
            db_context = ""
            if results['matches']:
                for match in results['matches']:
                    m_text = match['id']
                    m_score = round(match['score'], 2)
                    m_meta = match.get('metadata', {})
                    m_time = m_meta.get('updated_at', 'Unknown Time')
                    m_type = m_meta.get('type', 'Standard')
                    db_context += f"- Record: {m_text} | Similarity: {m_score} | Updated: {m_time} | Priority: {m_type}\n"
            else:
                db_context = "No existing records found in database."

            # C. Create the AI Prompt
            prompt = f"""
            You are a Sales Operations Expert. Analyze the new lead based on these COMPANY RULES:
            {company_rules}

            NEW LEAD: {new_lead}
            
            DATABASE MATCHES FOUND:
            {db_context}

            TASK:
            1. Determine if this is a DUPLICATE, RELATED (same society, different bldg), or NEW lead.
            2. Apply Rule 2 (Hierarchy): Priority goes to 'Sales Hierarchy' over 'Agent Hierarchy'.
            3. Apply Rule 3 (Recency): If matches are equal, pick the one with the latest 'Updated' date.
            4. Provide a clear 'Decision' and 'Assignment Suggestion'.
            """
            
            # D. Get AI Response
            try:
                response = llm.generate_content(prompt)
                st.subheader("📋 AI Decision")
                st.info(response.text)
            except Exception as e:
                st.warning("The AI is busy, but here are the matches from the database for you to check:")
                st.write(db_context)

            # E. Save Logic
            # If the closest match is very different (< 85%), we save it as a new record
            if not results['matches'] or results['matches'][0]['score'] < 0.85:
                now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                # Metadata helps with Rule 2 and 3 in the next search
                index.upsert(vectors=[(new_lead, vector, {
                    "updated_at": now_str, 
                    "type": "Sales Hierarchy" # Default for first-time entries
                })])
                st.success(f"New lead '{new_lead}' has been saved to the database.")
            else:
                st.warning("This lead is too similar to an existing entry. Not saving to prevent duplicate data.")
    else:
        st.error("Please enter a lead name to qualify.")

st.markdown("---")
st.caption("Powered by Pinecone Vector DB & Google Gemini AI")

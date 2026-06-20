import streamlit as st
from pinecone import Pinecone
import google.generativeai as genai
from sentence_transformers import SentenceTransformer
from datetime import datetime

# 1. API Setup
PINECONE_KEY = st.secrets["PINECONE_KEY"]
GEMINI_KEY = st.secrets["GEMINI_KEY"]

pc = Pinecone(api_key=PINECONE_KEY)
index = pc.Index("lead-index")
embed_model = SentenceTransformer('all-MiniLM-L6-v2')
genai.configure(api_key=GEMINI_KEY)
llm = genai.GenerativeModel('gemini-1.5-flash')

st.title("Lead Qualifier AI (Hierarchy & Recency Enabled)")

with open("rules.txt", "r") as f:
    company_rules = f.read()

new_lead = st.text_input("Enter lead name (Society/Address):")

if st.button("Check & Qualify"):
    if new_lead:
        vector = embed_model.encode(new_lead).tolist()
        
        # We search for Top 3 matches now to help with Rule 2 and Rule 3
        results = index.query(vector=vector, top_k=3, include_metadata=True)
        
        db_context = ""
        if results['matches']:
            for match in results['matches']:
                db_context += f"- Match: {match['id']}, Score: {match['score']}, Metadata: {match['metadata']}\n"
        else:
            db_context = "No existing match found."

        # AI Prompt including your Hierarchy and Recency rules
        prompt = f"""
        Rules: {company_rules}
        Current Time: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
        
        Incoming Lead: {new_lead}
        Database Matches:
        {db_context}
        
        Task:
        - Analyze the hierarchy (Rule 2).
        - If multiple matches exist, pick the most recent (Rule 3).
        - Qualify as Duplicate, Related, or New.
        - Explain your reasoning based on the rules.
        """
        
        response = llm.generate_content(prompt)
        st.subheader("AI Decision")
        st.write(response.text)
        
        if not results['matches'] or results['matches'][0]['score'] < 0.85:
            # Save new lead with Timestamp and a dummy Hierarchy for now
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            index.upsert(vectors=[(new_lead, vector, {"updated_at": timestamp, "type": "Sales Hierarchy"})])
            st.success(f"New lead saved at {timestamp}")
    else:
        st.error("Please enter a name.")

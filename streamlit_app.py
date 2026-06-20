import streamlit as st
from pinecone import Pinecone
import google.generativeai as genai
from sentence_transformers import SentenceTransformer

# 1. Connect to your API Keys (we will set these in the next phase)
PINECONE_KEY = st.secrets["PINECONE_KEY"]
GEMINI_KEY = st.secrets["GEMINI_KEY"]

# 2. Initialize the AI tools
pc = Pinecone(api_key=PINECONE_KEY)
index = pc.Index("lead-index") # Matches your Pinecone name
embed_model = SentenceTransformer('all-MiniLM-L6-v2')
genai.configure(api_key=GEMINI_KEY)
llm = genai.GenerativeModel('gemini-1.5-flash')

st.title("Lead Qualifier AI")

# Load rules from your file
with open("rules.txt", "r") as f:
    company_rules = f.read()

# UI Input
new_lead = st.text_input("Enter lead name (Society/Address):")

if st.button("Check & Qualify"):
    if new_lead:
        # Convert text to numbers
        vector = embed_model.encode(new_lead).tolist()
        
        # Search Pinecone
        results = index.query(vector=vector, top_k=1, include_metadata=True)
        
        match_info = "No existing match found."
        if results['matches'] and results['matches'][0]['score'] > 0.80:
            match_info = f"Existing Match: {results['matches'][0]['id']}"

        # Ask Gemini to decide
        prompt = f"Rules: {company_rules}\nNew Lead: {new_lead}\nDatabase: {match_info}\nQualify this lead and explain why."
        response = llm.generate_content(prompt)
        
        st.subheader("AI Decision")
        st.write(response.text)
        
        # Save to DB if it's new
        if "No existing match" in match_info:
            index.upsert(vectors=[(new_lead, vector)])
            st.success("New lead saved to database!")
    else:
        st.error("Please enter a name.")

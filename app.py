import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import pdfplumber
import re
import os
import json
import google.generativeai as genai
from dotenv import load_dotenv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(
    page_title="CareerCopilot AI", 
    page_icon="🚀", 
    layout="wide" # This makes your app take up more of the screen!
)
# --- UTILITIES ---
def reset_app():
    """Clears all session data and restarts the app."""
    st.cache_data.clear()
    st.rerun()

# --- 1. SETUP & CONFIG ---
# --- 1. SETUP & CONFIG ---
load_dotenv()  # This loads the .env file

# Try to get the key from the environment
api_key = os.getenv("GOOGLE_API_KEY") 

# If it's empty, try the other common name
if not api_key:
    api_key = os.getenv("GEMINI_API_KEY")

# Safety Check: If still no key, show a clear error in Streamlit
if not api_key:
    st.error("🔑 API Key not found! Please check your .env file.")
    st.stop() 

genai.configure(api_key=api_key)
# Try the most stable production name
model = genai.GenerativeModel('gemini-2.5-flash') # The current active model

# Temporary debug line - check your VS Code terminal for the output
#print([m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods])
# --- 2. AI CORE FUNCTIONS ---



def get_ai_suggestions(resume_text, job_desc, missing_skills):
    """Generates specific career advice to bridge the gap."""
    prompt = f"""
    Compare this resume to the job description. 
    Missing Skills: {', '.join(missing_skills)}
    
    Provide 3 specific, actionable bullet points on how this candidate can 
    improve their resume to match this job. 
    
    Resume Snippet: {resume_text[:1500]}
    JD Snippet: {job_desc[:1500]}
    """
    try:
        response = model.generate_content(prompt)
        return response.text
    except:
        return "Could not generate suggestions."

def get_interview_questions(job_desc):
    """Generates interview questions and answers based on the JD in JSON format."""
    prompt = f"""
    You are an expert technical recruiter. Analyze this Job Description, identify the core technical skills, and generate 5 targeted interview questions.
    
    For each question, provide a detailed "Ideal Answer" that a top candidate would give.
    
    Job Description: {job_desc[:2000]}
    
    Return ONLY a valid JSON array of objects with "question" and "answer" keys. Do not include any markdown outside the JSON.
    Example format:
    [
      {{
        "question": "What is the difference between a list and a tuple in Python?",
        "answer": "A list is mutable, meaning it can be changed after creation, while a tuple is immutable..."
      }}
    ]
    """
    try:
        response = model.generate_content(prompt)
        # Clean the response to ensure it's valid JSON
        clean_json = response.text.replace('```json', '').replace('```', '').strip()
        data = json.loads(clean_json)
        return data
    except Exception as e:
        # Fallback if AI fails so the app doesn't crash
        return [{"question": "System Error", "answer": f"Could not generate questions: {e}"}]

# --- 3. UTILITIES ---

def extract_text_from_pdf(uploaded_file):
    text = ""
    with pdfplumber.open(uploaded_file) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text: text += page_text + " "
    return text

def calculate_match_accurate(resume_text, job_desc):
    # This prompt asks the AI to extract AND match in one go
    prompt = f"""
    You are an expert ATS System. Compare the Resume to the Job Description.
    
    JD: {job_desc[:2000]}
    Resume: {resume_text[:2000]}
    
    Extract matched skills, missing skills, and provide a match score (0-100).
    Return ONLY a JSON object:
    {{
        "score": 85,
        "matched": ["Skill A", "Skill B"],
        "missing": ["Skill C"]
    }}
    """
    try:
        response = model.generate_content(prompt)
        # Robust cleaning for JSON
        clean_json = response.text.replace('```json', '').replace('```', '').strip()
        data = json.loads(clean_json)
        
        ai_skill_score = data.get("score", 0)
        matched = data.get("matched", [])
        missing = data.get("missing", [])
        
        # We still use TF-IDF as a "sanity check" for the final score
        vectorizer = TfidfVectorizer(stop_words='english')
        tfidf = vectorizer.fit_transform([resume_text.lower(), job_desc.lower()])
        cosine_sim = cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0]
        
        # Weighted Final Score: 80% AI Logic, 20% Math Similarity
        final_score = (ai_skill_score * 0.8) + (cosine_sim * 100 * 0.2)
        
        return round(final_score, 2), matched, missing
    except Exception as e:
        st.error(f"Analysis Error: {e}")
        return 0, [], []


# --- 4. STREAMLIT UI ---

st.title("🚀 CareerCopilot: Resume & Interview AI")
st.caption("Your all-in-one assistant for ATS screening, skill matching, and interview prep.")
st.markdown("---")

# 1. MOVE INPUTS TO A SIDEBAR
with st.sidebar:
    st.header("📋 Input Data")
    job_description = st.text_area("Job Description", height=250, placeholder="Paste JD here...")
    uploaded_files = st.file_uploader("Upload Resumes (PDF)", type=['pdf'], accept_multiple_files=True)
    analyze_btn = st.button("Analyze Match", type="primary", use_container_width=True)

    if st.button("🗑️ Clear All", use_container_width=True, type="secondary"):
        reset_app()

    st.divider() # Adds a clean line between the button and the tips
    st.markdown("### 💡 How to use:")
    st.info("""
    1. Paste the **Job Description**.
    2. Upload one or more **Resumes** (PDF).
    3. Click **Analyze Match**.
    4. Explore the tabs for insights!
    """)

# 2. MAIN SCREEN RESULTS
if analyze_btn and uploaded_files and job_description:
    results = []
    with st.spinner("AI is analyzing skills & generating insights..."):
        for file in uploaded_files:
            raw_text = extract_text_from_pdf(file)
            score, matched, missing = calculate_match_accurate(raw_text, job_description)

            results.append({
                "name": file.name, 
                "score": score, 
                "matched": matched, 
                "missing": missing,
                "raw": raw_text
            })

    results = sorted(results, key=lambda x: x["score"], reverse=True)
    top = results[0]

    # Display the Top Candidate in a nice container with a Metric
    st.success(f"### 🏆 Top Candidate: {top['name']}")
    st.metric(label="Overall ATS Match", value=f"{top['score']}%")
    st.progress(int(top['score'])) # Visual progress bar
    st.divider()

    # 3. USE TABS FOR CLEAN ORGANIZATION
    tab1, tab2, tab3 = st.tabs(["📊 Candidate Breakdown", "💡 AI Advice", "🎯 Interview Prep"])

    with tab1:
        st.subheader("All Candidates")
        df = pd.DataFrame(results)[["name", "score"]]
        st.bar_chart(df.set_index("name"))
        
        for res in results:
            with st.expander(f"Skill Breakdown: {res['name']}"):
                st.write(f"**✅ Matched:** {', '.join(res['matched']) if res['matched'] else 'None'}")
                st.write(f"**❌ Missing:** {', '.join(res['missing']) if res['missing'] else 'None'}")

    with tab2:
        st.subheader("Resume Improvement Plan")
        with st.spinner("Generating targeted advice..."):
            advice = get_ai_suggestions(top['raw'], job_description, top['missing'])
            st.info(advice)

    with tab3:
        st.subheader("Mock Interview Questions")
        with st.spinner("Generating questions from JD..."):
            qa_list = get_interview_questions(job_description)
            if isinstance(qa_list, list):
                for i, item in enumerate(qa_list):
                    st.markdown(f"**Q{i+1}: {item.get('question', 'Error loading question')}**")
                    with st.expander("View Ideal Answer"):
                        st.write(item.get('answer', 'No answer available.'))
            else:
                st.error("There was an issue formatting the interview questions.")

elif analyze_btn:
    st.warning("Please provide both a job description and at least one resume in the sidebar.")
import streamlit as st
import pandas as pd
import pdfplumber
import json
import os
import io
import google.generativeai as genai
from dotenv import load_dotenv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ── PAGE CONFIG ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CareerCopilot AI",
    page_icon="🚀",
    layout="wide",
)

# ── CUSTOM CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Sidebar refinements */
    [data-testid="stSidebar"] { 
    background-color: #1a1a2e !important; 
    color: #ffffff;
}

[data-testid="stSidebar"] * {
    color: #e0e0e0 !important;
}

[data-testid="stSidebar"] textarea,
[data-testid="stSidebar"] input {
    background-color: #2a2a3e !important;
    color: #ffffff !important;
    border: 1px solid #444466 !important;
    border-radius: 8px !important;
}

[data-testid="stSidebar"] .stButton > button {
    background-color: #534AB7 !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 8px !important;
}

    /* Metric cards */
    [data-testid="metric-container"] {
        background: #ffffff;
        border: 1px solid #e8e6f0;
        border-radius: 10px;
        padding: 12px 16px;
    }

    /* Skill pill tags */
    .pill-matched {
        display: inline-block;
        background: #d4edda; color: #155724;
        border-radius: 20px; padding: 3px 10px;
        font-size: 12px; margin: 3px 2px;
    }
    .pill-missing {
        display: inline-block;
        background: #f8d7da; color: #721c24;
        border-radius: 20px; padding: 3px 10px;
        font-size: 12px; margin: 3px 2px;
    }
    .pill-keyword {
        display: inline-block;
        background: #e8e6ff; color: #3a2e8c;
        border-radius: 20px; padding: 3px 10px;
        font-size: 12px; margin: 3px 2px;
    }

    /* Score bar colors */
    .score-high { color: #1e7e34; font-weight: 600; }
    .score-mid  { color: #856404; font-weight: 600; }
    .score-low  { color: #721c24; font-weight: 600; }

    /* Advice block */
    .advice-box {
        border-left: 4px solid #534AB7;
        background: #f4f3ff;
        padding: 14px 16px;
        border-radius: 0 8px 8px 0;
        font-size: 14px;
        line-height: 1.7;
        color: #2d2d2d;
    }
    div[data-testid="stExpander"] { border: 1px solid #e8e6f0 !important; border-radius: 8px !important; }
</style>
""", unsafe_allow_html=True)

# ── SETUP ──────────────────────────────────────────────────────────────────────
load_dotenv()

api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
if not api_key:
    st.error("🔑 API Key not found! Please check your .env file.")
    st.stop()

@st.cache_resource
def load_model():
    genai.configure(api_key=api_key)
    return genai.GenerativeModel("gemini-1.5-flash-latest")

model = load_model()

# ── SESSION STATE ──────────────────────────────────────────────────────────────
if "history" not in st.session_state:
    st.session_state.history = []   # list of {label, results, jd}
if "results" not in st.session_state:
    st.session_state.results = []
if "job_description" not in st.session_state:
    st.session_state.job_description = ""

# ── AI HELPERS ─────────────────────────────────────────────────────────────────

def _json_schema_call(prompt: str, schema: dict):
    """Shared helper for structured JSON calls."""
    response = model.generate_content(
        prompt,
        generation_config=genai.GenerationConfig(
            response_mime_type="application/json",
            response_schema=schema,
        ),
    )
    return json.loads(response.text)


def get_ai_suggestions(resume_text: str, job_desc: str, missing_skills: list) -> str:
    prompt = f"""
    You are a career coach reviewing a resume against a job description.
    Missing skills: {', '.join(missing_skills) or 'None identified'}

    Give exactly 4 specific, actionable bullet points (start each with •) on how
    the candidate can improve their resume to better match this role.

    Resume (first 1500 chars): {resume_text[:1500]}
    JD (first 1500 chars):     {job_desc[:1500]}
    """
    try:
        return model.generate_content(prompt).text
    except Exception as e:
        return f"Could not generate suggestions: {e}"


def get_interview_questions(job_desc: str) -> list:
    prompt = f"""
    You are an expert technical recruiter. Analyze this Job Description and return
    exactly 5 interview questions with detailed ideal answers.
    Job Description: {job_desc[:2000]}
    """
    schema = {
        "type": "ARRAY",
        "items": {
            "type": "OBJECT",
            "properties": {
                "question": {"type": "STRING"},
                "answer":   {"type": "STRING"},
            },
            "required": ["question", "answer"],
        },
    }
    try:
        return _json_schema_call(prompt, schema)
    except Exception as e:
        return [{"question": "Error", "answer": str(e)}]


def generate_cover_letter(resume_text: str, job_desc: str, candidate_name: str) -> str:
    prompt = f"""
    Write a professional, compelling cover letter for this candidate applying to
    this role. Keep it to 3 paragraphs. Be specific and reference real skills
    from the resume.

    Candidate name (use if available, else 'the candidate'): {candidate_name}
    Resume: {resume_text[:2000]}
    Job Description: {job_desc[:2000]}
    """
    try:
        return model.generate_content(prompt).text
    except Exception as e:
        return f"Could not generate cover letter: {e}"


def extract_jd_keywords(job_desc: str) -> list:
    """Extract top TF-IDF keywords from the JD."""
    try:
        vectorizer = TfidfVectorizer(stop_words="english", max_features=15)
        vectorizer.fit([job_desc])
        return list(vectorizer.get_feature_names_out())
    except Exception:
        return []

# ── CORE ANALYSIS ──────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def extract_text_from_pdf(file_bytes: bytes) -> str:
    """Extract text from PDF bytes (cached so same file isn't re-parsed)."""
    text = ""
    try:
        with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + " "
    except Exception as e:
        st.warning(f"PDF read error: {e}")
    return text.strip()


def calculate_match(resume_text: str, job_desc: str):
    prompt = f"""
    You are an expert ATS system. Compare the resume to the job description.
    Extract matched skills, missing skills, and a match score (0–100).

    JD:     {job_desc[:2000]}
    Resume: {resume_text[:2000]}
    """
    schema = {
        "type": "OBJECT",
        "properties": {
            "score":   {"type": "INTEGER"},
            "matched": {"type": "ARRAY", "items": {"type": "STRING"}},
            "missing": {"type": "ARRAY", "items": {"type": "STRING"}},
        },
        "required": ["score", "matched", "missing"],
    }
    try:
        data = _json_schema_call(prompt, schema)
        ai_score = int(data.get("score", 0))
        matched  = data.get("matched", [])
        missing  = data.get("missing", [])

        vectorizer   = TfidfVectorizer(stop_words="english")
        tfidf        = vectorizer.fit_transform([resume_text.lower(), job_desc.lower()])
        cosine_sim   = cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0]

        final_score = min(100, round((ai_score * 0.8) + (cosine_sim * 100 * 0.2), 1))
        return final_score, matched, missing

    except Exception as e:
        st.error(f"Analysis error: {e}")
        return 0, [], []

# ── SCORE COLOR HELPER ─────────────────────────────────────────────────────────

def score_color(score: float) -> str:
    if score >= 70:
        return "score-high"
    elif score >= 45:
        return "score-mid"
    return "score-low"

# ── SIDEBAR ────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 🚀 CareerCopilot AI")
    st.caption("ATS screening · Skill matching · Interview prep")
    st.divider()

    job_description = st.text_area(
        "Job Description",
        height=220,
        placeholder="Paste the full job description here…",
        key="jd_input",
    )
    uploaded_files = st.file_uploader(
        "Upload Resumes (PDF)",
        type=["pdf"],
        accept_multiple_files=True,
    )

    analyze_btn = st.button("🔍 Analyze Candidates", type="primary", use_container_width=True)

    if st.button("🗑️ Clear All", use_container_width=True):
        st.session_state.results = []
        st.session_state.job_description = ""
        st.cache_data.clear()
        st.rerun()

    st.divider()

    # Session history
    if st.session_state.history:
        st.markdown("#### 🕓 History")
        for i, entry in enumerate(reversed(st.session_state.history[-5:])):
            if st.button(entry["label"], key=f"hist_{i}", use_container_width=True):
                st.session_state.results        = entry["results"]
                st.session_state.job_description = entry["jd"]
                st.rerun()
    else:
        st.markdown("#### 💡 How to use")
        st.info("1. Paste the **Job Description**\n2. Upload one or more **PDFs**\n3. Click **Analyze Candidates**\n4. Explore the result tabs")

# ── ANALYSIS ───────────────────────────────────────────────────────────────────

if analyze_btn:
    if not uploaded_files or not job_description.strip():
        st.warning("Please provide both a job description and at least one resume.")
    else:
        results = []
        progress = st.progress(0, text="Analyzing resumes…")
        for idx, file in enumerate(uploaded_files):
            file_bytes = file.read()
            raw_text   = extract_text_from_pdf(file_bytes)

            if len(raw_text.strip()) < 100:
                st.warning(
                    f"⚠️ **{file.name}** appears to be a scanned/image PDF. "
                    "Text extraction returned very little content — results may be inaccurate."
                )

            score, matched, missing = calculate_match(raw_text, job_description)
            results.append({
                "name":    file.name,
                "score":   score,
                "matched": matched,
                "missing": missing,
                "raw":     raw_text,
            })
            progress.progress((idx + 1) / len(uploaded_files), text=f"Analyzed {file.name}")

        progress.empty()
        results.sort(key=lambda x: x["score"], reverse=True)

        # Save to session & history
        st.session_state.results        = results
        st.session_state.job_description = job_description
        label = f"{uploaded_files[0].name[:20]}… ({len(uploaded_files)} files)"
        st.session_state.history.append({"label": label, "results": results, "jd": job_description})

# ── RESULTS ────────────────────────────────────────────────────────────────────

results      = st.session_state.results
job_desc_ctx = st.session_state.job_description

if results:
    top = results[0]

    # ── Summary metrics ──────────────────────────────────────────────────────
    st.markdown("### 📊 Analysis Summary")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("🏆 Top Match",       f"{top['score']}%",         top["name"])
    c2.metric("👥 Candidates",      len(results),               "reviewed")
    c3.metric("✅ Matched Skills",  len(top["matched"]),        "in top resume")
    c4.metric("❌ Skills Gap",      len(top["missing"]),        "missing from top")

    # Banner based on top score
    if top["score"] >= 75:
        st.success(f"🎉 Strong match found! **{top['name']}** scores {top['score']}% — well aligned with the JD.")
    elif top["score"] >= 50:
        st.warning(f"⚠️ Moderate match. **{top['name']}** scores {top['score']}% — some key skills are missing.")
    else:
        st.error(f"🔴 Low match. Best candidate scores only {top['score']}% — consider revisiting the candidate pool.")

    st.divider()

    # ── Tabs ─────────────────────────────────────────────────────────────────
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📋 Candidates", "🧩 Skill Map", "💡 Advice", "🎯 Interview Prep", "✉️ Cover Letter"
    ])

    # ─── TAB 1: Candidate Breakdown ──────────────────────────────────────────
    with tab1:
        col_left, col_right = st.columns([2, 1])

        with col_left:
            st.markdown("#### Ranked candidates")
            for i, res in enumerate(results):
                rank_emoji = ["🥇", "🥈", "🥉"][i] if i < 3 else f"#{i+1}"
                css_class  = score_color(res["score"])
                bar_val    = int(res["score"])

                with st.container():
                    r1, r2 = st.columns([3, 1])
                    with r1:
                        st.markdown(f"{rank_emoji} **{res['name']}**")
                        st.progress(bar_val / 100)
                    with r2:
                        st.markdown(
                            f"<span class='{css_class}'>{res['score']}%</span>",
                            unsafe_allow_html=True,
                        )
                    with st.expander("Skill details"):
                        m_col, x_col = st.columns(2)
                        with m_col:
                            st.markdown("**✅ Matched**")
                            pills = " ".join(
                                f"<span class='pill-matched'>{s}</span>"
                                for s in res["matched"]
                            ) or "<i>None</i>"
                            st.markdown(pills, unsafe_allow_html=True)
                        with x_col:
                            st.markdown("**❌ Missing**")
                            pills = " ".join(
                                f"<span class='pill-missing'>{s}</span>"
                                for s in res["missing"]
                            ) or "<i>None</i>"
                            st.markdown(pills, unsafe_allow_html=True)
                st.write("")

        with col_right:
            st.markdown("#### Score distribution")
            df = pd.DataFrame(results)[["name", "score"]]
            df["name"] = df["name"].str.replace(".pdf", "", regex=False).str[:18]
            st.bar_chart(df.set_index("name"), height=300)

        # Export CSV
        st.divider()
        export_df = pd.DataFrame([
            {
                "File":           r["name"],
                "Score (%)":      r["score"],
                "Matched Skills": ", ".join(r["matched"]),
                "Missing Skills": ", ".join(r["missing"]),
            }
            for r in results
        ])
        csv_bytes = export_df.to_csv(index=False).encode()
        st.download_button(
            "⬇️ Export results as CSV",
            data=csv_bytes,
            file_name="careerecopilot_results.csv",
            mime="text/csv",
        )

    # ─── TAB 2: Skill Map ────────────────────────────────────────────────────
    with tab2:
        st.markdown("#### JD keyword radar")
        keywords = extract_jd_keywords(job_desc_ctx)
        if keywords:
            st.markdown("**Top keywords the ATS is weighting most from the JD:**")
            pills = " ".join(
                f"<span class='pill-keyword'>{k}</span>" for k in keywords
            )
            st.markdown(pills, unsafe_allow_html=True)
        st.divider()

        # Side-by-side candidate comparison
        st.markdown("#### Compare two candidates")
        names = [r["name"] for r in results]
        if len(names) >= 2:
            cc1, cc2 = st.columns(2)
            cand_a = cc1.selectbox("Candidate A", names, index=0, key="cmp_a")
            cand_b = cc2.selectbox("Candidate B", names, index=1, key="cmp_b")

            res_a = next(r for r in results if r["name"] == cand_a)
            res_b = next(r for r in results if r["name"] == cand_b)

            col_a, col_b = st.columns(2)
            for col, res in [(col_a, res_a), (col_b, res_b)]:
                with col:
                    st.markdown(f"**{res['name']}** — {res['score']}%")
                    st.progress(int(res["score"]) / 100)
                    matched_pills = " ".join(
                        f"<span class='pill-matched'>{s}</span>" for s in res["matched"]
                    ) or "<i>None</i>"
                    missing_pills = " ".join(
                        f"<span class='pill-missing'>{s}</span>" for s in res["missing"]
                    ) or "<i>None</i>"
                    st.markdown("✅ **Matched**", unsafe_allow_html=False)
                    st.markdown(matched_pills, unsafe_allow_html=True)
                    st.markdown("❌ **Missing**", unsafe_allow_html=False)
                    st.markdown(missing_pills, unsafe_allow_html=True)

            # Unique to each
            only_a = set(res_a["matched"]) - set(res_b["matched"])
            only_b = set(res_b["matched"]) - set(res_a["matched"])
            if only_a or only_b:
                st.divider()
                st.markdown("#### Unique strengths")
                ua_col, ub_col = st.columns(2)
                with ua_col:
                    st.markdown(f"**Only in {res_a['name'][:20]}…**")
                    for s in only_a:
                        st.markdown(f"<span class='pill-matched'>{s}</span>", unsafe_allow_html=True)
                with ub_col:
                    st.markdown(f"**Only in {res_b['name'][:20]}…**")
                    for s in only_b:
                        st.markdown(f"<span class='pill-matched'>{s}</span>", unsafe_allow_html=True)
        else:
            st.info("Upload at least 2 resumes to use the comparison view.")

    # ─── TAB 3: Advice ───────────────────────────────────────────────────────
    with tab3:
        st.markdown("#### Resume improvement plan")
        selected_name = st.selectbox(
            "Generate advice for:", [r["name"] for r in results], key="advice_sel"
        )
        selected = next(r for r in results if r["name"] == selected_name)

        if st.button("✨ Generate advice", key="gen_advice"):
            with st.spinner("Generating tailored advice…"):
                advice = get_ai_suggestions(selected["raw"], job_desc_ctx, selected["missing"])
            st.markdown(
                f"<div class='advice-box'>{advice.replace(chr(10), '<br>')}</div>",
                unsafe_allow_html=True,
            )

    # ─── TAB 4: Interview Prep ───────────────────────────────────────────────
    with tab4:
        st.markdown("#### Mock interview questions")
        st.caption("Generated from the Job Description")

        if st.button("🎯 Generate questions", key="gen_iq"):
            with st.spinner("Generating interview questions…"):
                qa_list = get_interview_questions(job_desc_ctx)
            if isinstance(qa_list, list):
                for i, item in enumerate(qa_list, 1):
                    with st.expander(f"Q{i}: {item.get('question', 'Error')}"):
                        st.markdown(f"**Ideal answer:**\n\n{item.get('answer', 'N/A')}")
            else:
                st.error("Could not format interview questions.")

    # ─── TAB 5: Cover Letter ─────────────────────────────────────────────────
    with tab5:
        st.markdown("#### Cover letter generator")
        cl_name = st.selectbox(
            "Generate cover letter for:", [r["name"] for r in results], key="cl_sel"
        )
        cl_candidate_name = st.text_input(
            "Candidate's full name (optional)", placeholder="e.g. Sarah Ahmed"
        )
        cl_selected = next(r for r in results if r["name"] == cl_name)

        if st.button("✉️ Generate cover letter", key="gen_cl"):
            with st.spinner("Writing cover letter…"):
                letter = generate_cover_letter(
                    cl_selected["raw"], job_desc_ctx, cl_candidate_name or "the candidate"
                )
            st.text_area("Cover letter", value=letter, height=350)
            st.download_button(
                "⬇️ Download as .txt",
                data=letter.encode(),
                file_name=f"cover_letter_{cl_name.replace('.pdf','')}.txt",
                mime="text/plain",
            )

else:
    # Empty state
    st.markdown("## 👋 Welcome to CareerCopilot AI")
    st.markdown(
        "Paste a **Job Description** and upload **PDF resumes** in the sidebar, "
        "then click **Analyze Candidates** to get started."
    )
    st.info(
        "**What you'll get:**\n"
        "- ATS match score per candidate\n"
        "- Matched & missing skill breakdown\n"
        "- Side-by-side candidate comparison\n"
        "- Tailored resume improvement advice\n"
        "- AI-generated interview questions\n"
        "- One-click cover letter generation\n"
        "- CSV export of all results"
    )

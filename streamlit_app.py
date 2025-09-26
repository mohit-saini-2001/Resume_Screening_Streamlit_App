# streamlit_app.py
import streamlit as st
import pdfplumber
import io
import re
from sentence_transformers import SentenceTransformer, util
import numpy as np
import os
import pickle


st.set_page_config(page_title="Smart Resume Screener", layout="wide")

# A starter skill set
SKILL_SET = {
    # Programming Languages
    "python","java","c++","c#","javascript","typescript","scala","go","rust","php","ruby","swift","kotlin",
    
    # Data Science & ML
    "machine learning","deep learning","nlp","computer vision","reinforcement learning","generative ai",
    "time series forecasting","recommendation systems","feature engineering","mlops","model deployment",
    
    # AI/ML Tools & Frameworks
    "tensorflow","pytorch","scikit-learn","keras","xgboost","lightgbm","huggingface","fastai",
    
    # Data Analytics & Visualization
    "pandas","numpy","matplotlib","seaborn","plotly","dash","streamlit","d3.js","tableau","powerbi","excel",
    "alteryx","sas","spss",
    
    # Databases & Data Engineering
    "sql","postgresql","mysql","mongodb","cassandra","redis","snowflake","bigquery","redshift",
    "hadoop","spark","apache airflow","apache kafka","etl pipelines",
    
    # Web & App Development
    "html","css","react","vue.js","angular","next.js","nodejs","express.js","django","flask","spring boot",
    "graphql","rest api design","mobile development","flutter","react native",
    
    # Cloud & DevOps
    "aws","azure","gcp","docker","kubernetes","terraform","ansible","jenkins","ci/cd pipelines",
    "openshift","cloudflare","serverless","aws lambda","gcp cloud functions","azure functions",
    
    # Operating Systems & Scripting
    "linux","bash","shell scripting","git","version control",
    
    # Cybersecurity
    "penetration testing","ethical hacking","network security","encryption standards","identity and access management",
    
    # Emerging Technologies
    "blockchain","ethereum","solidity","web3.js","smart contracts","iot","mqtt","edge computing",
    "ar/vr development","unity","unreal engine","quantum computing","qiskit","cirq",
    
    # Business & Management Skills
    "project management","program management","product management","agile","scrum","kanban",
    "waterfall methodology","business analysis","strategic planning","stakeholder management",
    "risk management","resource allocation","team leadership","conflict resolution","decision making",
    "negotiation","critical thinking","communication","presentation skills","time management",
    "change management","lean management","six sigma","design thinking",
    
    # Collaboration & Productivity Tools
    "jira","confluence","notion","microsoft teams","slack automation","asana","trello","monday.com"
}

# load SentenceTransformer model once (caches in memory)
@st.cache_resource(show_spinner=False)
def load_embedding_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

model = load_embedding_model()

# ---------------------------
# Helper Functions
# ---------------------------
def extract_text_from_pdf_bytes(pdf_bytes: bytes) -> str:
    """Extract text from PDF bytes using pdfplumber."""
    text_pages = []
    try:
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            for p in pdf.pages:
                t = p.extract_text()
                if t:
                    text_pages.append(t)
    except Exception as e:
        try:
            return pdf_bytes.decode('utf-8', errors='ignore')
        except:
            return ""
    return "\n".join(text_pages)

def clean_text_simple(text: str) -> str:
    if not text:
        return ""
    cleanText = re.sub('http\S+\s', ' ', text)
    cleanText = re.sub('RT|cc', ' ', cleanText)
    cleanText = re.sub('#\S+\s', ' ', cleanText)
    cleanText = re.sub('@\S+', '  ', cleanText)
    cleanText = re.sub('[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ', cleanText)
    cleanText = re.sub(r'[^\x00-\x7f]', ' ', cleanText)
    cleanText = re.sub('\s+', ' ', cleanText)
    text = cleanText.strip()
    return text

def extract_skills_from_text(text: str, skill_set=SKILL_SET):
    """Return two lists: matched skills and text_tokens for context."""
    text_low = text.lower()
    found = set()
    for skill in skill_set:
        # word boundary search (allow multi-word skills)
        pattern = r'\b' + re.escape(skill.lower()) + r'\b'
        if re.search(pattern, text_low):
            found.add(skill)
    return sorted(found)

def compute_embedding_similarity(resume_text: str, jd_text: str):
    """Compute cosine similarity between sentence-transformer embeddings."""
    if not resume_text or not jd_text:
        return 0.0
    emb_r = model.encode(resume_text, convert_to_tensor=True)
    emb_j = model.encode(jd_text, convert_to_tensor=True)
    cos_sim = util.pytorch_cos_sim(emb_r, emb_j).item()
    # clip to [0,1] in case model returns slight negatives
    return float(max(0.0, min(1.0, cos_sim)))

def compute_skill_overlap(resume_skills, jd_skills):
    if not jd_skills:
        return 0.0
    return len(set(resume_skills).intersection(set(jd_skills))) / max(1, len(set(jd_skills)))

def compute_years_experience_from_text(text: str):
    """
    Very naive heuristic: find 'X years' or 'X+ years' in resume.
    Returns max found (int) or 0.
    """
    matches = re.findall(r'(\d{1,2})\+?\s+years', text.lower())
    if matches:
        try:
            values = [int(m) for m in matches]
            return max(values)
        except:
            return 0
    return 0

def compute_hybrid_score(emb_sim, skill_overlap, years_exp, jd_years_req=None):
    w_emb, w_skill, w_exp = 0.60, 0.35, 0.05
    emb_part = emb_sim  # already in [0,1]
    skill_part = skill_overlap  # [0,1]
    # experience match fraction: if jd_years_req provided, clip
    if jd_years_req is not None and jd_years_req > 0:
        exp_part = min(1.0, years_exp / jd_years_req)
    else:
        # if no jd requirement, use a normalized function (cap years at 10)
        exp_part = min(1.0, years_exp / 10.0)
    combined = w_emb * emb_part + w_skill * skill_part + w_exp * exp_part
    score = combined * 100.0
    return float(max(0.0, min(100.0, score)))

def score_to_category(score):
    """Map numeric score to category Good/Medium/Poor."""
    if score >= 70:
        return "Good Fit"
    elif score >= 40:
        return "Medium Fit"
    else:
        return "Poor Fit"
    
# Load pre-trained model
le = pickle.load(open('encoder.pkl', 'rb'))
svc_model = pickle.load(open('clf.pkl', 'rb'))
tfidf = pickle.load(open('tfidf.pkl', 'rb'))

def predict(text):
    text=clean_text_simple(text)
    vectorized_text = tfidf.transform([text])
    vectorized_text = vectorized_text.toarray()
    predicted_category = svc_model.predict(vectorized_text)
    predicted_category_name = le.inverse_transform(predicted_category)
    return predicted_category_name[0]


# ---------------------------
# Streamlit UI
# ---------------------------
st.title("Smart Resume Screener — Resume + JD → Fit Score")
st.markdown("Upload a resume (PDF) or paste resume text, paste a Job Description (JD), then click **Compute**.")

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Resume Input")
    upload = st.file_uploader("Upload resume (PDF) or paste text below", type=["pdf","txt"], key="resume")
    resume_text_input = st.text_area("Or paste resume text here", height=200)
    if upload:
        bytes_data = upload.read()
        if upload.type == "application/pdf" or upload.name.lower().endswith(".pdf"):
            extracted = extract_text_from_pdf_bytes(bytes_data)
            resume_text = clean_text_simple(extracted)
        else:
            # treat as plain text
            try:
                resume_text = bytes_data.decode('utf-8', errors='ignore')
            except:
                resume_text = resume_text_input or ""
    else:
        resume_text = resume_text_input or ""

    st.write("---")
    years_exp_override = st.number_input("Years of experience (if not present in resume)", min_value=0, max_value=50, value=0, step=1)
    auto_detect_years = st.checkbox("Auto-detect 'X years' in resume text (heuristic)", value=True)

with col2:
    st.subheader("Job Description (JD) Input")
    jd_text = st.text_area("Paste job description here", height=420)
    st.write("---")
    jd_years_req = st.number_input("JD required years of experience (optional)", min_value=0, max_value=50, value=0, step=1)
    use_skill_list = st.checkbox("Use internal skill list for matching (can edit list in code)", value=True)

st.write("")
if st.button("Compute Fit Score"):
    # validate inputs
    if not resume_text or not jd_text:
        st.error("Please provide both resume (upload or paste) and job description text.")
    else:
        # clean
        resume_text_c = clean_text_simple(resume_text)
        jd_text_c = clean_text_simple(jd_text)

        # years experience
        years_from_text = compute_years_experience_from_text(resume_text_c) if auto_detect_years else 0
        years_exp = years_exp_override if years_exp_override > 0 else years_from_text

        # skills extraction (naive)
        resume_skills = extract_skills_from_text(resume_text_c, SKILL_SET) if use_skill_list else []
        jd_skills = extract_skills_from_text(jd_text_c, SKILL_SET) if use_skill_list else []

        # embeddings similarity
        with st.spinner("Computing semantic similarity..."):
            emb_sim = compute_embedding_similarity(resume_text_c, jd_text_c)

        # skill overlap
        skill_overlap = compute_skill_overlap(resume_skills, jd_skills)

        # compute hybrid score
        jd_req = jd_years_req if jd_years_req > 0 else None
        score = compute_hybrid_score(emb_sim, skill_overlap, years_exp, jd_req)

        # output
        st.success(f"Fit Score: {score:.2f} / 100")
        category = score_to_category(score)
        st.info(f"Categorical Label (simple): **{category}**")

        # Resume profile prediction
        Profile_result  = predict(resume_text_c)
        st.write(f"- Profile based on Resume only: **{Profile_result}** ")

        # detailed breakdown
        st.write("### Breakdown")
        st.write(f"- Embedding similarity (semantic): **{emb_sim:.3f}**  (weight 0.60)")
        st.write(f"- Skill overlap (fraction of JD skills present): **{skill_overlap:.3f}**  (weight 0.35)")
        st.write(f"- Years experience used: **{years_exp}**  (weight 0.05)")

        # Show matched / missing skills if using skill list
        if use_skill_list:
            st.write("### Skills (from internal skill list)")
            col_a, col_b = st.columns(2)
            with col_a:
                st.write("Resume skills found:")
                if resume_skills:
                    st.write(", ".join(resume_skills))
                else:
                    st.write("_No skills found from internal list_")
            with col_b:
                st.write("JD skills found:")
                if jd_skills:
                    st.write(", ".join(jd_skills))
                else:
                    st.write("_No skills found from internal list_")

            # matched and missing
            matched = sorted(set(resume_skills).intersection(set(jd_skills)))
            missing = sorted(set(jd_skills).difference(set(resume_skills)))
            st.write("Matched skills:", matched if matched else "None")
            st.write("Missing JD skills:", missing if missing else "None")

        # show short suggestions
        st.write("### Suggestions")
        suggestions = []
        if skill_overlap < 0.5:
            suggestions.append("Consider adding more JD-specific skills to the resume (use exact keywords).")
        if emb_sim < 0.5:
            suggestions.append("Rewrite headline/summary to include role keywords and responsibilities.")
        if years_exp < (jd_req or 3):
            suggestions.append("If possible, highlight relevant project experience to show applicable years/impact.")
        if not suggestions:
            suggestions.append("Resume appears to be a reasonable match. Highlight keywords and achievements for better fit.")
        for s in suggestions:
            st.write("- " + s)

st.write("---")
st.markdown("**Notes:** This demo uses a small internal skill list and sentence-transformer embeddings for semantic comparison. "
            "For production: expand skill list, use section-aware parsing (Experience, Education), and optionally a trained classifier on labeled pairs.")

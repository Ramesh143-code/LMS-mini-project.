import streamlit as st
import numpy as np
import re
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import PyPDF2
import docx

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------
st.set_page_config(page_title="Smart NLP Summarizer", layout="centered")

# --------------------------------------------------
# CUSTOM CSS
# --------------------------------------------------
st.markdown("""
<style>

/* Container width */
.block-container {
    max-width: 900px;
    padding-top: 40px;
}

/* Title */
.title {
    font-size: 42px;
    font-weight: 800;
    color: #1E88E5;
    margin-bottom: 10px;
}

/* Subtitle */
.subtitle {
    color: #9E9E9E;
    margin-bottom: 30px;
}

/* Section headers */
.section-header {
    font-size: 22px;
    font-weight: 600;
    margin-top: 30px;
    margin-bottom: 10px;
}

/* Summary box */
.summary-card {
    background-color: #E8F5E9;
    padding: 20px;
    border-radius: 10px;
    border: 1px solid #C8E6C9;
    color: #1B5E20;
    font-size: 16px;
    line-height: 1.6;
    margin-bottom: 25px;
}

/* Extra spacing for download */
.download-space {
    margin-top: 15px;
}

</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# HEADER
# --------------------------------------------------
st.markdown('<div class="title">📄 Smart NLP Text Summarizer</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Upload a document or paste text to generate a clean summary</div>', unsafe_allow_html=True)

# --------------------------------------------------
# LOAD MODEL (CACHE)
# --------------------------------------------------
@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

model = load_model()

# --------------------------------------------------
# TEXT PROCESSING
# --------------------------------------------------
def sentence_tokenization(text):
    sentences = re.split(r'(?<=[.!?]) +', text)
    return [s.strip() for s in sentences if len(s.split()) > 5]

def basic_cleaning(text):
    text = text.lower()
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

# --------------------------------------------------
# EMBEDDING + MMR
# --------------------------------------------------
def embedding_mmr_summary(text, word_limit=50, lambda_param=0.7):
    text = basic_cleaning(text)
    sentences = sentence_tokenization(text)

    if len(sentences) == 0:
        return "Not enough meaningful content to summarize."

    embeddings = model.encode(sentences)
    centroid = np.mean(embeddings, axis=0)

    sim_to_centroid = cosine_similarity([centroid], embeddings)[0]
    sentence_sim = cosine_similarity(embeddings)

    selected = []
    candidates = list(range(len(sentences)))

    first = np.argmax(sim_to_centroid)
    selected.append(first)
    candidates.remove(first)

    while len(selected) < len(sentences):
        mmr_scores = []
        for idx in candidates:
            relevance = sim_to_centroid[idx]
            redundancy = max(sentence_sim[idx][sel] for sel in selected)
            score = lambda_param * relevance - (1 - lambda_param) * redundancy
            mmr_scores.append((score, idx))

        best = max(mmr_scores)[1]
        selected.append(best)
        candidates.remove(best)

        summary_temp = " ".join([sentences[i] for i in sorted(selected)])
        if len(summary_temp.split()) >= word_limit:
            break

    final_summary = " ".join([sentences[i] for i in sorted(selected)])
    return " ".join(final_summary.split()[:word_limit])

# --------------------------------------------------
# FILE UPLOAD
# --------------------------------------------------
st.markdown('<div class="section-header">📂 Upload PDF or Word Document</div>', unsafe_allow_html=True)

uploaded_file = st.file_uploader("", type=["pdf", "docx"])

text_input = ""

if uploaded_file:
    if uploaded_file.type == "application/pdf":
        pdf_reader = PyPDF2.PdfReader(uploaded_file)
        for page in pdf_reader.pages:
            if page.extract_text():
                text_input += page.extract_text()
    elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        doc = docx.Document(uploaded_file)
        for para in doc.paragraphs:
            text_input += para.text + "\n"

# --------------------------------------------------
# TEXT AREA
# --------------------------------------------------
st.markdown('<div class="section-header">✍️ Or Paste Text Below</div>', unsafe_allow_html=True)

user_text = st.text_area("", height=250, value=text_input)

word_count = len(user_text.split())
st.markdown(f"**📊 Word Count:** {word_count}")

# --------------------------------------------------
# SUMMARY LENGTH
# --------------------------------------------------
st.markdown('<div class="section-header">📏 Select Summary Length (Words)</div>', unsafe_allow_html=True)

summary_length = st.slider("", 10, 500, 50)

# --------------------------------------------------
# GENERATE BUTTON (LEFT ALIGNED)
# --------------------------------------------------
generate = st.button("🚀 Generate Summary")

# --------------------------------------------------
# OUTPUT
# --------------------------------------------------
if generate:
    if word_count < 20:
        st.warning("Please enter at least 20 words.")
    else:
        summary = embedding_mmr_summary(user_text, word_limit=summary_length)

        st.markdown('<div class="section-header">📝 Generated Summary</div>', unsafe_allow_html=True)

        st.markdown(
            f'<div class="summary-card">{summary}</div>',
            unsafe_allow_html=True
        )

        st.markdown('<div class="download-space"></div>', unsafe_allow_html=True)

        st.download_button(
            label="⬇️ Download Summary",
            data=summary,
            file_name="summary.txt",
            mime="text/plain"
        )

# streamlit run app.py
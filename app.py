import streamlit as st
from pathlib import Path
import os

from scr.chunking import chunk_text_with_metadata
from scr.ingestion import load_multiple_pdfs
from scr.retriever import EmbeddingModel, VectorRetriever
from scr.config import CONFIG

import google.generativeai as genai

# -----------------------------
# STREAMLIT CONFIG (ONLY ONCE)
# -----------------------------
st.set_page_config(page_title="RAG Chatbot (Gemini)", layout="wide")

# -----------------------------
# LOAD GEMINI MODEL (CACHED)
# -----------------------------
# API KEY INPUT (UI)
# -----------------------------
st.sidebar.markdown("## 🔑 Gemini API Key")

if "api_key" not in st.session_state:
    st.session_state.api_key = ""

api_key_input = st.sidebar.text_input(
    "Enter your Gemini API Key:",
    type="password",
    value=st.session_state.api_key,
    placeholder="Paste your API key here..."
)

col1, col2 = st.sidebar.columns([1, 1])

with col1:
    if st.button("✅ Save Key"):
        if api_key_input.strip():
            st.session_state.api_key = api_key_input.strip()
            st.success("API Key saved!")
        else:
            st.warning("Please enter a valid API key.")

with col2:
    st.markdown(
        """
        <a href="https://ai.google.dev/" target="_blank">
            <button style="
                width:100%;
                padding:6px;
                border:none;
                border-radius:5px;
                background-color:#4CAF50;
                color:white;
                cursor:pointer;">
                🔗 Get Key
            </button>
        </a>
        """,
        unsafe_allow_html=True
    )

# -----------------------------
# LOAD GEMINI MODEL
# -----------------------------
@st.cache_resource
def load_model(api_key: str):
    genai.configure(api_key=api_key)
    return genai.GenerativeModel("gemini-2.5-flash")

model = None

if st.session_state.api_key:
    try:
        model = load_model(st.session_state.api_key)
        st.sidebar.success("✅ Gemini Ready")
    except Exception as e:
        st.sidebar.error(f"❌ Invalid API Key: {str(e)}")
else:
    st.warning("🔑 Please enter your Gemini API key to continue.")
    st.stop()
# -----------------------------
# DATA PATH
# -----------------------------
DATA_PATH = Path(CONFIG["pdf_path"]).parent

# -----------------------------
# SIDEBAR TOPIC SELECTION
# -----------------------------
pdf_files = list(DATA_PATH.glob("*.pdf"))
topics = [f.stem for f in pdf_files]

selected_topic = st.sidebar.selectbox(
    "📄 Select a topic",
    ["-- Select a topic --"] + topics
)

if selected_topic == "-- Select a topic --":
    st.info("📄 Please select a topic from the sidebar to continue.")
    st.stop()

selected_pdf = next(f for f in pdf_files if f.stem == selected_topic)
st.sidebar.success(f"Selected PDF: {selected_pdf.name}")

# -----------------------------
# LOAD PIPELINE (CACHED)
# -----------------------------
@st.cache_resource
def load_pipeline_single(pdf_path: Path):
    documents = load_multiple_pdfs(pdf_path.parent)
    text = documents.get(pdf_path.name, None)

    if text is None:
        raise FileNotFoundError(f"{pdf_path} not found")

    chunks = chunk_text_with_metadata(text, source=pdf_path.name)

    embedding_model = EmbeddingModel()
    retriever = VectorRetriever(embedding_model)
    retriever.build_index([c["text"] for c in chunks])
    retriever.chunk_metadata = chunks

    return retriever

# Build retriever
with st.spinner(f"⚙️ Processing '{selected_topic}'..."):
    retriever = load_pipeline_single(selected_pdf)

st.success("✅ PDF processed! You can now ask questions.")

# -----------------------------
# CUSTOM STYLING
# -----------------------------
st.markdown("""
<style>
.chat-container {
    max-height: 70vh;
    overflow-y: auto;
    padding: 10px;
    border: 1px solid #ddd;
    border-radius: 10px;
    background-color: #f5f5f5;
}
.user-message {
    text-align: right;
    padding: 8px 12px;
    margin: 5px;
    background-color: #dcf8c6;
    border-radius: 10px 0 10px 10px;
    display: inline-block;
    max-width: 70%;
}
.assistant-message {
    text-align: left;
    padding: 8px 12px;
    margin: 5px;
    background-color: #e8e8e8;
    border-radius: 0 10px 10px 10px;
    display: inline-block;
    max-width: 70%;
}
</style>
""", unsafe_allow_html=True)

# -----------------------------
# TITLE
# -----------------------------
st.title("📄 Retrieval-Augmented Generation Assistant")
st.text(
    "Ask questions about the selected PDF document. "
    "This assistant uses retrieved chunks + Gemini for accurate answers.\n"
    "Built by Dr. Muhammad Adnan Anwar."
)

# -----------------------------
# CHAT HISTORY
# -----------------------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if st.button("🧹 Clear Chat"):
    st.session_state.chat_history = []

# -----------------------------
# ANSWER GENERATION
# -----------------------------
def generate_answer(query, retrieved_chunks, history=None, top_k=5):
    context = "\n\n".join([r["text"] for r in retrieved_chunks[:top_k]])

    history_text = ""
    if history:
        for msg in history:
            history_text += f"{msg['role'].capitalize()}: {msg['message']}\n"

    prompt = f"""
You are a helpful assistant. Use the context and conversation history.

Conversation:
{history_text}

Context:
{context}

Question:
{query}

Answer clearly and concisely:
"""

    try:
        response = model.generate_content(prompt)
        return response.text.strip() if response.text else "I don't know."
    except Exception as e:
        return f"Error: {str(e)}"

# -----------------------------
# CHAT INPUT (BETTER UX)
# -----------------------------
query = st.chat_input("Ask something about the document...")

if query:
    results = retriever.search(query, top_k=5)

    answer = generate_answer(
        query,
        results,
        history=st.session_state.chat_history
    )

    st.session_state.chat_history.append(
        {"role": "user", "message": query}
    )
    st.session_state.chat_history.append(
        {"role": "assistant", "message": answer}
    )

# -----------------------------
# DISPLAY CHAT
# -----------------------------
st.markdown('<div class="chat-container">', unsafe_allow_html=True)

for chat in st.session_state.chat_history:
    if chat["role"] == "user":
        st.markdown(
            f'<div class="user-message">{chat["message"]}</div>',
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            f'<div class="assistant-message">{chat["message"]}</div>',
            unsafe_allow_html=True
        )

st.markdown('</div>', unsafe_allow_html=True)

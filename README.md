# 📄 RAG Chatbot with Google Gemini & Streamlit

A **production-ready Retrieval-Augmented Generation (RAG) chatbot** that allows users to interact with PDF documents using natural language.

Built using:
- 🔍 Semantic Search (FAISS)
- 🧠 Google Gemini (LLM)
- ⚡ Streamlit (UI)
- 🧩 Modular Python Pipeline

---

## 🚀 Live Demo

👉 *(Add your Streamlit link here after deployment)*  
`https://your-app-name.streamlit.app`

---

## 🎯 Key Features

### 📄 Document Intelligence
- Load and process PDF documents
- Multi-document support via sidebar selection
- Clean text extraction pipeline

### 🔍 Smart Retrieval
- Semantic search using embeddings
- FAISS-based vector indexing
- Context-aware chunk retrieval

### 🧠 LLM-Powered Answers
- Powered by Google Gemini (`gemini-2.5-flash`)
- Uses retrieved context for grounded responses
- Reduces hallucination via RAG pipeline

### 💬 Conversational Memory
- Maintains chat history
- Context-aware multi-turn conversations

### 🔐 API Key UI (Unique Feature ⭐)
- Users can input their own Gemini API key
- No need for backend secrets
- "Get API Key" button integrated in UI

### ⚡ Performance Optimized
- Cached pipelines using Streamlit caching
- Memory-efficient chunking
- Fast retrieval

---

## 🧱 Project Architecture

User selects PDF → Streamlit reads the PDF from data/.
Text Chunking → Converts PDF text into small chunks with metadata.
Embedding & Vector Store → Each chunk is embedded and added to FAISS for fast retrieval.
User query → Retrieved relevant chunks + conversation history → sent to Gemini API.
Response → Gemini generates an answer → Streamlit displays it in chat format.

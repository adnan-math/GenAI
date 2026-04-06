# 📄 RAG Chatbot with Google Gemini & Streamlit

A **production-ready Retrieval-Augmented Generation (RAG) chatbot** that allows users to interact with PDF documents using natural language — including **their own uploaded files**.

Built using:
- 🧠 Google Gemini (LLM)
- ⚡ Streamlit (UI)
- 🔍 Custom Vector Retrieval (no external DB)
- 🧩 Modular Python Pipeline

---

## 🚀 Live Demo

👉 https://madnan-rag.streamlit.app/

---

## 🎯 Key Features

### 📄 Document Intelligence
- Load and process PDF documents  
- Select from preloaded documents **or upload your own PDF**  
- Clean text extraction and preprocessing pipeline  

---

### 📤 User Document Upload (NEW ⭐)
- Upload custom PDFs directly from the UI  
- Automatic ingestion and indexing  
- Works seamlessly with the same RAG pipeline  

---

### 🔍 Smart Retrieval
- Lightweight custom vector retriever  
- Embedding-based semantic search  
- Relevant chunk selection without FAISS  

---

### 🧠 LLM-Powered Answers
- Powered by Google Gemini (`gemini-2.5-flash`)  
- Context-grounded responses using retrieved chunks  
- Reduced hallucination via RAG pipeline  

---

### 💬 Conversational Memory
- Maintains chat history  
- Supports multi-turn, context-aware conversations  

---

### 🔐 API Key UI (Unique Feature ⭐)
- Users can input their own Gemini API key  
- No need for backend secrets or environment variables  
- Integrated “Get API Key” button in UI  

---

### ⚡ Performance Optimized
- Streamlit caching for fast reloads  
- Efficient chunking and retrieval  
- Lightweight pipeline (no heavy vector DB)  

---

## 🧱 Project Architecture

```
rag-project/
│
├── app.py                  # Streamlit UI
│
├── scr/
│   ├── ingestion.py        # PDF loading & cleaning
│   ├── chunking.py         # Text chunking with metadata
│   ├── retriever.py        # Embedding + custom vector search
│   ├── config.py           # Configuration
│
├── data/                   # Sample PDFs
├── requirements.txt
```

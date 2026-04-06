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

+-----------------------------------------------------------+
|                       Streamlit UI                        |
|  - Sidebar: Select PDF                                    |
|  - Text Input: Ask Questions                              |
|  - Chat History Display                                   |
+-----------------------------------------------------------+
                     |
                     v
+-----------------------------------------------------------+
|                   RAG Pipeline Layer                      |
|  - PDF Loader (ingestion.py)                              |
|      * Reads multiple PDFs                                 |
|  - Text Chunking (chunking.py)                            |
|      * Word-based / Paragraph / Char-based chunks        |
|      * Metadata enriched for RAG                          |
|  - Embedding + Vector Store (retriever.py)               |
|      * EmbeddingModel                                     |
|      * VectorRetriever (FAISS)                            |
+-----------------------------------------------------------+
                     |
                     v
+-----------------------------------------------------------+
|                    Gemini API Layer                        |
|  - Google Gemini 2.5-flash                                 |
|  - Uses prompt with:                                       |
|      * Conversation History                                |
|      * Retrieved Text Chunks                               |
|  - Generates AI Response                                   |
+-----------------------------------------------------------+
                     |
                     v
+-----------------------------------------------------------+
|                  Response Rendering                        |
|  - Display answer in Streamlit UI                         |
|  - Update chat history                                     |
+-----------------------------------------------------------+

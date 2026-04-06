from typing import List, Dict, Any
import numpy as np
import re

# -----------------------------
# EMBEDDING MODEL (LIGHTWEIGHT)
# -----------------------------

class EmbeddingModel:
    """
    Simple embedding model using random vectors (or could replace with API-based embeddings like Gemini/OpenAI).
    CPU-friendly and Streamlit Cloud deployable.
    """

    def __init__(self, embedding_dim: int = 512):
        self.embedding_dim = embedding_dim

    def encode(self, texts: List[str]) -> np.ndarray:
        # For demo purposes: random embeddings
        return np.random.rand(len(texts), self.embedding_dim).astype("float32")

    def encode_single(self, text: str) -> np.ndarray:
        return self.encode([text])[0]


# -----------------------------
# VECTOR RETRIEVER
# -----------------------------

class VectorRetriever:
    def __init__(self, embedding_model: EmbeddingModel):
        self.embedding_model = embedding_model
        self.embeddings = None
        self.chunks = []

    # -------------------------
    # BUILD INDEX
    # -------------------------
    def build_index(self, chunks: List[str]):
        self.chunks = chunks
        self.embeddings = self.embedding_model.encode(chunks)
        print(f"[INFO] Index built with {len(chunks)} chunks.")

    # -------------------------
    # SEARCH FUNCTION
    # -------------------------
    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        if self.embeddings is None:
            raise ValueError("Index not built. Call build_index() first.")

        query_vec = self.embedding_model.encode_single(query)
        query_vec = query_vec / np.linalg.norm(query_vec)  # normalize

        # cosine similarity
        norms = np.linalg.norm(self.embeddings, axis=1, keepdims=True)
        embeddings_norm = self.embeddings / norms
        sims = embeddings_norm @ query_vec

        top_indices = sims.argsort()[::-1][:top_k]

        results = []
        for rank, idx in enumerate(top_indices):
            results.append({
                "rank": rank + 1,
                "chunk_id": idx,
                "text": self.chunks[idx],
                "score": float(sims[idx])
            })
        return results


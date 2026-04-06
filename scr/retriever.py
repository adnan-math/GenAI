from typing import List, Dict, Any
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from scr.config import CONFIG
import re

# -----------------------------
# EMBEDDING MODEL
# -----------------------------

class EmbeddingModel:
    """
    Wrapper around sentence-transformers model.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def encode(self, texts: List[str]) -> np.ndarray:
        embeddings = self.model.encode(texts, show_progress_bar=False)
        return np.array(embeddings).astype("float32")

    def encode_single(self, text: str) -> np.ndarray:
        return self.encode([text])[0]


# -----------------------------
# RETRIEVER CLASS
# -----------------------------

class VectorRetriever:
    def __init__(self, embedding_model: EmbeddingModel):
        self.embedding_model = embedding_model
        self.index = None
        self.chunks = []

    # -------------------------
    # BUILD INDEX
    # -------------------------

    def build_index(self, chunks: List[str]):
        """
        Build FAISS index from text chunks.
        """

        self.chunks = chunks

        embeddings = self.embedding_model.encode(chunks)

        dimension = embeddings.shape[1]

        # L2 similarity index
        self.index = faiss.IndexFlatL2(dimension)

        self.index.add(embeddings)

        print(f"[INFO] FAISS index built with {len(chunks)} chunks")

    # -------------------------
    # SEARCH FUNCTION
    # -------------------------

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Perform semantic search over chunks.
        """

        if self.index is None:
            raise ValueError("Index not built. Call build_index() first.")

        query_embedding = self.embedding_model.encode_single(query)
        query_embedding = np.array([query_embedding]).astype("float32")

        distances, indices = self.index.search(query_embedding, top_k)

        results = []

        for rank, idx in enumerate(indices[0]):
            # Skip invalid indices
            if idx == -1 or idx >= len(self.chunks):
                continue

            results.append({
                "rank": rank + 1,
                "chunk_id": idx,
                "text": self.chunks[idx],
                "score": float(distances[0][rank])
            })

        return results
    
    


# -----------------------------
# TEST RUN
# -----------------------------

if __name__ == "__main__":
    from ingestion import process_pdf
    from chunking import chunk_text

    # Step 1: Load document
    text = process_pdf(CONFIG["pdf_path"])
    text = re.sub(r'\d{2,4}-\d{2,4}', '', text)  # remove ISBN, years
    text = re.sub(r'\bDOI\b.*', '', text)
    text = re.sub(r'[^A-Za-z0-9.,;:() \n]+', ' ', text)  # remove non-text chars

    # Step 2: Chunk document
    chunks = chunk_text(text)

    # Step 3: Build retriever
    embedding_model = EmbeddingModel()
    retriever = VectorRetriever(embedding_model)

    retriever.build_index(chunks)

    # Step 4: Query system
    query = "What is the purpose of finite element method?"

    results = retriever.search(query, top_k=10)

    print("\nTop Results:\n")

    for r in results:
        print(f"Rank {r['rank']}")
        print(f"Score: {r['score']}")
        print(r["text"][:300])
        print("-" * 50)
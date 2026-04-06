from typing import List, Dict, Any
import re
from scr.config import CONFIG


# -----------------------------
# TEXT NORMALIZATION
# -----------------------------
def normalize_text(text: str) -> str:
    """
    Normalize text while preserving paragraph structure.
    - Removes extra spaces
    - Keeps meaningful line breaks
    """
    text = re.sub(r'[ \t]+', ' ', text)   # collapse spaces/tabs
    text = re.sub(r'\n+', '\n', text)     # normalize newlines
    return text.strip()


# -----------------------------
# STREAMING WORD-BASED CHUNKING
# -----------------------------
def chunk_text_streaming(
    text: str,
    chunk_size: int = None,
    overlap: int = None
) -> List[str]:
    """
    Memory-efficient word-based chunking:
    - Suitable for large documents
    - Maintains overlap between chunks
    """
    if chunk_size is None:
        chunk_size = CONFIG["chunk_size"]
    if overlap is None:
        overlap = CONFIG["overlap"]

    text = normalize_text(text)
    paragraphs = re.split(r'\n\s*\n', text)

    chunks = []
    current_chunk = []
    current_len = 0

    for para in paragraphs:
        words = para.split()
        if not words:
            continue

        idx = 0
        while idx < len(words):
            remaining_space = chunk_size - current_len
            slice_end = min(idx + remaining_space, len(words))

            current_chunk.extend(words[idx:slice_end])
            current_len += slice_end - idx
            idx = slice_end

            if current_len >= chunk_size:
                chunk_text = " ".join(current_chunk)
                chunks.append(chunk_text)

                # Apply overlap
                overlap_words = current_chunk[-overlap:] if overlap > 0 else []
                current_chunk = overlap_words
                current_len = len(current_chunk)

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks


# -----------------------------
# PARAGRAPH-BASED CHUNKING
# -----------------------------
def chunk_by_paragraphs(
    text: str,
    chunk_size: int = None,
    overlap: int = None
) -> List[str]:
    """
    Paragraph-aware chunking with overlap.
    """
    if chunk_size is None:
        chunk_size = CONFIG["chunk_size"]
    if overlap is None:
        overlap = CONFIG["overlap"]

    text = normalize_text(text)
    paragraphs = re.split(r'\n\s*\n', text)

    chunks = []
    current_chunk = []
    current_len = 0

    for para in paragraphs:
        words = para.split()
        if not words:
            continue

        if current_len + len(words) <= chunk_size:
            current_chunk.extend(words)
            current_len += len(words)
        else:
            if current_chunk:
                chunks.append(" ".join(current_chunk))

            overlap_words = current_chunk[-overlap:] if overlap > 0 else []
            current_chunk = overlap_words + words
            current_len = len(current_chunk)

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks


# -----------------------------
# CHARACTER-BASED CHUNKING
# -----------------------------
def chunk_by_characters(
    text: str,
    chunk_size: int = 1000,
    overlap: int = 200
) -> List[str]:
    """
    Character-based fallback chunking.
    Useful for poorly formatted text.
    """
    text = normalize_text(text)
    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = max(end - overlap, 0)

    return chunks


# -----------------------------
# MAIN CHUNKING PIPELINE
# -----------------------------
def chunk_text(
    text: str,
    method: str = "words",
    chunk_size: int = None,
    overlap: int = None
) -> List[str]:
    """
    High-level chunking interface.
    Cleans text and applies selected strategy.
    """
    if chunk_size is None:
        chunk_size = CONFIG["chunk_size"]
    if overlap is None:
        overlap = CONFIG["overlap"]

    text = normalize_text(text)

    # Light cleaning (preserve scientific notation)
    text = re.sub(r'\bDOI\b.*', '', text)
    text = re.sub(r'\d{4}-\d{4}', '', text)  # remove year ranges if needed
    text = re.sub(r'[^A-Za-z0-9.,;:()/%+\- \n]+', ' ', text)

    if method == "words":
        chunks = chunk_text_streaming(text, chunk_size, overlap)
    elif method == "paragraphs":
        chunks = chunk_by_paragraphs(text, chunk_size, overlap)
    elif method == "chars":
        chunks = chunk_by_characters(text, chunk_size, overlap)
    else:
        raise ValueError("method must be 'words', 'paragraphs', or 'chars'")

    # Filter out very small chunks (improves retrieval quality)
    chunks = [c for c in chunks if len(c.split()) > 30]

    return chunks


# -----------------------------
# METADATA-ENRICHED CHUNKING
# -----------------------------
def chunk_text_with_metadata(
    text: str,
    source: str = "unknown",
    chunk_size: int = None,
    overlap: int = None,
    method: str = "words"
) -> List[Dict[str, Any]]:
    """
    Returns chunks with metadata for RAG:
    - chunk_id
    - source
    - text
    - length
    """
    chunks = chunk_text(
        text,
        method=method,
        chunk_size=chunk_size,
        overlap=overlap
    )

    return [
        {
            "chunk_id": i,
            "source": source,
            "text": c,
            "length": len(c)
        }
        for i, c in enumerate(chunks)
    ]


# -----------------------------
# TEST RUN
# -----------------------------
if __name__ == "__main__":
    from pathlib import Path
    from scr.ingestion import process_pdf

    pdf_path = Path(CONFIG["pdf_path"])

    print(f"[INFO] Loading PDF: {pdf_path}")
    text = process_pdf(pdf_path)

    print(f"[INFO] Characters: {len(text)}")

    chunks = chunk_text(text, method="words")
    print(f"[INFO] Total chunks: {len(chunks)}")
    print(f"\nSample chunk:\n{chunks[0][:500]}")

    chunks_meta = chunk_text_with_metadata(text, source=pdf_path.name)
    print(f"\nSample metadata:\n{chunks_meta[0]}")
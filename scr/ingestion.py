from pypdf import PdfReader
from pathlib import Path
import re
from typing import List, Dict, Union
from scr.config import CONFIG



# -----------------------------
# PDF LOADING
# -----------------------------

def load_pdf(file_path: Union[str, Path]) -> str:
    """
    Extract raw text from a PDF file.

    Args:
        file_path (str or Path): Path to PDF file

    Returns:
        str: Extracted raw text
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    reader = PdfReader(str(file_path))

    text_pages = []
    for page_num, page in enumerate(reader.pages):
        try:
            page_text = page.extract_text()
            if page_text:
                text_pages.append(page_text)
        except Exception as e:
            print(f"[WARN] Failed to read page {page_num}: {e}")

    return "\n".join(text_pages)


# -----------------------------
# TEXT CLEANING
# -----------------------------

def clean_text(text: str) -> str:
    """
    Clean extracted PDF text for NLP/RAG pipelines.

    Args:
        text (str): Raw extracted text

    Returns:
        str: Cleaned text
    """

    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text)

    # Remove page numbers (basic heuristic)
    text = re.sub(r'\b\d{1,4}\b(?=\s)', '', text)

    # Remove non-ASCII characters (optional, depends on use case)
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)

    # Remove excessive punctuation noise
    text = re.sub(r'\.{2,}', '.', text)

    return text.strip()


# -----------------------------
# FULL PIPELINE
# -----------------------------

def process_pdf(file_path: Union[str, Path]) -> str:
    """
    Full pipeline: PDF → Raw text → Clean text

    Args:
        file_path (str or Path): PDF path

    Returns:
        str: Cleaned document text
    """
    raw_text = load_pdf(file_path)
    cleaned_text = clean_text(raw_text)
    return cleaned_text


# -----------------------------
# MULTI-FILE SUPPORT (Optional but recommended)
# -----------------------------

def load_multiple_pdfs(folder_path: Union[str, Path]) -> Dict[str, str]:
    """
    Load and process all PDFs in a folder.

    Args:
        folder_path (str or Path): Folder containing PDFs

    Returns:
        dict: {filename: cleaned_text}
    """
    folder_path = Path(folder_path)

    if not folder_path.exists():
        raise FileNotFoundError(f"Folder not found: {folder_path}")

    pdf_files = list(folder_path.glob("*.pdf"))

    documents = {}

    for pdf in pdf_files:
        try:
            print(f"[INFO] Processing: {pdf.name}")
            documents[pdf.name] = process_pdf(pdf)
        except Exception as e:
            print(f"[ERROR] Failed to process {pdf.name}: {e}")

    return documents


# -----------------------------
# DEBUG / TEST RUN
# -----------------------------

if __name__ == "__main__":
    text = process_pdf(CONFIG["pdf_path"])

    try:
        text = process_pdf(CONFIG["pdf_path"])

        print("\n==============================")
        print("TEXT EXTRACTION SUCCESSFUL")
        print("==============================")
        print(f"Characters: {len(text)}")
        print("\nPreview:\n")
        print(text[:1000])

    except Exception as e:
        print(f"[ERROR] {e}")
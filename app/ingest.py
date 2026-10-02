"""Ingestion pipeline: documents -> chunks -> embeddings -> FAISS vector store.

Run it with:   uv run python -m app.ingest
"""

import json
import logging

import faiss
from docx import Document
from pypdf import PdfReader

from app.config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    CHUNKS_DIR,
    CHUNKS_PATH,
    DATA_DIR,
    INDEX_DIR,
    INDEX_PATH,
    setup_logging,
)
from app.embeddings import embed_texts

logger = logging.getLogger(__name__)

SUPPORTED_TYPES = {".pdf", ".txt", ".docx"}


# ---------------------------------------------------------------------------
# STEP 1: Load documents
# Each loader returns a list of "pages": {"text", "source", "page"}
# ---------------------------------------------------------------------------
def load_pdf(path):
    pages = []
    reader = PdfReader(str(path))
    for number, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if text:
            pages.append({"text": text, "source": path.name, "page": number})
    return pages


def load_txt(path):
    text = path.read_text(encoding="utf-8", errors="ignore").strip()
    # Text files have no pages, so page is None
    return [{"text": text, "source": path.name, "page": None}] if text else []


def load_docx(path):
    document = Document(str(path))
    text = "\n".join(p.text for p in document.paragraphs if p.text.strip())
    # Word files have no fixed pages either, so page is None
    return [{"text": text, "source": path.name, "page": None}] if text else []


def load_documents():
    """Load every supported file from the data/ folder."""
    if not DATA_DIR.exists():
        raise FileNotFoundError(f"Data folder not found: {DATA_DIR}")

    loaders = {".pdf": load_pdf, ".txt": load_txt, ".docx": load_docx}
    pages = []

    for path in sorted(DATA_DIR.iterdir()):
        if path.suffix.lower() not in SUPPORTED_TYPES:
            continue
        try:
            loaded = loaders[path.suffix.lower()](path)
            logger.info("Loaded %-45s -> %d page(s)/section(s)", path.name, len(loaded))
            pages.extend(loaded)
        except Exception as error:  # one bad file should not stop everything
            logger.warning("Skipping %s because of an error: %s", path.name, error)

    if not pages:
        raise FileNotFoundError(
            f"No readable .pdf/.txt/.docx files found in {DATA_DIR}. "
            "Add your documents there and try again."
        )
    return pages


# ---------------------------------------------------------------------------
# STEP 2: Split text into chunks
# ---------------------------------------------------------------------------
def split_text(text, chunk_size, overlap):
    """Cut long text into pieces of about `chunk_size` characters.

    - We try to cut at a paragraph / sentence / space, so words are not broken.
    - Each chunk repeats the last `overlap` characters of the previous one,
      so a sentence cut in half is still fully visible in one of the chunks.
    """
    chunks = []
    start = 0

    while start < len(text):
        end = min(start + chunk_size, len(text))

        if end < len(text):
            # Look for a nice place to cut in the last 40% of this chunk
            search_from = start + int(chunk_size * 0.6)
            for separator in ["\n\n", "\n", ". ", " "]:
                position = text.rfind(separator, search_from, end)
                if position != -1:
                    end = position + len(separator)
                    break

        piece = text[start:end].strip()
        if piece:
            chunks.append(piece)

        if end >= len(text):
            break
        start = max(end - overlap, start + 1)  # step back a little (overlap)

    return chunks


def create_chunks(pages):
    """Split every page and remember where each chunk came from."""
    chunks = []
    for page in pages:
        for piece in split_text(page["text"], CHUNK_SIZE, CHUNK_OVERLAP):
            chunks.append(
                {
                    "id": len(chunks),
                    "text": piece,
                    "source": page["source"],
                    "page": page["page"],
                }
            )
    return chunks


# ---------------------------------------------------------------------------
# STEP 3 + 4: Embed the chunks and save everything to disk
# ---------------------------------------------------------------------------
def save_store(embeddings, chunks):
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    CHUNKS_DIR.mkdir(parents=True, exist_ok=True)

    # IndexFlatIP = exact search using inner product (cosine for normalized vectors)
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    faiss.write_index(index, str(INDEX_PATH))

    # Save the readable text of each chunk (position i here == vector i in FAISS)
    CHUNKS_PATH.write_text(json.dumps(chunks, indent=2, ensure_ascii=False), encoding="utf-8")


def build_index():
    """Run the full pipeline. Returns the number of chunks stored."""
    pages = load_documents()
    logger.info("Loaded %d page(s)/section(s) in total", len(pages))

    chunks = create_chunks(pages)
    logger.info("Created %d chunks (size=%d, overlap=%d)", len(chunks), CHUNK_SIZE, CHUNK_OVERLAP)

    logger.info("Creating embeddings with OpenAI (this can take a few seconds)...")
    embeddings = embed_texts([chunk["text"] for chunk in chunks])

    save_store(embeddings, chunks)
    logger.info("Saved FAISS index to %s", INDEX_PATH)
    logger.info("Saved chunks to %s", CHUNKS_PATH)
    return len(chunks)


def main():
    setup_logging()
    try:
        total = build_index()
        print(f"\nDone! {total} chunks are stored. You can now start the chatbot.")
    except Exception as error:
        logger.error("Ingestion failed: %s", error)
        raise SystemExit(1)


if __name__ == "__main__":
    main()

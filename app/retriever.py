"""Retriever: question -> embedding -> FAISS search -> most relevant chunks."""

import json
import logging

import faiss

from app.config import CHUNKS_PATH, INDEX_PATH, MIN_SCORE, TOP_K
from app.embeddings import embed_texts

logger = logging.getLogger(__name__)

# Keep the loaded index in memory so we do not read the files for every question
_cache = {"stamp": None, "index": None, "chunks": None}


def store_exists() -> bool:
    return INDEX_PATH.exists() and CHUNKS_PATH.exists()


def load_store():
    """Load the FAISS index and chunks (reloads automatically after re-ingestion)."""
    if not store_exists():
        raise FileNotFoundError(
            "The vector store was not found. Run ingestion first:  uv run python -m app.ingest"
        )

    stamp = INDEX_PATH.stat().st_mtime
    if _cache["stamp"] != stamp:
        _cache["index"] = faiss.read_index(str(INDEX_PATH))
        _cache["chunks"] = json.loads(CHUNKS_PATH.read_text(encoding="utf-8"))
        _cache["stamp"] = stamp
        logger.debug("Loaded vector store with %d chunks", len(_cache["chunks"]))

    return _cache["index"], _cache["chunks"]


def retrieve(query: str, top_k: int = TOP_K) -> list[dict]:
    """Return the top-k chunks for the question, best match first.

    Each result looks like: {"text", "source", "page", "score"}
    Chunks whose score is below MIN_SCORE are dropped (not relevant enough).
    """
    index, chunks = load_store()

    query_vector = embed_texts([query])
    scores, ids = index.search(query_vector, min(top_k, index.ntotal))

    results = []
    for score, chunk_id in zip(scores[0], ids[0]):
        if chunk_id == -1 or score < MIN_SCORE:
            continue
        chunk = chunks[chunk_id]
        results.append(
            {
                "text": chunk["text"],
                "source": chunk["source"],
                "page": chunk["page"],
                "score": float(score),
            }
        )

    logger.debug("Retrieved %d chunk(s) for: %s", len(results), query)
    return results

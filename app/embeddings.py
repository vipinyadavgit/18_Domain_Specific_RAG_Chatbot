"""Turn text into embeddings (lists of numbers) using OpenAI.

Used in two places:
  - ingest.py    -> to embed every document chunk
  - retriever.py -> to embed the user's question
"""

import faiss
import numpy as np

from app.config import EMBEDDING_MODEL, get_openai_client

BATCH_SIZE = 100  # how many texts we send to OpenAI in one request


def embed_texts(texts: list[str]) -> np.ndarray:
    """Return a (number_of_texts x vector_size) array of normalized embeddings.

    Normalizing makes every vector length 1, so "inner product" search
    in FAISS is the same as cosine similarity (1.0 = identical meaning).
    """
    client = get_openai_client()
    vectors = []

    for start in range(0, len(texts), BATCH_SIZE):
        batch = texts[start : start + BATCH_SIZE]
        response = client.embeddings.create(model=EMBEDDING_MODEL, input=batch)
        vectors.extend(item.embedding for item in response.data)

    array = np.array(vectors, dtype="float32")
    faiss.normalize_L2(array)
    return array

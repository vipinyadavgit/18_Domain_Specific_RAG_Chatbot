"""Central settings for the whole project.

Everything that can change (paths, model names, chunk size, top-k) lives here,
so the other files stay short and easy to read.
"""

import logging
import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

# ---------------------------------------------------------------------------
# Paths (built from this file's location, so nothing is hardcoded)
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
INDEX_DIR = BASE_DIR / "store" / "vector_index"
CHUNKS_DIR = BASE_DIR / "store" / "chunks"
INDEX_PATH = INDEX_DIR / "index.faiss"
CHUNKS_PATH = CHUNKS_DIR / "chunks.json"

# Read the values from the .env file (if it exists)
load_dotenv(BASE_DIR / ".env")


def _read_int(name: str, default: int) -> int:
    """Read an integer from the environment, or use the default."""
    try:
        return int(os.getenv(name, default))
    except ValueError:
        return default


def _read_float(name: str, default: float) -> float:
    """Read a decimal number from the environment, or use the default."""
    try:
        return float(os.getenv(name, default))
    except ValueError:
        return default


# ---------------------------------------------------------------------------
# Settings (all can be changed in the .env file)
# ---------------------------------------------------------------------------
CHAT_MODEL = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")
EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")

CHUNK_SIZE = _read_int("CHUNK_SIZE", 800)
CHUNK_OVERLAP = _read_int("CHUNK_OVERLAP", 100)
TOP_K = _read_int("TOP_K", 4)
MIN_SCORE = _read_float("MIN_SCORE", 0.10)

# Overlap must be smaller than the chunk size, otherwise chunking never moves forward
if CHUNK_OVERLAP >= CHUNK_SIZE:
    CHUNK_OVERLAP = CHUNK_SIZE // 5


def setup_logging() -> None:
    """Print helpful messages like: 2025-01-01 10:00:00 | INFO | Loaded 3 files."""
    level_name = os.getenv("LOG_LEVEL", "INFO").upper()
    logging.basicConfig(
        level=getattr(logging, level_name, logging.INFO),
        format="%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%H:%M:%S",
    )
    # Silence noisy library logs
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)


def get_openai_client() -> OpenAI:
    """Create the OpenAI client using the key stored in .env."""
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key or api_key == "your_openai_api_key_here":
        raise ValueError(
            "OPENAI_API_KEY is missing. Open the .env file and paste your OpenAI key."
        )
    return OpenAI(api_key=api_key)

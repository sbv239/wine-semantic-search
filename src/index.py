"""
index.py — build a FAISS IndexFlatL2 from embeddings and persist to disk.

Notes:
- We use IndexFlatIP (inner product) because embeddings are L2-normalised,
  so inner product == cosine similarity.  Higher score = more similar.
- IndexFlatIP gives exact (brute-force) search — fine for ~10k wines.
"""

import logging
import os

import faiss
import numpy as np

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def build_index(embeddings: np.ndarray) -> faiss.IndexFlatIP:
    """
    Build a FAISS flat inner-product index from a float32 embedding matrix.

    Args:
        embeddings: np.ndarray of shape (N, dim), already L2-normalised.

    Returns:
        Populated faiss.IndexFlatIP.
    """
    if embeddings.dtype != np.float32:
        embeddings = embeddings.astype(np.float32)

    dim = embeddings.shape[1]
    logger.info(f"Building IndexFlatIP  dim={dim}  n={len(embeddings)}")
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    logger.info(f"Index total vectors: {index.ntotal}")
    return index


def save_index(index: faiss.IndexFlatIP, path: str = config.FAISS_INDEX_PATH) -> None:
    """Persist FAISS index to disk."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    faiss.write_index(index, path)
    logger.info(f"Saved FAISS index → {path}")


def load_index(path: str = config.FAISS_INDEX_PATH) -> faiss.IndexFlatIP:
    """Load FAISS index from disk."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"FAISS index not found at '{path}'. "
            "Run scripts/build_index.py first."
        )
    index = faiss.read_index(path)
    logger.info(f"Loaded FAISS index from {path}  ntotal={index.ntotal}")
    return index


def build_and_save(
    embeddings: np.ndarray | None = None,
    embeddings_path: str = config.EMBEDDINGS_PATH,
    index_path: str = config.FAISS_INDEX_PATH,
) -> faiss.IndexFlatIP:
    """
    Build index from embeddings (loaded from disk if not passed) and save.
    Returns the built index.
    """
    if embeddings is None:
        logger.info(f"Loading embeddings from {embeddings_path}")
        embeddings = np.load(embeddings_path)

    index = build_index(embeddings)
    save_index(index, index_path)
    return index


if __name__ == "__main__":
    build_and_save()
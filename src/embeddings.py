"""
embeddings.py — encode wine descriptions with SentenceTransformer, save as .npy
"""

import logging
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def load_descriptions(data_path: str = config.DATA_PROCESSED_PATH) -> tuple[pd.DataFrame, list[str]]:
    """Load cleaned dataset and return (dataframe, list of descriptions)."""
    logger.info(f"Loading dataset from {data_path}")
    df = pd.read_csv(data_path)
    descriptions = df["description"].tolist()
    logger.info(f"Loaded {len(descriptions)} descriptions")
    return df, descriptions


def encode_descriptions(
    descriptions: list[str],
    model_name: str = config.EMBEDDING_MODEL,
    batch_size: int = 64,
    show_progress: bool = True,
) -> np.ndarray:
    """
    Encode a list of descriptions into embeddings.

    Returns:
        np.ndarray of shape (N, embedding_dim), dtype float32
    """
    logger.info(f"Loading model: {model_name}")
    model = SentenceTransformer(model_name)

    logger.info(f"Encoding {len(descriptions)} descriptions (batch_size={batch_size})")
    embeddings = model.encode(
        descriptions,
        batch_size=batch_size,
        show_progress_bar=show_progress,
        convert_to_numpy=True,
        normalize_embeddings=True,   # L2-normalise → cosine sim == dot product
    )
    logger.info(f"Embeddings shape: {embeddings.shape}")
    return embeddings.astype(np.float32)


def save_embeddings(embeddings: np.ndarray, path: str = config.EMBEDDINGS_PATH) -> None:
    """Save embeddings array to disk as .npy."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.save(path, embeddings)
    logger.info(f"Saved embeddings → {path}  ({embeddings.shape})")


def load_embeddings(path: str = config.EMBEDDINGS_PATH) -> np.ndarray:
    """Load embeddings array from disk."""
    embeddings = np.load(path)
    logger.info(f"Loaded embeddings from {path}  shape={embeddings.shape}")
    return embeddings


def build_and_save(
    data_path: str = config.DATA_PROCESSED_PATH,
    embeddings_path: str = config.EMBEDDINGS_PATH,
    model_name: str = config.EMBEDDING_MODEL,
) -> np.ndarray:
    """Full pipeline: load → encode → save. Returns the embeddings array."""
    _, descriptions = load_descriptions(data_path)
    embeddings = encode_descriptions(descriptions, model_name=model_name)
    save_embeddings(embeddings, embeddings_path)
    return embeddings


if __name__ == "__main__":
    build_and_save()
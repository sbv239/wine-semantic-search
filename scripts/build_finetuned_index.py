"""
scripts/build_finetuned_index.py

Rebuild embeddings + FAISS index for a finetuned model.
Run once after training, before evaluating the finetuned model.

Usage:
    python scripts/build_finetuned_index.py --model models/finetuned_.../final
"""

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
import faiss
from sentence_transformers import SentenceTransformer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def build_finetuned_index(model_path: str) -> None:
    log.info("Loading data from %s", config.DATA_PROCESSED_PATH)
    df           = pd.read_csv(config.DATA_PROCESSED_PATH)
    descriptions = df["description"].fillna("").tolist()
    log.info("Loaded %d wines", len(df))

    log.info("Loading model: %s", model_path)
    model = SentenceTransformer(model_path)

    log.info("Encoding %d descriptions…", len(descriptions))
    t0         = time.time()
    embeddings = model.encode(
        descriptions,
        batch_size=64,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    ).astype("float32")
    log.info("Encoded in %.1fs — shape: %s", time.time() - t0, embeddings.shape)

    np.save(config.FINETUNED_EMBEDDINGS_PATH, embeddings)
    log.info("Embeddings → %s", config.FINETUNED_EMBEDDINGS_PATH)

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    faiss.write_index(index, config.FINETUNED_INDEX_PATH)
    log.info("FAISS index → %s (%d vectors)", config.FINETUNED_INDEX_PATH, index.ntotal)

    log.info("Done. Run evaluation:")
    log.info("  python -m src.evaluation --model %s", model_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build FAISS index for a finetuned SentenceTransformer model."
    )
    parser.add_argument("--model", required=True,
                        help="Path to finetuned model dir (e.g. models/finetuned_.../final)")
    args = parser.parse_args()

    if not Path(args.model).exists():
        print(f"Model path not found: {args.model}")
        sys.exit(1)

    build_finetuned_index(args.model)


if __name__ == "__main__":
    main()
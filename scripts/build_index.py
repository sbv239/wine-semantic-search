"""
build_index.py — one-time pipeline runner.

Usage:
    python scripts/build_index.py

Steps:
    1. Preprocess raw CSV → wines_clean.csv  (skipped if already exists)
    2. Encode descriptions → embeddings.npy
    3. Build FAISS index   → faiss_index.bin
"""

import logging
import os
import sys
import time

# Ensure project root is on the path regardless of working directory
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import config
from src.preprocessing import preprocess
from src.embeddings import build_and_save as build_embeddings
from src.index import build_and_save as build_index

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)


def main() -> None:
    total_start = time.time()

    # ── Step 1: Preprocessing ────────────────────────────────────────────────
    if os.path.exists(config.DATA_PROCESSED_PATH):
        logger.info(f"[Step 1] Cleaned data already exists at {config.DATA_PROCESSED_PATH} — skipping.")
    else:
        logger.info("[Step 1] Preprocessing raw data…")
        t0 = time.time()
        preprocess(config.DATA_RAW_PATH, config.DATA_PROCESSED_PATH)
        logger.info(f"[Step 1] Done in {time.time() - t0:.1f}s")

    # ── Step 2: Embeddings ───────────────────────────────────────────────────
    if os.path.exists(config.EMBEDDINGS_PATH):
        logger.info(f"[Step 2] Embeddings already exist at {config.EMBEDDINGS_PATH} — skipping.")
    else:
        logger.info("[Step 2] Encoding descriptions → embeddings…")
        t0 = time.time()
        build_embeddings()
        logger.info(f"[Step 2] Done in {time.time() - t0:.1f}s")

    # ── Step 3: FAISS index ──────────────────────────────────────────────────
    if os.path.exists(config.FAISS_INDEX_PATH):
        logger.info(f"[Step 3] FAISS index already exists at {config.FAISS_INDEX_PATH} — skipping.")
    else:
        logger.info("[Step 3] Building FAISS index…")
        t0 = time.time()
        build_index()
        logger.info(f"[Step 3] Done in {time.time() - t0:.1f}s")

    logger.info(f"Pipeline complete in {time.time() - total_start:.1f}s")
    logger.info("  %-30s %s", "Cleaned data:", config.DATA_PROCESSED_PATH)
    logger.info("  %-30s %s", "Embeddings:", config.EMBEDDINGS_PATH)
    logger.info("  %-30s %s", "FAISS index:", config.FAISS_INDEX_PATH)


if __name__ == "__main__":
    main()
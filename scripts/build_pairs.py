"""
scripts/build_pairs.py

One-time pair building script. Run once before training.

Outputs:
    data/processed/train_wines.csv
    data/processed/val_wines.csv
    data/processed/train_pairs.parquet
    data/processed/val_pairs.parquet
    data/processed/pairs_meta.json

Usage:
    python scripts/build_pairs.py
"""

from __future__ import annotations

import json
import logging
import random
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from src.finetune import parse_primary_grape, compute_pair_score, _norm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

PAIRS_DIR = Path(config.PAIRS_DIR)


# ---------------------------------------------------------------------------
# Train / val split
# ---------------------------------------------------------------------------

def split_wines(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Stratified 80/20 split by Colour × Body."""
    df = df.copy()
    df["_stratum"] = (
        df["Colour"].fillna("Unknown").str.strip() + "_" +
        df["Body"].fillna("Unknown").str.strip()
    )
    # Merge rare strata (< 2 samples) into Other_Other
    counts = df["_stratum"].value_counts()
    rare   = counts[counts < 2].index
    df.loc[df["_stratum"].isin(rare), "_stratum"] = "Other_Other"

    train_df, val_df = train_test_split(
        df,
        test_size=config.VAL_SIZE,
        stratify=df["_stratum"],
        random_state=config.RANDOM_STATE,
    )
    train_df = train_df.drop(columns=["_stratum"]).reset_index(drop=True)
    val_df   = val_df.drop(columns=["_stratum"]).reset_index(drop=True)

    log.info("Split: %d train wines / %d val wines (80%% / 20%%)", len(train_df), len(val_df))
    return train_df, val_df


# ---------------------------------------------------------------------------
# Pair scoring — grouped by (Colour, Body) to reduce O(n²) cost
# ---------------------------------------------------------------------------

def score_pairs_grouped(
    df: pd.DataFrame,
    embeddings: np.ndarray | None,
    max_positive: int,
    hard_neg_ratio: float,
    easy_neg_ratio: float,
    label: str = "train",
) -> pd.DataFrame:
    """
    Build and score pairs grouped by (Colour × Body).

    Within-group: score every pair → collect positives + hard negatives.
    Cross-group:  sample easy negatives (different Colour guaranteed).

    Returns DataFrame with columns: desc_a, desc_b, label, pair_type.
    """
    records  = df.reset_index(drop=True)
    desc_col = "description"

    positive_rows:       list[dict] = []
    hard_neg_candidates: list[dict] = []

    records["_group"] = (
        records["Colour"].fillna("Unknown").str.strip() + "_" +
        records["Body"].fillna("Unknown").str.strip()
    )
    groups       = records.groupby("_group")
    total_groups = len(groups)
    log.info("[%s] Scoring pairs across %d groups (Colour × Body)…", label, total_groups)

    for g_idx, (group_name, group_df) in enumerate(groups, start=1):
        idx = group_df.index.tolist()
        if len(idx) < 2:
            continue

        if len(idx) > 3000:
            idx = random.sample(idx, 3000)

        if g_idx % 5 == 0 or g_idx == total_groups:
            log.info("[%s] Group %d/%d (%s): %d wines", label, g_idx, total_groups, group_name, len(idx))

        for ii in range(len(idx)):
            for jj in range(ii + 1, len(idx)):
                i, j   = idx[ii], idx[jj]
                a, b   = records.iloc[i], records.iloc[j]
                score  = compute_pair_score(a, b)

                if score >= config.POSITIVE_THRESHOLD:
                    positive_rows.append({
                        "desc_a": a[desc_col], "desc_b": b[desc_col],
                        "label": 1.0, "pair_type": "positive",
                    })
                elif score < config.WEAK_POSITIVE_MIN and embeddings is not None:
                    sim = float(np.dot(embeddings[i], embeddings[j]))
                    if config.HARD_NEG_SIMILARITY_MIN <= sim <= config.HARD_NEG_SIMILARITY_MAX:
                        hard_neg_candidates.append({
                            "desc_a": a[desc_col], "desc_b": b[desc_col],
                            "label": 0.0, "pair_type": "hard_neg", "_sim": sim,
                        })

    log.info("[%s] Raw counts — positive: %d | hard_neg candidates: %d",
             label, len(positive_rows), len(hard_neg_candidates))

    # Cap positives
    random.shuffle(positive_rows)
    positive_rows = positive_rows[:max_positive]
    n_pos         = len(positive_rows)

    # Hard negatives: hardest (highest similarity) up to ratio
    n_hard = int(n_pos * hard_neg_ratio)
    hard_neg_candidates.sort(key=lambda x: x["_sim"], reverse=True)
    hard_neg_rows = hard_neg_candidates[:n_hard]
    for r in hard_neg_rows:
        del r["_sim"]

    # Easy negatives: cross-colour pairs
    n_easy    = int(n_pos * easy_neg_ratio)
    all_idx   = records.index.tolist()
    easy_rows: list[dict] = []
    attempts  = 0

    while len(easy_rows) < n_easy and attempts < n_easy * 10:
        i, j = random.sample(all_idx, 2)
        a, b = records.iloc[i], records.iloc[j]
        if _norm(a.get("Colour", "")) != _norm(b.get("Colour", "")):
            easy_rows.append({
                "desc_a": a[desc_col], "desc_b": b[desc_col],
                "label": 0.0, "pair_type": "easy_neg",
            })
        attempts += 1

    all_rows = positive_rows + hard_neg_rows + easy_rows
    random.shuffle(all_rows)
    result_df = pd.DataFrame(all_rows)

    log.info("[%s] Final: %d positive | %d hard_neg | %d easy_neg = %d total",
             label, n_pos, len(hard_neg_rows), len(easy_rows), len(result_df))
    return result_df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    PAIRS_DIR.mkdir(parents=True, exist_ok=True)

    # Idempotent — skip if already built
    existing = [Path(p) for p in (config.TRAIN_PAIRS, config.VAL_PAIRS,
                                   config.TRAIN_WINES, config.VAL_WINES)]
    if all(p.exists() for p in existing):
        log.info("Pairs already exist in %s — delete them to rebuild.", PAIRS_DIR)
        for p in existing:
            log.info("  %s  (%.1f MB)", p, p.stat().st_size / 1024 / 1024)
        return

    log.info("=" * 60)
    log.info("Building pairs — one-time run")
    log.info("=" * 60)

    # Load & filter
    log.info("Loading data from %s", config.DATA_PROCESSED_PATH)
    df      = pd.read_csv(config.DATA_PROCESSED_PATH)
    initial = len(df)
    df      = df[df["description"].str.len() >= config.MIN_DESC_LENGTH_FT].reset_index(drop=True)
    log.info("After length filter (>=%d): %d / %d wines", config.MIN_DESC_LENGTH_FT, len(df), initial)

    # Split
    train_df, val_df = split_wines(df)
    train_df.to_csv(config.TRAIN_WINES, index=False)
    val_df.to_csv(config.VAL_WINES, index=False)
    log.info("Saved train_wines.csv and val_wines.csv")

    # Encode train wines for hard-neg mining
    log.info("Loading model for hard-neg mining: %s", config.EMBEDDING_MODEL)
    model = SentenceTransformer(config.EMBEDDING_MODEL)

    log.info("Encoding %d train wines…", len(train_df))
    train_emb = model.encode(
        train_df["description"].tolist(),
        batch_size=64,
        show_progress_bar=True,
        normalize_embeddings=True,
    )

    # Build pairs
    log.info("Building TRAIN pairs…")
    train_pairs_df = score_pairs_grouped(
        train_df, embeddings=train_emb,
        max_positive=config.MAX_POSITIVE_TRAIN,
        hard_neg_ratio=config.HARD_NEG_RATIO,
        easy_neg_ratio=config.EASY_NEG_RATIO,
        label="train",
    )

    log.info("Building VAL pairs…")
    val_pairs_df = score_pairs_grouped(
        val_df, embeddings=None,
        max_positive=config.MAX_POSITIVE_VAL,
        hard_neg_ratio=0.0,
        easy_neg_ratio=config.EASY_NEG_RATIO,
        label="val",
    )

    # Save
    train_pairs_df.to_parquet(config.TRAIN_PAIRS, index=False)
    val_pairs_df.to_parquet(config.VAL_PAIRS, index=False)
    log.info("Saved train_pairs.parquet (%.1f MB)", Path(config.TRAIN_PAIRS).stat().st_size / 1024 / 1024)
    log.info("Saved val_pairs.parquet   (%.1f MB)", Path(config.VAL_PAIRS).stat().st_size / 1024 / 1024)

    # Save meta
    meta = {
        "built_at":           datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
        "base_model":         config.EMBEDDING_MODEL,
        "min_desc_length":    config.MIN_DESC_LENGTH_FT,
        "positive_threshold": config.POSITIVE_THRESHOLD,
        "hard_neg_sim_range": [config.HARD_NEG_SIMILARITY_MIN, config.HARD_NEG_SIMILARITY_MAX],
        "val_size":           config.VAL_SIZE,
        "random_state":       config.RANDOM_STATE,
        "train_wines":        len(train_df),
        "val_wines":          len(val_df),
        "train_pairs":        len(train_pairs_df),
        "val_pairs":          len(val_pairs_df),
        "train_pair_types":   train_pairs_df["pair_type"].value_counts().to_dict(),
        "val_pair_types":     val_pairs_df["pair_type"].value_counts().to_dict(),
    }
    with open(config.PAIRS_META, "w") as f:
        json.dump(meta, f, indent=2)
    log.info("Saved pairs_meta.json")

    log.info("=" * 60)
    log.info("Done. Run training: python scripts/run_finetune.py")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
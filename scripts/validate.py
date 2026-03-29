"""
validate.py — sanity check after Sprint 2.

Checks:
  1. CSV quality      — nulls, duplicates, description lengths
  2. Embeddings       — shape, dtype, norm, variance
  3. FAISS index      — ntotal, dim, test search
  4. Semantic sanity  — similar wines close, different wines far

Usage:
    python scripts/validate.py
"""

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np
import pandas as pd
import faiss

import config

PASS = "✅"
FAIL = "❌"
WARN = "⚠️ "


def section(title: str) -> None:
    print(f"\n{'─' * 50}")
    print(f"  {title}")
    print(f"{'─' * 50}")


def check(label: str, ok: bool, detail: str = "") -> None:
    status = PASS if ok else FAIL
    line = f"  {status}  {label}"
    if detail:
        line += f"  →  {detail}"
    print(line)


# ── 1. CSV Quality ────────────────────────────────────────────────────────────

def validate_csv() -> pd.DataFrame:
    section("1. CSV Quality")
    df = pd.read_csv(config.DATA_PROCESSED_PATH)

    check("File loaded", True, f"{len(df):,} rows, {len(df.columns)} columns")

    # Nulls in key fields
    for col in ["description", "title", "Region", "Grapes"]:
        if col not in df.columns:
            check(f"Column '{col}' exists", False, "missing")
            continue
        n_null = df[col].isna().sum()
        check(f"Nulls in '{col}'", n_null == 0, f"{n_null} nulls" if n_null else "clean")

    # Duplicates
    n_dup = df["description"].duplicated().sum()
    check("No duplicate descriptions", n_dup == 0, f"{n_dup} duplicates" if n_dup else "clean")

    # Description lengths
    lengths = df["description"].str.len()
    short = (lengths < config.MIN_DESC_LENGTH).sum() if hasattr(config, "MIN_DESC_LENGTH") else 0
    check(
        "Description lengths",
        lengths.min() >= 50,
        f"min={lengths.min()}, median={lengths.median():.0f}, max={lengths.max()}"
    )
    if short:
        print(f"       {WARN} {short} descriptions shorter than MIN_DESC_LENGTH")

    return df


# ── 2. Embeddings ─────────────────────────────────────────────────────────────

def validate_embeddings(df: pd.DataFrame) -> np.ndarray:
    section("2. Embeddings")
    emb = np.load(config.EMBEDDINGS_PATH)

    check("File loaded", True, f"shape={emb.shape}")
    check("dtype is float32", emb.dtype == np.float32, str(emb.dtype))
    check("Row count matches CSV", emb.shape[0] == len(df), f"{emb.shape[0]} vs {len(df)}")
    check("Embedding dim is 384", emb.shape[1] == 384, str(emb.shape[1]))

    # L2 norms should all be ~1.0 (normalised)
    norms = np.linalg.norm(emb, axis=1)
    norm_ok = np.allclose(norms, 1.0, atol=1e-5)
    check("Embeddings are L2-normalised", norm_ok, f"mean norm={norms.mean():.6f}")

    # No zero / NaN vectors
    has_nan = np.isnan(emb).any()
    has_zero = (norms < 1e-6).any()
    check("No NaN values", not has_nan)
    check("No zero vectors", not has_zero)

    # Variance — if all embeddings are identical something is wrong
    col_var = emb.var(axis=0).mean()
    check("Non-trivial variance", col_var > 1e-4, f"mean col variance={col_var:.6f}")

    return emb


# ── 3. FAISS Index ────────────────────────────────────────────────────────────

def validate_index(emb: np.ndarray) -> faiss.IndexFlatIP:
    section("3. FAISS Index")
    index = faiss.read_index(config.FAISS_INDEX_PATH)

    check("File loaded", True)
    check("ntotal matches embeddings", index.ntotal == emb.shape[0], f"{index.ntotal} vectors")
    check("Dimension is 384", index.d == 384, str(index.d))

    # Test search — query with first embedding, top result should be itself (score ≈ 1.0)
    query = emb[0:1]
    scores, ids = index.search(query, 1)
    self_match = ids[0][0] == 0
    self_score = scores[0][0]
    check("Self-search returns self", self_match, f"id={ids[0][0]}, score={self_score:.6f}")
    check("Self-search score ≈ 1.0", abs(self_score - 1.0) < 1e-4, f"{self_score:.6f}")

    return index


# ── 4. Semantic Sanity ────────────────────────────────────────────────────────

def validate_semantics(df: pd.DataFrame, emb: np.ndarray) -> None:
    section("4. Semantic Sanity")

    if "Region" not in df.columns or "Grapes" not in df.columns:
        print(f"  {WARN} Skipping — 'Region' or 'Grapes' column missing")
        return

    # Find two wines from the same region
    region_counts = df["Region"].value_counts()
    common_regions = region_counts[region_counts >= 5].index.tolist()

    if not common_regions:
        print(f"  {WARN} No region with 5+ wines found, skipping similarity check")
        return

    region = common_regions[0]
    same_idx = df[df["Region"] == region].index.tolist()[:2]

    # Find a wine from a very different region
    other_regions = [r for r in common_regions if r != region]
    diff_idx = df[df["Region"] == other_regions[-1]].index[0] if other_regions else None

    if len(same_idx) >= 2:
        sim_same = float(np.dot(emb[same_idx[0]], emb[same_idx[1]]))
        t0 = df.iloc[same_idx[0]]["title"][:50]
        t1 = df.iloc[same_idx[1]]["title"][:50]
        print(f"\n  Same region ({region}):")
        print(f"    Wine A: {t0}")
        print(f"    Wine B: {t1}")
        print(f"    Cosine similarity: {sim_same:.4f}")
        check("Same-region similarity > 0.5", sim_same > 0.5, f"{sim_same:.4f}")

    if diff_idx is not None:
        sim_diff = float(np.dot(emb[same_idx[0]], emb[diff_idx]))
        t_diff = df.iloc[diff_idx]["title"][:50]
        print(f"\n  Different region ({other_regions[-1]}):")
        print(f"    Wine C: {t_diff}")
        print(f"    Cosine similarity with Wine A: {sim_diff:.4f}")
        check("Cross-region similarity < same-region", sim_diff < sim_same, f"{sim_diff:.4f}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print("\n🍷  Wine Semantic Search — Validation Report")

    df  = validate_csv()
    emb = validate_embeddings(df)
    idx = validate_index(emb)
    validate_semantics(df, emb)

    print(f"\n{'─' * 50}")
    print("  Done. Fix any ❌ before moving to Sprint 3.")
    print(f"{'─' * 50}\n")


if __name__ == "__main__":
    main()
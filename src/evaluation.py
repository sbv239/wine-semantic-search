"""
src/evaluation.py
-----------------
Offline evaluation of the semantic search system.

Metrics
-------
- Region Hit Rate  : fraction of top-K results sharing the region of the query wine
- Grape Recall     : fraction of top-K results sharing the primary grape of the query wine

Usage
-----
    python -m src.evaluation                  # default 200 sample queries, top-5
    python -m src.evaluation --n 500 --top_k 10
    python -m src.evaluation --verbose        # print per-query breakdown
"""

from __future__ import annotations

import argparse
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

import config
from src.search import WineSearcher

MIN_DESC_LENGTH = 50  # minimum characters to keep a description (mirrors preprocessing)

logging.basicConfig(
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _primary_grape(grapes_str: str) -> str:
    """
    Extract the first (primary) grape variety from a string like
    '85% Nebbiolo, 15% Barbera' or '100% Chardonnay' or 'Pinot Noir'.
    Returns lower-cased grape name or '' if unknown.
    """
    if not grapes_str or str(grapes_str).strip().lower() in ("unknown", "nan", ""):
        return ""
    # Drop percentage prefix/suffix, take first token group
    cleaned = re.sub(r"\d+%?\s*", "", str(grapes_str))
    # Split on comma or semicolon, take first variety
    first = re.split(r"[,;]", cleaned)[0].strip()
    return first.lower()


def _normalise_region(region_str: str) -> str:
    return str(region_str).strip().lower() if region_str else ""


# ---------------------------------------------------------------------------
# Dataclass for results
# ---------------------------------------------------------------------------

@dataclass
class EvalResults:
    n_queries: int
    top_k: int
    region_hit_rate: float          # mean over queries
    grape_recall: float             # mean over queries
    region_hits_per_query: list[float] = field(repr=False, default_factory=list)
    grape_recalls_per_query: list[float] = field(repr=False, default_factory=list)
    skipped_no_region: int = 0
    skipped_no_grape: int = 0

    def summary(self) -> str:
        lines = [
            "",
            "=" * 52,
            "  Wine Semantic Search — Evaluation Results",
            "=" * 52,
            f"  Queries evaluated : {self.n_queries}",
            f"  Top-K             : {self.top_k}",
            f"  Skipped (no region): {self.skipped_no_region}",
            f"  Skipped (no grape) : {self.skipped_no_grape}",
            "-" * 52,
            f"  Region Hit Rate   : {self.region_hit_rate:.3f}  "
            f"({self.region_hit_rate * 100:.1f}%)",
            f"  Grape Recall      : {self.grape_recall:.3f}  "
            f"({self.grape_recall * 100:.1f}%)",
            "=" * 52,
            "",
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Core evaluation function
# ---------------------------------------------------------------------------

def evaluate(
    searcher: WineSearcher,
    df: pd.DataFrame,
    n_queries: int = 200,
    top_k: int = 5,
    seed: int = 42,
    verbose: bool = False,
) -> EvalResults:
    """
    Sample `n_queries` wines from `df`, use each wine's description as a query,
    then check how many of the top-K results share the same region / primary grape.

    The query wine itself is excluded from results (search returns index IDs;
    the query wine will always appear as the #1 hit with similarity ≈ 1.0,
    so we drop it before scoring).
    """
    rng = np.random.default_rng(seed)

    # Sample query wines — only rows with known region or grape for fair eval
    sample_df = df.sample(n=min(n_queries, len(df)), random_state=seed)

    region_hits: list[float] = []
    grape_recalls: list[float] = []
    skipped_region = 0
    skipped_grape = 0

    evaluated = 0

    for _, row in sample_df.iterrows():
        if evaluated >= n_queries:
            break

        query_desc: str = str(row.get("description", ""))
        query_region: str = _normalise_region(row.get("Region", ""))
        query_grape: str = _primary_grape(row.get("Grapes", ""))
        query_idx: int = int(row.name)  # DataFrame index = FAISS id

        if not query_desc or len(query_desc) < MIN_DESC_LENGTH:
            continue

        # Fetch top_k + 1 to account for the query wine itself appearing in results
        results = searcher.search(query_desc, top_k=top_k + 1)

        # Exclude the query wine from results
        results = [r for r in results if r.id != query_idx][:top_k]

        if not results:
            continue

        evaluated += 1

        # --- Region Hit Rate ---
        if query_region and query_region != "unknown":
            region_match = sum(
                1 for r in results
                if _normalise_region(r.region) == query_region
            )
            hit_rate = region_match / len(results)
            region_hits.append(hit_rate)

            if verbose:
                log.info(
                    "Query region=%s | hits=%d/%d | desc='%.60s...'",
                    query_region, region_match, len(results), query_desc,
                )
        else:
            skipped_region += 1

        # --- Grape Recall ---
        if query_grape:
            grape_match = sum(
                1 for r in results
                if _primary_grape(r.grapes) == query_grape
            )
            recall = grape_match / len(results)
            grape_recalls.append(recall)
        else:
            skipped_grape += 1

    mean_region = float(np.mean(region_hits)) if region_hits else 0.0
    mean_grape = float(np.mean(grape_recalls)) if grape_recalls else 0.0

    return EvalResults(
        n_queries=evaluated,
        top_k=top_k,
        region_hit_rate=mean_region,
        grape_recall=mean_grape,
        region_hits_per_query=region_hits,
        grape_recalls_per_query=grape_recalls,
        skipped_no_region=skipped_region,
        skipped_no_grape=skipped_grape,
    )


# ---------------------------------------------------------------------------
# Percentile breakdown (for README table)
# ---------------------------------------------------------------------------

def percentile_breakdown(values: list[float], label: str) -> str:
    if not values:
        return f"{label}: no data"
    arr = np.array(values)
    return (
        f"{label}: "
        f"mean={arr.mean():.3f}  "
        f"p25={np.percentile(arr, 25):.3f}  "
        f"p50={np.percentile(arr, 50):.3f}  "
        f"p75={np.percentile(arr, 75):.3f}  "
        f"p100={arr.max():.3f}"
    )


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate wine semantic search quality."
    )
    parser.add_argument(
        "--n", type=int, default=200,
        help="Number of query wines to evaluate (default: 200)",
    )
    parser.add_argument(
        "--top_k", type=int, default=5,
        help="Top-K results to score (default: 5)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for sampling (default: 42)",
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Print per-query region match info",
    )
    args = parser.parse_args()

    # --- Load data ---
    log.info("Loading dataset from %s …", config.DATA_PROCESSED_PATH)
    df = pd.read_csv(config.DATA_PROCESSED_PATH)
    log.info("Dataset loaded: %d rows", len(df))

    # --- Load searcher ---
    log.info("Initialising WineSearcher …")
    t0 = time.perf_counter()
    searcher = WineSearcher()
    log.info("WineSearcher ready in %.1fs", time.perf_counter() - t0)

    # --- Run evaluation ---
    log.info("Running evaluation: n=%d, top_k=%d, seed=%d …", args.n, args.top_k, args.seed)
    t1 = time.perf_counter()
    results = evaluate(
        searcher=searcher,
        df=df,
        n_queries=args.n,
        top_k=args.top_k,
        seed=args.seed,
        verbose=args.verbose,
    )
    elapsed = time.perf_counter() - t1
    log.info("Evaluation finished in %.1fs (%.0f ms/query)", elapsed, elapsed / max(results.n_queries, 1) * 1000)

    # --- Print summary ---
    print(results.summary())
    print(percentile_breakdown(results.region_hits_per_query, "Region Hit Rate"))
    print(percentile_breakdown(results.grape_recalls_per_query, "Grape Recall   "))
    print()

    # --- Save results ---
    _save_results(results, args)


def _save_results(results: EvalResults, args) -> None:
    """Append evaluation run to eval_results.json for cross-run comparison."""
    import json
    import os
    from datetime import datetime

    out_path = "eval_results.json"
    history = []
    if os.path.exists(out_path):
        with open(out_path) as f:
            history = json.load(f)

    entry = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "model": config.EMBEDDING_MODEL,
        "n_queries": results.n_queries,
        "top_k": results.top_k,
        "seed": args.seed,
        "region_hit_rate": round(results.region_hit_rate, 4),
        "grape_recall": round(results.grape_recall, 4),
        "percentiles": {
            "region": {
                "p25": round(float(np.percentile(results.region_hits_per_query, 25)), 4),
                "p50": round(float(np.percentile(results.region_hits_per_query, 50)), 4),
                "p75": round(float(np.percentile(results.region_hits_per_query, 75)), 4),
            },
            "grape": {
                "p25": round(float(np.percentile(results.grape_recalls_per_query, 25)), 4),
                "p50": round(float(np.percentile(results.grape_recalls_per_query, 50)), 4),
                "p75": round(float(np.percentile(results.grape_recalls_per_query, 75)), 4),
            },
        },
    }
    history.append(entry)
    with open(out_path, "w") as f:
        json.dump(history, f, indent=2)
    log.info("Results appended to %s", out_path)


if __name__ == "__main__":
    main()
"""
src/evaluation.py

Offline evaluation of the semantic search system.

Metrics:
    Region Hit Rate  — fraction of top-K results sharing the region of the query wine
    Grape Recall     — fraction of top-K results sharing the primary grape of the query wine

Usage:
    python -m src.evaluation                                       # base model
    python -m src.evaluation --model models/finetuned_.../final   # finetuned model
    python -m src.evaluation --n 500 --top_k 10 --verbose
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import time
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np
import pandas as pd

import config
from src.search import WineSearcher

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
    if not grapes_str or str(grapes_str).strip().lower() in ("unknown", "nan", ""):
        return ""
    cleaned = re.sub(r"\d+%?\s*", "", str(grapes_str))
    return re.split(r"[,;]", cleaned)[0].strip().lower()


def _normalise_region(region_str: str) -> str:
    return str(region_str).strip().lower() if region_str else ""


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class EvalResults:
    n_queries: int
    top_k: int
    region_hit_rate: float
    grape_recall: float
    region_hits_per_query:  list[float] = field(repr=False, default_factory=list)
    grape_recalls_per_query: list[float] = field(repr=False, default_factory=list)
    skipped_no_region: int = 0
    skipped_no_grape:  int = 0

    def summary(self, model_name: str = "") -> str:
        display = ("..." + model_name[-45:]) if len(model_name) > 48 else model_name
        lines = ["", "=" * 52, "  Wine Semantic Search — Evaluation Results", "=" * 52]
        if model_name:
            lines.append(f"  Model             : {display}")
        lines += [
            f"  Queries evaluated : {self.n_queries}",
            f"  Top-K             : {self.top_k}",
            f"  Skipped (no region): {self.skipped_no_region}",
            f"  Skipped (no grape) : {self.skipped_no_grape}",
            "-" * 52,
            f"  Region Hit Rate   : {self.region_hit_rate:.3f}  ({self.region_hit_rate*100:.1f}%)",
            f"  Grape Recall      : {self.grape_recall:.3f}  ({self.grape_recall*100:.1f}%)",
            "=" * 52, "",
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Core evaluation
# ---------------------------------------------------------------------------

def evaluate(
    searcher: WineSearcher,
    df: pd.DataFrame,
    n_queries: int = 200,
    top_k: int     = 5,
    seed: int      = 42,
    verbose: bool  = False,
) -> EvalResults:
    sample_df = df.sample(n=min(n_queries, len(df)), random_state=seed)

    region_hits:    list[float] = []
    grape_recalls:  list[float] = []
    skipped_region = skipped_grape = evaluated = 0

    for _, row in sample_df.iterrows():
        if evaluated >= n_queries:
            break

        query_desc   = str(row.get("description", ""))
        query_region = _normalise_region(row.get("Region", ""))
        query_grape  = _primary_grape(row.get("Grapes", ""))
        query_idx    = int(row.name)

        if not query_desc or len(query_desc) < config.MIN_DESC_LENGTH:
            continue

        results = searcher.search(query_desc, top_k=top_k + 1)
        results = [r for r in results if r.id != query_idx][:top_k]
        if not results:
            continue

        evaluated += 1

        if query_region and query_region != "unknown":
            hits = sum(1 for r in results if _normalise_region(r.region) == query_region)
            region_hits.append(hits / len(results))
            if verbose:
                log.info("region=%s | hits=%d/%d | %.60s…", query_region, hits, len(results), query_desc)
        else:
            skipped_region += 1

        if query_grape:
            grape_recalls.append(
                sum(1 for r in results if _primary_grape(r.grapes) == query_grape) / len(results)
            )
        else:
            skipped_grape += 1

    return EvalResults(
        n_queries=evaluated,
        top_k=top_k,
        region_hit_rate=float(np.mean(region_hits)) if region_hits else 0.0,
        grape_recall=float(np.mean(grape_recalls)) if grape_recalls else 0.0,
        region_hits_per_query=region_hits,
        grape_recalls_per_query=grape_recalls,
        skipped_no_region=skipped_region,
        skipped_no_grape=skipped_grape,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def percentile_breakdown(values: list[float], label: str) -> str:
    if not values:
        return f"{label}: no data"
    arr = np.array(values)
    return (f"{label}: mean={arr.mean():.3f}  p25={np.percentile(arr,25):.3f}  "
            f"p50={np.percentile(arr,50):.3f}  p75={np.percentile(arr,75):.3f}  "
            f"p100={arr.max():.3f}")


def _save_results(results: EvalResults, args, model_name: str, model_type: str) -> None:
    history = []
    if os.path.exists(config.EVAL_RESULTS_PATH):
        with open(config.EVAL_RESULTS_PATH) as f:
            history = json.load(f)

    def pct(vals, p):
        return round(float(np.percentile(vals, p)), 4) if vals else 0.0

    history.append({
        "timestamp":       datetime.now().isoformat(timespec="seconds"),
        "model":           model_name,
        "model_type":      model_type,
        "n_queries":       results.n_queries,
        "top_k":           results.top_k,
        "seed":            args.seed,
        "region_hit_rate": round(results.region_hit_rate, 4),
        "grape_recall":    round(results.grape_recall, 4),
        "percentiles": {
            "region": {"p25": pct(results.region_hits_per_query, 25),
                       "p50": pct(results.region_hits_per_query, 50),
                       "p75": pct(results.region_hits_per_query, 75)},
            "grape":  {"p25": pct(results.grape_recalls_per_query, 25),
                       "p50": pct(results.grape_recalls_per_query, 50),
                       "p75": pct(results.grape_recalls_per_query, 75)},
        },
    })
    with open(config.EVAL_RESULTS_PATH, "w") as f:
        json.dump(history, f, indent=2)
    log.info("Results appended to %s", config.EVAL_RESULTS_PATH)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate wine semantic search quality.")
    parser.add_argument("--n",       type=int,  default=200)
    parser.add_argument("--top_k",   type=int,  default=5)
    parser.add_argument("--seed",    type=int,  default=42)
    parser.add_argument("--model",   type=str,  default=None,
                        help="HuggingFace ID or local path. Defaults to config.EMBEDDING_MODEL.")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    model_name = args.model or config.EMBEDDING_MODEL
    model_type = "finetuned" if args.model else "base"

    log.info("Loading dataset from %s …", config.DATA_PROCESSED_PATH)
    df = pd.read_csv(config.DATA_PROCESSED_PATH)
    log.info("Dataset loaded: %d rows", len(df))

    log.info("Initialising WineSearcher: %s …", model_name)
    t0       = time.perf_counter()
    searcher = WineSearcher(model_name=model_name)
    log.info("Ready in %.1fs", time.perf_counter() - t0)

    log.info("Running evaluation: n=%d, top_k=%d, seed=%d …", args.n, args.top_k, args.seed)
    t1      = time.perf_counter()
    results = evaluate(searcher=searcher, df=df, n_queries=args.n,
                       top_k=args.top_k, seed=args.seed, verbose=args.verbose)
    elapsed = time.perf_counter() - t1
    log.info("Done in %.1fs (%.0f ms/query)", elapsed, elapsed / max(results.n_queries, 1) * 1000)

    print(results.summary(model_name=model_name))
    print(percentile_breakdown(results.region_hits_per_query, "Region Hit Rate"))
    print(percentile_breakdown(results.grape_recalls_per_query, "Grape Recall   "))
    print()

    _save_results(results, args, model_name=model_name, model_type=model_type)


if __name__ == "__main__":
    main()
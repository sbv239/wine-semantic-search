"""
scripts/compare_models.py

Unified comparison table across all evaluation runs.
Reads from two sources and merges them:
  - eval_results.json        (from src/evaluation.py — base + finetuned)
  - models/eval/run_*.json   (from src/finetune.py — training run details)

Usage:
    python scripts/compare_models.py
    python scripts/compare_models.py --sort region_hit_rate
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare evaluation results across models.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--sort", default="region_hit_rate",
                        choices=["region_hit_rate", "grape_recall", "model", "timestamp"],
                        help="Sort rows by this column (descending)")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Load from eval_results.json (evaluation.py output)
# ---------------------------------------------------------------------------

def load_eval_results() -> list[dict]:
    path = Path(config.EVAL_RESULTS_PATH)
    if not path.exists():
        return []
    with open(path) as f:
        entries = json.load(f)

    rows = []
    for e in entries:
        rows.append({
            "source":          "evaluation.py",
            "timestamp":       e.get("timestamp", ""),
            "model":           e.get("model", ""),
            "model_type":      e.get("model_type", "unknown"),
            "n_queries":       e.get("n_queries", 0),
            "top_k":           e.get("top_k", 5),
            "region_hit_rate": e.get("region_hit_rate", 0.0),
            "grape_recall":    e.get("grape_recall", 0.0),
            "map_at_5":        None,   # not available in evaluation.py output
        })
    return rows


# ---------------------------------------------------------------------------
# Load from models/eval/run_*.json (finetune.py output)
# ---------------------------------------------------------------------------

def load_finetune_runs() -> list[dict]:
    runs_dir = Path(config.EVAL_RUNS_DIR)
    if not runs_dir.exists():
        return []

    rows = []
    for path in sorted(runs_dir.glob("run_*.json")):
        with open(path) as f:
            r = json.load(f)
        bm = r.get("best_metrics", {})
        rows.append({
            "source":          "finetune.py",
            "timestamp":       r.get("run_id", ""),
            "model":           r.get("model_name", ""),
            "model_type":      r.get("model_type", "finetuned"),
            "n_queries":       r.get("val_pairs", 0),
            "top_k":           5,
            "region_hit_rate": bm.get("region_hit_rate", 0.0),
            "grape_recall":    bm.get("grape_recall", 0.0),
            "map_at_5":        bm.get("map_at_5", None),
        })
    return rows


# ---------------------------------------------------------------------------
# Print table
# ---------------------------------------------------------------------------

def print_table(rows: list[dict], sort_by: str) -> None:
    if not rows:
        print("No evaluation results found.")
        print(f"  - Run: python -m src.evaluation")
        print(f"  - Run: python scripts/run_finetune.py")
        return

    reverse = sort_by not in ("model", "timestamp")
    rows = sorted(rows, key=lambda r: r.get(sort_by) or 0, reverse=reverse)

    # Column widths
    W_MODEL  = 55
    W_TYPE   = 12
    W_SOURCE = 15
    W_NUM    =  8
    W_MET    = 14

    header = (
        f"{'Model':<{W_MODEL}} "
        f"{'Type':<{W_TYPE}} "
        f"{'Source':<{W_SOURCE}} "
        f"{'Queries':>{W_NUM}} "
        f"{'RegionHit':>{W_MET}} "
        f"{'GrapeRecall':>{W_MET}} "
        f"{'MAP@5':>{W_MET}}"
    )
    sep = "─" * len(header)

    print()
    print("Model Comparison")
    print(sep)
    print(header)
    print(sep)

    for r in rows:
        model_display = r["model"]
        if len(model_display) > W_MODEL:
            model_display = "…" + model_display[-(W_MODEL - 1):]
        map_str = f"{r['map_at_5']:{W_MET}.4f}" if r["map_at_5"] is not None else f"{'n/a':>{W_MET}}"
        print(
            f"{model_display:<{W_MODEL}} "
            f"{r['model_type']:<{W_TYPE}} "
            f"{r['source']:<{W_SOURCE}} "
            f"{r['n_queries']:>{W_NUM}} "
            f"{r['region_hit_rate']:>{W_MET}.4f} "
            f"{r['grape_recall']:>{W_MET}.4f} "
            f"{map_str}"
        )

    print(sep)
    print(f"Sorted by: {sort_by} (descending)")

    # Delta summary: best finetuned vs best base (from evaluation.py)
    eval_rows     = [r for r in rows if r["source"] == "evaluation.py"]
    base_rows     = [r for r in eval_rows if r["model_type"] == "base"]
    finetuned_rows = [r for r in eval_rows if r["model_type"] == "finetuned"]

    if base_rows and finetuned_rows:
        best_base = max(base_rows, key=lambda r: r["region_hit_rate"])
        best_ft   = max(finetuned_rows, key=lambda r: r["region_hit_rate"])
        print()
        print("Delta (best finetuned vs best base):")
        d_region = best_ft["region_hit_rate"] - best_base["region_hit_rate"]
        d_grape  = best_ft["grape_recall"]    - best_base["grape_recall"]
        print(f"  Region Hit Rate : {best_base['region_hit_rate']:.4f} → {best_ft['region_hit_rate']:.4f}  "
              f"({'+'if d_region>=0 else ''}{d_region:.4f}  {d_region/best_base['region_hit_rate']*100:+.1f}%)")
        print(f"  Grape Recall    : {best_base['grape_recall']:.4f} → {best_ft['grape_recall']:.4f}  "
              f"({'+'if d_grape>=0 else ''}{d_grape:.4f}  {d_grape/best_base['grape_recall']*100:+.1f}%)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    eval_rows    = load_eval_results()
    finetune_rows = load_finetune_runs()
    all_rows     = eval_rows + finetune_rows

    print(f"Loaded {len(eval_rows)} evaluation run(s) + {len(finetune_rows)} finetune run(s)")
    print_table(all_rows, sort_by=args.sort)


if __name__ == "__main__":
    main()
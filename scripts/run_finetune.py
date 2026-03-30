"""
scripts/run_finetune.py

CLI runner for the fine-tuning pipeline.

Examples:
    python scripts/run_finetune.py
    python scripts/run_finetune.py --epochs 10 --batch-size 64 --patience 3
    python scripts/run_finetune.py --base-model BAAI/bge-small-en-v1.5
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from src.finetune import run_finetuning


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune SentenceTransformer on wine tasting notes.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--base-model",  default=config.EMBEDDING_MODEL,
                        help="HuggingFace model ID to fine-tune")
    parser.add_argument("--epochs",      type=int, default=config.FINETUNE_EPOCHS)
    parser.add_argument("--batch-size",  type=int, default=config.FINETUNE_BATCH_SIZE)
    parser.add_argument("--patience",    type=int, default=config.FINETUNE_PATIENCE,
                        help="Early stopping patience")
    parser.add_argument("--eval-sample", type=int, default=config.FINETUNE_EVAL_SAMPLE,
                        help="Val wines sampled for MAP@5 per epoch")
    parser.add_argument("--output-dir",  default=None,
                        help="Override output directory (default: auto-generated)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    result = run_finetuning(
        base_model_name = args.base_model,
        epochs          = args.epochs,
        batch_size      = args.batch_size,
        patience        = args.patience,
        eval_sample     = args.eval_sample,
        output_dir      = args.output_dir,
    )

    print("\n" + "=" * 50)
    print("Fine-tuning complete")
    print("=" * 50)
    print(f"Run ID        : {result.run_id}")
    print(f"Base model    : {result.base_model}")
    print(f"Epochs trained: {result.epochs_trained}")
    print(f"Best epoch    : {result.best_epoch}")
    print(f"Training pairs: {result.training_pairs}")
    print()
    print("Best metrics:")
    for k, v in result.best_metrics.items():
        if k != "epoch":
            print(f"  {k:<20} {v:.4f}")
    print()
    print(f"Model saved to : {result.output_dir}/final")
    print(f"Eval JSON      : {config.EVAL_RUNS_DIR}/run_{result.run_id}.json")


if __name__ == "__main__":
    main()
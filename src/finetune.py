"""
src/finetune.py

Fine-tuning pipeline for wine semantic search.
Expects prebuilt pairs from scripts/build_pairs.py.

Usage:
    python scripts/run_finetune.py --epochs 5 --batch-size 32
"""

from __future__ import annotations

import json
import logging
import random
import re
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

import config

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class PairStats:
    total_wines:    int = 0
    train_wines:    int = 0
    val_wines:      int = 0
    train_pairs:    int = 0
    val_pairs:      int = 0
    positive_train: int = 0
    hard_neg_train: int = 0
    easy_neg_train: int = 0


@dataclass
class EvalResult:
    epoch:           int
    map_at_5:        float
    region_hit_rate: float
    grape_recall:    float


@dataclass
class FinetuneRunResult:
    run_id:         str
    model_name:     str
    model_type:     str
    base_model:     str
    embedding_dim:  int
    training_pairs: int
    val_pairs:      int
    epochs_trained: int
    best_epoch:     int
    loss:           str
    batch_size:     int
    pair_stats:     dict
    eval_history:   list
    best_metrics:   dict
    output_dir:     str


# ---------------------------------------------------------------------------
# Pair scoring helpers (also used by build_pairs.py)
# ---------------------------------------------------------------------------

def parse_primary_grape(grapes_str: str) -> Optional[str]:
    """
    Extract primary grape from Grapes field.
    Returns None for blends (multiple varieties) and unknowns.
    """
    if not isinstance(grapes_str, str) or grapes_str.strip().lower() == "unknown":
        return None
    parts = [p.strip() for p in grapes_str.split(",")]
    if len(parts) > 1:
        return None  # blend → no match
    match = re.match(r"^\d+\s*%\s*(.+)$", parts[0])
    if match:
        return match.group(1).strip().lower()
    return parts[0].lower()


def _norm(val) -> str:
    if not isinstance(val, str):
        return ""
    return val.strip().lower()


def compute_pair_score(a: pd.Series, b: pd.Series) -> float:
    """
    Compute similarity score between two wine rows.
    Score >= config.POSITIVE_THRESHOLD → positive pair.
    Score <  config.WEAK_POSITIVE_MIN  → negative candidate.
    """
    score = 0.0

    grape_a = parse_primary_grape(a.get("Grapes", ""))
    grape_b = parse_primary_grape(b.get("Grapes", ""))
    if grape_a and grape_b and grape_a == grape_b:
        score += config.W_GRAPE

    app_a = _norm(a.get("Appellation", ""))
    app_b = _norm(b.get("Appellation", ""))
    reg_a = _norm(a.get("Region", ""))
    reg_b = _norm(b.get("Region", ""))

    if app_a and app_b and app_a != "unknown" and app_a == app_b:
        score += config.W_APPELLATION
    elif reg_a and reg_b and reg_a != "unknown" and reg_a == reg_b:
        score += config.W_REGION

    if _norm(a.get("Body", ""))      == _norm(b.get("Body", ""))      != "unknown":
        score += config.W_BODY
    if _norm(a.get("Oak", ""))       == _norm(b.get("Oak", ""))       != "unknown":
        score += config.W_OAK
    if _norm(a.get("Sweetness", "")) == _norm(b.get("Sweetness", "")) != "unknown":
        score += config.W_SWEETNESS

    prod_a = _norm(a.get("Producer", ""))
    prod_b = _norm(b.get("Producer", ""))
    vint_a = str(a.get("Vintage", "")).strip()
    vint_b = str(b.get("Vintage", "")).strip()
    if prod_a and prod_a != "unknown" and prod_a == prod_b and vint_a != vint_b:
        score += config.W_SAME_PRODUCER_VINTAGE

    return score


# ---------------------------------------------------------------------------
# Pre-tokenized dataset
# ---------------------------------------------------------------------------

class TokenizedPairDataset(Dataset):
    """
    Tokenizes all pairs once at construction.
    Avoids repeated tokenization per batch (was causing 12+ sec/batch).
    """
    def __init__(self, examples: list[InputExample], tokenizer, max_length: int):
        log.info("Pre-tokenizing %d pairs (max_length=%d)…", len(examples), max_length)
        texts_a = [e.texts[0] for e in examples]
        texts_b = [e.texts[1] for e in examples]
        self.enc_a = tokenizer(
            texts_a, padding=True, truncation=True,
            max_length=max_length, return_tensors="pt",
        )
        self.enc_b = tokenizer(
            texts_b, padding=True, truncation=True,
            max_length=max_length, return_tensors="pt",
        )
        self.labels = torch.tensor([e.label for e in examples], dtype=torch.float32)
        log.info("Pre-tokenization done.")

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        return (
            {k: v[idx] for k, v in self.enc_a.items()},
            {k: v[idx] for k, v in self.enc_b.items()},
            self.labels[idx],
        )


def collate_pretokenized(batch):
    """Stack pre-tokenized tensors into a batch."""
    a_list, b_list, labels = zip(*batch)

    def stack(dicts):
        return {k: torch.stack([d[k] for d in dicts]) for k in dicts[0]}

    return stack(a_list), stack(b_list), torch.stack(labels)


# ---------------------------------------------------------------------------
# Load prebuilt pairs
# ---------------------------------------------------------------------------

def load_pairs(
    pairs_dir: Path = Path(config.PAIRS_DIR),
) -> tuple[list[InputExample], list[InputExample], PairStats]:
    """Load prebuilt pairs from parquet files produced by build_pairs.py."""
    train_path = pairs_dir / "train_pairs.parquet"
    val_path   = pairs_dir / "val_pairs.parquet"
    meta_path  = pairs_dir / "pairs_meta.json"

    if not train_path.exists() or not val_path.exists():
        raise FileNotFoundError(
            f"Pairs not found in {pairs_dir}.\n"
            "Run first:  python scripts/build_pairs.py"
        )

    log.info("Loading prebuilt pairs from %s", pairs_dir)
    train_df = pd.read_parquet(train_path)
    val_df   = pd.read_parquet(val_path)

    train_examples = [
        InputExample(texts=[row["desc_a"], row["desc_b"]], label=float(row["label"]))
        for _, row in train_df.iterrows()
    ]
    val_examples = [
        InputExample(texts=[row["desc_a"], row["desc_b"]], label=float(row["label"]))
        for _, row in val_df.iterrows()
    ]

    stats = PairStats(train_pairs=len(train_examples), val_pairs=len(val_examples))
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)
        stats.train_wines    = meta.get("train_wines", 0)
        stats.val_wines      = meta.get("val_wines", 0)
        pt                   = meta.get("train_pair_types", {})
        stats.positive_train = pt.get("positive", 0)
        stats.hard_neg_train = pt.get("hard_neg", 0)
        stats.easy_neg_train = pt.get("easy_neg", 0)

    log.info("Loaded %d train pairs / %d val pairs", len(train_examples), len(val_examples))
    return train_examples, val_examples, stats


# ---------------------------------------------------------------------------
# MAP@5 evaluation (inline during training)
# ---------------------------------------------------------------------------

def compute_map_at_k(
    model: SentenceTransformer,
    val_wines_path: Path,
    k: int = 5,
    sample: int = config.FINETUNE_EVAL_SAMPLE,
) -> dict[str, float]:
    """Compute MAP@k, region_hit_rate, grape_recall on val wines."""
    if not val_wines_path.exists():
        log.warning("val_wines.csv not found — skipping eval")
        return {"map@5": 0.0, "region_hit_rate": 0.0, "grape_recall": 0.0}

    records   = pd.read_csv(val_wines_path).reset_index(drop=True)
    query_idx = random.sample(range(len(records)), min(sample, len(records)))

    all_emb = model.encode(
        records["description"].tolist(),
        batch_size=64,
        normalize_embeddings=True,
        show_progress_bar=False,
    )

    ap_scores:     list[float] = []
    region_hits:   list[float] = []
    grape_recalls: list[float] = []

    for qi in query_idx:
        sims     = np.dot(all_emb, all_emb[qi])
        sims[qi] = -1.0
        top_k    = np.argsort(sims)[::-1][:k]
        qrow     = records.iloc[qi]

        q_region = _norm(qrow.get("Region", ""))
        q_grape  = parse_primary_grape(qrow.get("Grapes", ""))

        n_rel, psum = 0, 0.0
        for rank, idx in enumerate(top_k, start=1):
            if compute_pair_score(qrow, records.iloc[idx]) >= config.POSITIVE_THRESHOLD:
                n_rel += 1
                psum  += n_rel / rank
        ap_scores.append(psum / min(k, max(n_rel, 1)))

        if q_region and q_region != "unknown":
            region_hits.append(float(
                any(_norm(records.iloc[idx].get("Region", "")) == q_region for idx in top_k)
            ))
        if q_grape:
            grape_recalls.append(
                sum(1 for idx in top_k
                    if parse_primary_grape(records.iloc[idx].get("Grapes", "")) == q_grape) / k
            )

    return {
        "map@5":           round(float(np.mean(ap_scores)), 4),
        "region_hit_rate": round(float(np.mean(region_hits)) if region_hits else 0.0, 4),
        "grape_recall":    round(float(np.mean(grape_recalls)) if grape_recalls else 0.0, 4),
    }


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_one_epoch(
    model: SentenceTransformer,
    dataloader: DataLoader,
    loss_fn: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    device: torch.device,
    epoch: int,
) -> float:
    model.train()
    total_loss, steps = 0.0, 0

    for enc_a, enc_b, labels in tqdm(dataloader, desc=f"Epoch {epoch}", unit="batch"):
        enc_a  = {k: v.to(device) for k, v in enc_a.items()}
        enc_b  = {k: v.to(device) for k, v in enc_b.items()}
        labels = labels.to(device)

        optimizer.zero_grad()
        loss_val = loss_fn([enc_a, enc_b], labels)
        loss_val.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        total_loss += loss_val.item()
        steps      += 1

    return total_loss / max(steps, 1)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_finetuning(
    base_model_name: str      = config.EMBEDDING_MODEL,
    epochs: int               = config.FINETUNE_EPOCHS,
    batch_size: int           = config.FINETUNE_BATCH_SIZE,
    warmup_ratio: float       = config.FINETUNE_WARMUP_RATIO,
    max_length: int           = config.FINETUNE_MAX_LENGTH,
    output_dir: Optional[str] = None,
    pairs_dir: Path           = Path(config.PAIRS_DIR),
    eval_sample: int          = config.FINETUNE_EVAL_SAMPLE,
    patience: int             = config.FINETUNE_PATIENCE,
    save_eval_json: bool      = True,
) -> FinetuneRunResult:

    run_id = datetime.now().strftime("%Y%m%dT%H%M%S")

    if output_dir is None:
        slug       = base_model_name.replace("/", "_").replace("-", "_")
        output_dir = f"models/finetuned_{slug}_{run_id}"

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    Path(config.EVAL_RUNS_DIR).mkdir(parents=True, exist_ok=True)

    log.info("=" * 60)
    log.info("Run %s | model: %s", run_id, base_model_name)
    log.info("=" * 60)

    # 1. Load pairs
    train_examples, _, pair_stats = load_pairs(pairs_dir)
    positive_only = [e for e in train_examples if e.label == 1.0]
    log.info("Positive pairs for MNRL: %d", len(positive_only))

    # 2. Load model + select device
    log.info("Loading %s", base_model_name)
    model   = SentenceTransformer(base_model_name)
    emb_dim = model.get_sentence_embedding_dimension()

    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    log.info("Device: %s | Embedding dim: %d", device, emb_dim)
    model = model.to(device)

    # 3. Pre-tokenize + DataLoader
    dataset = TokenizedPairDataset(positive_only, model.tokenizer, max_length)
    train_dataloader = DataLoader(
        dataset,
        shuffle=True,
        batch_size=batch_size,
        collate_fn=collate_pretokenized,
        num_workers=0,
        pin_memory=False,
    )

    # 4. Loss / optimiser / scheduler
    loss_fn      = losses.MultipleNegativesRankingLoss(model).to(device)
    total_steps  = len(train_dataloader) * epochs
    warmup_steps = int(total_steps * warmup_ratio)
    optimizer    = AdamW(model.parameters(), lr=config.FINETUNE_LR, eps=1e-6)
    scheduler    = LinearLR(
        optimizer,
        start_factor=1e-3,
        end_factor=1.0,
        total_iters=max(warmup_steps, 1),
    )
    log.info("Total steps: %d | Warmup: %d | LR: %s", total_steps, warmup_steps, config.FINETUNE_LR)

    # 5. Epoch loop with early stopping
    val_wines_path         = pairs_dir / "val_wines.csv"
    best_map, best_epoch   = -1.0, 0
    no_improve             = 0
    eval_history: list[EvalResult] = []
    best_ckpt              = str(Path(output_dir) / "best_checkpoint")

    for epoch in range(1, epochs + 1):
        log.info("--- Epoch %d / %d ---", epoch, epochs)
        loss = train_one_epoch(
            model, train_dataloader, loss_fn, optimizer, scheduler, device, epoch
        )
        log.info("Loss: %.4f", loss)

        metrics     = compute_map_at_k(model, val_wines_path, k=5, sample=eval_sample)
        current_map = metrics["map@5"]
        eval_history.append(EvalResult(
            epoch=epoch,
            map_at_5=current_map,
            region_hit_rate=metrics["region_hit_rate"],
            grape_recall=metrics["grape_recall"],
        ))
        log.info("MAP@5=%.4f  region_hit=%.4f  grape_recall=%.4f",
                 current_map, metrics["region_hit_rate"], metrics["grape_recall"])

        if current_map > best_map:
            best_map, best_epoch, no_improve = current_map, epoch, 0
            model.save(best_ckpt)
            log.info("✓ Best MAP@5=%.4f — checkpoint saved", best_map)
        else:
            no_improve += 1
            log.info("No improvement %d/%d (best %.4f @ epoch %d)",
                     no_improve, patience, best_map, best_epoch)
            if no_improve >= patience:
                log.info("Early stopping.")
                break

    # 6. Restore best checkpoint → save as final
    model = SentenceTransformer(best_ckpt)
    final_path = str(Path(output_dir) / "final")
    model.save(final_path)
    log.info("Final model → %s", final_path)

    # 7. Save eval JSON
    best_metrics = asdict(next(e for e in eval_history if e.epoch == best_epoch))
    run_result   = FinetuneRunResult(
        run_id=run_id,
        model_name=Path(output_dir).name + "/final",
        model_type="finetuned",
        base_model=base_model_name,
        embedding_dim=emb_dim,
        training_pairs=len(positive_only),
        val_pairs=pair_stats.val_pairs,
        epochs_trained=len(eval_history),
        best_epoch=best_epoch,
        loss="MultipleNegativesRankingLoss",
        batch_size=batch_size,
        pair_stats=asdict(pair_stats),
        eval_history=[asdict(e) for e in eval_history],
        best_metrics=best_metrics,
        output_dir=str(Path(output_dir).resolve()),
    )

    if save_eval_json:
        p = Path(config.EVAL_RUNS_DIR) / f"run_{run_id}.json"
        with open(p, "w", encoding="utf-8") as f:
            json.dump(asdict(run_result), f, indent=2, ensure_ascii=False)
        log.info("Eval JSON → %s", p)

    log.info("Done. Best epoch %d  MAP@5 %.4f", best_epoch, best_map)
    return run_result
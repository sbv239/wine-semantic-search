"""
src/search.py

Search logic: encode query → FAISS search → return top-K wines with metadata.

Usage:
    python -m src.search
    python -m src.search --query "elegant red with dried cherry" --top_k 5
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd

import config
from src.index import load_index
from sentence_transformers import SentenceTransformer


# ---------------------------------------------------------------------------
# Data class
# ---------------------------------------------------------------------------

@dataclass
class WineResult:
    id: int
    title: str
    score: float
    region: str
    country: str
    grapes: str
    colour: str
    vintage: str
    description: str
    similarity: float

    def __str__(self) -> str:
        return (
            f"[{self.id}] {self.title}\n"
            f"  Score: {self.score}  |  {self.colour}  |  {self.region}, {self.country}  |  {self.vintage}\n"
            f"  Grapes: {self.grapes}\n"
            f"  Similarity: {self.similarity:.4f}\n"
            f"  \"{self.description[:200]}{'...' if len(self.description) > 200 else ''}\"\n"
        )


# ---------------------------------------------------------------------------
# Searcher
# ---------------------------------------------------------------------------

class WineSearcher:
    """Loads FAISS index + metadata once; exposes .search(), .similar(), .get_wine()."""

    def __init__(
        self,
        model_name: str       = config.EMBEDDING_MODEL,
        index_path: str | None = None,
        data_path: str        = config.DATA_PROCESSED_PATH,
    ) -> None:
        # Auto-select index: finetuned model → finetuned index (if exists)
        if index_path is None:
            if (model_name != config.EMBEDDING_MODEL
                    and os.path.exists(config.FINETUNED_INDEX_PATH)):
                index_path = config.FINETUNED_INDEX_PATH
            else:
                index_path = config.FAISS_INDEX_PATH

        self._model_name = model_name
        self._index_path = index_path
        self._model      = SentenceTransformer(model_name)
        self._index      = load_index(index_path)
        self._df         = pd.read_csv(data_path).reset_index(drop=True)

        assert len(self._df) == self._index.ntotal, (
            f"Mismatch: CSV has {len(self._df)} rows but FAISS has {self._index.ntotal} vectors.\n"
            f"Index: {index_path}\n"
            "Run: python scripts/build_finetuned_index.py --model <model_path>"
        )

    def __len__(self) -> int:
        return self._index.ntotal

    def search(self, query: str, top_k: int = config.TOP_K) -> list[WineResult]:
        if not query.strip():
            raise ValueError("Query must not be empty.")
        vec = self._model.encode(
            [query], convert_to_numpy=True, normalize_embeddings=True
        ).astype("float32")
        distances, indices = self._index.search(vec, top_k)
        return self._build_results(indices[0], distances[0])

    def similar(self, wine_id: int, top_k: int = config.TOP_K) -> list[WineResult]:
        if wine_id < 0 or wine_id >= len(self._df):
            raise IndexError(f"wine_id {wine_id} out of range (0–{len(self._df) - 1}).")
        vec = self._index.reconstruct(wine_id).reshape(1, -1).astype("float32")
        distances, indices = self._index.search(vec, top_k + 1)
        mask = indices[0] != wine_id
        return self._build_results(indices[0][mask][:top_k], distances[0][mask][:top_k])

    def get_wine(self, wine_id: int) -> WineResult:
        if wine_id < 0 or wine_id >= len(self._df):
            raise IndexError(f"wine_id {wine_id} out of range (0–{len(self._df) - 1}).")
        return self._row_to_result(wine_id, self._df.iloc[wine_id], similarity=1.0)

    def _build_results(self, indices, distances) -> list[WineResult]:
        return [
            self._row_to_result(int(idx), self._df.iloc[int(idx)], float(dist))
            for idx, dist in zip(indices, distances)
            if idx != -1
        ]

    @staticmethod
    def _row_to_result(idx: int, row: pd.Series, similarity: float) -> WineResult:
        return WineResult(
            id=idx,
            title=str(row.get("title", "Unknown")),
            score=float(row.get("score", 0)),
            region=str(row.get("Region", "Unknown")),
            country=str(row.get("Country", "Unknown")),
            grapes=str(row.get("Grapes", "Unknown")),
            colour=str(row.get("Colour", "Unknown")),
            vintage=str(row.get("Vintage", "Unknown")),
            description=str(row.get("description", "")),
            similarity=similarity,
        )


# ---------------------------------------------------------------------------
# Console runner
# ---------------------------------------------------------------------------

EXAMPLE_QUERIES = [
    "elegant red with dried cherry and mineral finish",
    "crisp white with citrus and green apple, high acidity",
    "full-bodied Bordeaux blend with cassis and cedar",
    "floral Riesling with honeyed apricot and slate minerality",
    "earthy Pinot Noir with red berry and forest floor",
    "rich buttery Chardonnay with vanilla and toasty oak",
    "peppery Syrah with dark fruit and smoked meat",
    "sparkling wine with brioche and green apple, persistent bubbles",
    "sweet dessert wine with botrytis and marmalade notes",
    "fresh rosé with strawberry and watermelon, dry finish",
]


def run_console(query: str | None, top_k: int) -> None:
    print("=" * 70)
    print("  Wine Semantic Search — Console Runner")
    print("=" * 70)
    print("\nLoading index and model…")
    t0 = time.time()
    searcher = WineSearcher()
    print(f"Ready in {time.time() - t0:.1f}s\n")

    for q in ([query] if query else EXAMPLE_QUERIES):
        print(f"\n{'─' * 70}")
        print(f"  Query: \"{q}\"")
        print(f"{'─' * 70}")
        t1      = time.time()
        results = searcher.search(q, top_k=top_k)
        print(f"  Found {len(results)} results in {(time.time()-t1)*1000:.1f} ms\n")
        for i, r in enumerate(results, 1):
            print(f"  #{i} {r}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Wine semantic search — console runner")
    parser.add_argument("--query",  type=str, default=None)
    parser.add_argument("--top_k", type=int, default=config.TOP_K)
    args = parser.parse_args()
    run_console(query=args.query, top_k=args.top_k)


if __name__ == "__main__":
    main()
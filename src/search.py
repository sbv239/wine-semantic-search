"""
src/search.py
-------------
Search logic: encode query → FAISS search → return top-K wines with metadata.

Usage (console runner):
    python -m src.search
    python -m src.search --query "elegant red with dried cherry" --top_k 5
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import config
from src.index import load_index
from sentence_transformers import SentenceTransformer


# ---------------------------------------------------------------------------
# Data classes
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
# Searcher (singleton-friendly — load once, query many times)
# ---------------------------------------------------------------------------

class WineSearcher:
    """Loads FAISS index + metadata once; exposes .search() and .similar()."""

    def __init__(
        self,
        index_path: str = config.FAISS_INDEX_PATH,
        data_path: str = config.DATA_PROCESSED_PATH,
        model_name: str = config.EMBEDDING_MODEL,
    ) -> None:
        # Load model once — reused for all queries
        self._model = SentenceTransformer(model_name)

        # Load FAISS index
        self._index = load_index(index_path)

        # Load metadata
        self._df = pd.read_csv(data_path)
        self._df = self._df.reset_index(drop=True)  # ensure 0-based positional index

        assert len(self._df) == self._index.ntotal, (
            f"Mismatch: CSV has {len(self._df)} rows but FAISS has {self._index.ntotal} vectors."
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def search(self, query: str, top_k: int = config.TOP_K) -> list[WineResult]:
        """Encode free-text query and return top-K similar wines."""
        if not query.strip():
            raise ValueError("Query must not be empty.")

        # Encode query — L2-normalised, shape (1, 384)
        query_vec = self._model.encode(
            [query], convert_to_numpy=True, normalize_embeddings=True
        ).astype("float32")

        distances, indices = self._index.search(query_vec, top_k)

        return self._build_results(indices[0], distances[0])

    def similar(self, wine_id: int, top_k: int = config.TOP_K) -> list[WineResult]:
        """Return wines similar to a given wine (by its positional index in the dataset)."""
        if wine_id < 0 or wine_id >= len(self._df):
            raise IndexError(f"wine_id {wine_id} out of range (0–{len(self._df) - 1}).")

        # Reconstruct vector from FAISS (works for IndexFlat*)
        vec = self._index.reconstruct(wine_id).reshape(1, -1).astype("float32")

        # top_k + 1 to exclude the wine itself from results
        distances, indices = self._index.search(vec, top_k + 1)

        # Drop the query wine itself
        mask = indices[0] != wine_id
        filtered_indices = indices[0][mask][:top_k]
        filtered_distances = distances[0][mask][:top_k]

        return self._build_results(filtered_indices, filtered_distances)

    def get_wine(self, wine_id: int) -> WineResult:
        """Retrieve a single wine by its positional index."""
        if wine_id < 0 or wine_id >= len(self._df):
            raise IndexError(f"wine_id {wine_id} out of range (0–{len(self._df) - 1}).")
        row = self._df.iloc[wine_id]
        return self._row_to_result(wine_id, row, similarity=1.0)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_results(
        self, indices: np.ndarray, distances: np.ndarray
    ) -> list[WineResult]:
        results = []
        for idx, dist in zip(indices, distances):
            if idx == -1:  # FAISS returns -1 when fewer results than top_k
                continue
            row = self._df.iloc[int(idx)]
            results.append(self._row_to_result(int(idx), row, similarity=float(dist)))
        return results

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
# Console runner  (python -m src.search)
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
    print("  Wine Semantic Search — Sprint 3 Console Runner")
    print("=" * 70)

    print("\nLoading index and model…")
    t0 = time.time()
    searcher = WineSearcher()
    print(f"Ready in {time.time() - t0:.1f}s\n")

    queries = [query] if query else EXAMPLE_QUERIES

    for q in queries:
        print(f"\n{'─' * 70}")
        print(f"  Query: \"{q}\"")
        print(f"{'─' * 70}")

        t1 = time.time()
        results = searcher.search(q, top_k=top_k)
        elapsed = (time.time() - t1) * 1000

        print(f"  Found {len(results)} results in {elapsed:.1f} ms\n")
        for i, r in enumerate(results, 1):
            print(f"  #{i} {r}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Wine semantic search — console runner")
    parser.add_argument("--query", type=str, default=None, help="Custom query string")
    parser.add_argument("--top_k", type=int, default=config.TOP_K, help="Number of results")
    args = parser.parse_args()

    run_console(query=args.query, top_k=args.top_k)


if __name__ == "__main__":
    main()
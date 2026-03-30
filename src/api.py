"""
src/api.py — FastAPI app for Wine Semantic Search
Sprint 4: POST /search, GET /wine/{id}, GET /similar/{id}, GET /health
"""

import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

import config
from src.search import WineSearcher

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Singleton: one WineSearcher for the entire process lifetime
# ---------------------------------------------------------------------------

_searcher: WineSearcher | None = None


def get_searcher() -> WineSearcher:
    """Return the process-wide WineSearcher instance."""
    if _searcher is None:
        raise RuntimeError("WineSearcher not initialised — startup failed?")
    return _searcher


# ---------------------------------------------------------------------------
# Lifespan: load model + index once at startup
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    global _searcher
    logger.info("Loading WineSearcher (model + FAISS index)…")
    t0 = time.perf_counter()
    _searcher = WineSearcher()
    elapsed = time.perf_counter() - t0
    logger.info("WineSearcher ready in %.2fs — %d wines indexed", elapsed, len(_searcher))
    yield
    logger.info("Shutting down — releasing resources")
    _searcher = None


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Wine Semantic Search",
    description=(
        "Semantic search over ~10 000 Decanter wine reviews. "
        "Find similar wines by free-text tasting-note query."
    ),
    version="0.1.0",
    lifespan=lifespan,
)

# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------

class SearchRequest(BaseModel):
    query: str = Field(
        ...,
        min_length=3,
        max_length=500,
        examples=["elegant red with dried cherry and mineral finish"],
    )
    top_k: int = Field(
        default=config.TOP_K,
        ge=1,
        le=50,
        description="Number of results to return (1–50)",
    )


class WineResult(BaseModel):
    id: int
    title: str
    score: float
    region: str
    grapes: str
    description: str
    similarity: float


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health", tags=["meta"])
def health():
    """Liveness check — also reports how many wines are indexed."""
    searcher = get_searcher()
    return {"status": "ok", "wines_indexed": len(searcher)}


@app.post("/search", response_model=list[WineResult], tags=["search"])
def search(req: SearchRequest):
    """
    Semantic search: encode the query and return the top-K most similar wines.
    """
    searcher = get_searcher()
    logger.info("POST /search  query=%r  top_k=%d", req.query, req.top_k)

    t0 = time.perf_counter()
    results = searcher.search(req.query, top_k=req.top_k)
    elapsed_ms = (time.perf_counter() - t0) * 1000

    logger.info("Search completed in %.1f ms — %d results returned", elapsed_ms, len(results))
    return results


@app.get("/wine/{wine_id}", response_model=WineResult, tags=["search"])
def get_wine(wine_id: int):
    """
    Retrieve a single wine by its index ID.
    """
    searcher = get_searcher()
    logger.info("GET /wine/%d", wine_id)

    wine = searcher.get_wine(wine_id)
    if wine is None:
        raise HTTPException(status_code=404, detail=f"Wine id={wine_id} not found")
    return wine


@app.get("/similar/{wine_id}", response_model=list[WineResult], tags=["search"])
def similar(wine_id: int, top_k: int = config.TOP_K):
    """
    Find wines most similar to the wine at the given index ID.
    The source wine itself is excluded from results.
    """
    if top_k < 1 or top_k > 50:
        raise HTTPException(status_code=422, detail="top_k must be between 1 and 50")

    searcher = get_searcher()
    logger.info("GET /similar/%d  top_k=%d", wine_id, top_k)

    results = searcher.similar(wine_id, top_k=top_k)
    if results is None:
        raise HTTPException(status_code=404, detail=f"Wine id={wine_id} not found")
    return results
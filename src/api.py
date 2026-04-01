"""
src/api.py — FastAPI app for Wine Semantic Search

Endpoints:
    POST /search          — semantic search by free-text query
    GET  /wine/{id}       — wine by index ID
    GET  /similar/{id}    — find similar wines
    GET  /health          — healthcheck + active model info

Usage:
    # Base model (default)
    uvicorn src.api:app --reload

    # Finetuned model
    WINE_MODEL=models/finetuned_.../final uvicorn src.api:app --reload
"""

import logging
import os
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

import config
from src.search import WineSearcher

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_searcher: WineSearcher | None = None


def get_searcher() -> WineSearcher:
    if _searcher is None:
        raise RuntimeError("WineSearcher not initialised — startup failed?")
    return _searcher


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    global _searcher

    # Model can be overridden via env var — useful for switching base/finetuned
    model_name = os.environ.get("WINE_MODEL", config.EMBEDDING_MODEL)

    logger.info("=" * 55)
    logger.info("Starting Wine Semantic Search API")
    logger.info("Model : %s", model_name)
    logger.info("=" * 55)

    t0        = time.perf_counter()
    _searcher = WineSearcher(model_name=model_name)
    elapsed   = time.perf_counter() - t0

    logger.info("WineSearcher ready in %.2fs — %d wines indexed", elapsed, len(_searcher))
    logger.info("Index : %s", _searcher._index_path)

    yield

    logger.info("Shutting down.")
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
# Schemas
# ---------------------------------------------------------------------------

class SearchRequest(BaseModel):
    query: str = Field(
        ..., min_length=3, max_length=500,
        examples=["elegant red with dried cherry and mineral finish"],
    )
    top_k: int = Field(default=config.TOP_K, ge=1, le=50)


class WineResponse(BaseModel):
    id:          int
    title:       str
    score:       float
    region:      str
    grapes:      str
    description: str
    similarity:  float


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health", tags=["meta"])
def health():
    """Liveness check — reports active model and index."""
    searcher = get_searcher()
    return {
        "status":       "ok",
        "wines_indexed": len(searcher),
        "model":        searcher._model_name,
        "index":        searcher._index_path,
    }


@app.post("/search", response_model=list[WineResponse], tags=["search"])
def search(req: SearchRequest):
    """Semantic search: encode query → return top-K most similar wines."""
    searcher = get_searcher()
    logger.info("POST /search  query=%r  top_k=%d", req.query, req.top_k)

    t0      = time.perf_counter()
    results = searcher.search(req.query, top_k=req.top_k)
    elapsed = (time.perf_counter() - t0) * 1000

    logger.info("Search done in %.1f ms — %d results", elapsed, len(results))
    return results


@app.get("/wine/{wine_id}", response_model=WineResponse, tags=["search"])
def get_wine(wine_id: int):
    """Retrieve a single wine by its index ID."""
    searcher = get_searcher()
    logger.info("GET /wine/%d", wine_id)
    try:
        return searcher.get_wine(wine_id)
    except IndexError:
        raise HTTPException(status_code=404, detail=f"Wine id={wine_id} not found")
    
    
@app.get("/similar/{wine_id}", response_model=list[WineResponse], tags=["search"])
def similar(
    wine_id: int,
    top_k: int = Query(default=config.TOP_K, ge=1, le=50),
    ):
    searcher = get_searcher()
    logger.info("GET /similar/%d  top_k=%d", wine_id, top_k)
    try:
        return searcher.similar(wine_id, top_k=top_k)
    except IndexError:
        raise HTTPException(status_code=404, detail=f"Wine id={wine_id} not found")    
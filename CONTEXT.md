# Wine Semantic Search — Project Context

## Goal

Build a semantic search system for wine recommendations. The system understands
the **meaning** of tasting notes and finds similar wines by free-text query —
a simplified version of what Liv-ex does internally.

Example: query `"elegant red with dried cherry and mineral finish"` returns
structurally similar wines regardless of exact wording.

## Dataset

**Source:** Custom-scraped from Decanter using our own scraper
([decanter-wine-scraper](https://github.com/sbv239/decanter-wine-scraper))

**Size:** ~10,000 wine reviews (9,978 with tasting notes)

**Fields collected:**

| Field | Description |
|---|---|
| `url` | Source URL |
| `title` | Full wine name |
| `description` | Tasting notes — **primary field for embeddings** |
| `score` | Decanter score (e.g. 93) |
| `Producer` | Producer name |
| `Brand` | Sub-label / cuvée |
| `Vintage` | Vintage year |
| `Wine Type` | Still / Sparkling / Fortified |
| `Colour` | Red / White / Rosé / Orange |
| `Country` | Country of origin |
| `Region` | Wine region |
| `Appellation` | Appellation or MGA |
| `Sweetness` | Dry / Off-dry / Sweet |
| `Closure` | Cork / Screwcap / etc. |
| `Alcohol` | ABV percentage |
| `Body` | Light / Medium / Full |
| `Oak` | Yes / No |
| `Grapes` | Grape varieties with percentages |

**Why Decanter over Wine Enthusiast (Kaggle):**
- More structured metadata (appellation, body, oak, alcohol)
- Higher-quality tasting notes written by professional critics
- Recent vintages (2020–2025)
- We own the collection pipeline — can add more data at any time

## Stack

| Component | Choice | Reason |
|---|---|---|
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` | Fast, good quality for MVP |
| Vector DB | FAISS (`IndexFlatL2`) | No external service, saves to disk, strong CV signal |
| API | FastAPI | Already familiar, async, clean |
| Deployment | AWS EC2 | Required for CV line |
| Fine-tuning | Sentence Transformers + contrastive loss | Sprint 6, after MVP is working |

## Pipeline

```
wine_dataset.csv
      ↓
preprocessing.py       # clean, deduplicate, filter short descriptions
      ↓
embeddings.py          # encode descriptions → save as .npy
      ↓
index.py               # build FAISS index → save as .bin
      ↓
search.py              # query → encode → FAISS → top-K results
      ↓
api.py                 # FastAPI endpoints
      ↓
EC2                    # deployment
```

## Project Structure

```
wine-semantic-search/
├── data/
│   ├── raw/
│   │   └── wine_dataset.csv          # original scraped data (gitignored)
│   └── processed/
│       └── wines_clean.csv           # after preprocessing (gitignored)
├── notebooks/
│   └── 01_eda.ipynb                  # distribution analysis, null check
├── models/
│   ├── embeddings.npy                # encoded descriptions (gitignored)
│   └── faiss_index.bin               # FAISS index (gitignored)
├── scripts/
│   └── build_index.py                # one-time runner: preprocess → embed → index
├── src/
│   ├── preprocessing.py              # text cleaning, dedup, filtering
│   ├── embeddings.py                 # encode with SentenceTransformer, save .npy
│   ├── index.py                      # build and load FAISS index
│   ├── search.py                     # search logic: query → top-K wines
│   ├── api.py                        # FastAPI app
│   └── evaluation.py                 # region hit rate, grape recall metrics
├── config.py                         # all parameters here, nothing hardcoded
├── requirements.txt
├── .gitignore
└── README.md
```

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/search` | Query → top-K wines |
| `GET` | `/wine/{id}` | Wine by index ID |
| `GET` | `/similar/{id}` | Find similar to a given wine |
| `GET` | `/health` | Healthcheck |

**Request example:**
```json
POST /search
{
  "query": "elegant red with dried cherry and mineral finish",
  "top_k": 5
}
```

**Response example:**
```json
[
  {
    "id": 142,
    "title": "Pio Cesare, Barolo, Piedmont, Italy, 2022",
    "score": 93,
    "region": "Piedmont",
    "grapes": "100% Nebbiolo",
    "description": "Balsamic herb, dusty earth, black tea and dried cherry...",
    "similarity": 0.94
  }
]
```

## Config (config.py)

All parameters in one place — nothing hardcoded in modules:

```python
EMBEDDING_MODEL   = "sentence-transformers/all-MiniLM-L6-v2"
TOP_K             = 5
MIN_DESC_LENGTH   = 50      # minimum characters to keep a description
FAISS_INDEX_PATH  = "models/faiss_index.bin"
EMBEDDINGS_PATH   = "models/embeddings.npy"
DATA_PATH         = "data/processed/wines_clean.csv"
```

## Evaluation Metrics

| Metric | Description |
|---|---|
| Region Hit Rate | % of top-5 results from the same region as query wine |
| Grape Recall | % of top-5 results with same primary grape variety |

Computed in `src/evaluation.py`, results documented in README.

## Decisions & Conventions

- `description` is the **only field used for embeddings** — everything else is metadata
- No chunking — Decanter tasting notes are short (<500 chars), indexed whole
- FAISS index built **once offline** via `scripts/build_index.py`, loaded at API startup
- Embeddings saved as `.npy` — no re-encoding on every restart
- Fine-tuning (contrastive loss on same-region/same-appellation pairs) — Sprint 6 only
- Deployment: EC2, no Docker for MVP
- All params in `config.py`, nothing hardcoded in modules
- `data/` and `models/` are gitignored — too large and contain scraped content

## CV Line

> Built a semantic search system for wine recommendations using transformer-based
> embeddings (Sentence-BERT), FAISS vector index, and FastAPI;
> trained on a custom dataset of ~10,000 wines scraped from Decanter;
> deployed on AWS EC2.

---

## Sprints

### ✅ Sprint 0 — Planning (done)
Stack, architecture, project structure, context.md.

### ⏳ Sprint 1 — Data & Preprocessing
**Goal:** clean dataset ready for encoding

- EDA notebook: description lengths, null counts, score distribution, top regions/grapes
- `src/preprocessing.py`: lowercase, strip HTML, remove duplicates, filter by `MIN_DESC_LENGTH`
- Output: `data/processed/wines_clean.csv`

### ⏳ Sprint 2 — Embeddings + FAISS Index
**Goal:** working vector index on disk

- `src/embeddings.py`: encode all descriptions with `all-MiniLM-L6-v2`, save as `.npy`
- `src/index.py`: build `IndexFlatL2`, add embeddings, save `.bin`
- `scripts/build_index.py`: one-time runner combining both steps
- Sanity check in notebook: cosine similarity for known wine pairs

### ⏳ Sprint 3 — Search Logic
**Goal:** search works in console

- `src/search.py`: encode query → FAISS search → return top-K with metadata
- Console runner: `python -m src.search`
- Manual qualitative check: 10 example queries, results make sense

### ⏳ Sprint 4 — FastAPI
**Goal:** working REST API

- `src/api.py`: POST /search, GET /wine/{id}, GET /similar/{id}, GET /health
- Load index + embeddings at startup (singleton pattern)
- Input validation with Pydantic
- Logging: query received, search time, results count

### ⏳ Sprint 5 — Evaluation
**Goal:** measurable quality, portfolio-ready metrics

- `src/evaluation.py`: region hit rate, grape recall over sample queries
- Evaluation runner: `python -m src.evaluation`
- Results documented in README

### ⏳ Sprint 6 — Fine-tuning
**Goal:** domain-adapted embeddings, stronger CV signal

- Build training pairs: (wine_A, wine_B) positive if same region + same primary grape
- Fine-tune with `SentenceTransformer` + cosine similarity loss
- Save fine-tuned model to `models/`
- Re-run evaluation: compare base vs fine-tuned metrics

### ⏳ Sprint 7 — Deployment
**Goal:** API live on AWS EC2

- Launch EC2 instance (t2.micro or t3.small)
- Install deps, copy project, run `build_index.py`
- Serve with `uvicorn` in `screen`
- README: deployment steps, live endpoint URL

### ⏳ Sprint 8 — README & Polish
**Goal:** portfolio-ready project

- README: Problem → Solution → Architecture diagram → Tech stack → Example results → How to run
- Clean up notebooks
- Final qualitative examples with real queries and results
# Wine Semantic Search

A semantic search engine for wine recommendations. The system understands the **meaning** of tasting notes and finds structurally similar wines by free-text query — no keyword matching required.

> Query: *"elegant red with dried cherry and mineral finish"*
> → Returns Barolo, St-Émilion, and Priorat wines with matching flavour profiles

---

## The Problem

Traditional wine search is keyword-based: search "cherry" and you get wines that literally contain the word "cherry". But two critics describing the same wine might write "dried cherry notes" and "kirsch on the palate" — completely different words, same meaning.

This system encodes tasting notes as vectors in semantic space. Similar flavour profiles cluster together regardless of wording. A query about "forest floor and red fruit" will find wines described with "sous bois and raspberry" because the model understands they mean the same thing.

---

## Architecture

```
Query: "elegant red with dried cherry"
         ↓
  SentenceTransformer          ← encodes query into 384-dim vector
         ↓
    FAISS Index                ← finds nearest neighbours (cosine similarity)
  (9,970 wines)
         ↓
   Top-K Results               ← wine metadata + similarity score
         ↓
    FastAPI                    ← REST API served on AWS EC2
```

**Why this stack:**
- `sentence-transformers/all-MiniLM-L6-v2` — fast, strong baseline for semantic similarity
- `FAISS IndexFlatIP` — exact nearest-neighbour search, no external service, saves to disk
- `FastAPI` — async, Pydantic validation, auto-generated Swagger docs
- `AWS EC2` — simple, reproducible deployment without containerisation overhead

---

## Dataset

~10,000 professional wine reviews scraped from Decanter using a [custom scraper](https://github.com/sbv239/decanter-wine-scraper).

| Field | Description |
|---|---|
| `description` | Tasting notes — the only field used for embeddings |
| `score` | Decanter critic score (85–100) |
| `Region` / `Appellation` | Geographic origin |
| `Colour` / `Body` / `Oak` | Style profile |
| `Grapes` | Varieties with percentages |
| `Vintage` | Year |

**Why Decanter over Kaggle datasets:** structured metadata (appellation, body, oak), professional-grade notes, recent vintages (2020–2025), and we own the collection pipeline.

---

## Fine-tuning

The base model was fine-tuned on domain-specific pairs using `MultipleNegativesRankingLoss`.

**Pair construction:**
- **Positive pairs** — wines scored ≥ 5 on a weighted similarity function: appellation ×3, grape variety ×3, region ×2, body/oak/sweetness ×1, same producer + different vintage −1
- **Hard negatives** — same colour/body group, cosine similarity 0.55–0.80 with base model (look similar, but structurally different)
- **Easy negatives** — cross-colour pairs (Red vs White)
- 80k total pairs: 20k positive + 20k hard negative + 40k easy negative
- 80/20 train/val split stratified by Colour × Body, zero pair leakage

**Results (n=200, top\_k=5):**

| Model | Region Hit Rate | Grape Recall |
|---|---|---|
| `all-MiniLM-L6-v2` (base) | 0.375 | 0.324 |
| Fine-tuned (epoch 4/5) | **0.430** | **0.380** |
| Δ | **+14.7%** | **+17.3%** |

---

## Example Results

**Query:** `"elegant red with dried cherry and mineral finish"`

```json
[
  {
    "id": 2045,
    "title": "Celler Masroig, Etnic, Priorat, Catalonia, Spain 2022",
    "score": 89,
    "region": "Catalonia",
    "grapes": "Grenache / Garnacha, Syrah",
    "description": "Brilliant red cherry fruit with subtle spice and a trace of graphite. Poised and flavourful, finishing with grippy tannin and an elegant sense of control.",
    "similarity": 0.7247
  },
  {
    "id": 261,
    "title": "Château Montlabert, Croix de Montlabert, St-Émilion 2023",
    "score": 89,
    "region": "Bordeaux",
    "grapes": "Cabernet Sauvignon, Merlot",
    "description": "Lightly textured and bodied, this has a nice purity of creamy red cherry fruit with touches of flint and wet stone on the mineral finish.",
    "similarity": 0.7218
  },
  {
    "id": 2074,
    "title": "Clos Alkio, Caminito a Motel, Priorat, Catalonia, Spain 2021",
    "score": 88,
    "region": "Catalonia",
    "grapes": "35% Cabernet Sauvignon, 35% Grenache, 30% Carignan",
    "description": "Cherry and redcurrant purity framed by subtle earthy nuance and spice. A poised fruit presence carries through the finish. Elegant and complex.",
    "similarity": 0.7178
  }
]
```

**Query:** `"rich buttery Chardonnay with toasted oak and vanilla"`

```json
[
  {
    "id": 5821,
    "title": "Jules Taylor, Chardonnay, Gimblett Gravels, Hawke's Bay 2023",
    "score": 92,
    "region": "Hawke's Bay",
    "grapes": "100% Chardonnay",
    "description": "Creamy and full, with buttered brioche, grilled pineapple and vanilla custard. Toasty oak well-integrated, long finish.",
    "similarity": 0.7891
  }
]
```

---

## API

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/search` | Free-text query → top-K wines |
| `GET` | `/wine/{id}` | Wine by index ID |
| `GET` | `/similar/{id}` | Find wines similar to a given wine |
| `GET` | `/health` | Status + active model info |

**Request:**
```bash
curl -X POST http://localhost:8000/search \
  -H "Content-Type: application/json" \
  -d '{"query": "elegant red with dried cherry and mineral finish", "top_k": 5}'
```

Interactive docs available at `/docs` (Swagger UI).

---

## Project Structure

```
wine-semantic-search/
├── src/
│   ├── preprocessing.py      # text cleaning, dedup, filtering
│   ├── embeddings.py         # encode descriptions → .npy
│   ├── index.py              # build and load FAISS index
│   ├── search.py             # query → top-K wines
│   ├── api.py                # FastAPI endpoints
│   ├── evaluation.py         # region hit rate, grape recall
│   └── finetune.py           # pair scoring, training loop
├── scripts/
│   ├── build_index.py        # one-time: preprocess → embed → index
│   ├── build_pairs.py        # one-time: scored pairs for fine-tuning
│   ├── build_finetuned_index.py
│   ├── run_finetune.py
│   └── compare_models.py     # base vs finetuned delta table
├── notebooks/
│   └── 01_eda.ipynb          # distribution analysis, null check
├── config.py                 # all parameters, nothing hardcoded
├── eval_results.json         # evaluation history
└── requirements.txt
```

---

## How to Run

**1. Clone and install dependencies**
```bash
git clone https://github.com/sbv239/wine-semantic-search.git
cd wine-semantic-search
python3.11 -m venv venv && source venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

**2. Add data** *(not in repo — scraped content)*

Place `wines_clean.csv` in `data/processed/`.

**3. Build the index**
```bash
python scripts/build_index.py
```

This runs preprocessing → embedding → FAISS index (~2 min on CPU).

**4. Start the API**
```bash
# Base model
uvicorn src.api:app --reload

# Fine-tuned model
WINE_MODEL=models/finetuned_.../final uvicorn src.api:app --reload
```

**5. Run evaluation**
```bash
python -m src.evaluation --model models/finetuned_.../final
```

---

## Fine-tuning Pipeline

```bash
# Build training pairs (~15 min, one-time)
python scripts/build_pairs.py

# Train (~30 min on Apple M-series CPU)
python scripts/run_finetune.py

# Rebuild FAISS index for fine-tuned model (~25s)
python scripts/build_finetuned_index.py

# Compare base vs fine-tuned
python scripts/compare_models.py
```

---

## Deployment

Deployed on AWS EC2 (t3.small, Amazon Linux 2023). Served with `uvicorn` inside a `screen` session.

```bash
screen -S wine-api
source venv/bin/activate
uvicorn src.api:app --host 0.0.0.0 --port 8000
# Ctrl+A, D to detach
```

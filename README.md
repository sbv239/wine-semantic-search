# Wine Semantic Search

A semantic search engine for wine recommendations. The system understands the **meaning** of tasting notes and finds structurally similar wines by free-text query — no keyword matching required.

> Query: *"elegant red with dried cherry and mineral finish"*
> → Returns Barolo, St-Émilion, and Priorat wines with matching flavour profiles

---

## The Problem

Traditional wine search is keyword-based: search "cherry" and you get wines that literally contain the word "cherry". But two critics describing the same wine might write "dried cherry notes" and "kirsch on the palate" — completely different words, same meaning.

This system encodes tasting notes as vectors in semantic space. Similar flavour profiles cluster together regardless of wording. A query about "forest floor and red fruit" will find wines described with "sous bois and raspberry" because the model understands they mean the same thing.

---

## Use Cases

- Wine recommendation engines (Vivino-style apps)
- Retail search (semantic instead of keyword filters)
- Sommelier tools for wine pairing
- Market analysis of flavour trends

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

Improves real-world retrieval quality, not just cosine similarity
---

## Example Results

All results produced by the fine-tuned model.

**Query:** `"elegant red with dried cherry and mineral finish"`

```json
[
  {
    "id": 2486,
    "title": "Vina Prelac, Refošk, Momjan, Istria, Coastal, Croatia, 2023",
    "score": 90,
    "region": "Istria",
    "grapes": "Refosco 100%",
    "description": "Youthful red plum, rosehip and a lifted herbaceous note. Dark cherry and blackberry depth with subtle leathery spice. Firm structure with balance, clean and juicy, finishing a little astringent yet appealing.",
    "similarity": 0.9328
  },
  {
    "id": 7360,
    "title": "Clementi, Valpolicella Classico, Veneto, Italy, 2024",
    "score": 92,
    "region": "Veneto",
    "grapes": "Corvina 35%, Corvinone 30%, Rondinella 30%",
    "description": "Light ruby with perfumed red cherry, cedar, cigar and smoked leather over subtle tertiary tones. Juicy with chalky tannins, woodsy spice and an elegant, medium-length finish.",
    "similarity": 0.9283
  },
  {
    "id": 7323,
    "title": "Serego Alighieri, Montepiazzo, Valpolicella Classico",
    "score": 92,
    "region": "Veneto",
    "grapes": "Corvina 70%, Rondinella 20%, Molinara 10%",
    "description": "Cola, root and balsamic notes over oxidative tones, with dusty red and black cherry fruit. Ripe and rounded yet slightly tannic and drying on the finish.",
    "similarity": 0.9276
  }
]
```

**Query:** `"rich buttery Chardonnay with toasted oak and vanilla"`

```json
[
  {
    "id": 7239,
    "title": "Greywacke, Chardonnay, Marlborough, New Zealand, 2014",
    "score": 89,
    "region": "Marlborough",
    "grapes": "Chardonnay 100%",
    "description": "An elegant Chardonnay beaming with tropical fruit, pineapple, ginger and crème brûlée sprinkled with toasted hazelnuts and finishing with a warm lick of alcohol.",
    "similarity": 0.9669
  },
  {
    "id": 6630,
    "title": "Tapiz, Reserve Chardonnay, Uco Valley, San Pablo, 2022",
    "score": 90,
    "region": "Mendoza",
    "grapes": "Chardonnay 100%",
    "description": "Toasty, creamy Chardonnay with a dense fruit core and lemon ice cream lift. Full-bodied, rich and generous, warm and firm in structure, then with freshness tapering on a long, creamy finish.",
    "similarity": 0.9629
  },
  {
    "id": 6635,
    "title": "El Enemigo, Chardonnay, Uco Valley, Mendoza, Argentina, 2023",
    "score": 86,
    "region": "Mendoza",
    "grapes": "Chardonnay 100%",
    "description": "Smoky, bold Chardonnay with ripe pineapple, citrus and sweet lemon notes. Creamy peach richness, flinty lift and plenty of intensity through the finish.",
    "similarity": 0.9619
  }
]
```

**Query:** `"fresh sparkling wine with brioche and green apple"`

```json
[
  {
    "id": 6931,
    "title": "Camporè, Metodo Classico Nerello Mascalese Brut, Etna, 2021",
    "score": 91,
    "region": "Sicily",
    "grapes": "Nerello Mascalese 100%",
    "description": "Creamy brioche, lemon and floral notes introduce a bright and fresh sparkling wine with good intensity of lemon peel and white peach. It spends four years on its lees, helping to round out the edges and gain complexity.",
    "similarity": 0.9386
  },
  {
    "id": 4406,
    "title": "Huré Frères, Insouciance Rosé Brut, Champagne, France",
    "score": 93,
    "region": "Champagne",
    "grapes": "Chardonnay 25%, Pinot Meunier 35%, Pinot Noir 40%",
    "description": "Fresh citrus and red fruits, with lively red plum and berry notes. Light, delicate palate with brioche and creamy texture; broad yet crisp, toasty finish.",
    "similarity": 0.9301
  },
  {
    "id": 7665,
    "title": "Segura Viudas, Vintage Brut, Cava, Penedès, Spain",
    "score": 88,
    "region": "Penedès",
    "grapes": "Macabeo 39%, Parellada 21%, Xarel-lo 34%",
    "description": "Juicy white fruit and green citrus zest aromas with hints of brioche and sweet peach. Green fruits on the palate and touches of apple peel bitterness, with good weight thanks to nine months of ageing on lees.",
    "similarity": 0.9294
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

# Fine-tuned model (replace with your actual path from models/)
WINE_MODEL=models/finetuned_<timestamp>/final uvicorn src.api:app --reload
```

**5. Run evaluation**
```bash
python -m src.evaluation --model models/finetuned_sentence_transformers_all_MiniLM_L6_v2_20260330T194333/final
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

Note: EC2 public IP changes on every instance restart. Update the IP in any
client configs after stopping and restarting the instance.

```bash
screen -S wine-api
source venv/bin/activate
uvicorn src.api:app --host 0.0.0.0 --port 8000
# Ctrl+A, D to detach
```


# =============================================================================
# config.py — single source of truth for all parameters
# =============================================================================

# ---------------------------------------------------------------------------
# Paths — data
# ---------------------------------------------------------------------------
DATA_RAW_PATH       = "data/raw/wine_dataset.csv"
DATA_PROCESSED_PATH = "data/processed/wines_clean.csv"

# ---------------------------------------------------------------------------
# Paths — base model artefacts
# ---------------------------------------------------------------------------
EMBEDDINGS_PATH  = "models/embeddings.npy"
FAISS_INDEX_PATH = "models/faiss_index.bin"

# ---------------------------------------------------------------------------
# Paths — finetuned model artefacts
# ---------------------------------------------------------------------------
FINETUNED_EMBEDDINGS_PATH = "models/embeddings_finetuned.npy"
FINETUNED_INDEX_PATH      = "models/faiss_index_finetuned.bin"

# ---------------------------------------------------------------------------
# Paths — fine-tuning pair data
# ---------------------------------------------------------------------------
PAIRS_DIR      = "data/processed"
TRAIN_PAIRS    = "data/processed/train_pairs.parquet"
VAL_PAIRS      = "data/processed/val_pairs.parquet"
TRAIN_WINES    = "data/processed/train_wines.csv"
VAL_WINES      = "data/processed/val_wines.csv"
PAIRS_META     = "data/processed/pairs_meta.json"

# ---------------------------------------------------------------------------
# Paths — eval results
# ---------------------------------------------------------------------------
EVAL_RESULTS_PATH = "eval_results.json"   # evaluation.py output
EVAL_RUNS_DIR     = "models/eval"         # finetune run JSONs

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------
TOP_K = 5

# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------
MIN_DESC_LENGTH = 50    # minimum description length to keep (preprocessing)

# ---------------------------------------------------------------------------
# Fine-tuning — pair building
# ---------------------------------------------------------------------------
MIN_DESC_LENGTH_FT      = 100   # stricter filter for fine-tuning pairs
VAL_SIZE                = 0.20
RANDOM_STATE            = 42

POSITIVE_THRESHOLD      = 5     # pair score >= this → positive pair
WEAK_POSITIVE_MIN       = 3     # pair score in [3,4] → weak positive (not used as neg)

# Pair scoring weights
W_APPELLATION           = 3
W_REGION                = 2
W_GRAPE                 = 3
W_BODY                  = 1
W_OAK                   = 1
W_SWEETNESS             = 1
W_SAME_PRODUCER_VINTAGE = -1    # same producer, different vintage → slight penalty

# Hard negative mining similarity range
HARD_NEG_SIMILARITY_MIN = 0.55
HARD_NEG_SIMILARITY_MAX = 0.80

# Pair caps
MAX_POSITIVE_TRAIN = 20_000
MAX_POSITIVE_VAL   = 5_000
HARD_NEG_RATIO     = 1.0        # hard negatives = 1× positives
EASY_NEG_RATIO     = 2.0        # easy negatives = 2× positives

# ---------------------------------------------------------------------------
# Fine-tuning — training
# ---------------------------------------------------------------------------
FINETUNE_EPOCHS      = 5
FINETUNE_BATCH_SIZE  = 32
FINETUNE_LR          = 2e-5
FINETUNE_WARMUP_RATIO = 0.1
FINETUNE_MAX_LENGTH  = 128      # max token length for tokenizer
FINETUNE_PATIENCE    = 2        # early stopping patience
FINETUNE_EVAL_SAMPLE = 200      # val wines sampled for MAP@5 per epoch
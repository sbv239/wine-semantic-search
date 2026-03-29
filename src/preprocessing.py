"""
preprocessing.py — Clean and prepare the raw Decanter dataset for embedding.

Input:  data/raw/wine_dataset.csv
Output: data/processed/wines_clean.csv

Steps:
    1. Drop rows with missing descriptions
    2. Drop rows with score = 0.0 (parsing artifact)
    3. Drop duplicate descriptions (keep first occurrence)
    4. Strip HTML tags from descriptions
    5. Drop the Brand column (not used in search or metadata)
    6. Fill nulls in metadata columns with "Unknown"
    7. Reset index
    8. Save to processed/
"""

import re
import pandas as pd
from config import DATA_RAW_PATH, DATA_PROCESSED_PATH


def strip_html(text: str) -> str:
    """Remove HTML tags from a string."""
    return re.sub(r"<[^>]+>", "", text).strip()


def preprocess(input_path: str, output_path: str) -> pd.DataFrame:
    df = pd.read_csv(input_path)
    print(f"Loaded: {len(df)} rows")

    # 1. Drop missing descriptions
    df = df.dropna(subset=["description"])
    print(f"After dropping null descriptions: {len(df)} rows")

    # 2. Drop score = 0.0
    df = df[df["score"] != 0.0]
    print(f"After dropping score=0: {len(df)} rows")

    # 3. Drop duplicate descriptions
    df = df.drop_duplicates(subset=["description"], keep="first")
    print(f"After dropping duplicate descriptions: {len(df)} rows")

    # 4. Strip HTML tags from descriptions
    df["description"] = df["description"].apply(strip_html)

    # 5. Drop Brand column
    df = df.drop(columns=["Brand"])

    # 6. Fill nulls in metadata columns with "Unknown"
    metadata_cols = [c for c in df.columns if c not in ("description", "score", "url")]
    df[metadata_cols] = df[metadata_cols].fillna("Unknown")
    print(f"Filled nulls in metadata columns with 'Unknown'")

    # 7. Reset index
    df = df.reset_index(drop=True)

    # 8. Save
    df.to_csv(output_path, index=False)
    print(f"Saved: {output_path}")

    return df


if __name__ == "__main__":
    preprocess(DATA_RAW_PATH, DATA_PROCESSED_PATH)
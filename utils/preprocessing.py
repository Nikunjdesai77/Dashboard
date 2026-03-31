"""
utils/preprocessing.py — SINGLE unified preprocessing pipeline for CarIQ
This is the ONLY preprocessing file in the project. Handles:
  1. Raw data loading
  2. Cleaning & feature engineering
  3. ML feature matrix building
  4. Single-row encoding for inference
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

from config import (
    CURRENT_YEAR, DATASET_CSV, OWNER_MAP, ML_FEATURES,
    TARGET_COL, ENCODE_COLS, TOP_BRAND_COUNT, DEFAULT_CITIES,
    RANDOM_SEED,
)
from utils.logger import get_logger

logger = get_logger("preprocessing")


# ══════════════════════════════════════════════════════════════════════════════
# 1. DATA LOADING
# ══════════════════════════════════════════════════════════════════════════════
def load_raw(path: str = None) -> pd.DataFrame:
    """Load raw CSV dataset with fallback encoding."""
    path = path or str(DATASET_CSV)
    logger.info(f"Loading raw data from: {path}")
    try:
        df = pd.read_csv(path, encoding="utf-8")
    except UnicodeDecodeError:
        df = pd.read_csv(path, encoding="latin-1")
    logger.info(f"Loaded {len(df):,} rows × {len(df.columns)} columns")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# 2. CLEANING HELPERS
# ══════════════════════════════════════════════════════════════════════════════
def extract_brand(name_series: pd.Series) -> pd.Series:
    """Extract first word of car name as brand."""
    return name_series.astype(str).str.strip().str.split().str[0]


def clean_numeric(series: pd.Series) -> pd.Series:
    """Coerce to numeric, NaN for non-parseable."""
    return pd.to_numeric(series, errors="coerce")


def encode_owner(owner_series: pd.Series) -> pd.Series:
    """Map owner string → numeric rank."""
    return owner_series.map(OWNER_MAP).fillna(1).astype(int)


def remove_outliers(df: pd.DataFrame, col: str,
                    q_low: float = 0.005, q_high: float = 0.995) -> pd.DataFrame:
    """Remove outliers using quantile boundaries."""
    lo = df[col].quantile(q_low)
    hi = df[col].quantile(q_high)
    before = len(df)
    df = df[(df[col] >= lo) & (df[col] <= hi)]
    removed = before - len(df)
    if removed > 0:
        logger.debug(f"Outlier removal on '{col}': removed {removed} rows")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# 3. MASTER PREPROCESSING PIPELINE
# ══════════════════════════════════════════════════════════════════════════════
def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """
    Full preprocessing pipeline: cleaning → feature engineering → outlier removal.
    Returns analysis-ready DataFrame (NOT ML-encoded).
    """
    logger.info("Starting preprocessing pipeline...")
    df = df.copy()

    # Normalise column names
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    # Handle alternative price column name
    if "selling_price" not in df.columns and "price" in df.columns:
        df.rename(columns={"price": "selling_price"}, inplace=True)

    # Drop full duplicates
    n_before = len(df)
    df.drop_duplicates(inplace=True)
    logger.debug(f"Removed {n_before - len(df)} duplicate rows")

    # Type coercion
    df["selling_price"] = clean_numeric(df.get("selling_price", pd.Series(dtype=float)))
    df["year"]          = clean_numeric(df.get("year", pd.Series(dtype=float)))
    df["km_driven"]     = clean_numeric(df.get("km_driven", pd.Series(dtype=float)))

    # Drop rows missing critical values
    df.dropna(subset=["selling_price", "year", "km_driven"], inplace=True)

    # Brand extraction
    df["brand"] = extract_brand(df["name"])

    # Derived features
    df["car_age"]    = (CURRENT_YEAR - df["year"].astype(int)).clip(lower=0)
    df["price_lakh"] = (df["selling_price"] / 1e5).round(2)
    df["owner_num"]  = encode_owner(df.get("owner", pd.Series(["First Owner"] * len(df))))
    df["km_per_year"] = (df["km_driven"] / df["car_age"].replace(0, 1)).round(0)

    # Simulate city (deterministic via name hash) if missing
    if "city" not in df.columns:
        df["city"] = df["name"].apply(
            lambda n: DEFAULT_CITIES[hash(str(n)) % len(DEFAULT_CITIES)]
        )

    # Fill categoricals with mode
    for col in ["fuel", "seller_type", "transmission", "owner"]:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].mode()[0])

    # Seller type normalisation
    if "seller_type" in df.columns:
        df["seller_type"] = df["seller_type"].str.strip()
        df.loc[df["seller_type"].str.contains("Trustmark", na=False), "seller_type"] = "Trustmark Dealer"

    # Outlier removal
    df = remove_outliers(df, "selling_price")
    df = remove_outliers(df, "km_driven")
    df = df[(df["km_driven"] > 0) & (df["km_driven"] < 800_000)]
    df = df[(df["year"] >= 1990) & (df["year"] <= CURRENT_YEAR)]
    df = df[df["selling_price"] > 0]
    df = df[df["car_age"] >= 0]

    # Brand bucketing (keep top N)
    top_brands = df["brand"].value_counts().head(TOP_BRAND_COUNT).index.tolist()
    df["brand_clean"] = df["brand"].apply(lambda x: x if x in top_brands else "Other")

    df.reset_index(drop=True, inplace=True)
    logger.info(f"Preprocessing complete: {len(df):,} rows retained")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# 4. ML FEATURE MATRIX BUILDER (for training)
# ══════════════════════════════════════════════════════════════════════════════
def build_ml_df(df: pd.DataFrame) -> tuple:
    """
    Build ML-ready feature matrix from preprocessed DataFrame.
    Returns: (X, y, feature_cols, label_encoders)
    
    IMPORTANT: Label encoders are fit HERE during training and persisted.
    During inference, use encode_single() with the saved encoders.
    This prevents data leakage.
    """
    logger.info("Building ML feature matrix...")
    ml = df.copy()
    encoders = {}

    for col in ENCODE_COLS:
        key = "brand" if col == "brand_clean" else col
        enc_name = "brand_enc" if col == "brand_clean" else f"{col}_enc"
        if col in ml.columns:
            le = LabelEncoder()
            ml[enc_name] = le.fit_transform(ml[col].astype(str))
            encoders[key] = le
        else:
            ml[enc_name] = 0

    feat_cols = [c for c in ML_FEATURES if c in ml.columns]
    X = ml[feat_cols].fillna(0).astype(float)
    y = ml[TARGET_COL]

    logger.info(f"Feature matrix: {X.shape[0]} samples × {X.shape[1]} features")
    logger.info(f"Features: {feat_cols}")
    return X, y, feat_cols, encoders


# ══════════════════════════════════════════════════════════════════════════════
# 5. SINGLE-ROW ENCODER (for inference — uses SAVED encoders, no leakage)
# ══════════════════════════════════════════════════════════════════════════════
def encode_single(row_dict: dict, encoders: dict) -> dict:
    """
    Encode a single input row using previously fitted label encoders.
    This is used during prediction — encoders come from training artifacts.
    """
    out = dict(row_dict)
    enc_map = {
        "fuel":         ("fuel",         "fuel_enc"),
        "seller_type":  ("seller_type",  "seller_type_enc"),
        "transmission": ("transmission", "transmission_enc"),
        "brand_clean":  ("brand",        "brand_enc"),
    }
    for raw_col, (enc_key, enc_col) in enc_map.items():
        le = encoders.get(enc_key)
        val = str(row_dict.get(raw_col, "")).strip()
        if le is not None and val in le.classes_:
            out[enc_col] = int(le.transform([val])[0])
        else:
            out[enc_col] = 0
    return out

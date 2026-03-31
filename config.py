"""
config.py — Global Configuration for CarIQ
Centralized settings: paths, constants, feature lists, random seeds.
All modules import from here — single source of truth.
"""

import os
from pathlib import Path

# ── Project root ──────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(os.path.dirname(os.path.abspath(__file__)))

# ── Random seed (reproducibility) ─────────────────────────────────────────────
RANDOM_SEED = 42

# ── Current year (used for car_age computation) ───────────────────────────────
from datetime import datetime
CURRENT_YEAR = datetime.now().year

# ── File paths ────────────────────────────────────────────────────────────────
DATA_DIR       = PROJECT_ROOT / "data"
DATA_RAW_DIR   = DATA_DIR / "raw"
DATA_PROC_DIR  = DATA_DIR / "processed"
DATASET_CSV    = DATA_RAW_DIR / "dataset.csv"

MODEL_DIR      = PROJECT_ROOT / "model"
ARTIFACTS_DIR  = MODEL_DIR / "artifacts"
MODEL_PKL      = ARTIFACTS_DIR / "model.pkl"
ENCODERS_PKL   = ARTIFACTS_DIR / "encoders.pkl"
SCALER_PKL     = ARTIFACTS_DIR / "scaler.pkl"
META_JSON      = ARTIFACTS_DIR / "model_meta.json"

ASSETS_DIR     = PROJECT_ROOT / "assets"
STYLES_CSS     = ASSETS_DIR / "styles.css"
LOG_DIR        = PROJECT_ROOT / "logs"

# ── Owner mapping ────────────────────────────────────────────────────────────
OWNER_MAP = {
    "First Owner":          1,
    "Second Owner":         2,
    "Third Owner":          3,
    "Fourth & Above Owner": 4,
    "Test Drive Car":       0,
}

# ── ML feature schema (used by Pipeline + ColumnTransformer) ─────────────────
NUM_FEATURES = ["car_age", "km_driven", "km_per_year", "owner_num"]
CAT_FEATURES = ["fuel", "seller_type", "transmission", "brand_clean"]
ML_FEATURES  = NUM_FEATURES + CAT_FEATURES   # Pipeline column order

TARGET_COL = "selling_price"

# ── Encoding columns (used by ColumnTransformer internally) ──────────────────
ENCODE_COLS = CAT_FEATURES

# ── Train/test split ────────────────────────────────────────────────────────
TEST_SIZE = 0.20

# ── Model hyperparameter grids (for tuning) ──────────────────────────────────
HYPERPARAM_GRIDS = {
    "Ridge Regression": {
        "alpha": [0.1, 1.0, 10.0, 50.0, 100.0],
    },
    "Random Forest": {
        "n_estimators": [200, 300, 500],
        "max_depth": [10, 14, 18, None],
        "min_samples_leaf": [1, 2, 4],
        "max_features": [0.6, 0.8, "sqrt"],
    },
    "Gradient Boosting": {
        "n_estimators": [200, 300, 500],
        "learning_rate": [0.04, 0.06, 0.1],
        "max_depth": [4, 5, 7],
        "subsample": [0.8, 0.85, 0.9],
        "min_samples_leaf": [2, 3, 5],
    },
    "XGBoost": {
        "n_estimators": [300, 500, 700],
        "learning_rate": [0.03, 0.04, 0.06],
        "max_depth": [5, 6, 8],
        "subsample": [0.8, 0.85, 0.9],
        "colsample_bytree": [0.7, 0.8, 0.9],
        "reg_alpha": [0.1, 0.5, 1.0],
        "reg_lambda": [1.0, 2.0, 5.0],
        "min_child_weight": [1, 3, 5],
    },
}

# ── Cross-validation ────────────────────────────────────────────────────────
CV_FOLDS = 5

# ── App constants ────────────────────────────────────────────────────────────
APP_NAME = "CarIQ"
APP_VERSION = "3.0.0"
APP_TAGLINE = "Used Car Intelligence Platform"

# ── Simulated cities (for datasets without city column) ──────────────────────
DEFAULT_CITIES = [
    "Mumbai", "Delhi", "Bangalore", "Hyderabad",
    "Chennai", "Pune", "Kolkata", "Ahmedabad",
]

# ── Top brands to keep (everything else → "Other") ──────────────────────────
TOP_BRAND_COUNT = 25

# ── Plotting ─────────────────────────────────────────────────────────────────
PALETTE = [
    "#7C3AED", "#06B6D4", "#F59E0B", "#10B981", "#F43F5E",
    "#4F46E5", "#EC4899", "#8B5CF6", "#14B8A6", "#F97316",
    "#3B82F6", "#A855F7", "#22D3EE", "#84CC16", "#EAB308",
]
ACCENT_VIOLET = "#7C3AED"
ACCENT_CYAN   = "#06B6D4"

"""
model/predict.py — Inference module for CarIQ
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Loads the FULL sklearn Pipeline (ColumnTransformer + model) from disk.
Pipeline handles all preprocessing internally — no manual encode/scale.

Inference flow:
    raw_dict → DataFrame row → Pipeline.predict() → price
"""

import os, sys, pickle, json
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import MODEL_PKL, ENCODERS_PKL, SCALER_PKL, META_JSON
from utils.logger import get_logger

logger = get_logger("model.predict")

# Feature schema (must match training)
NUM_FEATURES = ["car_age", "km_driven", "km_per_year", "owner_num"]
CAT_FEATURES = ["fuel", "seller_type", "transmission", "brand_clean"]
ALL_FEATURES = NUM_FEATURES + CAT_FEATURES


# ══════════════════════════════════════════════════════════════════════════════
# ARTIFACT LOADING
# ══════════════════════════════════════════════════════════════════════════════
def load_artifacts() -> tuple:
    """
    Load the trained Pipeline + metadata.
    Returns: (pipeline, preprocessor, preprocessor, meta)
    
    NOTE: Returns 4 values for backward compatibility with app.py.
    The pipeline contains everything — enc/scaler are the same ColumnTransformer.
    """
    if not os.path.exists(MODEL_PKL) or not os.path.exists(META_JSON):
        logger.warning(f"Missing artifacts: {MODEL_PKL} or {META_JSON}")
        return None, None, None, None

    try:
        with open(MODEL_PKL, "rb")  as f: pipeline = pickle.load(f)
        with open(META_JSON)        as f: meta     = json.load(f)

        # Backward-compat: also load enc/scaler if present (they're the same ColumnTransformer)
        enc = scaler = None
        if os.path.exists(ENCODERS_PKL):
            with open(ENCODERS_PKL, "rb") as f: enc = pickle.load(f)
        if os.path.exists(SCALER_PKL):
            with open(SCALER_PKL, "rb") as f: scaler = pickle.load(f)

        logger.info(f"Loaded: {meta.get('best_name','?')} v{meta.get('version','?')}")
        return pipeline, enc, scaler, meta

    except Exception as e:
        logger.error(f"Artifact load failed: {e}")
        return None, None, None, None


# ══════════════════════════════════════════════════════════════════════════════
# SINGLE PREDICTION
# ══════════════════════════════════════════════════════════════════════════════
def predict_price(
    pipeline,       # trained sklearn Pipeline (or raw model for legacy)
    meta: dict,
    encoders,       # unused with Pipeline approach (kept for compat)
    scaler,         # unused with Pipeline approach (kept for compat)
    input_row: dict,
) -> dict:
    """
    Predict price for a single vehicle.

    Args:
        pipeline: Trained sklearn Pipeline (preprocessor + model)
        meta:     Model metadata dict
        encoders: (unused — Pipeline handles preprocessing)
        scaler:   (unused — Pipeline handles preprocessing)
        input_row: Dict with raw feature values:
            {car_age, km_driven, km_per_year, owner_num,
             fuel, seller_type, transmission, brand_clean}

    Returns:
        Dict: {price, price_low, price_high, model_name, r2, version}
    """
    try:
        # Build a single-row DataFrame with correct column order
        row_df = pd.DataFrame([{
            feat: input_row.get(feat, 0)
            for feat in ALL_FEATURES
        }])

        # Ensure numeric types
        for col in NUM_FEATURES:
            row_df[col] = pd.to_numeric(row_df[col], errors="coerce").fillna(0)
        for col in CAT_FEATURES:
            row_df[col] = row_df[col].astype(str)

        # Pipeline.predict handles ColumnTransformer + model
        if hasattr(pipeline, 'predict') and hasattr(pipeline, 'named_steps'):
            raw_pred = float(pipeline.predict(row_df)[0])
        else:
            # Legacy fallback: pipeline is a raw model, use encoders/scaler
            from utils.preprocessing import encode_single
            row_enc   = encode_single(input_row, encoders)
            feat_cols = meta.get("feat_cols", [])
            X_in      = np.array([[row_enc.get(c, 0) for c in feat_cols]], dtype=float)
            if meta.get("use_scaled") and scaler is not None:
                X_in = scaler.transform(X_in)
            raw_pred = float(pipeline.predict(X_in)[0])

        pred = max(raw_pred, 0)
        r2   = meta["results"][meta["best_name"]]["r2"]

        result = {
            "price":      pred,
            "price_low":  pred * 0.90,
            "price_high": pred * 1.10,
            "model_name": meta["best_name"],
            "r2":         r2,
            "version":    meta.get("version", "unknown"),
        }
        logger.info(f"Predicted: ₹{pred:,.0f} ({meta['best_name']}, R²={r2:.4f})")
        return result

    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise


def batch_predict(pipeline, meta, encoders, scaler, rows: list[dict]) -> list[dict]:
    """Batch prediction for multiple vehicles."""
    return [predict_price(pipeline, meta, encoders, scaler, r) for r in rows]

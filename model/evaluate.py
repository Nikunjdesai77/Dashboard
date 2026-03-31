"""
model/evaluate.py — Evaluation & Explainability for CarIQ
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Provides:
  1. Metric computation (R², RMSE, MAE, MAPE)
  2. Model comparison tables
  3. Feature importance with business interpretation
  4. SHAP-based explainability (when shap is installed)
"""

import os, sys
import numpy as np
import pandas as pd
from sklearn.metrics import (
    r2_score, mean_squared_error, mean_absolute_error,
    mean_absolute_percentage_error,
)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.logger import get_logger

logger = get_logger("model.evaluate")

# Feature name mapping for display
FEATURE_LABELS = {
    "car_age":       "Car Age (Years)",
    "km_driven":     "KM Driven",
    "km_per_year":   "KM per Year",
    "owner_num":     "Owner Count",
    "fuel":          "Fuel Type",
    "seller_type":   "Seller Type",
    "transmission":  "Transmission",
    "brand_clean":   "Brand",
    # Legacy encoded names (backward compat)
    "fuel_enc":         "Fuel Type",
    "seller_type_enc":  "Seller Type",
    "transmission_enc": "Transmission",
    "brand_enc":        "Brand",
}

BUSINESS_IMPACT = {
    "Car Age (Years)":  "Depreciation over time — strongest price driver for most cars",
    "KM Driven":        "Physical wear indicator — higher km = lower value",
    "KM per Year":      "Usage intensity — helps distinguish city vs highway cars",
    "Owner Count":      "Perceived care — first-owner cars command premium",
    "Fuel Type":        "Fuel economics & demand — diesel/electric premiums",
    "Seller Type":      "Trust factor — dealers vs individuals",
    "Transmission":     "Urban convenience — automatics fetch 20–40% more",
    "Brand":            "Brand equity — premium vs budget segment pricing",
}


def compute_metrics(y_true, y_pred) -> dict:
    """Comprehensive regression metrics."""
    return {
        "r2":   round(float(r2_score(y_true, y_pred)), 4),
        "rmse": round(float(np.sqrt(mean_squared_error(y_true, y_pred))), 2),
        "mae":  round(float(mean_absolute_error(y_true, y_pred)), 2),
        "mape": round(float(mean_absolute_percentage_error(y_true, y_pred) * 100), 2),
    }


def generate_comparison_table(meta: dict) -> pd.DataFrame:
    """Model comparison table for dashboard display."""
    rows = []
    for name, res in meta.get("results", {}).items():
        rows.append({
            "Model":      name,
            "R² Score":   res["r2"],
            "RMSE (₹)":  f"₹{res['rmse']:,.0f}",
            "MAE (₹)":   f"₹{res.get('mae', 0):,.0f}",
            "CV R² Mean": f"{res.get('cv_r2_mean', 0):.4f}",
            "CV R² Std":  f"±{res.get('cv_r2_std', 0):.4f}",
            "Best?":      "✅" if name == meta.get("best_name") else "",
        })
    return pd.DataFrame(rows).sort_values("R² Score", ascending=False).reset_index(drop=True)


def feature_importance_analysis(meta: dict) -> pd.DataFrame | None:
    """
    Feature importance from the best model, with business impact interpretation.
    Prefers SHAP values when available, falls back to tree importance / coefficients.
    """
    best = meta.get("best_name")
    if not best:
        return None

    feat_cols = meta.get("feat_cols", [])

    # Priority: SHAP > tree importance > coefficients
    shap_imp  = meta.get("shap_importance")
    tree_imp  = meta["results"].get(best, {}).get("feat_imp")

    fi_raw   = shap_imp if shap_imp else tree_imp
    fi_source = "SHAP" if shap_imp else "Model"

    if not fi_raw or len(fi_raw) == 0:
        return None

    min_len = min(len(fi_raw), len(feat_cols))
    features = feat_cols[:min_len]
    values   = fi_raw[:min_len]

    df = pd.DataFrame({
        "Feature Code":  features,
        "Feature":       [FEATURE_LABELS.get(c, c) for c in features],
        "Importance":    values,
        "Source":        fi_source,
    }).sort_values("Importance", ascending=False).reset_index(drop=True)

    total = df["Importance"].sum()
    if total > 0:
        df["Contribution (%)"] = (df["Importance"] / total * 100).round(1)

    df["Business Impact"] = df["Feature"].map(BUSINESS_IMPACT).fillna("")

    return df


def model_summary(meta: dict) -> str:
    """Generate text summary for reports/downloads."""
    best  = meta.get("best_name", "Unknown")
    r2    = meta["results"][best]["r2"]
    rmse  = meta["results"][best]["rmse"]
    mae   = meta["results"][best].get("mae", "N/A")
    ver   = meta.get("version", "?")
    n     = len(meta.get("results", {}))
    rows  = meta.get("dataset_rows", "?")
    t     = meta.get("training_time_sec", "?")

    lines = [
        f"CarIQ Model Report (v{ver})",
        "=" * 50,
        f"Best Model:      {best}",
        f"R² Score:        {r2:.4f}",
        f"RMSE:            ₹{rmse:,.0f}",
        f"MAE:             ₹{mae:,.0f}" if isinstance(mae, (int, float)) else "",
        f"Models Trained:  {n}",
        f"Training Data:   {rows:,} rows" if isinstance(rows, int) else "",
        f"Test Split:      {meta.get('test_size', 0.2)*100:.0f}%",
        f"CV Folds:        {meta.get('cv_folds', 'N/A')}",
        f"Random Seed:     {meta.get('random_seed', 'N/A')}",
        f"Pipeline:        {meta.get('pipeline_type', 'N/A')}",
        f"Training Time:   {t}s",
        f"SHAP Available:  {'Yes' if meta.get('shap_importance') else 'No'}",
        "=" * 50,
    ]
    return "\n".join(l for l in lines if l)

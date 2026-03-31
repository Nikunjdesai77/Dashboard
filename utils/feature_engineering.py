"""
utils/feature_engineering.py — Feature engineering utilities
Separates feature logic from preprocessing for clarity.
"""

import pandas as pd
import numpy as np
from config import CURRENT_YEAR, OWNER_MAP
from utils.logger import get_logger

logger = get_logger("feature_engineering")


def compute_car_age(year_series: pd.Series) -> pd.Series:
    """Compute car age from manufacturing year."""
    return (CURRENT_YEAR - year_series.astype(int)).clip(lower=0)


def compute_km_per_year(km_series: pd.Series, age_series: pd.Series) -> pd.Series:
    """Compute average km driven per year."""
    return (km_series / age_series.replace(0, 1)).round(0)


def compute_price_lakh(price_series: pd.Series) -> pd.Series:
    """Convert price to lakhs for display."""
    return (price_series / 1e5).round(2)


def bucket_brands(brand_series: pd.Series, top_n: int = 25) -> pd.Series:
    """Keep top N brands, bucket rest as 'Other'."""
    top_brands = brand_series.value_counts().head(top_n).index.tolist()
    return brand_series.apply(lambda x: x if x in top_brands else "Other")


def generate_prediction_explanation(
    car_age: int,
    km_driven: int,
    fuel: str,
    transmission: str,
    owner_num: int,
    predicted_price: float,
    r2_score: float,
) -> dict:
    """
    Generate a human-readable explanation of the prediction in business terms.
    Returns dict with 'reasons' list and 'summary' string.
    """
    reasons = []

    # Age factor
    if car_age <= 2:
        reasons.append(("🟢", "Nearly new vehicle (≤ 2 yrs)", "positive"))
    elif car_age <= 5:
        reasons.append(("🟢", "Relatively new (3–5 yrs) — sweet spot for buyers", "positive"))
    elif car_age <= 10:
        reasons.append(("🟡", "Moderate age (6–10 yrs) — standard depreciation", "neutral"))
    else:
        reasons.append(("🔴", "Older vehicle (>10 yrs) — significant depreciation", "negative"))

    # KM factor
    if km_driven < 20_000:
        reasons.append(("🟢", "Very low mileage — minimal wear", "positive"))
    elif km_driven < 50_000:
        reasons.append(("🟢", "Low mileage — good condition likely", "positive"))
    elif km_driven < 100_000:
        reasons.append(("🟡", "Moderate mileage — standard usage", "neutral"))
    else:
        reasons.append(("🔴", "High mileage (>1L km) — may need maintenance", "negative"))

    # Fuel
    if fuel == "Diesel":
        reasons.append(("🟡", "Diesel — higher resale for highway users, but may decline in metro cities", "neutral"))
    elif fuel == "Electric":
        reasons.append(("🟢", "Electric — growing demand and premium pricing", "positive"))
    elif fuel == "CNG":
        reasons.append(("🟢", "CNG — economical fuel, high demand in urban areas", "positive"))

    # Transmission
    if transmission == "Automatic":
        reasons.append(("🟢", "Automatic — commands 20–40% premium for urban convenience", "positive"))

    # Ownership
    if owner_num == 1:
        reasons.append(("🟢", "First owner — commands 15–25% premium over multi-owner", "positive"))
    elif owner_num == 2:
        reasons.append(("🟡", "Second owner — slight discount vs first owner", "neutral"))
    elif owner_num >= 3:
        reasons.append(("🔴", "Multiple owners — discount applied due to perceived wear", "negative"))

    # Confidence
    positive_count = sum(1 for _, _, s in reasons if s == "positive")
    negative_count = sum(1 for _, _, s in reasons if s == "negative")

    if positive_count > negative_count:
        summary = "This vehicle has strong value indicators — expect competitive market positioning."
    elif negative_count > positive_count:
        summary = "Some depreciation factors are present — price reflects market adjustments."
    else:
        summary = "Balanced mix of positive and negative factors — price is in the expected range."

    return {
        "reasons": reasons,
        "summary": summary,
        "confidence_label": _confidence_label(r2_score),
    }


def _confidence_label(r2: float) -> tuple:
    """Return (label, color_class) based on R² score."""
    conf = min(int(r2 * 100), 97)
    if conf >= 85:
        return conf, "High Confidence", "emerald"
    elif conf >= 70:
        return conf, "Medium Confidence", "amber"
    else:
        return conf, "Low Confidence", "rose"

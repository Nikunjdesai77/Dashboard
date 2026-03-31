"""
utils/validation.py — Input validation for CarIQ
Validates user inputs before prediction, data integrity checks.
"""

import pandas as pd
from config import CURRENT_YEAR, OWNER_MAP
from utils.logger import get_logger

logger = get_logger(__name__)


class ValidationError(Exception):
    """Raised when user input fails validation."""
    pass


def validate_prediction_input(
    brand: str,
    year: int,
    fuel: str,
    transmission: str,
    seller_type: str,
    km_driven: int,
    owner_label: str,
    valid_brands: list,
    valid_fuels: list,
    valid_trans: list,
    valid_sellers: list,
) -> dict:
    """
    Validate all prediction inputs. Returns cleaned dict or raises ValidationError.
    """
    errors = []

    # Year
    if not isinstance(year, int) or year < 1990 or year > CURRENT_YEAR:
        errors.append(f"Year must be between 1990 and {CURRENT_YEAR}.")

    # KM Driven
    if not isinstance(km_driven, (int, float)) or km_driven <= 0 or km_driven >= 1_000_000:
        errors.append("KM Driven must be between 1 and 10,00,000.")

    # Categorical checks
    if brand not in valid_brands:
        errors.append(f"Brand '{brand}' not recognized. Choose from the dropdown.")
    if fuel not in valid_fuels:
        errors.append(f"Fuel type '{fuel}' not recognized.")
    if transmission not in valid_trans:
        errors.append(f"Transmission '{transmission}' not recognized.")
    if seller_type not in valid_sellers:
        errors.append(f"Seller type '{seller_type}' not recognized.")
    if owner_label not in OWNER_MAP:
        errors.append(f"Owner label '{owner_label}' not recognized.")

    if errors:
        msg = " | ".join(errors)
        logger.warning(f"Validation failed: {msg}")
        raise ValidationError(msg)

    car_age = CURRENT_YEAR - year
    owner_num = OWNER_MAP[owner_label]

    return {
        "brand_clean":  brand,
        "year":         year,
        "car_age":      car_age,
        "fuel":         fuel,
        "transmission": transmission,
        "seller_type":  seller_type,
        "km_driven":    km_driven,
        "km_per_year":  round(km_driven / max(car_age, 1)),
        "owner_num":    owner_num,
        "owner_label":  owner_label,
    }


def validate_dataframe(df: pd.DataFrame, required_cols: list[str]) -> bool:
    """Check that dataframe has all required columns and is non-empty."""
    if df.empty:
        logger.error("DataFrame is empty.")
        return False
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        logger.error(f"Missing columns: {missing}")
        return False
    return True

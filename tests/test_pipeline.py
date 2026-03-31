"""
tests/test_pipeline.py — Tests for CarIQ ML Pipeline
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Run: python -m pytest tests/ -v
"""

import sys, os
import pytest
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import RANDOM_SEED, ML_FEATURES, NUM_FEATURES, CAT_FEATURES, OWNER_MAP, TARGET_COL
from utils.preprocessing import load_raw, preprocess
from utils.validation import validate_prediction_input, ValidationError
from utils.feature_engineering import (
    compute_car_age, compute_km_per_year, generate_prediction_explanation,
)


class TestDataLoading:
    """Test data loading and raw file access."""

    def test_load_raw_returns_dataframe(self):
        df = load_raw()
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0

    def test_load_raw_has_required_columns(self):
        df = load_raw()
        # Must have at least name, year, selling_price, km_driven
        has_price = "selling_price" in df.columns or "price" in df.columns
        assert "name" in df.columns
        assert has_price


class TestPreprocessing:
    """Test the unified preprocessing pipeline."""

    @pytest.fixture
    def clean_df(self):
        return preprocess(load_raw())

    def test_returns_required_columns(self, clean_df):
        required = ["selling_price", "year", "km_driven", "car_age",
                     "brand_clean", "owner_num", "fuel", "transmission"]
        for col in required:
            assert col in clean_df.columns, f"Missing: {col}"

    def test_no_nulls_in_critical(self, clean_df):
        for col in ["selling_price", "year", "km_driven"]:
            assert clean_df[col].isna().sum() == 0

    def test_positive_prices(self, clean_df):
        assert (clean_df["selling_price"] > 0).all()

    def test_car_age_non_negative(self, clean_df):
        assert (clean_df["car_age"] >= 0).all()

    def test_ml_features_present(self, clean_df):
        """All ML features must exist in the preprocessed DataFrame."""
        for feat in ML_FEATURES:
            assert feat in clean_df.columns, f"ML feature missing: {feat}"

    def test_no_duplicate_index(self, clean_df):
        assert clean_df.index.is_unique


class TestValidation:
    """Test input validation."""

    def test_valid_input_passes(self):
        result = validate_prediction_input(
            brand="Maruti", year=2020, fuel="Petrol",
            transmission="Manual", seller_type="Individual",
            km_driven=30000, owner_label="First Owner",
            valid_brands=["Maruti"], valid_fuels=["Petrol"],
            valid_trans=["Manual"], valid_sellers=["Individual"],
        )
        assert result["car_age"] >= 0
        assert result["owner_num"] == 1
        assert result["km_per_year"] > 0

    def test_negative_km_fails(self):
        with pytest.raises(ValidationError):
            validate_prediction_input(
                brand="Maruti", year=2020, fuel="Petrol",
                transmission="Manual", seller_type="Individual",
                km_driven=-100, owner_label="First Owner",
                valid_brands=["Maruti"], valid_fuels=["Petrol"],
                valid_trans=["Manual"], valid_sellers=["Individual"],
            )

    def test_ancient_year_fails(self):
        with pytest.raises(ValidationError):
            validate_prediction_input(
                brand="Maruti", year=1800, fuel="Petrol",
                transmission="Manual", seller_type="Individual",
                km_driven=30000, owner_label="First Owner",
                valid_brands=["Maruti"], valid_fuels=["Petrol"],
                valid_trans=["Manual"], valid_sellers=["Individual"],
            )

    def test_unknown_brand_fails(self):
        with pytest.raises(ValidationError):
            validate_prediction_input(
                brand="UnknownBrand", year=2020, fuel="Petrol",
                transmission="Manual", seller_type="Individual",
                km_driven=30000, owner_label="First Owner",
                valid_brands=["Maruti"], valid_fuels=["Petrol"],
                valid_trans=["Manual"], valid_sellers=["Individual"],
            )


class TestFeatureEngineering:
    """Test feature engineering utilities."""

    def test_compute_car_age(self):
        ages = compute_car_age(pd.Series([2020, 2015, 2000]))
        assert (ages >= 0).all()

    def test_compute_km_per_year(self):
        kpy = compute_km_per_year(pd.Series([50000, 0, 100000]),
                                  pd.Series([5, 0, 10]))
        assert len(kpy) == 3
        assert kpy.iloc[0] == 10000.0

    def test_prediction_explanation_structure(self):
        result = generate_prediction_explanation(
            car_age=3, km_driven=25000, fuel="Petrol",
            transmission="Automatic", owner_num=1,
            predicted_price=500000, r2_score=0.85,
        )
        assert "reasons" in result
        assert "summary" in result
        assert "confidence_label" in result
        assert len(result["reasons"]) > 0

    def test_prediction_explanation_high_mileage(self):
        result = generate_prediction_explanation(
            car_age=8, km_driven=120000, fuel="Diesel",
            transmission="Manual", owner_num=3,
            predicted_price=200000, r2_score=0.70,
        )
        # Should have negative sentiment for high mileage + many owners
        sentiments = [s for _, _, s in result["reasons"]]
        assert "negative" in sentiments


class TestPipelineIntegration:
    """Integration tests for the training → prediction flow."""

    def test_feature_schema_consistency(self):
        """NUM + CAT features must equal ML_FEATURES."""
        assert NUM_FEATURES + CAT_FEATURES == ML_FEATURES

    def test_owner_map_completeness(self):
        """Owner map must have all expected labels."""
        expected = ["First Owner", "Second Owner", "Third Owner",
                    "Fourth & Above Owner", "Test Drive Car"]
        for label in expected:
            assert label in OWNER_MAP

    def test_preprocessed_data_fits_feature_schema(self):
        """After preprocessing, all ML_FEATURES columns must be present."""
        df = preprocess(load_raw())
        for feat in ML_FEATURES:
            assert feat in df.columns, f"Feature {feat} missing after preprocessing"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

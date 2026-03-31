"""
model/pipeline.py — High-level ML Pipeline Abstraction for CarIQ
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Single interface for the dashboard to interact with the ML system.

Usage:
    from model.pipeline import CarIQPipeline
    pipe = CarIQPipeline()
    pipe.ensure_ready()
    result = pipe.predict({"car_age": 5, "km_driven": 50000, ...})
"""

import os, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.train   import train
from model.predict  import load_artifacts, predict_price, batch_predict
from model.evaluate import (
    generate_comparison_table,
    feature_importance_analysis,
    model_summary,
)
from utils.logger import get_logger

logger = get_logger("model.pipeline")


class CarIQPipeline:
    """
    Unified ML pipeline interface.
    Wraps training, loading, prediction, and evaluation.
    """

    def __init__(self):
        self.pipeline = None   # trained sklearn Pipeline
        self.encoders = None   # ColumnTransformer (for compat)
        self.scaler   = None   # ColumnTransformer (for compat)
        self.meta     = None
        self._loaded  = False

    def load(self) -> bool:
        """Load artifacts from disk."""
        self.pipeline, self.encoders, self.scaler, self.meta = load_artifacts()
        self._loaded = self.pipeline is not None
        if self._loaded:
            logger.info(f"Pipeline ready: {self.meta.get('best_name')} v{self.meta.get('version')}")
        return self._loaded

    def train_and_load(self, **kwargs) -> dict:
        """Train models, then load best."""
        logger.info("Training from scratch…")
        meta = train(**kwargs)
        self.load()
        return meta

    def ensure_ready(self, **train_kwargs) -> None:
        """Load if possible, train if no artifacts exist."""
        if not self.load():
            self.train_and_load(**train_kwargs)

    @property
    def is_ready(self) -> bool:
        return self._loaded and self.pipeline is not None

    def predict(self, input_row: dict) -> dict:
        """Predict price for one vehicle."""
        assert self.is_ready, "Pipeline not ready — call ensure_ready() first"
        return predict_price(self.pipeline, self.meta, self.encoders, self.scaler, input_row)

    def predict_batch(self, rows: list[dict]) -> list[dict]:
        """Batch prediction."""
        assert self.is_ready, "Pipeline not ready"
        return batch_predict(self.pipeline, self.meta, self.encoders, self.scaler, rows)

    def get_comparison_table(self):
        return generate_comparison_table(self.meta) if self.meta else None

    def get_feature_importance(self):
        return feature_importance_analysis(self.meta) if self.meta else None

    def get_summary(self) -> str:
        return model_summary(self.meta) if self.meta else "No model trained."

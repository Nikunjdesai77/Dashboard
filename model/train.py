import sys, io
if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

"""
model/train.py — Industry-level training pipeline for CarIQ
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
KEY DESIGN DECISIONS:
  1. sklearn Pipeline + ColumnTransformer  → no manual encode/scale
  2. Encoders fit INSIDE Pipeline          → zero data leakage
  3. Train/test split BEFORE any fitting   → correct evaluation
  4. Cross-validation on training fold     → unbiased CV scores
  5. RandomizedSearchCV optional           → hyperparameter tuning
  6. SHAP values computed post-training    → model explainability
  7. All artifacts versioned via timestamp → reproducibility

Run:
    python -m model.train              # basic
    python -m model.train --tune       # + hyperparameter search
    python -m model.train --tune --cv 10
"""

import os, sys, time, pickle, json
import numpy as np
import pandas as pd
from datetime import datetime

from sklearn.model_selection import train_test_split, cross_val_score, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline as SkPipeline
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

import warnings
warnings.filterwarnings("ignore")

try:
    from xgboost import XGBRegressor
    XGBOOST_OK = True
except ImportError:
    XGBOOST_OK = False

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    RANDOM_SEED, TEST_SIZE, ARTIFACTS_DIR,
    MODEL_PKL, ENCODERS_PKL, SCALER_PKL, META_JSON,
    CV_FOLDS, HYPERPARAM_GRIDS,
    NUM_FEATURES, CAT_FEATURES,
)
from utils.preprocessing import load_raw, preprocess
from utils.logger import get_logger

logger = get_logger("model.train")

# ── Feature schema (imported from config — single source of truth) ────────────
ALL_FEATURES = NUM_FEATURES + CAT_FEATURES
TARGET = "selling_price"


# ══════════════════════════════════════════════════════════════════════════════
# SKLEARN PREPROCESSOR  (ColumnTransformer)
# ══════════════════════════════════════════════════════════════════════════════
def _build_preprocessor() -> ColumnTransformer:
    """
    ColumnTransformer that:
      - Scales numeric features with StandardScaler
      - Ordinal-encodes categoricals (unknown → -1, then treated as numeric)
    This is embedded INSIDE each Pipeline, so it fits only on training data.
    """
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUM_FEATURES),
            ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), CAT_FEATURES),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


# ══════════════════════════════════════════════════════════════════════════════
# MODEL REGISTRY — each returns a full sklearn Pipeline
# ══════════════════════════════════════════════════════════════════════════════
def _get_pipelines() -> dict:
    """Returns dict: name → sklearn Pipeline(preprocessor → model)."""
    pre = _build_preprocessor

    pipelines = {
        "Ridge Regression": SkPipeline([
            ("pre", pre()),
            ("model", Ridge(alpha=10.0, random_state=RANDOM_SEED)),
        ]),
        "Random Forest": SkPipeline([
            ("pre", pre()),
            ("model", RandomForestRegressor(
                n_estimators=300, max_depth=14, min_samples_leaf=2,
                max_features=0.8, random_state=RANDOM_SEED, n_jobs=-1,
            )),
        ]),
        "Gradient Boosting": SkPipeline([
            ("pre", pre()),
            ("model", GradientBoostingRegressor(
                n_estimators=300, learning_rate=0.06, max_depth=5,
                subsample=0.85, min_samples_leaf=3, random_state=RANDOM_SEED,
            )),
        ]),
    }

    if XGBOOST_OK:
        pipelines["XGBoost"] = SkPipeline([
            ("pre", pre()),
            ("model", XGBRegressor(
                n_estimators=500, learning_rate=0.04, max_depth=6,
                subsample=0.85, colsample_bytree=0.80,
                reg_alpha=0.5, reg_lambda=2.0, min_child_weight=3,
                random_state=RANDOM_SEED, n_jobs=-1, verbosity=0,
                eval_metric="rmse",
            )),
        ])

    return pipelines


# ══════════════════════════════════════════════════════════════════════════════
# TRAINING PIPELINE
# ══════════════════════════════════════════════════════════════════════════════
def train(
    verbose: bool = True,
    tune_hyperparams: bool = False,
    cv_folds: int = CV_FOLDS,
    progress_callback=None,
) -> dict:
    """
    Full training pipeline.

    Args:
        verbose:           Print to console
        tune_hyperparams:  Run RandomizedSearchCV
        cv_folds:          Number of cross-validation folds
        progress_callback: Optional callable(dict) receiving status updates.
                           Keys: stage, message, pct, detail (optional)
                           Example: callback({"stage":"training","message":"Training XGBoost...","pct":60})

    Data flow:
        raw CSV -> preprocess() -> train/test split -> Pipeline.fit(train) -> evaluate(test)
                                                        ^ ColumnTransformer fits HERE
                                                        ^ no leakage
    """
    def _emit(stage, message, pct, **extra):
        if verbose:
            print(f"  [{stage}] {message}")
        if progress_callback:
            progress_callback(dict(stage=stage, message=message, pct=pct, **extra))

    t0 = time.time()
    logger.info("=" * 60)
    logger.info("TRAINING PIPELINE START")
    logger.info("=" * 60)

    os.makedirs(ARTIFACTS_DIR, exist_ok=True)

    # ── 1. Load & preprocess ──────────────────────────────────────────────────
    _emit("load", "Loading & preprocessing dataset...", 5)
    raw = load_raw()
    df  = preprocess(raw)
    logger.info(f"Clean dataset: {len(df):,} rows")
    _emit("load", f"Dataset ready: {len(df):,} rows", 10, rows=len(df))

    # ── 2. Split BEFORE any encoding/scaling (no leakage) ─────────────────────
    X = df[ALL_FEATURES].copy()
    y = df[TARGET].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED,
    )
    _emit("split", f"Train: {len(X_train):,}  |  Test: {len(X_test):,}", 15,
          train_size=len(X_train), test_size=len(X_test))
    logger.info(f"Train: {len(X_train)} | Test: {len(X_test)}")

    # ── 3. Train all candidate pipelines ──────────────────────────────────────
    pipelines = _get_pipelines()
    results   = {}
    n_models  = len(pipelines)

    # Hyperparameter grid prefix for Pipeline: "model__param"
    def _prefix_grid(grid: dict) -> dict:
        return {f"model__{k}": v for k, v in grid.items()}

    for i, (name, pipe) in enumerate(pipelines.items()):
        logger.info(f"Training: {name}")
        base_pct = 15 + int((i / n_models) * 55)  # 15% -> 70%
        _emit("training", f"Training {name}...", base_pct, model_name=name, model_index=i)

        # Optional tuning
        if tune_hyperparams and name in HYPERPARAM_GRIDS:
            _emit("tuning", f"Tuning {name} (RandomizedSearchCV, {cv_folds} folds)...",
                  base_pct + 5, model_name=name)
            search = RandomizedSearchCV(
                pipe,
                param_distributions=_prefix_grid(HYPERPARAM_GRIDS[name]),
                n_iter=20, cv=cv_folds, scoring="r2",
                random_state=RANDOM_SEED, n_jobs=-1, verbose=0,
            )
            search.fit(X_train, y_train)
            pipe = search.best_estimator_
            logger.info(f"  Best params: {search.best_params_}")
            _emit("tuning", f"{name}: best params found", base_pct + 10,
                  model_name=name, best_params=search.best_params_)
        else:
            pipe.fit(X_train, y_train)

        # ── Evaluate on held-out test set ─────────────────────────────────────
        preds = pipe.predict(X_test)
        r2    = float(round(r2_score(y_test, preds), 4))
        rmse  = float(round(np.sqrt(mean_squared_error(y_test, preds)), 2))
        mae   = float(round(mean_absolute_error(y_test, preds), 2))

        # ── Cross-validation (on training set only) ───────────────────────────
        cv_scores = cross_val_score(pipe, X_train, y_train,
                                    cv=cv_folds, scoring="r2", n_jobs=-1)
        cv_mean = float(round(cv_scores.mean(), 4))
        cv_std  = float(round(cv_scores.std(), 4))

        # ── Feature importance ────────────────────────────────────────────────
        mdl = pipe.named_steps["model"]
        if hasattr(mdl, "feature_importances_"):
            fi = mdl.feature_importances_.tolist()
        elif hasattr(mdl, "coef_"):
            fi = np.abs(mdl.coef_).tolist()
        else:
            fi = None

        results[name] = dict(
            r2=r2, rmse=rmse, mae=mae,
            cv_r2_mean=cv_mean, cv_r2_std=cv_std,
            pipeline=pipe, feat_imp=fi,
        )

        done_pct = 15 + int(((i + 1) / n_models) * 55)
        _emit("trained", f"{name} done — R²={r2:.4f}  RMSE={rmse:,.0f}", done_pct,
              model_name=name, r2=r2, rmse=rmse, mae=mae,
              cv_r2_mean=cv_mean, cv_r2_std=cv_std)

        logger.info(f"  {name}: R2={r2:.4f}  RMSE={rmse:,.0f}  CV={cv_mean:.4f}+/-{cv_std:.4f}")

    # ── 4. Select best ───────────────────────────────────────────────────────
    best_name = max(results, key=lambda k: results[k]["r2"])
    best_pipe = results[best_name]["pipeline"]
    logger.info(f"BEST: {best_name} (R2={results[best_name]['r2']:.4f})")
    _emit("best", f"Best model: {best_name} (R²={results[best_name]['r2']:.4f})", 75,
          best_name=best_name, r2=results[best_name]["r2"])

    # ── 5. SHAP values (post-training explainability) ─────────────────────────
    _emit("shap", "Computing SHAP explainability...", 78)
    shap_values_list = None
    try:
        import shap
        pre       = best_pipe.named_steps["pre"]
        mdl       = best_pipe.named_steps["model"]
        X_test_t  = pre.transform(X_test)

        feat_names = NUM_FEATURES + CAT_FEATURES

        if hasattr(mdl, "feature_importances_"):
            explainer   = shap.TreeExplainer(mdl)
            shap_vals   = explainer.shap_values(X_test_t[:200])
            mean_abs    = np.abs(shap_vals).mean(axis=0).tolist()
            shap_values_list = mean_abs
            logger.info("SHAP values computed successfully")
            _emit("shap", "SHAP explainability computed", 85)
        else:
            logger.info("SHAP: skipping (model has no tree structure)")
            _emit("shap", "SHAP skipped (no tree structure)", 85)
    except ImportError:
        logger.info("SHAP not installed - skipping explainability")
        _emit("shap", "SHAP not installed (pip install shap)", 85)
    except Exception as e:
        logger.warning(f"SHAP computation failed: {e}")
        _emit("shap", f"SHAP failed: {e}", 85)

    # ── 6. Persist artifacts ─────────────────────────────────────────────────
    _emit("save", "Saving model artifacts...", 90)

    with open(MODEL_PKL, "wb") as f:
        pickle.dump(best_pipe, f)
    with open(ENCODERS_PKL, "wb") as f:
        pickle.dump(best_pipe.named_steps["pre"], f)
    with open(SCALER_PKL, "wb") as f:
        pickle.dump(best_pipe.named_steps["pre"], f)

    elapsed = round(time.time() - t0, 2)

    feat_names = NUM_FEATURES + CAT_FEATURES

    meta = {
        "best_name":         best_name,
        "feat_cols":         feat_names,
        "num_features":      NUM_FEATURES,
        "cat_features":      CAT_FEATURES,
        "use_scaled":        False,
        "version":           datetime.now().strftime("%Y%m%d_%H%M%S"),
        "training_time_sec": elapsed,
        "dataset_rows":      len(df),
        "test_size":         TEST_SIZE,
        "cv_folds":          cv_folds,
        "random_seed":       RANDOM_SEED,
        "pipeline_type":     "sklearn.pipeline.Pipeline + ColumnTransformer",
        "shap_importance":   shap_values_list,
        "tuned":             tune_hyperparams,
        "results": {
            k: {
                "r2":         r["r2"],
                "rmse":       r["rmse"],
                "mae":        r["mae"],
                "cv_r2_mean": r["cv_r2_mean"],
                "cv_r2_std":  r["cv_r2_std"],
                "feat_imp":   r["feat_imp"],
            }
            for k, r in results.items()
        },
    }
    with open(META_JSON, "w") as f:
        json.dump(meta, f, indent=2)

    logger.info(f"Artifacts -> {ARTIFACTS_DIR}  ({elapsed:.1f}s)")
    _emit("done", f"Training complete in {elapsed:.1f}s", 100,
          elapsed=elapsed, best_name=best_name, version=meta["version"])

    return meta


# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Train CarIQ ML pipeline")
    p.add_argument("--tune", action="store_true", help="Hyperparameter search")
    p.add_argument("--cv",   type=int, default=CV_FOLDS, help="CV folds")
    args = p.parse_args()
    train(verbose=True, tune_hyperparams=args.tune, cv_folds=args.cv)

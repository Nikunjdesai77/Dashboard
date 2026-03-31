<h1 align="center">🚗 CarIQ v3.0 — Used Car Price Intelligence Platform</h1>

<p align="center">
  <b>Production-grade ML pipeline + SaaS-style Analytics Dashboard</b><br>
  <i>sklearn Pipeline · ColumnTransformer · Cross-Validation · SHAP · Model Versioning</i>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11+-blue?style=flat-square&logo=python" alt="Python">
  <img src="https://img.shields.io/badge/Streamlit-1.32+-red?style=flat-square&logo=streamlit" alt="Streamlit">
  <img src="https://img.shields.io/badge/XGBoost-2.0+-green?style=flat-square" alt="XGBoost">
  <img src="https://img.shields.io/badge/Scikit--learn-1.4+-orange?style=flat-square&logo=scikit-learn" alt="Sklearn">
  <img src="https://img.shields.io/badge/SHAP-0.43+-purple?style=flat-square" alt="SHAP">
  <img src="https://img.shields.io/badge/License-MIT-green?style=flat-square" alt="License">
</p>

---

## 🌟 Why This Is Industry-Level

| Feature | Implementation |
|---------|---------------|
| **sklearn Pipeline** | End-to-end `Pipeline(ColumnTransformer → Model)` — preprocessing baked into the model |
| **Zero Data Leakage** | Train/test split BEFORE any fitting; ColumnTransformer fits only on training data |
| **Cross-Validation** | K-fold CV (configurable) on training set only |
| **Hyperparameter Tuning** | `RandomizedSearchCV` with Pipeline-prefixed param grids |
| **Model Explainability** | SHAP TreeExplainer + tree importance with business-terms interpretation |
| **Model Versioning** | Timestamped metadata — every training run is traceable |
| **Centralized Config** | Single `config.py` — paths, random seeds, feature schemas, hyperparameter grids |
| **Logging System** | Rotating file + console logger (`logs/cariq.log`) |
| **Input Validation** | Dedicated validation module with typed error handling |
| **Testing** | pytest-based smoke tests for preprocessing, validation, and features |
| **Premium Dashboard** | 4-page dark SaaS UI with glassmorphism, animated cards, and interactive charts |

---

## 🏗️ Architecture

```
CarIQ/
│
├── app.py                        # Streamlit entry point
├── config.py                     # Global config (paths, seeds, feature schema, grids)
├── requirements.txt
├── README.md
├── .gitignore
│
├── data/
│   ├── raw/
│   │   └── dataset.csv           # CarDekho raw dataset
│   ├── processed/                # Cached / transformed data
│   └── external/                 # External data sources
│
├── model/
│   ├── __init__.py               # Package exports
│   ├── train.py                  # Training: Pipeline + CV + tuning + SHAP
│   ├── predict.py                # Inference: load Pipeline → predict
│   ├── evaluate.py               # Metrics + comparison + explainability
│   ├── pipeline.py               # CarIQPipeline abstraction
│   └── artifacts/
│       ├── model.pkl             # Serialized sklearn Pipeline
│       ├── encoders.pkl          # ColumnTransformer (for compat)
│       ├── scaler.pkl            # ColumnTransformer (for compat)
│       └── model_meta.json       # Versioned metadata
│
├── utils/
│   ├── __init__.py               # Package exports
│   ├── preprocessing.py          # SINGLE data cleaning pipeline
│   ├── feature_engineering.py    # Feature computation + prediction explanation
│   ├── helpers.py                # Chart factory + formatters + CSS loader
│   ├── logger.py                 # Rotating file + console logger
│   └── validation.py             # Input validation
│
├── pages/
│   ├── 1_Overview.py             # KPIs, brand analysis, data explorer
│   ├── 2_Market_Analysis.py      # Deep-dive analytics
│   ├── 3_Price_Prediction.py     # AI prediction + model comparison + SHAP
│   └── 4_Insights.py             # Market intelligence
│
├── assets/
│   └── styles.css                # Premium dark-mode stylesheet
│
├── logs/
│   └── cariq.log                 # Auto-generated logs
│
├── notebooks/                    # EDA / experiments
└── tests/
    └── test_pipeline.py          # pytest smoke tests
```

---

## ⚙️ ML Pipeline Architecture

```
┌─────────────────────────────────────────────────────────┐
│                   sklearn.Pipeline                       │
│ ┌─────────────────────┐  ┌──────────────────────────┐   │
│ │  ColumnTransformer   │  │                          │   │
│ │  ┌─────────────────┐ │  │   XGBRegressor           │   │
│ │  │ StandardScaler   │ │→│   (or RF, GBM, Ridge)    │→ ŷ│
│ │  │ (num features)   │ │  │                          │   │
│ │  ├─────────────────┤ │  │                          │   │
│ │  │ OrdinalEncoder   │ │  │                          │   │
│ │  │ (cat features)   │ │  │                          │   │
│ │  └─────────────────┘ │  └──────────────────────────┘   │
│ └─────────────────────┘                                  │
└─────────────────────────────────────────────────────────┘
         ↑ fits ONLY on training data (no leakage)
```

**Key difference from amateur projects:** Most beginner projects manually call `LabelEncoder.fit_transform()` on the full dataset, then split. This leaks test-set information into the encoder. Our Pipeline ensures the ColumnTransformer fits **only on training data**.

---

## 🚀 Quick Start

```bash
# 1. Install
git clone https://github.com/your-username/cariq.git
cd cariq
pip install -r requirements.txt

# 2. Train (4 models, cross-validation, auto-best selection)
python -m model.train

# 3. Train with hyperparameter tuning
python -m model.train --tune

# 4. Launch dashboard
streamlit run app.py
```

---

## 🤖 Model Results

| Model | R² Score | RMSE | MAE | CV R² |
|-------|----------|------|-----|-------|
| Ridge Regression | 0.4768 | ₹2.73L | ₹1.83L | 0.4911 |
| Random Forest | 0.6704 | ₹2.17L | ₹1.32L | 0.6961 |
| Gradient Boosting | 0.7033 | ₹2.06L | ₹1.30L | 0.7278 |
| **XGBoost** | **0.7354** | **₹1.94L** | **₹1.26L** | **0.7325** |

---

## 📊 Dashboard Pages

| Page | Features |
|------|----------|
| **🏠 Home** | 8 KPI cards, price trend, fuel distribution, sidebar filters |
| **📊 Overview** | 3-tab layout (Overview/Trends/Distribution), data explorer |
| **📈 Market Analysis** | KM vs Price scatter, city comparison, depreciation curves, correlation |
| **🤖 Prediction** | Form → Pipeline.predict() → business explanation → similar cars → model comparison → SHAP |
| **💡 Insights** | Best cities, resale brands, fuel economics, depreciation milestones |

---

## 🔧 Configuration (`config.py`)

```python
RANDOM_SEED      = 42           # Reproducibility
TEST_SIZE        = 0.20         # 80/20 split
CV_FOLDS         = 5            # Cross-validation
NUM_FEATURES     = ["car_age", "km_driven", "km_per_year", "owner_num"]
CAT_FEATURES     = ["fuel", "seller_type", "transmission", "brand_clean"]
TOP_BRAND_COUNT  = 25           # Brands to keep (rest → "Other")

HYPERPARAM_GRIDS = {            # For RandomizedSearchCV
    "XGBoost": {
        "n_estimators": [300, 500, 700],
        "learning_rate": [0.03, 0.04, 0.06],
        "max_depth": [5, 6, 8],
        ...
    }
}
```

---

## 📝 Logging

```
2026-04-01 01:00:52 | INFO | cariq.preprocessing  | Loaded 4,340 rows × 8 columns
2026-04-01 01:00:53 | INFO | cariq.model.train     | BEST: XGBoost (R2=0.7354)
2026-04-01 01:01:55 | INFO | cariq.model.predict   | Predicted: ₹3,36,635 (XGBoost, R²=0.7354)
```

---

## 🧪 Tests

```bash
python -m pytest tests/ -v
```

Tests cover:
- Data loading & column presence
- Preprocessing (nulls, positive prices, feature availability)
- Input validation (valid inputs, edge cases, error handling)
- Feature engineering (car_age, km_per_year, prediction explanation)
- Schema consistency (NUM + CAT = ML_FEATURES)

---

## 📦 Deployment

```bash
# Streamlit Cloud
# 1. Push to GitHub  2. share.streamlit.io  3. Set app.py as entry
```

---

## 👤 Author

**Industry-grade ML project** demonstrating:
- ✅ `sklearn.Pipeline` + `ColumnTransformer` (not manual encoding)
- ✅ Zero data leakage (split → fit pipeline → evaluate)
- ✅ Cross-validation + hyperparameter tuning
- ✅ SHAP model explainability
- ✅ Model versioning + centralized config
- ✅ Professional logging + input validation
- ✅ pytest test suite
- ✅ Business-terms prediction explanation
- ✅ Production SaaS dashboard design

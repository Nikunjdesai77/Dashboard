# 🚗 CarIQ v3.0 — Used Car Price Prediction

> sklearn Pipeline + ColumnTransformer · XGBoost · SHAP · Streamlit Dashboard

---

## Setup

```bash
pip install -r requirements.txt
python -m model.train          # train (or use Model Lab page)
streamlit run app.py           # launch dashboard
```

## Project Structure

```
├── app.py                  # Streamlit entry point
├── config.py               # Paths, seeds, feature schema, hyperparameter grids
├── model/
│   ├── train.py            # Pipeline training + CV + SHAP
│   ├── predict.py          # Pipeline.predict() inference
│   ├── evaluate.py         # Metrics + feature importance
│   ├── pipeline.py         # CarIQPipeline abstraction
│   └── artifacts/          # model.pkl, model_meta.json
├── utils/
│   ├── preprocessing.py    # Data cleaning + feature engineering
│   ├── feature_engineering.py
│   ├── validation.py       # Input validation
│   ├── helpers.py          # Chart factory + formatters
│   └── logger.py           # Rotating file logger
├── pages/
│   ├── 1_Overview.py
│   ├── 2_Market_Analysis.py
│   ├── 3_Price_Prediction.py
│   ├── 4_Insights.py
│   └── 5_Model_Lab.py     # Automated training UI
├── data/raw/dataset.csv
└── tests/test_pipeline.py
```

## ML Pipeline

```
Raw CSV → preprocess() → train_test_split → Pipeline.fit(X_train)
                                              ├── ColumnTransformer
                                              │   ├── StandardScaler (numeric)
                                              │   └── OrdinalEncoder (categorical)
                                              └── Model (XGBoost / RF / GBM / Ridge)
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

| Model | R² | RMSE | CV R² |
|-------|-----|------|-------|
| Ridge | 0.48 | ₹2.73L | 0.49 |
| Random Forest | 0.67 | ₹2.17L | 0.70 |
| Gradient Boosting | 0.70 | ₹2.06L | 0.73 |
| **XGBoost** | **0.74** | **₹1.94L** | **0.73** |

## Commands

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



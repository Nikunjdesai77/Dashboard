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

- **No data leakage** — encoders fit only on training data (inside Pipeline)
- **4 models** compared: Ridge, Random Forest, Gradient Boosting, XGBoost
- **Cross-validation** on training set, evaluation on held-out test set
- **SHAP** explainability (optional, `pip install shap`)

## Results

| Model | R² | RMSE | CV R² |
|-------|-----|------|-------|
| Ridge | 0.48 | ₹2.73L | 0.49 |
| Random Forest | 0.67 | ₹2.17L | 0.70 |
| Gradient Boosting | 0.70 | ₹2.06L | 0.73 |
| **XGBoost** | **0.74** | **₹1.94L** | **0.73** |

## Commands

```bash
python -m model.train              # standard training
python -m model.train --tune       # + hyperparameter search
python -m model.train --tune --cv 10
python -m pytest tests/ -v         # run tests
```

## Tech Stack

Python 3.11+ · Streamlit · scikit-learn · XGBoost · SHAP · Plotly

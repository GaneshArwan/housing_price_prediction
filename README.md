# Ames Housing Price Prediction

[![QA & Security](https://github.com/GaneshArwan/housing_price_prediction/actions/workflows/ml_pipeline.yml/badge.svg)](https://github.com/GaneshArwan/housing_price_prediction/actions/workflows/ml_pipeline.yml)
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://ames-housing-price-predictor.streamlit.app/)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/GaneshArwan/housing_price_prediction/blob/main/notebooks/analysis.ipynb)
![Python 3.11](https://img.shields.io/badge/Python-3.11-blue)
![License: MIT](https://img.shields.io/badge/License-MIT-green)

Regression pipeline predicting residential sale prices in Ames, Iowa (2006–2010), paired with MLflow tracking, Pydantic output guardrails, an interactive Streamlit app, a runnable Colab notebook, and an automated two-column IEEE PDF report.

- **Live App:** https://ames-housing-price-predictor.streamlit.app/
- **Colab Notebook:** https://colab.research.google.com/github/GaneshArwan/housing_price_prediction/blob/main/notebooks/analysis.ipynb
- **IEEE Report:** [Housing_Price_Prediction_Report.pdf](reports/figures/Housing_Price_Prediction_Report.pdf)

## Dataset
- **Source:** Ames Housing Dataset (De Cock, 2011) — `2,930` sales across `81` columns (`80` explanatory variables plus `Sale_Price`).
- **Target distribution:** Prices range from `$12,789` to `$755,000` (mean `$180,796`, median `$160,000`, SD `$79,887`, skewness `1.74`). Only `64` sales (`2.2%`) equal or exceed `$400,000`.
- **Missing data:** In this mirror, `NaN` values appear in `Misc_Feature` (`2,824`) and `Mas_Vnr_Type` (`1,775`), both filled with `'Unknown'`. Zero-valued `Lot_Frontage` (`490` rows) and `'No_Garage'` `Garage_Type` (`157` rows) are already encoded in the CSV.

## Pipeline Design
1. **Preprocessing (`src/data_loader.py`):** Drops legacy identifier/sparse columns if present (`Id`, `PID`, `Alley`, `PoolQC`, `Fence`, `MiscFeature`), renames `Sale_Price` to `SalePrice`, imputes numeric `NaN`s with column medians and categorical `NaN`s with `'Unknown'`, and one-hot encodes categoricals (`drop_first=True` for training, producing `300` features without feature scaling).
2. **Inference Parity (`app.py`, `src/predict.py`):** Single-row Streamlit inference uses `drop_first=False` and reindexes onto the `300` training columns so selected categories are retained. Model loading enforces `ALLOWED_MODELS = {"random_forest", "linear_regression"}` and `PredictionGuardrail` rejects non-finite values or prices outside `$1,000`–`$10,000,000`.
3. **Modeling (`src/train.py`):** Trains OLS `LinearRegression` and `RandomForestRegressor` (`n_estimators=100`, `max_depth=None`, `random_state=42`) on an 80/20 split (`2,344` train / `586` test) with MLflow tracking.

## Holdout Results (`n=586`)
| Metric | Linear Regression | Random Forest |
|---|---:|---:|
| RMSE | \$29,026.76 | \$28,471.76 |
| Relative RMSE (% Median \$160,000) | 18.14% | 17.79% |
| MAE | \$16,478.20 | \$16,641.57 |
| R² | 0.8949 | 0.8989 |
| MAPE | 8.99% | 9.31% |
| Standard Homes RMSE (<\$400k, `n=566`) | \$24,690.27 | \$24,242.60 |
| Luxury Homes RMSE (≥\$400k, `n=20`) | \$86,225.57 | \$84,378.68 |

## Findings & Limitations
1. **Numeric linear correlations:** `Gr_Liv_Area` (`r=0.7068`), `Garage_Cars` (`r=0.6476`), `Garage_Area` (`r=0.6401`), `Total_Bsmt_SF` (`r=0.6325`), and `First_Flr_SF` (`r=0.6217`) show the highest Pearson correlations with `Sale_Price`. `Garage_Cars` and `Garage_Area` are strongly collinear (`r=0.89`).
2. **Random Forest MDI splits:** `Garage_Cars` (`29.78%`), `Gr_Liv_Area` (`14.95%`), `Year_Built` (`10.89%`), `Exter_Qual_Typical` (`6.75%`), `First_Flr_SF` (`5.38%`), and `Total_Bsmt_SF` (`5.36%`) lead tree impurity reduction.
3. **One-hot dilution of ordinal quality:** Because `Overall_Qual` is stored as text labels and one-hot encoded into 9 dummy columns, its tree importance is split across dummies (`2.20%` combined MDI).
4. **Upper-tail error:** Random Forest RMSE increases `3.48x` on luxury homes (`≥$400k`, `n=20`), where tree averaging underpredicts scarce upper-tail comparables. Next steps: ordinal-encode quality ratings, model `log1p(SalePrice)`, and benchmark gradient boosting with 5-fold cross-validation.

## Quickstart
```bash
git clone https://github.com/GaneshArwan/housing_price_prediction.git
cd housing_price_prediction
python -m venv venv
venv\Scripts\activate        # Windows (macOS/Linux: source venv/bin/activate)
pip install -r requirements.txt

python -m src.train
python -m src.evaluate
python -m src.generate_pdf_report
python -m src.predict
streamlit run app.py
pytest
```

Run in Docker:
```bash
docker build -t housing-price-predictor . && docker run -p 8501:8501 housing-price-predictor
```

## Project Structure
```
housing_price_prediction/
├── .github/workflows/ml_pipeline.yml   # Path-filtered Bandit + Pytest CI
├── data/ames_housing.csv               # Static Ames Housing dataset (2,930 x 81)
├── notebooks/analysis.ipynb            # Self-contained Colab analysis notebook
├── reports/figures/
│   ├── correlation_heatmap.png
│   ├── feature_importance.png
│   ├── residual_plot.png
│   └── Housing_Price_Prediction_Report.pdf
├── src/
│   ├── config.py                       # Pydantic configuration schemas
│   ├── data_loader.py                  # Data ingestion and preprocessing
│   ├── evaluate.py                     # Evaluation plot generation
│   ├── format_utils.py                 # Babel currency and RMSE formatting
│   ├── generate_pdf_report.py          # Data-driven IEEE two-column PDF generator
│   ├── guardrails.py                   # Pydantic + finite-number prediction guardrails
│   ├── predict.py                      # Guardrailed model inference CLI/API
│   └── train.py                        # Training with MLflow experiment tracking
├── tests/                              # Unit, guardrail, report, and Streamlit UI tests
├── app.py                              # Self-bootstrapping Streamlit dashboard
├── conftest.py                         # Pytest sys.path setup
├── Dockerfile                          # Non-root container image
├── LICENSE
├── pyproject.toml
└── requirements.txt
```

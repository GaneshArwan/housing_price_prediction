<div align="center">
  
# 🏡 Ames Housing Price Prediction 🚀

[![MLOps Pipeline](https://github.com/GaneshArwan/housing_price_prediction/actions/workflows/ml_pipeline.yml/badge.svg)](https://github.com/GaneshArwan/housing_price_prediction/actions/workflows/ml_pipeline.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=flat&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![MLflow](https://img.shields.io/badge/MLflow-Tracking-0194E2.svg?logo=mlflow&logoColor=white)](https://mlflow.org/)
[![Pydantic](https://img.shields.io/badge/Pydantic-Validation-red.svg)](https://docs.pydantic.dev/)

*An end-to-end Machine Learning pipeline to predict house sale prices using the rich Ames Housing Dataset.*

</div>

---

## 🌟 Overview

What features most influence home sale prices, and how accurately can prices be predicted? This project answers these questions by exploring the Ames Housing Dataset (2,930 observations, 80 features) and building powerful predictive models like Random Forests and Linear Regression.

We take a **data-driven approach** to understand which features a homeowner could change to increase their property value, backed by statistical correlation heatmaps and feature importance analysis.

---

## 🏗️ Project Structure

```
housing_price_prediction/
├── .github/
│   └── workflows/
│       └── ml_pipeline.yml        # CI/CD pipeline (GitHub Actions)
├── data/
│   └── ames_housing.csv           # Ames Housing Dataset (2,930 rows)
├── models/
│   ├── linear_regression.joblib   # Serialized LR model
│   └── random_forest.joblib       # Serialized RF model
├── notebooks/
│   └── analysis.ipynb             # Exploratory data analysis
├── reports/
│   └── figures/
│       ├── feature_importance.png
│       ├── correlation_heatmap.png
│       ├── residual_plot.png
│       └── Housing_Price_Prediction_Report.pdf
├── src/
│   ├── __init__.py
│   ├── config.py                  # Pydantic configuration & validation
│   ├── data_loader.py             # Data ingestion & preprocessing
│   ├── evaluate.py                # Metrics & visualization generation
│   ├── format_utils.py            # i18n currency formatting (Babel)
│   ├── generate_pdf_report.py     # Platypus PDF report generator
│   ├── guardrails.py              # Prediction value validation
│   ├── predict.py                 # Model inference with guardrails
│   └── train.py                   # Model training with MLflow tracking
├── tests/
│   ├── __init__.py
│   ├── test_data_loader.py        # QA/QC: data pipeline tests
│   ├── test_evaluate.py           # QA/QC: evaluation pipeline tests
│   ├── test_guardrails.py         # QA/QC: guardrail boundary tests
│   └── test_train.py              # QA/QC: config validation tests
├── .gitignore
├── conftest.py                    # Pytest root config
├── pyproject.toml                 # Project metadata & tool config
├── LICENSE                        # MIT License
├── README.md
└── requirements.txt
```

### 📁 Directory Breakdown

| Directory / File | Description |
| :--- | :--- |
| `.github/workflows/` | CI/CD pipeline — runs security scans, tests, training, and uploads artifacts on every push |
| `src/` | Core ML pipeline: data loading, training, evaluation, prediction, PDF reporting, and guardrails |
| `data/` | Raw dataset committed to the repo for reproducible builds |
| `models/` | Serialized trained models (`.joblib`, git-ignored) |
| `notebooks/` | Jupyter notebooks for exploratory data analysis and visual insights |
| `reports/figures/` | Generated plots and the Platypus PDF report |
| `tests/` | Pytest test suite for QA/QC (data validation, config, guardrails, evaluation) |
| `requirements.txt` | Python dependencies required to run the pipeline |
| `LICENSE` | Open-source [MIT License](LICENSE) |

---

## 🔄 Machine Learning Workflow

The pipeline is completely automated. Simply run the scripts to load data, train the models, evaluate, and generate professional reports.

```mermaid
graph TD
    A[Data Ingestion] --> B[Preprocessing & Encoding]
    B --> C{Train/Test Split}
    C -->|Train Set| D[Model Training]
    D --> E[Linear Regression]
    D --> F[Random Forest]
    E --> G[MLflow Tracking]
    F --> G
    G --> H[Log Params, Metrics, Artifacts]
    C -->|Test Set| I[Model Evaluation]
    H --> I
    I --> J[Generate Visualizations]
    J --> K((Feature Importance))
    J --> L((Residual Plot))
    J --> M((Correlation Heatmap))
    J --> N[Platypus PDF Report]
    I --> O[Prediction Inference]
    O --> P{Guardrails Check}
    P -->|Pass| Q[Validated Prediction]
    P -->|Fail| R[Flagged / NaN]

    style G fill:#0194E2,color:#fff
    style N fill:#d63384,color:#fff
    style P fill:#ffc107,color:#000
```

---

## 📊 Key Results

- **Linear Regression RMSE:** ~$29,000
- **Random Forest RMSE:** ~$28,400

Through our analysis, the most impactful features for home value are **Overall Quality**, **Living Area**, and **Basement/Garage Condition**. Homeowners should focus on these aspects for the best return on investment.

---

## 🚀 How to Run

### 1. Set up the virtual environment
```bash
python -m venv venv
.\venv\Scripts\activate       # Windows
# source venv/bin/activate    # macOS/Linux
pip install -r requirements.txt
```

### 2. Train the models (with MLflow tracking)
```bash
python -m src.train
```

### 3. Generate evaluation plots
```bash
python -m src.evaluate
```

### 4. Generate the professional PDF report
```bash
python -m src.generate_pdf_report
```

### 5. Run predictions on new data
```bash
python -m src.predict
```

### 6. Run the test suite
```bash
pytest tests/ -v
```

### 7. View MLflow experiment dashboard
```bash
mlflow ui
# Open http://127.0.0.1:5000 in your browser
```

---
**License**: This project is licensed under the [MIT License](LICENSE).

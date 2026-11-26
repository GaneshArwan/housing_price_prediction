import os
import math
import pandas as pd
import pytest
from src.config import config
from src.data_loader import preprocess_data
from src.format_utils import format_price, format_rmse
from src.guardrails import PredictionGuardrail
from src.predict import load_model, predict
from src.train import train_models


@pytest.fixture(scope="module", autouse=True)
def _ensure_artifacts():
    rf_path = os.path.join(config.paths.model_dir, "random_forest.joblib")
    x_test_path = os.path.join(config.paths.data_dir, "X_test.csv")
    if not (os.path.exists(rf_path) and os.path.exists(x_test_path)):
        train_models()


def test_load_model_rejects_invalid_name():
    with pytest.raises(ValueError, match="Invalid model"):
        load_model("../evil")


def test_predict_on_test_slice():
    X_test = pd.read_csv(os.path.join(config.paths.data_dir, "X_test.csv")).head(5)
    preds = predict(X_test, model_name="random_forest")
    assert len(preds) == 5
    assert all(math.isfinite(p) and p > 0 for p in preds)


def test_format_utils():
    assert format_price(185432.5) == "$185,432.50"
    assert format_rmse(28400.0) == "Root Mean Squared Error (RMSE): $28,400.00"


def test_guardrail_rejects_non_finite():
    assert PredictionGuardrail.validate_prediction(float("nan")) is False
    assert PredictionGuardrail.validate_prediction(float("inf")) is False
    assert PredictionGuardrail.validate_prediction(float("-inf")) is False


def test_preprocess_single_row_preserves_categoricals():
    row = pd.DataFrame([{"Overall_Qual": "Very_Excellent", "Neighborhood": "North_Ames", "Gr_Liv_Area": 1500}])
    processed = preprocess_data(row, drop_first=False)
    assert "Overall_Qual_Very_Excellent" in processed.columns
    assert "Neighborhood_North_Ames" in processed.columns
    assert bool(processed["Overall_Qual_Very_Excellent"].iloc[0]) is True
    assert bool(processed["Neighborhood_North_Ames"].iloc[0]) is True

import os
import joblib
import pandas as pd
from src.config import config
from src.guardrails import PredictionGuardrail


ALLOWED_MODELS = {"random_forest", "linear_regression"}


def load_model(model_name: str = "random_forest"):
    """Load a trained model from disk."""
    if model_name not in ALLOWED_MODELS:
        raise ValueError(
            f"Invalid model '{model_name}'. Allowed models: {sorted(ALLOWED_MODELS)}"
        )
    model_path = os.path.join(config.paths.model_dir, f"{model_name}.joblib")
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model '{model_name}' not found at {model_path}. Run train.py first."
        )
    return joblib.load(model_path)


def predict(input_data: pd.DataFrame, model_name: str = "random_forest") -> list[float]:
    """
    Run predictions on new data using a trained model.

    Args:
        input_data: DataFrame with the same features used during training.
        model_name: Which model to load ('random_forest' or 'linear_regression').

    Returns:
        List of validated predicted prices.
    """
    model = load_model(model_name)
    raw_predictions = model.predict(input_data)

    validated = []
    for i, price in enumerate(raw_predictions):
        if PredictionGuardrail.validate_prediction(price):
            validated.append(price)
        else:
            print(f"Row {i}: prediction ${price:,.2f} failed guardrail — setting to NaN.")
            validated.append(float("nan"))

    return validated


if __name__ == "__main__":
    # Demo: load test data and predict
    x_test_path = os.path.join(config.paths.data_dir, "X_test.csv")
    if not os.path.exists(x_test_path):
        print("No test data found. Run train.py first to generate X_test.csv.")
    else:
        X_test = pd.read_csv(x_test_path)
        predictions = predict(X_test)
        print(f"Generated {len(predictions)} predictions.")
        print(f"Sample (first 5): {[f'${p:,.2f}' for p in predictions[:5]]}")

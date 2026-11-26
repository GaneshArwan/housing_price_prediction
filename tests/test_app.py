import os
import re
import pytest
from streamlit.testing.v1 import AppTest
from src.config import config
from src.train import train_models


@pytest.fixture(scope="module", autouse=True)
def _ensure_trained_artifacts():
    """Train models if artifacts are missing (e.g., fresh clone or CI)."""
    rf_path = os.path.join(config.paths.model_dir, "random_forest.joblib")
    lr_path = os.path.join(config.paths.model_dir, "linear_regression.joblib")
    x_test_path = os.path.join(config.paths.data_dir, "X_test.csv")
    if not (os.path.exists(rf_path) and os.path.exists(lr_path) and os.path.exists(x_test_path)):
        train_models()


def _parse_price(success_text: str) -> float:
    match = re.search(r"\$([\d,]+)", success_text)
    assert match is not None, f"Could not parse price from: {success_text}"
    return float(match.group(1).replace(",", ""))


def test_app_loads_without_errors():
    """Test that the Streamlit app boots up completely without Python exceptions."""
    at = AppTest.from_file("../app.py", default_timeout=15).run()
    assert not at.exception, f"App crashed on startup: {at.exception}"


def test_predict_button_flow():
    """Test the core UX flow: filling inputs and clicking 'Predict Price'."""
    at = AppTest.from_file("../app.py", default_timeout=15).run()

    assert at.title[0].value == "🏠 Predict Housing Price"

    predict_btn = [btn for btn in at.button if btn.label == "Predict Price"]
    assert len(predict_btn) == 1, "Predict button not found in UI"

    predict_btn[0].click().run()

    assert not at.exception, f"App crashed during prediction: {at.exception}"
    assert len(at.success) > 0, "No success message shown after prediction"
    assert "Estimated Sale Price" in at.success[0].value


def test_predict_linear_regression():
    """Test switching the model to Linear Regression and making a prediction."""
    at = AppTest.from_file("../app.py", default_timeout=15).run()

    model_radio = [r for r in at.radio if r.label == "Select ML Model"][0]
    model_radio.set_value("Linear Regression (Baseline)").run()

    assert not at.exception

    predict_btn = [btn for btn in at.button if btn.label == "Predict Price"]
    predict_btn[0].click().run()

    assert not at.exception
    assert len(at.success) > 0, "Linear regression prediction failed"


def test_categorical_feature_affects_prediction():
    """Regression test: changing Overall Quality must change the predicted price."""
    at = AppTest.from_file("../app.py", default_timeout=15).run()

    model_radio = [r for r in at.radio if r.label == "Select ML Model"][0]
    model_radio.set_value("Linear Regression (Baseline)").run()

    qual_box = [s for s in at.selectbox if s.label == "Overall Quality"][0]
    qual_box.set_value("Very_Poor").run()
    [btn for btn in at.button if btn.label == "Predict Price"][0].click().run()
    assert not at.exception
    poor_price = _parse_price(at.success[0].value)

    qual_box = [s for s in at.selectbox if s.label == "Overall Quality"][0]
    qual_box.set_value("Very_Excellent").run()
    [btn for btn in at.button if btn.label == "Predict Price"][0].click().run()
    assert not at.exception
    excellent_price = _parse_price(at.success[0].value)

    assert excellent_price > poor_price, (
        f"Expected Very_Excellent ({excellent_price}) > Very_Poor ({poor_price})"
    )


def test_explore_data_filtering():
    """Test the multiselect filtering feature on the Explore Data page."""
    at = AppTest.from_file("../app.py", default_timeout=15).run()

    nav_radio = [r for r in at.radio if r.label == "Navigate"][0]
    nav_radio.set_value("Explore Data").run()
    assert not at.exception

    assert len(at.multiselect) == 1

    neighborhood_filter = at.multiselect[0]
    neighborhood_filter.select("North_Ames").run()

    assert not at.exception
    showing_metric = [m for m in at.metric if m.label == "Showing Homes"][0]
    assert int(showing_metric.value.replace(",", "")) < 2930

    download_btns = [btn for btn in at.download_button if "Download Filtered Data as CSV" in btn.label]
    assert len(download_btns) == 1


def test_model_performance_page():
    """Test navigating to the Model Performance page and verifying images load."""
    at = AppTest.from_file("../app.py", default_timeout=15).run()

    nav_radio = [r for r in at.radio if r.label == "Navigate"][0]
    nav_radio.set_value("Model Performance").run()

    assert not at.exception
    assert not at.warning, f"Expected all plots present, got warnings: {at.warning}"
    assert at.title[0].value == "📈 Model Performance"

import os
import pytest
from src.config import config
from src.generate_pdf_report import compute_report_metrics, generate_report
from src.train import train_models


@pytest.fixture(scope="module", autouse=True)
def _ensure_artifacts():
    rf_path = os.path.join(config.paths.model_dir, "random_forest.joblib")
    lr_path = os.path.join(config.paths.model_dir, "linear_regression.joblib")
    x_test_path = os.path.join(config.paths.data_dir, "X_test.csv")
    if not (os.path.exists(rf_path) and os.path.exists(lr_path) and os.path.exists(x_test_path)):
        train_models()


def test_reports_dir_exists():
    """The reports directory should be created on import."""
    assert os.path.isdir(config.paths.reports_dir)


def test_model_dir_exists():
    """The models directory should be created on import."""
    assert os.path.isdir(config.paths.model_dir)


def test_data_dir_exists():
    """The data directory should be created on import."""
    assert os.path.isdir(config.paths.data_dir)


def test_compute_report_metrics_accuracy():
    """Verify report metrics match live data and holdout split invariants."""
    m = compute_report_metrics()
    assert m["raw_rows"] == 2930
    assert m["raw_cols"] == 81
    assert m["n_test"] == 586
    assert m["n_standard_test"] + m["n_luxury_test"] == 586
    assert 0.8 < m["lr"]["r2"] < 1.0
    assert 0.8 < m["rf"]["r2"] < 1.0
    assert m["rf"]["rmse"] < m["lr"]["rmse"]
    assert m["top_corrs"][0][0] == "Gr_Liv_Area"
    assert m["top_rf_imp"][0][0] == "Garage_Cars"


def test_generate_report_creates_nonempty_pdf():
    """Verify IEEE PDF generation succeeds and writes a non-empty PDF."""
    generate_report()
    pdf_path = os.path.join(config.paths.reports_dir, "Housing_Price_Prediction_Report.pdf")
    assert os.path.isfile(pdf_path)
    assert os.path.getsize(pdf_path) > 10_000

import pytest
import pandas as pd
import numpy as np
import os
from unittest.mock import patch
from src.data_loader import fetch_ames_housing_data, preprocess_data


def _make_sample_df():
    """Create a small synthetic DataFrame that mirrors the Ames schema."""
    np.random.seed(42)
    return pd.DataFrame({
        "Id": [1, 2, 3],
        "PID": [100, 200, 300],
        "Lot_Area": [8000, 9500, 11000],
        "Overall_Qual": [5, 7, 6],
        "Gr_Liv_Area": [1200, 1500, 1800],
        "Alley": [np.nan, np.nan, np.nan],
        "PoolQC": [np.nan, np.nan, np.nan],
        "Fence": [np.nan, "GdPrv", np.nan],
        "MiscFeature": [np.nan, np.nan, np.nan],
        "MS_Zoning": ["RL", "RM", "RL"],
        "Sale_Price": [200000, 150000, 250000],
    })


def test_fetch_data_local():
    """If the CSV exists locally, fetch should return a non-empty DataFrame."""
    data_path = os.path.join("data", "ames_housing.csv")
    if not os.path.exists(data_path):
        pytest.skip("Dataset not available locally; skipping network-dependent test.")
    df = fetch_ames_housing_data()
    assert not df.empty, "Dataset should not be empty."
    assert "Sale_Price" in df.columns or "SalePrice" in df.columns


def test_preprocess_drops_columns():
    df = _make_sample_df()
    processed = preprocess_data(df)
    for col in ["Id", "PID", "Alley", "PoolQC", "Fence", "MiscFeature"]:
        assert col not in processed.columns, f"{col} should be dropped."


def test_preprocess_renames_target():
    df = _make_sample_df()
    processed = preprocess_data(df)
    assert "SalePrice" in processed.columns, "Sale_Price should be renamed to SalePrice."


def test_preprocess_no_nans():
    df = _make_sample_df()
    processed = preprocess_data(df)
    assert processed.isnull().sum().sum() == 0, "No missing values after preprocessing."

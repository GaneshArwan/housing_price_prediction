import pytest
import pandas as pd
from src.data_loader import fetch_ames_housing_data, preprocess_data

def test_fetch_data():
    df = fetch_ames_housing_data()
    assert not df.empty, "Dataset should not be empty."
    assert 'Sale_Price' in df.columns or 'SalePrice' in df.columns, "Target variable should be present."

def test_preprocess_data():
    df = fetch_ames_housing_data()
    processed_df = preprocess_data(df)
    
    # Check that dropped columns are gone
    dropped_cols = ["Id", "PID", "Alley", "PoolQC", "Fence", "MiscFeature"]
    for col in dropped_cols:
        assert col not in processed_df.columns
        
    # Check that SalePrice is present and correct name
    assert 'SalePrice' in processed_df.columns
    
    # Check for NaNs
    assert processed_df.isnull().sum().sum() == 0, "There should be no missing values after preprocessing."

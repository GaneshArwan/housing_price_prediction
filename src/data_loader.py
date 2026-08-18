import pandas as pd
import os
import urllib.request
from src.config import config

def fetch_ames_housing_data():
    """
    Downloads the Ames Housing Dataset from a stable source.
    """
    data_path = os.path.join(config.paths.data_dir, "ames_housing.csv")
    
    if not os.path.exists(data_path):
        print("Downloading Ames Housing Dataset...")
        # Using a reliable github mirror since OpenML was giving 503
        url = "https://raw.githubusercontent.com/amueller/mglearn_course/master/data/ames_housing.csv"
        try:
            urllib.request.urlretrieve(url, data_path)  # nosec B310
            print("Download complete.")
        except Exception as e:
            print(f"Failed to download from primary mirror: {e}")
            print("Attempting to read directly via pandas from OpenML URL as fallback...")
            try:
                # OpenML URL fallback
                openml_url = "https://www.openml.org/data/get_csv/20649135/phpOJxEM9" 
                df = pd.read_csv(openml_url)
                df.to_csv(data_path, index=False)
            except Exception as e2:
                print(f"Failed fallback: {e2}")
                raise e2
    else:
        print("Dataset already exists locally.")
        
    return pd.read_csv(data_path)

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess the Ames Housing dataset for modeling.
    - Handles missing values
    - Encodes categorical variables
    """
    # Create a copy
    processed = df.copy()
    
    # Drop columns with too many missing values or identifiers
    cols_to_drop = ["Id", "PID", "Alley", "PoolQC", "Fence", "MiscFeature"]
    processed.drop(columns=[c for c in cols_to_drop if c in processed.columns], inplace=True)
    
    # Target variable is typically Sale_Price
    if 'Sale_Price' in processed.columns:
        processed.rename(columns={'Sale_Price': 'SalePrice'}, inplace=True)
    elif 'SalePrice' not in processed.columns:
        raise ValueError("Target variable 'SalePrice' not found in dataset.")
        
    # Fill numeric NaNs with median
    numeric_cols = processed.select_dtypes(include=['number']).columns
    processed[numeric_cols] = processed[numeric_cols].fillna(processed[numeric_cols].median())
    
    # Fill categorical NaNs with 'Unknown'
    cat_cols = processed.select_dtypes(exclude=['number']).columns
    processed[cat_cols] = processed[cat_cols].fillna('Unknown')
    
    # One-hot encode categorical variables
    processed = pd.get_dummies(processed, columns=cat_cols, drop_first=True)
    
    return processed

if __name__ == "__main__":
    df = fetch_ames_housing_data()
    print(f"Dataset shape: {df.shape}")
    processed_df = preprocess_data(df)
    print(f"Processed shape: {processed_df.shape}")

import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error
import mlflow
import mlflow.sklearn
from src.data_loader import fetch_ames_housing_data, preprocess_data
from src.config import config
from src.format_utils import format_rmse

def train_models():
    """
    Train Linear Regression and Random Forest models on the dataset.
    """
    print("Fetching and preprocessing data...")
    df = fetch_ames_housing_data()
    processed_df = preprocess_data(df)
    
    # Target variable
    X = processed_df.drop(columns=['SalePrice'])
    y = processed_df['SalePrice']
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config.model.test_size, random_state=config.model.random_state
    )
    
    mlflow.set_experiment("Housing_Price_Prediction")
    with mlflow.start_run():
        mlflow.log_param("random_state", config.model.random_state)
        mlflow.log_param("test_size", config.model.test_size)
        mlflow.log_param("rf_n_estimators", config.model.rf_n_estimators)
        mlflow.log_param("rf_max_depth", config.model.rf_max_depth)
        
        print("Training Linear Regression...")
        lr = LinearRegression()
        lr.fit(X_train, y_train)
        lr_preds = lr.predict(X_test)
        lr_rmse = root_mean_squared_error(y_test, lr_preds)
        print(f"Linear Regression {format_rmse(lr_rmse)}")
        mlflow.log_metric("lr_rmse", lr_rmse)
    
        print("Training Random Forest...")
        rf = RandomForestRegressor(
            n_estimators=config.model.rf_n_estimators,
            max_depth=config.model.rf_max_depth,
            random_state=config.model.random_state
        )
        rf.fit(X_train, y_train)
        rf_preds = rf.predict(X_test)
        rf_rmse = root_mean_squared_error(y_test, rf_preds)
        print(f"Random Forest {format_rmse(rf_rmse)}")
        mlflow.log_metric("rf_rmse", rf_rmse)
        
        # Save models
        joblib.dump(lr, os.path.join(config.paths.model_dir, "linear_regression.joblib"))
        joblib.dump(rf, os.path.join(config.paths.model_dir, "random_forest.joblib"))
        
        # Log to MLflow
        mlflow.sklearn.log_model(lr, "linear_regression_model")
        mlflow.sklearn.log_model(rf, "random_forest_model")
        
        # Save test set for evaluation script
        X_test.to_csv(os.path.join(config.paths.data_dir, "X_test.csv"), index=False)
        y_test.to_csv(os.path.join(config.paths.data_dir, "y_test.csv"), index=False)
        print("Models and test data saved successfully.")

if __name__ == "__main__":
    train_models()

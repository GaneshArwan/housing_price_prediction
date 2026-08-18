import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from src.config import config
from src.data_loader import fetch_ames_housing_data

def evaluate_and_plot():
    """
    Generate evaluation plots: feature importance, correlation heatmap, residual plot.
    """
    print("Loading test data and models...")
    try:
        X_test = pd.read_csv(os.path.join(config.paths.data_dir, "X_test.csv"))
        y_test = pd.read_csv(os.path.join(config.paths.data_dir, "y_test.csv")).squeeze()
        rf = joblib.load(os.path.join(config.paths.model_dir, "random_forest.joblib"))
    except FileNotFoundError:
        print("Required files not found. Run train.py first.")
        return

    # 1. Feature Importance
    print("Generating Feature Importance Plot...")
    importances = rf.feature_importances_
    # Get top 20 features
    top_indices = importances.argsort()[-20:][::-1]
    top_features = X_test.columns[top_indices]
    top_importances = importances[top_indices]

    plt.figure(figsize=(10, 8))
    sns.barplot(x=top_importances, y=top_features)
    plt.title("Top 20 Feature Importances (Random Forest)")
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.savefig(os.path.join(config.paths.reports_dir, "feature_importance.png"))
    plt.close()

    # 2. Residual Plot
    print("Generating Residual Plot...")
    rf_preds = rf.predict(X_test)
    residuals = y_test - rf_preds

    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=rf_preds, y=residuals, alpha=0.5)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.title("Residual Plot (Random Forest)")
    plt.xlabel("Predicted Sale Price")
    plt.ylabel("Residuals")
    plt.tight_layout()
    plt.savefig(os.path.join(config.paths.reports_dir, "residual_plot.png"))
    plt.close()

    # 3. Correlation Heatmap
    print("Generating Correlation Heatmap...")
    df = fetch_ames_housing_data()
    # Select numeric columns
    numeric_df = df.select_dtypes(include=['number'])
    # Correlation with SalePrice
    if 'SalePrice' in numeric_df.columns:
        correlations = numeric_df.corr()['SalePrice'].sort_values(ascending=False)
        top_corr_features = correlations.index[1:11] # Top 10 excluding SalePrice
        
        plt.figure(figsize=(10, 8))
        top_corr_matrix = numeric_df[['SalePrice'] + list(top_corr_features)].corr()
        sns.heatmap(top_corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
        plt.title("Correlation Heatmap (Top Features vs SalePrice)")
        plt.tight_layout()
        plt.savefig(os.path.join(config.paths.reports_dir, "correlation_heatmap.png"))
        plt.close()
    
    print(f"All plots saved to {config.paths.reports_dir}/")

if __name__ == "__main__":
    evaluate_and_plot()

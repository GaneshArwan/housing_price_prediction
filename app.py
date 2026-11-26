import streamlit as st
import pandas as pd
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
from src.config import config
from src.data_loader import fetch_ames_housing_data, preprocess_data
from src.guardrails import PredictionGuardrail

st.set_page_config(
    page_title="Housing Price Predictor",
    page_icon="🏠",
    layout="wide",
)

ALLOWED_MODELS = {"random_forest", "linear_regression"}


# --- Load model and data ---
@st.cache_resource
def load_model(name="random_forest"):
    if name not in ALLOWED_MODELS:
        raise ValueError(f"Invalid model '{name}'. Choose from: {sorted(ALLOWED_MODELS)}")
    path = os.path.join(config.paths.model_dir, f"{name}.joblib")
    if not os.path.exists(path):
        return None
    return joblib.load(path)

@st.cache_data
def load_dataset():
    df = fetch_ames_housing_data()
    return df

@st.cache_data
def get_training_columns():
    """Get the exact column order used during training."""
    x_test_path = os.path.join(config.paths.data_dir, "X_test.csv")
    if os.path.exists(x_test_path):
        return pd.read_csv(x_test_path, nrows=0).columns.tolist()
    return None

# --- Sidebar ---
st.sidebar.title("🏠 Housing Price Predictor")
st.sidebar.markdown("Predict Ames, Iowa home prices using a trained Random Forest model.")
st.sidebar.markdown("---")

page = st.sidebar.radio("Navigate", ["Predict", "Explore Data", "Model Performance"])

# --- Predict Page ---
if page == "Predict":
    st.title("🏠 Predict Housing Price")
    st.markdown("Adjust the home features below to get an estimated sale price.")

    model_choice = st.radio("Select ML Model", ["Random Forest (Recommended)", "Linear Regression (Baseline)"], horizontal=True)
    model_name = "random_forest" if "Random Forest" in model_choice else "linear_regression"
    
    model = load_model(model_name)
    raw_df = load_dataset()
    training_cols = get_training_columns()

    if model is None or training_cols is None:
        st.error("Model or training data not found. Run `python -m src.train` first.")
        st.stop()

    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("Quality & Age")
        overall_qual = st.selectbox("Overall Quality", [
            "Very_Poor", "Poor", "Fair", "Below_Average", "Average",
            "Above_Average", "Good", "Very_Good", "Excellent", "Very_Excellent"
        ], index=4, help="Overall material and finish of the house")
        year_built = st.slider("Year Built", 1872, 2010, 1990)
        year_remod = st.slider("Year Remodeled", 1950, 2010, 2000, help="Same as construction date if no remodeling or additions")
        overall_cond = st.selectbox("Overall Condition", [
            "Very_Poor", "Poor", "Fair", "Below_Average", "Average",
            "Above_Average", "Good", "Very_Good", "Excellent", "Very_Excellent"
        ], index=4, help="Overall condition rating of the house")

    with col2:
        st.subheader("Size & Area")
        gr_liv_area = st.slider("Above Ground Living Area (sqft)", 300, 5700, 1500, help="Does not include basement area")
        total_bsmt_sf = st.slider("Total Basement Area (sqft)", 0, 6200, 1000)
        lot_area = st.slider("Lot Area (sqft)", 1300, 50000, 10000)
        first_flr_sf = st.slider("First Floor Area (sqft)", 300, 4700, 1100)
        garage_area = st.slider("Garage Area (sqft)", 0, 1500, 480)

    with col3:
        st.subheader("Rooms & Features")
        full_bath = st.selectbox("Full Bathrooms", [0, 1, 2, 3, 4], index=1)
        bedroom = st.selectbox("Bedrooms Above Ground", [0, 1, 2, 3, 4, 5, 6], index=2, help="Does not include basement bedrooms")
        garage_cars = st.selectbox("Garage Car Capacity", [0, 1, 2, 3, 4, 5], index=2)
        fireplaces = st.selectbox("Fireplaces", [0, 1, 2, 3, 4], index=0)
        totrms = st.selectbox("Total Rooms Above Ground", list(range(2, 16)), index=4, help="Does not include bathrooms")
        neighborhood = st.selectbox("Neighborhood", sorted(raw_df["Neighborhood"].unique().tolist()))

    st.markdown("---")

    if st.button("Predict Price", type="primary", use_container_width=True):
        # Build a single-row DataFrame matching the raw dataset schema
        input_row = {}
        for col in raw_df.columns:
            if col == "Sale_Price" or col == "SalePrice":
                continue
            if raw_df[col].dtype == "object":
                input_row[col] = raw_df[col].mode()[0]
            else:
                input_row[col] = raw_df[col].median()

        # Override with user selections
        input_row["Overall_Qual"] = overall_qual
        input_row["Overall_Cond"] = overall_cond
        input_row["Year_Built"] = year_built
        input_row["Year_Remod_Add"] = year_remod
        input_row["Gr_Liv_Area"] = gr_liv_area
        input_row["Total_Bsmt_SF"] = total_bsmt_sf
        input_row["Lot_Area"] = lot_area
        input_row["First_Flr_SF"] = first_flr_sf
        input_row["Garage_Area"] = garage_area
        input_row["Garage_Cars"] = garage_cars
        input_row["Full_Bath"] = full_bath
        input_row["Bedroom_AbvGr"] = bedroom
        input_row["Fireplaces"] = fireplaces
        input_row["TotRms_AbvGrd"] = totrms
        input_row["Neighborhood"] = neighborhood

        with st.spinner("Analyzing market data and calculating estimate..."):
            input_df = pd.DataFrame([input_row])

            # Preserve single-row categorical levels before aligning to training columns
            input_df = preprocess_data(input_df, drop_first=False)
            input_df = input_df.reindex(columns=training_cols, fill_value=0)

            prediction = model.predict(input_df)[0]

        if PredictionGuardrail.validate_prediction(prediction):
            st.success(f"### Estimated Sale Price: ${prediction:,.0f}")
        else:
            st.warning(f"Prediction of ${prediction:,.0f} failed guardrail validation. The inputs may be outside the model's training range.")

# --- Explore Data Page ---
elif page == "Explore Data":
    st.title("📊 Explore the Ames Housing Dataset")
    raw_df = load_dataset()

    st.subheader("Dataset Overview")
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Homes", f"{len(raw_df):,}")
    col2.metric("Features", f"{raw_df.shape[1]}")
    col3.metric("Median Price", f"${raw_df['Sale_Price'].median():,.0f}")

    st.subheader("Sale Price Distribution")
    fig, ax = plt.subplots(figsize=(10, 4))
    sns.histplot(raw_df["Sale_Price"], bins=50, kde=True, ax=ax)
    ax.set_xlabel("Sale Price ($)")
    ax.set_ylabel("Count")
    st.pyplot(fig)
    plt.close()

    st.subheader("Price by Neighborhood")
    fig2, ax2 = plt.subplots(figsize=(12, 5))
    neighborhood_median = raw_df.groupby("Neighborhood")["Sale_Price"].median().sort_values(ascending=False)
    sns.barplot(x=neighborhood_median.index, y=neighborhood_median.values, ax=ax2)
    ax2.set_xticks(range(len(neighborhood_median.index)))
    ax2.set_xticklabels(neighborhood_median.index, rotation=45, ha="right", fontsize=7)
    ax2.set_ylabel("Median Sale Price ($)")
    plt.tight_layout()
    st.pyplot(fig2)
    plt.close()

    st.subheader("Filter & Export Data")
    all_neighborhoods = sorted(raw_df["Neighborhood"].unique().tolist())
    selected_neighborhoods = st.multiselect("Filter by Neighborhood(s)", all_neighborhoods, default=[])
    
    if selected_neighborhoods:
        filtered_df = raw_df[raw_df["Neighborhood"].isin(selected_neighborhoods)]
    else:
        filtered_df = raw_df

    st.metric("Showing Homes", f"{len(filtered_df):,}")
    st.dataframe(filtered_df.head(100), use_container_width=True)
    
    st.download_button(
        label="📥 Download Filtered Data as CSV",
        data=filtered_df.to_csv(index=False).encode("utf-8"),
        file_name="ames_housing_filtered.csv",
        mime="text/csv",
        help="Download the currently filtered dataset for offline analysis."
    )

# --- Model Performance Page ---
elif page == "Model Performance":
    st.title("📈 Model Performance")

    img_dir = config.paths.reports_dir

    col1, col2 = st.columns(2)

    heatmap_path = os.path.join(img_dir, "correlation_heatmap.png")
    importance_path = os.path.join(img_dir, "feature_importance.png")
    residual_path = os.path.join(img_dir, "residual_plot.png")

    with col1:
        if os.path.exists(heatmap_path):
            st.image(heatmap_path, caption="Correlation Heatmap", use_container_width=True)
        else:
            st.warning("Correlation heatmap not found. Run `python -m src.evaluate` first.")

    with col2:
        if os.path.exists(importance_path):
            st.image(importance_path, caption="Feature Importance (MDI)", use_container_width=True)
        else:
            st.warning("Feature importance plot not found. Run `python -m src.evaluate` first.")

    if os.path.exists(residual_path):
        st.image(residual_path, caption="Residual Plot", use_container_width=True)
    else:
        st.warning("Residual plot not found. Run `python -m src.evaluate` first.")

    report_path = os.path.join(img_dir, "Housing_Price_Prediction_Report.pdf")
    if os.path.exists(report_path):
        with open(report_path, "rb") as f:
            st.download_button("Download IEEE PDF Report", f, file_name="Housing_Price_Prediction_Report.pdf", mime="application/pdf")

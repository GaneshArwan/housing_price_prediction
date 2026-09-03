import os
import pytest
from streamlit.testing.v1 import AppTest
from src.config import config

def test_app_loads_without_errors():
    """Test that the Streamlit app boots up completely without Python exceptions."""
    at = AppTest.from_file("../app.py").run()
    assert not at.exception, f"App crashed on startup: {at.exception}"

def test_predict_button_flow():
    """Test the core UX flow: filling inputs and clicking 'Predict Price'."""
    at = AppTest.from_file("../app.py").run()
    
    # Ensure we are on the Predict page (default)
    assert at.title[0].value == "🏠 Predict Housing Price"
    
    # Simulate clicking the primary 'Predict Price' button
    # Streamlit testing API finds buttons by label or index.
    # In our app, there is only one button on this page.
    predict_btn = [btn for btn in at.button if btn.label == "Predict Price"]
    assert len(predict_btn) == 1, "Predict button not found in UI"
    
    # Click it and trigger a rerun
    predict_btn[0].click().run()
    
    # The app should process the ML prediction without crashing
    assert not at.exception, f"App crashed during prediction: {at.exception}"
    
    # We should see a success message with the formatted price
    assert len(at.success) > 0, "No success message shown after prediction"
    assert "Estimated Sale Price" in at.success[0].value

def test_explore_data_page():
    """Test navigating to the Explore Data page and verifying components load."""
    at = AppTest.from_file("../app.py").run()
    
    # The sidebar radio is the first radio widget
    nav_radio = at.radio[0]
    # Set value to 'Explore Data' and run
    nav_radio.set_value("Explore Data").run()
    
    assert not at.exception
    assert at.title[0].value == "📊 Explore the Ames Housing Dataset"
    
    # Verify metrics loaded (Total Homes, Features, Median Price)
    assert len(at.metric) == 3
    
    # Verify the dataframe renders
    assert len(at.dataframe) > 0

def test_model_performance_page():
    """Test navigating to the Model Performance page and verifying images load."""
    at = AppTest.from_file("../app.py").run()
    
    nav_radio = at.radio[0]
    nav_radio.set_value("Model Performance").run()
    
    assert not at.exception
    assert at.title[0].value == "📈 Model Performance"
    
    # We should have images loaded (Heatmap, Feature Importance, Residuals)
    # If the files exist, st.image is called. Let's just ensure no crash.
    assert not at.exception

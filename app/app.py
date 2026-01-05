import streamlit as st
import pandas as pd
import numpy as np
import joblib
import sys
from datetime import datetime, time
from pathlib import Path


sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils import AIRLINES, SOURCES, DESTINATIONS, ADDITIONAL_INFO, DAY_NAMES

# =============================================================================
# ROUTE-BASED DURATION ESTIMATION
# =============================================================================

ROUTE_DURATIONS = {
    # From Banglore
    ('Banglore', 'Cochin'): {0: 60, 1: 300, 2: 600, 3: 900, 4: 1200},
    ('Banglore', 'Delhi'): {0: 165, 1: 480, 2: 780, 3: 1100, 4: 1400},
    ('Banglore', 'Hyderabad'): {0: 75, 1: 320, 2: 620, 3: 920, 4: 1220},
    ('Banglore', 'Kolkata'): {0: 165, 1: 500, 2: 800, 3: 1100, 4: 1400},
    ('Banglore', 'New Delhi'): {0: 170, 1: 490, 2: 790, 3: 1100, 4: 1400},
    
    # From Chennai
    ('Chennai', 'Cochin'): {0: 75, 1: 340, 2: 640, 3: 940, 4: 1240},
    ('Chennai', 'Delhi'): {0: 170, 1: 510, 2: 810, 3: 1110, 4: 1410},
    ('Chennai', 'Hyderabad'): {0: 70, 1: 310, 2: 610, 3: 910, 4: 1210},
    ('Chennai', 'Kolkata'): {0: 140, 1: 460, 2: 760, 3: 1060, 4: 1360},
    ('Chennai', 'New Delhi'): {0: 175, 1: 520, 2: 820, 3: 1120, 4: 1420},
    
    # From Delhi
    ('Delhi', 'Banglore'): {0: 165, 1: 480, 2: 780, 3: 1080, 4: 1380},
    ('Delhi', 'Cochin'): {0: 195, 1: 560, 2: 860, 3: 1160, 4: 1460},
    ('Delhi', 'Hyderabad'): {0: 130, 1: 420, 2: 720, 3: 1020, 4: 1320},
    ('Delhi', 'Kolkata'): {0: 135, 1: 440, 2: 740, 3: 1040, 4: 1340},
    ('Delhi', 'New Delhi'): {0: 30, 1: 120, 2: 240, 3: 360, 4: 480},  
    
    # From Kolkata
    ('Kolkata', 'Banglore'): {0: 165, 1: 500, 2: 800, 3: 1100, 4: 1400},
    ('Kolkata', 'Cochin'): {0: 195, 1: 580, 2: 880, 3: 1180, 4: 1480},
    ('Kolkata', 'Delhi'): {0: 135, 1: 440, 2: 740, 3: 1040, 4: 1340},
    ('Kolkata', 'Hyderabad'): {0: 140, 1: 460, 2: 760, 3: 1060, 4: 1360},
    ('Kolkata', 'New Delhi'): {0: 140, 1: 450, 2: 750, 3: 1050, 4: 1350},
    
    # From Mumbai
    ('Mumbai', 'Banglore'): {0: 95, 1: 380, 2: 680, 3: 980, 4: 1280},
    ('Mumbai', 'Cochin'): {0: 115, 1: 420, 2: 720, 3: 1020, 4: 1320},
    ('Mumbai', 'Delhi'): {0: 130, 1: 420, 2: 720, 3: 1020, 4: 1320},
    ('Mumbai', 'Hyderabad'): {0: 85, 1: 360, 2: 660, 3: 960, 4: 1260},
    ('Mumbai', 'Kolkata'): {0: 155, 1: 480, 2: 780, 3: 1080, 4: 1380},
    ('Mumbai', 'New Delhi'): {0: 135, 1: 430, 2: 730, 3: 1030, 4: 1330},
}

# Fallback durations if route not found
DEFAULT_DURATIONS = {0: 120, 1: 480, 2: 780, 3: 1080, 4: 1380}


def get_estimated_duration(source: str, destination: str, stops: int) -> int:
    """
    Get estimated flight duration based on route and number of stops.
    
    Parameters:
    -----------
    source : str - Departure city
    destination : str - Arrival city
    stops : int - Number of stops (0-4)
    
    Returns:
    --------
    int : Estimated duration in minutes
    """
    route = (source, destination)
    
    if route in ROUTE_DURATIONS:
        return ROUTE_DURATIONS[route].get(stops, DEFAULT_DURATIONS[stops])
    else:
        return DEFAULT_DURATIONS[stops]


def format_duration(minutes: int) -> str:
    """Format duration in hours and minutes."""
    hours = minutes // 60
    mins = minutes % 60
    if hours > 0 and mins > 0:
        return f"{hours}h {mins}m"
    elif hours > 0:
        return f"{hours}h"
    else:
        return f"{mins}m"


# =============================================================================
# STREAMLIT APP
# =============================================================================

# Page config
st.set_page_config(
    page_title="Flight Price Predictor",
    page_icon="✈️",
    layout="centered"
)

# Load model
@st.cache_resource
def load_artifacts():
    """Load the trained pipeline (preprocessor + model)."""
    base_path = Path(__file__).parent.parent
    pipeline_path = base_path / "models" / "flight_price_pipeline.joblib"
    
    pipeline = joblib.load(pipeline_path)
    return pipeline

try:
    pipeline = load_artifacts()
    model_loaded = True
except Exception as e:
    model_loaded = False
    st.error(f"Error loading model: {e}")

# Main page
st.title("✈️ Flight Price Predictor")
st.markdown("Predict flight prices in India based on route, airline, and timing.")
st.markdown("---")

# Input form
col1, col2 = st.columns(2)

with col1:
    st.subheader("🛫 Flight Details")
    airline = st.selectbox("Airline", options=AIRLINES)
    source = st.selectbox("Source", options=SOURCES)
    destination = st.selectbox("Destination", options=DESTINATIONS)
    stops = st.selectbox(
        "Number of Stops", options=[0, 1, 2, 3, 4],
        format_func=lambda x: 'Non-stop' if x == 0 else f'{x} stop{"s" if x > 1 else ""}'
    )

with col2:
    st.subheader("📅 Journey Details")
    journey_date = st.date_input("Journey Date", value=datetime.now())
    dep_time = st.time_input("Departure Time", value=time(10, 0))
    additional_info = st.selectbox("Additional Info", options=ADDITIONAL_INFO)

# Calculate route-based duration
duration_minutes = get_estimated_duration(source, destination, stops)
duration_formatted = format_duration(duration_minutes)

# Show estimated duration
st.markdown("---")
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("🛣️ Route", f"{source} → {destination}")
with col2:
    st.metric("🛑 Stops", "Non-stop" if stops == 0 else f"{stops} stop{'s' if stops > 1 else ''}")
with col3:
    st.metric("⏱️ Est. Duration", duration_formatted)

# Derived features
journey_day = journey_date.day
journey_month = journey_date.month
journey_day_of_week = journey_date.weekday()
dep_hour = dep_time.hour
dep_minute = dep_time.minute

st.markdown("---")

# Validation
if source == destination:
    st.error("❌ Source and Destination cannot be the same!")

# Predict button
if st.button("🔮 Predict Price", type="primary", use_container_width=True):
    if source == destination:
        st.error("❌ Source and Destination cannot be the same!")
    elif not model_loaded:
        st.error("❌ Model not loaded.")
    else:
        input_data = pd.DataFrame({
            'Airline': [airline], 'Source': [source], 'Destination': [destination],
            'Additional_Info': [additional_info], 'Duration_Minutes': [duration_minutes],
            'Stops': [stops], 'Journey_Day': [journey_day], 'Journey_Month': [journey_month],
            'Journey_Day_of_Week': [journey_day_of_week], 'Dep_Hour': [dep_hour],
            'Dep_Minute': [dep_minute]
        })
        
        try:
            prediction = max(0, pipeline.predict(input_data)[0])
            
            st.success("✅ Prediction Complete!")
            
            col1, col2, col3 = st.columns(3)
            col1.metric("💰 Predicted Price", f"₹{prediction:,.0f}")
            col2.metric("📊 Price Range", f"₹{max(0,prediction-659):,.0f} - ₹{prediction+659:,.0f}")
            col3.metric("🎯 Confidence", "91.7%")
            
            # Flight summary
            st.markdown("### 📋 Flight Summary")
            summary_data = {
                "Detail": ["Route", "Airline", "Stops", "Est. Duration", "Date", "Departure"],
                "Value": [
                    f"{source} → {destination}",
                    airline,
                    'Non-stop' if stops == 0 else f'{stops} stop{"s" if stops > 1 else ""}',
                    f"{duration_formatted} ({duration_minutes} min)",
                    f"{journey_date.strftime('%d %b %Y')} ({DAY_NAMES[journey_day_of_week]})",
                    dep_time.strftime('%H:%M')
                ]
            }
            st.table(pd.DataFrame(summary_data))
            st.caption("* Duration estimated based on route and stops")
            
        except Exception as e:
            st.error(f"❌ Prediction failed: {str(e)}")


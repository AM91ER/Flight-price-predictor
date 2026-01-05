# ✈️ Flight Price Predictor

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://flight-price-predictor.streamlit.app)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An end-to-end machine learning project that predicts domestic flight prices in India with **91.7% accuracy**. Built with Python, XGBoost, and deployed on Streamlit Cloud.

##  Model Performance

| Metric | Value |
|--------|-------|
| **Model** | XGBoost (Tuned) |
| **Test R²** | 0.9167 (91.7%) |
| **Mean Absolute Error** | ₹659 |
| **RMSE** | ₹1,163 |
| **MAPE** | 7.95% |

##  Live Demo

**[Try the App →](https://flight-price-predictor.streamlit.app)**

##  Key Findings

###  Counterintuitive Discovery
**Non-stop flights are CHEAPER than connecting flights!**
- Non-stop average: ₹4,999
- 1-stop average: ₹9,471 (+89% premium)
- *Reason*: More stops → longer duration → higher operating costs

###  Airline Pricing Tiers
| Tier | Airlines | Avg Price Range |
|------|----------|-----------------|
| Premium | Jet Airways, Vistara Premium | ₹10,000 - ₹13,000 |
| Mid-range | Air India, Vistara | ₹7,500 - ₹10,000 |
| Budget | SpiceJet, GoAir, IndiGo | ₹4,500 - ₹7,500 |

###  Best Times to Book
- **Cheapest day**: Friday
- **Most expensive**: Sunday (+10%)
- **Cheapest time**: Red-eye (0-4 AM)
- **Cheapest month**: April

##  Project Structure

```
flight-price-predictor/
├── streamlit_app.py        # Main Streamlit application
├── app/                    # Alternative Streamlit app location
├── src/
│   ├── __init__.py         # Package initialization
│   ├── data_pipeline.py    # Data loading, cleaning, feature engineering
│   ├── model.py            # Model definition and configuration
│   ├── train.py            # Training logic and evaluation
│   ├── inference.py        # Prediction on new data
│   └── utils.py            # Constants and helper functions
├── models/
│   └── flight_price_pipeline.joblib  # Complete ML pipeline (2.1 MB)
├── data/
│   ├── raw/                # Original dataset
│   └── processed/          # Cleaned dataset
├── Notebooks/              # Jupyter notebooks (Phase 1-6)
├── main.py                 # CLI entry point for training
├── requirements.txt        # Python dependencies
└── README.md
```

## 🛠️ Installation

### Prerequisites
- Python 3.9 or higher
- pip package manager

### Local Setup

1. **Clone the repository**
```bash
git clone https://github.com/AM91ER/flight-price-predictor.git
cd flight-price-predictor
```

2. **Create virtual environment (recommended)**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Run the Streamlit app**
```bash
streamlit run streamlit_app.py
```

The app will open at `http://localhost:8501`

##  Dataset

The dataset contains **10,683 flight records** from major Indian airlines (March-June 2019):

| Feature | Description |
|---------|-------------|
| Airline | 11 carriers (IndiGo, Jet Airways, Air India, etc.) |
| Source | 5 cities (Delhi, Mumbai, Bangalore, Chennai, Kolkata) |
| Destination | 6 cities (Cochin, Bangalore, Delhi, New Delhi, Hyderabad, Kolkata) |
| Duration | Flight duration |
| Total_Stops | Number of stops (0-4) |
| Date_of_Journey | Travel date |
| Dep_Time | Departure time |
| Price | Ticket price in INR (target) |

### Data Cleaning Pipeline

| Step | Records Removed | Remaining |
|------|-----------------|-----------|
| Original | — | 10,683 |
| Duplicates | 219 (2.1%) | 10,464 |
| Missing Values | 2 (0.02%) | 10,462 |
| Price Outliers (IQR) | 94 (0.9%) | 10,368 |
| Duration Outliers | 5 (0.05%) | 10,363 |
| **Final Dataset** | **320 (3.0%)** | **10,363 (97%)** |

##  Model Pipeline

### Feature Engineering
- **Duration**: `'2h 50m'` → `170` minutes
- **Stops**: `'non-stop'` → `0`, `'1 stop'` → `1`
- **Date**: Extracted day, month, day_of_week
- **Time**: Extracted hour, minute

### Preprocessing
- **Categorical**: OneHotEncoder (Airline, Source, Destination, Additional_Info)
- **Numerical**: StandardScaler (Duration_Minutes), Passthrough (others)
- **Final Features**: 30 (after encoding)

### Models Evaluated
| Model | Test R² | MAE |
|-------|---------|-----|
| **XGBoost (Tuned)** | **0.9167** | **₹659** |
| Random Forest | 0.9071 | ₹653 |
| Gradient Boosting | 0.8286 | ₹1,186 |
| Linear Regression | 0.7030 | ₹1,632 |

### Deployment Architecture
The model uses a **scikit-learn Pipeline** that combines preprocessing and prediction into a single, deployment-safe artifact. This eliminates common serialization issues and simplifies production deployment.

**Pipeline Structure:**
```
Pipeline([
  ('preprocessor', ColumnTransformer([...])),
  ('model', XGBRegressor([...]))
])
```

##  Usage

### Using the Inference Module

```python
from src.inference import FlightPricePredictor

# New pipeline approach (recommended)
predictor = FlightPricePredictor(
    pipeline_path='models/flight_price_pipeline.joblib'
)

# Legacy separate files approach (still supported)
predictor = FlightPricePredictor(
    model_path='models/best_model.pkl',
    preprocessor_path='artifacts/preprocessor.pkl'
)

result = predictor.predict(
    airline='IndiGo',
    source='Delhi',
    destination='Cochin',
    additional_info='No info',
    duration_minutes=180,
    stops=0,
    journey_day=15,
    journey_month=5,
    journey_day_of_week=2,
    dep_hour=10,
    dep_minute=30
)

print(f"Predicted Price: {result['price_formatted']}")
# Output: Predicted Price: ₹5,432
```

### Direct Pipeline Usage

```python
import joblib
import pandas as pd

# Load the complete pipeline
pipeline = joblib.load('models/flight_price_pipeline.joblib')

# Prepare input data
input_data = pd.DataFrame({
    'Airline': ['IndiGo'],
    'Source': ['Delhi'],
    'Destination': ['Cochin'],
    'Additional_Info': ['No info'],
    'Duration_Minutes': [180],
    'Stops': [0],
    'Journey_Day': [15],
    'Journey_Month': [5],
    'Journey_Day_of_Week': [2],
    'Dep_Hour': [10],
    'Dep_Minute': [30]
})

# Single-step prediction (preprocessing + model)
prediction = pipeline.predict(input_data)[0]
print(f"Predicted Price: ₹{prediction:,.0f}")
```

### Training from Scratch

```bash
# Basic training
python main.py --data data/raw/Data_Train.xlsx --model xgb

# With hyperparameter tuning
python main.py --data data/raw/Data_Train.xlsx --model xgb --tune
```

## 📈 Project Phases

| Phase | Description | Notebook |
|-------|-------------|----------|
| 1 | Data Understanding | EDA, quality assessment |
| 2 | Data Cleaning | Missing values, duplicates, outliers |
| 3 | Analysis | 8 business questions answered |
| 4 | ML Preprocessing | Feature engineering, encoding |
| 5 | Model Training | 7 models compared, XGBoost selected |
| 6 | Deployment | Streamlit app created |

## 🔑 Key Insights

1. **Airline is the #1 price predictor** (22% importance)
2. **Non-stop flights** are often BOTH cheaper AND faster
3. **Budget airlines** offer 40-60% savings
4. **Red-eye flights** (0-4 AM) are cheapest
5. **Friday** is the cheapest day to fly
6. **April** is the cheapest month

## 👤 Author

**Amer Tarek**
- GitHub: https://github.com/AM91ER
- LinkedIn: https://www.linkedin.com/in/aamer-tarek/

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Dataset source: Flight booking data from Indian domestic airlines
- Built with: Python, Scikit-learn Pipeline, XGBoost, Streamlit
- Deployed on: Streamlit Cloud
- Architecture: Single deployment-safe pipeline for reliable production deployment

---

⭐ **If you found this project useful, please give it a star!**

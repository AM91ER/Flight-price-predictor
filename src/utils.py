"""
Flight Price Predictor - Utility Functions and Constants
Author: Amer Tarek
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any

# =============================================================================
# CONSTANTS
# =============================================================================

AIRLINES = [
    'Air Asia', 'Air India', 'GoAir', 'IndiGo', 'Jet Airways',
    'Multiple carriers', 'Multiple carriers Premium economy',
    'SpiceJet', 'Trujet', 'Vistara', 'Vistara Premium economy'
]

SOURCES = ['Banglore', 'Chennai', 'Delhi', 'Kolkata', 'Mumbai']

DESTINATIONS = ['Banglore', 'Cochin', 'Delhi', 'Hyderabad', 'Kolkata', 'New Delhi']

ADDITIONAL_INFO = [
    'No info', 'In-flight meal not included', 'No check-in baggage included',
    '1 Long layover', 'Change airports', 'Business class'
]

DAY_NAMES = {
    0: 'Monday', 1: 'Tuesday', 2: 'Wednesday', 3: 'Thursday',
    4: 'Friday', 5: 'Saturday', 6: 'Sunday'
}

MONTH_NAMES = {
    1: 'January', 2: 'February', 3: 'March', 4: 'April',
    5: 'May', 6: 'June', 7: 'July', 8: 'August',
    9: 'September', 10: 'October', 11: 'November', 12: 'December'
}

# Minimum duration thresholds by number of stops (in minutes)
# Used for outlier detection and input validation
MIN_DURATION_BY_STOPS = {
    0: 30,    # Non-stop: minimum 30 minutes (short domestic)
    1: 120,   # 1 stop: minimum 2 hours (flight + layover + flight)
    2: 240,   # 2 stops: minimum 4 hours
    3: 360,   # 3 stops: minimum 6 hours
    4: 480    # 4 stops: minimum 8 hours
}

# Columns that cause data leakage
LEAKAGE_COLUMNS = ['Route_Pair', 'Arrival_Hour', 'Arrival_Minute']

# Feature columns for model
CATEGORICAL_FEATURES = ['Airline', 'Source', 'Destination', 'Additional_Info']
NUMERICAL_FEATURES = ['Duration_Minutes', 'Stops', 'Journey_Day', 'Journey_Month',
                      'Journey_Day_of_Week', 'Dep_Hour', 'Dep_Minute']
SCALE_FEATURES = ['Duration_Minutes']
PASSTHROUGH_FEATURES = ['Stops', 'Journey_Day', 'Journey_Month', 
                        'Journey_Day_of_Week', 'Dep_Hour', 'Dep_Minute']

TARGET = 'Price'

# Random state for reproducibility
RANDOM_STATE = 42

# Train/test split ratio
TEST_SIZE = 0.2


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def parse_duration(duration_str: str) -> int:
    """Parse duration string to minutes. E.g., '2h 50m' -> 170"""
    total_minutes = 0
    duration_str = str(duration_str).strip()
    
    if 'h' in duration_str:
        parts = duration_str.split('h')
        hours = int(parts[0].strip())
        total_minutes += hours * 60
        
        if len(parts) > 1 and 'm' in parts[1]:
            minutes = int(parts[1].replace('m', '').strip())
            total_minutes += minutes
    elif 'm' in duration_str:
        total_minutes = int(duration_str.replace('m', '').strip())
    
    return total_minutes


def parse_stops(stops_str: str) -> int:
    """Parse stops string to integer. E.g., 'non-stop' -> 0, '1 stop' -> 1"""
    stops_str = str(stops_str).lower().strip()
    
    if 'non-stop' in stops_str:
        return 0
    elif 'stop' in stops_str:
        return int(stops_str.split()[0])
    else:
        return int(stops_str)


def is_valid_duration(duration_minutes: int, stops: int) -> bool:
    """
    Check if duration is valid for given number of stops.
    Returns True if duration meets minimum threshold.
    """
    min_duration = MIN_DURATION_BY_STOPS.get(stops, 30)
    return duration_minutes >= min_duration


def get_time_bucket(hour: int) -> str:
    """Categorize hour into time bucket."""
    if 4 <= hour < 8:
        return 'Early Morning (4-8)'
    elif 8 <= hour < 12:
        return 'Morning (8-12)'
    elif 12 <= hour < 16:
        return 'Afternoon (12-16)'
    elif 16 <= hour < 20:
        return 'Evening (16-20)'
    elif 20 <= hour < 24:
        return 'Night (20-24)'
    else:
        return 'Red-eye (0-4)'


def format_price(price: float) -> str:
    """Format price with Indian Rupee symbol."""
    return f"₹{price:,.0f}"


def validate_input(airline: str, source: str, destination: str,
                   additional_info: str, duration_minutes: int,
                   stops: int, journey_day: int, journey_month: int,
                   journey_day_of_week: int, dep_hour: int, 
                   dep_minute: int) -> Dict[str, Any]:
    """Validate input values for prediction."""
    errors = []
    
    if airline not in AIRLINES:
        errors.append(f"Invalid airline: {airline}")
    if source not in SOURCES:
        errors.append(f"Invalid source: {source}")
    if destination not in DESTINATIONS:
        errors.append(f"Invalid destination: {destination}")
    if source == destination:
        errors.append("Source and destination cannot be the same")
    if not 0 <= stops <= 4:
        errors.append("Stops must be between 0 and 4")
    if not 1 <= journey_day <= 31:
        errors.append("Journey day must be between 1 and 31")
    if not 1 <= journey_month <= 12:
        errors.append("Journey month must be between 1 and 12")
    if not 0 <= journey_day_of_week <= 6:
        errors.append("Day of week must be between 0 and 6")
    if not 0 <= dep_hour <= 23:
        errors.append("Departure hour must be between 0 and 23")
    if not 0 <= dep_minute <= 59:
        errors.append("Departure minute must be between 0 and 59")
    if duration_minutes <= 0:
        errors.append("Duration must be positive")
    
    # Validate duration against stops
    if not is_valid_duration(duration_minutes, stops):
        min_required = MIN_DURATION_BY_STOPS.get(stops, 30)
        errors.append(f"Duration too short for {stops} stops. Minimum: {min_required} minutes")
    
    return {'valid': len(errors) == 0, 'errors': errors}

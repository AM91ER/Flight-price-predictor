"""
Flight Price Predictor - Source Package
Author: Amer Tarek
"""

from .utils import (
    AIRLINES, SOURCES, DESTINATIONS, ADDITIONAL_INFO,
    DAY_NAMES, MONTH_NAMES, MIN_DURATION_BY_STOPS,
    CATEGORICAL_FEATURES, NUMERICAL_FEATURES,
    parse_duration, parse_stops, is_valid_duration,
    format_price, validate_input
)

__version__ = "1.0.0"
__author__ = "Amer Tarek"

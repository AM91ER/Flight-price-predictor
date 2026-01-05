import numpy as np
import pandas as pd
import joblib
from typing import Dict, Any, Optional

from .utils import (
    AIRLINES, SOURCES, DESTINATIONS, ADDITIONAL_INFO,
    validate_input, format_price
)


class FlightPricePredictor:
    """Flight Price Predictor for inference."""
    
    def __init__(self, model_path: str = None, preprocessor_path: str = None, pipeline_path: str = None):
        self.model = None
        self.preprocessor = None
        self.pipeline = None
        self.feature_names = None
        
        if pipeline_path:
            self.load_pipeline(pipeline_path)
        elif model_path and preprocessor_path:
            self.load(model_path, preprocessor_path)
    
    def load(self, model_path: str, preprocessor_path: str,
             feature_names_path: Optional[str] = None):
        """Load model and preprocessor from separate files."""
        self.model = joblib.load(model_path)
        self.preprocessor = joblib.load(preprocessor_path)
        
        if feature_names_path:
            self.feature_names = joblib.load(feature_names_path)
        
        print(f"Model loaded from: {model_path}")
    
    def load_pipeline(self, pipeline_path: str):
        """Load complete pipeline from single file."""
        self.pipeline = joblib.load(pipeline_path)
        print(f"Pipeline loaded from: {pipeline_path}")
    
    def predict(self, airline: str, source: str, destination: str,
                additional_info: str, duration_minutes: int, stops: int,
                journey_day: int, journey_month: int, journey_day_of_week: int,
                dep_hour: int, dep_minute: int,
                validate: bool = True) -> Dict[str, Any]:
        """Predict flight price."""
        if self.pipeline is None and (self.model is None or self.preprocessor is None):
            raise ValueError("Model not loaded. Call load() or load_pipeline() first.")
        
        if validate:
            validation = validate_input(
                airline, source, destination, additional_info,
                duration_minutes, stops, journey_day, journey_month,
                journey_day_of_week, dep_hour, dep_minute
            )
            
            if not validation['valid']:
                return {'success': False, 'errors': validation['errors'], 'price': None}
        
        input_data = pd.DataFrame({
            'Airline': [airline],
            'Source': [source],
            'Destination': [destination],
            'Additional_Info': [additional_info],
            'Duration_Minutes': [duration_minutes],
            'Stops': [stops],
            'Journey_Day': [journey_day],
            'Journey_Month': [journey_month],
            'Journey_Day_of_Week': [journey_day_of_week],
            'Dep_Hour': [dep_hour],
            'Dep_Minute': [dep_minute]
        })
        
        if self.pipeline is not None:
            # Use pipeline (handles preprocessing + prediction)
            prediction = self.pipeline.predict(input_data)[0]
        else:
            # Use separate model and preprocessor
            input_processed = self.preprocessor.transform(input_data)
            prediction = self.model.predict(input_processed)[0]
        
        prediction = max(0, prediction)
        
        return {
            'success': True,
            'price': prediction,
            'price_formatted': format_price(prediction),
            'input': {
                'airline': airline, 'source': source, 'destination': destination,
                'stops': stops, 'duration_minutes': duration_minutes
            }
        }
    
    def predict_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        """Predict prices for multiple flights."""
        if self.pipeline is not None:
            predictions = self.pipeline.predict(df)
        else:
            X = self.preprocessor.transform(df)
            predictions = self.model.predict(X)
        
        predictions = np.maximum(0, predictions)
        
        result = df.copy()
        result['Predicted_Price'] = predictions
        return result
    
    def get_price_range(self, prediction: float, mae: float = 659) -> Dict[str, float]:
        """Get price range based on MAE."""
        return {
            'min': max(0, prediction - mae),
            'max': prediction + mae,
            'min_formatted': format_price(max(0, prediction - mae)),
            'max_formatted': format_price(prediction + mae)
        }
    
    @staticmethod
    def get_available_options() -> Dict[str, list]:
        """Get available options for input fields."""
        return {
            'airlines': AIRLINES,
            'sources': SOURCES,
            'destinations': DESTINATIONS,
            'additional_info': ADDITIONAL_INFO,
            'stops': [0, 1, 2, 3, 4]
        }

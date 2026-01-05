import numpy as np
import pandas as pd
from typing import Tuple, Optional
import warnings

from .utils import (
    parse_duration, parse_stops, LEAKAGE_COLUMNS, TARGET,
    CATEGORICAL_FEATURES, NUMERICAL_FEATURES
)

warnings.filterwarnings('ignore')


class DataPipeline:
    """Data pipeline for loading, cleaning, and feature engineering."""
    
    def __init__(self):
        self.raw_data = None
        self.cleaned_data = None
    
    def load_data(self, filepath) -> pd.DataFrame:
        """Load data from file (.xlsx, .csv, or .pkl)."""
        filepath = str(filepath)
        if filepath.endswith('.xlsx'):
            self.raw_data = pd.read_excel(filepath)
        elif filepath.endswith('.csv'):
            self.raw_data = pd.read_csv(filepath)
        elif filepath.endswith('.pkl'):
            self.raw_data = pd.read_pickle(filepath)
        else:
            raise ValueError("Unsupported file format. Use .xlsx, .csv, or .pkl")
        
        print(f"Loaded data: {self.raw_data.shape[0]} rows, {self.raw_data.shape[1]} columns")
        return self.raw_data
    
    def clean_data(self, df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """Clean data: handle missing values, duplicates, and outliers."""
        if df is None:
            df = self.raw_data.copy()
        else:
            df = df.copy()
        
        initial_rows = len(df)
        
        # Drop duplicates
        df = df.drop_duplicates()
        print(f"Dropped {initial_rows - len(df)} duplicates")
        
        # Drop rows with missing values
        missing_before = df.isnull().sum().sum()
        df = df.dropna()
        print(f"Dropped rows with {missing_before} missing values")
        
        # Handle outliers using IQR method (for Price)
        if TARGET in df.columns:
            Q1 = df[TARGET].quantile(0.25)
            Q3 = df[TARGET].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = ((df[TARGET] < lower_bound) | (df[TARGET] > upper_bound)).sum()
            df = df[(df[TARGET] >= lower_bound) & (df[TARGET] <= upper_bound)]
            print(f"Removed {outliers} outliers")
        
        print(f"Final rows: {len(df)}")
        return df
    
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create new features from existing columns."""
        df = df.copy()
        
        # Parse Duration to minutes
        if 'Duration' in df.columns:
            df['Duration_Minutes'] = df['Duration'].apply(parse_duration)
            df = df.drop('Duration', axis=1)
            print("Created: Duration_Minutes")
        
        # Parse Total_Stops to integer
        if 'Total_Stops' in df.columns:
            df['Stops'] = df['Total_Stops'].apply(parse_stops)
            df = df.drop('Total_Stops', axis=1)
            print("Created: Stops")
        
        # Parse Date_of_Journey
        if 'Date_of_Journey' in df.columns:
            df['Journey_Date'] = pd.to_datetime(df['Date_of_Journey'], format='%d/%m/%Y')
            df['Journey_Day'] = df['Journey_Date'].dt.day
            df['Journey_Month'] = df['Journey_Date'].dt.month
            df['Journey_Day_of_Week'] = df['Journey_Date'].dt.dayofweek
            df = df.drop(['Date_of_Journey', 'Journey_Date'], axis=1)
            print("Created: Journey_Day, Journey_Month, Journey_Day_of_Week")
        
        # Parse Dep_Time
        if 'Dep_Time' in df.columns:
            df['Dep_Hour'] = pd.to_datetime(df['Dep_Time']).dt.hour
            df['Dep_Minute'] = pd.to_datetime(df['Dep_Time']).dt.minute
            df = df.drop('Dep_Time', axis=1)
            print("Created: Dep_Hour, Dep_Minute")
        
        # Parse Arrival_Time
        if 'Arrival_Time' in df.columns:
            df['Arrival_Time_Clean'] = df['Arrival_Time'].str.split(' ').str[0]
            df['Arrival_Hour'] = pd.to_datetime(df['Arrival_Time_Clean'], format='%H:%M').dt.hour
            df['Arrival_Minute'] = pd.to_datetime(df['Arrival_Time_Clean'], format='%H:%M').dt.minute
            df = df.drop(['Arrival_Time', 'Arrival_Time_Clean'], axis=1)
            print("Created: Arrival_Hour, Arrival_Minute")
        
        # Create Route_Pair
        if 'Source' in df.columns and 'Destination' in df.columns:
            df['Route_Pair'] = df['Source'] + '_' + df['Destination']
            print("Created: Route_Pair")
        
        # Standardize Additional_Info
        if 'Additional_Info' in df.columns:
            df['Additional_Info'] = df['Additional_Info'].replace('No Info', 'No info')
        
        # Drop Route column if exists
        if 'Route' in df.columns:
            df = df.drop('Route', axis=1)
            print("Dropped: Route")
        
        return df
    
    def remove_leakage_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        
        df = df.copy()
        
        for col in LEAKAGE_COLUMNS:
            if col in df.columns:
                df = df.drop(col, axis=1)
                print(f"Dropped leakage column: {col}")
        
        return df
    
    def run_pipeline(self, filepath: str) -> pd.DataFrame:
        """Run full data pipeline."""
        print("=" * 50)
        print("RUNNING DATA PIPELINE")
        print("=" * 50)
        
        print("\n1. Loading data...")
        df = self.load_data(filepath)
        
        print("\n2. Cleaning data...")
        df = self.clean_data(df)
        
        print("\n3. Engineering features...")
        df = self.engineer_features(df)
        
        print("\n4. Removing leakage columns...")
        df = self.remove_leakage_columns(df)
        
        self.cleaned_data = df
        
        print("\n" + "=" * 50)
        print(f"Pipeline complete: {df.shape[0]} rows, {df.shape[1]} columns")
        print("=" * 50)
        
        return df
    
    def save_processed_data(self, df: pd.DataFrame, filepath):
        """Save processed data to file."""
        filepath = str(filepath)
        if filepath.endswith('.xlsx'):
            df.to_excel(filepath, index=False)
        elif filepath.endswith('.csv'):
            df.to_csv(filepath, index=False)
        elif filepath.endswith('.pkl'):
            df.to_pickle(filepath)
        else:
            raise ValueError("Unsupported file format. Use .xlsx, .csv, or .pkl")
        
        print(f"Saved processed data to: {filepath}")
    
    def get_feature_target_split(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """Split DataFrame into features and target."""
        X = df.drop(TARGET, axis=1)
        y = df[TARGET]
        return X, y
    
    def validate_data(self, df: pd.DataFrame) -> bool:
        """Validate processed data integrity."""
        print("Validating processed data...")
        
        # Check if DataFrame is not empty
        if df.empty:
            raise ValueError("DataFrame is empty")
        
        # Check for target column
        if TARGET not in df.columns:
            raise ValueError(f"Target column '{TARGET}' not found in data")
        
        # Check for required categorical features
        missing_cat_features = [col for col in CATEGORICAL_FEATURES if col not in df.columns]
        if missing_cat_features:
            raise ValueError(f"Missing categorical features: {missing_cat_features}")
        
        # Check for required numerical features
        missing_num_features = [col for col in NUMERICAL_FEATURES if col not in df.columns]
        if missing_num_features:
            raise ValueError(f"Missing numerical features: {missing_num_features}")
        
        # Check for data types
        if not pd.api.types.is_numeric_dtype(df[TARGET]):
            raise ValueError(f"Target column '{TARGET}' must be numeric")
        
        # Check for missing values
        missing_values = df.isnull().sum().sum()
        if missing_values > 0:
            raise ValueError(f"Data contains {missing_values} missing values")
        
        print("✓ Data validation passed")
        return True

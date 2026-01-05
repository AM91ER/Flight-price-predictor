import numpy as np
import pandas as pd
import joblib
import time
from typing import Dict, Any, Tuple, Optional
from pathlib import Path

from sklearn.model_selection import train_test_split, cross_val_score, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from .model import FlightPriceModel
from .utils import RANDOM_STATE, TEST_SIZE


class ModelTrainer:
    """Model trainer for Flight Price Predictor."""
    
    def __init__(self, model: FlightPriceModel):
        self.model = model
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.X_train_raw = None
        self.X_test_raw = None
        self.metrics = {}
        self.training_time = None
    
    def split_data(self, X: pd.DataFrame, y: pd.Series,
                   test_size: float = TEST_SIZE,
                   random_state: int = RANDOM_STATE) -> Tuple:
        """Split data into train and test sets."""
        self.X_train_raw, self.X_test_raw, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        print(f"Train set: {len(self.X_train_raw)} samples")
        print(f"Test set: {len(self.X_test_raw)} samples")
        
        return self.X_train_raw, self.X_test_raw, self.y_train, self.y_test
    
    def preprocess(self):
        """Fit preprocessor on train data and transform both train and test."""
        if self.X_train_raw is None:
            raise ValueError("Data not split. Call split_data first.")
        
        if self.model.preprocessor is None:
            self.model.build_preprocessor()
        
        # Fit on train ONLY (prevent data leakage)
        self.X_train = self.model.preprocessor.fit_transform(self.X_train_raw)
        self.X_test = self.model.preprocessor.transform(self.X_test_raw)
        
        self.model.get_feature_names()
        print(f"Preprocessed features: {self.X_train.shape[1]}")
    
    def train(self) -> float:
        """Train the model."""
        if self.X_train is None:
            raise ValueError("Data not preprocessed. Call preprocess first.")
        
        print(f"Training {self.model.model_type}...")
        
        start_time = time.time()
        self.model.model.fit(self.X_train, self.y_train)
        self.training_time = time.time() - start_time
        
        print(f"Training completed in {self.training_time:.2f} seconds")
        return self.training_time
    
    def evaluate(self) -> Dict[str, float]:
        """Evaluate model on train and test sets."""
        y_train_pred = self.model.model.predict(self.X_train)
        y_test_pred = self.model.model.predict(self.X_test)
        
        self.metrics = {
            'train_r2': r2_score(self.y_train, y_train_pred),
            'test_r2': r2_score(self.y_test, y_test_pred),
            'train_rmse': np.sqrt(mean_squared_error(self.y_train, y_train_pred)),
            'test_rmse': np.sqrt(mean_squared_error(self.y_test, y_test_pred)),
            'train_mae': mean_absolute_error(self.y_train, y_train_pred),
            'test_mae': mean_absolute_error(self.y_test, y_test_pred),
            'test_mape': np.mean(np.abs((self.y_test - y_test_pred) / self.y_test)) * 100,
            'training_time': self.training_time
        }
        
        return self.metrics
    
    def cross_validate(self, cv: int = 5, scoring: str = 'r2') -> Dict[str, float]:
        """Perform cross-validation."""
        scores = cross_val_score(
            self.model.model, self.X_train, self.y_train, cv=cv, scoring=scoring
        )
        return {'cv_scores': scores, 'cv_mean': scores.mean(), 'cv_std': scores.std()}
    
    def tune_hyperparameters(self, param_grid: Dict, n_iter: int = 20, cv: int = 5) -> Dict:
        """Tune hyperparameters using RandomizedSearchCV."""
        print(f"Tuning {self.model.model_type}...")
        
        search = RandomizedSearchCV(
            estimator=self.model.model, param_distributions=param_grid,
            n_iter=n_iter, cv=cv, scoring='r2', random_state=RANDOM_STATE,
            n_jobs=-1, verbose=1
        )
        
        start_time = time.time()
        search.fit(self.X_train, self.y_train)
        tuning_time = time.time() - start_time
        
        self.model.model = search.best_estimator_
        self.model.hyperparameters = search.best_params_
        
        print(f"Best CV R²: {search.best_score_:.4f}")
        return {'best_params': search.best_params_, 'best_score': search.best_score_}
    
    def print_metrics(self):
        """Print evaluation metrics."""
        print("\n" + "=" * 50)
        print(f"MODEL: {self.model.model_type.upper()}")
        print("=" * 50)
        print(f"{'Metric':<15} {'Train':<15} {'Test':<15}")
        print("-" * 45)
        print(f"{'R²':<15} {self.metrics['train_r2']:<15.4f} {self.metrics['test_r2']:<15.4f}")
        print(f"{'RMSE':<15} {self.metrics['train_rmse']:<15,.0f} {self.metrics['test_rmse']:<15,.0f}")
        print(f"{'MAE':<15} {self.metrics['train_mae']:<15,.0f} {self.metrics['test_mae']:<15,.0f}")
        print(f"{'MAPE':<15} {'-':<15} {self.metrics['test_mape']:<15.2f}%")
        print("=" * 50)
    
    def save_model(self, pipeline_path: str):
        """Save trained pipeline (preprocessor + model)."""
        if self.model.pipeline is None:
            raise ValueError("Pipeline not built. Call build_pipeline first.")
        
        Path(pipeline_path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model.pipeline, pipeline_path)
        print(f"Pipeline saved to: {pipeline_path}")
    
    def save_model_separate(self, model_path: str, preprocessor_path: str,
                           feature_names_path: Optional[str] = None):
        """Save trained model and preprocessor separately (legacy method)."""
        Path(model_path).parent.mkdir(parents=True, exist_ok=True)
        Path(preprocessor_path).parent.mkdir(parents=True, exist_ok=True)
        
        joblib.dump(self.model.model, model_path)
        print(f"Model saved to: {model_path}")
        
        joblib.dump(self.model.preprocessor, preprocessor_path)
        print(f"Preprocessor saved to: {preprocessor_path}")
        
        if feature_names_path:
            Path(feature_names_path).parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(self.model.feature_names, feature_names_path)
            print(f"Feature names saved to: {feature_names_path}")
    
    def save_processed_data(self, output_dir: str):
        """Save processed train/test data."""
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        np.save(f"{output_dir}/X_train.npy", self.X_train)
        np.save(f"{output_dir}/X_test.npy", self.X_test)
        np.save(f"{output_dir}/y_train.npy", self.y_train)
        np.save(f"{output_dir}/y_test.npy", self.y_test)
        
        print(f"Processed data saved to: {output_dir}")
    
    def get_training_summary(self) -> Dict[str, Any]:
        """Get complete training summary."""
        return {
            'model_type': self.model.model_type,
            'hyperparameters': self.model.hyperparameters,
            'train_samples': len(self.y_train) if self.y_train is not None else None,
            'test_samples': len(self.y_test) if self.y_test is not None else None,
            'n_features': self.X_train.shape[1] if self.X_train is not None else None,
            'metrics': self.metrics,
            'training_time': self.training_time
        }

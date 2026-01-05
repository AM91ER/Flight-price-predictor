import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor

from .utils import (
    CATEGORICAL_FEATURES, SCALE_FEATURES, PASSTHROUGH_FEATURES, RANDOM_STATE
)


class FlightPriceModel:
    """Flight Price Prediction Model."""
    
    DEFAULT_PARAMS = {
        'linear': {},
        'ridge': {'alpha': 1.0, 'random_state': RANDOM_STATE},
        'rf': {
            'n_estimators': 100, 'max_depth': None, 'min_samples_split': 2,
            'min_samples_leaf': 1, 'random_state': RANDOM_STATE, 'n_jobs': -1
        },
        'gb': {
            'n_estimators': 100, 'max_depth': 3, 'learning_rate': 0.1,
            'random_state': RANDOM_STATE
        },
        'xgb': {
            'n_estimators': 100, 'max_depth': 6, 'learning_rate': 0.1,
            'subsample': 0.8, 'colsample_bytree': 0.8,
            'random_state': RANDOM_STATE, 'n_jobs': -1, 'verbosity': 0
        }
    }
    
    TUNED_PARAMS = {
        'rf': {
            'n_estimators': 200, 'max_depth': 20, 'min_samples_split': 2,
            'min_samples_leaf': 1, 'max_features': 'sqrt',
            'random_state': RANDOM_STATE, 'n_jobs': -1
        },
        'xgb': {
            'n_estimators': 200, 'max_depth': 7, 'learning_rate': 0.1,
            'subsample': 0.9, 'colsample_bytree': 0.9,
            'random_state': RANDOM_STATE, 'n_jobs': -1, 'verbosity': 0
        }
    }
    
    def __init__(self, model_type: str = 'xgb', use_tuned: bool = True,
                 custom_params: Optional[Dict] = None):
        self.model_type = model_type
        self.model = None
        self.preprocessor = None
        self.feature_names = None
        
        if use_tuned and model_type in self.TUNED_PARAMS:
            self.hyperparameters = self.TUNED_PARAMS[model_type].copy()
        else:
            self.hyperparameters = self.DEFAULT_PARAMS.get(model_type, {}).copy()
        
        if custom_params:
            self.hyperparameters.update(custom_params)
        
        self._init_model()
    
    def _init_model(self):
        """Initialize the model based on model_type."""
        if self.model_type == 'linear':
            self.model = LinearRegression(**self.hyperparameters)
        elif self.model_type == 'ridge':
            self.model = Ridge(**self.hyperparameters)
        elif self.model_type == 'rf':
            self.model = RandomForestRegressor(**self.hyperparameters)
        elif self.model_type == 'gb':
            self.model = GradientBoostingRegressor(**self.hyperparameters)
        elif self.model_type == 'xgb':
            self.model = XGBRegressor(**self.hyperparameters)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
    
    def build_preprocessor(self, categorical_features: List[str] = None,
                           scale_features: List[str] = None,
                           passthrough_features: List[str] = None) -> ColumnTransformer:
        """Build preprocessing pipeline."""
        if categorical_features is None:
            categorical_features = CATEGORICAL_FEATURES
        if scale_features is None:
            scale_features = SCALE_FEATURES
        if passthrough_features is None:
            passthrough_features = PASSTHROUGH_FEATURES
        
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('scale', StandardScaler(), scale_features),
                ('passthrough', 'passthrough', passthrough_features),
                ('onehot', OneHotEncoder(drop='first', sparse_output=False, 
                                         handle_unknown='ignore'), categorical_features)
            ],
            remainder='drop'
        )
        
        return self.preprocessor
    
    def build_pipeline(self, categorical_features: List[str] = None,
                       scale_features: List[str] = None,
                       passthrough_features: List[str] = None) -> Pipeline:
        """Build complete ML pipeline with preprocessor and model."""
        if categorical_features is None:
            categorical_features = CATEGORICAL_FEATURES
        if scale_features is None:
            scale_features = SCALE_FEATURES
        if passthrough_features is None:
            passthrough_features = PASSTHROUGH_FEATURES
        
        # Build preprocessor
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('scale', StandardScaler(), scale_features),
                ('passthrough', 'passthrough', passthrough_features),
                ('onehot', OneHotEncoder(drop='first', sparse_output=False, 
                                         handle_unknown='ignore'), categorical_features)
            ],
            remainder='drop'
        )
        
        # Build complete pipeline
        self.pipeline = Pipeline([
            ('preprocessor', self.preprocessor),
            ('model', self.model)
        ])
        
        return self.pipeline
    
    def get_feature_names(self) -> List[str]:
        """Get feature names after preprocessing."""
        if self.preprocessor is None:
            raise ValueError("Preprocessor not built.")
        
        feature_names = []
        
        for name, transformer, columns in self.preprocessor.transformers_:
            if name == 'scale':
                feature_names.extend(columns)
            elif name == 'passthrough':
                feature_names.extend(columns)
            elif name == 'onehot':
                ohe = transformer
                for i, col in enumerate(columns):
                    categories = ohe.categories_[i][1:]
                    for cat in categories:
                        feature_names.append(f"{col}_{cat}")
        
        self.feature_names = feature_names
        return feature_names
    
    def get_feature_importance(self) -> pd.DataFrame:
        """Get feature importance from trained model."""
        if self.model is None:
            raise ValueError("Model not trained yet.")
        
        if hasattr(self.model, 'feature_importances_'):
            importance = self.model.feature_importances_
        elif hasattr(self.model, 'coef_'):
            importance = np.abs(self.model.coef_)
        else:
            raise ValueError("Model doesn't have feature importance.")
        
        df = pd.DataFrame({
            'Feature': self.feature_names,
            'Importance': importance
        }).sort_values('Importance', ascending=False).reset_index(drop=True)
        
        return df
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information."""
        return {
            'model_type': self.model_type,
            'hyperparameters': self.hyperparameters,
            'n_features': len(self.feature_names) if self.feature_names else None
        }

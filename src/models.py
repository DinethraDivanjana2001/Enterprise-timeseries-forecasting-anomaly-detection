"""
Forecasting models: Baseline and XGBoost implementations.
"""

import logging
from pathlib import Path
from typing import Optional, Dict, Any
import numpy as np
import pandas as pd
import joblib
from abc import ABC, abstractmethod

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BaseForecaster(ABC):
    """
    Abstract base class for forecasting models.
    """
    
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.model = None
        
    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Train the model."""
        pass
    
    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Generate predictions."""
        pass
    
    def save_model(self, save_path: str) -> None:
        """Save the trained model to disk."""
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, save_path)
        logger.info(f"Model saved to {save_path}")
    
    def load_model(self, load_path: str) -> None:
        """Load a trained model from disk."""
        self.model = joblib.load(load_path)
        logger.info(f"Model loaded from {load_path}")


class SeasonalNaiveForecaster(BaseForecaster):
    """
    Seasonal naive baseline: predicts the value from the same period in the previous season.
    """
    
    def __init__(self, seasonal_period: int = 24):
        """
        Initialize seasonal naive forecaster.
        
        Args:
            seasonal_period: Number of periods in a season (e.g., 24 for hourly data with daily seasonality)
        """
        super().__init__("seasonal_naive")
        self.seasonal_period = seasonal_period
        self.historical_values: Optional[np.ndarray] = None
        
    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        """
        Store historical values for seasonal naive forecasting.
        
        Args:
            X: Feature DataFrame (not used for this baseline)
            y: Target values
        """
        self.historical_values = y.values
        logger.info(f"Seasonal naive model fitted with {len(y)} historical values")
        
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict using seasonal naive approach.
        
        Args:
            X: Feature DataFrame (not used, but kept for interface consistency)
            
        Returns:
            Array of predictions
        """
        if self.historical_values is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        n_predictions = len(X)
        predictions = np.zeros(n_predictions)
        
        # For each prediction, use the value from seasonal_period steps back
        for i in range(n_predictions):
            # Use the last available seasonal value
            seasonal_idx = len(self.historical_values) - self.seasonal_period + (i % self.seasonal_period)
            if seasonal_idx >= 0:
                predictions[i] = self.historical_values[seasonal_idx]
            else:
                # Fallback to mean if not enough history
                predictions[i] = np.mean(self.historical_values)
        
        return predictions
    
    def save_model(self, save_path: str) -> None:
        """Save the trained model to disk."""
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        model_data = {
            'seasonal_period': self.seasonal_period,
            'historical_values': self.historical_values
        }
        joblib.dump(model_data, save_path)
        logger.info(f"Model saved to {save_path}")
    
    def load_model(self, load_path: str) -> None:
        """Load a trained model from disk."""
        model_data = joblib.load(load_path)
        self.seasonal_period = model_data['seasonal_period']
        self.historical_values = model_data['historical_values']
        logger.info(f"Model loaded from {load_path}")


class MovingAverageForecaster(BaseForecaster):
    """
    Moving average baseline forecaster.
    """
    
    def __init__(self, window_size: int = 24):
        """
        Initialize moving average forecaster.
        
        Args:
            window_size: Number of periods to average
        """
        super().__init__("moving_average")
        self.window_size = window_size
        self.historical_values: Optional[np.ndarray] = None
        
    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        """
        Store historical values for moving average forecasting.
        
        Args:
            X: Feature DataFrame (not used for this baseline)
            y: Target values
        """
        self.historical_values = y.values
        logger.info(f"Moving average model fitted with {len(y)} historical values")
        
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict using moving average.
        
        Args:
            X: Feature DataFrame
            
        Returns:
            Array of predictions
        """
        if self.historical_values is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        n_predictions = len(X)
        
        # Use the last window_size values to compute the average
        if len(self.historical_values) >= self.window_size:
            prediction_value = np.mean(self.historical_values[-self.window_size:])
        else:
            prediction_value = np.mean(self.historical_values)
        
        # Return the same average for all predictions (simple approach)
        predictions = np.full(n_predictions, prediction_value)
        
        return predictions
    
    def save_model(self, save_path: str) -> None:
        """Save the trained model to disk."""
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        model_data = {
            'window_size': self.window_size,
            'historical_values': self.historical_values
        }
        joblib.dump(model_data, save_path)
        logger.info(f"Model saved to {save_path}")
    
    def load_model(self, load_path: str) -> None:
        """Load a trained model from disk."""
        model_data = joblib.load(load_path)
        self.window_size = model_data['window_size']
        self.historical_values = model_data['historical_values']
        logger.info(f"Model loaded from {load_path}")


class XGBoostForecaster(BaseForecaster):
    """
    XGBoost regressor for time-series forecasting with lag features.
    """
    
    def __init__(self, params: Optional[Dict[str, Any]] = None):
        """
        Initialize XGBoost forecaster.
        
        Args:
            params: XGBoost hyperparameters
        """
        super().__init__("xgboost")
        
        # Import here to avoid dependency issues if not installed
        try:
            import xgboost as xgb
            self.xgb = xgb
        except ImportError:
            raise ImportError("XGBoost not installed. Install with: pip install xgboost")
        
        # Default parameters
        default_params = {
            'objective': 'reg:squarederror',
            'n_estimators': 100,
            'max_depth': 6,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42
        }
        
        if params:
            default_params.update(params)
        
        self.params = default_params
        self.model = self.xgb.XGBRegressor(**self.params)
        
    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        """
        Train XGBoost model.
        
        Args:
            X: Feature DataFrame
            y: Target values
        """
        logger.info(f"Training XGBoost with {len(X)} samples and {X.shape[1]} features")
        self.model.fit(X, y)
        logger.info("XGBoost training completed")
        
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Generate predictions using XGBoost.
        
        Args:
            X: Feature DataFrame
            
        Returns:
            Array of predictions
        """
        if self.model is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        predictions = self.model.predict(X)
        return predictions
    
    def get_feature_importance(self) -> pd.DataFrame:
        """
        Get feature importance from the trained model.
        
        Returns:
            DataFrame with feature names and importance scores
        """
        if self.model is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        importance_df = pd.DataFrame({
            'feature': self.model.feature_names_in_,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        return importance_df

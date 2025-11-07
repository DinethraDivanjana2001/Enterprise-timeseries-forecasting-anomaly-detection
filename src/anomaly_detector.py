"""
Anomaly detection using forecasting residuals.
"""

import logging
from typing import Optional, Dict, Any
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
import joblib
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AnomalyDetector:
    """
    Base class for anomaly detection using forecasting residuals.
    """
    
    def __init__(self, method: str = "zscore", **kwargs):
        """
        Initialize anomaly detector.
        
        Args:
            method: Detection method - 'zscore' or 'isolation_forest'
            **kwargs: Additional parameters for the detection method
        """
        self.method = method
        self.kwargs = kwargs
        self.detector = None
        
        if method == "zscore":
            self.threshold = kwargs.get("threshold", 3.0)
            self.rolling_window = kwargs.get("rolling_window", None)
        elif method == "isolation_forest":
            self.contamination = kwargs.get("contamination", 0.1)
            self.random_state = kwargs.get("random_state", 42)
            self.detector = IsolationForest(
                contamination=self.contamination,
                random_state=self.random_state
            )
        else:
            raise ValueError(f"Unknown method: {method}. Use 'zscore' or 'isolation_forest'")
    
    def compute_residuals(self, actual: np.ndarray, predicted: np.ndarray) -> np.ndarray:
        """
        Compute residuals (forecast errors).
        
        Args:
            actual: Actual values
            predicted: Predicted values
            
        Returns:
            Array of residuals
        """
        residuals = actual - predicted
        return residuals
    
    def detect_zscore(self, residuals: np.ndarray) -> np.ndarray:
        """
        Detect anomalies using Z-score threshold.
        
        Args:
            residuals: Array of residuals
            
        Returns:
            Boolean array indicating anomalies (True = anomaly)
        """
        if self.rolling_window:
            # Use rolling statistics
            residuals_series = pd.Series(residuals)
            rolling_mean = residuals_series.rolling(window=self.rolling_window, min_periods=1).mean()
            rolling_std = residuals_series.rolling(window=self.rolling_window, min_periods=1).std()
            
            # Avoid division by zero
            rolling_std = rolling_std.replace(0, 1e-6)
            
            z_scores = np.abs((residuals - rolling_mean) / rolling_std)
        else:
            # Use global statistics
            mean = np.mean(residuals)
            std = np.std(residuals)
            
            if std == 0:
                std = 1e-6
            
            z_scores = np.abs((residuals - mean) / std)
        
        anomalies = z_scores > self.threshold
        
        logger.info(f"Z-score detection: {np.sum(anomalies)} anomalies found ({100*np.mean(anomalies):.2f}%)")
        
        return anomalies
    
    def detect_isolation_forest(self, residuals: np.ndarray) -> np.ndarray:
        """
        Detect anomalies using Isolation Forest.
        
        Args:
            residuals: Array of residuals
            
        Returns:
            Boolean array indicating anomalies (True = anomaly)
        """
        # Reshape for sklearn
        residuals_reshaped = residuals.reshape(-1, 1)
        
        # Fit and predict
        predictions = self.detector.fit_predict(residuals_reshaped)
        
        # Isolation Forest returns -1 for anomalies, 1 for normal
        anomalies = predictions == -1
        
        logger.info(f"Isolation Forest detection: {np.sum(anomalies)} anomalies found ({100*np.mean(anomalies):.2f}%)")
        
        return anomalies
    
    def detect(self, actual: np.ndarray, predicted: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Detect anomalies in forecasting residuals.
        
        Args:
            actual: Actual values
            predicted: Predicted values
            
        Returns:
            Dictionary with residuals and anomaly flags
        """
        residuals = self.compute_residuals(actual, predicted)
        
        if self.method == "zscore":
            anomalies = self.detect_zscore(residuals)
        elif self.method == "isolation_forest":
            anomalies = self.detect_isolation_forest(residuals)
        else:
            raise ValueError(f"Unknown method: {self.method}")
        
        return {
            'residuals': residuals,
            'anomalies': anomalies,
            'anomaly_indices': np.where(anomalies)[0]
        }
    
    def save_detector(self, save_path: str) -> None:
        """Save the detector to disk."""
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        detector_config = {
            'method': self.method,
            'kwargs': self.kwargs,
            'detector': self.detector
        }
        
        joblib.dump(detector_config, save_path)
        logger.info(f"Detector saved to {save_path}")
    
    def load_detector(self, load_path: str) -> None:
        """Load a detector from disk."""
        detector_config = joblib.load(load_path)
        
        self.method = detector_config['method']
        self.kwargs = detector_config['kwargs']
        self.detector = detector_config['detector']
        
        # Restore method-specific attributes
        if self.method == "zscore":
            self.threshold = self.kwargs.get("threshold", 3.0)
            self.rolling_window = self.kwargs.get("rolling_window", None)
        elif self.method == "isolation_forest":
            self.contamination = self.kwargs.get("contamination", 0.1)
            self.random_state = self.kwargs.get("random_state", 42)
        
        logger.info(f"Detector loaded from {load_path}")


class ResidualFeatureExtractor:
    """
    Extract features from residuals for more sophisticated anomaly detection.
    """
    
    @staticmethod
    def extract_features(residuals: np.ndarray, window_size: int = 5) -> pd.DataFrame:
        """
        Extract statistical features from residuals.
        
        Args:
            residuals: Array of residuals
            window_size: Window size for rolling statistics
            
        Returns:
            DataFrame with extracted features
        """
        residuals_series = pd.Series(residuals)
        
        features = pd.DataFrame({
            'residual': residuals,
            'abs_residual': np.abs(residuals),
            'squared_residual': residuals ** 2,
            'rolling_mean': residuals_series.rolling(window=window_size, min_periods=1).mean(),
            'rolling_std': residuals_series.rolling(window=window_size, min_periods=1).std(),
            'rolling_max': residuals_series.rolling(window=window_size, min_periods=1).max(),
            'rolling_min': residuals_series.rolling(window=window_size, min_periods=1).min()
        })
        
        return features

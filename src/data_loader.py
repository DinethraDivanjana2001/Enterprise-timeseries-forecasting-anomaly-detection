"""
Data loading and preprocessing utilities for time-series data.
"""

import logging
from pathlib import Path
from typing import Tuple, Optional
import pandas as pd
import numpy as np
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TimeSeriesDataLoader:
    """
    Loads and preprocesses time-series data for forecasting and anomaly detection.
    """
    
    def __init__(self, data_path: str, timestamp_col: str = "timestamp", value_col: str = "value"):
        """
        Initialize the data loader.
        
        Args:
            data_path: Path to the CSV data file
            timestamp_col: Name of the timestamp column
            value_col: Name of the value column
        """
        self.data_path = Path(data_path)
        self.timestamp_col = timestamp_col
        self.value_col = value_col
        self.data: Optional[pd.DataFrame] = None
        
    def load_data(self) -> pd.DataFrame:
        """
        Load time-series data from CSV file.
        
        Returns:
            DataFrame with parsed timestamps
        """
        logger.info(f"Loading data from {self.data_path}")
        
        if not self.data_path.exists():
            raise FileNotFoundError(f"Data file not found: {self.data_path}")
        
        self.data = pd.read_csv(self.data_path)
        
        # Parse timestamp column
        if self.timestamp_col in self.data.columns:
            self.data[self.timestamp_col] = pd.to_datetime(self.data[self.timestamp_col])
            self.data = self.data.sort_values(self.timestamp_col).reset_index(drop=True)
        
        logger.info(f"Loaded {len(self.data)} records")
        logger.info(f"Date range: {self.data[self.timestamp_col].min()} to {self.data[self.timestamp_col].max()}")
        
        return self.data
    
    def split_data(
        self,
        train_end: str,
        val_end: str,
        test_end: Optional[str] = None
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split data into train, validation, and test sets based on dates.
        
        Args:
            train_end: End date for training set (inclusive)
            val_end: End date for validation set (inclusive)
            test_end: End date for test set (inclusive), None means use all remaining data
            
        Returns:
            Tuple of (train_df, val_df, test_df)
        """
        if self.data is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        train_end_dt = pd.to_datetime(train_end)
        val_end_dt = pd.to_datetime(val_end)
        
        train_df = self.data[self.data[self.timestamp_col] <= train_end_dt].copy()
        val_df = self.data[
            (self.data[self.timestamp_col] > train_end_dt) & 
            (self.data[self.timestamp_col] <= val_end_dt)
        ].copy()
        
        if test_end is not None:
            test_end_dt = pd.to_datetime(test_end)
            test_df = self.data[
                (self.data[self.timestamp_col] > val_end_dt) & 
                (self.data[self.timestamp_col] <= test_end_dt)
            ].copy()
        else:
            test_df = self.data[self.data[self.timestamp_col] > val_end_dt].copy()
        
        logger.info(f"Train set: {len(train_df)} records")
        logger.info(f"Validation set: {len(val_df)} records")
        logger.info(f"Test set: {len(test_df)} records")
        
        return train_df, val_df, test_df
    
    def create_lag_features(
        self,
        df: pd.DataFrame,
        lags: list[int],
        rolling_windows: Optional[list[int]] = None
    ) -> pd.DataFrame:
        """
        Create lag features and rolling statistics for time-series forecasting.
        
        Args:
            df: Input DataFrame
            lags: List of lag periods to create
            rolling_windows: List of window sizes for rolling statistics
            
        Returns:
            DataFrame with lag features
        """
        result_df = df.copy()
        
        # Create lag features
        for lag in lags:
            result_df[f'lag_{lag}'] = result_df[self.value_col].shift(lag)
        
        # Create rolling statistics
        if rolling_windows:
            for window in rolling_windows:
                result_df[f'rolling_mean_{window}'] = result_df[self.value_col].rolling(window=window).mean()
                result_df[f'rolling_std_{window}'] = result_df[self.value_col].rolling(window=window).std()
                result_df[f'rolling_min_{window}'] = result_df[self.value_col].rolling(window=window).min()
                result_df[f'rolling_max_{window}'] = result_df[self.value_col].rolling(window=window).max()
        
        # Add time-based features
        if self.timestamp_col in result_df.columns:
            result_df['hour'] = result_df[self.timestamp_col].dt.hour
            result_df['day_of_week'] = result_df[self.timestamp_col].dt.dayofweek
            result_df['day_of_month'] = result_df[self.timestamp_col].dt.day
            result_df['month'] = result_df[self.timestamp_col].dt.month
        
        # Drop rows with NaN values created by lag/rolling operations
        result_df = result_df.dropna().reset_index(drop=True)
        
        logger.info(f"Created features. Resulting shape: {result_df.shape}")
        
        return result_df

"""
Unit tests for data loader module.
"""

import unittest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.data_loader import TimeSeriesDataLoader


class TestTimeSeriesDataLoader(unittest.TestCase):
    """Test cases for TimeSeriesDataLoader class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a small test dataset
        self.test_data = pd.DataFrame({
            'timestamp': pd.date_range('2020-01-01', periods=100, freq='H'),
            'value': np.random.randn(100) * 10 + 100
        })
        
        self.test_file = Path(__file__).parent / 'test_data.csv'
        self.test_data.to_csv(self.test_file, index=False)
        
    def tearDown(self):
        """Clean up test files."""
        if self.test_file.exists():
            self.test_file.unlink()
    
    def test_load_data(self):
        """Test data loading."""
        loader = TimeSeriesDataLoader(
            data_path=str(self.test_file),
            timestamp_col='timestamp',
            value_col='value'
        )
        
        data = loader.load_data()
        
        self.assertEqual(len(data), 100)
        self.assertIn('timestamp', data.columns)
        self.assertIn('value', data.columns)
        self.assertTrue(pd.api.types.is_datetime64_any_dtype(data['timestamp']))
    
    def test_create_lag_features(self):
        """Test lag feature creation."""
        loader = TimeSeriesDataLoader(
            data_path=str(self.test_file),
            timestamp_col='timestamp',
            value_col='value'
        )
        
        loader.load_data()
        features = loader.create_lag_features(
            loader.data,
            lags=[1, 2, 3],
            rolling_windows=[3]
        )
        
        # Check that lag features were created
        self.assertIn('lag_1', features.columns)
        self.assertIn('lag_2', features.columns)
        self.assertIn('lag_3', features.columns)
        self.assertIn('rolling_mean_3', features.columns)
        
        # Check that NaN rows were dropped
        self.assertFalse(features.isnull().any().any())


if __name__ == '__main__':
    unittest.main()

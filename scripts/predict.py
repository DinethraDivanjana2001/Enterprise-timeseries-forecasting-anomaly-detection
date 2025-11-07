"""
Generate predictions using trained models on test data.
"""

import logging
import sys
from pathlib import Path
import yaml
import pandas as pd
import numpy as np

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.data_loader import TimeSeriesDataLoader
from src.models import SeasonalNaiveForecaster, MovingAverageForecaster, XGBoostForecaster

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def main():
    """Main execution function."""
    # Paths
    project_root = Path(__file__).parent.parent
    dataset_config_path = project_root / "configs" / "dataset_config.yaml"
    
    # Load configuration
    logger.info("Loading configuration...")
    dataset_config = load_config(dataset_config_path)
    
    # Initialize data loader
    data_path = project_root / dataset_config['dataset']['path']
    loader = TimeSeriesDataLoader(
        data_path=str(data_path),
        timestamp_col=dataset_config['dataset']['timestamp_col'],
        value_col=dataset_config['dataset']['value_col']
    )
    
    # Load data
    logger.info("Loading data...")
    loader.load_data()
    
    # Split data
    logger.info("Splitting data...")
    train_df, val_df, test_df = loader.split_data(
        train_end=dataset_config['split']['train_end'],
        val_end=dataset_config['split']['val_end'],
        test_end=dataset_config['split']['test_end']
    )
    
    # Create lag features for test data
    logger.info("Creating lag features for test data...")
    test_features = loader.create_lag_features(
        test_df,
        lags=dataset_config['features']['lags'],
        rolling_windows=dataset_config['features']['rolling_windows']
    )
    
    # Prepare features and target
    feature_cols = [col for col in test_features.columns 
                   if col not in [dataset_config['dataset']['timestamp_col'], 
                                 dataset_config['dataset']['value_col']]]
    
    X_test = test_features[feature_cols]
    y_test = test_features[dataset_config['dataset']['value_col']]
    timestamps_test = test_features[dataset_config['dataset']['timestamp_col']]
    
    logger.info(f"Test features shape: {X_test.shape}")
    
    # Load models and generate predictions
    predictions = {}
    
    # Seasonal Naive
    logger.info("\n" + "="*50)
    logger.info("Loading Seasonal Naive model and generating predictions...")
    seasonal_naive = SeasonalNaiveForecaster()
    seasonal_naive.load_model(str(project_root / "models" / "seasonal_naive.pkl"))
    predictions['Seasonal_Naive'] = seasonal_naive.predict(X_test)
    
    # Moving Average
    logger.info("Loading Moving Average model and generating predictions...")
    moving_avg = MovingAverageForecaster()
    moving_avg.load_model(str(project_root / "models" / "moving_average.pkl"))
    predictions['Moving_Average'] = moving_avg.predict(X_test)
    
    # XGBoost
    logger.info("Loading XGBoost model and generating predictions...")
    xgb_model = XGBoostForecaster()
    xgb_model.load_model(str(project_root / "models" / "xgboost.pkl"))
    predictions['XGBoost'] = xgb_model.predict(X_test)
    
    # Save predictions
    logger.info("\n" + "="*50)
    logger.info("Saving predictions...")
    
    predictions_df = pd.DataFrame({
        'timestamp': timestamps_test,
        'actual': y_test.values,
        'seasonal_naive': predictions['Seasonal_Naive'],
        'moving_average': predictions['Moving_Average'],
        'xgboost': predictions['XGBoost']
    })
    
    output_path = project_root / "outputs" / "predictions.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    predictions_df.to_csv(output_path, index=False)
    
    logger.info(f"Predictions saved to {output_path}")
    logger.info(f"Total predictions: {len(predictions_df)}")
    
    # Display sample predictions
    logger.info("\nSample predictions:")
    logger.info(f"\n{predictions_df.head(10).to_string(index=False)}")
    
    logger.info("\n" + "="*50)
    logger.info("Prediction generation completed successfully!")
    logger.info("="*50)


if __name__ == "__main__":
    main()

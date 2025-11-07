"""
Train baseline forecasting models (Seasonal Naive and Moving Average).
"""

import logging
import sys
from pathlib import Path
import yaml
import pandas as pd

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.data_loader import TimeSeriesDataLoader
from src.models import SeasonalNaiveForecaster, MovingAverageForecaster

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
    baseline_config_path = project_root / "configs" / "baseline_config.yaml"
    
    # Load configurations
    logger.info("Loading configurations...")
    dataset_config = load_config(dataset_config_path)
    baseline_config = load_config(baseline_config_path)
    
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
    
    # Extract target values
    y_train = train_df[dataset_config['dataset']['value_col']]
    
    # Train Seasonal Naive model
    logger.info("\n" + "="*50)
    logger.info("Training Seasonal Naive model...")
    logger.info("="*50)
    
    seasonal_naive = SeasonalNaiveForecaster(
        seasonal_period=baseline_config['seasonal_naive']['seasonal_period']
    )
    seasonal_naive.fit(train_df, y_train)
    
    # Save model
    model_save_path = project_root / "models" / "seasonal_naive.pkl"
    seasonal_naive.save_model(str(model_save_path))
    
    # Train Moving Average model
    logger.info("\n" + "="*50)
    logger.info("Training Moving Average model...")
    logger.info("="*50)
    
    moving_avg = MovingAverageForecaster(
        window_size=baseline_config['moving_average']['window_size']
    )
    moving_avg.fit(train_df, y_train)
    
    # Save model
    model_save_path = project_root / "models" / "moving_average.pkl"
    moving_avg.save_model(str(model_save_path))
    
    logger.info("\n" + "="*50)
    logger.info("Baseline models training completed successfully!")
    logger.info("="*50)


if __name__ == "__main__":
    main()

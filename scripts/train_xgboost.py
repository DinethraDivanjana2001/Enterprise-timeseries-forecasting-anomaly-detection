"""
Train XGBoost forecasting model with lag features.
"""

import logging
import sys
from pathlib import Path
import yaml
import pandas as pd

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.data_loader import TimeSeriesDataLoader
from src.models import XGBoostForecaster

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
    xgboost_config_path = project_root / "configs" / "xgboost_config.yaml"
    
    # Load configurations
    logger.info("Loading configurations...")
    dataset_config = load_config(dataset_config_path)
    xgboost_config = load_config(xgboost_config_path)
    
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
    
    # Create lag features
    logger.info("Creating lag features for training data...")
    train_features = loader.create_lag_features(
        train_df,
        lags=dataset_config['features']['lags'],
        rolling_windows=dataset_config['features']['rolling_windows']
    )
    
    logger.info("Creating lag features for validation data...")
    val_features = loader.create_lag_features(
        val_df,
        lags=dataset_config['features']['lags'],
        rolling_windows=dataset_config['features']['rolling_windows']
    )
    
    # Prepare features and target
    feature_cols = [col for col in train_features.columns 
                   if col not in [dataset_config['dataset']['timestamp_col'], 
                                 dataset_config['dataset']['value_col']]]
    
    X_train = train_features[feature_cols]
    y_train = train_features[dataset_config['dataset']['value_col']]
    
    X_val = val_features[feature_cols]
    y_val = val_features[dataset_config['dataset']['value_col']]
    
    logger.info(f"Training features shape: {X_train.shape}")
    logger.info(f"Validation features shape: {X_val.shape}")
    
    # Train XGBoost model
    logger.info("\n" + "="*50)
    logger.info("Training XGBoost model...")
    logger.info("="*50)
    
    xgb_model = XGBoostForecaster(params=xgboost_config['xgboost'])
    xgb_model.fit(X_train, y_train)
    
    # Evaluate on validation set
    logger.info("\nEvaluating on validation set...")
    val_predictions = xgb_model.predict(X_val)
    
    from src.evaluation import ForecastEvaluator
    val_metrics = ForecastEvaluator.calculate_metrics(y_val.values, val_predictions)
    
    logger.info("Validation metrics:")
    for metric, value in val_metrics.items():
        logger.info(f"  {metric}: {value:.4f}")
    
    # Feature importance
    logger.info("\nTop 10 important features:")
    feature_importance = xgb_model.get_feature_importance()
    logger.info(f"\n{feature_importance.head(10).to_string(index=False)}")
    
    # Save model
    model_save_path = project_root / "models" / "xgboost.pkl"
    xgb_model.save_model(str(model_save_path))
    
    # Save feature importance
    importance_save_path = project_root / "reports" / "xgboost_feature_importance.csv"
    importance_save_path.parent.mkdir(parents=True, exist_ok=True)
    feature_importance.to_csv(importance_save_path, index=False)
    logger.info(f"Feature importance saved to {importance_save_path}")
    
    logger.info("\n" + "="*50)
    logger.info("XGBoost model training completed successfully!")
    logger.info("="*50)


if __name__ == "__main__":
    main()

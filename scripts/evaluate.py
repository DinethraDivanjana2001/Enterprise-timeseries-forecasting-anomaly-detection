"""
Evaluate forecasting models and generate comparison reports.
"""

import logging
import sys
from pathlib import Path
import yaml
import pandas as pd
import numpy as np

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.evaluation import ForecastEvaluator

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
    predictions_path = project_root / "outputs" / "predictions.csv"
    
    # Load predictions
    logger.info("Loading predictions...")
    predictions_df = pd.read_csv(predictions_path)
    predictions_df['timestamp'] = pd.to_datetime(predictions_df['timestamp'])
    
    logger.info(f"Loaded {len(predictions_df)} predictions")
    
    # Extract data
    timestamps = predictions_df['timestamp']
    actual = predictions_df['actual'].values
    
    predictions_dict = {
        'Seasonal Naive': predictions_df['seasonal_naive'].values,
        'Moving Average': predictions_df['moving_average'].values,
        'XGBoost': predictions_df['xgboost'].values
    }
    
    # Calculate and compare metrics
    logger.info("\n" + "="*50)
    logger.info("Evaluating models...")
    logger.info("="*50)
    
    metrics_path = project_root / "reports" / "metrics.csv"
    comparison_df = ForecastEvaluator.compare_models(
        actual=actual,
        predictions_dict=predictions_dict,
        save_path=str(metrics_path)
    )
    
    # Generate forecast plot
    logger.info("\n" + "="*50)
    logger.info("Generating forecast visualization...")
    
    forecast_plot_path = project_root / "reports" / "forecast_comparison.png"
    ForecastEvaluator.plot_forecast(
        timestamps=timestamps,
        actual=actual,
        predictions_dict=predictions_dict,
        title="Forecast Comparison - All Models",
        save_path=str(forecast_plot_path),
        show_last_n=500  # Show last 500 points for clarity
    )
    
    # Generate residuals plot
    logger.info("Generating residuals visualization...")
    
    residuals_dict = {
        model_name: actual - predictions 
        for model_name, predictions in predictions_dict.items()
    }
    
    residuals_plot_path = project_root / "reports" / "residuals_comparison.png"
    ForecastEvaluator.plot_residuals(
        timestamps=timestamps,
        residuals_dict=residuals_dict,
        title="Forecast Residuals - All Models",
        save_path=str(residuals_plot_path)
    )
    
    # Calculate per-model statistics
    logger.info("\n" + "="*50)
    logger.info("Model-specific statistics:")
    logger.info("="*50)
    
    for model_name, predictions in predictions_dict.items():
        residuals = actual - predictions
        
        logger.info(f"\n{model_name}:")
        logger.info(f"  Prediction range: [{predictions.min():.2f}, {predictions.max():.2f}]")
        logger.info(f"  Residual mean: {residuals.mean():.4f}")
        logger.info(f"  Residual std: {residuals.std():.4f}")
        logger.info(f"  Residual range: [{residuals.min():.2f}, {residuals.max():.2f}]")
    
    # Summary
    logger.info("\n" + "="*50)
    logger.info("EVALUATION SUMMARY")
    logger.info("="*50)
    
    logger.info("\nBest model by metric:")
    for metric in ['MAE', 'RMSE', 'MAPE']:
        best_model = comparison_df.loc[comparison_df[metric].idxmin(), 'Model']
        best_value = comparison_df[metric].min()
        logger.info(f"  {metric}: {best_model} ({best_value:.4f})")
    
    logger.info("\nGenerated outputs:")
    logger.info(f"  - Metrics table: {metrics_path}")
    logger.info(f"  - Forecast plot: {forecast_plot_path}")
    logger.info(f"  - Residuals plot: {residuals_plot_path}")
    
    logger.info("\n" + "="*50)
    logger.info("Evaluation completed successfully!")
    logger.info("="*50)


if __name__ == "__main__":
    main()

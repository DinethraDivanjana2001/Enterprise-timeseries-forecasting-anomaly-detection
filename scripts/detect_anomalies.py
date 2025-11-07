"""
Detect anomalies in forecasting residuals.
"""

import logging
import sys
from pathlib import Path
import yaml
import pandas as pd
import numpy as np

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.anomaly_detector import AnomalyDetector
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
    anomaly_config_path = project_root / "configs" / "anomaly_config.yaml"
    predictions_path = project_root / "outputs" / "predictions.csv"
    
    # Load configuration
    logger.info("Loading configuration...")
    anomaly_config = load_config(anomaly_config_path)
    
    # Load predictions
    logger.info("Loading predictions...")
    predictions_df = pd.read_csv(predictions_path)
    predictions_df['timestamp'] = pd.to_datetime(predictions_df['timestamp'])
    
    logger.info(f"Loaded {len(predictions_df)} predictions")
    
    # Extract data
    timestamps = predictions_df['timestamp']
    actual = predictions_df['actual'].values
    
    # Initialize anomaly detector
    method = anomaly_config['method']
    logger.info(f"\nUsing anomaly detection method: {method}")
    
    if method == "zscore":
        detector = AnomalyDetector(
            method="zscore",
            **anomaly_config['zscore']
        )
    elif method == "isolation_forest":
        detector = AnomalyDetector(
            method="isolation_forest",
            **anomaly_config['isolation_forest']
        )
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Detect anomalies for each model
    anomaly_results = {}
    
    models = ['seasonal_naive', 'moving_average', 'xgboost']
    
    for model_name in models:
        logger.info("\n" + "="*50)
        logger.info(f"Detecting anomalies for {model_name}...")
        logger.info("="*50)
        
        predicted = predictions_df[model_name].values
        
        # Detect anomalies
        results = detector.detect(actual, predicted)
        
        anomaly_results[model_name] = results
        
        # Log statistics
        n_anomalies = np.sum(results['anomalies'])
        anomaly_pct = 100 * n_anomalies / len(actual)
        
        logger.info(f"Anomalies detected: {n_anomalies} ({anomaly_pct:.2f}%)")
        logger.info(f"Residual statistics:")
        logger.info(f"  Mean: {np.mean(results['residuals']):.4f}")
        logger.info(f"  Std: {np.std(results['residuals']):.4f}")
        logger.info(f"  Min: {np.min(results['residuals']):.4f}")
        logger.info(f"  Max: {np.max(results['residuals']):.4f}")
        
        # Generate anomaly plot
        plot_path = project_root / "reports" / f"anomalies_{model_name}.png"
        ForecastEvaluator.plot_anomalies(
            timestamps=timestamps,
            actual=actual,
            predicted=predicted,
            anomaly_flags=results['anomalies'],
            model_name=model_name.replace('_', ' ').title(),
            title=f"Anomaly Detection - {model_name.replace('_', ' ').title()}",
            save_path=str(plot_path)
        )
    
    # Save anomaly flags
    logger.info("\n" + "="*50)
    logger.info("Saving anomaly detection results...")
    
    anomaly_df = pd.DataFrame({
        'timestamp': timestamps,
        'actual': actual,
        'seasonal_naive_anomaly': anomaly_results['seasonal_naive']['anomalies'],
        'moving_average_anomaly': anomaly_results['moving_average']['anomalies'],
        'xgboost_anomaly': anomaly_results['xgboost']['anomalies']
    })
    
    output_path = project_root / "outputs" / "anomalies.csv"
    anomaly_df.to_csv(output_path, index=False)
    
    logger.info(f"Anomaly flags saved to {output_path}")
    
    # Save detector
    detector_path = project_root / "models" / "anomaly_detector.pkl"
    detector.save_detector(str(detector_path))
    logger.info(f"Detector saved to {detector_path}")
    
    logger.info("\n" + "="*50)
    logger.info("Anomaly detection completed successfully!")
    logger.info("="*50)


if __name__ == "__main__":
    main()

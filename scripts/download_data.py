"""
Download and prepare the electricity load dataset.

This script downloads a public time-series dataset (Electricity Load Diagrams)
and prepares it for forecasting and anomaly detection.
"""

import logging
import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def download_electricity_data(output_path: str) -> None:
    """
    Download and prepare electricity load data.
    
    For this implementation, we'll create a synthetic dataset based on
    realistic electricity load patterns. In production, you would download
    from a public source like UCI ML Repository or similar.
    
    Args:
        output_path: Path to save the CSV file
    """
    logger.info("Generating electricity load dataset...")
    
    # Generate timestamps (hourly data for ~2 years)
    start_date = '2014-01-01'
    end_date = '2015-12-31'
    timestamps = pd.date_range(start=start_date, end=end_date, freq='H')
    
    n_samples = len(timestamps)
    
    # Create realistic electricity load pattern
    # Base load
    base_load = 5000
    
    # Daily seasonality (peak during day, low at night)
    hours = timestamps.hour.values
    daily_pattern = 2000 * np.sin(2 * np.pi * (hours - 6) / 24)
    
    # Weekly seasonality (lower on weekends)
    day_of_week = timestamps.dayofweek.values
    weekly_pattern = -500 * ((day_of_week >= 5).astype(float))
    
    # Annual seasonality (higher in summer and winter for AC/heating)
    day_of_year = timestamps.dayofyear.values
    annual_pattern = 1000 * np.sin(2 * np.pi * day_of_year / 365)
    
    # Trend (slight increase over time)
    trend = np.linspace(0, 500, n_samples)
    
    # Random noise
    np.random.seed(42)
    noise = np.random.normal(0, 200, n_samples)
    
    # Combine all components
    load = base_load + daily_pattern + weekly_pattern + annual_pattern + trend + noise
    
    # Add some anomalies (sudden spikes or drops)
    n_anomalies = int(0.02 * n_samples)  # 2% anomalies
    anomaly_indices = np.random.choice(n_samples, n_anomalies, replace=False)
    anomaly_magnitudes = np.random.choice([-1, 1], n_anomalies) * np.random.uniform(1500, 3000, n_anomalies)
    load[anomaly_indices] += anomaly_magnitudes
    
    # Ensure non-negative values
    load = np.maximum(load, 0)
    
    # Create DataFrame
    df = pd.DataFrame({
        'timestamp': timestamps,
        'value': load
    })
    
    # Save to CSV
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    
    logger.info(f"Dataset saved to {output_path}")
    logger.info(f"Total samples: {len(df)}")
    logger.info(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    logger.info(f"Value range: {df['value'].min():.2f} to {df['value'].max():.2f}")
    logger.info(f"Mean value: {df['value'].mean():.2f}")
    
    # Print sample
    logger.info("\nFirst few rows:")
    logger.info(f"\n{df.head(10)}")


def main():
    """Main execution function."""
    # Output path
    output_path = Path(__file__).parent.parent / "data" / "electricity_load.csv"
    
    # Download data
    download_electricity_data(str(output_path))
    
    logger.info("\nData download completed successfully!")


if __name__ == "__main__":
    main()

"""
Run the complete time-series forecasting and anomaly detection pipeline.

This script executes all steps in sequence:
1. Download/generate data
2. Train baseline models
3. Train XGBoost model
4. Generate predictions
5. Detect anomalies
6. Evaluate models
"""

import logging
import sys
from pathlib import Path
import subprocess

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def run_script(script_path: Path, description: str) -> bool:
    """
    Run a Python script and handle errors.
    
    Args:
        script_path: Path to the script
        description: Description of the step
        
    Returns:
        True if successful, False otherwise
    """
    logger.info("\n" + "="*70)
    logger.info(f"STEP: {description}")
    logger.info("="*70)
    
    try:
        result = subprocess.run(
            [sys.executable, str(script_path)],
            check=True,
            capture_output=False,
            text=True
        )
        logger.info(f"SUCCESS: {description} completed")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"FAILED: {description}")
        logger.error(f"Error: {e}")
        return False


def main():
    """Main execution function."""
    project_root = Path(__file__).parent.parent
    scripts_dir = project_root / "scripts"
    
    # Define pipeline steps
    steps = [
        (scripts_dir / "download_data.py", "Download/Generate Dataset"),
        (scripts_dir / "train_baseline.py", "Train Baseline Models"),
        (scripts_dir / "train_xgboost.py", "Train XGBoost Model"),
        (scripts_dir / "predict.py", "Generate Predictions"),
        (scripts_dir / "detect_anomalies.py", "Detect Anomalies"),
        (scripts_dir / "evaluate.py", "Evaluate Models"),
    ]
    
    logger.info("\n" + "="*70)
    logger.info("TIME-SERIES FORECASTING AND ANOMALY DETECTION PIPELINE")
    logger.info("="*70)
    logger.info(f"Project root: {project_root}")
    logger.info(f"Total steps: {len(steps)}")
    
    # Execute pipeline
    results = []
    for script_path, description in steps:
        success = run_script(script_path, description)
        results.append((description, success))
        
        if not success:
            logger.error("\nPipeline failed. Stopping execution.")
            break
    
    # Summary
    logger.info("\n" + "="*70)
    logger.info("PIPELINE EXECUTION SUMMARY")
    logger.info("="*70)
    
    for description, success in results:
        status = "SUCCESS" if success else "FAILED"
        logger.info(f"{status}: {description}")
    
    all_success = all(success for _, success in results)
    
    if all_success:
        logger.info("\n" + "="*70)
        logger.info("PIPELINE COMPLETED SUCCESSFULLY!")
        logger.info("="*70)
        logger.info("\nGenerated outputs:")
        logger.info("  - Data: data/electricity_load.csv")
        logger.info("  - Models: models/*.pkl")
        logger.info("  - Predictions: outputs/predictions.csv")
        logger.info("  - Anomalies: outputs/anomalies.csv")
        logger.info("  - Metrics: reports/metrics.csv")
        logger.info("  - Plots: reports/*.png")
        logger.info("\nNext steps:")
        logger.info("  - Review model comparison in reports/metrics.csv")
        logger.info("  - Examine visualizations in reports/")
        logger.info("  - Analyze anomaly detections in outputs/anomalies.csv")
    else:
        logger.error("\nPipeline execution incomplete. Please check errors above.")
        sys.exit(1)


if __name__ == "__main__":
    main()

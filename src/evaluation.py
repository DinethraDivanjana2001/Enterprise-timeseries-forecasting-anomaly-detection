"""
Evaluation metrics and reporting utilities.
"""

import logging
from typing import Dict, List, Optional
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set plotting style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)


class ForecastEvaluator:
    """
    Evaluate forecasting models and generate reports.
    """
    
    @staticmethod
    def calculate_metrics(actual: np.ndarray, predicted: np.ndarray) -> Dict[str, float]:
        """
        Calculate forecasting metrics.
        
        Args:
            actual: Actual values
            predicted: Predicted values
            
        Returns:
            Dictionary of metrics
        """
        # Mean Absolute Error
        mae = np.mean(np.abs(actual - predicted))
        
        # Root Mean Squared Error
        rmse = np.sqrt(np.mean((actual - predicted) ** 2))
        
        # Mean Absolute Percentage Error
        # Avoid division by zero
        mask = actual != 0
        if np.sum(mask) > 0:
            mape = np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100
        else:
            mape = np.nan
        
        # R-squared
        ss_res = np.sum((actual - predicted) ** 2)
        ss_tot = np.sum((actual - np.mean(actual)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else np.nan
        
        metrics = {
            'MAE': mae,
            'RMSE': rmse,
            'MAPE': mape,
            'R2': r2
        }
        
        return metrics
    
    @staticmethod
    def compare_models(
        actual: np.ndarray,
        predictions_dict: Dict[str, np.ndarray],
        save_path: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Compare multiple models and create a metrics table.
        
        Args:
            actual: Actual values
            predictions_dict: Dictionary mapping model names to predictions
            save_path: Optional path to save the comparison table
            
        Returns:
            DataFrame with model comparison
        """
        results = []
        
        for model_name, predictions in predictions_dict.items():
            metrics = ForecastEvaluator.calculate_metrics(actual, predictions)
            metrics['Model'] = model_name
            results.append(metrics)
        
        comparison_df = pd.DataFrame(results)
        comparison_df = comparison_df[['Model', 'MAE', 'RMSE', 'MAPE', 'R2']]
        
        # Sort by RMSE (lower is better)
        comparison_df = comparison_df.sort_values('RMSE')
        
        logger.info("Model comparison:")
        logger.info(f"\n{comparison_df.to_string(index=False)}")
        
        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            comparison_df.to_csv(save_path, index=False)
            logger.info(f"Comparison table saved to {save_path}")
        
        return comparison_df
    
    @staticmethod
    def plot_forecast(
        timestamps: pd.Series,
        actual: np.ndarray,
        predictions_dict: Dict[str, np.ndarray],
        title: str = "Forecast vs Actual",
        save_path: Optional[str] = None,
        show_last_n: Optional[int] = None
    ) -> None:
        """
        Plot forecast vs actual values for multiple models.
        
        Args:
            timestamps: Timestamp series
            actual: Actual values
            predictions_dict: Dictionary mapping model names to predictions
            title: Plot title
            save_path: Optional path to save the plot
            show_last_n: Show only the last N points
        """
        plt.figure(figsize=(14, 7))
        
        if show_last_n:
            timestamps = timestamps.iloc[-show_last_n:]
            actual = actual[-show_last_n:]
            predictions_dict = {k: v[-show_last_n:] for k, v in predictions_dict.items()}
        
        # Plot actual values
        plt.plot(timestamps, actual, label='Actual', color='black', linewidth=2, alpha=0.7)
        
        # Plot predictions for each model
        colors = ['blue', 'red', 'green', 'orange', 'purple']
        for idx, (model_name, predictions) in enumerate(predictions_dict.items()):
            color = colors[idx % len(colors)]
            plt.plot(timestamps, predictions, label=model_name, color=color, 
                    linewidth=1.5, alpha=0.7, linestyle='--')
        
        plt.xlabel('Time', fontsize=12)
        plt.ylabel('Value', fontsize=12)
        plt.title(title, fontsize=14, fontweight='bold')
        plt.legend(loc='best', fontsize=10)
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Forecast plot saved to {save_path}")
        
        plt.close()
    
    @staticmethod
    def plot_residuals(
        timestamps: pd.Series,
        residuals_dict: Dict[str, np.ndarray],
        title: str = "Forecast Residuals",
        save_path: Optional[str] = None
    ) -> None:
        """
        Plot residuals for multiple models.
        
        Args:
            timestamps: Timestamp series
            residuals_dict: Dictionary mapping model names to residuals
            title: Plot title
            save_path: Optional path to save the plot
        """
        n_models = len(residuals_dict)
        fig, axes = plt.subplots(n_models, 1, figsize=(14, 4 * n_models))
        
        if n_models == 1:
            axes = [axes]
        
        for idx, (model_name, residuals) in enumerate(residuals_dict.items()):
            ax = axes[idx]
            ax.plot(timestamps, residuals, color='blue', alpha=0.6)
            ax.axhline(y=0, color='red', linestyle='--', linewidth=1)
            ax.fill_between(timestamps, residuals, 0, alpha=0.3)
            ax.set_xlabel('Time', fontsize=10)
            ax.set_ylabel('Residual', fontsize=10)
            ax.set_title(f'{model_name} - Residuals', fontsize=12, fontweight='bold')
            ax.tick_params(axis='x', rotation=45)
        
        plt.suptitle(title, fontsize=14, fontweight='bold', y=1.00)
        plt.tight_layout()
        
        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Residuals plot saved to {save_path}")
        
        plt.close()
    
    @staticmethod
    def plot_anomalies(
        timestamps: pd.Series,
        actual: np.ndarray,
        predicted: np.ndarray,
        anomaly_flags: np.ndarray,
        model_name: str = "Model",
        title: str = "Anomaly Detection",
        save_path: Optional[str] = None
    ) -> None:
        """
        Plot anomalies detected in forecast residuals.
        
        Args:
            timestamps: Timestamp series
            actual: Actual values
            predicted: Predicted values
            anomaly_flags: Boolean array indicating anomalies
            model_name: Name of the model
            title: Plot title
            save_path: Optional path to save the plot
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
        
        # Plot 1: Actual vs Predicted with anomalies highlighted
        ax1.plot(timestamps, actual, label='Actual', color='black', linewidth=2, alpha=0.7)
        ax1.plot(timestamps, predicted, label='Predicted', color='blue', 
                linewidth=1.5, alpha=0.7, linestyle='--')
        
        # Highlight anomalies
        anomaly_timestamps = timestamps[anomaly_flags]
        anomaly_values = actual[anomaly_flags]
        ax1.scatter(anomaly_timestamps, anomaly_values, color='red', s=100, 
                   label='Anomaly', zorder=5, alpha=0.8, edgecolors='darkred', linewidths=2)
        
        ax1.set_xlabel('Time', fontsize=12)
        ax1.set_ylabel('Value', fontsize=12)
        ax1.set_title(f'{model_name} - Forecast with Anomalies', fontsize=13, fontweight='bold')
        ax1.legend(loc='best', fontsize=10)
        ax1.tick_params(axis='x', rotation=45)
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Residuals with anomalies highlighted
        residuals = actual - predicted
        ax2.plot(timestamps, residuals, color='blue', alpha=0.6, label='Residuals')
        ax2.axhline(y=0, color='black', linestyle='--', linewidth=1)
        ax2.fill_between(timestamps, residuals, 0, alpha=0.3, color='blue')
        
        # Highlight anomalies in residuals
        anomaly_residuals = residuals[anomaly_flags]
        ax2.scatter(anomaly_timestamps, anomaly_residuals, color='red', s=100, 
                   label='Anomaly', zorder=5, alpha=0.8, edgecolors='darkred', linewidths=2)
        
        ax2.set_xlabel('Time', fontsize=12)
        ax2.set_ylabel('Residual', fontsize=12)
        ax2.set_title(f'{model_name} - Residuals with Anomalies', fontsize=13, fontweight='bold')
        ax2.legend(loc='best', fontsize=10)
        ax2.tick_params(axis='x', rotation=45)
        ax2.grid(True, alpha=0.3)
        
        plt.suptitle(title, fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Anomaly plot saved to {save_path}")
        
        plt.close()

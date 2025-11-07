# Time-Series Forecasting and Anomaly Detection

## Project Overview

This repository implements a production-grade machine learning pipeline for time-series forecasting and anomaly detection on electricity load data. The system predicts future electricity demand and identifies unusual consumption patterns that may indicate equipment failures, data quality issues, or unexpected events.

### Business Context

Accurate electricity load forecasting is critical for:
- Grid operators to balance supply and demand
- Energy traders to optimize market positions
- Utilities to plan infrastructure investments
- Facility managers to detect equipment anomalies

This project demonstrates an end-to-end ML workflow that combines multiple forecasting approaches with residual-based anomaly detection, providing both predictions and quality monitoring in a single pipeline.

## Key Features

- Multiple forecasting models with automated comparison
- Configurable anomaly detection using statistical and ML methods
- Comprehensive evaluation metrics and visualizations
- Clean, modular codebase with type hints and logging
- YAML-based configuration for easy experimentation
- Production-ready code structure (not notebook-only)

## Project Structure

```
timeseries-forecast-anomaly-mlops/
├── configs/                    # Configuration files
│   ├── dataset_config.yaml    # Data paths and split dates
│   ├── baseline_config.yaml   # Baseline model parameters
│   ├── xgboost_config.yaml    # XGBoost hyperparameters
│   └── anomaly_config.yaml    # Anomaly detection settings
├── data/                       # Dataset storage
│   └── electricity_load.csv   # Time-series data
├── src/                        # Source code modules
│   ├── __init__.py
│   ├── data_loader.py         # Data loading and preprocessing
│   ├── models.py              # Forecasting models
│   ├── anomaly_detector.py    # Anomaly detection
│   └── evaluation.py          # Metrics and visualization
├── scripts/                    # Execution scripts
│   ├── download_data.py       # Data acquisition
│   ├── train_baseline.py      # Train baseline models
│   ├── train_xgboost.py       # Train XGBoost model
│   ├── predict.py             # Generate predictions
│   ├── detect_anomalies.py    # Detect anomalies
│   └── evaluate.py            # Model evaluation
├── models/                     # Trained model artifacts
├── outputs/                    # Predictions and anomaly flags
├── reports/                    # Metrics and plots
├── tests/                      # Unit tests
├── notebooks/                  # Exploratory analysis
├── requirements.txt           # Python dependencies
└── README.md                  # This file
```

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Setup

**Option 1: Automated Setup (Recommended)**

For Windows:
```bash
setup_venv.bat
```

For Linux/Mac:
```bash
chmod +x setup_venv.sh
./setup_venv.sh
```

This will automatically:
- Create a virtual environment in `venv/`
- Install all dependencies
- Verify the installation

**Option 2: Manual Setup**

1. Create a virtual environment:
```bash
python -m venv venv
```

2. Activate the virtual environment:
```bash
# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

**IMPORTANT**: Always activate the virtual environment before running any scripts!

```bash
# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

### Quick Start

**Option 1: Run Complete Pipeline (Automated)**

For Windows:
```bash
run_pipeline.bat
```

For Linux/Mac (after activating venv):
```bash
python scripts/run_pipeline.py
```

**Option 2: Run Individual Steps**

Make sure the virtual environment is activated, then run:

```bash
# 1. Download/generate the dataset
python scripts/download_data.py

# 2. Train baseline models
python scripts/train_baseline.py

# 3. Train XGBoost model
python scripts/train_xgboost.py

# 4. Generate predictions on test data
python scripts/predict.py

# 5. Detect anomalies
python scripts/detect_anomalies.py

# 6. Evaluate and compare models
python scripts/evaluate.py
```

### Detailed Workflow

#### 1. Data Preparation

The `download_data.py` script generates a synthetic electricity load dataset with realistic patterns:
- Hourly data spanning 2 years (2014-2015)
- Daily seasonality (peak during day, low at night)
- Weekly seasonality (lower on weekends)
- Annual seasonality (higher in summer/winter)
- Injected anomalies for testing detection

```bash
python scripts/download_data.py
```

**Output**: `data/electricity_load.csv`

#### 2. Train Baseline Models

Two simple baseline models are trained for comparison:
- **Seasonal Naive**: Predicts the value from the same period in the previous season
- **Moving Average**: Uses a rolling average of recent values

```bash
python scripts/train_baseline.py
```

**Output**: 
- `models/seasonal_naive.pkl`
- `models/moving_average.pkl`

#### 3. Train XGBoost Model

The main forecasting model uses XGBoost with engineered lag features:
- Lag features: 1h, 2h, 3h, 6h, 12h, 24h, 48h, 1 week
- Rolling statistics: mean, std, min, max over multiple windows
- Time-based features: hour, day of week, day of month, month

```bash
python scripts/train_xgboost.py
```

**Output**: 
- `models/xgboost.pkl`
- `reports/xgboost_feature_importance.csv`

#### 4. Generate Predictions

Generate forecasts on the test set using all trained models:

```bash
python scripts/predict.py
```

**Output**: `outputs/predictions.csv`

#### 5. Detect Anomalies

Identify anomalies using forecast residuals (actual - predicted):

Two methods available (configured in `configs/anomaly_config.yaml`):
- **Z-score**: Flags points with residuals beyond a threshold (default: 3 std)
- **Isolation Forest**: ML-based outlier detection

```bash
python scripts/detect_anomalies.py
```

**Output**: 
- `outputs/anomalies.csv`
- `reports/anomalies_*.png` (visualization for each model)
- `models/anomaly_detector.pkl`

#### 6. Evaluate Models

Compare all models and generate comprehensive reports:

```bash
python scripts/evaluate.py
```

**Output**: 
- `reports/metrics.csv` - Model comparison table
- `reports/forecast_comparison.png` - Forecast vs actual plot
- `reports/residuals_comparison.png` - Residual analysis

## Model Comparison Results

The following table shows the performance of each model on the test set:

| Model | MAE | RMSE | MAPE | R2 |
|-------|-----|------|------|-----|
| XGBoost | TBD | TBD | TBD | TBD |
| Seasonal Naive | TBD | TBD | TBD | TBD |
| Moving Average | TBD | TBD | TBD | TBD |

*Note: Run the pipeline to generate actual results. The table will be auto-generated in `reports/metrics.csv`.*

### Expected Performance

Based on the implementation:
- **XGBoost** should significantly outperform baselines due to lag features and non-linear modeling
- **Seasonal Naive** provides a strong baseline for data with clear seasonality
- **Moving Average** is the simplest baseline, useful for sanity checking

## Configuration

All hyperparameters and settings are defined in YAML files under `configs/`:

### Dataset Configuration (`dataset_config.yaml`)

```yaml
dataset:
  path: "data/electricity_load.csv"
  timestamp_col: "timestamp"
  value_col: "value"

split:
  train_end: "2014-12-31"
  val_end: "2015-01-31"
  test_end: null

features:
  lags: [1, 2, 3, 6, 12, 24, 48, 168]
  rolling_windows: [6, 12, 24, 48]
```

### XGBoost Configuration (`xgboost_config.yaml`)

```yaml
xgboost:
  n_estimators: 200
  max_depth: 8
  learning_rate: 0.05
  subsample: 0.8
  colsample_bytree: 0.8
  # ... more parameters
```

### Anomaly Detection Configuration (`anomaly_config.yaml`)

```yaml
method: "zscore"  # or "isolation_forest"

zscore:
  threshold: 3.0
  rolling_window: 24

isolation_forest:
  contamination: 0.05
  random_state: 42
```

## Design Decisions

### Why XGBoost?

- Excellent performance on tabular data with lag features
- Fast training and inference
- Built-in regularization prevents overfitting
- Feature importance for interpretability
- Production-ready with broad library support

### Why Residual-Based Anomaly Detection?

- Leverages forecasting model's understanding of normal patterns
- Anomalies are context-aware (unusual given recent history)
- Flexible: works with any forecasting model
- Interpretable: large residuals indicate unexpected behavior

### Feature Engineering Strategy

- **Lag features**: Capture autoregressive patterns
- **Rolling statistics**: Smooth out noise and capture trends
- **Time features**: Model daily/weekly/seasonal patterns
- **Multiple windows**: Different time scales for short and long-term patterns

## Limitations and Future Work

### Current Limitations

1. **Single time series**: Currently processes one series at a time
2. **No external features**: Weather, holidays, or economic indicators not included
3. **Simple train/val/test split**: No cross-validation or walk-forward validation
4. **Synthetic data**: Real-world data may have different characteristics
5. **No deployment**: Models are trained locally, not deployed to production

### Future Enhancements

1. **Multi-series forecasting**: Extend to multiple related time series
2. **Deep learning models**: Add LSTM/GRU for comparison
3. **Automated hyperparameter tuning**: Implement Optuna or similar
4. **Online learning**: Update models with new data
5. **API deployment**: Serve predictions via REST API
6. **Monitoring dashboard**: Real-time visualization of predictions and anomalies
7. **Advanced anomaly detection**: Multivariate methods, contextual anomalies
8. **Probabilistic forecasting**: Prediction intervals and uncertainty quantification

## Technical Stack

- **Python 3.8+**: Core language
- **NumPy/Pandas**: Data manipulation
- **XGBoost**: Gradient boosting framework
- **Scikit-learn**: ML utilities and Isolation Forest
- **Matplotlib/Seaborn**: Visualization
- **PyYAML**: Configuration management
- **Joblib**: Model serialization

## Testing

Unit tests are located in the `tests/` directory. Run tests with:

```bash
pytest tests/
```

*Note: Test suite to be implemented.*

## Contributing

This is a demonstration project. For production use, consider:
- Adding comprehensive unit tests
- Implementing CI/CD pipelines
- Adding data validation checks
- Implementing model versioning
- Adding monitoring and alerting

## License

This project is for educational and demonstration purposes.

## Contact

For questions or feedback, please open an issue in the repository.

---

**Disclaimer**: This implementation uses synthetic data for demonstration. Real-world deployment requires careful validation, monitoring, and domain expertise.

# Quick Start Guide

## Time-Series Forecasting and Anomaly Detection

### Installation

**Windows:**
```bash
setup_venv.bat
```

**Linux/Mac:**
```bash
chmod +x setup_venv.sh
./setup_venv.sh
```

### Activate Virtual Environment

**ALWAYS activate the virtual environment before running any scripts!**

**Windows:**
```bash
venv\Scripts\activate
```

**Linux/Mac:**
```bash
source venv/bin/activate
```

### Run Complete Pipeline

**Windows:**
```bash
run_pipeline.bat
```

**Linux/Mac (venv activated):**
```bash
python scripts/run_pipeline.py
```

### Run Individual Steps

```bash
# Step 1: Download/generate data
python scripts/download_data.py

# Step 2: Train baseline models
python scripts/train_baseline.py

# Step 3: Train XGBoost model
python scripts/train_xgboost.py

# Step 4: Generate predictions
python scripts/predict.py

# Step 5: Detect anomalies
python scripts/detect_anomalies.py

# Step 6: Evaluate models
python scripts/evaluate.py
```

### Configuration

Edit YAML files in `configs/` to customize:
- Dataset paths and split dates
- Model hyperparameters
- Anomaly detection method and thresholds

### Outputs

After running the pipeline:
- **Models**: `models/*.pkl`
- **Predictions**: `outputs/predictions.csv`
- **Anomalies**: `outputs/anomalies.csv`
- **Metrics**: `reports/metrics.csv`
- **Visualizations**: `reports/*.png`

### Key Files

- `src/models.py` - Forecasting models
- `src/anomaly_detector.py` - Anomaly detection
- `src/evaluation.py` - Metrics and plots
- `configs/anomaly_config.yaml` - Switch between zscore/isolation_forest

### Customization

To use your own data:
1. Place CSV file in `data/`
2. Update `configs/dataset_config.yaml` with:
   - File path
   - Column names
   - Split dates
3. Run pipeline

### Troubleshooting

**Import errors**: Run `python setup.py` to install dependencies

**Missing data**: Run `python scripts/download_data.py` first

**Configuration errors**: Check YAML syntax in `configs/`

For detailed documentation, see `README.md`

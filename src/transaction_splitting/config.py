"""
Configuration file for transaction splitting detection project.
"""
import os
from pathlib import Path

# Base directories
BASE_DIR = Path(__file__).parent.parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODEL_DIR = BASE_DIR / "models"

# Ensure directories exist
MODEL_DIR.mkdir(exist_ok=True)
PROCESSED_DATA_DIR.mkdir(exist_ok=True)

# Model paths
SCALER_PATH = MODEL_DIR / "scaler.pkl"
DBSCAN_MODEL_PATH = MODEL_DIR / "dbscan_model.pkl"
IF_MODEL_PATH = MODEL_DIR / "if_model.pkl"

# Data processing parameters
WINDOW_DURATION = "24h"
MIN_TRANSACTIONS_IN_GROUP = 3

# Feature names
NUMERICAL_FEATURES = [
    "transaction_count",
    "total_group_amount",
    "amount_stdev",
    "amount_median",
    "amount_min",
    "amount_max",
    "time_span_minutes",
    "amount_coeff_of_variation",
    "amount_entropy",
    "avg_time_delta_minutes"
]

# Default model parameters
DEFAULT_IF_PARAMS = {
    "contamination": 0.02,
    "n_estimators": 300,
    "random_state": 42
}

DEFAULT_DBSCAN_PARAMS = {
    "eps": 0.9,
    "min_samples": 15
}

# MLflow configuration
MLFLOW_EXPERIMENT_NAME = "transaction-splitting-detection"
ALERT_THRESHOLD_PERCENTILE = 95
ENSEMBLE_WEIGHTS = {
    "isolation_forest": 0.7,
    "dbscan": 0.3
} 
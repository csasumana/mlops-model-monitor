import os
from pathlib import Path

# Root paths
BASE_DIR = Path(__file__).resolve().parents[2]

DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

ARTIFACTS_DIR = BASE_DIR / "artifacts"
MODELS_DIR = ARTIFACTS_DIR / "models"
METRICS_DIR = ARTIFACTS_DIR / "metrics"

# Input dataset
RAW_DATA_PATH = RAW_DATA_DIR / "WA_Fn-UseC_-Telco-Customer-Churn.csv"

# Output files
TRAIN_PROCESSED_PATH = PROCESSED_DATA_DIR / "train_processed.csv"
TEST_PROCESSED_PATH = PROCESSED_DATA_DIR / "test_processed.csv"

BEST_MODEL_PATH = MODELS_DIR / "best_model.joblib"
PREPROCESSOR_PATH = MODELS_DIR / "preprocessor.joblib"
MODEL_METADATA_PATH = METRICS_DIR / "model_metadata.json"
BASELINE_METRICS_PATH = METRICS_DIR / "baseline_metrics.json"

# Random state
RANDOM_STATE = 42

# Target column
TARGET_COLUMN = "Churn"

# MLflow / Registry config
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
MLFLOW_MODEL_NAME = os.getenv("MLFLOW_MODEL_NAME", "telco_churn_classifier")
MLFLOW_MODEL_VERSION = os.getenv("MLFLOW_MODEL_VERSION", "1")
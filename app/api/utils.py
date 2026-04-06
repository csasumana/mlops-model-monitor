import json
from typing import Tuple, Optional
import os

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow.tracking import MlflowClient

from app.training.config import BEST_MODEL_PATH, MODEL_METADATA_PATH

REGISTERED_MODEL_NAME = os.getenv("MLFLOW_MODEL_NAME", "telco_churn_classifier")
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")


def load_metadata() -> dict:
    if MODEL_METADATA_PATH.exists():
        with open(MODEL_METADATA_PATH, "r") as f:
            return json.load(f)
    return {}


def get_latest_registered_version(model_name: str) -> Optional[str]:
    """
    Return the highest registered model version number from MLflow Model Registry.
    Do NOT filter by READY status because local MLflow setups may not reliably expose it.
    """
    client = MlflowClient()

    try:
        versions = list(client.search_model_versions(f"name='{model_name}'"))

        print(f"[DEBUG] Found {len(versions)} model version(s) for {model_name}", flush=True)

        if not versions:
            return None

        for v in versions:
            print(
                f"[DEBUG] version={v.version}, status={getattr(v, 'status', None)}, stage={getattr(v, 'current_stage', None)}",
                flush=True,
            )

        latest = max(versions, key=lambda v: int(v.version))
        return str(latest.version)

    except Exception as e:
        print(f"[DEBUG] search_model_versions FAILED: {repr(e)}", flush=True)
        return None


def load_model() -> Tuple[object, str, Optional[str]]:
    """
    Try loading from MLflow registered model first.
    Fallback to local joblib pipeline.
    Returns:
        model, model_source, registered_model_version
    """
    try:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

        latest_version = get_latest_registered_version(REGISTERED_MODEL_NAME)
        print(f"[DEBUG] Latest registered version for {REGISTERED_MODEL_NAME}: {latest_version}", flush=True)

        if latest_version is not None:
            model_uri = f"models:/{REGISTERED_MODEL_NAME}/{latest_version}"
            print(f"[DEBUG] Trying MLflow registry load: {model_uri}", flush=True)

            model = mlflow.sklearn.load_model(model_uri)

            print(f"[DEBUG] Successfully loaded from MLflow registry: {model_uri}", flush=True)
            return model, "mlflow_registry", latest_version
        else:
            print(f"[DEBUG] No versions found in MLflow registry.", flush=True)

    except Exception as e:
        print(f"[DEBUG] MLflow registry load FAILED: {repr(e)}", flush=True)

    # Fallback to local pipeline
    if BEST_MODEL_PATH.exists():
        print(f"[DEBUG] Falling back to local joblib: {BEST_MODEL_PATH}", flush=True)
        model = joblib.load(BEST_MODEL_PATH)
        return model, "local_joblib", None

    raise FileNotFoundError("No model could be loaded from MLflow registry or local artifact.")


def predict_with_model(model, input_df: pd.DataFrame):
    """
    Predict using sklearn pipeline model.
    Works for both MLflow sklearn-loaded and local joblib-loaded pipeline.
    """
    prediction = int(model.predict(input_df)[0])

    probability = 0.5
    if hasattr(model, "predict_proba"):
        probability = float(model.predict_proba(input_df)[0][1])

    return prediction, probability
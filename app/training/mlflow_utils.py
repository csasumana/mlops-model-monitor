import os
import time

import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow.tracking import MlflowClient

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
EXPERIMENT_NAME = "telco-churn-mlops"
REGISTERED_MODEL_NAME = "telco_churn_classifier"


def setup_mlflow(experiment_name: str = EXPERIMENT_NAME):
    """
    Configure MLflow to use the tracking server.
    """
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(experiment_name)


def log_model_run(
    model_name: str,
    pipeline,
    metrics: dict,
    params: dict,
    X_sample: pd.DataFrame,
    artifact_dir: str = "model_artifacts",
):
    """
    Log a single model run into MLflow and return a stable runs:/ URI.
    """
    with mlflow.start_run(run_name=model_name) as run:
        run_id = run.info.run_id

        # Params
        mlflow.log_param("model_name", model_name)
        for k, v in params.items():
            mlflow.log_param(k, v)

        # Metrics
        for k, v in metrics.items():
            if v is not None:
                mlflow.log_metric(k, float(v))

        # Save sample input locally and log it
        os.makedirs("artifacts/metrics", exist_ok=True)
        sample_path = f"artifacts/metrics/{model_name}_sample_input.csv"
        X_sample.to_csv(sample_path, index=False)
        mlflow.log_artifact(sample_path, artifact_path="sample_input")

        # Log model artifact
        mlflow.sklearn.log_model(
            sk_model=pipeline,
            artifact_path=artifact_dir,
            input_example=X_sample,
        )

        # IMPORTANT: use stable runs:/ URI manually
        model_uri = f"runs:/{run_id}/{artifact_dir}"

    return run_id, model_uri


def register_best_model(model_uri: str, model_name: str = REGISTERED_MODEL_NAME):
    """
    Register the best model in MLflow Model Registry and wait until READY.
    """
    client = MlflowClient()

    # Ensure registered model exists
    try:
        client.get_registered_model(model_name)
    except Exception:
        client.create_registered_model(model_name)

    registered_model = mlflow.register_model(
        model_uri=model_uri,
        name=model_name,
    )

    version = registered_model.version

    # Wait until model version is READY
    for _ in range(30):
        mv = client.get_model_version(name=model_name, version=version)
        status = mv.status
        if status == "READY":
            return version
        elif status == "FAILED_REGISTRATION":
            raise RuntimeError(
                f"MLflow model registration failed for {model_name} v{version}"
            )
        time.sleep(1)

    raise TimeoutError(
        f"Timed out waiting for MLflow model {model_name} v{version} to become READY"
    )
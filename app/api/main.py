import pandas as pd
from fastapi import FastAPI, HTTPException

from app.api.schemas import (
    PredictionRequest,
    PredictionResponse,
    HealthResponse,
)
from app.api.utils import load_model, load_metadata, predict_with_model, REGISTERED_MODEL_NAME

app = FastAPI(
    title="Telco Churn MLOps API",
    version="1.0.0",
    description="FastAPI service for churn prediction using MLflow-registered model",
)

MODEL = None
MODEL_SOURCE = "not_loaded"
REGISTERED_VERSION = None


@app.on_event("startup")
def startup_event():
    global MODEL, MODEL_SOURCE, REGISTERED_VERSION
    try:
        MODEL, MODEL_SOURCE, REGISTERED_VERSION = load_model()
    except Exception as e:
        MODEL = None
        MODEL_SOURCE = f"load_failed: {str(e)}"
        REGISTERED_VERSION = None


@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(
        status="ok" if MODEL is not None else "degraded",
        model_loaded=MODEL is not None,
        model_source=MODEL_SOURCE,
        registered_model_name=REGISTERED_MODEL_NAME if MODEL_SOURCE == "mlflow_registry" else None,
        registered_model_version=REGISTERED_VERSION if MODEL_SOURCE == "mlflow_registry" else None,
    )


@app.get("/model-info")
def model_info():
    metadata = load_metadata()

    return {
        "registered_model_name": metadata.get("registered_model_name", REGISTERED_MODEL_NAME),
        "registered_model_version": metadata.get("registered_model_version", REGISTERED_VERSION),
        "best_model_name": metadata.get("best_model_name"),
        "best_run_id": metadata.get("best_run_id"),
        "best_model_uri": metadata.get("best_model_uri"),
        "trained_at": metadata.get("trained_at"),
        "target_column": metadata.get("target_column"),
        "feature_columns": metadata.get("feature_columns", []),
        "baseline_metrics": metadata.get("baseline_metrics", {}),
        "all_model_results": metadata.get("all_model_results", {}),
        "model_source": MODEL_SOURCE,
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(payload: PredictionRequest):
    if MODEL is None:
        raise HTTPException(status_code=503, detail="Model is not loaded.")

    try:
        input_df = pd.DataFrame([payload.model_dump()])
        prediction, probability = predict_with_model(MODEL, input_df)

        churn_label = "Yes" if prediction == 1 else "No"

        return PredictionResponse(
            prediction=prediction,
            probability=probability,
            churn_label=churn_label,
            model_source=MODEL_SOURCE,
            registered_model_name=REGISTERED_MODEL_NAME,
            registered_model_version=REGISTERED_VERSION,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
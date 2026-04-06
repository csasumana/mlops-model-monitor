import json
from datetime import datetime

import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from app.common.logger import get_logger
from app.training.config import (
    RAW_DATA_PATH,
    TRAIN_PROCESSED_PATH,
    TEST_PROCESSED_PATH,
    RANDOM_STATE,
    TARGET_COLUMN,
    MODELS_DIR,
    METRICS_DIR,
    BEST_MODEL_PATH,
    PREPROCESSOR_PATH,
    MODEL_METADATA_PATH,
    BASELINE_METRICS_PATH,
)
from app.training.preprocess import (
    load_data,
    clean_telco_data,
    split_features_target,
    build_preprocessor,
)
from app.training.evaluate import evaluate_model, to_native
from app.training.mlflow_utils import (
    setup_mlflow,
    log_model_run,
    register_best_model,
    REGISTERED_MODEL_NAME,
)

logger = get_logger(__name__)


def ensure_directories():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    TRAIN_PROCESSED_PATH.parent.mkdir(parents=True, exist_ok=True)
    TEST_PROCESSED_PATH.parent.mkdir(parents=True, exist_ok=True)


def save_processed_data(X_train, X_test, y_train, y_test):
    train_df = X_train.copy()
    train_df[TARGET_COLUMN] = y_train.values
    train_df.to_csv(TRAIN_PROCESSED_PATH, index=False)

    test_df = X_test.copy()
    test_df[TARGET_COLUMN] = y_test.values
    test_df.to_csv(TEST_PROCESSED_PATH, index=False)


def train_and_evaluate():
    logger.info("Setting up MLflow...")
    setup_mlflow()

    logger.info("Loading dataset...")
    df = load_data(RAW_DATA_PATH)

    logger.info("Cleaning dataset...")
    df = clean_telco_data(df)

    logger.info("Splitting features and target...")
    X, y = split_features_target(df, TARGET_COLUMN)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    save_processed_data(X_train, X_test, y_train, y_test)

    logger.info("Building preprocessor...")
    preprocessor = build_preprocessor(X_train)

    models = {
        "logistic_regression": {
            "estimator": LogisticRegression(
                max_iter=3000,
                solver="lbfgs",
                random_state=RANDOM_STATE,
            ),
            "params": {
                "max_iter": 3000,
                "solver": "lbfgs",
            },
        },
        "random_forest": {
            "estimator": RandomForestClassifier(
                n_estimators=200,
                max_depth=10,
                min_samples_split=5,
                random_state=RANDOM_STATE,
                n_jobs=-1,
            ),
            "params": {
                "n_estimators": 200,
                "max_depth": 10,
                "min_samples_split": 5,
            },
        },
        "xgboost": {
            "estimator": XGBClassifier(
                n_estimators=300,
                max_depth=5,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=RANDOM_STATE,
                eval_metric="logloss",
                n_jobs=-1,
            ),
            "params": {
                "n_estimators": 300,
                "max_depth": 5,
                "learning_rate": 0.05,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "eval_metric": "logloss",
            },
        },
    }

    all_results = {}
    run_ids = {}
    model_uris = {}
    best_model_name = None
    best_pipeline = None
    best_f1 = -1.0
    best_run_id = None
    best_model_uri = None
    registered_model_version = None

    X_sample = X_test.head(20).copy()
    for col in X_sample.select_dtypes(include=["int64", "int32"]).columns:
        X_sample[col] = X_sample[col].astype(float)

    for model_name, model_info in models.items():
        logger.info(f"Training model: {model_name}")

        model = model_info["estimator"]

        pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model", model),
            ]
        )

        pipeline.fit(X_train, y_train)

        y_pred = pipeline.predict(X_test)
        y_prob = None
        if hasattr(pipeline, "predict_proba"):
            y_prob = pipeline.predict_proba(X_test)[:, 1]

        metrics = evaluate_model(y_test, y_pred, y_prob)
        all_results[model_name] = metrics

        logger.info(f"{model_name} metrics: {metrics}")

        run_id, model_uri = log_model_run(
            model_name=model_name,
            pipeline=pipeline,
            metrics=metrics,
            params=model_info["params"],
            X_sample=X_sample,
        )
        run_ids[model_name] = run_id
        model_uris[model_name] = model_uri
        logger.info(f"Logged MLflow run for {model_name}: {run_id}")
        logger.info(f"Model URI for {model_name}: {model_uri}")

        if metrics["f1_score"] > best_f1:
            best_f1 = metrics["f1_score"]
            best_model_name = model_name
            best_pipeline = pipeline
            best_run_id = run_id
            best_model_uri = model_uri

    logger.info(f"Best model selected: {best_model_name} (F1 = {best_f1:.4f})")

    logger.info(f"Registering best model '{best_model_name}' as '{REGISTERED_MODEL_NAME}'...")
    registered_model_version = register_best_model(best_model_uri, REGISTERED_MODEL_NAME)
    logger.info(f"Registered model version: {registered_model_version}")

    # Save best full pipeline locally too
    joblib.dump(best_pipeline, BEST_MODEL_PATH)

    # Save fitted preprocessor separately
    fitted_preprocessor = best_pipeline.named_steps["preprocessor"]
    joblib.dump(fitted_preprocessor, PREPROCESSOR_PATH)

    metadata = {
        "best_model_name": best_model_name,
        "best_run_id": best_run_id,
        "best_model_uri": best_model_uri,
        "registered_model_name": REGISTERED_MODEL_NAME,
        "registered_model_version": registered_model_version,
        "trained_at": datetime.utcnow().isoformat(),
        "target_column": TARGET_COLUMN,
        "feature_columns": X_train.columns.tolist(),
        "baseline_metrics": all_results[best_model_name],
        "all_model_results": all_results,
        "run_ids": run_ids,
        "model_uris": model_uris,
    }

    with open(MODEL_METADATA_PATH, "w") as f:
        json.dump(to_native(metadata), f, indent=4)

    with open(BASELINE_METRICS_PATH, "w") as f:
        json.dump(to_native(all_results), f, indent=4)

    logger.info("Artifacts saved:")
    logger.info(f"Best model pipeline: {BEST_MODEL_PATH}")
    logger.info(f"Preprocessor: {PREPROCESSOR_PATH}")
    logger.info(f"Model metadata: {MODEL_METADATA_PATH}")
    logger.info(f"Baseline metrics: {BASELINE_METRICS_PATH}")


if __name__ == "__main__":
    ensure_directories()
    train_and_evaluate()
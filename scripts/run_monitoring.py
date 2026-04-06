import json
from pathlib import Path

import pandas as pd
import joblib

from app.training.config import (
    DATA_DIR,
    ARTIFACTS_DIR,
    BEST_MODEL_PATH,
    MODEL_METADATA_PATH,
)
from app.monitoring.drift import (
    calculate_custom_drift_score,
    generate_evidently_report,
    save_drift_summary,
)
from app.monitoring.performance import (
    evaluate_batch_performance,
    append_metrics_history,
)
from app.monitoring.alerts import (
    generate_alerts,
    append_alerts,
)


REFERENCE_PATH = DATA_DIR / "reference" / "reference_data.csv"
BATCHES_DIR = DATA_DIR / "batches"

REPORTS_DIR = ARTIFACTS_DIR / "reports"
METRICS_HISTORY_PATH = ARTIFACTS_DIR / "metrics" / "metrics_history.csv"
ALERTS_LOG_PATH = ARTIFACTS_DIR / "alerts" / "alerts_log.csv"


def load_metadata():
    with open(MODEL_METADATA_PATH, "r") as f:
        return json.load(f)


def main():
    if not REFERENCE_PATH.exists():
        raise FileNotFoundError(f"Reference dataset missing: {REFERENCE_PATH}")

    if not BEST_MODEL_PATH.exists():
        raise FileNotFoundError(f"Best model missing: {BEST_MODEL_PATH}")

    metadata = load_metadata()
    baseline_f1 = metadata["baseline_metrics"]["f1_score"]

    reference_df = pd.read_csv(REFERENCE_PATH)
    model = joblib.load(BEST_MODEL_PATH)

    batch_files = sorted(BATCHES_DIR.glob("batch_*.csv"))

    if not batch_files:
        raise FileNotFoundError(f"No batch files found in {BATCHES_DIR}")

    for batch_file in batch_files:
        print(f"\nProcessing {batch_file.name} ...")

        batch_df = pd.read_csv(batch_file)

        batch_id = int(batch_df["batch_id"].iloc[0])
        batch_timestamp = batch_df["batch_timestamp"].iloc[0]
        is_drifted_batch = bool(batch_df["is_drifted_batch"].iloc[0])

        # Preserve label
        y_true = batch_df["Churn"].copy()

        # Remove monitoring metadata + label for inference
        X_batch = batch_df.drop(columns=["Churn", "batch_id", "batch_timestamp", "is_drifted_batch"])

        # Reference features only (drop label)
        X_reference = reference_df.drop(columns=["Churn"])

        # Predictions
        y_pred = model.predict(X_batch)
        batch_metrics = evaluate_batch_performance(y_true, y_pred)

        # Drift summary
        drift_summary = calculate_custom_drift_score(X_reference, X_batch)

        # Save Evidently HTML report (best-effort)
        report_path = REPORTS_DIR / f"drift_report_batch_{batch_id:03d}.html"
        report_created = generate_evidently_report(X_reference, X_batch, report_path)

        drift_summary["batch_id"] = batch_id
        drift_summary["batch_timestamp"] = batch_timestamp
        drift_summary["is_drifted_batch"] = is_drifted_batch
        drift_summary["evidently_report_created"] = report_created
        drift_summary["evidently_report_path"] = str(report_path)

        summary_path = REPORTS_DIR / f"drift_summary_batch_{batch_id:03d}.json"
        save_drift_summary(drift_summary, summary_path)

        # Metrics history
        metrics_row = {
            "batch_id": batch_id,
            "batch_timestamp": batch_timestamp,
            "is_drifted_batch": is_drifted_batch,
            "accuracy": batch_metrics["accuracy"],
            "precision": batch_metrics["precision"],
            "recall": batch_metrics["recall"],
            "f1_score": batch_metrics["f1_score"],
            "baseline_f1": baseline_f1,
            "overall_drift_score": drift_summary["overall_drift_score"],
            "dataset_drift_detected": drift_summary["dataset_drift_detected"],
            "drifted_feature_count": drift_summary["drifted_feature_count"],
        }
        append_metrics_history(metrics_row, METRICS_HISTORY_PATH)

        # Alerts
        alerts = generate_alerts(
            batch_id=batch_id,
            drift_summary=drift_summary,
            batch_metrics=batch_metrics,
            baseline_f1=baseline_f1,
        )
        append_alerts(alerts, ALERTS_LOG_PATH)

        print(f"Batch metrics: {batch_metrics}")
        print(f"Drift summary: score={drift_summary['overall_drift_score']:.4f}, detected={drift_summary['dataset_drift_detected']}")
        print(f"Evidently report created: {report_created}")
        print(f"Alerts triggered: {len(alerts)}")

    print("\nMonitoring run complete.")
    print(f"Metrics history: {METRICS_HISTORY_PATH}")
    print(f"Alerts log: {ALERTS_LOG_PATH}")
    print(f"Reports dir: {REPORTS_DIR}")


if __name__ == "__main__":
    main()
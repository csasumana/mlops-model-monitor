import pandas as pd


def generate_alerts(
    batch_id: int,
    drift_summary: dict,
    batch_metrics: dict,
    baseline_f1: float,
):
    alerts = []

    if drift_summary.get("dataset_drift_detected", False):
        alerts.append({
            "batch_id": batch_id,
            "alert_type": "DATA_DRIFT",
            "severity": "HIGH" if drift_summary["overall_drift_score"] > 0.15 else "MEDIUM",
            "message": f"Dataset drift detected. Drift score={drift_summary['overall_drift_score']:.4f}",
        })

    current_f1 = batch_metrics["f1_score"]
    if current_f1 < baseline_f1 - 0.08:
        alerts.append({
            "batch_id": batch_id,
            "alert_type": "PERFORMANCE_DROP",
            "severity": "HIGH" if current_f1 < baseline_f1 - 0.15 else "MEDIUM",
            "message": f"F1 dropped from baseline {baseline_f1:.4f} to {current_f1:.4f}",
        })

    return alerts


def append_alerts(alerts: list, output_csv_path):
    if not alerts:
        return

    output_csv_path.parent.mkdir(parents=True, exist_ok=True)

    alerts_df = pd.DataFrame(alerts)

    if output_csv_path.exists():
        existing = pd.read_csv(output_csv_path)
        updated = pd.concat([existing, alerts_df], ignore_index=True)
        updated.to_csv(output_csv_path, index=False)
    else:
        alerts_df.to_csv(output_csv_path, index=False)
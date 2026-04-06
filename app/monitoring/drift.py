import json
from pathlib import Path

import numpy as np
import pandas as pd
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset


def calculate_custom_drift_score(reference_df: pd.DataFrame, current_df: pd.DataFrame) -> dict:
    """
    Simple custom drift score to avoid full dependency on Evidently internals.
    We compare:
    - numeric mean shifts (normalized)
    - categorical distribution shifts (top-category frequency delta)
    """
    numeric_cols = reference_df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_cols = reference_df.select_dtypes(include=["object", "bool"]).columns.tolist()

    # Remove label / metadata columns if present
    excluded_cols = {"Churn", "batch_id", "batch_timestamp", "is_drifted_batch"}
    numeric_cols = [c for c in numeric_cols if c not in excluded_cols]
    categorical_cols = [c for c in categorical_cols if c not in excluded_cols]

    numeric_shifts = {}
    for col in numeric_cols:
        ref_mean = reference_df[col].mean()
        cur_mean = current_df[col].mean()

        denom = abs(ref_mean) if abs(ref_mean) > 1e-6 else 1.0
        shift = abs(cur_mean - ref_mean) / denom
        numeric_shifts[col] = float(shift)

    categorical_shifts = {}
    for col in categorical_cols:
        ref_top = reference_df[col].value_counts(normalize=True, dropna=False)
        cur_top = current_df[col].value_counts(normalize=True, dropna=False)

        categories = set(ref_top.index).union(set(cur_top.index))
        total_shift = 0.0
        for cat in categories:
            total_shift += abs(ref_top.get(cat, 0.0) - cur_top.get(cat, 0.0))

        # Normalize rough total variation distance style
        categorical_shifts[col] = float(total_shift / 2.0)

    numeric_drift_score = float(np.mean(list(numeric_shifts.values()))) if numeric_shifts else 0.0
    categorical_drift_score = float(np.mean(list(categorical_shifts.values()))) if categorical_shifts else 0.0

    overall_drift_score = float((numeric_drift_score + categorical_drift_score) / 2.0)

    drifted_numeric_features = [k for k, v in numeric_shifts.items() if v > 0.10]
    drifted_categorical_features = [k for k, v in categorical_shifts.items() if v > 0.10]

    return {
        "overall_drift_score": overall_drift_score,
        "numeric_drift_score": numeric_drift_score,
        "categorical_drift_score": categorical_drift_score,
        "numeric_feature_shifts": numeric_shifts,
        "categorical_feature_shifts": categorical_shifts,
        "drifted_numeric_features": drifted_numeric_features,
        "drifted_categorical_features": drifted_categorical_features,
        "drifted_feature_count": len(drifted_numeric_features) + len(drifted_categorical_features),
        "dataset_drift_detected": overall_drift_score > 0.09,
    }


def generate_evidently_report(reference_df: pd.DataFrame, current_df: pd.DataFrame, output_html_path: Path):
    """
    Generate Evidently HTML report.
    If Evidently fails, return False but don't kill pipeline.
    """
    try:
        report = Report(metrics=[DataDriftPreset()])
        report.run(reference_data=reference_df, current_data=current_df)
        report.save_html(str(output_html_path))
        return True
    except Exception as e:
        print(f"[WARN] Evidently report generation failed: {e}")
        return False


def save_drift_summary(summary: dict, output_json_path: Path):
    output_json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json_path, "w") as f:
        json.dump(summary, f, indent=4)
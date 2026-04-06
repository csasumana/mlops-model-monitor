import json
from pathlib import Path

import pandas as pd
import streamlit as st

from app.training.config import ARTIFACTS_DIR, MODEL_METADATA_PATH


METRICS_HISTORY_PATH = ARTIFACTS_DIR / "metrics" / "metrics_history.csv"
ALERTS_LOG_PATH = ARTIFACTS_DIR / "alerts" / "alerts_log.csv"
REPORTS_DIR = ARTIFACTS_DIR / "reports"


st.set_page_config(
    page_title="ML Model Monitor Dashboard",
    layout="wide",
)


def load_metadata():
    if MODEL_METADATA_PATH.exists():
        with open(MODEL_METADATA_PATH, "r") as f:
            return json.load(f)
    return {}


def load_metrics_history():
    if METRICS_HISTORY_PATH.exists():
        return pd.read_csv(METRICS_HISTORY_PATH)
    return pd.DataFrame()


def load_alerts():
    if ALERTS_LOG_PATH.exists():
        return pd.read_csv(ALERTS_LOG_PATH)
    return pd.DataFrame()


def get_latest_drift_summary():
    if not REPORTS_DIR.exists():
        return None, None

    summaries = sorted(REPORTS_DIR.glob("drift_summary_batch_*.json"))
    if not summaries:
        return None, None

    latest = summaries[-1]
    with open(latest, "r") as f:
        data = json.load(f)

    return latest.name, data


def get_report_files():
    if not REPORTS_DIR.exists():
        return []

    return sorted([p.name for p in REPORTS_DIR.glob("*")])


def main():
    st.title("📊 ML Model Monitor Dashboard")

    metadata = load_metadata()
    metrics_df = load_metrics_history()
    alerts_df = load_alerts()
    latest_summary_name, latest_summary = get_latest_drift_summary()
    report_files = get_report_files()

    # -------------------------------
    # Top summary
    # -------------------------------
    st.subheader("Model Summary")

    col1, col2, col3, col4 = st.columns(4)

    registered_model_name = metadata.get("registered_model_name", "N/A")
    registered_model_version = metadata.get("registered_model_version", "N/A")
    best_model_name = metadata.get("best_model_name", "N/A")
    baseline_f1 = metadata.get("baseline_metrics", {}).get("f1_score", None)

    col1.metric("Registered Model", registered_model_name or "N/A")
    col2.metric("Model Version", str(registered_model_version) if registered_model_version else "N/A")
    col3.metric("Best Model", best_model_name or "N/A")
    col4.metric("Baseline F1", f"{baseline_f1:.4f}" if baseline_f1 is not None else "N/A")

    

    st.markdown("---")

    # -------------------------------
    # Latest batch status
    # -------------------------------
    st.subheader("Latest Batch Status")

    if latest_summary:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Latest Batch ID", latest_summary.get("batch_id", "N/A"))
        c2.metric("Drift Score", f"{latest_summary.get('overall_drift_score', 0):.4f}")
        c3.metric("Drift Detected", str(latest_summary.get("dataset_drift_detected", False)))
        c4.metric("Drifted Features", latest_summary.get("drifted_feature_count", 0))

        with st.expander(f"View latest drift summary ({latest_summary_name})"):
            st.json(latest_summary)
    else:
        st.info("No drift summary found yet.")

    st.markdown("---")

    # -------------------------------
    # Metrics trends
    # -------------------------------
    st.subheader("Batch Metrics Trends")

    if not metrics_df.empty:
        st.dataframe(metrics_df, use_container_width=True)

        plot_df = metrics_df.sort_values("batch_id").copy()
        plot_df = plot_df.set_index("batch_id")

        st.markdown("### Performance Over Time")
        st.line_chart(plot_df[["accuracy", "precision", "recall", "f1_score"]])

        st.markdown("### Drift Over Time")
        st.line_chart(plot_df[["overall_drift_score"]])

        st.markdown("### Drifted Feature Count")
        st.bar_chart(plot_df[["drifted_feature_count"]])
    else:
        st.warning("metrics_history.csv not found.")

    st.markdown("---")

    # -------------------------------
    # Alerts
    # -------------------------------
    st.subheader("Alerts Log")

    if not alerts_df.empty:
        st.dataframe(alerts_df, use_container_width=True)
    else:
        st.success("No alerts logged yet.")

    st.markdown("---")

    # -------------------------------
    # Reports
    # -------------------------------
    st.subheader("Generated Reports")

    if report_files:
        reports_df = pd.DataFrame({"report_file": report_files})
        st.dataframe(reports_df, use_container_width=True)
        st.caption(f"Reports directory: {REPORTS_DIR}")
    else:
        st.info("No reports found.")


if __name__ == "__main__":
    main()
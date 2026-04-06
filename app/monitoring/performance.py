import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def evaluate_batch_performance(y_true, y_pred) -> dict:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1_score": float(f1_score(y_true, y_pred, zero_division=0)),
    }


def append_metrics_history(metrics_row: dict, output_csv_path):
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)

    row_df = pd.DataFrame([metrics_row])

    if output_csv_path.exists():
        existing = pd.read_csv(output_csv_path)
        updated = pd.concat([existing, row_df], ignore_index=True)
        updated.to_csv(output_csv_path, index=False)
    else:
        row_df.to_csv(output_csv_path, index=False)
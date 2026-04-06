import argparse
from datetime import datetime

import numpy as np
import pandas as pd

from app.training.config import TEST_PROCESSED_PATH, DATA_DIR, RANDOM_STATE


BATCHES_DIR = DATA_DIR / "batches"


def apply_drift(df: pd.DataFrame, severity: float = 0.2) -> pd.DataFrame:
    """
    Apply controlled drift to selected features.
    severity ~ 0.1 to 0.5 recommended
    """
    drifted = df.copy()

    # 1) Increase MonthlyCharges
    if "MonthlyCharges" in drifted.columns:
        drifted["MonthlyCharges"] = drifted["MonthlyCharges"] * (1 + severity)

    # 2) Increase TotalCharges slightly
    if "TotalCharges" in drifted.columns:
        drifted["TotalCharges"] = drifted["TotalCharges"] * (1 + severity * 0.6)

    # 3) Shift tenure downward (simulate newer customers)
    if "tenure" in drifted.columns:
        drifted["tenure"] = np.maximum(
            0,
            (drifted["tenure"] * (1 - severity)).round()
        ).astype(int)

    # 4) Contract distribution drift
    if "Contract" in drifted.columns:
        mask = np.random.rand(len(drifted)) < (0.25 + severity * 0.3)
        drifted.loc[mask, "Contract"] = "Month-to-month"

    # 5) Payment method drift
    if "PaymentMethod" in drifted.columns:
        mask = np.random.rand(len(drifted)) < (0.20 + severity * 0.2)
        drifted.loc[mask, "PaymentMethod"] = "Electronic check"

    return drifted


def generate_batch(batch_id: int, size: int = 200, drift: bool = False, severity: float = 0.2):
    if not TEST_PROCESSED_PATH.exists():
        raise FileNotFoundError(
            f"Processed test data not found: {TEST_PROCESSED_PATH}. Run training first."
        )

    df = pd.read_csv(TEST_PROCESSED_PATH)

    if size > len(df):
        size = len(df)

    batch = df.sample(n=size, random_state=RANDOM_STATE + batch_id).reset_index(drop=True)

    if drift:
        batch = apply_drift(batch, severity=severity)

    batch["batch_id"] = batch_id
    batch["batch_timestamp"] = datetime.utcnow().isoformat()
    batch["is_drifted_batch"] = drift

    BATCHES_DIR.mkdir(parents=True, exist_ok=True)

    filename = f"batch_{batch_id:03d}.csv"
    output_path = BATCHES_DIR / filename
    batch.to_csv(output_path, index=False)

    print(f"Saved batch to: {output_path}")
    print(f"Batch ID: {batch_id}")
    print(f"Rows: {len(batch)}")
    print(f"Drift applied: {drift}")
    print(f"Severity: {severity if drift else 0.0}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-id", type=int, required=True)
    parser.add_argument("--size", type=int, default=200)
    parser.add_argument("--drift", action="store_true")
    parser.add_argument("--severity", type=float, default=0.2)

    args = parser.parse_args()

    generate_batch(
        batch_id=args.batch_id,
        size=args.size,
        drift=args.drift,
        severity=args.severity,
    )


if __name__ == "__main__":
    main()
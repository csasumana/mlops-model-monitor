import pandas as pd

from app.training.config import TRAIN_PROCESSED_PATH, DATA_DIR


REFERENCE_PATH = DATA_DIR / "reference" / "reference_data.csv"


def main():
    if not TRAIN_PROCESSED_PATH.exists():
        raise FileNotFoundError(
            f"Processed training data not found: {TRAIN_PROCESSED_PATH}. Run training first."
        )

    df = pd.read_csv(TRAIN_PROCESSED_PATH)

    REFERENCE_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(REFERENCE_PATH, index=False)

    print(f"Reference dataset saved to: {REFERENCE_PATH}")
    print(f"Rows: {len(df)}, Columns: {len(df.columns)}")


if __name__ == "__main__":
    main()
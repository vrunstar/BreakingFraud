import pandas as pd
from pathlib import Path

RAW_DATA_DIR = Path("backend/data/raw")


def load_raw_data():
    """Loads and left-merges IEEE-CIS transaction + identity data on TransactionID.

    Identity data only exists for ~24% of transactions, so most rows will
    have NaN in identity columns (DeviceType, DeviceInfo, etc.) after the
    merge — that's expected, not a bug.
    """
    transaction_path = RAW_DATA_DIR / "train_transaction.csv"
    identity_path = RAW_DATA_DIR / "train_identity.csv"

    if not transaction_path.exists():
        raise FileNotFoundError(f"No file found at {transaction_path}")
    if not identity_path.exists():
        raise FileNotFoundError(f"No file found at {identity_path}")

    df_transaction = pd.read_csv(transaction_path)
    df_identity = pd.read_csv(identity_path)

    df = df_transaction.merge(df_identity, on="TransactionID", how="left")

    return df


def basic_schema_check(df):
    required_columns = ["TransactionID", "TransactionAmt", "isFraud"]

    missing = set(required_columns) - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    valid_classes = set(df["isFraud"].unique()).issubset({0, 1})
    if not valid_classes:
        raise ValueError("isFraud column has unexpected values")

    print(f"Shape: {df.shape}")
    print(f"Fraud distribution:\n{df['isFraud'].value_counts()}")
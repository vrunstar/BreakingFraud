import pandas as pd
from pathlib import Path

RAW_DATA_PATH = Path("data/raw/creditcard.csv")

def load_raw_data():
    if RAW_DATA_PATH.exists():
        df = pd.read_csv(RAW_DATA_PATH)
    else:
        raise FileNotFoundError(f"No file found at {RAW_DATA_PATH}")
    return df

def basic_schema_check(df):
    expected_columns = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount", "Class"]
    
    missing = set(expected_columns) - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    
    valid_classes = set(df['Class'].unique()).issubset({0, 1})
    if not valid_classes:
        raise ValueError("Class column has unexpected values")
    
    print(f"Shape: {df.shape}")
    print(f"Class distribution:\n{df['Class'].value_counts()}")
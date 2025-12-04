# src/data_prep.py
import numpy as np
import pandas as pd
from .config import RAW_DATA_PATH

COLUMNS_WITH_ZERO_AS_MISSING = [
    "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"
]


def load_raw_data(path: str | None = None) -> pd.DataFrame:
    """Load raw diabetes dataset."""
    csv_path = RAW_DATA_PATH if path is None else path
    df = pd.read_csv(csv_path)
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Handle duplicates, constant columns, and zero-as-missing."""
    # Drop duplicates
    df = df.drop_duplicates()

    # Drop constant columns
    constant_cols = [c for c in df.columns if df[c].nunique() == 1]
    df = df.drop(columns=constant_cols, axis=1)

    # Replace zeros with NaN and impute with median
    df[COLUMNS_WITH_ZERO_AS_MISSING] = df[COLUMNS_WITH_ZERO_AS_MISSING].replace(
        0, np.nan
    )
    df[COLUMNS_WITH_ZERO_AS_MISSING] = df[COLUMNS_WITH_ZERO_AS_MISSING].fillna(
        df[COLUMNS_WITH_ZERO_AS_MISSING].median()
    )

    return df


def get_features_and_target(df: pd.DataFrame):
    """Split dataframe into X (features) and y (target)."""
    X = df.drop("Outcome", axis=1)
    y = df["Outcome"]
    return X, y

# src/data_prep.py
import numpy as np
import pandas as pd
from .config import RAW_DATA_PATH

COLUMNS_WITH_ZERO_AS_MISSING = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]

def load_raw_data(path: str | None = None) -> pd.DataFrame:
    csv_path = RAW_DATA_PATH if path is None else path  #Load raw diabetes dataset.
    df = pd.read_csv(csv_path)
    return df

# Handle duplicates, constant columns, and zero-as-missing.
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop_duplicates()  # Drop Duplicate
    constant_cols = [c for c in df.columns if df[c].nunique() == 1] # Drop constant columns
    df = df.drop(columns=constant_cols, axis=1)

    # Replace zeros with NaN and impute with median
    df[COLUMNS_WITH_ZERO_AS_MISSING] = df[COLUMNS_WITH_ZERO_AS_MISSING].replace(0, np.nan)
    df[COLUMNS_WITH_ZERO_AS_MISSING] = df[COLUMNS_WITH_ZERO_AS_MISSING].fillna(
                                       df[COLUMNS_WITH_ZERO_AS_MISSING].median())
    return df

# Split dataframe into X (features) and y (target).
def get_features_and_target(df: pd.DataFrame):
    X = df.drop("Outcome", axis=1)
    y = df["Outcome"]
    return X, y

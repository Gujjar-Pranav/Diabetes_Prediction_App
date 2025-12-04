from pathlib import Path

# Project root = folder where this file lives two levels up
PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"

RAW_DATA_PATH = DATA_DIR / "diabetes.csv"
MODEL_PATH = MODELS_DIR / "diabetes_log_reg.pkl"

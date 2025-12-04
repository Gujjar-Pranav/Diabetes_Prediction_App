from pathlib import Path
# Project root = folder where all files stored for the project
PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"

# Paths for Dataset and Trained Model
RAW_DATA_PATH = DATA_DIR / "diabetes.csv"
MODEL_PATH = MODELS_DIR / "diabetes_log_reg.pkl"

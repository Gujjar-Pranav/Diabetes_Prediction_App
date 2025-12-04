# src/train_model.py
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import joblib

from .data_prep import load_raw_data, clean_data, get_features_and_target
from .config import MODEL_PATH, MODELS_DIR


def train_model():
    """Train logistic regression model and save it to models/diabetes_log_reg.pkl."""
    df_raw = load_raw_data()
    df = clean_data(df_raw)
    X, y = get_features_and_target(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    pipeline = Pipeline(steps=[
        ("scaler", StandardScaler()),
        ("log_reg", LogisticRegression(max_iter=1000, solver="lbfgs")),
    ])

    pipeline.fit(X_train, y_train)

    # Ensure models directory exists, then save
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)
    print(f"✅ Model trained and saved to: {MODEL_PATH}")

    return pipeline, (X_test, y_test, X.columns)


if __name__ == "__main__":
    train_model()

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from .config import MODEL_PATH
from .data_prep import load_raw_data, clean_data, get_features_and_target

from sklearn.metrics import (accuracy_score,precision_score,recall_score,f1_score, roc_auc_score,
                             roc_curve,classification_report,confusion_matrix,)

def evaluate_model():  # Load data and model
    df = clean_data(load_raw_data())
    X, y = get_features_and_target(df)

    model = joblib.load(MODEL_PATH)

    # Re-split to approximate original setup
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=42,stratify=y,)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba)

    print(f"Accuracy  : {acc:.3f}")
    print(f"Precision : {prec:.3f}")
    print(f"Recall    : {rec:.3f}")
    print(f"F1-score  : {f1:.3f}")
    print(f"ROC-AUC   : {roc_auc:.3f}\n")

    print("Classification report:\n")
    print(classification_report(y_test, y_pred))

    print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))
    """
    # ROC curve
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    plt.figure(figsize=(6, 4))
    plt.plot(fpr, tpr, label=f"ROC (AUC = {roc_auc:.3f})")
    plt.plot([0, 1], [0, 1], linestyle="--", label="Random")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve - Logistic Regression")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    """
    # Coefficients / odds ratios
    log_reg = model.named_steps["log_reg"]
    coefficients = log_reg.coef_[0]
    features = X.columns

    coef_df = pd.DataFrame({
        "Feature": features,
        "Coefficient": coefficients,
        "Magnitude (exp(coef))": np.exp(coefficients)
    }).sort_values(by="Coefficient", ascending=False)
    print("\nSorted coefficients:\n")
    print(coef_df.round(3))

if __name__ == "__main__":
    evaluate_model()

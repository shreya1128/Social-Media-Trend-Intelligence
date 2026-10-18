"""
ml_model.py
-----------
Data Science (traditional Machine Learning) component.

Trains and evaluates:
    1. Logistic Regression  (baseline classifier)
    2. Random Forest        (main Data Science model)

Target: trend_label (Low / Moderate / High)

Saves:
    models/random_forest.pkl        -> trained Random Forest (joblib)
    models/logistic_regression.pkl  -> trained Logistic Regression (joblib)
    models/ml_metrics.json          -> evaluation metrics for both models
                                        (consumed by the Model Comparison page)

Run:
    python utils/ml_model.py
"""

import os
import json
import numpy as np
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
)

try:
    from utils.preprocessing import build_processed_dataframe, prepare_model_data, MODEL_FEATURES
except ImportError:  # allow running as a plain script from utils/
    from preprocessing import build_processed_dataframe, prepare_model_data, MODEL_FEATURES

BASE_DIR = os.path.join(os.path.dirname(__file__), "..")
DATASET_PATH = os.path.join(BASE_DIR, "dataset", "social_media_trends.csv")
MODELS_DIR = os.path.join(BASE_DIR, "models")
LABELS = ["Low", "Moderate", "High"]  # class 0, 1, 2


def evaluate(y_true, y_pred, model_name):
    metrics = {
        "model": model_name,
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "precision": round(precision_score(y_true, y_pred, average="macro", zero_division=0), 4),
        "recall": round(recall_score(y_true, y_pred, average="macro", zero_division=0), 4),
        "f1_score": round(f1_score(y_true, y_pred, average="macro", zero_division=0), 4),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1, 2]).tolist(),
    }
    return metrics


def train_and_evaluate():
    os.makedirs(MODELS_DIR, exist_ok=True)

    df = build_processed_dataframe(DATASET_PATH)
    X_train, X_test, y_train, y_test, transformers = prepare_model_data(df, save_dir=MODELS_DIR)

    all_metrics = {}

    # -----------------------------------------------------------------
    # 1. Logistic Regression (baseline)
    # -----------------------------------------------------------------
    # NOTE: recent scikit-learn versions dropped the `multi_class` argument;
    # LogisticRegression now automatically uses a multinomial-equivalent
    # scheme for multi-class problems with the default 'lbfgs' solver.
    log_reg = LogisticRegression(max_iter=1000, random_state=42)
    log_reg.fit(X_train, y_train)
    y_pred_lr = log_reg.predict(X_test)
    lr_metrics = evaluate(y_test, y_pred_lr, "Logistic Regression")
    all_metrics["logistic_regression"] = lr_metrics
    joblib.dump(log_reg, os.path.join(MODELS_DIR, "logistic_regression.pkl"))
    print("Logistic Regression:", lr_metrics)

    # -----------------------------------------------------------------
    # 2. Random Forest (main DS model)
    # -----------------------------------------------------------------
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=12,
        min_samples_leaf=3,
        random_state=42,
        n_jobs=-1,
    )
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)
    rf_metrics = evaluate(y_test, y_pred_rf, "Random Forest")

    # Feature importance
    importances = dict(zip(MODEL_FEATURES, rf.feature_importances_.round(4).tolist()))
    rf_metrics["feature_importance"] = importances

    all_metrics["random_forest"] = rf_metrics
    joblib.dump(rf, os.path.join(MODELS_DIR, "random_forest.pkl"))
    print("Random Forest:", rf_metrics)

    # -----------------------------------------------------------------
    # Save metrics for the Flask Model Comparison page
    # -----------------------------------------------------------------
    metrics_path = os.path.join(MODELS_DIR, "ml_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"\nSaved metrics -> {metrics_path}")

    return all_metrics


if __name__ == "__main__":
    train_and_evaluate()

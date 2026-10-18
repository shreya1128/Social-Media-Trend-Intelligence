"""
dl_model.py
-----------
Deep Learning component: a lightweight Feed-Forward Artificial Neural
Network (MLP) built with TensorFlow/Keras, trained on the SAME
preprocessed features as the Data Science models (utils/ml_model.py),
so the DS vs DL comparison is apples-to-apples.

Architecture:
    Input -> Dense(64, ReLU) -> Dropout(0.3)
          -> Dense(32, ReLU) -> Dropout(0.2)
          -> Dense(3, Softmax)   [Low / Moderate / High]

Loss: sparse_categorical_crossentropy (multi-class, integer labels)
Optimizer: Adam
EarlyStopping on validation loss.

Saves:
    models/neural_network.keras
    models/dl_metrics.json          (accuracy/precision/recall/F1/confusion matrix)
    models/dl_history.json          (per-epoch training/validation accuracy & loss,
                                      used to plot the required graphs)

Runs entirely on CPU (no GPU required) - small dataset, small network.

Run:
    python utils/dl_model.py
"""

import os
import json
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
)

try:
    from utils.preprocessing import build_processed_dataframe, prepare_model_data
except ImportError:
    from preprocessing import build_processed_dataframe, prepare_model_data

BASE_DIR = os.path.join(os.path.dirname(__file__), "..")
DATASET_PATH = os.path.join(BASE_DIR, "dataset", "social_media_trends.csv")
MODELS_DIR = os.path.join(BASE_DIR, "models")

tf.random.set_seed(42)
np.random.seed(42)


def build_model(input_dim: int) -> keras.Model:
    model = keras.Sequential([
        keras.layers.Input(shape=(input_dim,)),
        keras.layers.Dense(64, activation="relu"),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(32, activation="relu"),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(3, activation="softmax"),  # Low / Moderate / High
    ])
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def train_and_evaluate():
    os.makedirs(MODELS_DIR, exist_ok=True)

    df = build_processed_dataframe(DATASET_PATH)
    # Reuse the exact same split/scaling logic as the ML models
    X_train, X_test, y_train, y_test, transformers = prepare_model_data(df, save_dir=None)

    X_train_np = X_train.values.astype("float32")
    X_test_np = X_test.values.astype("float32")
    y_train_np = np.asarray(y_train)
    y_test_np = np.asarray(y_test)

    model = build_model(input_dim=X_train_np.shape[1])
    model.summary()

    early_stop = keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=8, restore_best_weights=True
    )

    history = model.fit(
        X_train_np, y_train_np,
        validation_split=0.15,
        epochs=100,
        batch_size=32,
        callbacks=[early_stop],
        verbose=2,
    )

    # --- Evaluate on held-out test set ---
    y_pred_probs = model.predict(X_test_np, verbose=0)
    y_pred = np.argmax(y_pred_probs, axis=1)

    metrics = {
        "model": "Neural Network",
        "accuracy": round(accuracy_score(y_test_np, y_pred), 4),
        "precision": round(precision_score(y_test_np, y_pred, average="macro", zero_division=0), 4),
        "recall": round(recall_score(y_test_np, y_pred, average="macro", zero_division=0), 4),
        "f1_score": round(f1_score(y_test_np, y_pred, average="macro", zero_division=0), 4),
        "confusion_matrix": confusion_matrix(y_test_np, y_pred, labels=[0, 1, 2]).tolist(),
        "epochs_trained": len(history.history["loss"]),
    }
    print("\nNeural Network test metrics:", metrics)

    # --- Save model + metrics + training history (for accuracy/loss graphs) ---
    model.save(os.path.join(MODELS_DIR, "neural_network.keras"))

    with open(os.path.join(MODELS_DIR, "dl_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    history_dict = {
        "accuracy": [round(float(v), 4) for v in history.history["accuracy"]],
        "val_accuracy": [round(float(v), 4) for v in history.history["val_accuracy"]],
        "loss": [round(float(v), 4) for v in history.history["loss"]],
        "val_loss": [round(float(v), 4) for v in history.history["val_loss"]],
    }
    with open(os.path.join(MODELS_DIR, "dl_history.json"), "w") as f:
        json.dump(history_dict, f, indent=2)

    print(f"Saved model -> {os.path.join(MODELS_DIR, 'neural_network.keras')}")
    print(f"Saved metrics -> {os.path.join(MODELS_DIR, 'dl_metrics.json')}")
    print(f"Saved training history -> {os.path.join(MODELS_DIR, 'dl_history.json')}")

    return metrics, history_dict


if __name__ == "__main__":
    train_and_evaluate()

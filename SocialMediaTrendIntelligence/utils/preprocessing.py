"""
preprocessing.py
-----------------
Central Data Science preprocessing + feature engineering pipeline for the
Cross-Platform Social Media Trend Intelligence & Prediction System.

This module is imported by:
    - notebooks/01_data_preprocessing.ipynb
    - utils/ml_model.py   (Data Science models)
    - utils/dl_model.py   (Deep Learning model)
    - app.py              (Flask prediction endpoint)

so that the SAME cleaning / feature-engineering / encoding / scaling logic
is used everywhere (no train/serve skew).

PROJECT-DEFINED TREND SCORE
----------------------------
Trend Score = 0.4 * normalized(engagement_rate)
            + 0.3 * normalized(growth_rate)
            + 0.3 * normalized(views)          [as a proxy for "reach"]

This is NOT an official social-media industry metric. It is a formula
defined for this academic project to combine engagement, growth and reach
into a single interpretable trend indicator.

Trend Label thresholds (on the 0-100 Trend Score):
    Low Trend      : score <  40
    Moderate Trend : 40 <= score < 70
    High Trend     : score >= 70
"""

import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import joblib

RANDOM_STATE = 42

NUMERIC_FEATURES = [
    "views", "likes", "comments", "shares",
    "growth_rate", "sentiment_score", "engagement_rate",
]
CATEGORICAL_FEATURES = ["platform", "category"]

# Features actually fed into the ML/DL models to predict trend_label.
# NOTE: engagement_rate, growth_rate and views feed into trend_score/trend_label,
# so trend_score itself is EXCLUDED from the model inputs to avoid leakage.
MODEL_FEATURES = [
    "platform", "category",
    "views", "likes", "comments", "shares",
    "growth_rate", "sentiment_score", "engagement_rate",
]
TARGET = "trend_label"


# ---------------------------------------------------------------------------
# 1. LOAD + CLEAN
# ---------------------------------------------------------------------------
def load_raw_dataset(csv_path):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"Dataset not found at '{csv_path}'. Run utils/generate_dataset.py first."
        )
    return pd.read_csv(csv_path)


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Missing values, duplicates, dtypes, outlier handling."""
    df = df.copy()

    # --- Duplicate removal ---
    before = len(df)
    df = df.drop_duplicates(subset=["post_id"]).reset_index(drop=True)
    removed_dupes = before - len(df)

    # --- Data type conversion ---
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    numeric_cols = ["views", "likes", "comments", "shares", "growth_rate", "sentiment_score"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # --- Missing value handling ---
    # sentiment_score / growth_rate: impute with median (robust to outliers)
    for col in ["sentiment_score", "growth_rate"]:
        median_val = df[col].median()
        df[col] = df[col].fillna(median_val)

    # Drop any row still missing a critical field (post_id, platform, views)
    df = df.dropna(subset=["post_id", "platform", "category", "views"]).reset_index(drop=True)

    # --- Outlier handling (views) ---
    # Cap extreme viral outliers at the 99th percentile instead of dropping them,
    # so we keep the record but avoid the model being dominated by a handful of posts.
    cap = df["views"].quantile(0.99)
    df["views"] = np.where(df["views"] > cap, cap, df["views"])

    # Basic sanity clipping
    df["likes"] = df["likes"].clip(lower=0)
    df["comments"] = df["comments"].clip(lower=0)
    df["shares"] = df["shares"].clip(lower=0)
    df["sentiment_score"] = df["sentiment_score"].clip(-1, 1)

    df.attrs["removed_duplicates"] = removed_dupes
    return df


# ---------------------------------------------------------------------------
# 2. FEATURE ENGINEERING
# ---------------------------------------------------------------------------
def add_engagement_rate(df: pd.DataFrame) -> pd.DataFrame:
    """Engagement Rate = ((Likes + Comments + Shares) / Views) * 100, safe div-by-zero."""
    df = df.copy()
    safe_views = df["views"].replace(0, np.nan)
    df["engagement_rate"] = ((df["likes"] + df["comments"] + df["shares"]) / safe_views) * 100
    df["engagement_rate"] = df["engagement_rate"].fillna(0)
    df["engagement_rate"] = df["engagement_rate"].clip(0, 100)
    return df


def add_trend_score(df: pd.DataFrame) -> pd.DataFrame:
    """
    Project-defined Trend Score (0-100):
        0.4 * normalized(engagement_rate)
      + 0.3 * normalized(growth_rate)
      + 0.3 * normalized(views)

    Normalization method: PERCENTILE RANK (0-1), not plain min-max.
    Reason: views/likes/etc. are naturally right-skewed (lognormal-like) with
    a long tail of viral posts. Plain min-max normalization lets a handful of
    extreme outliers compress almost every other post toward 0, producing a
    badly imbalanced Low/Moderate/High split. Percentile-rank normalization
    (each value's rank among all posts, scaled to [0,1]) is robust to that
    skew and produces a well-balanced, more meaningful trend distribution.
    """
    df = df.copy()

    def norm(series):
        return series.rank(pct=True)

    norm_engagement = norm(df["engagement_rate"])
    norm_growth = norm(df["growth_rate"])
    norm_views = norm(df["views"])

    trend_score = (0.4 * norm_engagement + 0.3 * norm_growth + 0.3 * norm_views) * 100
    df["trend_score"] = trend_score.round(2)

    def label(score):
        if score >= 70:
            return "High"
        elif score >= 40:
            return "Moderate"
        return "Low"

    df["trend_label"] = df["trend_score"].apply(label)
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = add_engagement_rate(df)
    df = add_trend_score(df)
    return df


# ---------------------------------------------------------------------------
# 3. FULL PIPELINE: clean -> engineer -> return analysis-ready dataframe
# ---------------------------------------------------------------------------
def build_processed_dataframe(csv_path) -> pd.DataFrame:
    """Used by the Flask app / EDA / dashboard: cleaned + feature-engineered,
    but NOT yet encoded/scaled (still human-readable)."""
    raw = load_raw_dataset(csv_path)
    clean = clean_dataset(raw)
    processed = engineer_features(clean)
    return processed


# ---------------------------------------------------------------------------
# 4. ENCODING + SCALING + TRAIN/TEST SPLIT (for ML/DL model training)
# ---------------------------------------------------------------------------
def prepare_model_data(df: pd.DataFrame, save_dir=None, test_size=0.2):
    """
    Encodes categoricals, scales numerics, and splits into train/test.
    Returns X_train, X_test, y_train, y_test, and the fitted transformers
    (so the SAME transformers can be reused at prediction time -> no leakage,
    no train/serve skew).
    """
    df = df.copy()

    X = df[MODEL_FEATURES].copy()
    y = df[TARGET].copy()

    # --- Categorical encoding ---
    platform_encoder = LabelEncoder()
    category_encoder = LabelEncoder()
    X["platform"] = platform_encoder.fit_transform(X["platform"])
    X["category"] = category_encoder.fit_transform(X["category"])

    # --- Target encoding ---
    label_encoder = LabelEncoder()
    # Fix a deterministic, human-friendly order: Low < Moderate < High
    label_encoder.classes_ = np.array(["Low", "Moderate", "High"])
    y_encoded = y.map({"Low": 0, "Moderate": 1, "High": 2}).values

    # --- Train/test split (BEFORE scaling, to avoid leakage) ---
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=test_size, random_state=RANDOM_STATE, stratify=y_encoded
    )

    # --- Numerical feature scaling (fit on train only) ---
    numeric_cols = ["views", "likes", "comments", "shares",
                     "growth_rate", "sentiment_score", "engagement_rate"]
    scaler = StandardScaler()
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()
    X_train_scaled[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
    X_test_scaled[numeric_cols] = scaler.transform(X_test[numeric_cols])

    transformers = {
        "platform_encoder": platform_encoder,
        "category_encoder": category_encoder,
        "scaler": scaler,
        "numeric_cols": numeric_cols,
        "model_features": MODEL_FEATURES,
        "label_map": {0: "Low", 1: "Moderate", 2: "High"},
    }

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        joblib.dump(transformers, os.path.join(save_dir, "preprocessing.pkl"))

    return X_train_scaled, X_test_scaled, y_train, y_test, transformers


def transform_single_input(record: dict, transformers: dict) -> pd.DataFrame:
    """
    Transforms a single user-submitted prediction-form record (dict) into a
    model-ready row using the SAME fitted encoders/scaler from training.
    `record` keys: platform, category, views, likes, comments, shares,
                    growth_rate, sentiment_score
    """
    row = pd.DataFrame([record])

    # engagement rate is derived, not entered by the user
    safe_views = row["views"].replace(0, np.nan)
    row["engagement_rate"] = ((row["likes"] + row["comments"] + row["shares"]) / safe_views) * 100
    row["engagement_rate"] = row["engagement_rate"].fillna(0).clip(0, 100)

    platform_encoder = transformers["platform_encoder"]
    category_encoder = transformers["category_encoder"]

    # Handle unseen categories gracefully (fallback to the first known class)
    def safe_encode(encoder, value):
        if value in encoder.classes_:
            return encoder.transform([value])[0]
        return 0

    row["platform"] = safe_encode(platform_encoder, row["platform"].iloc[0])
    row["category"] = safe_encode(category_encoder, row["category"].iloc[0])

    row = row[transformers["model_features"]]
    numeric_cols = transformers["numeric_cols"]
    row[numeric_cols] = transformers["scaler"].transform(row[numeric_cols])
    return row


if __name__ == "__main__":
    base = os.path.dirname(__file__)
    csv_path = os.path.join(base, "..", "dataset", "social_media_trends.csv")
    df = build_processed_dataframe(csv_path)
    print("Processed shape:", df.shape)
    print("\nTrend label distribution:\n", df["trend_label"].value_counts())
    print("\nSample rows:\n", df.head(3).to_string())

    X_train, X_test, y_train, y_test, tr = prepare_model_data(
        df, save_dir=os.path.join(base, "..", "models")
    )
    print("\nX_train shape:", X_train.shape, "X_test shape:", X_test.shape)
    print("Class balance (train):", np.bincount(y_train))
    print("Saved preprocessing.pkl ->", os.path.join(base, "..", "models", "preprocessing.pkl"))

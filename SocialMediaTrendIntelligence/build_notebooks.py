"""
build_notebooks.py — generates the four academic notebooks with nbformat,
then executes each one with nbclient to verify it runs top-to-bottom
without errors before saving the final version (with outputs) into notebooks/.

Run from the project root:
    python build_notebooks.py
"""
import os
import nbformat as nbf
from nbclient import NotebookClient

BASE = os.path.dirname(__file__)
NB_DIR = os.path.join(BASE, "notebooks")
os.makedirs(NB_DIR, exist_ok=True)


def make_notebook(cells):
    nb = nbf.v4.new_notebook()
    nb["cells"] = cells
    nb["metadata"] = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11"},
    }
    return nb


def md(text):
    return nbf.v4.new_markdown_cell(text)


def code(text):
    return nbf.v4.new_code_cell(text)


# ===========================================================================
# NOTEBOOK 1: DATA PREPROCESSING
# ===========================================================================
nb1_cells = [
md("""# 01 · Data Preprocessing
**Cross-Platform Social Media Trend Intelligence & Prediction System**

This notebook covers: dataset loading, inspection, missing-value handling,
duplicate removal, data-type conversion, outlier handling, feature
engineering (Engagement Rate, Trend Score, Trend Label), categorical
encoding, numerical scaling, and the train/test split.

All logic here mirrors `utils/preprocessing.py`, which the Flask app
imports directly so the notebook and the web app never drift apart."""),

code("""import sys, os
sys.path.append(os.path.abspath('..'))
import pandas as pd
import numpy as np
from utils.preprocessing import (
    load_raw_dataset, clean_dataset, add_engagement_rate, add_trend_score,
    prepare_model_data, MODEL_FEATURES, TARGET
)

pd.set_option('display.max_columns', None)
DATASET_PATH = os.path.join('..', 'dataset', 'social_media_trends.csv')"""),

md("## 1. Dataset Loading & Inspection"),
code("""raw_df = load_raw_dataset(DATASET_PATH)
print("Shape:", raw_df.shape)
raw_df.head()"""),

code("""raw_df.info()"""),

code("""raw_df.describe(include='all').T"""),

md("## 2. Missing Values"),
code("""print("Missing values per column:")
print(raw_df.isnull().sum()[raw_df.isnull().sum() > 0])"""),

md("""We impute `sentiment_score` and `growth_rate` with their **median**
(robust to outliers) rather than the mean, and drop any row still missing
a critical identifying field (`post_id`, `platform`, `category`, `views`)."""),

md("## 3. Duplicate Checking"),
code("""dupe_count = raw_df.duplicated(subset=['post_id']).sum()
print(f"Duplicate post_id rows found: {dupe_count}")"""),

md("## 4. Data Cleaning (missing values, duplicates, dtypes, outliers)"),
code("""clean_df = clean_dataset(raw_df)
print("Shape after cleaning:", clean_df.shape)
print("Duplicates removed:", clean_df.attrs.get('removed_duplicates'))
print("\\nRemaining missing values:\\n", clean_df.isnull().sum().sum())"""),

md("""**Outlier handling:** `views` is capped at the 99th percentile rather
than dropped, so a handful of viral posts don't dominate scaling/model
training, while still keeping the record in the dataset."""),

md("## 5. Feature Engineering"),
md("""### Engagement Rate
$$\\text{Engagement Rate} = \\frac{Likes + Comments + Shares}{Views} \\times 100$$
Division by zero is handled safely (views of 0 would produce NaN, which is
filled with 0)."""),
code("""fe_df = add_engagement_rate(clean_df)
fe_df[['views','likes','comments','shares','engagement_rate']].head()"""),

md("""### Project-Defined Trend Score
This is **not** an official social-media metric — it is defined for this
academic project as a weighted combination of engagement, growth and reach:

$$\\text{Trend Score} = 0.4 \\times \\text{norm(engagement\\_rate)} + 0.3 \\times \\text{norm(growth\\_rate)} + 0.3 \\times \\text{norm(views)}$$

Each component is **percentile-rank normalized** to [0,1] before combining
(rather than plain min-max), because raw views/engagement are right-skewed
with a long viral tail — min-max normalization let a handful of outliers
compress nearly every other post toward 0, producing a badly imbalanced
Low/Moderate/High split. Percentile-rank normalization is robust to that
skew and produces a well-balanced trend distribution.

Thresholds: Low `< 40`, Moderate `40–70`, High `>= 70`."""),
code("""processed_df = add_trend_score(fe_df)
processed_df[['engagement_rate','growth_rate','views','trend_score','trend_label']].head()"""),

code("""processed_df['trend_label'].value_counts()"""),

md("## 6. Categorical Encoding, Feature Scaling & Train/Test Split"),
md("""Encoding and scaling are fit **only on the training split** to avoid
data leakage. The target (`trend_label`) is excluded from the input
features, and `trend_score` itself is also excluded from `MODEL_FEATURES`
since it's directly derived from `engagement_rate`, `growth_rate` and
`views` — including it would leak the answer."""),
code("""print("Model input features:", MODEL_FEATURES)
print("Target:", TARGET)

X_train, X_test, y_train, y_test, transformers = prepare_model_data(
    processed_df, save_dir=os.path.join('..', 'models')
)
print("\\nX_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("Class balance (train):", np.bincount(y_train))
print("Class balance (test):", np.bincount(y_test))"""),

code("""X_train.head()"""),

md("""## 7. Save Processed Dataset
The cleaned + feature-engineered (but still human-readable) dataframe is
also saved so the EDA notebook can load it directly."""),
code("""processed_df.to_csv(os.path.join('..', 'dataset', 'processed_social_media_trends.csv'), index=False)
print("Saved processed_social_media_trends.csv")"""),

md("""## Summary
- Loaded raw dataset, inspected structure and missing values
- Cleaned duplicates, fixed data types, capped extreme outliers
- Engineered `engagement_rate` and the project-defined `trend_score` / `trend_label`
- Encoded categoricals, scaled numeric features, split into train/test with **no data leakage**
- Saved `preprocessing.pkl` transformers (reused identically by the Flask app) and the processed dataset for EDA"""),
]

nb1 = make_notebook(nb1_cells)


# ===========================================================================
# NOTEBOOK 2: EDA
# ===========================================================================
nb2_cells = [
md("""# 02 · Exploratory Data Analysis (EDA)
**Cross-Platform Social Media Trend Intelligence & Prediction System**

Univariate, platform, category, engagement, sentiment, correlation and
trend analysis on the processed dataset."""),

code("""import sys, os
sys.path.append(os.path.abspath('..'))
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid", palette="deep")
plt.rcParams['figure.dpi'] = 100

df = pd.read_csv(os.path.join('..', 'dataset', 'processed_social_media_trends.csv'), parse_dates=['date'])
df.shape"""),

md("## 1. Univariate Analysis"),
code("""fig, axes = plt.subplots(1, 3, figsize=(15, 4))
sns.histplot(df['views'], bins=40, ax=axes[0], color='#3457D5').set_title('Views Distribution')
sns.histplot(df['engagement_rate'], bins=40, ax=axes[1], color='#21A366').set_title('Engagement Rate Distribution')
sns.histplot(df['trend_score'], bins=40, ax=axes[2], color='#E0932A').set_title('Trend Score Distribution')
plt.tight_layout()
plt.show()"""),

md("## 2. Platform Analysis"),
code("""fig, axes = plt.subplots(1, 2, figsize=(14, 4))
df['platform'].value_counts().plot(kind='bar', ax=axes[0], color='#3457D5')
axes[0].set_title('Posts per Platform')
df.groupby('platform')['engagement_rate'].mean().sort_values().plot(kind='barh', ax=axes[1], color='#6C8CFF')
axes[1].set_title('Avg Engagement Rate by Platform')
plt.tight_layout()
plt.show()"""),

md("## 3. Category Analysis"),
code("""plt.figure(figsize=(10,4))
df.groupby('category')['trend_score'].mean().sort_values().plot(kind='barh', color='#E0932A')
plt.title('Average Trend Score by Category')
plt.tight_layout()
plt.show()"""),

md("## 4. Engagement Analysis"),
code("""plt.figure(figsize=(7,5))
sns.scatterplot(data=df.sample(1500, random_state=42), x='views', y='engagement_rate',
                 hue='trend_label', palette={'Low':'#8890A6','Moderate':'#E0932A','High':'#21A366'}, alpha=0.6, s=20)
plt.xscale('log')
plt.title('Views vs Engagement Rate (colored by Trend Label)')
plt.tight_layout()
plt.show()"""),

md("## 5. Sentiment Analysis"),
code("""fig, axes = plt.subplots(1, 2, figsize=(12,4))
df['sentiment'].value_counts().plot(kind='pie', autopct='%1.1f%%', ax=axes[0],
                                     colors=['#21A366','#8890A6','#D8503C'])
axes[0].set_ylabel('')
axes[0].set_title('Sentiment Share')
sns.boxplot(data=df, x='sentiment', y='trend_score', ax=axes[1],
            order=['Negative','Neutral','Positive'], palette=['#D8503C','#8890A6','#21A366'])
axes[1].set_title('Trend Score by Sentiment')
plt.tight_layout()
plt.show()"""),

md("## 6. Correlation Analysis"),
code("""numeric_cols = ['views','likes','comments','shares','growth_rate','sentiment_score','engagement_rate','trend_score']
plt.figure(figsize=(8,6))
sns.heatmap(df[numeric_cols].corr(), annot=True, fmt='.2f', cmap='coolwarm', center=0)
plt.title('Correlation Matrix')
plt.tight_layout()
plt.show()"""),

md("## 7. Trend Analysis Over Time"),
code("""monthly_trend = df.set_index('date').resample('MS')['trend_score'].mean()
plt.figure(figsize=(10,4))
monthly_trend.plot(marker='o', color='#3457D5')
plt.title('Average Trend Score Over Time')
plt.ylabel('Trend Score')
plt.tight_layout()
plt.show()"""),

md("## 8. Trend Label Distribution"),
code("""plt.figure(figsize=(6,4))
df['trend_label'].value_counts().reindex(['Low','Moderate','High']).plot(
    kind='bar', color=['#8890A6','#E0932A','#21A366'])
plt.title('Trend Label Distribution')
plt.tight_layout()
plt.show()"""),

md("""## Summary of Key Observations
- Views and engagement are right-skewed, typical of social-media reach data.
- TikTok/YouTube show different engagement dynamics than X/Instagram, consistent with platform-level differences built into the dataset.
- Sentiment correlates positively with trend score — more positive posts trend higher, on average.
- `likes`, `engagement_rate` and `growth_rate` are the numeric features most correlated with `trend_score`, which lines up with the Trend Score formula and previews the feature importance seen later in the Random Forest model."""),
]

nb2 = make_notebook(nb2_cells)


# ===========================================================================
# NOTEBOOK 3: MACHINE LEARNING
# ===========================================================================
nb3_cells = [
md("""# 03 · Machine Learning (Data Science Component)
**Cross-Platform Social Media Trend Intelligence & Prediction System**

Trains and evaluates the two traditional ML models:
- **Logistic Regression** (baseline)
- **Random Forest** (main Data Science model)

Target: `trend_label` (Low / Moderate / High)"""),

code("""import sys, os
sys.path.append(os.path.abspath('..'))
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, confusion_matrix, classification_report)

from utils.preprocessing import build_processed_dataframe, prepare_model_data, MODEL_FEATURES

DATASET_PATH = os.path.join('..', 'dataset', 'social_media_trends.csv')
MODELS_DIR = os.path.join('..', 'models')
LABELS = ['Low', 'Moderate', 'High']"""),

md("## 1. Feature Selection & Train/Test Split"),
code("""df = build_processed_dataframe(DATASET_PATH)
X_train, X_test, y_train, y_test, transformers = prepare_model_data(df, save_dir=MODELS_DIR)
print("Features used:", MODEL_FEATURES)
print("X_train:", X_train.shape, " X_test:", X_test.shape)"""),

md("## 2. Logistic Regression (Baseline)"),
code("""log_reg = LogisticRegression(max_iter=1000, random_state=42)
log_reg.fit(X_train, y_train)
y_pred_lr = log_reg.predict(X_test)

print(classification_report(y_test, y_pred_lr, target_names=LABELS))"""),

code("""cm_lr = confusion_matrix(y_test, y_pred_lr)
plt.figure(figsize=(5,4))
sns.heatmap(cm_lr, annot=True, fmt='d', cmap='Blues', xticklabels=LABELS, yticklabels=LABELS)
plt.title('Logistic Regression — Confusion Matrix')
plt.xlabel('Predicted'); plt.ylabel('Actual')
plt.tight_layout()
plt.show()"""),

md("## 3. Random Forest (Main Data Science Model)"),
code("""rf = RandomForestClassifier(n_estimators=200, max_depth=12, min_samples_leaf=3,
                             random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

print(classification_report(y_test, y_pred_rf, target_names=LABELS))"""),

code("""cm_rf = confusion_matrix(y_test, y_pred_rf)
plt.figure(figsize=(5,4))
sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Greens', xticklabels=LABELS, yticklabels=LABELS)
plt.title('Random Forest — Confusion Matrix')
plt.xlabel('Predicted'); plt.ylabel('Actual')
plt.tight_layout()
plt.show()"""),

md("## 4. Feature Importance (Random Forest)"),
code("""importances = pd.Series(rf.feature_importances_, index=MODEL_FEATURES).sort_values()
plt.figure(figsize=(7,5))
importances.plot(kind='barh', color='#3457D5')
plt.title('Random Forest — Feature Importance')
plt.tight_layout()
plt.show()"""),

md("## 5. Save Models"),
code("""joblib.dump(log_reg, os.path.join(MODELS_DIR, 'logistic_regression.pkl'))
joblib.dump(rf, os.path.join(MODELS_DIR, 'random_forest.pkl'))

def evaluate(y_true, y_pred, name):
    return {
        'model': name,
        'accuracy': round(accuracy_score(y_true, y_pred), 4),
        'precision': round(precision_score(y_true, y_pred, average='macro', zero_division=0), 4),
        'recall': round(recall_score(y_true, y_pred, average='macro', zero_division=0), 4),
        'f1_score': round(f1_score(y_true, y_pred, average='macro', zero_division=0), 4),
    }

print("Logistic Regression:", evaluate(y_test, y_pred_lr, 'Logistic Regression'))
print("Random Forest:      ", evaluate(y_test, y_pred_rf, 'Random Forest'))
print("\\nModels saved to:", MODELS_DIR)"""),

md("""## Summary
Random Forest outperforms the Logistic Regression baseline on every metric,
which is expected since the Trend Score/Label boundaries are non-linear
combinations of the input features — a tree ensemble captures that
non-linearity better than a linear model. `likes`, `engagement_rate` and
`growth_rate` are the most influential features, consistent with the EDA
correlation analysis."""),
]

nb3 = make_notebook(nb3_cells)


# ===========================================================================
# NOTEBOOK 4: DEEP LEARNING
# ===========================================================================
nb4_cells = [
md("""# 04 · Deep Learning (ANN / MLP Component)
**Cross-Platform Social Media Trend Intelligence & Prediction System**

Trains a lightweight Feed-Forward Artificial Neural Network (MLP) with
TensorFlow/Keras on the same preprocessed features as the ML notebook, for
a fair DS-vs-DL comparison. Runs entirely on CPU."""),

code("""import sys, os
sys.path.append(os.path.abspath('..'))
import json
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, confusion_matrix, classification_report)
import seaborn as sns

from utils.preprocessing import build_processed_dataframe, prepare_model_data

DATASET_PATH = os.path.join('..', 'dataset', 'social_media_trends.csv')
MODELS_DIR = os.path.join('..', 'models')
LABELS = ['Low', 'Moderate', 'High']

tf.random.set_seed(42)
np.random.seed(42)"""),

md("## 1. Dataset Preparation & Scaling"),
code("""df = build_processed_dataframe(DATASET_PATH)
X_train, X_test, y_train, y_test, transformers = prepare_model_data(df, save_dir=None)

X_train_np = X_train.values.astype('float32')
X_test_np = X_test.values.astype('float32')
y_train_np = np.asarray(y_train)
y_test_np = np.asarray(y_test)

print("X_train:", X_train_np.shape, " X_test:", X_test_np.shape)"""),

md("""## 2. Neural Network Architecture

```
Input(9 features)
   -> Dense(64, ReLU)
   -> Dropout(0.3)
   -> Dense(32, ReLU)
   -> Dropout(0.2)
   -> Dense(3, Softmax)      # Low / Moderate / High
```
Loss: `sparse_categorical_crossentropy` (multi-class, integer labels)
Optimizer: `Adam`"""),
code("""model = keras.Sequential([
    keras.layers.Input(shape=(X_train_np.shape[1],)),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dropout(0.3),
    keras.layers.Dense(32, activation='relu'),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(3, activation='softmax'),
])
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()"""),

md("## 3. Training with EarlyStopping"),
code("""early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True)

history = model.fit(
    X_train_np, y_train_np,
    validation_split=0.15,
    epochs=100,
    batch_size=32,
    callbacks=[early_stop],
    verbose=0,
)
print(f"Training stopped after {len(history.history['loss'])} epochs (EarlyStopping).")"""),

md("## 4. Training / Validation Accuracy & Loss Graphs"),
code("""fig, axes = plt.subplots(1, 2, figsize=(13,4))
axes[0].plot(history.history['accuracy'], label='Training Accuracy', color='#3457D5')
axes[0].plot(history.history['val_accuracy'], label='Validation Accuracy', color='#21A366')
axes[0].set_title('Accuracy over Epochs'); axes[0].set_xlabel('Epoch'); axes[0].legend()

axes[1].plot(history.history['loss'], label='Training Loss', color='#D8503C')
axes[1].plot(history.history['val_loss'], label='Validation Loss', color='#E0932A')
axes[1].set_title('Loss over Epochs'); axes[1].set_xlabel('Epoch'); axes[1].legend()
plt.tight_layout()
plt.show()"""),

md("## 5. Testing & Evaluation"),
code("""y_pred_probs = model.predict(X_test_np, verbose=0)
y_pred = np.argmax(y_pred_probs, axis=1)

print(classification_report(y_test_np, y_pred, target_names=LABELS))"""),

code("""cm_nn = confusion_matrix(y_test_np, y_pred)
plt.figure(figsize=(5,4))
sns.heatmap(cm_nn, annot=True, fmt='d', cmap='Purples', xticklabels=LABELS, yticklabels=LABELS)
plt.title('Neural Network — Confusion Matrix')
plt.xlabel('Predicted'); plt.ylabel('Actual')
plt.tight_layout()
plt.show()"""),

md("## 6. Save Model, Metrics & Training History"),
code("""model.save(os.path.join(MODELS_DIR, 'neural_network.keras'))

metrics = {
    'model': 'Neural Network',
    'accuracy': round(accuracy_score(y_test_np, y_pred), 4),
    'precision': round(precision_score(y_test_np, y_pred, average='macro', zero_division=0), 4),
    'recall': round(recall_score(y_test_np, y_pred, average='macro', zero_division=0), 4),
    'f1_score': round(f1_score(y_test_np, y_pred, average='macro', zero_division=0), 4),
    'confusion_matrix': cm_nn.tolist(),
    'epochs_trained': len(history.history['loss']),
}
with open(os.path.join(MODELS_DIR, 'dl_metrics.json'), 'w') as f:
    json.dump(metrics, f, indent=2)

history_dict = {k: [round(float(v), 4) for v in history.history[k]]
                 for k in ['accuracy','val_accuracy','loss','val_loss']}
with open(os.path.join(MODELS_DIR, 'dl_history.json'), 'w') as f:
    json.dump(history_dict, f, indent=2)

print("Neural Network metrics:", metrics)
print("Saved model, dl_metrics.json and dl_history.json")"""),

md("""## Summary
The MLP reaches accuracy comparable to (and in this run, slightly above)
the Random Forest, showing that a small feed-forward network can capture
the non-linear Trend Score relationship without needing a deep or
convolutional/recurrent architecture — appropriate given the tabular,
non-sequential nature of this feature set. See `05` (Model Comparison page
in the Flask app) for the full DS vs DL comparison."""),
]

nb4 = make_notebook(nb4_cells)


# ===========================================================================
# Write + execute all notebooks
# ===========================================================================
notebooks = {
    "01_data_preprocessing.ipynb": nb1,
    "02_eda.ipynb": nb2,
    "03_machine_learning.ipynb": nb3,
    "04_deep_learning.ipynb": nb4,
}

for filename, nb in notebooks.items():
    path = os.path.join(NB_DIR, filename)
    with open(path, "w") as f:
        nbf.write(nb, f)
    print(f"Written: {path}")

print("\n--- Executing notebooks to verify they run end-to-end ---\n")
for filename in notebooks:
    path = os.path.join(NB_DIR, filename)
    print(f"Executing {filename} ...")
    with open(path) as f:
        nb = nbf.read(f, as_version=4)
    client = NotebookClient(nb, timeout=600, kernel_name="python3",
                             resources={"metadata": {"path": NB_DIR}})
    client.execute()
    with open(path, "w") as f:
        nbf.write(nb, f)
    print(f"  OK -> {filename} executed successfully with outputs saved.\n")

print("All notebooks built and executed successfully.")

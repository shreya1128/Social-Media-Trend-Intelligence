"""
app.py
------
Flask backend for the Cross-Platform Social Media Trend Intelligence &
Prediction System.

Run:
    python app.py
Then open http://127.0.0.1:5000 in a browser.

Pages:
    /login              - authentication
    /dashboard           - overview cards + charts + recent predictions
    /trend-analysis       - filterable trend explorer
    /prediction           - single blended Trend Predictor (see note below)
    /content-strategy      - data-driven posting recommendations
    /logout

Note on /prediction:
    Two models run underneath (Random Forest + Neural Network), but rather
    than exposing them as two separate, competing outputs, their class
    probabilities are blended into a single result - the way a real
    product would present one confident answer rather than making the
    user reconcile two model opinions. The underlying per-model outputs
    are still logged to prediction_history for internal record-keeping,
    they're just not surfaced in the UI.

Note on model evaluation:
    Full technical evaluation (accuracy/precision/recall/F1, confusion
    matrices, feature importance, training curves for both the traditional
    ML models and the neural network) lives in notebooks/03_machine_learning.ipynb
    and notebooks/04_deep_learning.ipynb. This is deliberate: those numbers
    are academic evaluation artifacts for grading, not something a real
    end-user-facing product would display, so the live app stays productised.
"""

import os
import json
import traceback
from functools import wraps

import numpy as np
import pandas as pd
import joblib
from flask import Flask, render_template, request, redirect, url_for, session, flash

from utils.preprocessing import build_processed_dataframe, transform_single_input
from utils.database import (
    init_db, verify_user, save_prediction, get_recent_predictions, DatabaseError
)

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(__file__)
DATASET_PATH = os.path.join(BASE_DIR, "dataset", "social_media_trends.csv")
MODELS_DIR = os.path.join(BASE_DIR, "models")

app = Flask(__name__)
app.secret_key = "mca-academic-project-secret-key-2026"  # fine for a local academic prototype

LABEL_MAP = {0: "Low", 1: "Moderate", 2: "High"}

# ---------------------------------------------------------------------------
# Load dataset + trained models ONCE at startup
# ---------------------------------------------------------------------------
DF = None
RF_MODEL = None
LR_MODEL = None
NN_MODEL = None
TRANSFORMERS = None
ML_METRICS = None
DL_METRICS = None
DL_HISTORY = None
LOAD_ERROR = None


def load_everything():
    global DF, RF_MODEL, LR_MODEL, NN_MODEL, TRANSFORMERS
    global ML_METRICS, DL_METRICS, DL_HISTORY, LOAD_ERROR
    try:
        if not os.path.exists(DATASET_PATH):
            raise FileNotFoundError(
                "Dataset missing. Run: python utils/generate_dataset.py"
            )
        DF = build_processed_dataframe(DATASET_PATH)

        required_models = ["random_forest.pkl", "logistic_regression.pkl",
                            "neural_network.keras", "preprocessing.pkl"]
        missing = [m for m in required_models if not os.path.exists(os.path.join(MODELS_DIR, m))]
        if missing:
            raise FileNotFoundError(
                f"Missing trained model file(s): {missing}. "
                "Run: python utils/ml_model.py  and  python utils/dl_model.py"
            )

        RF_MODEL = joblib.load(os.path.join(MODELS_DIR, "random_forest.pkl"))
        LR_MODEL = joblib.load(os.path.join(MODELS_DIR, "logistic_regression.pkl"))
        TRANSFORMERS = joblib.load(os.path.join(MODELS_DIR, "preprocessing.pkl"))

        # TensorFlow import deferred here so the rest of the app still works
        # (with a friendly error) if TF is somehow unavailable.
        from tensorflow import keras
        NN_MODEL = keras.models.load_model(os.path.join(MODELS_DIR, "neural_network.keras"))

        with open(os.path.join(MODELS_DIR, "ml_metrics.json")) as f:
            ML_METRICS = json.load(f)
        with open(os.path.join(MODELS_DIR, "dl_metrics.json")) as f:
            DL_METRICS = json.load(f)
        with open(os.path.join(MODELS_DIR, "dl_history.json")) as f:
            DL_HISTORY = json.load(f)

        LOAD_ERROR = None
    except Exception as e:
        LOAD_ERROR = str(e)
        print("STARTUP LOAD ERROR:", LOAD_ERROR)


load_everything()
init_db()


# ---------------------------------------------------------------------------
# Auth helper
# ---------------------------------------------------------------------------
def login_required(view_func):
    @wraps(view_func)
    def wrapped(*args, **kwargs):
        if "username" not in session:
            return redirect(url_for("login"))
        return view_func(*args, **kwargs)
    return wrapped


def system_ready():
    return LOAD_ERROR is None


# ---------------------------------------------------------------------------
# ROUTES: AUTH
# ---------------------------------------------------------------------------
@app.route("/", methods=["GET"])
def index():
    return redirect(url_for("dashboard") if "username" in session else url_for("login"))


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")
        if not username or not password:
            flash("Please enter both username and password.", "danger")
            return render_template("login.html")
        try:
            if verify_user(username, password):
                session["username"] = username
                return redirect(url_for("dashboard"))
            else:
                flash("Invalid username or password.", "danger")
        except DatabaseError as de:
            flash(str(de), "danger")
        except Exception:
            flash("Login is temporarily unavailable. Please try again.", "danger")
    return render_template("login.html")


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))


# ---------------------------------------------------------------------------
# ROUTES: DASHBOARD
# ---------------------------------------------------------------------------
@app.route("/dashboard")
@login_required
def dashboard():
    if not system_ready():
        return render_template("dashboard.html", active_page="dashboard", error=LOAD_ERROR)

    total_posts = len(DF)
    total_platforms = DF["platform"].nunique()
    high_trend_posts = int((DF["trend_label"] == "High").sum())
    avg_engagement = round(DF["engagement_rate"].mean(), 2)

    platform_counts = DF["platform"].value_counts()
    avg_engagement_by_platform = DF.groupby("platform")["engagement_rate"].mean().round(2)
    trend_dist = DF["trend_label"].value_counts().reindex(["Low", "Moderate", "High"]).fillna(0)
    sentiment_dist = DF["sentiment"].value_counts().reindex(["Positive", "Neutral", "Negative"]).fillna(0)

    chart_data = {
        "platform_labels": platform_counts.index.tolist(),
        "platform_values": platform_counts.values.tolist(),
        "engagement_platform_labels": avg_engagement_by_platform.index.tolist(),
        "engagement_platform_values": avg_engagement_by_platform.values.tolist(),
        "trend_labels": trend_dist.index.tolist(),
        "trend_values": trend_dist.values.tolist(),
        "sentiment_labels": sentiment_dist.index.tolist(),
        "sentiment_values": sentiment_dist.values.tolist(),
    }

    recent_predictions = get_recent_predictions(limit=8)

    return render_template(
        "dashboard.html",
        active_page="dashboard",
        total_posts=total_posts,
        total_platforms=total_platforms,
        high_trend_posts=high_trend_posts,
        avg_engagement=avg_engagement,
        chart_data=json.dumps(chart_data),
        recent_predictions=recent_predictions,
        error=None,
    )


# ---------------------------------------------------------------------------
# ROUTES: TREND ANALYSIS
# ---------------------------------------------------------------------------
@app.route("/trend-analysis")
@login_required
def trend_analysis():
    if not system_ready():
        return render_template("trend_analysis.html", active_page="trend_analysis", error=LOAD_ERROR)

    platform = request.args.get("platform", "")
    category = request.args.get("category", "")
    topic = request.args.get("topic", "")
    date_from = request.args.get("date_from", "")
    date_to = request.args.get("date_to", "")
    page = max(int(request.args.get("page", 1)), 1)
    per_page = 15

    filtered = DF.copy()
    if platform:
        filtered = filtered[filtered["platform"] == platform]
    if category:
        filtered = filtered[filtered["category"] == category]
    if topic:
        filtered = filtered[filtered["topic"].str.contains(topic, case=False, na=False)]
    if date_from:
        filtered = filtered[filtered["date"] >= pd.to_datetime(date_from)]
    if date_to:
        filtered = filtered[filtered["date"] <= pd.to_datetime(date_to)]

    summary = {
        "num_posts": len(filtered),
        "avg_views": round(filtered["views"].mean(), 0) if len(filtered) else 0,
        "avg_likes": round(filtered["likes"].mean(), 0) if len(filtered) else 0,
        "avg_comments": round(filtered["comments"].mean(), 0) if len(filtered) else 0,
        "avg_shares": round(filtered["shares"].mean(), 0) if len(filtered) else 0,
        "avg_engagement_rate": round(filtered["engagement_rate"].mean(), 2) if len(filtered) else 0,
        "avg_trend_score": round(filtered["trend_score"].mean(), 2) if len(filtered) else 0,
    }

    # Charts (computed on the filtered subset)
    platform_engagement = filtered.groupby("platform")["engagement_rate"].mean().round(2)
    trend_score_bins = pd.cut(
        filtered["trend_score"], bins=[0, 20, 40, 60, 80, 100],
        labels=["0-20", "20-40", "40-60", "60-80", "80-100"], include_lowest=True
    ).value_counts().sort_index()
    sentiment_dist = filtered["sentiment"].value_counts().reindex(
        ["Positive", "Neutral", "Negative"]).fillna(0)
    trend_over_time = (
        filtered.set_index("date").resample("MS")["trend_score"].mean().round(2).dropna()
        if len(filtered) else pd.Series(dtype=float)
    )
    top_topics = filtered["topic"].value_counts().head(8)

    chart_data = {
        "platform_engagement_labels": platform_engagement.index.tolist(),
        "platform_engagement_values": platform_engagement.values.tolist(),
        "trend_score_bin_labels": trend_score_bins.index.astype(str).tolist(),
        "trend_score_bin_values": trend_score_bins.values.tolist(),
        "sentiment_labels": sentiment_dist.index.tolist(),
        "sentiment_values": sentiment_dist.values.tolist(),
        "time_labels": [d.strftime("%b %Y") for d in trend_over_time.index],
        "time_values": trend_over_time.values.tolist(),
        "topic_labels": top_topics.index.tolist(),
        "topic_values": top_topics.values.tolist(),
    }

    # Pagination for the records table
    total_records = len(filtered)
    total_pages = max((total_records + per_page - 1) // per_page, 1)
    page = min(page, total_pages)
    start = (page - 1) * per_page
    page_df = filtered.sort_values("date", ascending=False).iloc[start:start + per_page]
    records = page_df[[
        "post_id", "platform", "category", "topic", "date",
        "views", "likes", "comments", "shares",
        "engagement_rate", "trend_score", "trend_label"
    ]].copy()
    records["date"] = records["date"].dt.strftime("%Y-%m-%d")
    records = records.to_dict(orient="records")

    filter_options = {
        "platforms": sorted(DF["platform"].unique().tolist()),
        "categories": sorted(DF["category"].unique().tolist()),
    }

    return render_template(
        "trend_analysis.html",
        active_page="trend_analysis",
        summary=summary,
        chart_data=json.dumps(chart_data),
        records=records,
        page=page,
        total_pages=total_pages,
        filter_options=filter_options,
        selected={
            "platform": platform, "category": category, "topic": topic,
            "date_from": date_from, "date_to": date_to,
        },
        error=None,
    )


# ---------------------------------------------------------------------------
# ROUTES: PREDICTION
# ---------------------------------------------------------------------------
@app.route("/prediction", methods=["GET", "POST"])
@login_required
def prediction():
    if not system_ready():
        return render_template("prediction.html", active_page="prediction", error=LOAD_ERROR, result=None)

    filter_options = {
        "platforms": sorted(DF["platform"].unique().tolist()),
        "categories": sorted(DF["category"].unique().tolist()),
    }

    result = None
    form_values = {}

    if request.method == "POST":
        try:
            form_values = {
                "platform": request.form.get("platform", "").strip(),
                "category": request.form.get("category", "").strip(),
                "views": request.form.get("views", "").strip(),
                "likes": request.form.get("likes", "").strip(),
                "comments": request.form.get("comments", "").strip(),
                "shares": request.form.get("shares", "").strip(),
                "growth_rate": request.form.get("growth_rate", "").strip(),
                "sentiment_score": request.form.get("sentiment_score", "").strip(),
            }

            # --- Validation ---
            if not form_values["platform"] or not form_values["category"]:
                raise ValueError("Please select both platform and category.")

            try:
                views = float(form_values["views"])
                likes = float(form_values["likes"])
                comments = float(form_values["comments"])
                shares = float(form_values["shares"])
                growth_rate = float(form_values["growth_rate"])
                sentiment_score = float(form_values["sentiment_score"])
            except ValueError:
                raise ValueError("All numeric fields must contain valid numbers.")

            if views <= 0:
                raise ValueError("Views must be greater than zero.")
            if likes < 0 or comments < 0 or shares < 0:
                raise ValueError("Likes, comments and shares cannot be negative.")
            if not (-1 <= sentiment_score <= 1):
                raise ValueError("Sentiment score must be between -1 and 1.")

            record = {
                "platform": form_values["platform"],
                "category": form_values["category"],
                "views": views, "likes": likes, "comments": comments, "shares": shares,
                "growth_rate": growth_rate, "sentiment_score": sentiment_score,
            }

            model_input = transform_single_input(record, TRANSFORMERS)

            # --- Run both models internally ---
            rf_pred = RF_MODEL.predict(model_input)[0]
            rf_probs = RF_MODEL.predict_proba(model_input)[0]
            ds_label = LABEL_MAP[int(rf_pred)]
            ds_confidence = round(float(np.max(rf_probs)) * 100, 2)

            nn_probs = NN_MODEL.predict(model_input.values.astype("float32"), verbose=0)[0]
            nn_pred = int(np.argmax(nn_probs))
            dl_label = LABEL_MAP[nn_pred]
            dl_confidence = round(float(np.max(nn_probs)) * 100, 2)

            # --- Blend into ONE product-facing result ---
            # Unweighted average of the two models' class-probability
            # vectors, then take the top class. This is how many real
            # systems quietly combine several internal signals into a
            # single answer, instead of showing the user two competing
            # model outputs.
            blended_probs = (np.asarray(rf_probs) + np.asarray(nn_probs)) / 2.0
            final_class = int(np.argmax(blended_probs))
            final_label = LABEL_MAP[final_class]
            final_confidence = round(float(np.max(blended_probs)) * 100, 2)

            prob_breakdown = {
                LABEL_MAP[i]: round(float(blended_probs[i]) * 100, 2) for i in range(3)
            }

            # --- "Why" explanation: which of this post's metrics stand out ---
            engagement_rate_val = ((likes + comments + shares) / views) * 100 if views else 0
            feature_values = {
                "views": views, "likes": likes, "comments": comments, "shares": shares,
                "growth_rate": growth_rate, "sentiment_score": sentiment_score,
                "engagement_rate": engagement_rate_val,
            }
            FEATURE_LABELS = {
                "views": "Views", "likes": "Likes", "comments": "Comments", "shares": "Shares",
                "growth_rate": "Growth Rate", "sentiment_score": "Sentiment",
                "engagement_rate": "Engagement Rate",
            }
            importance_source = ML_METRICS.get("random_forest", {}).get("feature_importance", {})
            ranked_features = sorted(
                ((f, imp) for f, imp in importance_source.items() if f in feature_values),
                key=lambda x: x[1], reverse=True,
            )[:3]

            factors = []
            for feat, _imp in ranked_features:
                series = DF[feat]
                val = feature_values[feat]
                p25, p75 = series.quantile(0.25), series.quantile(0.75)
                if val >= p75:
                    note = "well above typical for similar posts"
                elif val <= p25:
                    note = "below typical for similar posts"
                else:
                    note = "around typical for similar posts"
                factors.append({"label": FEATURE_LABELS[feat], "note": note})

            result = {
                "label": final_label,
                "confidence": final_confidence,
                "probs": prob_breakdown,
                "factors": factors,
            }

            try:
                save_prediction({
                    "platform": record["platform"], "category": record["category"],
                    "views": views, "likes": likes, "comments": comments, "shares": shares,
                    "growth_rate": growth_rate, "sentiment_score": sentiment_score,
                    "ds_prediction": ds_label, "ds_confidence": ds_confidence,
                    "dl_prediction": dl_label, "dl_confidence": dl_confidence,
                    "final_prediction": final_label, "final_confidence": final_confidence,
                })
            except DatabaseError:
                # The prediction itself succeeded and is still shown below;
                # only the history log write failed, which is non-fatal.
                flash("Prediction generated, but it could not be saved to the history log.", "warning")

        except ValueError as ve:
            flash(str(ve), "danger")
        except Exception:
            traceback.print_exc()
            flash("Something went wrong while generating the prediction. Please check your inputs and try again.", "danger")

    return render_template(
        "prediction.html", active_page="prediction", filter_options=filter_options,
        result=result, form_values=form_values, error=None,
    )


# ---------------------------------------------------------------------------
# ROUTES: CONTENT STRATEGY  (data-driven posting recommendations)
# ---------------------------------------------------------------------------
MIN_SAMPLE_SIZE = 20  # ignore combinations with too few posts to be reliable


@app.route("/content-strategy")
@login_required
def content_strategy():
    if not system_ready():
        return render_template("content_strategy.html", active_page="content_strategy", error=LOAD_ERROR)

    platform = request.args.get("platform", "")
    category = request.args.get("category", "")

    df = DF.copy()
    df["day_of_week"] = df["date"].dt.day_name()

    # --- Best day of week to post (by average trend score) ---
    day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    by_day = df.groupby("day_of_week")["trend_score"].mean().reindex(day_order)
    best_day = by_day.idxmax()

    # --- Top platform + category combinations ---
    combo_stats = (
        df.groupby(["platform", "category"])
        .agg(avg_trend_score=("trend_score", "mean"), avg_engagement=("engagement_rate", "mean"), posts=("post_id", "count"))
        .reset_index()
    )
    combo_stats = combo_stats[combo_stats["posts"] >= MIN_SAMPLE_SIZE]
    top_combos = combo_stats.sort_values("avg_trend_score", ascending=False).head(8)

    # --- Top topics ---
    topic_stats = (
        df.groupby("topic")
        .agg(avg_trend_score=("trend_score", "mean"), posts=("post_id", "count"))
        .reset_index()
    )
    topic_stats = topic_stats[topic_stats["posts"] >= MIN_SAMPLE_SIZE]
    top_topics = topic_stats.sort_values("avg_trend_score", ascending=False).head(8)

    # --- Sentiment impact ---
    sentiment_avg = df.groupby("sentiment")["trend_score"].mean()
    positive_avg = sentiment_avg.get("Positive", 0)
    negative_avg = sentiment_avg.get("Negative", 0)
    sentiment_lift = round(((positive_avg - negative_avg) / negative_avg) * 100, 1) if negative_avg else 0

    # --- Personalized recommendation (if platform + category selected) ---
    personalized = None
    if platform and category:
        subset = df[(df["platform"] == platform) & (df["category"] == category)]
        if len(subset) >= MIN_SAMPLE_SIZE:
            subset_by_day = subset.groupby("day_of_week")["trend_score"].mean().reindex(day_order).dropna()
            personalized = {
                "platform": platform, "category": category,
                "sample_size": len(subset),
                "avg_trend_score": round(subset["trend_score"].mean(), 1),
                "avg_engagement": round(subset["engagement_rate"].mean(), 2),
                "best_day": subset_by_day.idxmax() if len(subset_by_day) else best_day,
            }
        else:
            personalized = {"insufficient_data": True, "platform": platform, "category": category, "sample_size": len(subset)}

    chart_data = {
        "day_labels": day_order,
        "day_values": by_day.round(2).fillna(0).tolist(),
        "combo_labels": [f"{r.platform} · {r.category}" for r in top_combos.itertuples()],
        "combo_values": top_combos["avg_trend_score"].round(2).tolist(),
        "topic_labels": top_topics["topic"].tolist(),
        "topic_values": top_topics["avg_trend_score"].round(2).tolist(),
    }

    filter_options = {
        "platforms": sorted(DF["platform"].unique().tolist()),
        "categories": sorted(DF["category"].unique().tolist()),
    }

    return render_template(
        "content_strategy.html",
        active_page="content_strategy",
        best_day=best_day,
        sentiment_lift=sentiment_lift,
        top_combos=top_combos.to_dict(orient="records"),
        top_topics=top_topics.to_dict(orient="records"),
        chart_data=json.dumps(chart_data),
        personalized=personalized,
        filter_options=filter_options,
        selected={"platform": platform, "category": category},
        error=None,
    )


# ---------------------------------------------------------------------------
# ERROR HANDLERS
# ---------------------------------------------------------------------------
@app.errorhandler(404)
def not_found_error(e):
    return render_template("login.html", error=None), 404


@app.errorhandler(500)
def server_error(e):
    return "An unexpected error occurred. Please restart the application.", 500


if __name__ == "__main__":
    app.run(debug=False, host="127.0.0.1", port=5000)

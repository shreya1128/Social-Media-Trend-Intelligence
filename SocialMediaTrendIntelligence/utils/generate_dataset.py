"""
generate_dataset.py
--------------------
Generates a realistic SYNTHETIC social-media dataset for the
Cross-Platform Social Media Trend Intelligence & Prediction System.

This is NOT real social-media data. It is statistically simulated so that
views/likes/comments/shares/growth/sentiment are correlated in a believable
way, giving the ML/DL models genuine patterns to learn instead of noise.

Run:
    python utils/generate_dataset.py
Output:
    dataset/social_media_trends.csv
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import os

# Reproducibility
RNG = np.random.default_rng(42)

N_RECORDS = 8000

PLATFORMS = ["Instagram", "YouTube", "X", "TikTok"]
# Different platforms have different typical reach / engagement behaviour
PLATFORM_PROFILE = {
    "Instagram": {"base_views": 9000,  "view_sigma": 0.9, "engage_mult": 1.10},
    "YouTube":   {"base_views": 15000, "view_sigma": 1.1, "engage_mult": 0.85},
    "X":         {"base_views": 6000,  "view_sigma": 1.0, "engage_mult": 0.95},
    "TikTok":    {"base_views": 20000, "view_sigma": 1.2, "engage_mult": 1.30},
}

CATEGORIES = ["Technology", "Entertainment", "Sports", "Fashion",
              "Gaming", "News", "Education", "Lifestyle"]

TOPICS_BY_CATEGORY = {
    "Technology":    ["AI Tools", "Smartphones", "Gadget Review", "Coding Tips", "Startups"],
    "Entertainment": ["Movie Trailer", "Celebrity News", "Music Release", "TV Shows", "Awards"],
    "Sports":        ["Cricket Match", "Football League", "Olympics", "Athlete Update", "Tournament"],
    "Fashion":       ["Street Style", "Runway Show", "Outfit Ideas", "Designer Launch", "Trends"],
    "Gaming":        ["Game Launch", "Esports", "Game Review", "Live Stream", "Patch Update"],
    "News":          ["Breaking News", "Politics", "Weather Alert", "World Affairs", "Local News"],
    "Education":     ["Study Tips", "Online Course", "Exam Prep", "Career Advice", "Tutorial"],
    "Lifestyle":     ["Travel Vlog", "Food Recipe", "Fitness Routine", "Home Decor", "Daily Vlog"],
}

START_DATE = datetime(2024, 1, 1)
END_DATE = datetime(2025, 12, 31)
DATE_RANGE_DAYS = (END_DATE - START_DATE).days


def random_dates(n):
    offsets = RNG.integers(0, DATE_RANGE_DAYS, size=n)
    return [START_DATE + timedelta(days=int(o)) for o in offsets]


def generate():
    platforms = RNG.choice(PLATFORMS, size=N_RECORDS)
    categories = RNG.choice(CATEGORIES, size=N_RECORDS)
    topics = [RNG.choice(TOPICS_BY_CATEGORY[c]) for c in categories]
    dates = random_dates(N_RECORDS)

    # --- Underlying "quality" latent factor per post (drives everything) ---
    # This simulates that some posts are just inherently more engaging/viral.
    post_quality = RNG.beta(2, 5, size=N_RECORDS)  # skewed toward lower/mid quality, long tail

    views = np.zeros(N_RECORDS)
    engage_mult = np.zeros(N_RECORDS)
    for i, p in enumerate(platforms):
        profile = PLATFORM_PROFILE[p]
        base = profile["base_views"]
        sigma = profile["view_sigma"]
        # Lognormal views influenced by post quality (higher quality -> higher mean views)
        mean_log = np.log(base) + post_quality[i] * 1.8
        views[i] = RNG.lognormal(mean=mean_log, sigma=sigma * 0.35)
        engage_mult[i] = profile["engage_mult"]

    views = np.clip(views, 50, None).round().astype(int)

    # --- Sentiment: correlated with post quality (better content -> more positive) ---
    sentiment_score = np.clip(
        RNG.normal(loc=(post_quality - 0.35) * 1.4, scale=0.35), -1, 1
    )

    def sentiment_label(s):
        if s > 0.15:
            return "Positive"
        elif s < -0.15:
            return "Negative"
        return "Neutral"

    sentiment = [sentiment_label(s) for s in sentiment_score]

    # --- Engagement actions correlated with quality, sentiment, and platform ---
    base_engage_rate = 0.02 + post_quality * 0.18 * engage_mult
    base_engage_rate += np.clip(sentiment_score, 0, None) * 0.05  # positive sentiment boosts engagement
    noise = RNG.normal(0, 0.01, size=N_RECORDS)
    engage_rate_latent = np.clip(base_engage_rate + noise, 0.005, 0.6)

    likes = (views * engage_rate_latent * RNG.uniform(0.65, 0.85, N_RECORDS)).round().astype(int)
    comments = (views * engage_rate_latent * RNG.uniform(0.05, 0.15, N_RECORDS)).round().astype(int)
    shares = (views * engage_rate_latent * RNG.uniform(0.08, 0.20, N_RECORDS)).round().astype(int)

    likes = np.clip(likes, 0, None)
    comments = np.clip(comments, 0, None)
    shares = np.clip(shares, 0, None)

    # --- Growth rate: correlated with post quality + sentiment, with noise ---
    growth_rate = np.clip(
        (post_quality * 40) + (sentiment_score * 8) + RNG.normal(0, 6, N_RECORDS),
        -20, 100
    ).round(2)

    df = pd.DataFrame({
        "post_id": [f"P{100000+i}" for i in range(N_RECORDS)],
        "platform": platforms,
        "date": [d.strftime("%Y-%m-%d") for d in dates],
        "topic": topics,
        "category": categories,
        "views": views,
        "likes": likes,
        "comments": comments,
        "shares": shares,
        "growth_rate": growth_rate,
        "sentiment": sentiment,
        "sentiment_score": sentiment_score.round(3),
    })

    # Inject a small, realistic amount of messiness for the preprocessing stage to handle
    dup_idx = RNG.choice(df.index, size=40, replace=False)
    df = pd.concat([df, df.loc[dup_idx]], ignore_index=True)  # duplicates

    missing_idx = RNG.choice(df.index, size=60, replace=False)
    df.loc[missing_idx, "sentiment_score"] = np.nan  # missing values

    missing_idx2 = RNG.choice(df.index, size=30, replace=False)
    df.loc[missing_idx2, "growth_rate"] = np.nan

    # A few extreme outliers in views (viral posts / data glitches)
    outlier_idx = RNG.choice(df.index, size=15, replace=False)
    df.loc[outlier_idx, "views"] = df.loc[outlier_idx, "views"] * RNG.integers(15, 40, size=15)

    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    return df


if __name__ == "__main__":
    df = generate()
    os.makedirs(os.path.join(os.path.dirname(__file__), "..", "dataset"), exist_ok=True)
    out_path = os.path.join(os.path.dirname(__file__), "..", "dataset", "social_media_trends.csv")
    df.to_csv(out_path, index=False)
    print(f"Generated {len(df)} records -> {out_path}")
    print(df.describe(include='all').T)

"""
sentiment.py
------------
Lightweight sentiment-analysis component using VADER (Valence Aware
Dictionary and sEntiment Reasoner).

NOTE ON THIS PROJECT'S DATASET:
The synthetic dataset (dataset/social_media_trends.csv) is generated
statistically rather than scraped from real captions/comments, so it
already ships with a 'sentiment' label and 'sentiment_score' column
that stand in for what a VADER analysis of real post text would produce.

This module is provided so the sentiment-analysis component is a real,
working part of the pipeline (not just static data): it can score any
piece of text (e.g. a post caption/title typed in by a user) and is used
by the EDA notebook to demonstrate the technique, and is available for
ad-hoc scoring from the Flask app if free-text is ever supplied.
"""

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

_analyzer = SentimentIntensityAnalyzer()


def analyze_sentiment(text: str) -> dict:
    """
    Returns a dict: {label: 'Positive'|'Neutral'|'Negative', score: float in [-1,1]}
    Uses VADER's compound score, thresholded the standard way:
        score >  0.15  -> Positive
        score < -0.15  -> Negative
        else           -> Neutral
    """
    if not text or not isinstance(text, str):
        return {"label": "Neutral", "score": 0.0}

    scores = _analyzer.polarity_scores(text)
    compound = scores["compound"]

    if compound > 0.15:
        label = "Positive"
    elif compound < -0.15:
        label = "Negative"
    else:
        label = "Neutral"

    return {"label": label, "score": round(compound, 3)}


if __name__ == "__main__":
    samples = [
        "This new AI tool is absolutely amazing and saved me so much time!",
        "The match was okay, nothing special.",
        "Worst update ever, the app keeps crashing and I hate it.",
    ]
    for s in samples:
        print(s, "->", analyze_sentiment(s))

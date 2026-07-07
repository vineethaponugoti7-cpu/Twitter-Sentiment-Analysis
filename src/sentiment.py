"""
sentiment.py
------------
Sentiment analysis with two approaches:

1. VADER  - a rule-based analyzer (no training). Fast baseline.
2. A trained TF-IDF + Logistic Regression model, loaded from disk.

Both classify text as 'positive', 'negative', or 'neutral'.
"""

import re

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

_analyzer = SentimentIntensityAnalyzer()


def clean_text(text):
    """Remove @mentions, links, and non-letters; lowercase."""
    text = re.sub(r"@\w+", "", str(text))
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    return text.lower().strip()


def vader_sentiment(text):
    """Rule-based sentiment using VADER's compound score."""
    score = _analyzer.polarity_scores(str(text))["compound"]
    if score >= 0.05:
        return "positive"
    if score <= -0.05:
        return "negative"
    return "neutral"


def model_sentiment(texts, model, vectorizer):
    """
    Trained-model sentiment for a list of texts.
    Returns a list of predicted labels.
    """
    cleaned = [clean_text(t) for t in texts]
    features = vectorizer.transform(cleaned)
    return list(model.predict(features))

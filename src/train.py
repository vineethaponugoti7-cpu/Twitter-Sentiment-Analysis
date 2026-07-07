"""
train.py
--------
Trains the sentiment classifier on the labeled airline-tweets dataset and
saves the model + vectorizer to disk. Also prints the VADER baseline so you
can see how much the trained model improves over the rule-based approach.

Usage:
    python src/train.py
"""

import os
import pickle
import re

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

from sentiment import clean_text, vader_sentiment

DATA_PATH = "data/tweets.csv"
MODEL_DIR = "models"


def main():
    df = pd.read_csv(DATA_PATH)[["text", "airline_sentiment"]].dropna()
    df["clean"] = df["text"].apply(clean_text)

    # Hold out 20% as an unseen test set so accuracy is measured honestly.
    X_train, X_test, y_train, y_test = train_test_split(
        df["clean"], df["airline_sentiment"],
        test_size=0.2, random_state=42, stratify=df["airline_sentiment"],
    )
    print(f"Training on {len(X_train)} tweets, testing on {len(X_test)} unseen tweets.\n")

    # --- Baseline: VADER (no training) ---
    vader_preds = X_test.apply(vader_sentiment)
    vader_acc = accuracy_score(y_test, vader_preds)
    print(f"[BASELINE] VADER accuracy:                  {vader_acc*100:.1f}%")

    # --- Trained model: TF-IDF features + Logistic Regression ---
    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), stop_words="english")
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    model = LogisticRegression(max_iter=1000, class_weight="balanced")
    model.fit(X_train_vec, y_train)

    model_preds = model.predict(X_test_vec)
    model_acc = accuracy_score(y_test, model_preds)
    print(f"[TRAINED]  TF-IDF + Logistic Regression:     {model_acc*100:.1f}%")
    print(f"\nImprovement: +{(model_acc - vader_acc)*100:.1f} percentage points\n")
    print("Detailed report for the trained model:")
    print(classification_report(y_test, model_preds))

    # Save the model and vectorizer so the app can load them.
    os.makedirs(MODEL_DIR, exist_ok=True)
    with open(os.path.join(MODEL_DIR, "model.pkl"), "wb") as f:
        pickle.dump(model, f)
    with open(os.path.join(MODEL_DIR, "vectorizer.pkl"), "wb") as f:
        pickle.dump(vectorizer, f)
    print(f"Saved model and vectorizer to '{MODEL_DIR}/'.")


if __name__ == "__main__":
    main()

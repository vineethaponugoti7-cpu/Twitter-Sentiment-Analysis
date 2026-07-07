"""
app.py
------
Streamlit dashboard for Twitter sentiment analysis.

- Type any text and see how both VADER (rule-based) and the trained model
  classify it.
- View the sentiment distribution of the airline-tweets dataset.
- See the accuracy comparison between the baseline and trained model.

Run:
    streamlit run app.py
"""

import os
import pickle

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from src.sentiment import vader_sentiment, clean_text

st.set_page_config(page_title="Twitter Sentiment Analysis", page_icon="📊")

DATA_PATH = "data/tweets.csv"
MODEL_PATH = "models/model.pkl"
VECTORIZER_PATH = "models/vectorizer.pkl"


@st.cache_resource
def load_model():
    """Load the trained model and vectorizer, if they exist."""
    if os.path.exists(MODEL_PATH) and os.path.exists(VECTORIZER_PATH):
        with open(MODEL_PATH, "rb") as f:
            model = pickle.load(f)
        with open(VECTORIZER_PATH, "rb") as f:
            vectorizer = pickle.load(f)
        return model, vectorizer
    return None, None


@st.cache_data
def load_data():
    if os.path.exists(DATA_PATH):
        return pd.read_csv(DATA_PATH)
    return None


model, vectorizer = load_model()
data = load_data()

st.title("📊 Twitter Sentiment Analysis")
st.caption("Compare a rule-based analyzer (VADER) with a trained ML model.")

# ---------------------------------------------------------------------------
# Section 1: Try it on your own text
# ---------------------------------------------------------------------------
st.header("Try it yourself")
text = st.text_area("Enter a tweet or any sentence:", value="My flight was delayed for 3 hours and no one helped.")

if st.button("Analyze"):
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("VADER (rule-based)")
        v = vader_sentiment(text)
        st.markdown(f"### {v.capitalize()}")

    with col2:
        st.subheader("Trained model")
        if model is not None:
            features = vectorizer.transform([clean_text(text)])
            m = model.predict(features)[0]
            st.markdown(f"### {m.capitalize()}")
        else:
            st.info("Train the model first: `python src/train.py`")

# ---------------------------------------------------------------------------
# Section 2: Dataset overview
# ---------------------------------------------------------------------------
if data is not None:
    st.header("Dataset overview")
    st.write(f"This project uses **{len(data):,} labeled airline tweets**.")

    counts = data["airline_sentiment"].value_counts()
    fig, ax = plt.subplots(figsize=(6, 3))
    colors = {"negative": "#e74c3c", "neutral": "#95a5a6", "positive": "#2ecc71"}
    ax.bar(counts.index, counts.values, color=[colors.get(c, "#3498db") for c in counts.index])
    ax.set_ylabel("Number of tweets")
    ax.set_title("Sentiment distribution in the dataset")
    st.pyplot(fig)

    st.caption(
        "Most airline tweets are negative — this class imbalance is part of what "
        "makes the dataset challenging for a simple rule-based approach."
    )

# ---------------------------------------------------------------------------
# Section 3: Model comparison
# ---------------------------------------------------------------------------
st.header("Model comparison")
st.markdown(
    """
| Approach | Accuracy | How it works |
|---|---|---|
| **VADER** (baseline) | ~56% | Fixed sentiment dictionary + rules, no training |
| **TF-IDF + Logistic Regression** | ~75% | Learns sentiment patterns from the training tweets |

The trained model improves on the baseline by about **19 percentage points**,
because it learns from the data that words like *delayed* and *cancelled* are
strongly negative in this airline context — something the rule-based analyzer
can't pick up on its own.
"""
)

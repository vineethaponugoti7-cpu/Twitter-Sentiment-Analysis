# Twitter Sentiment Analysis

Sentiment analysis on airline tweets, comparing a rule-based analyzer (VADER) against a trained machine-learning model. The project shows the full workflow a real ML task follows: establish a baseline, then train a model that beats it, and measure both honestly on held-out data.

Built with Python, scikit-learn, VADER, and Streamlit.

## Demo

Type any sentence and see how both approaches classify it, view the dataset's sentiment distribution, and compare model accuracy.

*(Add a screenshot here once you've run the app — save it as `Assets/demo.png` and it will show up.)*

## What it does

- **Interactive analysis** — enter any text and see the sentiment predicted by both VADER and the trained model, side by side.
- **Baseline vs. trained comparison** — VADER (no training) reaches about 56% accuracy; a TF-IDF + Logistic Regression model reaches about 75%.
- **Dataset insight** — visualises the sentiment distribution of 14,600 labeled airline tweets.

## Results

Measured on a held-out test set of 2,928 tweets the models never saw during training:

| Approach | Accuracy | How it works |
|---|---|---|
| VADER (baseline) | ~56% | Fixed sentiment dictionary and rules, no training |
| TF-IDF + Logistic Regression | ~75% | Learns sentiment patterns from the training tweets |

The trained model improves on the baseline by about **19 percentage points**. It learns from the data that words like *delayed* and *cancelled* are strongly negative in an airline context — something the rule-based analyzer cannot infer on its own. Negative tweets are classified most accurately; neutral tweets are the hardest, since they carry the weakest signal.

## Dataset

The [Twitter US Airline Sentiment dataset](https://www.kaggle.com/datasets/crowdflower/twitter-airline-sentiment) (~14,600 tweets from February 2015), where each tweet is labeled by humans as positive, negative, or neutral. Because most tweets are complaints, the data is heavily skewed toward negative — a realistic challenge for sentiment models.

## Project structure

```
twitter-sentiment/
├── app.py                # Streamlit dashboard
├── src/
│   ├── sentiment.py      # VADER + trained-model sentiment logic
│   └── train.py          # trains the model, prints baseline vs trained
├── data/
│   └── tweets.csv        # the labeled dataset
├── models/               # saved model + vectorizer (created by train.py)
├── requirements.txt
└── README.md
```

## Setup

```bash
git clone https://github.com/vineethaponugoti7-cpu/twitter-sentiment.git
cd twitter-sentiment
pip install -r requirements.txt
```

## Usage

Train the model (prints the baseline-vs-trained comparison and saves the model):

```bash
python src/train.py
```

Then launch the dashboard:

```bash
streamlit run app.py
```

## How it works

1. **Cleaning** — tweets are stripped of @mentions, links, and punctuation, then lowercased.
2. **Baseline (VADER)** — a rule-based analyzer scores each tweet using a sentiment dictionary. Fast, but it can't adapt to the airline context.
3. **Features (TF-IDF)** — each tweet is turned into numbers reflecting which informative words it contains (common words are down-weighted, distinctive ones up-weighted).
4. **Trained model (Logistic Regression)** — learns from the training tweets which words signal each sentiment, then predicts on unseen test tweets.
5. **Honest evaluation** — accuracy is measured on a held-out 20% test split, so the reported numbers reflect how the models perform on tweets they never saw.

## Note on the original approach

An earlier version of this project fetched live tweets through the Twitter API. Because Twitter's API access changed and the free tier no longer supports this reliably, the project now uses a public labeled dataset instead — which has the added benefit of letting the model's accuracy be measured against human labels.

## Tech stack

Python, scikit-learn (TF-IDF, Logistic Regression), vaderSentiment, pandas, matplotlib, Streamlit.

## Limitations & future work

- Neutral tweets are the hardest to classify; more training data or a stronger model (e.g. a transformer) would help.
- The dataset is from 2015 and airline-specific, so the model reflects that domain.
- Possible extensions: try other classifiers, add cross-validation, or fine-tune a small transformer model for higher accuracy.

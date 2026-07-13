import pandas as pd
from transformers import pipeline

_FINBERT_MODEL = "ProsusAI/finbert"
_classifier = None


def _get_classifier():
    global _classifier
    if _classifier is None:
        _classifier = pipeline("text-classification", model=_FINBERT_MODEL, top_k=None)
    return _classifier


def score_headlines(headlines: list) -> pd.DataFrame:
    columns = ["headline", "positive", "negative", "neutral", "sentiment_score"]
    if not headlines:
        return pd.DataFrame(columns=columns)
    classifier = _get_classifier()
    results = classifier(list(headlines), batch_size=16, truncation=True)
    rows = []
    for headline, scores in zip(headlines, results):
        score_map = {s["label"]: s["score"] for s in scores}
        positive = score_map.get("positive", 0.0)
        negative = score_map.get("negative", 0.0)
        neutral = score_map.get("neutral", 0.0)
        rows.append({
            "headline": headline,
            "positive": positive,
            "negative": negative,
            "neutral": neutral,
            "sentiment_score": positive - negative,
        })
    return pd.DataFrame(rows, columns=columns)

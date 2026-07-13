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


def aggregate_daily_sentiment(scored_news: pd.DataFrame, effective_day: pd.Series) -> pd.DataFrame:
    df = scored_news.copy()
    df["effective_trading_day"] = effective_day.values
    grouped = df.groupby("effective_trading_day")["sentiment_score"]
    daily = grouped.agg(sentiment_mean="mean", sentiment_std="std", headline_count="count")
    daily.index.name = "date"
    return daily


def add_rolling_sentiment(daily_sentiment: pd.DataFrame, trading_days: pd.DatetimeIndex, windows=(3, 7)) -> pd.DataFrame:
    calendar = pd.DatetimeIndex(
        sorted(pd.DatetimeIndex(trading_days).tz_localize(None).normalize().unique())
    )
    reindexed = daily_sentiment.reindex(calendar)
    reindexed["headline_count"] = reindexed["headline_count"].fillna(0)
    for w in windows:
        reindexed[f"sentiment_roll{w}"] = reindexed["sentiment_mean"].rolling(window=w, min_periods=1).mean()
    return reindexed

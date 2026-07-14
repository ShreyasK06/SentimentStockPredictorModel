import pandas as pd

from src.features.technical import compute_technical_features
from src.features.trading_day import assign_effective_trading_day
from src.features.sentiment import aggregate_daily_sentiment, add_rolling_sentiment
from src.features.labels import generate_labels


def build_ticker_feature_matrix(price_df: pd.DataFrame, scored_news: pd.DataFrame, trading_days: pd.DatetimeIndex) -> pd.DataFrame:
    ticker = price_df["ticker"].iloc[0]
    technical = compute_technical_features(price_df)

    # technical.index comes from price_df.index, which per Phase 1's schema
    # is tz-naive. Normalize trading_days the same way before comparing, so
    # a tz-aware trading_days calendar doesn't produce a spurious mismatch.
    trading_days_idx = pd.DatetimeIndex(trading_days)
    if trading_days_idx.tz is not None:
        trading_days_idx = trading_days_idx.tz_localize(None)
    misaligned = technical.index.difference(trading_days_idx)
    if len(misaligned) > 0:
        print(
            f"WARNING: {ticker} has {len(misaligned)} price date(s) not present in the "
            f"trading_days calendar passed to build_ticker_feature_matrix - their sentiment "
            f"will be zero-filled rather than computed. First few: {list(misaligned[:5])}"
        )

    if not scored_news.empty:
        effective_day = assign_effective_trading_day(scored_news["published_at"], trading_days)
        daily_sentiment = aggregate_daily_sentiment(scored_news, effective_day)
    else:
        daily_sentiment = pd.DataFrame(columns=["sentiment_mean", "sentiment_std", "headline_count"])
    sentiment = add_rolling_sentiment(daily_sentiment, trading_days)

    labels = generate_labels(price_df["Close"])

    matrix = technical.join(sentiment, how="left").join(labels, how="left")
    matrix["ticker"] = ticker

    # No-news trading days produce NaN sentiment features (headline_count is
    # already filled to 0 by add_rolling_sentiment). Treat "no news" as a
    # neutral (0.0) sentiment signal so the final matrix has no NaN in any
    # feature column - headline_count == 0 still lets downstream consumers
    # distinguish "genuinely neutral news" from "no news at all".
    sentiment_cols = ["sentiment_mean", "sentiment_std", "sentiment_roll3", "sentiment_roll7"]
    matrix[sentiment_cols] = matrix[sentiment_cols].fillna(0.0)

    matrix = matrix.dropna(subset=technical.columns)
    matrix = matrix.dropna(subset=["label"])
    return matrix


def build_feature_matrix(price_panel: pd.DataFrame, scored_news_by_ticker: dict, trading_days: pd.DatetimeIndex) -> pd.DataFrame:
    empty_news = pd.DataFrame(columns=["headline", "ticker", "sentiment_score", "published_at"])
    frames = []
    for ticker, ticker_price_df in price_panel.groupby("ticker"):
        scored_news = scored_news_by_ticker.get(ticker, empty_news)
        frames.append(build_ticker_feature_matrix(ticker_price_df, scored_news, trading_days))
    return pd.concat(frames)

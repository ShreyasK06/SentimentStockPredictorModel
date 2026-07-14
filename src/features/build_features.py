import pandas as pd

from src.features.technical import compute_technical_features
from src.features.trading_day import assign_effective_trading_day
from src.features.sentiment import aggregate_daily_sentiment, add_rolling_sentiment
from src.features.labels import generate_labels


def build_ticker_feature_matrix(price_df: pd.DataFrame, scored_news: pd.DataFrame, trading_days: pd.DatetimeIndex) -> pd.DataFrame:
    ticker = price_df["ticker"].iloc[0]
    technical = compute_technical_features(price_df)

    if not scored_news.empty:
        effective_day = assign_effective_trading_day(scored_news["published_at"], trading_days)
        daily_sentiment = aggregate_daily_sentiment(scored_news, effective_day)
    else:
        daily_sentiment = pd.DataFrame(columns=["sentiment_mean", "sentiment_std", "headline_count"])
    sentiment = add_rolling_sentiment(daily_sentiment, trading_days)

    labels = generate_labels(price_df["Close"])

    matrix = technical.join(sentiment, how="left").join(labels, how="left")
    matrix["ticker"] = ticker
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

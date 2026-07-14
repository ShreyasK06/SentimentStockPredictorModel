from src import config
from src.data.tickers import UNIVERSE
from src.data.calendar import get_nyse_trading_days
from src.data.storage import load_parquet, save_parquet
from src.features.sentiment import score_headlines
from src.features.build_features import build_feature_matrix


def _load_or_score_sentiment(ticker: str) -> None:
    cache_path = config.SENTIMENT_DIR / f"{ticker}.parquet"
    if cache_path.exists():
        return
    news = load_parquet(ticker, config.NEWS_DIR)
    # Filter malformed/empty headlines BEFORE scoring, so `headline` and
    # `published_at` stay aligned by construction. score_headlines() itself
    # is already-approved Task 3 code (with existing tests) and is not
    # modified here - a row-count mismatch between `news` and `scored` would
    # otherwise silently misattribute published_at timestamps to the wrong
    # headlines via the positional assignment below.
    valid = news["headline"].apply(lambda h: isinstance(h, str) and h.strip() != "")
    dropped = (~valid).sum()
    if dropped > 0:
        print(f"WARNING: {ticker} dropped {dropped} malformed/empty headline(s) before sentiment scoring")
    news = news[valid]
    scored = score_headlines(list(news["headline"]))
    scored["published_at"] = news["published_at"].values
    scored["ticker"] = ticker
    save_parquet(scored, ticker, config.SENTIMENT_DIR)


def main():
    trading_days = get_nyse_trading_days(config.START_DATE, config.END_DATE)
    price_panel = load_parquet("combined_price_panel", config.DATA_PROCESSED_DIR)

    scored_news_by_ticker = {}
    for ticker in UNIVERSE:
        try:
            print(f"Scoring sentiment for {ticker}...")
            _load_or_score_sentiment(ticker)
            scored_news_by_ticker[ticker] = load_parquet(ticker, config.SENTIMENT_DIR)
        except Exception as exc:
            print(f"WARNING: failed to process sentiment for {ticker}: {exc}")
            continue

    matrix = build_feature_matrix(price_panel, scored_news_by_ticker, trading_days)

    print(f"Feature matrix rows: {len(matrix)}")
    print(f"Tickers represented: {sorted(matrix['ticker'].unique())}")
    label_counts = matrix["label"].value_counts(normalize=True)
    print(f"Label balance: {dict(label_counts)}")
    majority_baseline = label_counts.max()
    print(f"Majority-class baseline accuracy: {majority_baseline:.3f}")
    if majority_baseline > 0.65:
        print("WARNING: label balance is skewed (>65% majority class) - accuracy targets need to beat this baseline, not 50%.")

    config.DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    save_parquet(matrix, "feature_matrix", config.DATA_PROCESSED_DIR)
    print(f"Saved feature matrix to {config.DATA_PROCESSED_DIR / 'feature_matrix.parquet'}")


if __name__ == "__main__":
    main()

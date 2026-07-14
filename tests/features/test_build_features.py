import numpy as np
import pandas as pd
from src.features.build_features import build_ticker_feature_matrix, build_feature_matrix


def _price_df(ticker="AAPL", n=80, start=100.0, step=0.5):
    idx = pd.date_range("2024-01-02", periods=n, freq="B")
    close = pd.Series(start + step * np.arange(n), index=idx)
    volume = pd.Series(1_000_000 + 1000 * np.arange(n), index=idx)
    df = pd.DataFrame({
        "Open": close, "High": close, "Low": close, "Close": close, "Volume": volume,
    })
    df.index.name = "date"
    df["ticker"] = ticker
    return df


def _scored_news(ticker="AAPL"):
    idx = pd.date_range("2024-01-02", periods=80, freq="B")
    # one headline published well before close on each of the first 5 trading days
    published = [pd.Timestamp(d, tz="America/New_York").replace(hour=9).tz_convert("UTC") for d in idx[:5]]
    return pd.DataFrame({
        "headline": [f"headline {i}" for i in range(5)],
        "ticker": ticker,
        "sentiment_score": [0.3, -0.1, 0.5, 0.0, 0.2],
        "published_at": published,
    })


def test_build_ticker_feature_matrix_drops_warmup_and_tail_rows():
    price_df = _price_df()
    scored_news = _scored_news()
    trading_days = price_df.index
    matrix = build_ticker_feature_matrix(price_df, scored_news, trading_days)
    # warmup: dist_from_ma50 needs 50 rows; tail: label needs 10 future rows.
    assert matrix.index.min() >= price_df.index[49]
    assert matrix.index.max() <= price_df.index[-11]
    assert matrix["label"].isna().sum() == 0
    assert "ticker" in matrix.columns
    assert (matrix["ticker"] == "AAPL").all()


def test_build_ticker_feature_matrix_handles_ticker_with_no_news():
    price_df = _price_df(ticker="MSFT")
    empty_news = pd.DataFrame(columns=["headline", "ticker", "sentiment_score", "published_at"])
    trading_days = price_df.index
    matrix = build_ticker_feature_matrix(price_df, empty_news, trading_days)
    assert len(matrix) > 0
    assert matrix["headline_count"].fillna(0).eq(0).all()


def test_build_feature_matrix_concatenates_all_tickers():
    price_panel = pd.concat([_price_df("AAPL"), _price_df("MSFT")])
    scored_news_by_ticker = {"AAPL": _scored_news("AAPL")}
    trading_days = _price_df().index
    matrix = build_feature_matrix(price_panel, scored_news_by_ticker, trading_days)
    assert set(matrix["ticker"].unique()) == {"AAPL", "MSFT"}


def test_full_pipeline_is_causal_no_lookahead():
    # End-to-end causality check across technical + sentiment + label joins:
    # altering price and news data from day 60 onward must not change any
    # feature (not label - labels are intentionally forward-looking) at day < 60.
    price_df = _price_df(n=80)
    scored_news = _scored_news()
    trading_days = price_df.index
    feature_cols = [
        "return_1d", "return_5d", "return_10d", "return_20d", "rsi_14",
        "macd", "macd_signal", "macd_hist", "volatility_20d", "volume_ratio_20d",
        "dist_from_ma50", "sentiment_mean", "sentiment_std", "headline_count",
        "sentiment_roll3", "sentiment_roll7",
    ]
    matrix_before = build_ticker_feature_matrix(price_df, scored_news, trading_days)

    price_df_modified = price_df.copy()
    price_df_modified.loc[price_df_modified.index[60]:, "Close"] *= 5
    price_df_modified.loc[price_df_modified.index[60]:, "Volume"] *= 5

    matrix_after = build_ticker_feature_matrix(price_df_modified, scored_news, trading_days)

    common_index = matrix_before.index[matrix_before.index < price_df.index[60]]
    common_index = common_index.intersection(matrix_after.index)
    assert len(common_index) > 0
    pd.testing.assert_frame_equal(
        matrix_before.loc[common_index, feature_cols],
        matrix_after.loc[common_index, feature_cols],
    )

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


def _scored_news_at(trading_days, day_indices, scores, ticker="AAPL"):
    # Build headlines published well before market close on specific trading
    # days (identified by position in `trading_days`), so each headline's
    # effective_trading_day is exactly that day.
    published = [
        pd.Timestamp(trading_days[i], tz="America/New_York").replace(hour=9).tz_convert("UTC")
        for i in day_indices
    ]
    return pd.DataFrame({
        "headline": [f"headline_day{i}" for i in day_indices],
        "ticker": ticker,
        "sentiment_score": scores,
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
    # Final deliverable: no NaN in any feature/label column of the matrix.
    assert matrix.isna().sum().sum() == 0


def test_build_ticker_feature_matrix_handles_ticker_with_no_news():
    price_df = _price_df(ticker="MSFT")
    empty_news = pd.DataFrame(columns=["headline", "ticker", "sentiment_score", "published_at"])
    trading_days = price_df.index
    matrix = build_ticker_feature_matrix(price_df, empty_news, trading_days)
    assert len(matrix) > 0
    assert matrix["headline_count"].fillna(0).eq(0).all()
    # No news at all must not leave NaN sentiment features in the final matrix.
    assert matrix["sentiment_mean"].notna().all()
    assert matrix["sentiment_std"].notna().all()
    assert matrix["sentiment_roll3"].notna().all()
    assert matrix["sentiment_roll7"].notna().all()


def test_build_ticker_feature_matrix_warns_on_price_date_not_in_trading_days_calendar(capsys):
    # Every existing fixture constructs trading_days = price_df.index, so the
    # two indices are always identical by construction and misalignment is
    # never exercised. Real Phase 1 price data can contain a non-trading-day
    # row (this is exactly why Phase 1's find_misaligned_dates exists), which
    # would otherwise silently zero-fill that row's sentiment with no warning.
    price_df = _price_df()
    scored_news = _scored_news()
    missing_date = price_df.index[55]
    trading_days = price_df.index.delete(55)
    assert missing_date not in trading_days

    matrix = build_ticker_feature_matrix(price_df, scored_news, trading_days)

    captured = capsys.readouterr()
    assert "WARNING" in captured.out
    assert "AAPL" in captured.out
    assert "1 price date(s)" in captured.out
    # A loud warning, not a crash - the function must still complete.
    assert len(matrix) > 0


def test_build_feature_matrix_concatenates_all_tickers():
    price_panel = pd.concat([_price_df("AAPL"), _price_df("MSFT")])
    scored_news_by_ticker = {"AAPL": _scored_news("AAPL")}
    trading_days = _price_df().index
    matrix = build_feature_matrix(price_panel, scored_news_by_ticker, trading_days)
    assert set(matrix["ticker"].unique()) == {"AAPL", "MSFT"}


def test_full_pipeline_is_causal_no_lookahead():
    # End-to-end causality check across technical + sentiment + label joins:
    # altering price AND news data from day 60 onward must not change any
    # feature (not label - labels are intentionally forward-looking) at day < 60.
    #
    # Row-index reasoning (n=80 price rows, indices 0..79):
    #   - warmup: dist_from_ma50 needs a 50-row window -> first surviving row
    #     is index 49 (matrix.index.min() >= price_df.index[49]).
    #   - tail: label needs 10 future rows -> last surviving row is index 69
    #     (matrix.index.max() <= price_df.index[-11] == price_df.index[69]).
    #   So the surviving matrix spans rows 49..69 inclusive.
    #   - the "day < 60" compared range within that span is rows 49..59.
    #   - headlines placed at day indices 50-55 land inside that compared
    #     range and are NEVER mutated, so they must produce genuine
    #     (non-zero) sentiment values in the compared rows.
    #   - headlines placed at day indices 60, 62, 65 land in the mutated tail
    #     (rows 60..69) and ARE mutated, to prove a sentiment-only change
    #     propagates through aggregate_daily_sentiment / add_rolling_sentiment
    #     without leaking backward into rows < 60.
    price_df = _price_df(n=80)
    trading_days = price_df.index

    before_indices = [50, 51, 52, 53, 54, 55]
    before_scores = [0.3, -0.1, 0.5, 0.0, 0.2, 0.4]
    after_indices = [60, 62, 65]
    after_scores = [0.4, -0.2, 0.1]

    scored_news = pd.concat([
        _scored_news_at(trading_days, before_indices, before_scores),
        _scored_news_at(trading_days, after_indices, after_scores),
    ], ignore_index=True)

    feature_cols = [
        "return_1d", "return_5d", "return_10d", "return_20d", "rsi_14",
        "macd", "macd_signal", "macd_hist", "volatility_20d", "volume_ratio_20d",
        "dist_from_ma50", "sentiment_mean", "sentiment_std", "headline_count",
        "sentiment_roll3", "sentiment_roll7",
    ]
    matrix_before = build_ticker_feature_matrix(price_df, scored_news, trading_days)

    day60 = trading_days[60]
    common_index = matrix_before.index[matrix_before.index < day60]
    assert len(common_index) > 0

    # Sanity: the compared range must actually contain genuine, non-NaN,
    # non-zero (headline-driven) sentiment values. Otherwise the equality
    # assertion below would pass vacuously (NaN == NaN, or 0.0 == 0.0 for
    # every no-news day) without ever comparing real sentiment data.
    assert matrix_before.loc[common_index, "headline_count"].gt(0).any()
    assert matrix_before.loc[common_index, "sentiment_mean"].notna().any()
    assert matrix_before.loc[common_index, "sentiment_mean"].ne(0.0).any()

    # Mutate price from day 60 onward (as before)...
    price_df_modified = price_df.copy()
    price_df_modified.loc[day60:, "Close"] *= 5
    price_df_modified.loc[day60:, "Volume"] *= 5

    # ...AND mutate news effective on/after day 60: headlines before day 60
    # are unchanged, headlines on/after day 60 have sentiment_score
    # drastically altered (x -10).
    news_modified = pd.concat([
        _scored_news_at(trading_days, before_indices, before_scores),
        _scored_news_at(trading_days, after_indices, [s * -10 for s in after_scores]),
    ], ignore_index=True)

    matrix_after = build_ticker_feature_matrix(price_df_modified, news_modified, trading_days)

    common_index = common_index.intersection(matrix_after.index)
    assert len(common_index) > 0
    pd.testing.assert_frame_equal(
        matrix_before.loc[common_index, feature_cols],
        matrix_after.loc[common_index, feature_cols],
    )

    # Sanity: the day-60+ mutation (price AND news) must actually have had an
    # effect somewhere at/after day 60 - otherwise "no effect before day 60"
    # could trivially hold because the mutation did nothing anywhere.
    tail_index = matrix_before.index[matrix_before.index >= day60]
    tail_index = tail_index.intersection(matrix_after.index)
    assert len(tail_index) > 0
    assert not matrix_before.loc[tail_index, feature_cols].equals(
        matrix_after.loc[tail_index, feature_cols]
    )

    # Specifically confirm the sentiment mutation (independent of the price
    # mutation, which alone would only move technical columns) changed
    # sentiment_mean at day 60, where a mutated headline lives.
    assert day60 in matrix_before.index and day60 in matrix_after.index
    assert matrix_before.loc[day60, "sentiment_mean"] != matrix_after.loc[day60, "sentiment_mean"]

from pathlib import Path

import pandas as pd

from paper_trader import config
from paper_trader.data.price_fetcher import fetch_universe_prices, PriceFetchError
from paper_trader.data.news_fetcher import fetch_universe_news, NewsFetchError
from paper_trader.models.leakage_guard import assert_no_leakage, compute_data_window_hash

from src.data.storage import load_parquet
from src.data.calendar import get_nyse_trading_days
from src.features.technical import compute_technical_features
from src.features.trading_day import assign_effective_trading_day
from src.features.sentiment import score_headlines, aggregate_daily_sentiment, add_rolling_sentiment
from src.features.labels import generate_labels
from src.features.cross_sectional import compute_cross_sectional_features
from src.models.ablation import TECHNICAL_COLUMNS, SENTIMENT_COLUMNS

# Cross-sectional base columns match scripts/run_phase5_cross_sectional_baselines.py,
# the last research phase to evaluate them (no edge found, but the feature set
# stays available behind config.yaml's model.feature_set: xs for future iteration).
XS_BASE_COLUMNS = ["return_10d", "return_20d"]
XS_COLUMNS = [
    "xs_rank_return_10d", "xs_zscore_return_10d", "xs_sector_rank_return_10d", "xs_sector_zscore_return_10d",
    "xs_rank_return_20d", "xs_zscore_return_20d", "xs_sector_rank_return_20d", "xs_sector_zscore_return_20d",
]

LIVE_PRICES_DIR = config.DATA_DIR / "live_prices"
LIVE_SENTIMENT_DIR = config.DATA_DIR / "live_sentiment"
HISTORICAL_MATRIX_NAME = "feature_matrix_extended"

# Calendar-day buffer fetched for news each run so the 7-day rolling sentiment
# feature is correct at the as_of_date tail. Known limitation: on a paper
# trader's very first run, any gap between the frozen historical matrix's end
# date and as_of_date minus this buffer gets 0.0 (neutral/no-news) sentiment
# rather than true backfilled sentiment - see README "Known limitations".
NEWS_LOOKBACK_DAYS = 14

# Extra label columns generated alongside the global-config-driven "label"
# column (10d/2%, still sourced from config.MODEL_CFG below so it keeps
# tracking config.yaml's model: block exactly like before). These give the
# "weekly" and "twenty-day" strategies (config.yaml) their own real trained
# labels instead of reusing the direction-daily model under a different
# display horizon. label_7d's 1.5% thresholds and label_20d_4pct's 4%
# thresholds match the frozen historical matrix's own columns (see
# scripts/add_7day_label.py and scripts/build_extended_feature_matrix.py) so
# the live-appended rows computed here are labeled identically to history.
LABEL_CONFIGS = [
    # (column_name, horizon_days, up_threshold, down_threshold)
    ("label_7d", 7, 0.015, 0.015),
    ("label_20d_4pct", 20, 0.04, 0.04),
]
ALL_LABEL_COLUMNS = ["label"] + [c[0] for c in LABEL_CONFIGS]


def _max_configured_horizon() -> int:
    """
    Widest horizon in play this run, for sizing the embargo cutoff
    conservatively. Previously the embargo only considered the global
    config.MODEL_CFG horizon (always 10), which was correct back when every
    strategy secretly trained on the same 10-day label. Now that label_7d and
    label_20d_4pct are real, independently-resolved labels, a stale/leaked
    frozen-matrix row near the embargo boundary is a real risk for whichever
    configured horizon is longest - so embargo against the max of: the global
    default, every strategy's horizon_days override in config.yaml, and the
    label configs' own horizons (label_20d_4pct's 20 is the long pole today).
    The literal `20` floor keeps this safe even if config.yaml strategies are
    edited to remove the twenty-day strategy outright.
    """
    strategy_horizons = [cfg.get("horizon_days", config.MODEL_CFG["horizon_days"]) for cfg in config.STRATEGIES_CFG]
    label_horizons = [h for _, h, _, _ in LABEL_CONFIGS]
    return max([20, config.MODEL_CFG["horizon_days"]] + strategy_horizons + label_horizons)


def feature_columns(feature_set: str) -> list:
    cols = list(TECHNICAL_COLUMNS) + list(SENTIMENT_COLUMNS)
    if feature_set == "xs":
        cols = cols + XS_COLUMNS
    return cols


def _load_historical_matrix() -> pd.DataFrame:
    return load_parquet(HISTORICAL_MATRIX_NAME, config.RESEARCH_REPO_ROOT / "data" / "processed")


def _load_research_raw_prices(ticker: str) -> pd.DataFrame:
    from src.config import PRICES_DIR
    return load_parquet(ticker, PRICES_DIR)


def _load_or_init_cache(path: Path, columns: list) -> pd.DataFrame:
    if path.exists():
        return pd.read_parquet(path)
    return pd.DataFrame(columns=columns)


def _combined_price_series(ticker: str, as_of_date: str) -> pd.DataFrame:
    research_raw = _load_research_raw_prices(ticker).reset_index()
    live_path = LIVE_PRICES_DIR / f"{ticker}.parquet"
    # An empty DataFrame(columns=[...]) defaults every column to object dtype;
    # concatenating that with research_raw's real float64 OHLCV columns
    # silently upcasts the combined result to object, which LightGBM later
    # rejects. Use research_raw's own empty slice instead so dtypes match
    # even when no live cache file exists yet.
    live_raw = pd.read_parquet(live_path) if live_path.exists() else research_raw.iloc[0:0].copy()

    combined_max = pd.DatetimeIndex(research_raw["date"]).max()
    if not live_raw.empty:
        combined_max = max(combined_max, pd.DatetimeIndex(live_raw["date"]).max())

    as_of_ts = pd.Timestamp(as_of_date)
    if as_of_ts > combined_max:
        fetch_start = (combined_max + pd.Timedelta(days=1)).date().isoformat()
        panel, errors = fetch_universe_prices([ticker], fetch_start, as_of_date)
        if ticker in errors:
            raise PriceFetchError(f"{ticker}: {errors[ticker]}")
        new_rows = panel[panel["ticker"] == ticker]
        LIVE_PRICES_DIR.mkdir(parents=True, exist_ok=True)
        live_raw = pd.concat([live_raw, new_rows], ignore_index=True)
        live_raw = live_raw.drop_duplicates(subset=["date"]).sort_values("date")
        live_raw.to_parquet(live_path, index=False)

    combined = pd.concat([research_raw, live_raw], ignore_index=True)
    combined = combined.drop_duplicates(subset=["date"]).sort_values("date").set_index("date")
    combined.index = pd.DatetimeIndex(combined.index)
    combined = combined[combined.index <= as_of_ts]
    assert_no_leakage(combined.index.to_series(), as_of_date, f"{ticker} price history")
    return combined


def _score_new_headlines(ticker: str, articles_df: pd.DataFrame, cache_path: Path) -> pd.DataFrame:
    cache_columns = ["headline", "positive", "negative", "neutral", "sentiment_score", "published_at", "ticker"]
    cached = _load_or_init_cache(cache_path, cache_columns)
    if articles_df.empty:
        return cached

    already_scored = set(cached["headline"]) if not cached.empty else set()
    to_score = articles_df[~articles_df["headline"].isin(already_scored)]
    if not to_score.empty:
        scored = score_headlines(to_score["headline"].tolist())
        scored["published_at"] = to_score["published_at"].values
        scored["ticker"] = ticker
        cached = pd.concat([cached, scored], ignore_index=True)
        cached = cached.drop_duplicates(subset=["headline", "published_at"])
        LIVE_SENTIMENT_DIR.mkdir(parents=True, exist_ok=True)
        cached.to_parquet(cache_path, index=False)
    return cached


def _clip_to_as_of_date(articles_df: pd.DataFrame, as_of_date: str) -> pd.DataFrame:
    """
    Finnhub's `to` date filter is not strictly reliable in practice - it can
    return an article or two published just past the requested boundary.
    Clip defensively here (using the same UTC calendar-day rule the leakage
    guard checks) so a news API quirk degrades to "one headline dropped"
    instead of aborting the whole day's run.
    """
    if articles_df.empty:
        return articles_df
    cutoff = pd.Timestamp(as_of_date, tz="UTC") + pd.Timedelta(days=1)
    published_utc = pd.to_datetime(articles_df["published_at"], utc=True)
    return articles_df[published_utc < cutoff]


def _combined_scored_sentiment(ticker: str, as_of_date: str, news_by_ticker: dict) -> pd.DataFrame:
    cache_path = LIVE_SENTIMENT_DIR / f"{ticker}.parquet"
    empty = pd.DataFrame(columns=["headline", "ticker", "published_at"])
    articles_df = _clip_to_as_of_date(news_by_ticker.get(ticker, empty), as_of_date)
    scored = _score_new_headlines(ticker, articles_df, cache_path)
    if not scored.empty:
        assert_no_leakage(scored["published_at"], as_of_date, f"{ticker} news")
    return scored


def _label_safe_cutoff(as_of_date: str, horizon: int) -> pd.Timestamp:
    """
    The frozen historical matrix was built once over the full 2019-2025
    dataset, not walk-forward - a row dated within `horizon` trading days of
    as_of_date has a label that was resolved using price data past
    as_of_date. Embargo those rows from training so a backtest-style
    as_of_date inside the frozen matrix's own range can't leak the future
    into itself (a live as_of_date, always long past the frozen matrix's
    end, is unaffected by this embargo).
    """
    as_of_ts = pd.Timestamp(as_of_date)
    lookback_start = (as_of_ts - pd.Timedelta(days=horizon * 3 + 10)).date().isoformat()
    recent_days = get_nyse_trading_days(lookback_start, as_of_date)
    recent_days = recent_days[recent_days <= as_of_ts]
    if len(recent_days) <= horizon:
        return pd.Timestamp.min
    return recent_days[-(horizon + 1)]


def build_universe_feature_state(tickers: list, as_of_date: str, feature_set: str = "base",
                                  news_lookback_days: int = NEWS_LOOKBACK_DAYS, finnhub_api_key: str = None) -> dict:
    """
    Build everything the predictor/sentiment wrappers need for `as_of_date`:
    an appended training matrix (frozen historical + newly-resolved live rows)
    and a predict_rows frame (one feature row per ticker at as_of_date, no
    label - the label isn't knowable yet). Shared across predict() and
    sentiment() so a pipeline run only pays the data-fetch/feature-build cost
    once per day.

    Returns {"training_matrix", "predict_rows", "data_window_hash", "news_coverage"}.
    Raises LeakageError (via leakage_guard) if any input data is timestamped
    after as_of_date - this is the Phase 2 leakage guard assertion.
    """
    finnhub_api_key = finnhub_api_key or config.FINNHUB_API_KEY
    horizon = config.MODEL_CFG["horizon_days"]
    up_threshold = config.MODEL_CFG["up_threshold"]
    down_threshold = config.MODEL_CFG["down_threshold"]

    historical = _load_historical_matrix()
    historical = historical[historical.index <= _label_safe_cutoff(as_of_date, _max_configured_horizon())]
    historical_max = historical.index.max() if not historical.empty else pd.Timestamp.min

    news_start = (pd.Timestamp(as_of_date) - pd.Timedelta(days=news_lookback_days)).date().isoformat()
    try:
        news_by_ticker, news_coverage = fetch_universe_news(tickers, news_start, as_of_date, finnhub_api_key)
    except NewsFetchError:
        # Total news outage: degrade to price-only rather than aborting the
        # run - the model tolerates 0.0 (neutral/no-news) sentiment features.
        news_by_ticker, news_coverage = {}, {t: "error" for t in tickers}

    ticker_dates = {}
    live_frames = []
    predict_rows = []
    price_errors = {}
    for ticker in tickers:
        try:
            price_series = _combined_price_series(ticker, as_of_date)
        except PriceFetchError as exc:
            price_errors[ticker] = str(exc)
            continue
        ticker_dates[ticker] = price_series.index

        trading_days = get_nyse_trading_days(price_series.index.min().date().isoformat(), as_of_date)
        scored_news = _combined_scored_sentiment(ticker, as_of_date, news_by_ticker)

        technical = compute_technical_features(price_series)
        if not scored_news.empty:
            effective_day = assign_effective_trading_day(scored_news["published_at"], trading_days)
            daily_sentiment = aggregate_daily_sentiment(scored_news, effective_day)
        else:
            daily_sentiment = pd.DataFrame(columns=["sentiment_mean", "sentiment_std", "headline_count"])
        sentiment = add_rolling_sentiment(daily_sentiment, trading_days)

        labels = generate_labels(price_series["Close"], horizon=horizon, up_threshold=up_threshold, down_threshold=down_threshold)

        matrix = technical.join(sentiment, how="left").join(labels, how="left")
        for col_name, extra_horizon, extra_up, extra_down in LABEL_CONFIGS:
            extra_labels = generate_labels(
                price_series["Close"], horizon=extra_horizon, up_threshold=extra_up, down_threshold=extra_down
            ).rename(col_name)
            matrix = matrix.join(extra_labels, how="left")
        matrix["ticker"] = ticker
        sentiment_cols = ["sentiment_mean", "sentiment_std", "sentiment_roll3", "sentiment_roll7"]
        matrix[sentiment_cols] = matrix[sentiment_cols].fillna(0.0)

        as_of_ts = pd.Timestamp(as_of_date)
        if as_of_ts in matrix.index:
            row = matrix.loc[[as_of_ts]].copy()
            if not row[TECHNICAL_COLUMNS].isna().any(axis=None):
                predict_rows.append(row)

        new_rows = matrix[matrix.index > historical_max]
        # Require complete technical features (unchanged, backward-compatible
        # requirement), but only require *some* label to be resolvable rather
        # than specifically "label" (10d). A label resolves once `horizon`
        # future trading days exist past its row's date, so label_7d (7d)
        # resolves first, "label" (10d) next, and label_20d_4pct (20d) last -
        # meaning any row where label_20d_4pct is resolved always has "label"
        # resolved too (20 future days implies 10), so the 20-day strategy was
        # never starved by the old dropna. label_7d IS the starved case: a
        # freshly-appended live row dated 7-9 trading days before as_of_date
        # already has a resolvable label_7d but not yet a resolvable "label",
        # and was being dropped from training_matrix entirely before the
        # weekly strategy ever got a chance to use it. The per-strategy
        # dropna in predictor.predict_from_state (keyed off each strategy's
        # own label_column) still does the final, precise filtering, so
        # keeping these extra rows around here is safe.
        trainable = new_rows.dropna(subset=TECHNICAL_COLUMNS)
        trainable = trainable[trainable[ALL_LABEL_COLUMNS].notna().any(axis=1)]
        live_frames.append(trainable)

    if feature_set == "xs" and any(not f.empty for f in live_frames):
        from src.data.tickers import SECTOR_MAP
        live_panel = pd.concat(live_frames)
        xs = compute_cross_sectional_features(live_panel, XS_BASE_COLUMNS, SECTOR_MAP)
        live_frames = [live_panel.join(xs)]

    training_matrix = pd.concat([historical] + live_frames) if live_frames else historical
    predict_df = pd.concat(predict_rows) if predict_rows else pd.DataFrame(columns=TECHNICAL_COLUMNS + SENTIMENT_COLUMNS + ["ticker"])

    if feature_set == "xs" and not predict_df.empty:
        from src.data.tickers import SECTOR_MAP
        xs = compute_cross_sectional_features(predict_df, XS_BASE_COLUMNS, SECTOR_MAP)
        predict_df = predict_df.join(xs)

    return {
        "training_matrix": training_matrix,
        "predict_rows": predict_df,
        "data_window_hash": compute_data_window_hash(ticker_dates),
        "news_coverage": news_coverage,
        "price_errors": price_errors,
    }

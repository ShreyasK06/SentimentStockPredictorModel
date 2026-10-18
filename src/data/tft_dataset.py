"""Per-ticker daily panel for a Temporal Fusion Transformer.

One row per (ticker, trading day). Column groups:

- ``KNOWN_REALS`` / ``KNOWN_CATEGORICALS``: deterministic calendar covariates.
- ``PRICE_FEATURES``, ``TECHNICAL_FEATURES``, ``SENTIMENT_FEATURES``: observed
  covariates, using only information available by the close of day t.
- ``TARGET_COLUMNS``: forward-looking evaluation targets. These are the ONLY
  columns that look ahead and must never be model inputs.

Sentiment choices
-----------------
Articles are mapped to the first trading day whose close they can affect
(``assign_effective_trading_day``: before 16:00 ET on a trading day -> same
day, otherwise next trading day). Days without news get count 0, means 0
(neutral) and ``has_news`` 0. Rolling features treat missing days as 0
(neutral) and use windows over trading days ending at t (past/current only).

Known limitation: FinBERT-scored news only covers roughly 2025-08 to 2025-12,
so for most of the history every sentiment feature is the neutral fill.
"""

from __future__ import annotations

import logging
import math
from typing import Sequence

import numpy as np
import pandas as pd

from src.config import PRICES_DIR, SENTIMENT_DIR
from src.features.technical import compute_technical_features
from src.features.trading_day import assign_effective_trading_day

logger = logging.getLogger(__name__)

HORIZONS: tuple[int, ...] = (1, 3, 5, 7)

KNOWN_REALS: list[str] = [
    "day_of_week_sin",
    "day_of_week_cos",
    "month_sin",
    "month_cos",
    "day_of_month",
    "days_to_next_trading_day",
]
KNOWN_CATEGORICALS: list[str] = [
    "day_of_week",
    "month",
    "is_month_end",
    "is_quarter_end",
    "is_pre_holiday",
]
CALENDAR_FEATURES: list[str] = KNOWN_CATEGORICALS + KNOWN_REALS

PRICE_FEATURES: list[str] = ["log_ret_1d", "log_volume_chg"]

TECHNICAL_FEATURES: list[str] = [
    "return_1d", "return_5d", "return_10d", "return_20d",
    "rsi_14", "macd", "macd_signal", "macd_hist",
    "volatility_20d", "volume_ratio_20d", "dist_from_ma50",
    "bollinger_pctb", "bollinger_bandwidth", "atr_14_norm",
    "stochastic_k", "stochastic_d", "obv_zscore",
    "dist_from_52w_high", "dist_from_52w_low",
]

SENTIMENT_FEATURES: list[str] = [
    "sent_mean", "sent_pos_mean", "sent_neg_mean", "sent_neu_mean",
    "article_count", "sent_roll3", "sent_roll7", "has_news",
]

TARGET_COLUMNS: list[str] = (
    [f"fwd_ret_{h}d" for h in HORIZONS] + [f"fwd_dir_{h}d" for h in HORIZONS]
)

OBSERVED_FEATURES: list[str] = PRICE_FEATURES + TECHNICAL_FEATURES + SENTIMENT_FEATURES

ID_COLUMNS: list[str] = ["date", "ticker", "time_idx"]


def assert_no_target_leakage(feature_columns: Sequence[str]) -> None:
    """Raise if any forward-looking target column is used as a feature."""
    leaked = sorted(set(feature_columns) & set(TARGET_COLUMNS))
    if leaked:
        raise ValueError(f"Target columns used as features: {leaked}")


def _normalize_index(dates: pd.DatetimeIndex) -> pd.DatetimeIndex:
    idx = pd.DatetimeIndex(dates)
    if idx.tz is not None:
        idx = idx.tz_localize(None)
    return idx.normalize()


def build_calendar_features(dates: pd.DatetimeIndex) -> pd.DataFrame:
    """Deterministic calendar covariates, indexed by ``dates``.

    ``dates`` is treated as the trading calendar: month/quarter end flags and
    the gap to the next trading day are derived from it. For the final date the
    gap is unknown and falls back to a weekday rule (Friday -> 3, else 1).
    """
    idx = _normalize_index(dates)
    dow = pd.Series(idx.dayofweek, index=idx)
    month = pd.Series(idx.month, index=idx)

    date_s = pd.Series(idx, index=idx)
    next_date = date_s.shift(-1)
    gap = (next_date - date_s).dt.days
    expected_gap = pd.Series(np.where(idx.dayofweek == 4, 3, 1), index=idx)
    days_to_next = gap.fillna(expected_gap).astype(int)

    is_month_end = (next_date.dt.month != month) | next_date.isna()
    is_quarter_end = is_month_end & month.isin([3, 6, 9, 12])
    is_pre_holiday = days_to_next > expected_gap

    dow_angle = 2 * math.pi * dow / 5
    month_angle = 2 * math.pi * (month - 1) / 12
    out = pd.DataFrame(
        {
            "day_of_week": dow.astype(int),
            "month": month.astype(int),
            "is_month_end": is_month_end.astype(int),
            "is_quarter_end": is_quarter_end.astype(int),
            "is_pre_holiday": is_pre_holiday.astype(int),
            "day_of_week_sin": np.sin(dow_angle),
            "day_of_week_cos": np.cos(dow_angle),
            "month_sin": np.sin(month_angle),
            "month_cos": np.cos(month_angle),
            "day_of_month": idx.day.astype(int),
            "days_to_next_trading_day": days_to_next,
        },
        index=idx,
    )
    return out[CALENDAR_FEATURES]


def build_sentiment_features(
    scored_news: pd.DataFrame | None, trading_days: pd.DatetimeIndex
) -> pd.DataFrame:
    """Daily sentiment features indexed by ``trading_days``.

    ``scored_news`` needs columns published_at (naive UTC), positive, negative,
    neutral, sentiment_score. None or empty input yields neutral fill.
    """
    days = _normalize_index(trading_days)
    out = pd.DataFrame(0.0, index=days, columns=SENTIMENT_FEATURES)

    if scored_news is not None and len(scored_news) > 0:
        news = scored_news.reset_index(drop=True)
        effective = assign_effective_trading_day(news["published_at"], days)
        grouped = (
            news.assign(effective_trading_day=effective.values)
            .dropna(subset=["effective_trading_day"])
            .groupby("effective_trading_day")
        )
        daily = pd.DataFrame(
            {
                "sent_mean": grouped["sentiment_score"].mean(),
                "sent_pos_mean": grouped["positive"].mean(),
                "sent_neg_mean": grouped["negative"].mean(),
                "sent_neu_mean": grouped["neutral"].mean(),
                "article_count": grouped["sentiment_score"].count().astype(float),
            }
        )
        daily = daily.reindex(days).fillna(0.0)
        out[daily.columns] = daily

    out["has_news"] = (out["article_count"] > 0).astype(float)
    # Missing days are 0 (neutral); windows end at t, so no future information.
    out["sent_roll3"] = out["sent_mean"].rolling(3, min_periods=1).mean()
    out["sent_roll7"] = out["sent_mean"].rolling(7, min_periods=1).mean()
    return out[SENTIMENT_FEATURES]


def build_forward_targets(
    close: pd.Series, horizons: Sequence[int] = HORIZONS
) -> pd.DataFrame:
    """Forward log returns and direction labels (NaN where t+h is past the data)."""
    log_close = np.log(close.astype(float))
    cols: dict[str, pd.Series] = {}
    for h in horizons:
        cols[f"fwd_ret_{h}d"] = log_close.shift(-h) - log_close
    for h in horizons:
        ret = cols[f"fwd_ret_{h}d"]
        cols[f"fwd_dir_{h}d"] = (ret > 0).astype(float).where(ret.notna())
    return pd.DataFrame(cols, index=close.index)


def build_ticker_panel(
    prices: pd.DataFrame,
    scored_news: pd.DataFrame | None,
    trading_days: pd.DatetimeIndex,
    ticker: str | None = None,
) -> pd.DataFrame:
    """Full feature/target frame for one ticker on the shared ``trading_days``.

    Days missing from ``prices`` are forward-filled (OHLC) with zero volume so
    every ticker covers the same calendar. Warm-up NaN rows are NOT dropped
    here (see ``finalize_panel``).
    """
    days = _normalize_index(trading_days)
    px = prices.copy()
    px.index = _normalize_index(px.index)
    px = px[~px.index.duplicated(keep="last")].sort_index()
    px = px.reindex(days)
    ohlc = ["Open", "High", "Low", "Close"]
    px[ohlc] = px[ohlc].ffill()
    px["Volume"] = px["Volume"].fillna(0.0)

    close = px["Close"].astype(float)
    tech = compute_technical_features(px[["Open", "High", "Low", "Close", "Volume"]])
    tech = tech[TECHNICAL_FEATURES]

    log_close = np.log(close)
    price = pd.DataFrame(
        {
            "log_close": log_close,
            "log_ret_1d": log_close.diff(),
            "log_volume_chg": np.log1p(px["Volume"].astype(float)).diff(),
        },
        index=days,
    )

    frame = pd.concat(
        [
            price,
            tech,
            build_sentiment_features(scored_news, days),
            build_calendar_features(days),
            build_forward_targets(close),
        ],
        axis=1,
    )
    frame.index.name = "date"
    frame = frame.reset_index()
    frame.insert(1, "ticker", ticker if ticker is not None else "")
    return frame


def add_time_idx(panel: pd.DataFrame) -> pd.DataFrame:
    """Add a contiguous global ``time_idx`` (same date -> same index, from 0)."""
    out = panel.copy()
    unique_dates = pd.DatetimeIndex(sorted(out["date"].unique()))
    mapping = pd.Series(np.arange(len(unique_dates)), index=unique_dates)
    out["time_idx"] = out["date"].map(mapping).astype(int)
    cols = ID_COLUMNS + [c for c in out.columns if c not in ID_COLUMNS]
    return out[cols]


def finalize_panel(frame: pd.DataFrame) -> pd.DataFrame:
    """Drop warm-up rows consistently across tickers, add time_idx, validate."""
    required = PRICE_FEATURES + TECHNICAL_FEATURES + ["log_close"]
    valid = frame.dropna(subset=required)
    start = valid.groupby("ticker")["date"].min().max()
    out = valid[valid["date"] >= start]
    out = out.sort_values(["ticker", "date"]).reset_index(drop=True)
    out = add_time_idx(out)

    per_ticker = out.groupby("ticker")["time_idx"]
    contiguous = (per_ticker.max() - per_ticker.min() + 1) == per_ticker.count()
    if not contiguous.all():
        raise ValueError(f"Gaps in time_idx for: {list(contiguous[~contiguous].index)}")
    if per_ticker.count().nunique() != 1:
        raise ValueError("Tickers have unequal row counts after warm-up drop")
    assert_no_target_leakage(OBSERVED_FEATURES + CALENDAR_FEATURES)
    return out


def _load_ticker_inputs(ticker: str) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    prices = pd.read_parquet(PRICES_DIR / f"{ticker}.parquet")
    news_path = SENTIMENT_DIR / f"{ticker}.parquet"
    news = pd.read_parquet(news_path) if news_path.exists() else None
    return prices, news


def list_price_tickers() -> list[str]:
    return sorted(p.stem for p in PRICES_DIR.glob("*.parquet"))


def build_panel(tickers: list[str] | None = None) -> pd.DataFrame:
    """Load raw prices and scored news from disk and build the full panel."""
    tickers = tickers or list_price_tickers()
    inputs = {t: _load_ticker_inputs(t) for t in tickers}
    trading_days = pd.DatetimeIndex(
        sorted(set().union(*(_normalize_index(p.index) for p, _ in inputs.values())))
    )
    frames = []
    for t, (prices, news) in inputs.items():
        n_missing = len(trading_days) - _normalize_index(prices.index).nunique()
        if n_missing:
            logger.warning("%s missing %d trading days; forward-filling", t, n_missing)
        frames.append(build_ticker_panel(prices, news, trading_days, ticker=t))
    return finalize_panel(pd.concat(frames, ignore_index=True))


def sentiment_coverage_report(panel: pd.DataFrame) -> pd.DataFrame:
    """Per-ticker news coverage: first/last news date, row and article counts."""
    rows = []
    for ticker, g in panel.groupby("ticker"):
        news_days = g.loc[g["has_news"] > 0, "date"]
        rows.append(
            {
                "ticker": ticker,
                "first_news_date": news_days.min() if len(news_days) else pd.NaT,
                "last_news_date": news_days.max() if len(news_days) else pd.NaT,
                "n_days": len(g),
                "n_days_with_news": int(len(news_days)),
                "pct_days_with_news": 100.0 * len(news_days) / len(g) if len(g) else 0.0,
                "n_articles": int(g["article_count"].sum()),
            }
        )
    return pd.DataFrame(rows).set_index("ticker")

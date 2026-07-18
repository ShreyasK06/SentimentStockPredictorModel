import hashlib

import pandas as pd


class LeakageError(Exception):
    """Raised when input data extends past the prediction's as_of_date."""


def assert_no_leakage(dates: pd.Series, as_of_date: str, context: str) -> None:
    """
    Assert every timestamp in `dates` is <= as_of_date (normalized to a date,
    timezone stripped). Raises LeakageError naming the offending context
    (e.g. "AAPL price history") so a violation is traceable to its source.
    """
    if len(dates) == 0:
        return
    idx = pd.DatetimeIndex(pd.to_datetime(dates))
    if idx.tz is not None:
        # Compare in UTC, not exchange-local time: this guard is a
        # conservative safety backstop, independent of the ET trading-day
        # bucketing that assign_effective_trading_day already applies
        # upstream - it must never be more permissive than a raw UTC compare.
        idx = idx.tz_convert("UTC").tz_localize(None)
    max_date = idx.normalize().max()
    cutoff = pd.Timestamp(as_of_date).normalize()
    if max_date > cutoff:
        raise LeakageError(
            f"Leakage guard tripped for {context}: max data timestamp {max_date.date()} "
            f"is after as_of_date {cutoff.date()}"
        )


def compute_data_window_hash(ticker_dates: dict) -> str:
    """
    Hash the exact (ticker, date) pairs that fed a prediction, so the run log
    proves what data window was actually used without storing the full frame.
    `ticker_dates` maps ticker -> iterable of date-like values.
    """
    parts = []
    for ticker in sorted(ticker_dates):
        dates = pd.DatetimeIndex(pd.to_datetime(list(ticker_dates[ticker]))).normalize()
        for d in sorted(dates.unique()):
            parts.append(f"{ticker}:{pd.Timestamp(d).date().isoformat()}")
    digest = hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()
    return digest

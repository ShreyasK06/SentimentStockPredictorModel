import pandas as pd

from src.data.prices import fetch_ohlcv

PRICE_PANEL_COLUMNS = ["date", "Open", "High", "Low", "Close", "Volume", "ticker"]


class PriceFetchError(Exception):
    """Raised when price data could not be fetched for the whole universe (total outage)."""


def fetch_universe_prices(tickers: list, start: str, end: str) -> tuple:
    """
    Fetch daily OHLCV for every ticker in the universe.

    Returns (panel, errors) where panel is a DataFrame with PRICE_PANEL_COLUMNS
    covering every ticker that succeeded, and errors maps ticker -> error message
    for tickers that failed. Raises PriceFetchError only when every ticker fails,
    since a single-ticker gap is recoverable but a total outage is not.
    """
    frames = []
    errors = {}
    for ticker in tickers:
        try:
            df = fetch_ohlcv(ticker, start, end)
            frames.append(df.reset_index())
        except Exception as exc:
            errors[ticker] = str(exc)

    if not frames:
        raise PriceFetchError(
            f"No price data returned for any of {len(tickers)} tickers between {start} and {end}: {errors}"
        )

    panel = pd.concat(frames, ignore_index=True)
    return panel, errors

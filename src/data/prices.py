import pandas as pd
import yfinance as yf

REQUIRED_COLUMNS = ["Open", "High", "Low", "Close", "Volume"]


def fetch_ohlcv(ticker: str, start: str, end: str) -> pd.DataFrame:
    # auto_adjust=True: prices are split/dividend-adjusted, the standard choice for
    # return-based ML. This is intentional, not a bug - kept for future readers.
    df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True, interval="1d")
    if df.empty:
        raise ValueError(f"No price data returned for {ticker} between {start} and {end}")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df[REQUIRED_COLUMNS].copy()
    df.index.name = "date"
    df["ticker"] = ticker
    return df

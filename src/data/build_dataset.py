from pathlib import Path
import pandas as pd

from src.data.prices import fetch_ohlcv
from src.data.news import fetch_company_news, articles_to_dataframe
from src.data.storage import save_parquet, find_missing_trading_days, find_misaligned_dates

PRICE_PANEL_COLUMNS = ["date", "Open", "High", "Low", "Close", "Volume", "ticker"]


def build_price_panel(tickers: list, start: str, end: str, prices_dir: Path) -> pd.DataFrame:
    frames = []
    for ticker in tickers:
        try:
            df = fetch_ohlcv(ticker, start, end)
            save_parquet(df, ticker, prices_dir)
            frames.append(df.reset_index())
        except Exception as exc:
            print(f"WARNING: failed to fetch {ticker}: {exc}")
            continue
    if not frames:
        return pd.DataFrame(columns=PRICE_PANEL_COLUMNS)
    return pd.concat(frames, ignore_index=True)


def validate_price_panel(panel: pd.DataFrame, tickers: list, trading_days: pd.DatetimeIndex) -> dict:
    report = {}
    for ticker in tickers:
        sub = panel[panel["ticker"] == ticker].set_index("date")
        report[ticker] = {
            "missing": find_missing_trading_days(sub, trading_days),
            "misaligned": find_misaligned_dates(sub, trading_days),
        }
    return report


def build_news_archive(tickers: list, start: str, end: str, news_dir: Path, api_key: str) -> dict:
    counts = {}
    for ticker in tickers:
        try:
            articles = fetch_company_news(ticker, start, end, api_key)
            df = articles_to_dataframe(articles, ticker)
            save_parquet(df, ticker, news_dir)
            counts[ticker] = len(df)
        except Exception as exc:
            print(f"WARNING: failed to fetch {ticker}: {exc}")
            continue
    return counts

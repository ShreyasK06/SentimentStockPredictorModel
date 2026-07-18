import time

import pandas as pd
import requests

from src.data.news import fetch_company_news, articles_to_dataframe

NEWS_COLUMNS = ["date", "ticker", "headline", "source", "url", "published_at"]

MAX_RETRIES = 3
BACKOFF_BASE_SECONDS = 5


class NewsFetchError(Exception):
    """Raised when the news API is unreachable for the whole universe (total outage)."""


def _fetch_with_backoff(ticker: str, start: str, end: str, api_key: str) -> list:
    last_exc = None
    for attempt in range(MAX_RETRIES):
        try:
            return fetch_company_news(ticker, start, end, api_key)
        except requests.exceptions.HTTPError as exc:
            status = exc.response.status_code if exc.response is not None else None
            last_exc = exc
            if status == 429:
                time.sleep(BACKOFF_BASE_SECONDS * (2 ** attempt))
                continue
            raise
        except requests.exceptions.RequestException as exc:
            last_exc = exc
            time.sleep(BACKOFF_BASE_SECONDS * (2 ** attempt))
    raise last_exc


def fetch_universe_news(tickers: list, start: str, end: str, api_key: str) -> tuple:
    """
    Fetch per-ticker headlines for the lookback window.

    Returns (news_by_ticker, coverage) where news_by_ticker maps ticker -> DataFrame
    (empty DataFrame is a valid "quiet day" result, not an error) and coverage maps
    ticker -> "ok" | "empty" | "error". Raises NewsFetchError only when every ticker
    errors out (the API itself appears down), matching the degradation policy:
    news unavailable overall vs. partial per-ticker gaps are handled differently
    by the caller.
    """
    news_by_ticker = {}
    coverage = {}
    for ticker in tickers:
        try:
            articles = _fetch_with_backoff(ticker, start, end, api_key)
            df = articles_to_dataframe(articles, ticker)
            news_by_ticker[ticker] = df
            coverage[ticker] = "empty" if df.empty else "ok"
        except Exception:
            news_by_ticker[ticker] = pd.DataFrame(columns=NEWS_COLUMNS)
            coverage[ticker] = "error"

    if all(status == "error" for status in coverage.values()):
        raise NewsFetchError(f"News API unreachable for all {len(tickers)} tickers between {start} and {end}")

    return news_by_ticker, coverage

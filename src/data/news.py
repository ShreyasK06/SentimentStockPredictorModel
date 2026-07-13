import time
from datetime import date, timedelta
import pandas as pd
import requests

FINNHUB_URL = "https://finnhub.io/api/v1/company-news"


def _year_chunks(start: str, end: str):
    start_d = date.fromisoformat(start)
    end_d = date.fromisoformat(end)
    chunk_start = start_d
    while chunk_start <= end_d:
        chunk_end = min(date(chunk_start.year, 12, 31), end_d)
        yield chunk_start.isoformat(), chunk_end.isoformat()
        chunk_start = chunk_end + timedelta(days=1)


def fetch_company_news(ticker: str, start: str, end: str, api_key: str, sleep_seconds: float = 1.0) -> list:
    articles = []
    for chunk_start, chunk_end in _year_chunks(start, end):
        resp = requests.get(FINNHUB_URL, params={
            "symbol": ticker, "from": chunk_start, "to": chunk_end, "token": api_key,
        }, timeout=30)
        resp.raise_for_status()
        articles.extend(resp.json())
        time.sleep(sleep_seconds)
    return articles


def articles_to_dataframe(articles: list, ticker: str) -> pd.DataFrame:
    columns = ["date", "ticker", "headline", "source", "url"]
    if not articles:
        return pd.DataFrame(columns=columns)
    df = pd.DataFrame(articles)
    df["date"] = pd.to_datetime(df["datetime"], unit="s").dt.normalize()
    df["ticker"] = ticker
    return df[columns]

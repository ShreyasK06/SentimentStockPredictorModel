"""Chunked Finnhub /company-news fetcher.

Finnhub's free tier returns at most ~250 articles per request, so a long date
range must be split into short windows. Windows that hit the cap are re-split
into single days. Results are cached per (ticker, window) so runs are resumable.
The API key is never logged or written.
"""
import hashlib
import json
import logging
import time
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import requests

from src.features.trading_day import assign_effective_trading_day

FINNHUB_URL = "https://finnhub.io/api/v1/company-news"
logger = logging.getLogger(__name__)


class FinnhubUnavailable(RuntimeError):
    """Raised when Finnhub cannot be used (missing key, auth failure)."""


def _require_key(api_key):
    if not api_key or not str(api_key).strip():
        raise FinnhubUnavailable("FINNHUB_API_KEY is missing; set it in .env to fetch Finnhub news.")


def date_chunks(start: str, end: str, days: int = 7):
    """Inclusive (from, to) ISO date pairs covering [start, end], each <= `days` days."""
    if days < 1:
        raise ValueError("days must be >= 1")
    s, e = date.fromisoformat(start), date.fromisoformat(end)
    out = []
    while s <= e:
        t = min(s + timedelta(days=days - 1), e)
        out.append((s.isoformat(), t.isoformat()))
        s = t + timedelta(days=1)
    return out


def _cache_path(cache_dir, ticker, c_from, c_to):
    return Path(cache_dir) / ticker / f"{c_from}_{c_to}.json"


class FetchReport:
    def __init__(self):
        self.failed_chunks = []   # (ticker, from, to, reason)
        self.capped_chunks = []   # (ticker, from, to, n) capped even at 1-day granularity
        self.n_requests = 0


def _request_chunk(session, ticker, c_from, c_to, api_key, max_retries, sleep_seconds, report):
    """Return list of articles, or None on failure. Raises FinnhubUnavailable on auth errors."""
    params = {"symbol": ticker, "from": c_from, "to": c_to, "token": api_key}
    for attempt in range(max_retries + 1):
        report.n_requests += 1
        try:
            resp = session.get(FINNHUB_URL, params=params, timeout=30)
        except requests.RequestException as exc:
            logger.warning("%s %s..%s request error: %s", ticker, c_from, c_to, type(exc).__name__)
            time.sleep(sleep_seconds * (attempt + 1))
            continue
        status = resp.status_code
        if status == 429:
            wait = min(60.0, max(sleep_seconds, 1.0) * (2 ** attempt) * 5)
            logger.warning("%s %s..%s HTTP 429, backing off %.1fs", ticker, c_from, c_to, wait)
            time.sleep(wait)
            continue
        if status in (401, 403):
            raise FinnhubUnavailable(f"Finnhub rejected the API key (HTTP {status}).")
        if status >= 400:
            logger.warning("%s %s..%s HTTP %s", ticker, c_from, c_to, status)
            if status >= 500:
                time.sleep(sleep_seconds * (attempt + 1))
                continue
            report.failed_chunks.append((ticker, c_from, c_to, f"HTTP {status}"))
            return None
        try:
            data = resp.json()
        except ValueError:
            report.failed_chunks.append((ticker, c_from, c_to, "bad json"))
            return None
        if not isinstance(data, list):
            report.failed_chunks.append((ticker, c_from, c_to, "unexpected payload"))
            return None
        return data
    report.failed_chunks.append((ticker, c_from, c_to, "retries exhausted"))
    return None


def _fetch_window(session, ticker, c_from, c_to, api_key, cache_dir, max_retries,
                  sleep_seconds, cap_threshold, report, allow_split):
    cp = _cache_path(cache_dir, ticker, c_from, c_to) if cache_dir else None
    if cp is not None and cp.exists():
        try:
            return json.loads(cp.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            pass
    data = _request_chunk(session, ticker, c_from, c_to, api_key, max_retries, sleep_seconds, report)
    time.sleep(sleep_seconds)
    if data is None:
        return []
    if len(data) >= cap_threshold:
        if allow_split and c_from != c_to:
            parts = []
            for d_from, d_to in date_chunks(c_from, c_to, 1):
                parts.extend(_fetch_window(session, ticker, d_from, d_to, api_key, cache_dir,
                                           max_retries, sleep_seconds, cap_threshold, report, False))
            data = parts
        else:
            report.capped_chunks.append((ticker, c_from, c_to, len(data)))
    if cp is not None:
        cp.parent.mkdir(parents=True, exist_ok=True)
        cp.write_text(json.dumps(data), encoding="utf-8")
    return data


def _dedupe(articles):
    """Drop repeats by Finnhub id (when non-zero), then by url."""
    seen_ids, seen_urls, out = set(), set(), []
    for a in articles:
        aid, url = a.get("id"), a.get("url")
        if aid:
            if aid in seen_ids:
                continue
            seen_ids.add(aid)
        if url:
            if url in seen_urls:
                continue
            seen_urls.add(url)
        out.append(a)
    return out


def fetch_company_news_chunked(ticker, start, end, api_key, chunk_days=7, session=None,
                               sleep_seconds=1.1, cache_dir=None, max_retries=5,
                               cap_threshold=240, report=None):
    """Fetch [start, end] in small windows; return deduplicated raw article dicts.

    `report` (a FetchReport) collects failed / still-capped chunks.
    """
    _require_key(api_key)
    session = session or requests.Session()
    report = report if report is not None else FetchReport()
    articles = []
    for c_from, c_to in date_chunks(start, end, chunk_days):
        articles.extend(_fetch_window(session, ticker, c_from, c_to, api_key, cache_dir,
                                      max_retries, sleep_seconds, cap_threshold, report, True))
    return _dedupe(articles)


ARTICLE_COLUMNS = ["article_id", "ticker", "headline", "publisher", "url", "source",
                   "published_at", "effective_day", "naive_effective_day", "alignment", "summary"]


def to_articles_frame(raw_articles, ticker, trading_days) -> pd.DataFrame:
    """Normalize raw Finnhub articles to the Phase 8 article schema."""
    if not raw_articles:
        return pd.DataFrame(columns=ARTICLE_COLUMNS)
    df = pd.DataFrame(raw_articles)
    ts = pd.to_datetime(df["datetime"], unit="s", utc=True)
    eff = assign_effective_trading_day(ts, pd.DatetimeIndex(trading_days))
    eff = pd.to_datetime(eff).astype("datetime64[ns]")
    url = df["url"] if "url" in df else pd.Series([""] * len(df))
    if "id" in df:
        article_id = df["id"].astype(str)
    else:
        article_id = url.map(lambda u: hashlib.md5(str(u).encode()).hexdigest())
    out = pd.DataFrame({
        "article_id": article_id,
        "ticker": ticker,
        "headline": df["headline"] if "headline" in df else "",
        "publisher": df["source"] if "source" in df else "",
        "url": url,
        "source": "finnhub",
        "published_at": ts,
        "effective_day": eff,
        "naive_effective_day": eff,
        "alignment": "epoch",
        "summary": df["summary"] if "summary" in df else "",
    })
    out = out[out["headline"].fillna("").astype(str).str.strip() != ""]
    return out.reset_index(drop=True)

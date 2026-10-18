"""Fetch the recent Finnhub news window in small chunks (Phase 8 / WP1b).

Resumable: per-(ticker, chunk) JSON cache. The API key is read from .env and
never printed or logged.
"""
import argparse
import json
import logging
import os
import sys
from datetime import date, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402
import requests  # noqa: E402
from dotenv import load_dotenv  # noqa: E402

from src.data.calendar import get_nyse_trading_days  # noqa: E402
from src.data.finnhub_chunked import (  # noqa: E402
    ARTICLE_COLUMNS, FetchReport, FinnhubUnavailable, fetch_company_news_chunked, to_articles_frame,
)

TICKERS = ("AAPL ABBV AMZN BA BAC CAT COP CVX EOG GOOGL GS HD HON JNJ JPM KO MCD META MRK MS "
           "MSFT PFE PG SLB UNH WFC WMT XOM").split()


def main():
    today = date.today()
    ap = argparse.ArgumentParser()
    ap.add_argument("--tickers", nargs="+", default=TICKERS)
    ap.add_argument("--start", default=(today - timedelta(days=365)).isoformat())
    ap.add_argument("--end", default=today.isoformat())
    ap.add_argument("--chunk-days", type=int, default=7)
    ap.add_argument("--cache-dir", default="data/raw/finnhub_chunked/cache")
    ap.add_argument("--out", default="data/processed/phase8/articles_finnhub.parquet")
    ap.add_argument("--log", default="data/raw/finnhub_chunked/fetch.log")
    ap.add_argument("--sleep-seconds", type=float, default=1.1)
    args = ap.parse_args()

    def rel(p):
        return Path(p) if Path(p).is_absolute() else ROOT / p

    log_path = rel(args.log)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(filename=str(log_path), level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    log = logging.getLogger("phase8_fetch")

    load_dotenv(ROOT / ".env")
    key = os.getenv("FINNHUB_API_KEY")
    if not key:
        print("FINNHUB_API_KEY missing in .env; nothing fetched.")
        return 1

    cache_dir, out_path = rel(args.cache_dir), rel(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    end_pad = (date.fromisoformat(args.end) + timedelta(days=14)).isoformat()
    trading_days = get_nyse_trading_days(args.start, end_pad)
    session = requests.Session()
    report = FetchReport()
    frames = []
    log.info("start tickers=%d range=%s..%s chunk_days=%d", len(args.tickers), args.start, args.end,
             args.chunk_days)
    try:
        for i, t in enumerate(args.tickers, 1):
            raw = fetch_company_news_chunked(t, args.start, args.end, key, args.chunk_days, session,
                                             args.sleep_seconds, cache_dir, report=report)
            frames.append(to_articles_frame(raw, t, trading_days))
            log.info("done %s (%d/%d): %d articles, requests so far=%d, failed=%d, capped=%d",
                     t, i, len(args.tickers), len(raw), report.n_requests,
                     len(report.failed_chunks), len(report.capped_chunks))
    except FinnhubUnavailable as exc:
        log.error("Finnhub unavailable: %s", exc)
        print(f"Finnhub unavailable: {exc}")
        return 2

    nonempty = [f for f in frames if len(f)]
    df = pd.concat(nonempty, ignore_index=True) if nonempty else pd.DataFrame(columns=ARTICLE_COLUMNS)
    df.to_parquet(out_path, index=False)
    summary = {"range": [args.start, args.end], "n_articles": int(len(df)), "n_requests": report.n_requests,
               "failed_chunks": report.failed_chunks, "capped_chunks": report.capped_chunks}
    (log_path.parent / "fetch_summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    log.info("FINISHED articles=%d requests=%d failed=%d capped=%d", len(df), report.n_requests,
             len(report.failed_chunks), len(report.capped_chunks))
    print(f"wrote {out_path} ({len(df)} articles)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

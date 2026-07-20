"""Fetch daily OHLCV history for a single ticker and print it as JSON.

Used by the dashboard's Next.js route handler (app/api/stocks/[ticker]/history)
as a subprocess call, since Finnhub's /stock/candle endpoint is paywalled on
our API tier. Reuses the same yfinance fetcher (`fetch_ohlcv`) already proven
out for the 28-ticker research universe.

Usage:
    python scripts/fetch_ticker_history.py TICKER START_DATE END_DATE

Always prints exactly one JSON value to stdout:
  - success: a JSON array of {date, open, high, low, close, volume}
  - failure: {"error": "..."} (never a raw Python traceback)

Never raises past main() — the caller (a Node.js child_process with a 10s
timeout) expects stdout to always be parseable JSON.
"""

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.prices import fetch_ohlcv  # noqa: E402

RETRY_ATTEMPTS = 2


def main() -> None:
    if len(sys.argv) != 4:
        print(json.dumps({"error": "usage: fetch_ticker_history.py TICKER START END"}))
        return

    ticker, start, end = sys.argv[1], sys.argv[2], sys.argv[3]

    last_error: Exception | None = None
    for attempt in range(RETRY_ATTEMPTS):
        try:
            df = fetch_ohlcv(ticker, start, end)
            records = [
                {
                    "date": idx.strftime("%Y-%m-%d"),
                    "open": float(row["Open"]),
                    "high": float(row["High"]),
                    "low": float(row["Low"]),
                    "close": float(row["Close"]),
                    "volume": int(row["Volume"]),
                }
                for idx, row in df.iterrows()
            ]
            print(json.dumps(records))
            return
        except Exception as exc:  # noqa: BLE001 - deliberately broad, see docstring
            last_error = exc
            if attempt < RETRY_ATTEMPTS - 1:
                time.sleep(1)

    print(json.dumps({"error": str(last_error) if last_error else "unknown error"}))


if __name__ == "__main__":
    main()

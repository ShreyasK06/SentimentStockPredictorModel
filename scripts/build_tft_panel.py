"""Build the per-ticker daily TFT panel and write data/processed/tft_panel.parquet.

Run: python scripts/build_tft_panel.py [--tickers AAPL,MSFT,...]
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from src.config import DATA_PROCESSED_DIR  # noqa: E402
from src.data.tft_dataset import build_panel, sentiment_coverage_report  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tickers", default=None, help="comma-separated tickers (default: all)")
    args = parser.parse_args()
    tickers = [t.strip() for t in args.tickers.split(",") if t.strip()] if args.tickers else None

    panel = build_panel(tickers)
    DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out_path = DATA_PROCESSED_DIR / "tft_panel.parquet"
    panel.to_parquet(out_path, index=False)

    print(f"Wrote {out_path}")
    print(f"Shape: {panel.shape}")
    print(f"Date range: {panel['date'].min().date()} .. {panel['date'].max().date()}")
    print(f"time_idx range: {panel['time_idx'].min()} .. {panel['time_idx'].max()}")
    print("Rows per ticker:")
    print(panel.groupby("ticker").size().to_string())

    cov = sentiment_coverage_report(panel)
    with pd.option_context("display.width", 200, "display.max_rows", 100):
        print("Sentiment coverage per ticker:")
        print(cov.round(2).to_string())
    print(
        f"Overall: first news {cov['first_news_date'].min()}, last news {cov['last_news_date'].max()}, "
        f"{100.0 * panel['has_news'].mean():.2f}% of rows have news, "
        f"{int(panel['article_count'].sum())} articles"
    )


if __name__ == "__main__":
    main()

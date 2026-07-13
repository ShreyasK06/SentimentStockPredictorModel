from src import config
from src.data.tickers import UNIVERSE
from src.data.calendar import get_nyse_trading_days
from src.data.build_dataset import build_price_panel, validate_price_panel, build_news_archive
from src.data.storage import save_parquet


def main():
    trading_days = get_nyse_trading_days(config.START_DATE, config.END_DATE)

    panel = build_price_panel(UNIVERSE, config.START_DATE, config.END_DATE, config.PRICES_DIR)
    gap_report = validate_price_panel(panel, UNIVERSE, trading_days)
    total_missing = sum(len(v) for v in gap_report.values())

    print(f"Trading days expected: {len(trading_days)}")
    print(f"Tickers pulled: {len(UNIVERSE)}")
    print(f"Total price rows: {len(panel)}")
    print(f"Tickers with gaps: {[t for t, v in gap_report.items() if v]}")
    print(f"Total missing (ticker, day) cells: {total_missing}")

    if not config.FINNHUB_API_KEY:
        print("FINNHUB_API_KEY not set - skipping news pull. Set it in .env to fetch headlines.")
    else:
        news_counts = build_news_archive(UNIVERSE, config.START_DATE, config.END_DATE, config.NEWS_DIR, config.FINNHUB_API_KEY)
        print(f"Headline counts per ticker: {news_counts}")
        low_coverage = [t for t, c in news_counts.items() if c < 50]
        if low_coverage:
            print(f"WARNING: low news coverage (<50 headlines over 5y) for: {low_coverage}")

    config.DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    save_parquet(panel.set_index("date"), "combined_price_panel", config.DATA_PROCESSED_DIR)
    print(f"Saved combined panel to {config.DATA_PROCESSED_DIR / 'combined_price_panel.parquet'}")


if __name__ == "__main__":
    main()

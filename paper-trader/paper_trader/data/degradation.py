RUN_STATUS_OK = "ok"
RUN_STATUS_ABORTED_NO_PRICES = "aborted_no_prices"
RUN_STATUS_SKIPPED_NO_NEWS = "skipped_no_news"
RUN_STATUS_PARTIAL_NEWS = "partial_news"


def evaluate_data_coverage(tickers: list, price_errors: dict, news_coverage: dict, model_supports_price_only: bool) -> dict:
    """
    Apply the Phase 1 degradation policy and decide whether/how the run proceeds.

    Returns a dict: {status, proceed, use_news, tickers_missing_prices, tickers_missing_news}.

    Policy:
    - Every ticker missing prices -> abort the run entirely.
    - Every ticker missing news -> run price-only if the model supports it, else skip trading.
    - Partial news coverage -> proceed, recording which tickers had coverage.
    """
    tickers_missing_prices = [t for t in tickers if t in price_errors]
    tickers_missing_news = [t for t, status in news_coverage.items() if status == "error"]

    if len(tickers_missing_prices) == len(tickers):
        return {
            "status": RUN_STATUS_ABORTED_NO_PRICES,
            "proceed": False,
            "use_news": False,
            "tickers_missing_prices": tickers_missing_prices,
            "tickers_missing_news": tickers_missing_news,
        }

    all_news_missing = len(tickers_missing_news) == len(tickers)
    if all_news_missing:
        if model_supports_price_only:
            return {
                "status": RUN_STATUS_OK,
                "proceed": True,
                "use_news": False,
                "tickers_missing_prices": tickers_missing_prices,
                "tickers_missing_news": tickers_missing_news,
            }
        return {
            "status": RUN_STATUS_SKIPPED_NO_NEWS,
            "proceed": False,
            "use_news": False,
            "tickers_missing_prices": tickers_missing_prices,
            "tickers_missing_news": tickers_missing_news,
        }

    status = RUN_STATUS_PARTIAL_NEWS if tickers_missing_news else RUN_STATUS_OK
    return {
        "status": status,
        "proceed": True,
        "use_news": True,
        "tickers_missing_prices": tickers_missing_prices,
        "tickers_missing_news": tickers_missing_news,
    }

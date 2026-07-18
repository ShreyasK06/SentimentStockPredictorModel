from paper_trader.data.degradation import (
    evaluate_data_coverage,
    RUN_STATUS_OK, RUN_STATUS_ABORTED_NO_PRICES, RUN_STATUS_SKIPPED_NO_NEWS, RUN_STATUS_PARTIAL_NEWS,
)

TICKERS = ["AAPL", "MSFT", "GOOGL"]


def test_all_prices_missing_aborts_run():
    result = evaluate_data_coverage(TICKERS, price_errors={t: "boom" for t in TICKERS}, news_coverage={}, model_supports_price_only=True)
    assert result["status"] == RUN_STATUS_ABORTED_NO_PRICES
    assert result["proceed"] is False


def test_all_news_missing_price_only_model_still_proceeds():
    news_coverage = {t: "error" for t in TICKERS}
    result = evaluate_data_coverage(TICKERS, price_errors={}, news_coverage=news_coverage, model_supports_price_only=True)
    assert result["status"] == RUN_STATUS_OK
    assert result["proceed"] is True
    assert result["use_news"] is False


def test_all_news_missing_no_price_only_support_skips_trading():
    news_coverage = {t: "error" for t in TICKERS}
    result = evaluate_data_coverage(TICKERS, price_errors={}, news_coverage=news_coverage, model_supports_price_only=False)
    assert result["status"] == RUN_STATUS_SKIPPED_NO_NEWS
    assert result["proceed"] is False


def test_partial_news_proceeds_and_records_coverage():
    news_coverage = {"AAPL": "ok", "MSFT": "error", "GOOGL": "empty"}
    result = evaluate_data_coverage(TICKERS, price_errors={}, news_coverage=news_coverage, model_supports_price_only=True)
    assert result["status"] == RUN_STATUS_PARTIAL_NEWS
    assert result["proceed"] is True
    assert result["tickers_missing_news"] == ["MSFT"]


def test_single_ticker_price_gap_is_recoverable():
    result = evaluate_data_coverage(TICKERS, price_errors={"AAPL": "timeout"}, news_coverage={t: "ok" for t in TICKERS}, model_supports_price_only=True)
    assert result["status"] == RUN_STATUS_OK
    assert result["proceed"] is True
    assert result["tickers_missing_prices"] == ["AAPL"]


def test_all_ok_is_plain_ok():
    result = evaluate_data_coverage(TICKERS, price_errors={}, news_coverage={t: "ok" for t in TICKERS}, model_supports_price_only=True)
    assert result["status"] == RUN_STATUS_OK
    assert result["tickers_missing_prices"] == []
    assert result["tickers_missing_news"] == []

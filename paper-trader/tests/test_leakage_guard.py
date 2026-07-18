import pandas as pd
import pytest

from paper_trader.models.leakage_guard import assert_no_leakage, compute_data_window_hash, LeakageError


class TestAssertNoLeakage:
    def test_passes_when_all_dates_on_or_before_as_of_date(self):
        dates = pd.Series(pd.date_range("2024-01-01", "2024-01-10"))
        assert_no_leakage(dates, "2024-01-10", context="test")  # should not raise

    def test_raises_when_any_date_is_after_as_of_date(self):
        dates = pd.Series(pd.date_range("2024-01-01", "2024-01-10"))
        with pytest.raises(LeakageError):
            assert_no_leakage(dates, "2024-01-05", context="AAPL price history")

    def test_error_message_names_the_context(self):
        dates = pd.Series(pd.date_range("2024-01-01", "2024-01-10"))
        with pytest.raises(LeakageError, match="AAPL price history"):
            assert_no_leakage(dates, "2024-01-05", context="AAPL price history")

    def test_empty_series_never_raises(self):
        assert_no_leakage(pd.Series([], dtype="datetime64[ns]"), "2024-01-05", context="test")

    def test_timezone_aware_dates_are_handled(self):
        dates = pd.Series(pd.date_range("2024-01-01", "2024-01-05", tz="UTC"))
        assert_no_leakage(dates, "2024-01-05", context="test")
        with pytest.raises(LeakageError):
            assert_no_leakage(dates, "2024-01-04", context="test")


class TestDataWindowHash:
    def test_deterministic_for_same_input(self):
        ticker_dates = {"AAPL": pd.date_range("2024-01-01", "2024-01-03")}
        assert compute_data_window_hash(ticker_dates) == compute_data_window_hash(ticker_dates)

    def test_differs_when_dates_change(self):
        h1 = compute_data_window_hash({"AAPL": pd.date_range("2024-01-01", "2024-01-03")})
        h2 = compute_data_window_hash({"AAPL": pd.date_range("2024-01-01", "2024-01-04")})
        assert h1 != h2

    def test_order_independent_across_tickers(self):
        dates_a = pd.date_range("2024-01-01", "2024-01-02")
        dates_b = pd.date_range("2024-01-01", "2024-01-02")
        h1 = compute_data_window_hash({"AAPL": dates_a, "MSFT": dates_b})
        h2 = compute_data_window_hash({"MSFT": dates_b, "AAPL": dates_a})
        assert h1 == h2

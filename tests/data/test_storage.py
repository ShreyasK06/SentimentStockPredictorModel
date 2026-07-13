import pandas as pd
from src.data.storage import save_parquet, load_parquet, find_missing_trading_days, find_misaligned_dates


def test_save_and_load_roundtrip(tmp_path):
    idx = pd.date_range("2024-01-02", periods=3, freq="B")
    df = pd.DataFrame({"Open": [1, 2, 3], "Close": [1, 2, 3]}, index=idx)
    save_parquet(df, "AAPL", tmp_path)
    loaded = load_parquet("AAPL", tmp_path)
    assert list(loaded["Close"]) == [1, 2, 3]


def test_find_missing_trading_days_detects_gap():
    idx = pd.DatetimeIndex(["2024-01-02", "2024-01-04"])
    df = pd.DataFrame({"Close": [1, 2]}, index=idx)
    trading_days = pd.DatetimeIndex(["2024-01-02", "2024-01-03", "2024-01-04"])
    missing = find_missing_trading_days(df, trading_days)
    assert missing == ["2024-01-03"]


def test_find_missing_trading_days_empty_when_complete():
    idx = pd.DatetimeIndex(["2024-01-02", "2024-01-03"])
    df = pd.DataFrame({"Close": [1, 2]}, index=idx)
    trading_days = pd.DatetimeIndex(["2024-01-02", "2024-01-03"])
    missing = find_missing_trading_days(df, trading_days)
    assert missing == []


def test_find_missing_trading_days_tz_aware_index():
    # Regression test: tz-aware index should match tz-naive trading_days
    idx = pd.date_range("2024-01-02", periods=2, freq="B", tz="America/New_York")
    df = pd.DataFrame({"Close": [1, 2]}, index=idx)
    trading_days = pd.DatetimeIndex(["2024-01-02", "2024-01-03"])
    missing = find_missing_trading_days(df, trading_days)
    assert missing == []


def test_find_misaligned_dates_detects_spurious_non_trading_day():
    # 2024-01-06 and 2024-01-07 are a Saturday/Sunday - not real NYSE trading days.
    idx = pd.DatetimeIndex(["2024-01-02", "2024-01-03", "2024-01-06"])
    df = pd.DataFrame({"Close": [1, 2, 3]}, index=idx)
    trading_days = pd.DatetimeIndex(["2024-01-02", "2024-01-03"])
    misaligned = find_misaligned_dates(df, trading_days)
    assert misaligned == ["2024-01-06"]


def test_find_misaligned_dates_empty_when_fully_aligned():
    idx = pd.DatetimeIndex(["2024-01-02", "2024-01-03"])
    df = pd.DataFrame({"Close": [1, 2]}, index=idx)
    trading_days = pd.DatetimeIndex(["2024-01-02", "2024-01-03"])
    misaligned = find_misaligned_dates(df, trading_days)
    assert misaligned == []


def test_find_misaligned_dates_tz_aware_index():
    # Regression test: tz-aware index should match tz-naive trading_days
    idx = pd.date_range("2024-01-02", periods=2, freq="B", tz="America/New_York")
    df = pd.DataFrame({"Close": [1, 2]}, index=idx)
    trading_days = pd.DatetimeIndex(["2024-01-02", "2024-01-03"])
    misaligned = find_misaligned_dates(df, trading_days)
    assert misaligned == []

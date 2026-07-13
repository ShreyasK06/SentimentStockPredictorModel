import pandas as pd
from src.features.trading_day import assign_effective_trading_day


def test_before_close_on_trading_day_stays_same_day():
    trading_days = pd.DatetimeIndex(["2024-01-16", "2024-01-17", "2024-01-18"])
    # 2024-01-17 15:59 US/Eastern (winter, EST = UTC-5) = 20:59 UTC
    published = pd.Series([pd.Timestamp("2024-01-17 20:59:00", tz="UTC")])
    result = assign_effective_trading_day(published, trading_days)
    assert result.iloc[0] == pd.Timestamp("2024-01-17")


def test_after_close_on_trading_day_rolls_to_next_trading_day():
    trading_days = pd.DatetimeIndex(["2024-01-16", "2024-01-17", "2024-01-18"])
    # 2024-01-17 16:01 US/Eastern = 21:01 UTC
    published = pd.Series([pd.Timestamp("2024-01-17 21:01:00", tz="UTC")])
    result = assign_effective_trading_day(published, trading_days)
    assert result.iloc[0] == pd.Timestamp("2024-01-18")


def test_published_on_non_trading_day_rolls_to_next_trading_day():
    trading_days = pd.DatetimeIndex(["2024-01-16", "2024-01-17", "2024-01-22"])  # 18-21 = weekend/holiday gap
    # Saturday 2024-01-20, any time of day
    published = pd.Series([pd.Timestamp("2024-01-20 15:00:00", tz="UTC")])
    result = assign_effective_trading_day(published, trading_days)
    assert result.iloc[0] == pd.Timestamp("2024-01-22")


def test_handles_multiple_rows_independently():
    trading_days = pd.DatetimeIndex(["2024-01-16", "2024-01-17", "2024-01-18"])
    published = pd.Series([
        pd.Timestamp("2024-01-17 20:59:00", tz="UTC"),  # before close -> same day
        pd.Timestamp("2024-01-17 21:01:00", tz="UTC"),  # after close -> next day
    ])
    result = assign_effective_trading_day(published, trading_days)
    assert result.iloc[0] == pd.Timestamp("2024-01-17")
    assert result.iloc[1] == pd.Timestamp("2024-01-18")

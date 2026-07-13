import pandas as pd
from src.data.calendar import get_nyse_trading_days


def test_returns_datetimeindex():
    days = get_nyse_trading_days("2024-01-01", "2024-01-10")
    assert isinstance(days, pd.DatetimeIndex)
    assert len(days) > 0


def test_excludes_weekends():
    days = get_nyse_trading_days("2024-01-01", "2024-01-07")
    assert all(pd.Timestamp(d).weekday() < 5 for d in days)


def test_excludes_known_holiday():
    days = get_nyse_trading_days("2024-01-01", "2024-01-02")
    assert pd.Timestamp("2024-01-01").normalize() not in pd.DatetimeIndex(days).normalize()

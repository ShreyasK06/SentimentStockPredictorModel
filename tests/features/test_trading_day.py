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


def test_headline_after_last_trading_day_in_calendar_returns_nat():
    # A headline published well beyond the known calendar must NOT be silently
    # misattributed backward to a stale trading day.
    trading_days = pd.DatetimeIndex(["2024-01-16", "2024-01-17", "2024-01-18"])
    published = pd.Series([pd.Timestamp("2024-01-25 15:00:00", tz="UTC")])
    result = assign_effective_trading_day(published, trading_days)
    assert pd.isna(result.iloc[0])


def test_after_close_on_last_trading_day_in_calendar_returns_nat():
    # After-close on the LAST day in the calendar must roll forward, but there
    # is no next day known - so it must be NaT, not silently stay same-day.
    trading_days = pd.DatetimeIndex(["2024-01-16", "2024-01-17", "2024-01-18"])
    # 2024-01-18 16:01 ET = 21:01 UTC
    published = pd.Series([pd.Timestamp("2024-01-18 21:01:00", tz="UTC")])
    result = assign_effective_trading_day(published, trading_days)
    assert pd.isna(result.iloc[0])


def test_exact_market_close_time_rolls_to_next_day():
    # Exactly 16:00:00 ET is NOT "before close" - must roll to next day.
    trading_days = pd.DatetimeIndex(["2024-01-16", "2024-01-17", "2024-01-18"])
    # 2024-01-17 16:00:00 ET = 21:00:00 UTC exactly
    published = pd.Series([pd.Timestamp("2024-01-17 21:00:00", tz="UTC")])
    result = assign_effective_trading_day(published, trading_days)
    assert result.iloc[0] == pd.Timestamp("2024-01-18")


def test_before_close_in_summer_edt_stays_same_day():
    # July = EDT = UTC-4 (not UTC-5 like the winter EST tests). Confirms
    # tz_convert does real DST-aware conversion, not a hardcoded offset.
    trading_days = pd.DatetimeIndex(["2024-07-15", "2024-07-16", "2024-07-17"])
    # 2024-07-16 15:59 ET (EDT, UTC-4) = 19:59 UTC
    published = pd.Series([pd.Timestamp("2024-07-16 19:59:00", tz="UTC")])
    result = assign_effective_trading_day(published, trading_days)
    assert result.iloc[0] == pd.Timestamp("2024-07-16")


def test_tz_naive_published_at_is_treated_as_utc():
    # Phase 1's news schema documents published_at as UTC-aware, but there's
    # no defensive guard against tz-naive input reaching this function. A
    # tz-naive timestamp must be treated as UTC (not raise, not silently
    # assume US/Eastern), matching the equivalent tz-aware UTC case from
    # test_before_close_on_trading_day_stays_same_day.
    trading_days = pd.DatetimeIndex(["2024-01-16", "2024-01-17", "2024-01-18"])
    # Tz-naive, no tz= argument at all.
    published_naive = pd.Series([pd.Timestamp("2024-01-17 20:59:00")])
    result_naive = assign_effective_trading_day(published_naive, trading_days)

    published_utc = pd.Series([pd.Timestamp("2024-01-17 20:59:00", tz="UTC")])
    result_utc = assign_effective_trading_day(published_utc, trading_days)

    assert result_naive.iloc[0] == pd.Timestamp("2024-01-17")
    assert result_naive.iloc[0] == result_utc.iloc[0]

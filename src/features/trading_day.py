import numpy as np
import pandas as pd

MARKET_CLOSE_ET = pd.Timestamp("16:00").time()


def assign_effective_trading_day(published_at: pd.Series, trading_days: pd.DatetimeIndex) -> pd.Series:
    published_et = pd.to_datetime(published_at).dt.tz_convert("America/New_York")
    calendar_day_naive = published_et.dt.tz_localize(None).dt.normalize()
    before_close = published_et.dt.time < MARKET_CLOSE_ET

    sorted_days = pd.DatetimeIndex(
        sorted(pd.DatetimeIndex(trading_days).tz_localize(None).normalize().unique())
    )
    is_trading_day = calendar_day_naive.isin(sorted_days)
    same_day_eligible = before_close & is_trading_day

    positions_next = sorted_days.searchsorted(calendar_day_naive.values, side="right")
    out_of_bounds = positions_next >= len(sorted_days)
    safe_positions = np.where(out_of_bounds, 0, positions_next)
    next_trading_day = pd.Series(sorted_days[safe_positions], index=calendar_day_naive.index)
    next_trading_day = next_trading_day.mask(out_of_bounds, pd.NaT)

    effective = calendar_day_naive.where(same_day_eligible, next_trading_day)
    return effective.rename("effective_trading_day")

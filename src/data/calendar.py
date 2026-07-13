import pandas as pd
import pandas_market_calendars as mcal


def get_nyse_trading_days(start: str, end: str) -> pd.DatetimeIndex:
    nyse = mcal.get_calendar("NYSE")
    schedule = nyse.schedule(start_date=start, end_date=end)
    return pd.DatetimeIndex(schedule.index.date)

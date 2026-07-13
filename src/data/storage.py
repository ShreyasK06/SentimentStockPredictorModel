from pathlib import Path
import pandas as pd


def save_parquet(df: pd.DataFrame, ticker: str, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{ticker}.parquet"
    df.to_parquet(path)
    return path


def load_parquet(ticker: str, out_dir: Path) -> pd.DataFrame:
    path = out_dir / f"{ticker}.parquet"
    return pd.read_parquet(path)


def find_missing_trading_days(df: pd.DataFrame, trading_days: pd.DatetimeIndex) -> list:
    have = pd.DatetimeIndex(df.index).normalize()
    want = pd.DatetimeIndex(trading_days).normalize()
    # Strip timezone info to ensure comparison works regardless of tz-aware/tz-naive mismatch
    have = set(have.tz_localize(None) if have.tz is not None else have)
    want = set(want.tz_localize(None) if want.tz is not None else want)
    missing = sorted(want - have)
    return [d.strftime("%Y-%m-%d") for d in missing]

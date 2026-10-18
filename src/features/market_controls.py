"""Causal market/technical controls per (ticker, date) (Phase 8 WP2).

Every value at date E uses OHLCV up to and including session E only
(no next-day open/close). Input: long price frame with ticker, date, Open, High,
Low, Close, Volume; must include ticker 'SPY' for market features.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _rsi(close: pd.Series, n: int = 14) -> pd.Series:
    d = close.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    rs = up / dn.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def _one(g: pd.DataFrame) -> pd.DataFrame:
    g = g.sort_values("date").set_index("date")
    c = g["Close"]
    r = c.pct_change()
    out = pd.DataFrame(index=g.index)
    out["ctl_ret_1d"] = r
    out["ctl_ret_5d"] = c / c.shift(5) - 1
    out["ctl_ret_20d"] = c / c.shift(20) - 1
    out["ctl_vol_20d"] = r.rolling(20).std()
    out["ctl_rsi14"] = _rsi(c)
    out["ctl_gap"] = g["Open"] / c.shift(1) - 1
    lv = np.log1p(g["Volume"].astype(float))
    out["ctl_volume_z20"] = (lv - lv.rolling(20).mean()) / lv.rolling(20).std()
    out["ctl_range"] = (g["High"] - g["Low"]) / c
    out["ctl_dist_ma20"] = c / c.rolling(20).mean() - 1
    out["ctl_dist_ma50"] = c / c.rolling(50).mean() - 1
    return out


def build_market_controls(prices: pd.DataFrame) -> pd.DataFrame:
    prices = prices.copy()
    prices["date"] = pd.to_datetime(prices["date"]).dt.tz_localize(None).dt.normalize()
    spy = _one(prices[prices["ticker"] == "SPY"])
    mkt = pd.DataFrame({"ctl_spy_ret_1d": spy["ctl_ret_1d"], "ctl_spy_ret_5d": spy["ctl_ret_5d"],
                        "ctl_spy_vol_20d": spy["ctl_vol_20d"]})
    frames = []
    for t, g in prices[prices["ticker"] != "SPY"].groupby("ticker"):
        f = _one(g).join(mkt, how="left")
        cov = f["ctl_ret_1d"].rolling(60).cov(f["ctl_spy_ret_1d"])
        var = f["ctl_spy_ret_1d"].rolling(60).var()
        f["ctl_beta60"] = cov / var
        f["ctl_excess_ret_1d"] = f["ctl_ret_1d"] - f["ctl_spy_ret_1d"]
        f["ctl_dow"] = f.index.dayofweek.astype(float)
        f["ticker"] = t
        frames.append(f.reset_index())
    return pd.concat(frames, ignore_index=True)

import numpy as np
import pandas as pd
from scipy import stats


def _outcomes_df(outcomes: list, traded_only: bool = False) -> pd.DataFrame:
    df = pd.DataFrame(outcomes)
    if df.empty:
        return df
    if traded_only:
        df = df[df["traded"] == True]  # noqa: E712
    return df


def binomial_significance(n_correct: int, n_total: int, baseline: float):
    """Two-sided binomial test of the observed hit rate against a baseline rate."""
    if n_total == 0 or baseline is None or not (0 < baseline < 1):
        return None
    result = stats.binomtest(n_correct, n_total, baseline, alternative="two-sided")
    return float(result.pvalue)


def hit_rate_vs_baselines(outcomes: list, traded_only: bool = False) -> dict:
    """
    Directional hit rate plus the two honest baselines: coin flip (0.5) and
    always-up (the realized fraction of up-moves over the same date range),
    each with a binomial significance p-value. This is the number the report
    must state plainly, including when it shows no edge.
    """
    df = _outcomes_df(outcomes, traded_only=traded_only)
    if df.empty:
        return {
            "hit_rate": None, "n": 0,
            "coin_flip_baseline": 0.5, "coin_flip_p_value": None,
            "always_up_baseline": None, "always_up_p_value": None,
        }

    n_total = len(df)
    n_correct = int(df["correct"].sum())
    hit_rate = n_correct / n_total
    always_up = float((df["realized_direction"] == "up").mean())

    return {
        "hit_rate": hit_rate,
        "n": n_total,
        "coin_flip_baseline": 0.5,
        "coin_flip_p_value": binomial_significance(n_correct, n_total, 0.5),
        "always_up_baseline": always_up,
        "always_up_p_value": binomial_significance(n_correct, n_total, always_up),
    }


def portfolio_metrics(trades: list) -> dict:
    """
    Cumulative return, max drawdown, and annualized Sharpe computed from
    realized per-trade returns on exit, resampled onto a daily equity curve.
    """
    exits = [t for t in trades if t.get("event") == "exit" and "return_pct" in t]
    if not exits:
        return {"cumulative_return": None, "max_drawdown": None, "sharpe": None, "n_trades": 0}

    df = pd.DataFrame(exits)
    df["exit_date"] = pd.to_datetime(df["exit_date"])
    daily_returns = df.groupby("exit_date")["return_pct"].mean().sort_index()

    equity_curve = (1 + daily_returns).cumprod()
    cumulative_return = float(equity_curve.iloc[-1] - 1)

    running_max = equity_curve.cummax()
    drawdown = (equity_curve - running_max) / running_max
    max_drawdown = float(drawdown.min())

    std = daily_returns.std(ddof=0)
    sharpe = float(daily_returns.mean() / std * np.sqrt(252)) if std > 0 else None

    return {"cumulative_return": cumulative_return, "max_drawdown": max_drawdown, "sharpe": sharpe, "n_trades": len(exits)}


def spy_buy_and_hold_return(spy_close: pd.Series, start_date: str, end_date: str):
    """Buy-and-hold SPY return over [start_date, end_date] for portfolio comparison."""
    idx = pd.DatetimeIndex(spy_close.index)
    window = spy_close[(idx >= pd.Timestamp(start_date)) & (idx <= pd.Timestamp(end_date))]
    if len(window) < 2:
        return None
    return float(window.iloc[-1] / window.iloc[0] - 1)


def calibration_by_confidence_decile(outcomes: list) -> pd.DataFrame:
    """Does higher confidence actually mean more correct? Bucket hit rate by confidence decile."""
    df = _outcomes_df(outcomes)
    columns = ["decile", "mean_confidence", "hit_rate", "n"]
    if df.empty:
        return pd.DataFrame(columns=columns)

    n_bins = min(10, df["confidence"].nunique())
    if n_bins < 2:
        return pd.DataFrame(columns=columns)

    df = df.copy()
    df["decile"] = pd.qcut(df["confidence"], q=n_bins, duplicates="drop")
    grouped = df.groupby("decile", observed=True).agg(
        mean_confidence=("confidence", "mean"), hit_rate=("correct", "mean"), n=("correct", "size")
    ).reset_index()
    return grouped


def rolling_hit_rate(outcomes: list, window_days: int = 30) -> pd.Series:
    """Rolling hit rate over a trailing calendar window - the series the drift alert watches."""
    df = _outcomes_df(outcomes)
    if df.empty:
        return pd.Series(dtype=float)
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").set_index("date")
    return df["correct"].rolling(f"{window_days}D").mean()


def drift_alert(outcomes: list, always_up_baseline: float, window_days: int = 30, margin: float = 0.03):
    """
    Flag (not act on) model drift: rolling hit rate has fallen below
    (always_up_baseline - margin). Returns None if there isn't enough data or
    no drift, else a dict describing the breach for the report to surface.
    """
    series = rolling_hit_rate(outcomes, window_days=window_days)
    if series.empty or always_up_baseline is None:
        return None
    latest = series.iloc[-1]
    if pd.isna(latest):
        return None
    threshold = always_up_baseline - margin
    if latest < threshold:
        return {
            "triggered": True,
            "latest_rolling_hit_rate": float(latest),
            "threshold": float(threshold),
            "always_up_baseline": always_up_baseline,
            "window_days": window_days,
        }
    return None

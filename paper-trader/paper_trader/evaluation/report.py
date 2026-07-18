from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from paper_trader import config
from paper_trader.evaluation import metrics
from paper_trader.evaluation.outcomes import load_outcomes
from paper_trader.execution.ledger import load_trades

from src.data.prices import fetch_ohlcv


def _fetch_spy_close(start: str, end: str) -> pd.Series:
    try:
        df = fetch_ohlcv(config.EVALUATION_CFG["spy_ticker"], start, end)
        return df["Close"]
    except Exception:
        return pd.Series(dtype=float)


def _plot_equity_curve(trades: list, spy_close: pd.Series, out_path: Path) -> None:
    exits = [t for t in trades if t.get("event") == "exit" and "return_pct" in t]
    fig, ax = plt.subplots(figsize=(8, 4))
    if exits:
        df = pd.DataFrame(exits)
        df["exit_date"] = pd.to_datetime(df["exit_date"])
        daily = df.groupby("exit_date")["return_pct"].mean().sort_index()
        equity = (1 + daily).cumprod()
        ax.plot(equity.index, equity.values, label="strategy")
    if not spy_close.empty:
        spy_norm = spy_close / spy_close.iloc[0]
        ax.plot(spy_norm.index, spy_norm.values, label="SPY buy & hold", linestyle="--")
    ax.set_title("Equity curve vs SPY")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def _plot_rolling_hit_rate(outcomes: list, window_days: int, out_path: Path) -> None:
    series = metrics.rolling_hit_rate(outcomes, window_days=window_days)
    fig, ax = plt.subplots(figsize=(8, 4))
    if not series.empty:
        ax.plot(series.index, series.values)
    ax.axhline(0.5, color="gray", linestyle=":", label="coin flip")
    ax.set_title(f"Rolling {window_days}-day hit rate")
    ax.set_ylim(0, 1)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def _plot_calibration(outcomes: list, out_path: Path) -> None:
    cal = metrics.calibration_by_confidence_decile(outcomes)
    fig, ax = plt.subplots(figsize=(6, 6))
    if not cal.empty:
        ax.plot(cal["mean_confidence"], cal["hit_rate"], marker="o", label="observed")
    ax.plot([0.5, 1.0], [0.5, 1.0], linestyle="--", color="gray", label="perfect calibration")
    ax.set_xlabel("Mean confidence")
    ax.set_ylabel("Hit rate")
    ax.set_title("Calibration by confidence decile")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def _fmt_pct(x):
    return f"{x:.1%}" if x is not None else "n/a"


def _fmt_p(x):
    return f"{x:.3f}" if x is not None else "n/a"


def _strategy_section(strategy_name: str, start_date: str, end_date: str) -> list:
    outcomes = [o for o in load_outcomes(strategy_name) if start_date <= o["date"] <= end_date]
    trades = load_trades(strategy_name)

    all_stats = metrics.hit_rate_vs_baselines(outcomes, traded_only=False)
    traded_stats = metrics.hit_rate_vs_baselines(outcomes, traded_only=True)
    port = metrics.portfolio_metrics(trades)
    drift = metrics.drift_alert(
        outcomes, all_stats["always_up_baseline"],
        window_days=config.EVALUATION_CFG["rolling_window_days"],
        margin=config.EVALUATION_CFG["drift_margin"],
    )

    lines = [f"\n## Strategy: {strategy_name}\n"]

    if drift is not None:
        lines.append(
            f"\n> **DRIFT ALERT**: rolling {drift['window_days']}-day hit rate "
            f"({drift['latest_rolling_hit_rate']:.1%}) has fallen below the always-up "
            f"baseline minus margin ({drift['threshold']:.1%}). Flagged for human review - "
            f"no auto-retraining will occur.\n"
        )

    lines.append(
        f"\n- All predictions: n={all_stats['n']}, hit rate={_fmt_pct(all_stats['hit_rate'])}, "
        f"coin-flip baseline=50.0% (p={_fmt_p(all_stats['coin_flip_p_value'])}), "
        f"always-up baseline={_fmt_pct(all_stats['always_up_baseline'])} "
        f"(p={_fmt_p(all_stats['always_up_p_value'])})\n"
    )
    lines.append(
        f"- Traded-only: n={traded_stats['n']}, hit rate={_fmt_pct(traded_stats['hit_rate'])}, "
        f"always-up baseline={_fmt_pct(traded_stats['always_up_baseline'])} "
        f"(p={_fmt_p(traded_stats['always_up_p_value'])})\n"
    )
    if port["sharpe"] is not None:
        lines.append(
            f"- Portfolio: cumulative return={_fmt_pct(port['cumulative_return'])}, "
            f"max drawdown={_fmt_pct(port['max_drawdown'])}, Sharpe={port['sharpe']:.2f} "
            f"(n={port['n_trades']} closed trades)\n"
        )
    else:
        lines.append("- Portfolio: n/a (no closed trades yet)\n")

    if all_stats["n"] > 0 and all_stats["always_up_baseline"] is not None:
        p = all_stats["always_up_p_value"]
        significant = p is not None and p < 0.05
        beats_baseline = all_stats["hit_rate"] > all_stats["always_up_baseline"]
        verdict = (
            "evidence of edge at this horizon (significant, beats baseline)."
            if beats_baseline and significant
            else "no evidence of edge at this horizon."
        )
        lines.append(
            f"\n**Conclusion**: hit rate {all_stats['hit_rate']:.1%} vs. always-up baseline "
            f"{_fmt_pct(all_stats['always_up_baseline'])}, p={_fmt_p(p)} - {verdict}\n"
        )
    else:
        lines.append("\n**Conclusion**: not enough resolved outcomes yet to evaluate.\n")

    img_prefix = strategy_name.replace("/", "_")
    _plot_equity_curve(trades, _fetch_spy_close(start_date, end_date), config.REPORTS_DIR / f"{img_prefix}_equity.png")
    _plot_rolling_hit_rate(outcomes, config.EVALUATION_CFG["rolling_window_days"], config.REPORTS_DIR / f"{img_prefix}_rolling_hit_rate.png")
    _plot_calibration(outcomes, config.REPORTS_DIR / f"{img_prefix}_calibration.png")

    lines.append(f"\n![equity]({img_prefix}_equity.png)\n")
    lines.append(f"![rolling hit rate]({img_prefix}_rolling_hit_rate.png)\n")
    lines.append(f"![calibration]({img_prefix}_calibration.png)\n")

    return lines


def generate_report(start_date: str, end_date: str) -> Path:
    """
    Regenerate reports/latest.md, comparing every configured strategy on the
    identical [start_date, end_date] range. States conclusions plainly,
    including negative ones - no cherry-picked windows.
    """
    lines = [f"# Paper Trading Report\n\nDate range: {start_date} to {end_date}\n"]
    for cfg in config.STRATEGIES_CFG:
        lines.extend(_strategy_section(cfg["name"], start_date, end_date))

    report_path = config.REPORTS_DIR / "latest.md"
    report_path.write_text("\n".join(lines))
    return report_path

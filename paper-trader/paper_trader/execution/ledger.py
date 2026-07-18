import json
from datetime import datetime, timezone

from paper_trader import config


def load_trades(strategy_name: str) -> list:
    path = config.trades_log_path(strategy_name)
    if not path.exists():
        return []
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def append_trade(strategy_name: str, record: dict) -> None:
    """
    Append one entry or exit event to logs/trades/{strategy}.jsonl. Entry
    records: entry_date, ticker, side, qty, fill_price, notional,
    prediction_id (links back to logs/predictions), scheduled_exit_date.
    Exit records: the same keys plus event="exit" and exit_date/fill_price.
    """
    path = config.trades_log_path(strategy_name)
    path.parent.mkdir(parents=True, exist_ok=True)
    record = dict(record)
    record.setdefault("logged_at_utc", datetime.now(timezone.utc).isoformat())
    with open(path, "a") as f:
        f.write(json.dumps(record) + "\n")


def already_traded_today(strategy_name: str, ticker: str, entry_date: str) -> bool:
    """
    Idempotency check: has this (strategy, ticker, entry_date) already produced
    an entry trade? Re-running the pipeline for the same date must not
    duplicate trades.
    """
    for trade in load_trades(strategy_name):
        if trade.get("event") == "entry" and trade.get("ticker") == ticker and trade.get("entry_date") == entry_date:
            return True
    return False


def open_positions(strategy_name: str) -> dict:
    """
    Reconstruct currently-open positions from the trade log: an entry without
    a matching exit for the same (ticker, entry_date) is still open.
    Returns {ticker: {qty, entry_date, entry_price, scheduled_exit_date, side, notional, prediction_id}}.
    """
    open_by_key = {}
    for trade in load_trades(strategy_name):
        key = (trade["ticker"], trade["entry_date"])
        if trade.get("event") == "entry":
            open_by_key[key] = trade
        elif trade.get("event") == "exit":
            open_by_key.pop(key, None)

    positions = {}
    for (ticker, _), trade in open_by_key.items():
        positions[ticker] = {
            "qty": trade["qty"],
            "entry_date": trade["entry_date"],
            "entry_price": trade["fill_price"],
            "scheduled_exit_date": trade["scheduled_exit_date"],
            "side": trade["side"],
            "notional": trade["notional"],
            "prediction_id": trade.get("prediction_id"),
        }
    return positions


def positions_due_for_exit(strategy_name: str, as_of_date: str) -> list:
    """Mechanical, time-based exit selection - any open position whose scheduled_exit_date has arrived."""
    due = []
    for ticker, pos in open_positions(strategy_name).items():
        if pos["scheduled_exit_date"] <= as_of_date:
            due.append({"ticker": ticker, **pos})
    return due


def compute_virtual_equity(strategy_name: str, starting_equity: float, current_prices: dict) -> dict:
    """
    Per-strategy virtual sub-ledger equity: starting_equity + realized P&L from
    closed round trips + unrealized P&L on currently open positions (marked at
    current_prices). Alpaca exposes only one paper account, so every strategy
    (including log-only ones) tracks its own equity locally from its own
    trade log, letting strategies be compared on identical terms.
    """
    trades = load_trades(strategy_name)
    realized_pnl = 0.0
    cash = starting_equity
    entries_by_key = {}

    for trade in trades:
        key = (trade["ticker"], trade["entry_date"])
        if trade.get("event") == "entry":
            entries_by_key[key] = trade
            cash -= trade["notional"]
        elif trade.get("event") == "exit":
            entry = entries_by_key.get(key, {})
            entry_notional = entry.get("notional", 0.0)
            exit_notional = trade.get("notional", 0.0)
            realized_pnl += exit_notional - entry_notional
            cash += exit_notional

    unrealized_pnl = 0.0
    for ticker, pos in open_positions(strategy_name).items():
        price = current_prices.get(ticker)
        if price is None:
            continue
        unrealized_pnl += (price - pos["entry_price"]) * pos["qty"]

    equity = starting_equity + realized_pnl + unrealized_pnl
    return {"equity": equity, "cash": cash, "realized_pnl": realized_pnl, "unrealized_pnl": unrealized_pnl}

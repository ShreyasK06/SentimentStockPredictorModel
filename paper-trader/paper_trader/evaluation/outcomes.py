import json

import pandas as pd

from paper_trader import config
from paper_trader.models.prediction_log import load_predictions
from paper_trader.strategy.decision import compute_exit_date

from src.data.storage import load_parquet


def _get_close_price(ticker: str, date: str):
    """
    Look up the realized close price for a ticker on a date, checking the
    research repo's raw price cache first and the paper-trader's live price
    cache second - the same two sources feature_store combines for training.
    """
    from src.config import PRICES_DIR
    research = load_parquet(ticker, PRICES_DIR).reset_index()
    live_path = config.DATA_DIR / "live_prices" / f"{ticker}.parquet"
    frames = [research]
    if live_path.exists():
        frames.append(pd.read_parquet(live_path))
    combined = pd.concat(frames, ignore_index=True).drop_duplicates(subset=["date"])
    combined["date"] = pd.to_datetime(combined["date"])
    match = combined[combined["date"] == pd.Timestamp(date)]
    if match.empty:
        return None
    return float(match.iloc[0]["Close"])


def load_outcomes(strategy_name: str) -> list:
    path = config.outcomes_log_path(strategy_name)
    if not path.exists():
        return []
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _already_resolved(strategy_name: str) -> set:
    return {(o["date"], o["ticker"]) for o in load_outcomes(strategy_name)}


def resolve_outcomes(strategy_name: str, as_of_date: str, traded_keys: set = None) -> list:
    """
    For every logged prediction whose horizon has elapsed by as_of_date and
    that hasn't already been resolved, look up the realized close price at
    entry and exit and record correct/incorrect. Appends to
    logs/outcomes/{strategy}.jsonl and returns the newly-resolved records.
    traded_keys is an optional set of (date, ticker) that became real trades,
    so the metrics module can separate traded-only hit rate from all-predictions.
    """
    traded_keys = traded_keys or set()
    resolved_keys = _already_resolved(strategy_name)
    predictions = load_predictions(strategy_name)
    path = config.outcomes_log_path(strategy_name)
    path.parent.mkdir(parents=True, exist_ok=True)

    new_outcomes = []
    with open(path, "a") as f:
        for pred in predictions:
            key = (pred["date"], pred["ticker"])
            if key in resolved_keys:
                continue

            try:
                exit_date = compute_exit_date(pred["date"], pred["horizon"])
            except ValueError:
                continue
            if exit_date > as_of_date:
                continue  # horizon hasn't elapsed yet - not resolvable today

            entry_price = _get_close_price(pred["ticker"], pred["date"])
            exit_price = _get_close_price(pred["ticker"], exit_date)
            if entry_price is None or exit_price is None:
                continue

            realized_direction = "up" if exit_price >= entry_price else "down"
            correct = realized_direction == pred["direction"]

            record = {
                "date": pred["date"],
                "exit_date": exit_date,
                "ticker": pred["ticker"],
                "strategy": strategy_name,
                "predicted_direction": pred["direction"],
                "confidence": pred["confidence"],
                "realized_direction": realized_direction,
                "correct": correct,
                "entry_price": entry_price,
                "exit_price": exit_price,
                "return_pct": (exit_price - entry_price) / entry_price,
                "traded": key in traded_keys,
            }
            f.write(json.dumps(record) + "\n")
            new_outcomes.append(record)
            resolved_keys.add(key)

    return new_outcomes

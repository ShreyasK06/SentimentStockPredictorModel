def should_trade(direction_pred: dict, sentiment_pred: dict, confidence_threshold: float, allow_short: bool = False) -> dict:
    """
    Decision rule: trade ticker T iff confidence >= threshold AND
    sentiment.direction == model.direction. Long on up-signals; down-signals
    are skipped by default (allow_short=False) rather than opened as shorts.

    Returns {"should_trade": bool, "side": "buy"|"sell_short"|None, "reason": str}.
    """
    if direction_pred is None or sentiment_pred is None:
        return {"should_trade": False, "side": None, "reason": "missing_prediction"}

    confidence = direction_pred["confidence"]
    direction = direction_pred["direction"]
    sentiment_direction = sentiment_pred["direction"]

    if confidence < confidence_threshold:
        return {"should_trade": False, "side": None, "reason": "below_confidence_threshold"}

    if sentiment_direction != direction:
        return {"should_trade": False, "side": None, "reason": "sentiment_disagreement"}

    if direction == "up":
        return {"should_trade": True, "side": "buy", "reason": "confident_agreement_up"}

    if allow_short:
        return {"should_trade": True, "side": "sell_short", "reason": "confident_agreement_down"}
    return {"should_trade": False, "side": None, "reason": "down_signal_skipped_no_shorting"}


def compute_position_size(equity: float, position_size_fraction: float, cash_available: float, price: float) -> dict:
    """
    Fixed fraction of equity per position, sized to whole shares, never
    exceeding available cash. Returns {"shares": int, "notional": float}.
    """
    if price <= 0 or equity <= 0 or cash_available <= 0:
        return {"shares": 0, "notional": 0.0}

    target_notional = equity * position_size_fraction
    notional = min(target_notional, cash_available)
    shares = int(notional // price)
    return {"shares": shares, "notional": round(shares * price, 2)}


def select_new_positions(candidates: list, max_positions: int, current_open_count: int) -> list:
    """
    Cap how many NEW positions can open this run. `candidates` should already
    be ordered by the caller's priority (e.g. confidence descending) - this
    function only enforces the max_positions ceiling, it does not sort.
    """
    slots = max(0, max_positions - current_open_count)
    return candidates[:slots]


def compute_exit_date(entry_date: str, horizon_days: int) -> str:
    """
    Time-based, mechanical exit: the horizon-th NYSE trading day strictly
    after entry_date. No discretionary logic - exits are always scheduled,
    never judged.
    """
    import pandas as pd
    from src.data.calendar import get_nyse_trading_days

    entry_ts = pd.Timestamp(entry_date)
    # Calendar-day search window generous enough to contain horizon_days
    # trading days even across holidays/weekends.
    search_end = (entry_ts + pd.Timedelta(days=horizon_days * 2 + 15)).date().isoformat()
    trading_days = get_nyse_trading_days(entry_date, search_end)
    future_days = trading_days[trading_days > entry_ts]
    if len(future_days) < horizon_days:
        raise ValueError(f"Not enough trading days after {entry_date} to compute a {horizon_days}-day exit")
    return future_days[horizon_days - 1].date().isoformat()

import time

from paper_trader import config


class OrderUnfilledError(Exception):
    """Raised by poll_fill's caller-visible status when an order times out unfilled (not raised - see poll_fill)."""


class AlpacaClient:
    """
    Thin wrapper around alpaca-py's paper trading endpoint. The SDK is
    imported lazily inside __init__ (not at module load time) so the rest of
    the pipeline - including --dry-run runs and unit tests - can import this
    module without alpaca-py installed or credentials configured.
    """

    def __init__(self, key_id: str = None, secret_key: str = None):
        key_id = key_id or config.ALPACA_KEY_ID
        secret_key = secret_key or config.ALPACA_SECRET_KEY
        if not key_id or not secret_key:
            raise ValueError("ALPACA_KEY_ID / ALPACA_SECRET_KEY are not configured")

        from alpaca.trading.client import TradingClient
        self._client = TradingClient(key_id, secret_key, paper=True)

    def get_account(self) -> dict:
        account = self._client.get_account()
        return {
            "equity": float(account.equity),
            "cash": float(account.cash),
            "buying_power": float(account.buying_power),
        }

    def get_open_positions(self) -> dict:
        positions = self._client.get_all_positions()
        return {p.symbol: {"qty": float(p.qty), "side": p.side.value, "avg_entry_price": float(p.avg_entry_price)} for p in positions}

    def is_market_open_today(self, as_of_date: str) -> bool:
        from alpaca.trading.requests import GetCalendarRequest

        req = GetCalendarRequest(start=as_of_date, end=as_of_date)
        days = self._client.get_calendar(req)
        return len(days) > 0

    def submit_market_order(self, symbol: str, qty: int, side: str) -> dict:
        from alpaca.trading.enums import OrderSide, TimeInForce
        from alpaca.trading.requests import MarketOrderRequest

        order_side = OrderSide.BUY if side == "buy" else OrderSide.SELL
        req = MarketOrderRequest(symbol=symbol, qty=qty, side=order_side, time_in_force=TimeInForce.DAY)
        order = self._client.submit_order(req)
        return {"order_id": str(order.id), "symbol": symbol, "qty": qty, "side": side, "submitted_at": str(order.submitted_at)}

    def poll_fill(self, order_id: str, timeout_seconds: int = 300, poll_interval_seconds: int = 5) -> dict:
        """
        Poll an order until it fills or the timeout elapses. On timeout, cancel
        the order and return order_status="unfilled" rather than raising -
        callers log this status per the Phase 3 spec.
        """
        deadline = time.monotonic() + timeout_seconds
        last_status = None
        while time.monotonic() < deadline:
            order = self._client.get_order_by_id(order_id)
            last_status = order.status.value if hasattr(order.status, "value") else str(order.status)
            if last_status == "filled":
                return {
                    "order_status": "filled",
                    "fill_price": float(order.filled_avg_price) if order.filled_avg_price else None,
                    "filled_at": str(order.filled_at) if order.filled_at else None,
                }
            if last_status in ("canceled", "rejected", "expired"):
                return {"order_status": last_status, "fill_price": None, "filled_at": None}
            time.sleep(poll_interval_seconds)

        self._client.cancel_order_by_id(order_id)
        return {"order_status": "unfilled", "fill_price": None, "filled_at": None}


def reconcile_positions(local_positions: dict, alpaca_positions: dict) -> dict:
    """
    Compare the local position ledger to Alpaca's reported account state.
    Returns {"matched": bool, "mismatches": [...]}, one mismatch entry per
    ticker whose local qty disagrees with Alpaca's, including tickers present
    on only one side.
    """
    mismatches = []
    all_symbols = set(local_positions) | set(alpaca_positions)
    for symbol in sorted(all_symbols):
        local_qty = local_positions.get(symbol, {}).get("qty", 0)
        alpaca_qty = alpaca_positions.get(symbol, {}).get("qty", 0)
        if float(local_qty) != float(alpaca_qty):
            mismatches.append({"symbol": symbol, "local_qty": local_qty, "alpaca_qty": alpaca_qty})

    return {"matched": len(mismatches) == 0, "mismatches": mismatches}

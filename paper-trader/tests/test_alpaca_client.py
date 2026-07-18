"""
Integration test with the Alpaca paper API mocked via fake modules injected
into sys.modules. This stands in for "recorded responses": FakeTradingClient
returns scripted account/order/position data so AlpacaClient's fill-polling,
timeout/cancel, and reconciliation logic can be exercised end-to-end without
real network calls or real alpaca-py credentials.
"""
import sys
import types

import pytest


class FakeOrder:
    def __init__(self, order_id, symbol, qty, side):
        self.id = order_id
        self.symbol = symbol
        self.qty = qty
        self.side = types.SimpleNamespace(value=side)
        self.status = types.SimpleNamespace(value="new")
        self.filled_avg_price = None
        self.filled_at = None
        self.submitted_at = "2024-01-02T14:30:00Z"


class FakeAccount:
    def __init__(self, equity, cash, buying_power):
        self.equity = equity
        self.cash = cash
        self.buying_power = buying_power


class FakeTradingClient:
    """Stands in for alpaca-py's TradingClient with scripted, recorded-style responses."""

    def __init__(self, key_id, secret_key, paper=True):
        self.key_id = key_id
        self.secret_key = secret_key
        self.paper = paper
        self.orders = {}
        self.next_order_status = "filled"
        self.next_fill_price = 150.0
        self.positions = []
        self.calendar_days = ["2024-01-02"]
        self.canceled_order_ids = []

    def get_account(self):
        return FakeAccount(equity=100000.0, cash=50000.0, buying_power=50000.0)

    def get_all_positions(self):
        return self.positions

    def get_calendar(self, req):
        return self.calendar_days

    def submit_order(self, req):
        order_id = f"order-{len(self.orders) + 1}"
        side_value = req.side.value if hasattr(req.side, "value") else req.side
        self.orders[order_id] = FakeOrder(order_id, req.symbol, req.qty, side_value)
        return self.orders[order_id]

    def get_order_by_id(self, order_id):
        order = self.orders[order_id]
        order.status = types.SimpleNamespace(value=self.next_order_status)
        if self.next_order_status == "filled":
            order.filled_avg_price = self.next_fill_price
            order.filled_at = "2024-01-02T14:30:05Z"
        return order

    def cancel_order_by_id(self, order_id):
        self.canceled_order_ids.append(order_id)


class FakeOrderSide:
    BUY = types.SimpleNamespace(value="buy")
    SELL = types.SimpleNamespace(value="sell")


class FakeTimeInForce:
    DAY = "day"


class FakeMarketOrderRequest:
    def __init__(self, symbol, qty, side, time_in_force):
        self.symbol, self.qty, self.side, self.time_in_force = symbol, qty, side, time_in_force


class FakeGetCalendarRequest:
    def __init__(self, start, end):
        self.start, self.end = start, end


@pytest.fixture
def fake_alpaca(monkeypatch):
    modules = {
        "alpaca": types.ModuleType("alpaca"),
        "alpaca.trading": types.ModuleType("alpaca.trading"),
        "alpaca.trading.client": types.ModuleType("alpaca.trading.client"),
        "alpaca.trading.requests": types.ModuleType("alpaca.trading.requests"),
        "alpaca.trading.enums": types.ModuleType("alpaca.trading.enums"),
    }
    modules["alpaca.trading.client"].TradingClient = FakeTradingClient
    modules["alpaca.trading.requests"].MarketOrderRequest = FakeMarketOrderRequest
    modules["alpaca.trading.requests"].GetCalendarRequest = FakeGetCalendarRequest
    modules["alpaca.trading.enums"].OrderSide = FakeOrderSide
    modules["alpaca.trading.enums"].TimeInForce = FakeTimeInForce

    for name, mod in modules.items():
        monkeypatch.setitem(sys.modules, name, mod)
    return modules


def _client():
    from paper_trader.execution.alpaca_client import AlpacaClient
    return AlpacaClient(key_id="fake-key", secret_key="fake-secret")


def test_no_credentials_raises_before_touching_the_sdk(monkeypatch):
    from paper_trader.execution.alpaca_client import AlpacaClient
    with pytest.raises(ValueError):
        AlpacaClient(key_id="", secret_key="")


def test_get_account_returns_expected_fields(fake_alpaca):
    account = _client().get_account()
    assert account == {"equity": 100000.0, "cash": 50000.0, "buying_power": 50000.0}


def test_market_open_check_uses_calendar(fake_alpaca):
    client = _client()
    assert client.is_market_open_today("2024-01-02") is True

    client._client.calendar_days = []
    assert client.is_market_open_today("2024-01-01") is False


def test_submit_and_poll_fill_immediate(fake_alpaca):
    client = _client()
    order = client.submit_market_order("AAPL", 10, "buy")
    fill = client.poll_fill(order["order_id"], timeout_seconds=5, poll_interval_seconds=0)
    assert fill == {"order_status": "filled", "fill_price": 150.0, "filled_at": "2024-01-02T14:30:05Z"}


def test_poll_fill_times_out_and_cancels(fake_alpaca):
    client = _client()
    client._client.next_order_status = "pending_new"
    order = client.submit_market_order("AAPL", 10, "buy")

    fill = client.poll_fill(order["order_id"], timeout_seconds=0.3, poll_interval_seconds=0.1)

    assert fill["order_status"] == "unfilled"
    assert fill["fill_price"] is None
    assert order["order_id"] in client._client.canceled_order_ids


def test_reconcile_detects_mismatch(fake_alpaca):
    from paper_trader.execution.alpaca_client import reconcile_positions
    local = {"AAPL": {"qty": 10}}
    alpaca_positions = {"AAPL": {"qty": 5}, "MSFT": {"qty": 3}}

    result = reconcile_positions(local, alpaca_positions)

    assert result["matched"] is False
    assert {"symbol": "AAPL", "local_qty": 10, "alpaca_qty": 5} in result["mismatches"]
    assert {"symbol": "MSFT", "local_qty": 0, "alpaca_qty": 3} in result["mismatches"]


def test_reconcile_matches_when_equal(fake_alpaca):
    from paper_trader.execution.alpaca_client import reconcile_positions
    assert reconcile_positions({"AAPL": {"qty": 10}}, {"AAPL": {"qty": 10}})["matched"] is True

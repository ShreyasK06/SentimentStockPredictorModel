from paper_trader.execution import ledger


def _entry(ticker="AAPL", entry_date="2024-01-02", qty=10, fill_price=100.0, notional=1000.0, scheduled_exit_date="2024-01-16"):
    return {
        "event": "entry", "ticker": ticker, "entry_date": entry_date, "side": "buy",
        "qty": qty, "fill_price": fill_price, "notional": notional,
        "prediction_id": f"{entry_date}:{ticker}:direction-daily", "scheduled_exit_date": scheduled_exit_date,
    }


def test_no_trade_logged_is_not_already_traded(isolated_logs):
    assert ledger.already_traded_today("direction-daily", "AAPL", "2024-01-02") is False


def test_logged_entry_is_detected_as_already_traded(isolated_logs):
    ledger.append_trade("direction-daily", _entry())
    assert ledger.already_traded_today("direction-daily", "AAPL", "2024-01-02") is True


def test_rerunning_same_date_does_not_duplicate(isolated_logs):
    for _ in range(3):
        if not ledger.already_traded_today("direction-daily", "AAPL", "2024-01-02"):
            ledger.append_trade("direction-daily", _entry())
    trades = ledger.load_trades("direction-daily")
    entries = [t for t in trades if t["event"] == "entry"]
    assert len(entries) == 1


def test_idempotency_is_scoped_to_ticker_and_date(isolated_logs):
    ledger.append_trade("direction-daily", _entry(ticker="AAPL", entry_date="2024-01-02"))
    assert ledger.already_traded_today("direction-daily", "MSFT", "2024-01-02") is False
    assert ledger.already_traded_today("direction-daily", "AAPL", "2024-01-03") is False


def test_idempotency_is_scoped_to_strategy(isolated_logs):
    ledger.append_trade("direction-daily", _entry())
    assert ledger.already_traded_today("volatility", "AAPL", "2024-01-02") is False


def test_open_positions_excludes_exited_trades(isolated_logs):
    ledger.append_trade("direction-daily", _entry())
    assert "AAPL" in ledger.open_positions("direction-daily")

    ledger.append_trade("direction-daily", {
        "event": "exit", "ticker": "AAPL", "entry_date": "2024-01-02", "exit_date": "2024-01-16",
        "side": "buy", "qty": 10, "fill_price": 110.0, "notional": 1100.0,
        "entry_fill_price": 100.0, "return_pct": 0.10,
    })
    assert "AAPL" not in ledger.open_positions("direction-daily")


def test_positions_due_for_exit_respects_scheduled_date(isolated_logs):
    ledger.append_trade("direction-daily", _entry(scheduled_exit_date="2024-01-16"))
    assert ledger.positions_due_for_exit("direction-daily", "2024-01-10") == []
    due = ledger.positions_due_for_exit("direction-daily", "2024-01-16")
    assert len(due) == 1 and due[0]["ticker"] == "AAPL"

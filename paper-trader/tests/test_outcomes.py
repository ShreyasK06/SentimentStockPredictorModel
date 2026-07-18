import pytest

import paper_trader.evaluation.outcomes as outcomes_mod
from paper_trader.models import prediction_log


def _prediction(direction="up", confidence=0.7, horizon=10):
    return {
        "direction": direction, "confidence": confidence,
        "proba_up": confidence if direction == "up" else 1 - confidence,
        "horizon": horizon, "model_version": "abc123", "data_window_hash": "hash",
    }


def _log_one_prediction(direction="up"):
    prediction_log.log_predictions("direction-daily", "2024-01-02", {"AAPL": _prediction(direction=direction)}, {})


def test_correct_prediction_resolves_true(isolated_logs, monkeypatch):
    _log_one_prediction("up")
    monkeypatch.setattr(outcomes_mod, "compute_exit_date", lambda entry_date, horizon: "2024-01-16")
    prices = {("AAPL", "2024-01-02"): 100.0, ("AAPL", "2024-01-16"): 110.0}
    monkeypatch.setattr(outcomes_mod, "_get_close_price", lambda ticker, date: prices.get((ticker, date)))

    resolved = outcomes_mod.resolve_outcomes("direction-daily", "2024-01-16")
    assert len(resolved) == 1
    assert resolved[0]["correct"] is True
    assert resolved[0]["realized_direction"] == "up"
    assert resolved[0]["return_pct"] == pytest.approx(0.10)


def test_incorrect_prediction_resolves_false(isolated_logs, monkeypatch):
    _log_one_prediction("up")
    monkeypatch.setattr(outcomes_mod, "compute_exit_date", lambda entry_date, horizon: "2024-01-16")
    prices = {("AAPL", "2024-01-02"): 100.0, ("AAPL", "2024-01-16"): 90.0}
    monkeypatch.setattr(outcomes_mod, "_get_close_price", lambda ticker, date: prices.get((ticker, date)))

    resolved = outcomes_mod.resolve_outcomes("direction-daily", "2024-01-16")
    assert resolved[0]["correct"] is False
    assert resolved[0]["realized_direction"] == "down"


def test_unelapsed_horizon_is_not_resolved(isolated_logs, monkeypatch):
    _log_one_prediction("up")
    monkeypatch.setattr(outcomes_mod, "compute_exit_date", lambda entry_date, horizon: "2024-01-16")
    monkeypatch.setattr(outcomes_mod, "_get_close_price", lambda ticker, date: 100.0)

    resolved = outcomes_mod.resolve_outcomes("direction-daily", "2024-01-10")  # before exit_date
    assert resolved == []


def test_already_resolved_is_not_duplicated(isolated_logs, monkeypatch):
    _log_one_prediction("up")
    monkeypatch.setattr(outcomes_mod, "compute_exit_date", lambda entry_date, horizon: "2024-01-16")
    prices = {("AAPL", "2024-01-02"): 100.0, ("AAPL", "2024-01-16"): 110.0}
    monkeypatch.setattr(outcomes_mod, "_get_close_price", lambda ticker, date: prices.get((ticker, date)))

    first = outcomes_mod.resolve_outcomes("direction-daily", "2024-01-16")
    second = outcomes_mod.resolve_outcomes("direction-daily", "2024-01-16")
    assert len(first) == 1
    assert second == []
    assert len(outcomes_mod.load_outcomes("direction-daily")) == 1


def test_missing_price_data_skips_resolution(isolated_logs, monkeypatch):
    _log_one_prediction("up")
    monkeypatch.setattr(outcomes_mod, "compute_exit_date", lambda entry_date, horizon: "2024-01-16")
    monkeypatch.setattr(outcomes_mod, "_get_close_price", lambda ticker, date: None)

    resolved = outcomes_mod.resolve_outcomes("direction-daily", "2024-01-16")
    assert resolved == []

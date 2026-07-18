from paper_trader.strategy.decision import (
    should_trade, compute_position_size, select_new_positions, compute_exit_date,
)


def _pred(direction="up", confidence=0.7):
    return {"direction": direction, "confidence": confidence}


def _sent(direction="up"):
    return {"direction": direction}


class TestShouldTrade:
    def test_trades_when_confident_and_agreeing_up(self):
        result = should_trade(_pred("up", 0.7), _sent("up"), confidence_threshold=0.6)
        assert result["should_trade"] is True
        assert result["side"] == "buy"

    def test_confidence_exactly_at_threshold_still_trades(self):
        result = should_trade(_pred("up", 0.6), _sent("up"), confidence_threshold=0.6)
        assert result["should_trade"] is True

    def test_confidence_just_below_threshold_does_not_trade(self):
        result = should_trade(_pred("up", 0.599), _sent("up"), confidence_threshold=0.6)
        assert result["should_trade"] is False
        assert result["reason"] == "below_confidence_threshold"

    def test_sentiment_disagreement_blocks_trade(self):
        result = should_trade(_pred("up", 0.9), _sent("down"), confidence_threshold=0.6)
        assert result["should_trade"] is False
        assert result["reason"] == "sentiment_disagreement"

    def test_flat_sentiment_counts_as_disagreement(self):
        result = should_trade(_pred("up", 0.9), _sent("flat"), confidence_threshold=0.6)
        assert result["should_trade"] is False

    def test_down_signal_skipped_by_default(self):
        result = should_trade(_pred("down", 0.9), _sent("down"), confidence_threshold=0.6, allow_short=False)
        assert result["should_trade"] is False
        assert result["reason"] == "down_signal_skipped_no_shorting"

    def test_down_signal_shorts_when_allowed(self):
        result = should_trade(_pred("down", 0.9), _sent("down"), confidence_threshold=0.6, allow_short=True)
        assert result["should_trade"] is True
        assert result["side"] == "sell_short"

    def test_missing_prediction_never_trades(self):
        assert should_trade(None, _sent("up"), 0.6)["should_trade"] is False
        assert should_trade(_pred(), None, 0.6)["should_trade"] is False


class TestPositionSizing:
    def test_fixed_fraction_of_equity(self):
        sizing = compute_position_size(equity=100_000, position_size_fraction=0.10, cash_available=100_000, price=100)
        assert sizing["shares"] == 100  # 10% of 100k = 10k / $100 = 100 shares
        assert sizing["notional"] == 10_000.0

    def test_never_exceeds_available_cash(self):
        sizing = compute_position_size(equity=100_000, position_size_fraction=0.10, cash_available=500, price=100)
        assert sizing["shares"] == 5
        assert sizing["notional"] <= 500

    def test_zero_cash_yields_zero_shares(self):
        sizing = compute_position_size(equity=100_000, position_size_fraction=0.10, cash_available=0, price=100)
        assert sizing["shares"] == 0

    def test_price_higher_than_target_notional_yields_zero_shares(self):
        sizing = compute_position_size(equity=1000, position_size_fraction=0.10, cash_available=1000, price=500)
        assert sizing["shares"] == 0  # target notional = 100, can't afford even 1 share at $500


class TestSelectNewPositions:
    def test_caps_at_max_positions(self):
        candidates = [{"ticker": t} for t in ["A", "B", "C", "D"]]
        selected = select_new_positions(candidates, max_positions=5, current_open_count=3)
        assert len(selected) == 2
        assert selected == candidates[:2]

    def test_no_slots_left_returns_empty(self):
        candidates = [{"ticker": "A"}]
        assert select_new_positions(candidates, max_positions=3, current_open_count=3) == []

    def test_over_capacity_current_count_does_not_error(self):
        candidates = [{"ticker": "A"}]
        assert select_new_positions(candidates, max_positions=3, current_open_count=10) == []


class TestComputeExitDate:
    def test_exit_date_is_after_entry(self):
        exit_date = compute_exit_date("2024-01-02", horizon_days=10)
        assert exit_date > "2024-01-02"

    def test_exit_date_skips_weekends_and_holidays(self):
        # 2024-01-01 is a market holiday (New Year's Day); entry the day
        # before a long weekend should still land on a real trading day.
        exit_date = compute_exit_date("2023-12-29", horizon_days=1)
        assert exit_date == "2024-01-02"

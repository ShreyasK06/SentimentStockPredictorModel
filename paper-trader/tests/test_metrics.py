from paper_trader.evaluation import metrics


def _outcome(correct, realized_direction, confidence=0.7, traded=False):
    return {"correct": correct, "realized_direction": realized_direction, "confidence": confidence, "traded": traded}


def test_hit_rate_vs_baselines_empty_input():
    result = metrics.hit_rate_vs_baselines([])
    assert result["hit_rate"] is None
    assert result["n"] == 0


def test_hit_rate_matches_manual_fraction():
    outcomes = [_outcome(True, "up")] * 6 + [_outcome(False, "down")] * 4
    result = metrics.hit_rate_vs_baselines(outcomes)
    assert result["hit_rate"] == 0.6
    assert result["n"] == 10


def test_always_up_baseline_is_fraction_of_up_moves():
    outcomes = [_outcome(True, "up")] * 7 + [_outcome(True, "down")] * 3
    result = metrics.hit_rate_vs_baselines(outcomes)
    assert result["always_up_baseline"] == 0.7


def test_traded_only_filters_correctly():
    outcomes = [_outcome(True, "up", traded=True), _outcome(False, "down", traded=False)]
    all_stats = metrics.hit_rate_vs_baselines(outcomes, traded_only=False)
    traded_stats = metrics.hit_rate_vs_baselines(outcomes, traded_only=True)
    assert all_stats["n"] == 2
    assert traded_stats["n"] == 1
    assert traded_stats["hit_rate"] == 1.0


def test_binomial_significance_perfect_match_is_not_significant():
    # exactly at baseline rate -> high p-value, not significant
    p = metrics.binomial_significance(n_correct=50, n_total=100, baseline=0.5)
    assert p is not None
    assert p > 0.5


def test_binomial_significance_extreme_deviation_is_significant():
    p = metrics.binomial_significance(n_correct=95, n_total=100, baseline=0.5)
    assert p is not None
    assert p < 0.001


def test_binomial_significance_handles_degenerate_baseline():
    assert metrics.binomial_significance(10, 10, baseline=1.0) is None
    assert metrics.binomial_significance(0, 0, baseline=0.5) is None


def test_calibration_returns_empty_frame_for_no_data():
    df = metrics.calibration_by_confidence_decile([])
    assert df.empty


def test_drift_alert_none_when_above_threshold():
    import pandas as pd
    base_date = pd.Timestamp("2024-01-01")
    outcomes = [
        {**_outcome(True, "up", confidence=0.8), "date": (base_date + pd.Timedelta(days=i)).date().isoformat()}
        for i in range(40)
    ]
    alert = metrics.drift_alert(outcomes, always_up_baseline=0.55, window_days=30, margin=0.03)
    assert alert is None


def test_drift_alert_triggers_when_rolling_hit_rate_collapses():
    import pandas as pd
    outcomes = []
    base_date = pd.Timestamp("2024-01-01")
    for i in range(40):
        outcomes.append({
            "date": (base_date + pd.Timedelta(days=i)).date().isoformat(),
            "correct": False, "realized_direction": "up", "confidence": 0.7, "traded": False,
        })
    alert = metrics.drift_alert(outcomes, always_up_baseline=0.55, window_days=30, margin=0.03)
    assert alert is not None
    assert alert["triggered"] is True

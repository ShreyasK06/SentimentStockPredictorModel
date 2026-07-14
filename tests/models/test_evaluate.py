import numpy as np
import pandas as pd
import pytest
from src.models.evaluate import (
    majority_baseline, accuracy_at_threshold, breakdown_by_year,
    breakdown_by_regime, sweep_confidence_threshold,
)


def test_majority_baseline_computes_larger_class_proportion():
    y = pd.Series([1.0, 1.0, 1.0, 0.0])
    assert majority_baseline(y) == pytest.approx(0.75)


def test_accuracy_at_threshold_default_uses_all_predictions():
    y_true = pd.Series([1.0, 0.0, 1.0, 0.0])
    proba = np.array([0.9, 0.1, 0.4, 0.6])  # predictions: 1,0,0,1 at threshold 0.5
    result = accuracy_at_threshold(y_true, proba, threshold=0.5)
    assert result["accuracy"] == pytest.approx(0.5)  # 2/4 correct
    assert result["coverage"] == pytest.approx(1.0)
    assert result["n_predicted"] == 4


def test_accuracy_at_threshold_abstains_below_confidence():
    y_true = pd.Series([1.0, 0.0, 1.0, 0.0])
    proba = np.array([0.9, 0.1, 0.55, 0.45])  # only first two are confident at 0.6
    result = accuracy_at_threshold(y_true, proba, threshold=0.6)
    assert result["n_predicted"] == 2
    assert result["accuracy"] == pytest.approx(1.0)  # both confident predictions correct
    assert result["coverage"] == pytest.approx(0.5)


def test_breakdown_by_year_groups_correctly():
    idx = pd.DatetimeIndex(["2022-01-01", "2022-06-01", "2023-01-01"])
    y_true = pd.Series([1.0, 0.0, 1.0], index=idx)
    proba = np.array([0.9, 0.1, 0.9])
    result = breakdown_by_year(y_true, proba, idx, threshold=0.5)
    assert set(result.index) == {2022, 2023}
    assert result.loc[2022, "n"] == 2
    assert result.loc[2022, "accuracy"] == pytest.approx(1.0)
    assert "majority_baseline" in result.columns


def test_breakdown_by_regime_groups_correctly():
    idx = pd.DatetimeIndex(["2022-01-01", "2021-06-01"])  # 2022=down, 2021=bull
    y_true = pd.Series([1.0, 0.0], index=idx)
    proba = np.array([0.9, 0.1])
    result = breakdown_by_regime(y_true, proba, idx, threshold=0.5)
    assert set(result.index) == {"down", "bull"}


def test_sweep_confidence_threshold_returns_one_row_per_threshold():
    y_true = pd.Series([1.0, 0.0, 1.0, 0.0])
    proba = np.array([0.9, 0.1, 0.65, 0.35])
    result = sweep_confidence_threshold(y_true, proba, thresholds=(0.5, 0.6))
    assert list(result.index) == [0.5, 0.6]
    assert "accuracy" in result.columns
    assert "coverage" in result.columns

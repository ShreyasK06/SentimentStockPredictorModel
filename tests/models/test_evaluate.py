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
    # At threshold=0.5 the confident subset is the entire set (balanced 2/2),
    # so the local majority baseline should equal the full-set majority baseline.
    assert result["local_majority_baseline"] == pytest.approx(0.5)


def test_accuracy_at_threshold_abstains_below_confidence():
    y_true = pd.Series([1.0, 0.0, 1.0, 0.0])
    proba = np.array([0.9, 0.1, 0.55, 0.45])  # only first two are confident at 0.6
    result = accuracy_at_threshold(y_true, proba, threshold=0.6)
    assert result["n_predicted"] == 2
    assert result["accuracy"] == pytest.approx(1.0)  # both confident predictions correct
    assert result["coverage"] == pytest.approx(0.5)
    # Confident subset is [1.0, 0.0] -> perfectly balanced -> local majority baseline 0.5
    assert result["local_majority_baseline"] == pytest.approx(0.5)


def test_accuracy_at_threshold_local_majority_baseline_reflects_confident_subset_skew():
    # Full set is roughly balanced (5 of class 1, 5 of class 0), but the "confident"
    # subset (rows with extreme proba, |proba - 0.5| large) is skewed 4:1 toward class 1.
    # This directly reproduces the bug: comparing confident-subset accuracy against the
    # FULL-set majority baseline (0.5) is invalid when the confident subset is skewed
    # (local majority baseline should be 0.8, not 0.5).
    y_true = pd.Series([1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
    proba = np.array([
        0.95, 0.92, 0.91, 0.90,  # 4 confident rows, all class 1 (correctly predicted)
        0.05,                     # 1 confident row, class 0 (correctly predicted) -> 5 confident, 4/5 are class 1
        0.55, 0.52, 0.51, 0.49,   # not confident at threshold 0.9
        0.48,                     # not confident
    ])
    result = accuracy_at_threshold(y_true, proba, threshold=0.9)
    assert result["n_predicted"] == 5
    assert result["local_majority_baseline"] == pytest.approx(0.8)


def test_breakdown_by_year_groups_correctly():
    # Test with different label balance per year to verify per-subset majority_baseline
    # 2022: y=[1.0, 0.0] -> majority_baseline = 0.5 (balanced)
    # 2023: y=[1.0, 1.0, 0.0] -> majority_baseline = 2/3 (imbalanced, DIFFERENT from 2022)
    idx = pd.DatetimeIndex(["2022-01-01", "2022-06-01", "2023-01-01", "2023-02-01", "2023-03-01"])
    y_true = pd.Series([1.0, 0.0, 1.0, 1.0, 0.0], index=idx)
    proba = np.array([0.9, 0.1, 0.9, 0.9, 0.1])
    result = breakdown_by_year(y_true, proba, idx, threshold=0.5)
    assert set(result.index) == {2022, 2023}
    assert result.loc[2022, "n"] == 2
    assert result.loc[2022, "accuracy"] == pytest.approx(1.0)
    assert result.loc[2022, "majority_baseline"] == pytest.approx(0.5)
    assert result.loc[2023, "n"] == 3
    assert result.loc[2023, "accuracy"] == pytest.approx(1.0)
    assert result.loc[2023, "majority_baseline"] == pytest.approx(2 / 3)
    # Verify that different years have different majority_baseline values
    # (this catches regressions to a single global majority_baseline)
    assert result.loc[2022, "majority_baseline"] != result.loc[2023, "majority_baseline"]


def test_breakdown_by_regime_groups_correctly():
    # Test with different label balance per regime to verify per-subset majority_baseline
    # 2022 (down): y=[1.0, 0.0] -> majority_baseline = 0.5 (balanced)
    # 2021 (bull): y=[1.0, 1.0, 0.0] -> majority_baseline = 2/3 (imbalanced, DIFFERENT from down)
    idx = pd.DatetimeIndex(["2022-01-01", "2022-06-01", "2021-01-01", "2021-06-01", "2021-12-01"])
    y_true = pd.Series([1.0, 0.0, 1.0, 1.0, 0.0], index=idx)
    proba = np.array([0.9, 0.1, 0.9, 0.9, 0.1])
    result = breakdown_by_regime(y_true, proba, idx, threshold=0.5)
    assert set(result.index) == {"down", "bull"}
    assert result.loc["down", "n"] == 2
    assert result.loc["down", "accuracy"] == pytest.approx(1.0)
    assert result.loc["down", "majority_baseline"] == pytest.approx(0.5)
    assert result.loc["bull", "n"] == 3
    assert result.loc["bull", "accuracy"] == pytest.approx(1.0)
    assert result.loc["bull", "majority_baseline"] == pytest.approx(2 / 3)
    # Verify that different regimes have different majority_baseline values
    # (this catches regressions to a single global majority_baseline)
    assert result.loc["down", "majority_baseline"] != result.loc["bull", "majority_baseline"]


def test_sweep_confidence_threshold_returns_one_row_per_threshold():
    y_true = pd.Series([1.0, 0.0, 1.0, 0.0])
    proba = np.array([0.9, 0.1, 0.65, 0.35])
    result = sweep_confidence_threshold(y_true, proba, thresholds=(0.5, 0.6))
    assert list(result.index) == [0.5, 0.6]
    assert "accuracy" in result.columns
    assert "coverage" in result.columns

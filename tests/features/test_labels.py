import numpy as np
import pandas as pd
from src.features.labels import generate_labels


def test_label_1_when_up_threshold_hit_first():
    # entry=100 at index0; future = [101, 103, 104, 105, 106] (horizon=5)
    # up_target=102 hit at future index1 (103); down_target=98 never hit -> label 1
    idx = pd.date_range("2024-01-02", periods=6, freq="B")
    close = pd.Series([100, 101, 103, 104, 105, 106], index=idx, dtype=float)
    labels = generate_labels(close, horizon=5, up_threshold=0.02, down_threshold=0.02)
    assert labels.iloc[0] == 1.0


def test_label_0_when_down_threshold_hit_first():
    # entry=100; future = [99, 97, 96, 95, 94] (horizon=5)
    # down_target=98 hit at future index1 (97); up_target=102 never hit -> label 0
    idx = pd.date_range("2024-01-02", periods=6, freq="B")
    close = pd.Series([100, 99, 97, 96, 95, 94], index=idx, dtype=float)
    labels = generate_labels(close, horizon=5, up_threshold=0.02, down_threshold=0.02)
    assert labels.iloc[0] == 0.0


def test_label_nan_when_neither_threshold_hit():
    # entry=100; future stays within [98, 102] the whole horizon -> ambiguous, drop
    idx = pd.date_range("2024-01-02", periods=6, freq="B")
    close = pd.Series([100, 100.5, 101, 100.8, 101.2, 100.9], index=idx, dtype=float)
    labels = generate_labels(close, horizon=5, up_threshold=0.02, down_threshold=0.02)
    assert np.isnan(labels.iloc[0])


def test_label_nan_for_tail_rows_with_insufficient_future_data():
    # 6 rows total, horizon=5 -> only index0 has a full 5-day future window;
    # indices 1-5 must all be NaN regardless of their values.
    idx = pd.date_range("2024-01-02", periods=6, freq="B")
    close = pd.Series([100, 106, 106, 106, 106, 106], index=idx, dtype=float)
    labels = generate_labels(close, horizon=5, up_threshold=0.02, down_threshold=0.02)
    assert labels.iloc[0] == 1.0
    assert labels.iloc[1:].isna().all()


def test_labels_preserve_index():
    idx = pd.date_range("2024-01-02", periods=6, freq="B")
    close = pd.Series([100, 101, 103, 104, 105, 106], index=idx, dtype=float)
    labels = generate_labels(close, horizon=5)
    assert labels.index.equals(idx)
    assert labels.name == "label"


def test_label_resolves_correctly_when_both_thresholds_hit_within_horizon():
    # entry=100 at index0; future = [99, 97, 103, 104, 105] (horizon=5)
    # down_target=98 hit at future index1 (97); up_target=102 hit at future index2 (103)
    # down is hit FIRST (index1 < index2) -> label should be 0, even though
    # the up-threshold is also eventually crossed within the same horizon.
    idx = pd.date_range("2024-01-02", periods=6, freq="B")
    close = pd.Series([100, 99, 97, 103, 104, 105], index=idx, dtype=float)
    labels = generate_labels(close, horizon=5, up_threshold=0.02, down_threshold=0.02)
    assert labels.iloc[0] == 0.0


def test_label_resolves_correctly_when_up_hits_before_down_within_horizon():
    # entry=100 at index0; future = [103, 97, 96, 95, 94] (horizon=5)
    # up_target=102 hit at future index0 (103); down_target=98 hit at future index1 (97)
    # up is hit FIRST (index0 < index1) -> label should be 1.
    idx = pd.date_range("2024-01-02", periods=6, freq="B")
    close = pd.Series([100, 103, 97, 96, 95, 94], index=idx, dtype=float)
    labels = generate_labels(close, horizon=5, up_threshold=0.02, down_threshold=0.02)
    assert labels.iloc[0] == 1.0

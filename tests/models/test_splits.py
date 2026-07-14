import pytest
import pandas as pd
from src.models.splits import time_split


def _matrix():
    idx = pd.date_range("2019-01-01", "2025-12-31", freq="D")
    return pd.DataFrame({"x": range(len(idx))}, index=idx)


def test_time_split_partitions_chronologically():
    df = _matrix()
    train, validate, test = time_split(df, train_end="2022-12-31", validate_end="2023-12-31")
    assert train.index.max() <= pd.Timestamp("2022-12-31")
    assert validate.index.min() > pd.Timestamp("2022-12-31")
    assert validate.index.max() <= pd.Timestamp("2023-12-31")
    assert test.index.min() > pd.Timestamp("2023-12-31")


def test_time_split_covers_all_rows_exactly_once():
    df = _matrix()
    train, validate, test = time_split(df, train_end="2022-12-31", validate_end="2023-12-31")
    assert len(train) + len(validate) + len(test) == len(df)


def test_time_split_no_row_appears_in_two_splits():
    df = _matrix()
    train, validate, test = time_split(df, train_end="2022-12-31", validate_end="2023-12-31")
    train_idx = set(train.index)
    validate_idx = set(validate.index)
    test_idx = set(test.index)
    assert train_idx.isdisjoint(validate_idx)
    assert validate_idx.isdisjoint(test_idx)
    assert train_idx.isdisjoint(test_idx)


def test_time_split_preserves_row_order_no_shuffle():
    df = _matrix()
    train, validate, test = time_split(df, train_end="2022-12-31", validate_end="2023-12-31")
    assert train.index.is_monotonic_increasing
    assert validate.index.is_monotonic_increasing
    assert test.index.is_monotonic_increasing


def test_time_split_raises_when_train_end_not_before_validate_end():
    df = _matrix()
    with pytest.raises(ValueError):
        time_split(df, train_end="2023-12-31", validate_end="2022-12-31")

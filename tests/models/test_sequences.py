import numpy as np
import pandas as pd
import pytest
from src.models.sequences import build_sequences


def _matrix(ticker="AAPL", n=20, start_val=0.0):
    idx = pd.date_range("2024-01-02", periods=n, freq="B")
    return pd.DataFrame({
        "f1": start_val + np.arange(n, dtype=float),
        "f2": start_val + np.arange(n, dtype=float) * 2,
        "label": [1.0 if i % 2 == 0 else 0.0 for i in range(n)],
        "ticker": ticker,
    }, index=idx)


def test_build_sequences_shape():
    df = _matrix(n=20)
    X_seq, y, index, ticker = build_sequences(df, feature_columns=["f1", "f2"], lookback=10)
    # first 9 rows can't form a full 10-day lookback window; 20-10+1=11 sequences remain
    assert X_seq.shape == (11, 10, 2)
    assert y.shape == (11,)
    assert len(index) == 11
    assert len(ticker) == 11


def test_build_sequences_last_day_of_each_window_matches_label():
    df = _matrix(n=20)
    X_seq, y, index, ticker = build_sequences(df, feature_columns=["f1", "f2"], lookback=10)
    # the first sequence's anchor day is index 9 (0-indexed), i.e. df.index[9]
    assert index[0] == df.index[9]
    assert y[0] == df["label"].iloc[9]
    # the sequence's LAST row (X_seq[0, -1, :]) must match day 9's own features
    assert X_seq[0, -1, 0] == df["f1"].iloc[9]
    assert X_seq[0, -1, 1] == df["f2"].iloc[9]
    # the sequence's FIRST row (X_seq[0, 0, :]) must match day 0's features (9-10+1=0)
    assert X_seq[0, 0, 0] == df["f1"].iloc[0]


def test_build_sequences_never_crosses_ticker_boundary():
    df_a = _matrix(ticker="AAPL", n=15, start_val=0.0)
    df_b = _matrix(ticker="MSFT", n=15, start_val=1000.0)
    combined = pd.concat([df_a, df_b])
    X_seq, y, index, ticker = build_sequences(combined, feature_columns=["f1", "f2"], lookback=10)
    # AAPL contributes 15-10+1=6 sequences, MSFT contributes 6 sequences, total 12
    assert X_seq.shape[0] == 12
    assert set(ticker) == {"AAPL", "MSFT"}
    # no AAPL sequence should contain any MSFT feature values (>=1000) and vice versa
    aapl_mask = (ticker == "AAPL").values
    msft_mask = (ticker == "MSFT").values
    assert (X_seq[aapl_mask] < 1000).all()
    assert (X_seq[msft_mask] >= 1000).all()


def test_build_sequences_drops_tickers_with_insufficient_history():
    df = _matrix(n=5)  # fewer than lookback=10 rows
    X_seq, y, index, ticker = build_sequences(df, feature_columns=["f1", "f2"], lookback=10)
    assert X_seq.shape[0] == 0


def test_build_sequences_is_causal_no_lookahead():
    # Modifying a day AFTER a given sequence's anchor day must not change that sequence.
    df = _matrix(n=20)
    X_seq_before, _, _, _ = build_sequences(df, feature_columns=["f1", "f2"], lookback=10)
    df_modified = df.copy()
    df_modified.iloc[15:] = df_modified.iloc[15:] * 100  # perturb days 15+
    X_seq_after, _, _, _ = build_sequences(df_modified, feature_columns=["f1", "f2"], lookback=10)
    # sequences anchored at day 9 through day 14 (indices 0-5 in the output) must be unaffected
    # by the day-15+ perturbation, since their windows are entirely within [0,14].
    np.testing.assert_array_equal(X_seq_before[:6], X_seq_after[:6])

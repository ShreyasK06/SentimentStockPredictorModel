import numpy as np
import pandas as pd
import pytest
from src.features.cross_sectional import compute_cross_sectional_features


def _panel(n_days=5, tickers=("A", "B", "C", "D"), sector_map=None):
    idx = pd.date_range("2024-01-02", periods=n_days, freq="B")
    rows = []
    for day_i, day in enumerate(idx):
        for ticker_i, ticker in enumerate(tickers):
            # deterministic, ticker-and-day-dependent value so ranks are unambiguous
            value = day_i * 10 + ticker_i
            rows.append({"date": day, "ticker": ticker, "return_10d": value})
    df = pd.DataFrame(rows).set_index("date")
    return df


def test_universe_rank_is_0_to_1_and_orders_correctly():
    df = _panel()
    result = compute_cross_sectional_features(df, base_columns=["return_10d"], sector_map={"A": "S1", "B": "S1", "C": "S2", "D": "S2"})
    # within each day, ticker D always has the highest return_10d value (ticker_i=3, highest offset)
    # so its universe rank should be the maximum (1.0) every day.
    d_ranks = result[df["ticker"] == "D"]["xs_rank_return_10d"]
    assert (d_ranks == 1.0).all()
    a_ranks = result[df["ticker"] == "A"]["xs_rank_return_10d"]
    assert (a_ranks == a_ranks.min()).all()
    assert a_ranks.iloc[0] < d_ranks.iloc[0]


def test_universe_zscore_has_zero_mean_within_each_day():
    df = _panel()
    result = compute_cross_sectional_features(df, base_columns=["return_10d"], sector_map={"A": "S1", "B": "S1", "C": "S2", "D": "S2"})
    for day in df.index.unique():
        day_zscores = result.loc[result.index == day, "xs_zscore_return_10d"]
        assert day_zscores.mean() == pytest.approx(0.0, abs=1e-9)


def test_sector_rank_only_compares_within_same_sector():
    df = _panel(tickers=("A", "B", "C", "D"))
    sector_map = {"A": "S1", "B": "S1", "C": "S2", "D": "S2"}
    result = compute_cross_sectional_features(df, base_columns=["return_10d"], sector_map=sector_map)
    # within S1 (A, B), B always has a higher return_10d value than A (ticker_i 1 > 0)
    # so B's sector rank should be 1.0 (max within S1), regardless of C/D's values.
    b_sector_ranks = result[df["ticker"] == "B"]["xs_sector_rank_return_10d"]
    assert (b_sector_ranks == 1.0).all()
    a_sector_ranks = result[df["ticker"] == "A"]["xs_sector_rank_return_10d"]
    assert (a_sector_ranks == 0.0).all()


def test_returns_correct_shape_and_column_names():
    df = _panel()
    result = compute_cross_sectional_features(
        df, base_columns=["return_10d"],
        sector_map={"A": "S1", "B": "S1", "C": "S2", "D": "S2"},
    )
    expected_cols = {"xs_rank_return_10d", "xs_zscore_return_10d", "xs_sector_rank_return_10d", "xs_sector_zscore_return_10d"}
    assert set(result.columns) == expected_cols
    assert len(result) == len(df)
    assert result.index.equals(df.index)


def test_cross_sectional_features_are_causal_no_lookahead():
    # Perturbing one ticker's value on a LATER day must not change another
    # ticker's cross-sectional rank/zscore on an EARLIER day.
    df = _panel(n_days=5)
    sector_map = {"A": "S1", "B": "S1", "C": "S2", "D": "S2"}
    result_before = compute_cross_sectional_features(df, base_columns=["return_10d"], sector_map=sector_map)

    df_modified = df.copy()
    last_day = df.index.unique()[-1]
    mask = (df_modified.index == last_day) & (df_modified["ticker"] == "D")
    df_modified.loc[mask, "return_10d"] = 99999.0

    result_after = compute_cross_sectional_features(df_modified, base_columns=["return_10d"], sector_map=sector_map)

    earlier_days_mask = df.index != last_day
    pd.testing.assert_frame_equal(
        result_before.loc[earlier_days_mask],
        result_after.loc[earlier_days_mask],
    )

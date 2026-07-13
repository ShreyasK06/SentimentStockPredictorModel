import numpy as np
import pandas as pd
import pytest
from src.features.technical import compute_technical_features


def _price_df(n=80, start=100.0, step=1.0):
    idx = pd.date_range("2024-01-02", periods=n, freq="B")
    close = pd.Series(start + step * np.arange(n), index=idx)
    volume = pd.Series(1_000_000 + 1000 * np.arange(n), index=idx)
    return pd.DataFrame({"Close": close, "Volume": volume})


def test_returns_expected_columns():
    df = _price_df()
    features = compute_technical_features(df)
    assert list(features.columns) == [
        "return_1d", "return_5d", "return_10d", "return_20d",
        "rsi_14", "macd", "macd_signal", "macd_hist",
        "volatility_20d", "volume_ratio_20d", "dist_from_ma50",
    ]
    assert features.index.equals(df.index)


def test_rsi_is_100_for_strictly_increasing_series():
    df = _price_df(n=30, start=100.0, step=1.0)
    features = compute_technical_features(df)
    assert features["rsi_14"].iloc[-1] == 100.0


def test_rsi_is_0_for_strictly_decreasing_series():
    df = _price_df(n=30, start=200.0, step=-1.0)
    features = compute_technical_features(df)
    assert features["rsi_14"].iloc[-1] == 0.0


def test_returns_match_pct_change():
    df = _price_df()
    features = compute_technical_features(df)
    expected_5d = df["Close"].pct_change(5)
    pd.testing.assert_series_equal(features["return_5d"], expected_5d, check_names=False)


def test_macd_hist_equals_macd_minus_signal():
    df = _price_df()
    features = compute_technical_features(df)
    diff = features["macd"] - features["macd_signal"]
    pd.testing.assert_series_equal(features["macd_hist"], diff, check_names=False)


def test_technical_features_are_causal_no_lookahead():
    # The single most important property in this project: a feature at day t
    # must not change if future data (day t+1 onward) is altered.
    # Note: n=120, modify from row 70, compare rows [:70] to ensure dist_from_ma50
    # (window=50, first valid at row 49) has real (non-NaN) values in the slice.
    df = _price_df(n=120)
    features_before = compute_technical_features(df)
    df_modified = df.copy()
    df_modified.iloc[70:] = df_modified.iloc[70:] * 10
    features_after = compute_technical_features(df_modified)
    pd.testing.assert_frame_equal(features_before.iloc[:70], features_after.iloc[:70])


def test_volatility_matches_manual_calculation():
    df = _price_df(n=41)
    features = compute_technical_features(df)
    daily_returns = df["Close"].pct_change()
    expected = daily_returns.iloc[1:21].std()  # 20-day window of pct_change (rows 1-20)
    assert features["volatility_20d"].iloc[20] == pytest.approx(expected)


def test_volume_ratio_matches_manual_calculation():
    df = _price_df(n=40)
    features = compute_technical_features(df)
    expected = df["Volume"].iloc[19] / df["Volume"].iloc[0:20].mean()
    assert features["volume_ratio_20d"].iloc[19] == pytest.approx(expected)


def test_dist_from_ma50_matches_manual_calculation():
    df = _price_df(n=60)
    features = compute_technical_features(df)
    ma50 = df["Close"].iloc[0:50].mean()
    expected = (df["Close"].iloc[49] - ma50) / ma50
    assert features["dist_from_ma50"].iloc[49] == pytest.approx(expected)

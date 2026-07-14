import numpy as np
import pandas as pd
import pytest
from src.features.technical import compute_technical_features


def _price_df(n=80, start=100.0, step=1.0):
    idx = pd.date_range("2024-01-02", periods=n, freq="B")
    close = pd.Series(start + step * np.arange(n), index=idx)
    high = close + 1.0
    low = close - 1.0
    volume = pd.Series(1_000_000 + 1000 * np.arange(n), index=idx)
    return pd.DataFrame({"Close": close, "High": high, "Low": low, "Volume": volume})


def test_returns_expected_columns():
    df = _price_df()
    features = compute_technical_features(df)
    assert list(features.columns) == [
        "return_1d", "return_5d", "return_10d", "return_20d",
        "rsi_14", "macd", "macd_signal", "macd_hist",
        "volatility_20d", "volume_ratio_20d", "dist_from_ma50",
        "bollinger_pctb", "bollinger_bandwidth", "atr_14_norm",
        "stochastic_k", "stochastic_d", "obv_zscore",
        "dist_from_52w_high", "dist_from_52w_low",
        "day_of_week_sin", "day_of_week_cos",
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


def _ohlcv_df(n=280, start=100.0, step=1.0):
    idx = pd.date_range("2024-01-02", periods=n, freq="B")
    close = pd.Series(start + step * np.arange(n), index=idx)
    high = close + 1.0
    low = close - 1.0
    volume = pd.Series(1_000_000 + 1000 * np.arange(n), index=idx)
    return pd.DataFrame({"Close": close, "High": high, "Low": low, "Volume": volume})


def test_compute_technical_features_returns_20_columns():
    df = _ohlcv_df()
    features = compute_technical_features(df)
    assert len(features.columns) == 21
    expected_new = [
        "bollinger_pctb", "bollinger_bandwidth", "atr_14_norm",
        "stochastic_k", "stochastic_d", "obv_zscore",
        "dist_from_52w_high", "dist_from_52w_low",
        "day_of_week_sin", "day_of_week_cos",
    ]
    for col in expected_new:
        assert col in features.columns


def test_bollinger_pctb_matches_manual_calculation():
    df = _ohlcv_df(n=40)
    features = compute_technical_features(df)
    close = df["Close"]
    ma20 = close.iloc[0:20].mean()
    std20 = close.iloc[0:20].std()
    upper = ma20 + 2 * std20
    lower = ma20 - 2 * std20
    expected_pctb = (close.iloc[19] - lower) / (upper - lower)
    assert features["bollinger_pctb"].iloc[19] == pytest.approx(expected_pctb)


def test_atr_is_normalized_by_close_and_nonnegative():
    df = _ohlcv_df(n=30)
    features = compute_technical_features(df)
    valid_atr = features["atr_14_norm"].dropna()
    assert len(valid_atr) > 0
    assert (valid_atr >= 0).all()


def test_stochastic_k_matches_manual_calculation():
    df = _ohlcv_df(n=30)
    features = compute_technical_features(df)
    high14 = df["High"].iloc[0:14].max()
    low14 = df["Low"].iloc[0:14].min()
    close14 = df["Close"].iloc[13]
    expected_k = (close14 - low14) / (high14 - low14) * 100
    assert features["stochastic_k"].iloc[13] == pytest.approx(expected_k)


def test_stochastic_k_bounded_0_to_100():
    df = _ohlcv_df(n=40)
    features = compute_technical_features(df)
    valid_k = features["stochastic_k"].dropna()
    assert (valid_k >= 0).all() and (valid_k <= 100).all()


def test_dist_from_52w_high_is_never_positive():
    # close can never exceed its own trailing rolling max, so this is always <= 0
    df = _ohlcv_df(n=260)
    features = compute_technical_features(df)
    valid = features["dist_from_52w_high"].dropna()
    assert len(valid) > 0
    assert (valid <= 1e-9).all()


def test_dist_from_52w_low_is_never_negative():
    df = _ohlcv_df(n=260)
    features = compute_technical_features(df)
    valid = features["dist_from_52w_low"].dropna()
    assert len(valid) > 0
    assert (valid >= -1e-9).all()


def test_day_of_week_cyclical_values_bounded():
    df = _ohlcv_df(n=30)
    features = compute_technical_features(df)
    assert (features["day_of_week_sin"].abs() <= 1.0 + 1e-9).all()
    assert (features["day_of_week_cos"].abs() <= 1.0 + 1e-9).all()


def test_day_of_week_cyclical_distinguishes_monday_and_friday():
    df = _ohlcv_df(n=10)
    features = compute_technical_features(df)
    weekdays = df.index.dayofweek
    monday_rows = features.index[weekdays == 0]
    friday_rows = features.index[weekdays == 4]
    assert len(monday_rows) > 0 and len(friday_rows) > 0
    mon_val = (features.loc[monday_rows[0], "day_of_week_sin"], features.loc[monday_rows[0], "day_of_week_cos"])
    fri_val = (features.loc[friday_rows[0], "day_of_week_sin"], features.loc[friday_rows[0], "day_of_week_cos"])
    assert mon_val != fri_val


def test_new_indicators_are_causal_no_lookahead():
    df = _ohlcv_df(n=280)
    features_before = compute_technical_features(df)
    df_modified = df.copy()
    df_modified.iloc[260:] = df_modified.iloc[260:] * 10
    features_after = compute_technical_features(df_modified)
    new_cols = [
        "bollinger_pctb", "bollinger_bandwidth", "atr_14_norm",
        "stochastic_k", "stochastic_d", "obv_zscore",
        "dist_from_52w_high", "dist_from_52w_low",
    ]
    pd.testing.assert_frame_equal(
        features_before.loc[features_before.index < df.index[260], new_cols],
        features_after.loc[features_after.index < df.index[260], new_cols],
    )

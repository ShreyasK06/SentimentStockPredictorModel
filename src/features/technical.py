import pandas as pd
import math


def compute_returns(close: pd.Series, windows=(1, 5, 10, 20)) -> pd.DataFrame:
    return pd.DataFrame({f"return_{w}d": close.pct_change(w) for w in windows})


def compute_rsi(close: pd.Series, window: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=window, min_periods=window).mean()
    avg_loss = loss.rolling(window=window, min_periods=window).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi.rename("rsi_14")


def compute_macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd_line = (ema_fast - ema_slow) / close
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    hist = macd_line - signal_line
    return pd.DataFrame({"macd": macd_line, "macd_signal": signal_line, "macd_hist": hist})


def compute_volatility(close: pd.Series, window: int = 20) -> pd.Series:
    daily_returns = close.pct_change()
    return daily_returns.rolling(window=window, min_periods=window).std().rename("volatility_20d")


def compute_volume_ratio(volume: pd.Series, window: int = 20) -> pd.Series:
    avg_volume = volume.rolling(window=window, min_periods=window).mean()
    return (volume / avg_volume).rename("volume_ratio_20d")


def compute_distance_from_ma(close: pd.Series, window: int = 50) -> pd.Series:
    ma = close.rolling(window=window, min_periods=window).mean()
    return ((close - ma) / ma).rename("dist_from_ma50")


def compute_bollinger_bands(close: pd.Series, window: int = 20, num_std: int = 2) -> pd.DataFrame:
    ma = close.rolling(window=window, min_periods=window).mean()
    std = close.rolling(window=window, min_periods=window).std()
    upper = ma + num_std * std
    lower = ma - num_std * std
    pctb = (close - lower) / (upper - lower)
    bandwidth = (upper - lower) / ma
    return pd.DataFrame({"bollinger_pctb": pctb, "bollinger_bandwidth": bandwidth})


def compute_atr(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
    prev_close = close.shift(1)
    true_range = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    atr = true_range.rolling(window=window, min_periods=window).mean()
    return (atr / close).rename("atr_14_norm")


def compute_stochastic(high: pd.Series, low: pd.Series, close: pd.Series, k_window: int = 14, d_window: int = 3) -> pd.DataFrame:
    low_k = low.rolling(window=k_window, min_periods=k_window).min()
    high_k = high.rolling(window=k_window, min_periods=k_window).max()
    k = (close - low_k) / (high_k - low_k) * 100
    d = k.rolling(window=d_window, min_periods=d_window).mean()
    return pd.DataFrame({"stochastic_k": k, "stochastic_d": d})


def compute_obv_zscore(close: pd.Series, volume: pd.Series, window: int = 20) -> pd.Series:
    direction = close.diff().apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))
    obv = (direction * volume).cumsum()
    obv_mean = obv.rolling(window=window, min_periods=window).mean()
    obv_std = obv.rolling(window=window, min_periods=window).std()
    return ((obv - obv_mean) / obv_std).rename("obv_zscore")


def compute_distance_from_52w_extremes(close: pd.Series, window: int = 252) -> pd.DataFrame:
    rolling_max = close.rolling(window=window, min_periods=window).max()
    rolling_min = close.rolling(window=window, min_periods=window).min()
    dist_high = (close - rolling_max) / rolling_max
    dist_low = (close - rolling_min) / rolling_min
    return pd.DataFrame({"dist_from_52w_high": dist_high, "dist_from_52w_low": dist_low})


def compute_day_of_week_cyclical(index: pd.DatetimeIndex) -> pd.DataFrame:
    weekday = pd.Series(index.dayofweek, index=index)
    angle = 2 * math.pi * weekday / 5
    return pd.DataFrame({
        "day_of_week_sin": angle.apply(lambda a: math.sin(a)),
        "day_of_week_cos": angle.apply(lambda a: math.cos(a)),
    }, index=index)


def compute_technical_features(df: pd.DataFrame) -> pd.DataFrame:
    close = df["Close"]
    volume = df["Volume"]
    high = df["High"]
    low = df["Low"]
    return pd.concat([
        compute_returns(close),
        compute_rsi(close),
        compute_macd(close),
        compute_volatility(close),
        compute_volume_ratio(volume),
        compute_distance_from_ma(close),
        compute_bollinger_bands(close),
        compute_atr(high, low, close),
        compute_stochastic(high, low, close),
        compute_obv_zscore(close, volume),
        compute_distance_from_52w_extremes(close),
        compute_day_of_week_cyclical(df.index),
    ], axis=1)

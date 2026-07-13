import pandas as pd


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
    macd_line = ema_fast - ema_slow
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


def compute_technical_features(df: pd.DataFrame) -> pd.DataFrame:
    close = df["Close"]
    volume = df["Volume"]
    return pd.concat([
        compute_returns(close),
        compute_rsi(close),
        compute_macd(close),
        compute_volatility(close),
        compute_volume_ratio(volume),
        compute_distance_from_ma(close),
    ], axis=1)

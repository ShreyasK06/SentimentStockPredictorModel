import pandas as pd
import pytest
from unittest.mock import patch
from src.data.prices import fetch_ohlcv


def _fake_yf_frame():
    idx = pd.date_range("2024-01-02", periods=3, freq="B")
    return pd.DataFrame({
        "Open": [100.0, 101.0, 102.0],
        "High": [101.0, 102.0, 103.0],
        "Low": [99.0, 100.0, 101.0],
        "Close": [100.5, 101.5, 102.5],
        "Adj Close": [100.5, 101.5, 102.5],
        "Volume": [1000, 1100, 1200],
    }, index=idx)


@patch("src.data.prices.yf.download")
def test_fetch_ohlcv_returns_expected_columns(mock_download):
    mock_download.return_value = _fake_yf_frame()
    df = fetch_ohlcv("AAPL", "2024-01-01", "2024-01-05")
    assert list(df.columns) == ["Open", "High", "Low", "Close", "Volume", "ticker"]
    assert (df["ticker"] == "AAPL").all()
    assert len(df) == 3
    assert df.index.name == "date"


@patch("src.data.prices.yf.download")
def test_fetch_ohlcv_requests_yfinance_end_exclusive_of_requested_end(mock_download):
    # yfinance's `end` is exclusive; fetch_ohlcv must request one day past the
    # caller's `end` so the caller's own end date is actually included.
    mock_download.return_value = _fake_yf_frame()
    fetch_ohlcv("AAPL", "2024-01-01", "2024-01-05")
    call_kwargs = mock_download.call_args.kwargs
    assert call_kwargs["end"] == "2024-01-06"


@patch("src.data.prices.yf.download")
def test_fetch_ohlcv_raises_on_empty(mock_download):
    mock_download.return_value = pd.DataFrame()
    with pytest.raises(ValueError):
        fetch_ohlcv("BADTICKER", "2024-01-01", "2024-01-05")

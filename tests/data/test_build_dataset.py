import pandas as pd
from unittest.mock import patch
from src.data.build_dataset import build_price_panel, validate_price_panel


def _fake_prices(ticker, start, end):
    idx = pd.date_range("2024-01-02", periods=3, freq="B")
    df = pd.DataFrame({
        "Open": [1.0] * 3, "High": [1.0] * 3, "Low": [1.0] * 3, "Close": [1.0] * 3, "Volume": [100] * 3,
    }, index=idx)
    df.index.name = "date"
    df["ticker"] = ticker
    return df


@patch("src.data.build_dataset.save_parquet")
@patch("src.data.build_dataset.fetch_ohlcv")
def test_build_price_panel_concatenates_all_tickers(mock_fetch, mock_save, tmp_path):
    mock_fetch.side_effect = _fake_prices
    panel = build_price_panel(["AAPL", "MSFT"], "2024-01-01", "2024-01-05", tmp_path)
    assert set(panel["ticker"]) == {"AAPL", "MSFT"}
    assert len(panel) == 6
    assert "date" in panel.columns
    assert mock_save.call_count == 2


def test_validate_price_panel_flags_gap():
    panel = pd.DataFrame({
        "date": pd.DatetimeIndex(["2024-01-02", "2024-01-04"]),
        "ticker": ["AAPL", "AAPL"],
        "Close": [1, 2],
    })
    trading_days = pd.DatetimeIndex(["2024-01-02", "2024-01-03", "2024-01-04"])
    report = validate_price_panel(panel, ["AAPL"], trading_days)
    assert report["AAPL"] == ["2024-01-03"]


def test_validate_price_panel_empty_when_complete():
    panel = pd.DataFrame({
        "date": pd.DatetimeIndex(["2024-01-02", "2024-01-03"]),
        "ticker": ["AAPL", "AAPL"],
        "Close": [1, 2],
    })
    trading_days = pd.DatetimeIndex(["2024-01-02", "2024-01-03"])
    report = validate_price_panel(panel, ["AAPL"], trading_days)
    assert report["AAPL"] == []

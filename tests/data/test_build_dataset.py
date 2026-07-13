import pandas as pd
from unittest.mock import patch
from src.data.build_dataset import build_price_panel, validate_price_panel, build_news_archive


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


def _fake_articles(ticker, start, end, api_key):
    """Generate fake articles for testing."""
    if ticker == "AAPL":
        return [
            {"headline": "Apple News", "datetime": 1704153600, "source": "Reuters", "url": "http://example.com/1"}
        ]
    elif ticker == "MSFT":
        return [
            {"headline": "Microsoft News 1", "datetime": 1704153600, "source": "Reuters", "url": "http://example.com/2"},
            {"headline": "Microsoft News 2", "datetime": 1704240000, "source": "AP", "url": "http://example.com/3"}
        ]
    return []


def _fake_articles_to_dataframe(articles, ticker):
    """Convert articles to a simple DataFrame with ticker column."""
    df = pd.DataFrame(articles)
    df["ticker"] = ticker
    return df


@patch("src.data.build_dataset.save_parquet")
@patch("src.data.build_dataset.articles_to_dataframe")
@patch("src.data.build_dataset.fetch_company_news")
def test_build_news_archive_counts_and_saves_per_ticker(mock_fetch, mock_articles_df, mock_save, tmp_path):
    mock_fetch.side_effect = _fake_articles
    mock_articles_df.side_effect = _fake_articles_to_dataframe

    result = build_news_archive(["AAPL", "MSFT"], "2024-01-01", "2024-01-05", tmp_path, api_key="fake")

    # Verify the returned counts match the articles per ticker
    assert result == {"AAPL": 1, "MSFT": 2}

    # Verify save_parquet was called once per ticker
    assert mock_save.call_count == 2

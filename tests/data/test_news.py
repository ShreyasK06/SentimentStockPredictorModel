import pandas as pd
from unittest.mock import patch, MagicMock
from src.data.news import fetch_company_news, articles_to_dataframe, _year_chunks


def test_year_chunks_splits_by_calendar_year():
    chunks = list(_year_chunks("2019-06-01", "2021-02-15"))
    assert chunks == [
        ("2019-06-01", "2019-12-31"),
        ("2020-01-01", "2020-12-31"),
        ("2021-01-01", "2021-02-15"),
    ]


@patch("src.data.news.time.sleep", return_value=None)
@patch("src.data.news.requests.get")
def test_fetch_company_news_aggregates_chunks(mock_get, mock_sleep):
    mock_resp = MagicMock()
    mock_resp.json.return_value = [{"headline": "x", "datetime": 1704153600, "source": "Reuters", "url": "http://x"}]
    mock_resp.raise_for_status.return_value = None
    mock_get.return_value = mock_resp

    articles = fetch_company_news("AAPL", "2024-01-01", "2024-01-05", api_key="fake")
    assert len(articles) == 1
    assert mock_get.call_count == 1
    call_kwargs = mock_get.call_args.kwargs
    assert call_kwargs["params"]["symbol"] == "AAPL"
    assert call_kwargs["params"]["token"] == "fake"


@patch("src.data.news.time.sleep", return_value=None)
@patch("src.data.news.requests.get")
def test_fetch_company_news_multi_year_aggregation(mock_get, mock_sleep):
    """Test that fetch_company_news correctly aggregates articles across multiple calendar years."""
    # Create distinct response objects for each year chunk
    mock_resp_2019 = MagicMock()
    mock_resp_2019.json.return_value = [
        {"headline": "2019-article-1", "datetime": 1560000000, "source": "Reuters", "url": "http://x1"},
        {"headline": "2019-article-2", "datetime": 1560000001, "source": "Reuters", "url": "http://x2"},
    ]
    mock_resp_2019.raise_for_status.return_value = None

    mock_resp_2020 = MagicMock()
    mock_resp_2020.json.return_value = [
        {"headline": "2020-article-1", "datetime": 1577000000, "source": "Reuters", "url": "http://y1"},
    ]
    mock_resp_2020.raise_for_status.return_value = None

    mock_resp_2021 = MagicMock()
    mock_resp_2021.json.return_value = [
        {"headline": "2021-article-1", "datetime": 1609000000, "source": "Reuters", "url": "http://z1"},
        {"headline": "2021-article-2", "datetime": 1609000001, "source": "Reuters", "url": "http://z2"},
        {"headline": "2021-article-3", "datetime": 1609000002, "source": "Reuters", "url": "http://z3"},
    ]
    mock_resp_2021.raise_for_status.return_value = None

    # Use side_effect to return different responses for each call
    mock_get.side_effect = [mock_resp_2019, mock_resp_2020, mock_resp_2021]

    # Call with 3-year range that spans exactly 3 calendar years
    articles = fetch_company_news("AAPL", "2019-06-01", "2021-02-15", api_key="fake")

    # Verify we made 3 API calls (one per year chunk)
    assert mock_get.call_count == 3

    # Verify we slept 3 times (once per chunk)
    assert mock_sleep.call_count == 3

    # Verify all articles were aggregated correctly
    assert len(articles) == 6  # 2 + 1 + 3

    # Verify the order and content of aggregated articles
    headlines = [a["headline"] for a in articles]
    assert headlines == [
        "2019-article-1", "2019-article-2",
        "2020-article-1",
        "2021-article-1", "2021-article-2", "2021-article-3",
    ]

    # Verify each API call had correct parameters for its year chunk
    calls = mock_get.call_args_list
    assert calls[0].kwargs["params"]["symbol"] == "AAPL"
    assert calls[0].kwargs["params"]["token"] == "fake"
    assert calls[0].kwargs["params"]["from"] == "2019-06-01"
    assert calls[0].kwargs["params"]["to"] == "2019-12-31"

    assert calls[1].kwargs["params"]["from"] == "2020-01-01"
    assert calls[1].kwargs["params"]["to"] == "2020-12-31"

    assert calls[2].kwargs["params"]["from"] == "2021-01-01"
    assert calls[2].kwargs["params"]["to"] == "2021-02-15"


def test_articles_to_dataframe_normalizes_schema():
    articles = [{"headline": "AAPL rises", "datetime": 1704153600, "source": "Reuters", "url": "http://x"}]
    df = articles_to_dataframe(articles, "AAPL")
    assert list(df.columns) == ["date", "ticker", "headline", "source", "url", "published_at"]
    assert df.iloc[0]["ticker"] == "AAPL"
    # published_at must hold the exact UTC timestamp, not normalized to midnight
    expected_published_at = pd.Timestamp(1704153600, unit="s", tz="UTC")
    assert df.iloc[0]["published_at"] == expected_published_at


def test_articles_to_dataframe_empty_list():
    df = articles_to_dataframe([], "AAPL")
    assert len(df) == 0
    assert list(df.columns) == ["date", "ticker", "headline", "source", "url", "published_at"]


def test_articles_to_dataframe_buckets_by_us_eastern_calendar_day():
    # Epoch arithmetic (verified via pandas):
    #   A = 2024-01-15 23:30:00 UTC -> epoch 1705361400 -> ET 2024-01-15 18:30:00-05:00 (Jan 15 ET)
    #   B = 2024-01-16 03:30:00 UTC -> epoch 1705375800 -> ET 2024-01-15 22:30:00-05:00 (Jan 15 ET)
    #   C = 2024-01-16 09:00:00 UTC -> epoch 1705395600 -> ET 2024-01-16 04:00:00-05:00 (Jan 16 ET)
    # A and B are on different UTC calendar days (Jan 15 vs Jan 16) but the SAME US-Eastern
    # calendar day (Jan 15). C is a different US-Eastern calendar day (Jan 16).
    articles = [
        {"headline": "A", "datetime": 1705361400, "source": "Reuters", "url": "http://a"},
        {"headline": "B", "datetime": 1705375800, "source": "Reuters", "url": "http://b"},
        {"headline": "C", "datetime": 1705395600, "source": "Reuters", "url": "http://c"},
    ]
    df = articles_to_dataframe(articles, "AAPL")

    expected_jan15_et = pd.Timestamp("2024-01-15", tz="America/New_York")
    expected_jan16_et = pd.Timestamp("2024-01-16", tz="America/New_York")

    assert df.iloc[0]["date"] == expected_jan15_et
    assert df.iloc[1]["date"] == expected_jan15_et
    assert df.iloc[2]["date"] == expected_jan16_et

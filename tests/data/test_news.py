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


def test_articles_to_dataframe_normalizes_schema():
    articles = [{"headline": "AAPL rises", "datetime": 1704153600, "source": "Reuters", "url": "http://x"}]
    df = articles_to_dataframe(articles, "AAPL")
    assert list(df.columns) == ["date", "ticker", "headline", "source", "url"]
    assert df.iloc[0]["ticker"] == "AAPL"


def test_articles_to_dataframe_empty_list():
    df = articles_to_dataframe([], "AAPL")
    assert len(df) == 0
    assert list(df.columns) == ["date", "ticker", "headline", "source", "url"]

import pytest
from unittest.mock import patch, MagicMock
import pandas as pd
from src.features.sentiment import score_headlines, aggregate_daily_sentiment, add_rolling_sentiment


@patch("src.features.sentiment._get_classifier")
def test_score_headlines_computes_sentiment_score(mock_get_classifier):
    mock_classifier = MagicMock()
    mock_classifier.return_value = [
        [{"label": "positive", "score": 0.7}, {"label": "negative", "score": 0.1}, {"label": "neutral", "score": 0.2}],
        [{"label": "positive", "score": 0.05}, {"label": "negative", "score": 0.85}, {"label": "neutral", "score": 0.1}],
    ]
    mock_get_classifier.return_value = mock_classifier

    df = score_headlines(["Great earnings beat", "Massive layoffs announced"])

    assert list(df.columns) == ["headline", "positive", "negative", "neutral", "sentiment_score"]
    assert df.iloc[0]["sentiment_score"] == pytest.approx(0.6)
    assert df.iloc[1]["sentiment_score"] == pytest.approx(-0.8)
    mock_classifier.assert_called_once_with(
        ["Great earnings beat", "Massive layoffs announced"], batch_size=16, truncation=True
    )


def test_score_headlines_empty_list():
    df = score_headlines([])
    assert len(df) == 0
    assert list(df.columns) == ["headline", "positive", "negative", "neutral", "sentiment_score"]


def test_aggregate_daily_sentiment_computes_mean_std_count():
    scored_news = pd.DataFrame({
        "headline": ["a", "b", "c"],
        "sentiment_score": [0.5, -0.5, 0.2],
    })
    effective_day = pd.Series([
        pd.Timestamp("2024-01-16"), pd.Timestamp("2024-01-16"), pd.Timestamp("2024-01-17"),
    ])
    daily = aggregate_daily_sentiment(scored_news, effective_day)
    assert daily.loc[pd.Timestamp("2024-01-16"), "sentiment_mean"] == pytest.approx(0.0)
    assert daily.loc[pd.Timestamp("2024-01-16"), "headline_count"] == 2
    assert daily.loc[pd.Timestamp("2024-01-17"), "sentiment_mean"] == pytest.approx(0.2)
    assert daily.loc[pd.Timestamp("2024-01-17"), "headline_count"] == 1


def test_add_rolling_sentiment_fills_no_news_days_with_zero_count():
    trading_days = pd.DatetimeIndex(["2024-01-16", "2024-01-17", "2024-01-18"])
    daily = pd.DataFrame(
        {"sentiment_mean": [0.5], "sentiment_std": [float("nan")], "headline_count": [1]},
        index=pd.DatetimeIndex(["2024-01-16"]),
    )
    result = add_rolling_sentiment(daily, trading_days, windows=(3, 7))
    assert result.loc[pd.Timestamp("2024-01-17"), "headline_count"] == 0
    assert list(result.index) == list(pd.DatetimeIndex(trading_days))


def test_add_rolling_sentiment_rolling_average_skips_no_news_days():
    trading_days = pd.DatetimeIndex(["2024-01-16", "2024-01-17", "2024-01-18"])
    daily = pd.DataFrame(
        {"sentiment_mean": [0.6, 0.0], "sentiment_std": [0.0, 0.0], "headline_count": [1, 1]},
        index=pd.DatetimeIndex(["2024-01-16", "2024-01-18"]),
    )
    result = add_rolling_sentiment(daily, trading_days, windows=(3,))
    # 2024-01-17 has no news (NaN sentiment_mean); the 3-day rolling average on
    # 2024-01-18 should average only the two real observations (0.6, 0.0) -> 0.3,
    # not treat the missing day as 0.
    assert result.loc[pd.Timestamp("2024-01-18"), "sentiment_roll3"] == pytest.approx(0.3)


def test_aggregate_daily_sentiment_excludes_nat_effective_day():
    """Test that NaT effective_day values are excluded from aggregation."""
    scored_news = pd.DataFrame({
        "headline": ["a", "b", "c"],
        "sentiment_score": [0.5, -0.5, 0.2],
    })
    effective_day = pd.Series([
        pd.Timestamp("2024-01-16"),
        pd.NaT,
        pd.Timestamp("2024-01-17"),
    ])
    daily = aggregate_daily_sentiment(scored_news, effective_day)
    # Should not raise
    # NaT should not appear as an index value
    assert pd.NaT not in daily.index
    # Only 2 dates should be present
    assert len(daily) == 2
    # 2024-01-16 should have only 1 headline (the NaT one is excluded)
    assert daily.loc[pd.Timestamp("2024-01-16"), "headline_count"] == 1
    assert daily.loc[pd.Timestamp("2024-01-16"), "sentiment_mean"] == pytest.approx(0.5)
    # 2024-01-17 should be unchanged
    assert daily.loc[pd.Timestamp("2024-01-17"), "headline_count"] == 1
    assert daily.loc[pd.Timestamp("2024-01-17"), "sentiment_mean"] == pytest.approx(0.2)

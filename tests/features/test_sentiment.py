import pytest
from unittest.mock import patch, MagicMock
from src.features.sentiment import score_headlines


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

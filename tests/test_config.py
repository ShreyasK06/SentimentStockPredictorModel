from datetime import date
from src import config


def test_directories_are_paths():
    assert config.PRICES_DIR.name == "prices"
    assert config.NEWS_DIR.name == "news"
    assert config.DATA_PROCESSED_DIR.name == "processed"


def test_date_range_covers_at_least_five_years():
    start = date.fromisoformat(config.START_DATE)
    end = date.fromisoformat(config.END_DATE)
    assert (end - start).days >= 5 * 365


def test_date_range_includes_2022_bear_market():
    start = date.fromisoformat(config.START_DATE)
    end = date.fromisoformat(config.END_DATE)
    assert start <= date(2022, 1, 1)
    assert end >= date(2022, 12, 31)

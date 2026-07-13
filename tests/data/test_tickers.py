from src.data.tickers import UNIVERSE, SECTOR_MAP


def test_universe_size_in_range():
    assert 20 <= len(UNIVERSE) <= 30


def test_universe_has_no_duplicates():
    assert len(UNIVERSE) == len(set(UNIVERSE))


def test_every_ticker_has_a_sector():
    for ticker in UNIVERSE:
        assert ticker in SECTOR_MAP


def test_at_least_four_sectors_represented():
    sectors = {SECTOR_MAP[t] for t in UNIVERSE}
    assert len(sectors) >= 4

import pandas as pd
from src.models.regimes import label_regime, REGIME_BY_YEAR


def test_2022_is_labeled_down():
    idx = pd.DatetimeIndex(["2022-03-15", "2022-11-01"])
    result = label_regime(idx)
    assert (result == "down").all()


def test_2021_is_labeled_bull():
    idx = pd.DatetimeIndex(["2021-01-15", "2021-11-01"])
    result = label_regime(idx)
    assert (result == "bull").all()


def test_every_year_in_dataset_range_has_a_regime_assigned():
    for year in range(2019, 2026):
        assert year in REGIME_BY_YEAR


def test_label_regime_preserves_index():
    idx = pd.DatetimeIndex(["2022-03-15", "2023-06-01"])
    result = label_regime(idx)
    assert list(result.index) == list(idx)
    assert result.name == "regime"

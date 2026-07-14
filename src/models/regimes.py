import pandas as pd

REGIME_BY_YEAR = {
    2019: "bull",
    2020: "bull",
    2021: "bull",
    2022: "down",
    2023: "bull",
    2024: "bull",
    2025: "flat",
}


def label_regime(index: pd.DatetimeIndex) -> pd.Series:
    years = pd.DatetimeIndex(index).year
    regimes = [REGIME_BY_YEAR.get(y, "flat") for y in years]
    return pd.Series(regimes, index=index, name="regime")

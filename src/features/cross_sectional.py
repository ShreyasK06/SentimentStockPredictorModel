import pandas as pd


def compute_cross_sectional_features(df: pd.DataFrame, base_columns: list, sector_map: dict) -> pd.DataFrame:
    working = df.copy()
    working["_sector"] = working["ticker"].map(sector_map)

    result = pd.DataFrame(index=df.index)

    for col in base_columns:
        # Universe ranking: normalize to [0, 1] using (rank - 1) / (n - 1)
        universe_grouped = working.groupby(working.index)[col]
        rank = universe_grouped.rank()
        n = universe_grouped.transform("count")
        result[f"xs_rank_{col}"] = (rank - 1) / (n - 1)

        # Universe z-score
        universe_mean = universe_grouped.transform("mean")
        universe_std = universe_grouped.transform("std")
        result[f"xs_zscore_{col}"] = (working[col] - universe_mean) / universe_std

        # Sector ranking: normalize to [0, 1] using (rank - 1) / (n - 1)
        sector_grouped = working.groupby([working.index, "_sector"])[col]
        rank_sector = sector_grouped.rank()
        n_sector = sector_grouped.transform("count")
        result[f"xs_sector_rank_{col}"] = (rank_sector - 1) / (n_sector - 1)

        # Sector z-score
        sector_mean = sector_grouped.transform("mean")
        sector_std = sector_grouped.transform("std")
        result[f"xs_sector_zscore_{col}"] = (working[col] - sector_mean) / sector_std

    return result

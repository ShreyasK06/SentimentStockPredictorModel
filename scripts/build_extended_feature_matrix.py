import pandas as pd

from src import config
from src.data.storage import load_parquet, save_parquet
from src.data.tickers import SECTOR_MAP
from src.features.cross_sectional import compute_cross_sectional_features
from src.features.labels import generate_labels

RETURN_COLUMNS = ["return_10d", "return_20d"]


def main():
    matrix = load_parquet("feature_matrix", config.DATA_PROCESSED_DIR)
    panel = load_parquet("combined_price_panel", config.DATA_PROCESSED_DIR)

    xs_features = compute_cross_sectional_features(matrix, base_columns=RETURN_COLUMNS, sector_map=SECTOR_MAP)
    # matrix's DatetimeIndex has duplicate values (one per ticker per date), so an
    # index-based join() performs a many-to-many relational join and blows up the
    # row count. compute_cross_sectional_features preserves matrix's exact row
    # order, so assign columns positionally instead.
    extended = matrix.copy()
    for col in xs_features.columns:
        extended[col] = xs_features[col].to_numpy()

    alt_labels = []
    for ticker, group in panel.groupby("ticker"):
        group = group.sort_index()
        labels = generate_labels(group["Close"], horizon=20, up_threshold=0.04, down_threshold=0.04)
        df = labels.rename("label_20d_4pct").to_frame()
        df["ticker"] = ticker
        alt_labels.append(df)
    alt_labels = pd.concat(alt_labels)

    extended_reset = extended.reset_index()
    alt_labels_reset = alt_labels.reset_index()
    merged = extended_reset.merge(
        alt_labels_reset, on=[extended_reset.columns[0], "ticker"], how="left"
    )
    merged = merged.set_index(extended_reset.columns[0])
    merged.index.name = matrix.index.name

    print(f"Extended matrix shape: {merged.shape} (original: {matrix.shape})")
    print(f"Rows with non-null label_20d_4pct: {merged['label_20d_4pct'].notna().sum()} / {len(merged)}")
    xs_cols = [c for c in merged.columns if c.startswith("xs_")]
    print(f"Cross-sectional columns: {xs_cols}")
    print(f"NaN count in cross-sectional columns: {merged[xs_cols].isna().sum().sum()}")

    save_parquet(merged, "feature_matrix_extended", config.DATA_PROCESSED_DIR)
    print(f"Saved to {config.DATA_PROCESSED_DIR / 'feature_matrix_extended.parquet'}")


if __name__ == "__main__":
    main()

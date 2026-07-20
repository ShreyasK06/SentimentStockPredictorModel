"""
Adds a `label_7d` column to the already-frozen `feature_matrix_extended.parquet`
(built by scripts/build_extended_feature_matrix.py). This is an ADDITIVE merge
only - every existing column (including `label`, `label_20d_4pct`, and the
cross-sectional xs_* columns) must be preserved byte-for-byte, and no row may
be added or dropped. Mirrors build_extended_feature_matrix.py's own
per-ticker label-adding pattern (see its `alt_labels` block) so the merge
mechanics stay consistent across both label columns.
"""
import pandas as pd

from src import config
from src.data.storage import load_parquet, save_parquet
from src.features.labels import generate_labels

# 7-day horizon, 1.5% up/down thresholds: a defensible linear scale-down from
# the global 10-day/2% config (2% * 7/10 = 1.4%, rounded up slightly to a
# clean 1.5% - not implying more calibration precision than actually exists).
HORIZON = 7
UP_THRESHOLD = 0.015
DOWN_THRESHOLD = 0.015


def main():
    matrix = load_parquet("feature_matrix_extended", config.DATA_PROCESSED_DIR)
    panel = load_parquet("combined_price_panel", config.DATA_PROCESSED_DIR)

    before_shape = matrix.shape
    before_columns = list(matrix.columns)
    # Snapshot of a few existing columns to spot-check byte-identity after the merge.
    before_label = matrix["label"].copy()
    before_label_20d = matrix["label_20d_4pct"].copy()

    new_labels = []
    for ticker, group in panel.groupby("ticker"):
        group = group.sort_index()
        labels = generate_labels(group["Close"], horizon=HORIZON, up_threshold=UP_THRESHOLD, down_threshold=DOWN_THRESHOLD)
        df = labels.rename("label_7d").to_frame()
        df["ticker"] = ticker
        new_labels.append(df)
    new_labels = pd.concat(new_labels)

    matrix_reset = matrix.reset_index()
    new_labels_reset = new_labels.reset_index()
    merged = matrix_reset.merge(
        new_labels_reset, on=[matrix_reset.columns[0], "ticker"], how="left"
    )
    merged = merged.set_index(matrix_reset.columns[0])
    merged.index.name = matrix.index.name

    # --- verify additive-only merge before saving ---
    assert merged.shape[0] == before_shape[0], f"row count changed: {before_shape[0]} -> {merged.shape[0]}"
    assert set(before_columns) <= set(merged.columns), "an existing column was dropped"
    assert list(merged.columns) == before_columns + ["label_7d"], (
        f"unexpected column set/order: {list(merged.columns)}"
    )
    assert merged["label"].equals(before_label), "existing `label` column changed values"
    assert merged["label_20d_4pct"].equals(before_label_20d), "existing `label_20d_4pct` column changed values"

    print(f"Extended matrix shape: {merged.shape} (before: {before_shape})")
    print(f"Rows with non-null label_7d: {merged['label_7d'].notna().sum()} / {len(merged)}")
    print(f"Rows with non-null label (10d, unchanged): {merged['label'].notna().sum()} / {len(merged)}")
    print(f"Rows with non-null label_20d_4pct (unchanged): {merged['label_20d_4pct'].notna().sum()} / {len(merged)}")

    save_parquet(merged, "feature_matrix_extended", config.DATA_PROCESSED_DIR)
    print(f"Saved to {config.DATA_PROCESSED_DIR / 'feature_matrix_extended.parquet'}")


if __name__ == "__main__":
    main()

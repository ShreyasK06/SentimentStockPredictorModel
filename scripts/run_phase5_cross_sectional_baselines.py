from src import config
from src.data.storage import load_parquet
from src.models.splits import time_split
from src.models.train import fit_lightgbm, predict_proba
from src.models.evaluate import majority_baseline, accuracy_at_threshold
from src.models.ablation import TECHNICAL_COLUMNS, SENTIMENT_COLUMNS

BASE_FEATURE_COLUMNS = TECHNICAL_COLUMNS + SENTIMENT_COLUMNS
XS_COLUMNS = [
    "xs_rank_return_10d", "xs_zscore_return_10d", "xs_sector_rank_return_10d", "xs_sector_zscore_return_10d",
    "xs_rank_return_20d", "xs_zscore_return_20d", "xs_sector_rank_return_20d", "xs_sector_zscore_return_20d",
]
TRAIN_END = "2022-12-31"
VALIDATE_END = "2023-12-31"
CHOSEN_THRESHOLD = 0.6

CONFIGS = {
    "original_label_original_features": {"label_col": "label", "feature_cols": BASE_FEATURE_COLUMNS},
    "original_label_plus_xs_features": {"label_col": "label", "feature_cols": BASE_FEATURE_COLUMNS + XS_COLUMNS},
    "new_20d4pct_label_plus_xs_features": {"label_col": "label_20d_4pct", "feature_cols": BASE_FEATURE_COLUMNS + XS_COLUMNS},
}


def _split_xy(df, feature_cols, label_col):
    subset = df.dropna(subset=feature_cols + [label_col])
    return subset[feature_cols], subset[label_col], subset.index


def run_config(name, cfg, matrix, lines):
    lines.append(f"\n\n## Configuration: {name}\n\n")
    lines.append(f"Label column: `{cfg['label_col']}`, feature count: {len(cfg['feature_cols'])}\n")

    train_df, validate_df, test_df = time_split(matrix, TRAIN_END, VALIDATE_END)

    X_train, y_train, _ = _split_xy(train_df, cfg["feature_cols"], cfg["label_col"])
    X_validate, y_validate, validate_index = _split_xy(validate_df, cfg["feature_cols"], cfg["label_col"])

    lines.append(f"Train rows: {len(y_train)}, Validate rows: {len(y_validate)}\n")

    model = fit_lightgbm(X_train, y_train)
    val_proba = predict_proba(model, X_validate)

    lines.append(f"Validation majority baseline: {majority_baseline(y_validate):.3f}\n")
    lines.append(f"Validation accuracy @{CHOSEN_THRESHOLD}: {accuracy_at_threshold(y_validate, val_proba, CHOSEN_THRESHOLD)}\n")

    print("=" * 60)
    print(f"[{name}] VALIDATION COMPLETE. Test set has NOT been touched yet.")
    print("=" * 60)

    X_test, y_test, test_index = _split_xy(test_df, cfg["feature_cols"], cfg["label_col"])
    test_proba = predict_proba(model, X_test)

    lines.append(f"\nTest rows: {len(y_test)} - evaluated ONCE at threshold={CHOSEN_THRESHOLD}\n")
    result = accuracy_at_threshold(y_test, test_proba, CHOSEN_THRESHOLD)
    lines.append(f"Test result: {result}\n")
    lines.append(f"Beats local_majority_baseline? {'Yes' if result['accuracy'] > result['local_majority_baseline'] else 'No'}\n")


def main():
    matrix = load_parquet("feature_matrix_extended", config.DATA_PROCESSED_DIR)
    lines = ["# Phase 5 Cross-Sectional Feature Results\n"]
    lines.append("\nLightGBM only (most consistently well-behaved algorithm across prior phases). Same full-history walk-forward split throughout.\n")

    for name, cfg in CONFIGS.items():
        run_config(name, cfg, matrix, lines)

    report_path = config.PROJECT_ROOT / "docs" / "phase5_cross_sectional_results.md"
    report_path.write_text("\n".join(lines))
    print(f"Full results written to {report_path}")


if __name__ == "__main__":
    main()

"""Phase 3 baseline, redefined to the window where real sentiment data exists.

The original walk-forward baseline (scripts/run_phase3_baselines.py) trained on
2019-2022, validated on 2023, tested on 2024-2025 - spanning the full price history
including the 2022 bear market, per the original spec. But Phase 1's news pull
(Finnhub free tier) only returned recent headlines regardless of the requested
date range, so sentiment coverage only exists from 2025-08-08 to 2025-12-16.
Evaluated over the full 2019-2025 span, sentiment features were constant zero
for the entire train/validate period, making any technical-vs-sentiment
comparison structurally impossible over that window (see the CAVEAT in
docs/phase3_results.md).

This script redefines the walk-forward split entirely within the sentiment-
coverage window, so BOTH technical and sentiment features have real, non-zero
variation across train, validate, AND test. It uses ALL rows in the window
(not just headline_count > 0 rows) - a headline_count of 0 is itself a real,
informative signal (no news that day), not missing data.

This trades away the 2019-2022 bear-market coverage the original spec wanted,
in exchange for a walk-forward split where the technical-vs-sentiment ablation
is actually meaningful. The original 2019-2025 technical-only baseline in
docs/phase3_results.md remains the historical/full-history reference.
"""
from src import config
from src.data.storage import load_parquet
from src.models.splits import time_split
from src.models.train import fit_logistic_regression, fit_lightgbm, predict_proba
from src.models.evaluate import (
    majority_baseline, accuracy_at_threshold, breakdown_by_year,
    breakdown_by_regime, sweep_confidence_threshold,
)
from src.models.ablation import run_ablation, TECHNICAL_COLUMNS, SENTIMENT_COLUMNS

FEATURE_COLUMNS = TECHNICAL_COLUMNS + SENTIMENT_COLUMNS
WINDOW_START = "2025-08-08"
WINDOW_END = "2025-12-16"
TRAIN_END = "2025-10-23"
VALIDATE_END = "2025-11-18"


def _split_xy(df):
    return df[FEATURE_COLUMNS], df["label"]


def main():
    matrix = load_parquet("feature_matrix", config.DATA_PROCESSED_DIR)
    window = matrix[(matrix.index >= WINDOW_START) & (matrix.index <= WINDOW_END)]

    train_df, validate_df, test_df = time_split(window, TRAIN_END, VALIDATE_END)

    X_train, y_train = _split_xy(train_df)
    X_validate, y_validate = _split_xy(validate_df)

    lines = []
    lines.append("# Phase 3 Recent-Window Baseline Results\n")
    lines.append(
        "\nRedefined walk-forward split restricted to the sentiment-coverage window "
        f"({WINDOW_START} to {WINDOW_END}) - see module docstring in "
        "`scripts/run_phase3_recent_window_baseline.py` for why. This trades away "
        "2019-2022 bear-market coverage for a window where technical AND sentiment "
        "features both have real variation.\n"
    )
    lines.append(f"\nWindow: {window.index.min()} to {window.index.max()} ({len(window)} rows, {window['ticker'].nunique()} tickers)\n")
    lines.append(f"\nTrain: {train_df.index.min()} to {train_df.index.max()} ({len(train_df)} rows)\n")
    lines.append(f"\nValidate: {validate_df.index.min()} to {validate_df.index.max()} ({len(validate_df)} rows)\n")

    lr_model, scaler = fit_logistic_regression(X_train, y_train)
    lgbm_model = fit_lightgbm(X_train, y_train)

    lr_val_proba = predict_proba(lr_model, X_validate, scaler=scaler)
    lgbm_val_proba = predict_proba(lgbm_model, X_validate)

    lines.append("\n## Validation results\n")
    lines.append(f"\nMajority baseline: {majority_baseline(y_validate):.3f}\n")
    lines.append(f"\nLogistic regression accuracy @0.5: {accuracy_at_threshold(y_validate, lr_val_proba, 0.5)}\n")
    lines.append(f"\nLightGBM accuracy @0.5: {accuracy_at_threshold(y_validate, lgbm_val_proba, 0.5)}\n")

    lines.append("\n## Confidence-threshold sweep (tuned on validation only)\n")
    lines.append("\nLogistic regression:\n\n")
    lines.append(sweep_confidence_threshold(y_validate, lr_val_proba).to_string())
    lines.append("\n\nLightGBM:\n\n")
    lines.append(sweep_confidence_threshold(y_validate, lgbm_val_proba).to_string())

    lines.append("\n\n## Ablation (LightGBM, evaluated on validation) - technical vs sentiment now meaningful\n\n")
    ablation_result = run_ablation(X_train, y_train, X_validate, y_validate)
    lines.append(ablation_result.to_string())

    print("=" * 60)
    print("VALIDATION RESULTS COMPLETE. Review the above before proceeding to test.")
    print("The test set has NOT been touched yet.")
    print("=" * 60)

    chosen_threshold = 0.6

    X_test, y_test = _split_xy(test_df)
    lr_test_proba = predict_proba(lr_model, X_test, scaler=scaler)
    lgbm_test_proba = predict_proba(lgbm_model, X_test)

    lines.append(f"\n\n## TEST RESULTS - evaluated ONCE at threshold={chosen_threshold}\n\n")
    lines.append(f"\nTest: {test_df.index.min()} to {test_df.index.max()} ({len(test_df)} rows)\n")
    lines.append(f"\nTest-set majority baseline: {majority_baseline(y_test):.3f}\n")
    lines.append(f"\nLogistic regression: {accuracy_at_threshold(y_test, lr_test_proba, chosen_threshold)}\n")
    lines.append(f"\nLightGBM: {accuracy_at_threshold(y_test, lgbm_test_proba, chosen_threshold)}\n")

    lines.append("\n\n### Ablation on test set (LightGBM, technical vs sentiment)\n\n")
    test_ablation = run_ablation(X_train, y_train, X_test, y_test)
    lines.append(test_ablation.to_string())

    lines.append("\n\n### Per-year breakdown (LightGBM, test set)\n\n")
    lines.append(breakdown_by_year(y_test, lgbm_test_proba, X_test.index, threshold=chosen_threshold).to_string())

    lines.append("\n\n### Per-regime breakdown (LightGBM, test set)\n\n")
    lines.append(
        "\nNote: this entire window falls within a single regime (2025 = \"flat\" per "
        "src/models/regimes.py), so this breakdown is trivially one row - included for "
        "completeness, not because multiple regimes are represented.\n\n"
    )
    lines.append(breakdown_by_regime(y_test, lgbm_test_proba, X_test.index, threshold=chosen_threshold).to_string())

    report_path = config.PROJECT_ROOT / "docs" / "phase3_recent_window_results.md"
    report_path.write_text("\n".join(lines))
    print(f"Full results written to {report_path}")


if __name__ == "__main__":
    main()

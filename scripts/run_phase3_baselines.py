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
TRAIN_END = "2022-12-31"
VALIDATE_END = "2023-12-31"


def _split_xy(df):
    return df[FEATURE_COLUMNS], df["label"]


def main():
    matrix = load_parquet("feature_matrix", config.DATA_PROCESSED_DIR)
    train_df, validate_df, test_df = time_split(matrix, TRAIN_END, VALIDATE_END)

    X_train, y_train = _split_xy(train_df)
    X_validate, y_validate = _split_xy(validate_df)

    lines = []
    lines.append("# Phase 3 Baseline Results\n")
    lines.append(f"Train: {train_df.index.min()} to {train_df.index.max()} ({len(train_df)} rows)\n")
    lines.append(f"Validate: {validate_df.index.min()} to {validate_df.index.max()} ({len(validate_df)} rows)\n")
    lines.append(f"Test (untouched until this run): {test_df.index.min()} to {test_df.index.max()} ({len(test_df)} rows)\n")

    lr_model, scaler = fit_logistic_regression(X_train, y_train)
    lgbm_model = fit_lightgbm(X_train, y_train)

    lr_val_proba = predict_proba(lr_model, X_validate, scaler=scaler)
    lgbm_val_proba = predict_proba(lgbm_model, X_validate)

    lines.append("\n## Validation-year (2023) results\n")
    lines.append(f"Logistic regression majority baseline: {majority_baseline(y_validate):.3f}\n")
    lines.append(f"Logistic regression accuracy @0.5: {accuracy_at_threshold(y_validate, lr_val_proba, 0.5)}\n")
    lines.append(f"LightGBM accuracy @0.5: {accuracy_at_threshold(y_validate, lgbm_val_proba, 0.5)}\n")

    lines.append("\n## Confidence-threshold sweep (tuned on validation only)\n")
    lines.append("Logistic regression:\n")
    lines.append(sweep_confidence_threshold(y_validate, lr_val_proba).to_string())
    lines.append("\nLightGBM:\n")
    lines.append(sweep_confidence_threshold(y_validate, lgbm_val_proba).to_string())

    lines.append("\n## Ablation (LightGBM, evaluated on validation year)\n")
    ablation_result = run_ablation(X_train, y_train, X_validate, y_validate)
    lines.append(ablation_result.to_string())

    print("=" * 60)
    print("VALIDATION RESULTS COMPLETE. Review the above before proceeding to test.")
    print("The test set (2024-2025) has NOT been touched yet.")
    print("=" * 60)

    chosen_threshold = 0.6

    X_test, y_test = _split_xy(test_df)
    lr_test_proba = predict_proba(lr_model, X_test, scaler=scaler)
    lgbm_test_proba = predict_proba(lgbm_model, X_test)

    lines.append(f"\n## TEST SET RESULTS (2024-2025) - evaluated ONCE at threshold={chosen_threshold}\n")
    lines.append(f"Test-set majority baseline: {majority_baseline(y_test):.3f}\n")
    lines.append(f"Logistic regression: {accuracy_at_threshold(y_test, lr_test_proba, chosen_threshold)}\n")
    lines.append(f"LightGBM: {accuracy_at_threshold(y_test, lgbm_test_proba, chosen_threshold)}\n")

    lines.append("\n### Per-year breakdown (LightGBM, test set)\n")
    lines.append(breakdown_by_year(y_test, lgbm_test_proba, X_test.index, threshold=chosen_threshold).to_string())

    lines.append("\n### Per-regime breakdown (LightGBM, test set)\n")
    lines.append(breakdown_by_regime(y_test, lgbm_test_proba, X_test.index, threshold=chosen_threshold).to_string())

    report_path = config.PROJECT_ROOT / "docs" / "phase3_results.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines))
    print(f"Full results written to {report_path}")


if __name__ == "__main__":
    main()

"""Supplementary analysis: does sentiment help within the narrow window where it actually exists?

The main Phase 3 walk-forward baseline (scripts/run_phase3_baselines.py) trains on
2019-2022 and evaluates on 2023/2024-2025. Phase 1's news pull (Finnhub free tier)
only returned recent headlines regardless of the requested date range, so real
sentiment coverage (headline_count > 0) only exists from 2025-08-08 to 2025-12-16
(476 rows / 43,505, across 22 of 28 tickers). The main baseline's technical-vs-sentiment
ablation is therefore structurally inconclusive: sentiment columns are constant zero
across the entire 2019-2023 training/validation period, so no model trained on that
period could ever learn to use them.

This script restricts to the real-sentiment-coverage window only, splits it
chronologically by row count (not by date - headline volume is heavily back-loaded
within this window, so a date-percentile split produces a lopsided train/eval size),
and compares technical_only vs both on that narrow slice. This is an exploratory,
small-sample supplementary analysis, not a replacement for the main baseline - it is
NOT part of the project's official walk-forward test-set evaluation (which already
ran once and is final per scripts/run_phase3_baselines.py).
"""
from src import config
from src.data.storage import load_parquet
from src.models.train import fit_lightgbm, predict_proba
from src.models.evaluate import majority_baseline, accuracy_at_threshold
from src.models.ablation import TECHNICAL_COLUMNS, SENTIMENT_COLUMNS

FEATURE_COLUMNS = TECHNICAL_COLUMNS + SENTIMENT_COLUMNS
TRAIN_FRACTION = 0.7


def _split_xy(df):
    return df[FEATURE_COLUMNS], df["label"]


def main():
    matrix = load_parquet("feature_matrix", config.DATA_PROCESSED_DIR)
    covered = matrix[matrix["headline_count"] > 0].sort_index()

    split_row = int(len(covered) * TRAIN_FRACTION)
    train_df = covered.iloc[:split_row]
    eval_df = covered.iloc[split_row:]

    X_train, y_train = _split_xy(train_df)
    X_eval, y_eval = _split_xy(eval_df)

    variants = {
        "technical_only": TECHNICAL_COLUMNS,
        "both": TECHNICAL_COLUMNS + SENTIMENT_COLUMNS,
    }

    lines = []
    lines.append("\n\n---\n\n# Supplementary: Sentiment-Coverage Sub-Period Analysis\n")
    lines.append(
        "\nExploratory, small-sample analysis - NOT part of the official walk-forward "
        "test-set evaluation. See module docstring in "
        "`scripts/run_sentiment_coverage_analysis.py` for why this exists: Phase 1's "
        "news pull only captured recent headlines (Finnhub free-tier limitation), so "
        "real sentiment coverage (headline_count > 0) exists only in a narrow window, "
        "entirely within the already-evaluated test period.\n"
    )
    lines.append(
        f"\nCoverage window: {covered.index.min()} to {covered.index.max()} "
        f"({len(covered)} rows, {covered['ticker'].nunique()} of {matrix['ticker'].nunique()} tickers)\n"
    )
    lines.append(
        f"\nSub-split (70/30 by row count, chronological, NOT the official train/validate/test split): "
        f"train {train_df.index.min()} to {train_df.index.max()} ({len(train_df)} rows), "
        f"eval {eval_df.index.min()} to {eval_df.index.max()} ({len(eval_df)} rows)\n"
    )
    lines.append(
        f"\nCAVEAT: {len(train_df)} training rows is a very small sample for LightGBM - "
        "treat any signal difference here as a weak, exploratory hint, not a confirmed finding.\n"
    )
    lines.append(f"\nEval-set majority baseline: {majority_baseline(y_eval):.3f}\n")

    lines.append("\n| variant | accuracy@0.5 | n |\n|---|---|---|\n")
    for name, columns in variants.items():
        model = fit_lightgbm(X_train[columns], y_train)
        proba = predict_proba(model, X_eval[columns])
        result = accuracy_at_threshold(y_eval, proba, threshold=0.5)
        lines.append(f"| {name} | {result['accuracy']:.4f} | {result['n_predicted']} |\n")

    report_path = config.PROJECT_ROOT / "docs" / "phase3_results.md"
    with open(report_path, "a") as f:
        f.write("".join(lines))
    print(f"Appended sentiment-coverage sub-period analysis to {report_path}")


if __name__ == "__main__":
    main()

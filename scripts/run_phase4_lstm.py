import pandas as pd

from src import config
from src.data.storage import load_parquet
from src.models.splits import time_split
from src.models.sequences import build_sequences
from src.models.train_lstm import fit_lstm, predict_proba_lstm
from src.models.evaluate import majority_baseline, accuracy_at_threshold, breakdown_by_year, breakdown_by_regime
from src.models.ablation import TECHNICAL_COLUMNS, SENTIMENT_COLUMNS

FEATURE_COLUMNS = TECHNICAL_COLUMNS + SENTIMENT_COLUMNS
LOOKBACK = 10
CHOSEN_THRESHOLD = 0.6

EXPERIMENTS = {
    "full_history": {
        "window_start": None,
        "window_end": None,
        "train_end": "2022-12-31",
        "validate_end": "2023-12-31",
    },
    "recent_window": {
        "window_start": "2025-08-08",
        "window_end": "2025-12-16",
        "train_end": "2025-10-23",
        "validate_end": "2025-11-18",
    },
}


def run_experiment(name, cfg, matrix, lines):
    lines.append(f"\n\n## Experiment: {name}\n\n")

    data = matrix
    if cfg["window_start"] is not None:
        data = matrix[(matrix.index >= cfg["window_start"]) & (matrix.index <= cfg["window_end"])]

    train_df, validate_df, test_df = time_split(data, cfg["train_end"], cfg["validate_end"])

    X_seq_train, y_train, _, _ = build_sequences(train_df, FEATURE_COLUMNS, lookback=LOOKBACK)
    X_seq_validate, y_validate, validate_index, _ = build_sequences(validate_df, FEATURE_COLUMNS, lookback=LOOKBACK)

    lines.append(f"Train sequences: {len(y_train)}\n")
    lines.append(f"Validate sequences: {len(y_validate)}\n")

    if len(y_train) == 0 or len(y_validate) == 0:
        lines.append("Insufficient data to build sequences for this experiment (lookback window exceeds available rows) - skipped.\n")
        return

    model = fit_lstm(X_seq_train, y_train, hidden_size=16, epochs=30)
    val_proba = predict_proba_lstm(model, X_seq_validate)

    y_validate_series = pd.Series(y_validate)
    lines.append(f"\nValidation majority baseline: {majority_baseline(y_validate_series):.3f}\n")
    lines.append(f"LSTM validation accuracy @0.5: {accuracy_at_threshold(y_validate_series, val_proba, 0.5)}\n")
    lines.append(f"LSTM validation accuracy @{CHOSEN_THRESHOLD} (matches the threshold used for test below): {accuracy_at_threshold(y_validate_series, val_proba, CHOSEN_THRESHOLD)}\n")

    print("=" * 60)
    print(f"[{name}] VALIDATION COMPLETE. Test set has NOT been touched yet.")
    print("=" * 60)

    X_seq_test, y_test, test_index, _ = build_sequences(test_df, FEATURE_COLUMNS, lookback=LOOKBACK)
    if len(y_test) == 0:
        lines.append("Insufficient data to build test sequences for this experiment - skipped.\n")
        return

    test_proba = predict_proba_lstm(model, X_seq_test)
    y_test_series = pd.Series(y_test, index=test_index)

    lines.append(f"\nTest ({test_index.min()} to {test_index.max()}, {len(y_test)} sequences) - evaluated ONCE at threshold={CHOSEN_THRESHOLD}\n")
    result = accuracy_at_threshold(y_test_series, test_proba, CHOSEN_THRESHOLD)
    lines.append(f"LSTM test result: {result}\n")
    lines.append(f"Beats local_majority_baseline? {'Yes' if result['accuracy'] > result['local_majority_baseline'] else 'No'}\n")

    lines.append("\n### Per-year breakdown\n")
    lines.append(breakdown_by_year(y_test_series, test_proba, test_index, threshold=CHOSEN_THRESHOLD).to_string())
    lines.append("\n\n### Per-regime breakdown\n")
    lines.append(breakdown_by_regime(y_test_series, test_proba, test_index, threshold=CHOSEN_THRESHOLD).to_string())


def main():
    matrix = load_parquet("feature_matrix", config.DATA_PROCESSED_DIR)
    lines = ["# Phase 4 Small LSTM Results\n"]
    lines.append(f"\nLookback window: {LOOKBACK} days. Single-layer LSTM, hidden_size=16, per the spec's 'keep it small' guidance.\n")

    for name, cfg in EXPERIMENTS.items():
        run_experiment(name, cfg, matrix, lines)

    report_path = config.PROJECT_ROOT / "docs" / "phase4_lstm_results.md"
    report_path.write_text("\n".join(lines))
    print(f"Full results written to {report_path}")


if __name__ == "__main__":
    main()

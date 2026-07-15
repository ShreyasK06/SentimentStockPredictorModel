"""Robustness check: are Random Forest's nominal test-set wins (recorded in
docs/phase3_results.md and docs/phase3_recent_window_results.md) real signal, or
noise from RandomForestClassifier's own randomness plus confidence-threshold
selection bias?

This is a diagnostic on data already evaluated once in
scripts/run_phase3_baselines.py and scripts/run_phase3_recent_window_baseline.py
(both already ran and are final) - it does not create a new model-selection
decision or re-tune anything against either test set, so it carries none of those
scripts' "first touch" risk. It answers: if we retrain the identical Random Forest
config with many different random seeds (same train data, same test data, same
threshold), how often does the observed baseline-beating result recur? If it's
rare/centered below baseline, the original result was noise.
"""
import numpy as np
from src import config
from src.data.storage import load_parquet
from src.models.splits import time_split
from src.models.evaluate import majority_baseline, accuracy_at_threshold
from src.models.ablation import TECHNICAL_COLUMNS, SENTIMENT_COLUMNS
from sklearn.ensemble import RandomForestClassifier

FEATURE_COLUMNS = TECHNICAL_COLUMNS + SENTIMENT_COLUMNS
CHOSEN_THRESHOLD = 0.6
N_SEEDS = 50

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


def _split_xy(df):
    return df[FEATURE_COLUMNS], df["label"]


def run_experiment(name, cfg, matrix):
    print("=" * 70)
    print(f"EXPERIMENT: {name}")
    print("=" * 70)

    data = matrix
    if cfg["window_start"] is not None:
        data = matrix[(matrix.index >= cfg["window_start"]) & (matrix.index <= cfg["window_end"])]

    train_df, validate_df, test_df = time_split(data, cfg["train_end"], cfg["validate_end"])
    X_train, y_train = _split_xy(train_df)
    X_test, y_test = _split_xy(test_df)

    baseline = majority_baseline(y_test)
    print(f"Test-set majority baseline: {baseline:.4f} (n={len(y_test)})")
    print(f"Running {N_SEEDS} random seeds...")

    threshold_accuracies = []
    threshold_coverages = []
    full_accuracies = []

    for seed in range(N_SEEDS):
        model = RandomForestClassifier(n_estimators=200, max_depth=6, random_state=seed)
        model.fit(X_train, y_train)
        proba = model.predict_proba(X_test)[:, 1]

        result = accuracy_at_threshold(y_test, proba, threshold=CHOSEN_THRESHOLD)
        threshold_accuracies.append(result["accuracy"])
        threshold_coverages.append(result["n_predicted"])

        full_result = accuracy_at_threshold(y_test, proba, threshold=0.5)
        full_accuracies.append(full_result["accuracy"])

    threshold_accuracies = np.array(threshold_accuracies)
    full_accuracies = np.array(full_accuracies)

    print()
    print(f"=== At threshold={CHOSEN_THRESHOLD} (matching original report, coverage varies by seed) ===")
    print(f"Mean accuracy across {N_SEEDS} seeds: {threshold_accuracies.mean():.4f}")
    print(f"Std: {threshold_accuracies.std():.4f}")
    print(f"Min: {threshold_accuracies.min():.4f}, Max: {threshold_accuracies.max():.4f}")
    print(f"Median n_predicted (coverage): {np.median(threshold_coverages):.0f} / {len(y_test)}")
    beat_baseline = (threshold_accuracies > baseline).sum()
    print(f"Seeds that beat baseline ({baseline:.4f}): {beat_baseline}/{N_SEEDS} ({100*beat_baseline/N_SEEDS:.1f}%)")

    print()
    print(f"=== At threshold=0.5 (full coverage, less selection bias) ===")
    print(f"Mean accuracy across {N_SEEDS} seeds: {full_accuracies.mean():.4f}")
    print(f"Std: {full_accuracies.std():.4f}")
    print(f"Min: {full_accuracies.min():.4f}, Max: {full_accuracies.max():.4f}")
    beat_baseline_full = (full_accuracies > baseline).sum()
    print(f"Seeds that beat baseline ({baseline:.4f}): {beat_baseline_full}/{N_SEEDS} ({100*beat_baseline_full/N_SEEDS:.1f}%)")
    print()


def main():
    matrix = load_parquet("feature_matrix", config.DATA_PROCESSED_DIR)
    for name, cfg in EXPERIMENTS.items():
        run_experiment(name, cfg, matrix)


if __name__ == "__main__":
    main()

import numpy as np
import pandas as pd

from src.models.regimes import label_regime


def majority_baseline(y: pd.Series) -> float:
    counts = y.value_counts(normalize=True)
    return counts.max()


def accuracy_at_threshold(y_true: pd.Series, proba: np.ndarray, threshold: float = 0.5) -> dict:
    confident = (proba >= threshold) | (proba <= 1 - threshold)
    n_predicted = int(confident.sum())
    coverage = n_predicted / len(y_true) if len(y_true) > 0 else 0.0

    if n_predicted == 0:
        return {"accuracy": float("nan"), "coverage": coverage, "n_predicted": 0, "local_majority_baseline": float("nan")}

    confident_actual = np.asarray(y_true)[confident]
    predicted = (proba[confident] >= 0.5).astype(float)
    accuracy = (predicted == confident_actual).mean()
    local_baseline = majority_baseline(pd.Series(confident_actual))

    return {"accuracy": float(accuracy), "coverage": float(coverage), "n_predicted": n_predicted, "local_majority_baseline": float(local_baseline)}


def _breakdown_by_group(y_true: pd.Series, proba: np.ndarray, groups: np.ndarray, threshold: float = 0.5) -> pd.DataFrame:
    """
    Helper to compute accuracy breakdown by group.

    Args:
        y_true: Ground truth labels
        proba: Model probabilities
        groups: Already-computed grouping array/Series (e.g., years or regime labels)
        threshold: Confidence threshold

    Returns:
        DataFrame with per-group accuracy, majority_baseline, and n
    """
    rows = {}
    for group in sorted(set(groups)):
        mask = groups == group
        result = accuracy_at_threshold(pd.Series(np.asarray(y_true)[mask]), proba[mask], threshold)
        rows[group] = {
            "accuracy": result["accuracy"],
            "majority_baseline": majority_baseline(pd.Series(np.asarray(y_true)[mask])),
            "n": int(mask.sum()),
        }
    return pd.DataFrame.from_dict(rows, orient="index")


def breakdown_by_year(y_true: pd.Series, proba: np.ndarray, index: pd.DatetimeIndex, threshold: float = 0.5) -> pd.DataFrame:
    years = pd.DatetimeIndex(index).year
    return _breakdown_by_group(y_true, proba, years, threshold)


def breakdown_by_regime(y_true: pd.Series, proba: np.ndarray, index: pd.DatetimeIndex, threshold: float = 0.5) -> pd.DataFrame:
    regimes = label_regime(index).values
    return _breakdown_by_group(y_true, proba, regimes, threshold)


def sweep_confidence_threshold(y_true: pd.Series, proba: np.ndarray, thresholds=(0.5, 0.55, 0.6, 0.65, 0.7)) -> pd.DataFrame:
    rows = {}
    for t in thresholds:
        result = accuracy_at_threshold(y_true, proba, threshold=t)
        rows[t] = {"accuracy": result["accuracy"], "coverage": result["coverage"]}
    return pd.DataFrame.from_dict(rows, orient="index")

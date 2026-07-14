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
        return {"accuracy": float("nan"), "coverage": coverage, "n_predicted": 0}

    predicted = (proba[confident] >= 0.5).astype(float)
    actual = np.asarray(y_true)[confident]
    accuracy = (predicted == actual).mean()

    return {"accuracy": float(accuracy), "coverage": float(coverage), "n_predicted": n_predicted}


def breakdown_by_year(y_true: pd.Series, proba: np.ndarray, index: pd.DatetimeIndex, threshold: float = 0.5) -> pd.DataFrame:
    years = pd.DatetimeIndex(index).year
    rows = {}
    for year in sorted(set(years)):
        mask = years == year
        result = accuracy_at_threshold(pd.Series(np.asarray(y_true)[mask]), proba[mask], threshold)
        rows[year] = {
            "accuracy": result["accuracy"],
            "majority_baseline": majority_baseline(pd.Series(np.asarray(y_true)[mask])),
            "n": int(mask.sum()),
        }
    return pd.DataFrame.from_dict(rows, orient="index")


def breakdown_by_regime(y_true: pd.Series, proba: np.ndarray, index: pd.DatetimeIndex, threshold: float = 0.5) -> pd.DataFrame:
    regimes = label_regime(index).values
    rows = {}
    for regime in sorted(set(regimes)):
        mask = regimes == regime
        result = accuracy_at_threshold(pd.Series(np.asarray(y_true)[mask]), proba[mask], threshold)
        rows[regime] = {
            "accuracy": result["accuracy"],
            "majority_baseline": majority_baseline(pd.Series(np.asarray(y_true)[mask])),
            "n": int(mask.sum()),
        }
    return pd.DataFrame.from_dict(rows, orient="index")


def sweep_confidence_threshold(y_true: pd.Series, proba: np.ndarray, thresholds=(0.5, 0.55, 0.6, 0.65, 0.7)) -> pd.DataFrame:
    rows = {}
    for t in thresholds:
        result = accuracy_at_threshold(y_true, proba, threshold=t)
        rows[t] = {"accuracy": result["accuracy"], "coverage": result["coverage"]}
    return pd.DataFrame.from_dict(rows, orient="index")

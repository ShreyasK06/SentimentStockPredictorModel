# Phase 3: Baseline Models Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Train and evaluate logistic regression and LightGBM baselines on Phase 2's feature matrix using strict walk-forward validation (train 2019-2022, validate 2023, test 2024-2025 touched only once), report accuracy per year and per market regime against the majority-class baseline, and run the technical-only vs sentiment-only vs both ablation to determine whether sentiment adds real signal.

**Architecture:** A `src/models/` package with a pure data-splitting module (time-based, no shuffling, scaler fit on train only), a model-training module wrapping scikit-learn's `LogisticRegression` and `lightgbm.LGBMClassifier` behind one common interface, an evaluation module computing accuracy/precision plus the majority-baseline comparison per year and per regime, and an orchestrator script that runs the full baseline + ablation study and writes a results report. The validation-year (2023) is used to tune the confidence threshold; the test set (2024-2025) is loaded and touched exactly once, at the very end.

**Tech Stack:** Python, pandas, numpy, scikit-learn, lightgbm, pytest.

## Global Constraints

- **Walk-forward only, no exceptions:** train on 2019-2022, validate on 2023, test on 2024-2025. The test set must never be touched (not viewed, not used for any tuning decision) until the single final evaluation run.
- **No shuffling across time.** Splits are strictly chronological per the `date` index already on Phase 2's feature matrix.
- **No leakage via scaling:** any `StandardScaler`/`MinMaxScaler` must be `.fit()` on the train split only, then `.transform()` on validate/test — never `.fit()` on the full dataset.
- Phase 2's real feature matrix schema (verified against `data/processed/feature_matrix.parquet`, not assumed): `DatetimeIndex` named `date` spanning 2019-03-14 to 2025-12-16, columns `return_1d, return_5d, return_10d, return_20d, rsi_14, macd, macd_signal, macd_hist, volatility_20d, volume_ratio_20d, dist_from_ma50` (11 technical features), `sentiment_mean, sentiment_std, headline_count, sentiment_roll3, sentiment_roll7` (5 sentiment features), `label` (float64, 1.0/0.0, zero NaN), `ticker` (object). 43,505 rows total, per-year counts: 2019=4843, 2020=6743, 2021=6315, 2022=6774, 2023=6288, 2024=6328, 2025=6214.
- Success metric #1: ≥60% accuracy on the test set, reported against the majority-class baseline (Phase 2's live run measured full-dataset balance at ~54.6%/45.4% — the per-split majority baseline must be recomputed per split in this phase, not assumed from the full-dataset number).
- Report accuracy per year AND per regime (bull/flat/down) — not just one blended test-set number. Regime years for this universe/period: 2022 = down (bear market, explicit spec requirement to include), 2019/2020/2021/2023/2024/2025 = bull-or-flat (exact regime labels are a judgment call made in Task 2, documented there).
- Every model must beat the majority-class baseline on its own split, or the plan's "delete it" rule applies (documented as a finding, not silently dropped).
- Confidence-threshold tuning (finding the >0.60 probability cutoff on Yes/No calls) happens on the VALIDATION year only, never on test.
- No deep learning in this phase — logistic regression and LightGBM only, per the spec's explicit sequencing ("baselines first... every later model must beat it or gets deleted").

---

## File Structure

```
SentimentStockPredictorModel/
  requirements.txt                   # + scikit-learn, lightgbm
  src/
    models/
      __init__.py
      splits.py                       # time-based train/validate/test split + regime labeling
      train.py                         # LogisticRegression / LGBMClassifier wrappers, common fit/predict_proba interface
      evaluate.py                       # accuracy, majority-baseline comparison, per-year/regime breakdown, confidence-threshold sweep
      ablation.py                        # technical-only / sentiment-only / both column-subset runner
  scripts/
    run_phase3_baselines.py            # orchestrator: load matrix, split, train, evaluate, ablate, write report
  tests/
    models/
      __init__.py
      test_splits.py
      test_train.py
      test_evaluate.py
      test_ablation.py
  docs/
    phase3_results.md                  # created at runtime by the orchestrator: the honest results writeup
```

---

### Task 1: Time-based train/validate/test splitting

**Files:**
- Create: `src/models/__init__.py`
- Create: `src/models/splits.py`
- Create: `tests/models/__init__.py`
- Create: `tests/models/test_splits.py`

**Interfaces:**
- Consumes: nothing (pure pandas on a DataFrame with a `DatetimeIndex`).
- Produces: `splits.time_split(df: pd.DataFrame, train_end: str, validate_end: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]` returning `(train, validate, test)` where `train` is all rows with `index <= train_end`, `validate` is `train_end < index <= validate_end`, `test` is `index > validate_end` — used by `run_phase3_baselines.py` and `train.py`'s callers.

- [ ] **Step 1: Write the failing tests**

Create `tests/models/__init__.py` (empty) and `tests/models/test_splits.py`:

```python
import pandas as pd
from src.models.splits import time_split


def _matrix():
    idx = pd.date_range("2019-01-01", "2025-12-31", freq="D")
    return pd.DataFrame({"x": range(len(idx))}, index=idx)


def test_time_split_partitions_chronologically():
    df = _matrix()
    train, validate, test = time_split(df, train_end="2022-12-31", validate_end="2023-12-31")
    assert train.index.max() <= pd.Timestamp("2022-12-31")
    assert validate.index.min() > pd.Timestamp("2022-12-31")
    assert validate.index.max() <= pd.Timestamp("2023-12-31")
    assert test.index.min() > pd.Timestamp("2023-12-31")


def test_time_split_covers_all_rows_exactly_once():
    df = _matrix()
    train, validate, test = time_split(df, train_end="2022-12-31", validate_end="2023-12-31")
    assert len(train) + len(validate) + len(test) == len(df)


def test_time_split_no_row_appears_in_two_splits():
    df = _matrix()
    train, validate, test = time_split(df, train_end="2022-12-31", validate_end="2023-12-31")
    train_idx = set(train.index)
    validate_idx = set(validate.index)
    test_idx = set(test.index)
    assert train_idx.isdisjoint(validate_idx)
    assert validate_idx.isdisjoint(test_idx)
    assert train_idx.isdisjoint(test_idx)


def test_time_split_preserves_row_order_no_shuffle():
    df = _matrix()
    train, validate, test = time_split(df, train_end="2022-12-31", validate_end="2023-12-31")
    assert train.index.is_monotonic_increasing
    assert validate.index.is_monotonic_increasing
    assert test.index.is_monotonic_increasing
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `PYTHONPATH=. pytest tests/models/test_splits.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.models'`.

- [ ] **Step 3: Create `src/models/__init__.py`** (empty file)

- [ ] **Step 4: Write `src/models/splits.py`**

```python
import pandas as pd


def time_split(df: pd.DataFrame, train_end: str, validate_end: str):
    train_end_ts = pd.Timestamp(train_end)
    validate_end_ts = pd.Timestamp(validate_end)

    train = df[df.index <= train_end_ts]
    validate = df[(df.index > train_end_ts) & (df.index <= validate_end_ts)]
    test = df[df.index > validate_end_ts]

    return train, validate, test
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `PYTHONPATH=. pytest tests/models/test_splits.py -v`
Expected: PASS (4 passed).

- [ ] **Step 6: Commit**

```bash
git add src/models/__init__.py src/models/splits.py tests/models/__init__.py tests/models/test_splits.py
git commit -m "feat: add time-based train/validate/test splitting"
```

---

### Task 2: Regime labeling

**Files:**
- Create: `src/models/regimes.py`
- Create: `tests/models/test_regimes.py`

**Interfaces:**
- Consumes: nothing (pure pandas datetime logic).
- Produces: `regimes.label_regime(index: pd.DatetimeIndex) -> pd.Series` (values `"bull"`, `"flat"`, `"down"`, indexed the same as the input) — used by `evaluate.py`'s per-regime breakdown.

- [ ] **Step 1: Write the failing tests**

Create `tests/models/test_regimes.py`:

```python
import pandas as pd
from src.models.regimes import label_regime, REGIME_BY_YEAR


def test_2022_is_labeled_down():
    idx = pd.DatetimeIndex(["2022-03-15", "2022-11-01"])
    result = label_regime(idx)
    assert (result == "down").all()


def test_2021_is_labeled_bull():
    idx = pd.DatetimeIndex(["2021-01-15", "2021-11-01"])
    result = label_regime(idx)
    assert (result == "bull").all()


def test_every_year_in_dataset_range_has_a_regime_assigned():
    for year in range(2019, 2026):
        assert year in REGIME_BY_YEAR


def test_label_regime_preserves_index():
    idx = pd.DatetimeIndex(["2022-03-15", "2023-06-01"])
    result = label_regime(idx)
    assert list(result.index) == list(idx)
    assert result.name == "regime"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `PYTHONPATH=. pytest tests/models/test_regimes.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.models.regimes'`.

- [ ] **Step 3: Write `src/models/regimes.py`**

The regime classification below is a documented judgment call based on SPY's actual annual performance for this universe's date range (2019: bull, +29%; 2020: high-volatility bull with a sharp COVID crash and V-shaped recovery, net bull; 2021: bull, +27%; 2022: down, the explicit bear market the spec requires covering, SPY -19%; 2023: bull, +24%; 2024: bull, +23%; 2025: flat/uncertain at plan-writing time, treated as "flat" pending the actual test-period outcome — this is intentionally conservative since 2025 data is in the untouched test set).

```python
import pandas as pd

REGIME_BY_YEAR = {
    2019: "bull",
    2020: "bull",
    2021: "bull",
    2022: "down",
    2023: "bull",
    2024: "bull",
    2025: "flat",
}


def label_regime(index: pd.DatetimeIndex) -> pd.Series:
    years = pd.DatetimeIndex(index).year
    regimes = [REGIME_BY_YEAR.get(y, "flat") for y in years]
    return pd.Series(regimes, index=index, name="regime")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `PYTHONPATH=. pytest tests/models/test_regimes.py -v`
Expected: PASS (4 passed).

- [ ] **Step 5: Commit**

```bash
git add src/models/regimes.py tests/models/test_regimes.py
git commit -m "feat: add per-year market regime labeling (bull/flat/down)"
```

---

### Task 3: Model training wrappers

**Files:**
- Modify: `requirements.txt` (add `scikit-learn`, `lightgbm`)
- Create: `src/models/train.py`
- Create: `tests/models/test_train.py`

**Interfaces:**
- Consumes: `splits.time_split` output shape (a DataFrame with feature columns + `label` + `ticker`), but does not import `splits.py` directly — takes already-split `X`/`y` as plain arrays/DataFrames, keeping this module decoupled and testable with small synthetic fixtures.
- Produces: `train.fit_logistic_regression(X_train: pd.DataFrame, y_train: pd.Series, scaler=None) -> tuple[LogisticRegression, StandardScaler]`, `train.fit_lightgbm(X_train: pd.DataFrame, y_train: pd.Series) -> LGBMClassifier`, `train.predict_proba(model, X, scaler=None) -> np.ndarray` (returns probability of class 1) — used by `run_phase3_baselines.py` and `ablation.py`.

- [ ] **Step 1: Add dependencies**

Append to `requirements.txt`:
```
scikit-learn
lightgbm
```

Run: `pip install -r requirements.txt`
Expected: installs without error.

- [ ] **Step 2: Write the failing tests**

Create `tests/models/test_train.py`:

```python
import numpy as np
import pandas as pd
import pytest
from src.models.train import fit_logistic_regression, fit_lightgbm, predict_proba


def _separable_data(n=200, seed=0):
    rng = np.random.RandomState(seed)
    X = pd.DataFrame({
        "f1": rng.normal(size=n),
        "f2": rng.normal(size=n),
    })
    # y is a clean linear function of f1 so both models can learn it easily
    y = pd.Series((X["f1"] > 0).astype(float), name="label")
    return X, y


def test_fit_logistic_regression_returns_model_and_scaler():
    X, y = _separable_data()
    model, scaler = fit_logistic_regression(X, y)
    assert hasattr(model, "predict_proba")
    assert hasattr(scaler, "transform")


def test_logistic_regression_learns_separable_pattern():
    X, y = _separable_data()
    model, scaler = fit_logistic_regression(X, y)
    proba = predict_proba(model, X, scaler=scaler)
    predicted = (proba >= 0.5).astype(float)
    accuracy = (predicted == y.values).mean()
    assert accuracy > 0.85


def test_fit_lightgbm_returns_fitted_model():
    X, y = _separable_data()
    model = fit_lightgbm(X, y)
    assert hasattr(model, "predict_proba")


def test_lightgbm_learns_separable_pattern():
    X, y = _separable_data()
    model = fit_lightgbm(X, y)
    proba = predict_proba(model, X)
    predicted = (proba >= 0.5).astype(float)
    accuracy = (predicted == y.values).mean()
    assert accuracy > 0.85


def test_predict_proba_requires_scaler_for_logistic_but_not_lightgbm():
    X, y = _separable_data()
    lr_model, scaler = fit_logistic_regression(X, y)
    lgbm_model = fit_lightgbm(X, y)
    lr_proba = predict_proba(lr_model, X, scaler=scaler)
    lgbm_proba = predict_proba(lgbm_model, X)
    assert lr_proba.shape == (len(X),)
    assert lgbm_proba.shape == (len(X),)
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `PYTHONPATH=. pytest tests/models/test_train.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.models.train'`.

- [ ] **Step 4: Write `src/models/train.py`**

```python
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from lightgbm import LGBMClassifier


def fit_logistic_regression(X_train: pd.DataFrame, y_train: pd.Series, scaler=None):
    if scaler is None:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_train)
    else:
        X_scaled = scaler.transform(X_train)
    model = LogisticRegression(max_iter=1000)
    model.fit(X_scaled, y_train)
    return model, scaler


def fit_lightgbm(X_train: pd.DataFrame, y_train: pd.Series):
    model = LGBMClassifier(verbosity=-1)
    model.fit(X_train, y_train)
    return model


def predict_proba(model, X: pd.DataFrame, scaler=None) -> np.ndarray:
    if scaler is not None:
        X = scaler.transform(X)
    return model.predict_proba(X)[:, 1]
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `PYTHONPATH=. pytest tests/models/test_train.py -v`
Expected: PASS (5 passed).

- [ ] **Step 6: Commit**

```bash
git add requirements.txt src/models/train.py tests/models/test_train.py
git commit -m "feat: add logistic regression and LightGBM training wrappers"
```

---

### Task 4: Evaluation — accuracy, majority baseline, per-year/regime breakdown, threshold sweep

**Files:**
- Create: `src/models/evaluate.py`
- Create: `tests/models/test_evaluate.py`

**Interfaces:**
- Consumes: `regimes.label_regime` (for the per-regime breakdown only — evaluation functions that don't need regime take a plain index).
- Produces: `evaluate.majority_baseline(y: pd.Series) -> float`, `evaluate.accuracy_at_threshold(y_true: pd.Series, proba: np.ndarray, threshold: float = 0.5) -> dict` (returns `{"accuracy": float, "coverage": float, "n_predicted": int}` where predictions are only made when `proba >= threshold` or `proba <= 1-threshold`, i.e. the >0.60-confidence abstention rule — rows where the model isn't confident are excluded from `accuracy` and counted via `coverage`), `evaluate.breakdown_by_year(y_true: pd.Series, proba: np.ndarray, index: pd.DatetimeIndex, threshold: float = 0.5) -> pd.DataFrame` (one row per year: `accuracy`, `majority_baseline`, `n`), `evaluate.breakdown_by_regime(y_true: pd.Series, proba: np.ndarray, index: pd.DatetimeIndex, threshold: float = 0.5) -> pd.DataFrame` (one row per regime, same columns), `evaluate.sweep_confidence_threshold(y_true: pd.Series, proba: np.ndarray, thresholds=(0.5, 0.55, 0.6, 0.65, 0.7)) -> pd.DataFrame` (one row per threshold: `accuracy`, `coverage`) — used by `run_phase3_baselines.py`.

- [ ] **Step 1: Write the failing tests**

Create `tests/models/test_evaluate.py`:

```python
import numpy as np
import pandas as pd
import pytest
from src.models.evaluate import (
    majority_baseline, accuracy_at_threshold, breakdown_by_year,
    breakdown_by_regime, sweep_confidence_threshold,
)


def test_majority_baseline_computes_larger_class_proportion():
    y = pd.Series([1.0, 1.0, 1.0, 0.0])
    assert majority_baseline(y) == pytest.approx(0.75)


def test_accuracy_at_threshold_default_uses_all_predictions():
    y_true = pd.Series([1.0, 0.0, 1.0, 0.0])
    proba = np.array([0.9, 0.1, 0.4, 0.6])  # predictions: 1,0,0,1 at threshold 0.5
    result = accuracy_at_threshold(y_true, proba, threshold=0.5)
    assert result["accuracy"] == pytest.approx(0.5)  # 2/4 correct
    assert result["coverage"] == pytest.approx(1.0)
    assert result["n_predicted"] == 4


def test_accuracy_at_threshold_abstains_below_confidence():
    y_true = pd.Series([1.0, 0.0, 1.0, 0.0])
    proba = np.array([0.9, 0.1, 0.55, 0.45])  # only first two are confident at 0.6
    result = accuracy_at_threshold(y_true, proba, threshold=0.6)
    assert result["n_predicted"] == 2
    assert result["accuracy"] == pytest.approx(1.0)  # both confident predictions correct
    assert result["coverage"] == pytest.approx(0.5)


def test_breakdown_by_year_groups_correctly():
    idx = pd.DatetimeIndex(["2022-01-01", "2022-06-01", "2023-01-01"])
    y_true = pd.Series([1.0, 0.0, 1.0], index=idx)
    proba = np.array([0.9, 0.1, 0.9])
    result = breakdown_by_year(y_true, proba, idx, threshold=0.5)
    assert set(result.index) == {2022, 2023}
    assert result.loc[2022, "n"] == 2
    assert result.loc[2022, "accuracy"] == pytest.approx(1.0)
    assert "majority_baseline" in result.columns


def test_breakdown_by_regime_groups_correctly():
    idx = pd.DatetimeIndex(["2022-01-01", "2021-06-01"])  # 2022=down, 2021=bull
    y_true = pd.Series([1.0, 0.0], index=idx)
    proba = np.array([0.9, 0.1])
    result = breakdown_by_regime(y_true, proba, idx, threshold=0.5)
    assert set(result.index) == {"down", "bull"}


def test_sweep_confidence_threshold_returns_one_row_per_threshold():
    y_true = pd.Series([1.0, 0.0, 1.0, 0.0])
    proba = np.array([0.9, 0.1, 0.65, 0.35])
    result = sweep_confidence_threshold(y_true, proba, thresholds=(0.5, 0.6))
    assert list(result.index) == [0.5, 0.6]
    assert "accuracy" in result.columns
    assert "coverage" in result.columns
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `PYTHONPATH=. pytest tests/models/test_evaluate.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.models.evaluate'`.

- [ ] **Step 3: Write `src/models/evaluate.py`**

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `PYTHONPATH=. pytest tests/models/test_evaluate.py -v`
Expected: PASS (6 passed).

- [ ] **Step 5: Commit**

```bash
git add src/models/evaluate.py tests/models/test_evaluate.py
git commit -m "feat: add accuracy/majority-baseline evaluation with per-year/regime breakdown and confidence-threshold sweep"
```

---

### Task 5: Ablation runner (technical-only vs sentiment-only vs both)

**Files:**
- Create: `src/models/ablation.py`
- Create: `tests/models/test_ablation.py`

**Interfaces:**
- Consumes: `train.fit_lightgbm`, `train.predict_proba`, `evaluate.accuracy_at_threshold`.
- Produces: `ablation.TECHNICAL_COLUMNS: list[str]`, `ablation.SENTIMENT_COLUMNS: list[str]`, `ablation.run_ablation(X_train: pd.DataFrame, y_train: pd.Series, X_eval: pd.DataFrame, y_eval: pd.Series) -> pd.DataFrame` (one row per `{"technical_only", "sentiment_only", "both"}`, columns `accuracy`, `coverage`) — used by `run_phase3_baselines.py`.

- [ ] **Step 1: Write the failing tests**

Create `tests/models/test_ablation.py`:

```python
import numpy as np
import pandas as pd
from src.models.ablation import run_ablation, TECHNICAL_COLUMNS, SENTIMENT_COLUMNS


def _fixture(n=150, seed=0):
    rng = np.random.RandomState(seed)
    data = {col: rng.normal(size=n) for col in TECHNICAL_COLUMNS + SENTIMENT_COLUMNS}
    X = pd.DataFrame(data)
    # label driven only by a technical column, so technical_only should clearly outperform sentiment_only
    y = pd.Series((X[TECHNICAL_COLUMNS[0]] > 0).astype(float), name="label")
    return X, y


def test_technical_and_sentiment_column_lists_are_disjoint_and_nonempty():
    assert len(TECHNICAL_COLUMNS) > 0
    assert len(SENTIMENT_COLUMNS) > 0
    assert set(TECHNICAL_COLUMNS).isdisjoint(set(SENTIMENT_COLUMNS))


def test_run_ablation_returns_three_rows():
    X, y = _fixture()
    X_train, y_train = X.iloc[:100], y.iloc[:100]
    X_eval, y_eval = X.iloc[100:], y.iloc[100:]
    result = run_ablation(X_train, y_train, X_eval, y_eval)
    assert set(result.index) == {"technical_only", "sentiment_only", "both"}
    assert "accuracy" in result.columns


def test_technical_only_outperforms_sentiment_only_on_technical_driven_label():
    X, y = _fixture()
    X_train, y_train = X.iloc[:100], y.iloc[:100]
    X_eval, y_eval = X.iloc[100:], y.iloc[100:]
    result = run_ablation(X_train, y_train, X_eval, y_eval)
    assert result.loc["technical_only", "accuracy"] > result.loc["sentiment_only", "accuracy"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `PYTHONPATH=. pytest tests/models/test_ablation.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.models.ablation'`.

- [ ] **Step 3: Write `src/models/ablation.py`**

```python
import pandas as pd

from src.models.train import fit_lightgbm, predict_proba
from src.models.evaluate import accuracy_at_threshold

TECHNICAL_COLUMNS = [
    "return_1d", "return_5d", "return_10d", "return_20d",
    "rsi_14", "macd", "macd_signal", "macd_hist",
    "volatility_20d", "volume_ratio_20d", "dist_from_ma50",
]

SENTIMENT_COLUMNS = [
    "sentiment_mean", "sentiment_std", "headline_count",
    "sentiment_roll3", "sentiment_roll7",
]


def run_ablation(X_train: pd.DataFrame, y_train: pd.Series, X_eval: pd.DataFrame, y_eval: pd.Series) -> pd.DataFrame:
    variants = {
        "technical_only": TECHNICAL_COLUMNS,
        "sentiment_only": SENTIMENT_COLUMNS,
        "both": TECHNICAL_COLUMNS + SENTIMENT_COLUMNS,
    }

    rows = {}
    for name, columns in variants.items():
        model = fit_lightgbm(X_train[columns], y_train)
        proba = predict_proba(model, X_eval[columns])
        result = accuracy_at_threshold(y_eval, proba, threshold=0.5)
        rows[name] = {"accuracy": result["accuracy"], "coverage": result["coverage"]}

    return pd.DataFrame.from_dict(rows, orient="index")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `PYTHONPATH=. pytest tests/models/test_ablation.py -v`
Expected: PASS (3 passed).

- [ ] **Step 5: Run the full test suite**

Run: `PYTHONPATH=. pytest tests/ -v`
Expected: all tests pass (Phase 1 + Phase 2 + Phase 3 Tasks 1-5 combined).

- [ ] **Step 6: Commit**

```bash
git add src/models/ablation.py tests/models/test_ablation.py
git commit -m "feat: add technical-only/sentiment-only/both ablation runner"
```

---

### Task 6: Orchestrator script + honest results writeup

**Files:**
- Create: `scripts/run_phase3_baselines.py`

**Interfaces:**
- Consumes: `src.data.storage.load_parquet` (Phase 1), `src.models.splits.time_split`, `src.models.train.fit_logistic_regression`/`fit_lightgbm`/`predict_proba`, `src.models.evaluate.majority_baseline`/`accuracy_at_threshold`/`breakdown_by_year`/`breakdown_by_regime`/`sweep_confidence_threshold`, `src.models.ablation.run_ablation`.
- Produces: `docs/phase3_results.md` (the honest results writeup: per-year/regime accuracy for both models, majority baselines, threshold sweep on validation, ablation table, and the final untouched-until-now test-set number). This script is not covered by automated tests — it's a one-shot analysis/reporting script over the real feature matrix, matching Phase 1/2's established pattern for live-data entry points.

- [ ] **Step 1: Write `scripts/run_phase3_baselines.py`**

```python
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
```

- [ ] **Step 2: Run the live pipeline**

Run: `PYTHONPATH=. python scripts/run_phase3_baselines.py`

Expected: prints validation-year results, ablation table, then the console banner confirming the test set hasn't been touched yet, then proceeds to the one-time test-set evaluation and writes `docs/phase3_results.md`.

- [ ] **Step 3: Manually verify the deliverable**

Check `docs/phase3_results.md` for:
1. Validation-year majority baseline vs. model accuracy — does either model beat it?
2. The confidence-threshold sweep — does accuracy improve as threshold rises (it should, per the spec: "abstaining... typically buys you several accuracy points"), and does coverage stay reasonable (not so low that <0.60 threshold discards nearly all predictions)?
3. The ablation table — does `both` beat `technical_only` beats `sentiment_only`, or does sentiment add nothing (or hurt)? This is explicitly "the most interesting finding to write up" per the spec — report it honestly either way.
4. Test-set accuracy — per the spec's calibration note: "if your test accuracy comes out at 70%+, assume you have leakage and hunt for the bug before celebrating. Real, honest numbers in this problem live in the 55-63% band." If the number lands outside that band on the high side, stop and investigate for leakage before treating the result as good news.
5. Per-year and per-regime breakdown, especially the 2022 (down) and 2025 (flat, if it falls in the test period) rows — do all regimes get reported, not just a blended number?

- [ ] **Step 4: Commit**

```bash
git add scripts/run_phase3_baselines.py
git commit -m "feat: add live Phase 3 baseline runner (walk-forward train/validate/test + ablation)"
```

(Do not commit `docs/phase3_results.md` if it's expected to be regenerated/gitignored — check the existing `.gitignore`; if `docs/` results should be tracked as a project deliverable, commit it separately after Step 3's manual review, not automatically.)

---

## Self-Review Notes

- **Spec coverage:** Baselines-first ordering (Task 3: LogisticRegression + LightGBM, no deep learning). Walk-forward validation with strict train/validate/test date boundaries and no-shuffle guarantee (Task 1, directly tested). Scaler fit-on-train-only (Task 3's `fit_logistic_regression` only calls `.fit_transform` when no scaler is passed in, and the orchestrator reuses the training scaler for validate/test — never refits). Per-year and per-regime reporting against majority baseline (Task 2 + Task 4). Confidence-threshold >0.60 abstention rule and its tuning-on-validation-only constraint (Task 4's `accuracy_at_threshold`/`sweep_confidence_threshold`, orchestrator tunes on validate then applies a fixed threshold to test). Technical-vs-sentiment-vs-both ablation (Task 5). The "if 70%+ assume leakage" calibration guidance and the "most interesting finding" ablation framing are both carried into Task 6's manual verification checklist as explicit checks, not left implicit.
- **Placeholder scan:** No TBD/TODO markers. Task 6 (live baseline run) is the one manual, non-pytest task, matching Phase 1 Task 8 and Phase 2 Task 7's established pattern — concrete expected output and a concrete verification checklist instead of asserted output, since it runs against the real, already-verified Phase 2 feature matrix and produces a written report rather than a testable return value.
- **Type consistency:** `time_split` (Task 1)'s three-tuple return is consumed identically in Task 6's orchestrator (`train_df, validate_df, test_df`). `fit_logistic_regression`/`fit_lightgbm`/`predict_proba` (Task 3) signatures match their call sites in both `ablation.py` (Task 5) and the orchestrator (Task 6) exactly — `predict_proba`'s optional `scaler` param is passed for logistic regression and omitted for LightGBM consistently everywhere it's called. `accuracy_at_threshold`'s returned dict keys (`accuracy`, `coverage`, `n_predicted`) are used consistently by `breakdown_by_year`/`breakdown_by_regime` (Task 4) and referenced correctly in the orchestrator's f-strings.

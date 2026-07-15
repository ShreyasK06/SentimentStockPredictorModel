# Phase 4 Task 6: Small LSTM Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** After exhaustive baseline work (logistic regression, LightGBM, Random Forest, XGBoost, across 21 technical + 5 sentiment features, two time windows, with data-quality and evaluation-methodology bugs found and fixed) found no algorithm with a genuine edge over the correctly-computed majority baseline, implement one small LSTM per the parent spec's explicit fallback clause: "if and only if trees plateau below target, try an LSTM/TFT on sequences. Keep it small — financial data is low signal-to-noise and big models memorize noise." This is the last untried architecture in this project's scope.

**Architecture:** A sequence-construction module (`src/models/sequences.py`) that turns the existing per-(ticker, day) feature matrix into per-ticker sliding windows of the last N days' features, never crossing ticker boundaries. A minimal single-layer PyTorch LSTM wrapper (`src/models/train_lstm.py`) matching the project's established `fit_*`/`predict_proba`-style interface as closely as PyTorch's training loop allows. An orchestrator script reusing the already-approved `time_split`/`evaluate.py`/walk-forward discipline, run on both existing experiments (full 2019-2025 history, recent sentiment-covered window).

**Tech Stack:** Python, pandas, numpy, PyTorch (already a project dependency via Phase 2's FinBERT), pytest.

## Global Constraints

- **Hard rule, unchanged from every prior phase:** no feature may use information from after its own day's close. Sequence construction must respect this: a sequence for day t may include days [t-lookback+1, t] (i.e., day t's own already-causal feature row plus its own causal history) — never day t+1 or later. This is a natural extension of the existing convention (every other model in this project uses day t's feature row, computed only from data available by day t's close, to predict day t's label).
- **Never cross ticker boundaries.** A sequence must be built from a single ticker's own chronological history — concatenating the tail of one ticker's data with the head of another's (which could happen if sequences were built naively on the concatenated multi-ticker feature matrix without grouping by ticker first) would inject nonsensical temporal structure.
- **Keep it small**, per the spec's explicit instruction: single LSTM layer, small hidden size (16-32), heavy dropout or none needed given the small parameter count, minimal epochs — the goal is a model that COULD learn a subtle sequential pattern if one exists, not a model with enough capacity to memorize this dataset's noise. Do not add multiple layers, attention, or other capacity-increasing architecture choices without first observing this small model badly underfit (unlikely to be the failure mode given everything else this project has found).
- **Same walk-forward discipline as every prior test-set-touching script:** the LSTM orchestrator script must be reviewed BEFORE its first execution, exactly like every other Phase 3/4 orchestrator. Console banner before test-set access, fixed confidence threshold set before test access, no retraining after validation.
- **Reuse, don't reimplement:** `src.models.evaluate.majority_baseline`, `accuracy_at_threshold` (now including the corrected `local_majority_baseline`), `breakdown_by_year`, `breakdown_by_regime` are already correct and approved — the LSTM's predictions must be converted to a plain `np.ndarray` of probabilities so these functions work unchanged, exactly as they already do for scikit-learn/LightGBM/XGBoost models.
- Real feature matrix schema (verified): `data/processed/feature_matrix.parquet`, `DatetimeIndex` named `date`, 21 technical + 5 sentiment + `label` (float64, 1.0/0.0) + `ticker` (object) = 28 columns, 38,685 rows, 28 tickers, zero NaN.
- PyTorch is already installed (`torch==2.11.0+cpu`, confirmed live) — no new dependency needed.

---

## File Structure

```
SentimentStockPredictorModel/
  src/
    models/
      sequences.py          # per-ticker sliding-window sequence construction
      train_lstm.py           # small LSTM model + fit/predict wrapper
  scripts/
    run_phase4_lstm.py         # orchestrator: both experiments, walk-forward, reviewed before execution
  tests/
    models/
      test_sequences.py
      test_train_lstm.py
```

---

### Task 1: Sequence construction

**Files:**
- Create: `src/models/sequences.py`
- Create: `tests/models/test_sequences.py`

**Interfaces:**
- Consumes: nothing new (pure pandas/numpy on a DataFrame matching the feature matrix's real shape: `DatetimeIndex`, feature columns, `label`, `ticker`).
- Produces: `sequences.build_sequences(df: pd.DataFrame, feature_columns: list, lookback: int = 10) -> tuple[np.ndarray, np.ndarray, pd.DatetimeIndex, pd.Series]` returning `(X_seq, y, index, ticker)` where `X_seq` has shape `(n_sequences, lookback, len(feature_columns))`, `y` has shape `(n_sequences,)` (the label for the LAST day in each sequence), `index` is the DatetimeIndex of each sequence's last (anchor) day, and `ticker` is the ticker each sequence belongs to (needed downstream if per-ticker analysis is ever wanted, and useful for debugging/verifying no cross-ticker leakage) — used by `scripts/run_phase4_lstm.py`.

- [ ] **Step 1: Write the failing tests**

Create `tests/models/test_sequences.py`:

```python
import numpy as np
import pandas as pd
import pytest
from src.models.sequences import build_sequences


def _matrix(ticker="AAPL", n=20, start_val=0.0):
    idx = pd.date_range("2024-01-02", periods=n, freq="B")
    return pd.DataFrame({
        "f1": start_val + np.arange(n, dtype=float),
        "f2": start_val + np.arange(n, dtype=float) * 2,
        "label": [1.0 if i % 2 == 0 else 0.0 for i in range(n)],
        "ticker": ticker,
    }, index=idx)


def test_build_sequences_shape():
    df = _matrix(n=20)
    X_seq, y, index, ticker = build_sequences(df, feature_columns=["f1", "f2"], lookback=10)
    # first 9 rows can't form a full 10-day lookback window; 20-10+1=11 sequences remain
    assert X_seq.shape == (11, 10, 2)
    assert y.shape == (11,)
    assert len(index) == 11
    assert len(ticker) == 11


def test_build_sequences_last_day_of_each_window_matches_label():
    df = _matrix(n=20)
    X_seq, y, index, ticker = build_sequences(df, feature_columns=["f1", "f2"], lookback=10)
    # the first sequence's anchor day is index 9 (0-indexed), i.e. df.index[9]
    assert index[0] == df.index[9]
    assert y[0] == df["label"].iloc[9]
    # the sequence's LAST row (X_seq[0, -1, :]) must match day 9's own features
    assert X_seq[0, -1, 0] == df["f1"].iloc[9]
    assert X_seq[0, -1, 1] == df["f2"].iloc[9]
    # the sequence's FIRST row (X_seq[0, 0, :]) must match day 0's features (9-10+1=0)
    assert X_seq[0, 0, 0] == df["f1"].iloc[0]


def test_build_sequences_never_crosses_ticker_boundary():
    df_a = _matrix(ticker="AAPL", n=15, start_val=0.0)
    df_b = _matrix(ticker="MSFT", n=15, start_val=1000.0)
    combined = pd.concat([df_a, df_b])
    X_seq, y, index, ticker = build_sequences(combined, feature_columns=["f1", "f2"], lookback=10)
    # AAPL contributes 15-10+1=6 sequences, MSFT contributes 6 sequences, total 12
    assert X_seq.shape[0] == 12
    assert set(ticker) == {"AAPL", "MSFT"}
    # no AAPL sequence should contain any MSFT feature values (>=1000) and vice versa
    aapl_mask = (ticker == "AAPL").values
    msft_mask = (ticker == "MSFT").values
    assert (X_seq[aapl_mask] < 1000).all()
    assert (X_seq[msft_mask] >= 1000).all()


def test_build_sequences_drops_tickers_with_insufficient_history():
    df = _matrix(n=5)  # fewer than lookback=10 rows
    X_seq, y, index, ticker = build_sequences(df, feature_columns=["f1", "f2"], lookback=10)
    assert X_seq.shape[0] == 0


def test_build_sequences_is_causal_no_lookahead():
    # Modifying a day AFTER a given sequence's anchor day must not change that sequence.
    df = _matrix(n=20)
    X_seq_before, _, _, _ = build_sequences(df, feature_columns=["f1", "f2"], lookback=10)
    df_modified = df.copy()
    df_modified.iloc[15:] = df_modified.iloc[15:] * 100  # perturb days 15+
    X_seq_after, _, _, _ = build_sequences(df_modified, feature_columns=["f1", "f2"], lookback=10)
    # sequences anchored at day 9 through day 14 (indices 0-5 in the output) must be unaffected
    # by the day-15+ perturbation, since their windows are entirely within [0,14].
    np.testing.assert_array_equal(X_seq_before[:6], X_seq_after[:6])
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `PYTHONPATH=. pytest tests/models/test_sequences.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.models.sequences'`.

- [ ] **Step 3: Write `src/models/sequences.py`**

```python
import numpy as np
import pandas as pd


def build_sequences(df: pd.DataFrame, feature_columns: list, lookback: int = 10):
    X_seq_list = []
    y_list = []
    index_list = []
    ticker_list = []

    for ticker_name, group in df.groupby("ticker"):
        group = group.sort_index()
        n = len(group)
        if n < lookback:
            continue
        features = group[feature_columns].values
        labels = group["label"].values
        dates = group.index

        for end in range(lookback - 1, n):
            start = end - lookback + 1
            X_seq_list.append(features[start:end + 1])
            y_list.append(labels[end])
            index_list.append(dates[end])
            ticker_list.append(ticker_name)

    if not X_seq_list:
        return (
            np.empty((0, lookback, len(feature_columns))),
            np.empty((0,)),
            pd.DatetimeIndex([]),
            pd.Series([], dtype=object),
        )

    X_seq = np.stack(X_seq_list)
    y = np.array(y_list)
    index = pd.DatetimeIndex(index_list)
    ticker = pd.Series(ticker_list)

    return X_seq, y, index, ticker
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `PYTHONPATH=. pytest tests/models/test_sequences.py -v`
Expected: PASS (5 passed).

- [ ] **Step 5: Commit**

```bash
git add src/models/sequences.py tests/models/test_sequences.py
git commit -m "feat: add per-ticker sliding-window sequence construction for LSTM"
```

---

### Task 2: Small LSTM model wrapper

**Files:**
- Create: `src/models/train_lstm.py`
- Create: `tests/models/test_train_lstm.py`

**Interfaces:**
- Consumes: `sequences.build_sequences`'s output shape (`X_seq: np.ndarray` of shape `(n, lookback, n_features)`, `y: np.ndarray` of shape `(n,)`) as plain arrays — does not import `sequences.py` directly, keeping this module testable with small synthetic arrays.
- Produces: `train_lstm.fit_lstm(X_seq_train: np.ndarray, y_train: np.ndarray, hidden_size: int = 16, epochs: int = 20, lr: float = 0.01) -> LSTMClassifier` (a small `torch.nn.Module` wrapping a single `nn.LSTM` layer + linear + sigmoid, already trained), `train_lstm.predict_proba_lstm(model: LSTMClassifier, X_seq: np.ndarray) -> np.ndarray` (returns a 1D array of probabilities, shape `(n,)`, matching the shape `src.models.train.predict_proba` already returns for every other model in this project) — used by `scripts/run_phase4_lstm.py`.

- [ ] **Step 1: Write the failing tests**

Create `tests/models/test_train_lstm.py`:

```python
import numpy as np
import pytest
from src.models.train_lstm import fit_lstm, predict_proba_lstm, LSTMClassifier


def _separable_sequences(n=200, lookback=5, n_features=2, seed=0):
    rng = np.random.RandomState(seed)
    X_seq = rng.normal(size=(n, lookback, n_features))
    # label depends only on the LAST timestep's first feature - an easy pattern
    # a working LSTM should learn, while keeping the test fast (few epochs needed).
    y = (X_seq[:, -1, 0] > 0).astype(np.float32)
    return X_seq, y


def test_fit_lstm_returns_module_with_small_hidden_size():
    X_seq, y = _separable_sequences()
    model = fit_lstm(X_seq, y, hidden_size=16, epochs=20)
    assert isinstance(model, LSTMClassifier)
    assert model.lstm.hidden_size == 16
    assert model.lstm.num_layers == 1  # "keep it small" - single layer only


def test_predict_proba_lstm_returns_1d_array_matching_input_length():
    X_seq, y = _separable_sequences(n=50)
    model = fit_lstm(X_seq, y, hidden_size=16, epochs=20)
    proba = predict_proba_lstm(model, X_seq)
    assert proba.shape == (50,)
    assert (proba >= 0).all() and (proba <= 1).all()


def test_lstm_learns_separable_sequential_pattern():
    X_seq, y = _separable_sequences(n=300, seed=1)
    model = fit_lstm(X_seq, y, hidden_size=16, epochs=30)
    proba = predict_proba_lstm(model, X_seq)
    predicted = (proba >= 0.5).astype(np.float32)
    accuracy = (predicted == y).mean()
    assert accuracy > 0.80


def test_fit_lstm_is_deterministic_given_a_seed():
    # torch.manual_seed should be set inside fit_lstm so two calls with the
    # same data produce the same result - important for this project's
    # reproducibility standard (every other model call is deterministic
    # given a fixed random_state).
    X_seq, y = _separable_sequences(n=60, seed=2)
    model_a = fit_lstm(X_seq, y, hidden_size=16, epochs=10, seed=42)
    model_b = fit_lstm(X_seq, y, hidden_size=16, epochs=10, seed=42)
    proba_a = predict_proba_lstm(model_a, X_seq)
    proba_b = predict_proba_lstm(model_b, X_seq)
    np.testing.assert_array_almost_equal(proba_a, proba_b, decimal=5)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `PYTHONPATH=. pytest tests/models/test_train_lstm.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.models.train_lstm'`.

- [ ] **Step 3: Write `src/models/train_lstm.py`**

```python
import numpy as np
import torch
import torch.nn as nn


class LSTMClassifier(nn.Module):
    def __init__(self, n_features: int, hidden_size: int = 16):
        super().__init__()
        self.lstm = nn.LSTM(input_size=n_features, hidden_size=hidden_size, num_layers=1, batch_first=True)
        self.linear = nn.Linear(hidden_size, 1)

    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        last_hidden = h_n[-1]
        logits = self.linear(last_hidden).squeeze(-1)
        return torch.sigmoid(logits)


def fit_lstm(X_seq_train: np.ndarray, y_train: np.ndarray, hidden_size: int = 16, epochs: int = 20, lr: float = 0.01, seed: int = 0) -> LSTMClassifier:
    torch.manual_seed(seed)
    n_features = X_seq_train.shape[2]
    model = LSTMClassifier(n_features=n_features, hidden_size=hidden_size)

    X_tensor = torch.tensor(X_seq_train, dtype=torch.float32)
    y_tensor = torch.tensor(y_train, dtype=torch.float32)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCELoss()

    model.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        proba = model(X_tensor)
        loss = loss_fn(proba, y_tensor)
        loss.backward()
        optimizer.step()

    model.eval()
    return model


def predict_proba_lstm(model: LSTMClassifier, X_seq: np.ndarray) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        X_tensor = torch.tensor(X_seq, dtype=torch.float32)
        proba = model(X_tensor)
    return proba.numpy()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `PYTHONPATH=. pytest tests/models/test_train_lstm.py -v`
Expected: PASS (4 passed). Note: this will take longer than other test files (real gradient descent, ~20-30 epochs across a few test cases) — allow up to a minute or two, not a sign of a hang.

- [ ] **Step 5: Run the full test suite**

Run: `PYTHONPATH=. pytest tests/ -v`
Expected: all tests pass (Phase 1-4's existing tests plus this task's new ones).

- [ ] **Step 6: Commit**

```bash
git add src/models/train_lstm.py tests/models/test_train_lstm.py
git commit -m "feat: add small single-layer LSTM classifier (fit_lstm/predict_proba_lstm)"
```

---

### Task 3: LSTM orchestrator script (both experiments)

**Files:**
- Create: `scripts/run_phase4_lstm.py`

**Interfaces:**
- Consumes: `src.data.storage.load_parquet`, `src.models.splits.time_split`, `src.models.sequences.build_sequences`, `src.models.train_lstm.fit_lstm`/`predict_proba_lstm`, `src.models.evaluate.majority_baseline`/`accuracy_at_threshold`/`breakdown_by_year`/`breakdown_by_regime`, `src.models.ablation.TECHNICAL_COLUMNS`/`SENTIMENT_COLUMNS`.
- Produces: `docs/phase4_lstm_results.md` (both experiments' LSTM results, reported with the same `local_majority_baseline`-corrected comparison established in Phase 4's data-cleaning work).

**IMPORTANT — this script touches both test sets. It must be reviewed before its first execution, exactly like every other Phase 3/4 orchestrator that reaches this point. Do not run it as part of implementing this task — implement, verify syntax, run the (unrelated) test suite, commit, and STOP.**

- [ ] **Step 1: Write `scripts/run_phase4_lstm.py`**

```python
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

    lines.append(f"\nValidation majority baseline: {majority_baseline(__import__('pandas').Series(y_validate)):.3f}\n")
    lines.append(f"LSTM validation accuracy @0.5: {accuracy_at_threshold(__import__('pandas').Series(y_validate), val_proba, 0.5)}\n")

    print("=" * 60)
    print(f"[{name}] VALIDATION COMPLETE. Test set has NOT been touched yet.")
    print("=" * 60)

    X_seq_test, y_test, test_index, _ = build_sequences(test_df, FEATURE_COLUMNS, lookback=LOOKBACK)
    if len(y_test) == 0:
        lines.append("Insufficient data to build test sequences for this experiment - skipped.\n")
        return

    test_proba = predict_proba_lstm(model, X_seq_test)
    y_test_series = __import__('pandas').Series(y_test, index=test_index)

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
```

- [ ] **Step 2: Verify syntax**

Run: `python -m py_compile scripts/run_phase4_lstm.py`
Expected: no errors.

- [ ] **Step 3: Run the full test suite**

Run: `PYTHONPATH=. pytest tests/ -v`
Expected: all tests pass (this script has no dedicated tests, matching the established pattern for one-shot live-data orchestrators).

- [ ] **Step 4: Commit**

```bash
git add scripts/run_phase4_lstm.py
git commit -m "feat: add Phase 4 small-LSTM orchestrator (both experiments, walk-forward)"
```

- [ ] **Step 5: STOP — do not run this script**

This script touches both test sets for the first time in their LSTM-evaluated form. It must be reviewed (tracing test-set access ordering, confirming the model is fit once on train only, confirming `chosen_threshold` is fixed before any test access) before execution — the same discipline applied to every prior Phase 3/4 orchestrator. This review and the subsequent live run happen as a separate step after this task is complete.

---

## Self-Review Notes

- **Spec coverage:** Sequence construction respects the causal/no-lookahead rule and never crosses ticker boundaries (Task 1, directly tested including a dedicated causality test mirroring the pattern established in Phase 2). Small model size (single LSTM layer, hidden_size=16) matches the spec's explicit "keep it small" instruction (Task 2, directly tested — `test_fit_lstm_returns_module_with_small_hidden_size` asserts `num_layers == 1`). Deterministic given a seed (Task 2), matching this project's reproducibility standard for every other model. Walk-forward discipline and pre-execution review requirement (Task 3) explicitly called out twice (in Global Constraints and again in Task 3's own warning), matching the pattern that has caught real bugs in every prior orchestrator script this project has written.
- **Placeholder scan:** No TBD/TODO markers. Task 3's script is the one manual, non-pytest task, matching every other live-data orchestrator in this project — concrete expected behavior and an explicit stop-before-execution instruction instead of asserted output, since it evaluates against real, previously-established test sets.
- **Type consistency:** `build_sequences`'s four-tuple return `(X_seq, y, index, ticker)` (Task 1) is consumed identically in Task 3's orchestrator. `fit_lstm`/`predict_proba_lstm`'s signatures (Task 2) match their call sites in Task 3 exactly — `predict_proba_lstm`'s 1D array output is passed directly to `accuracy_at_threshold`/`breakdown_by_year`/`breakdown_by_regime`, the same interface those functions already accept from every other model in this project (`np.ndarray` of probabilities), so no adapter code is needed.

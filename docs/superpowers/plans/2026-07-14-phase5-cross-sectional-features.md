# Phase 5: Cross-Sectional Features Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Every feature tried so far (21 technical indicators, 5 sentiment features, across 4 label formulations) is either an absolute per-ticker transform of that ticker's own OHLCV history, or near-zero correlation with the label. A diagnostic showed all of it sits at the statistical noise floor (best correlation ≤0.036 across every label horizon tested). This phase adds a genuinely different information type — cross-sectional relative-strength features (how a ticker's momentum compares to the other 27 tickers on the same day, both universe-wide and sector-relative) — grounded in the most robust, well-replicated finding in equity return literature (cross-sectional momentum). It also tests a longer label horizon, since the same diagnostic showed longer horizons carry roughly double the correlation of the current tight 10-day window, consistent with where momentum effects are documented to be strongest (3-12 month formation, 1-3 month holding periods in the literature — this phase moves partway there, not all the way, to stay within a reasonable trading-relevant horizon).

**Architecture:** A new `src/features/cross_sectional.py` module that operates on the FULL multi-ticker feature matrix (not per-ticker, unlike every prior feature module) — for each date's cross-section of already-causal per-ticker return features, computes percentile rank and z-score, both universe-wide and sector-relative (using Phase 1's existing `SECTOR_MAP`). A script augments the existing feature matrix with these new columns plus an alternate longer-horizon label column, without touching the original matrix or its label. A new orchestrator script tests three configurations (current label + cross-sectional features; new label + original features; new label + cross-sectional features) via the same reviewed walk-forward discipline every prior orchestrator has used.

**Tech Stack:** Python, pandas, numpy, pytest. No new dependencies.

## Global Constraints

- **Hard rule, unchanged from every prior phase:** no feature for day t may use information from after day t's close. Cross-sectional features are a same-day AGGREGATION of already-causal per-ticker values (e.g., `return_20d`, itself already verified causal in Phase 2) — ranking/z-scoring across tickers on day t does not introduce new lookahead, since it never looks at day t+1 or later for ANY ticker. This must be directly tested (a causality test analogous to every prior feature module's, but checking that perturbing ticker B's FUTURE data doesn't change ticker A's PAST cross-sectional rank).
- Cross-sectional features (8 new columns): for each of `return_10d` and `return_20d` (the two return horizons closest to where the label-formulation diagnostic showed the most signal), compute: `xs_rank_{col}` (percentile rank 0-1 within that day's full 28-ticker universe), `xs_zscore_{col}` (z-score relative to that day's universe mean/std), `xs_sector_rank_{col}` (percentile rank within the ticker's own sector only, via `src.data.tickers.SECTOR_MAP`), `xs_sector_zscore_{col}` (z-score relative to the ticker's own sector's mean/std that day).
- Alternate longer-horizon label: horizon=20 trading days, up_threshold=0.04, down_threshold=0.04 (roughly double both the horizon and threshold of the current 10-day/2% label, consistent with the diagnostic's own finding that a 20-day/3-5% label showed roughly double the correlation of the current setting) — generated via the already-approved, unchanged `src.features.labels.generate_labels`, stored as a NEW column (`label_20d_4pct`) alongside the existing `label` column, not replacing it.
- The augmented feature matrix must preserve every existing column and row exactly (same 38,685-row set of already-causal, already-tested feature values) — this phase only ADDS columns, never modifies existing ones.
- Real feature matrix schema (verified, current state): `data/processed/feature_matrix.parquet`, `DatetimeIndex` named `date`, 21 technical + 5 sentiment + `label` + `ticker` = 28 columns, 38,685 rows, 28 tickers, zero NaN. `src.data.tickers.SECTOR_MAP` provides all 28 tickers' sector assignments across 6 sectors (Tech, Finance, Healthcare, Energy, Consumer, Industrials).
- Same walk-forward discipline as every prior test-set-touching script: the new orchestrator script must be reviewed BEFORE its first execution.

---

## File Structure

```
SentimentStockPredictorModel/
  src/
    features/
      cross_sectional.py          # new: universe/sector rank + z-score of per-ticker returns
  scripts/
    build_extended_feature_matrix.py   # augments feature_matrix.parquet with new columns + alt label
    run_phase5_cross_sectional_baselines.py  # orchestrator: 3 configs, walk-forward, reviewed before execution
  tests/
    features/
      test_cross_sectional.py
  docs/
    phase5_cross_sectional_results.md   # created at runtime by the orchestrator
```

---

### Task 1: Cross-sectional feature computation

**Files:**
- Create: `src/features/cross_sectional.py`
- Create: `tests/features/test_cross_sectional.py`

**Interfaces:**
- Consumes: `src.data.tickers.SECTOR_MAP` (real, already-committed data: `dict[str, str]` mapping all 28 tickers to one of 6 sector names).
- Produces: `cross_sectional.compute_cross_sectional_features(df: pd.DataFrame, base_columns: list, sector_map: dict) -> pd.DataFrame` — takes a multi-ticker DataFrame with a `DatetimeIndex`, a `ticker` column, and the columns named in `base_columns` already present (e.g. `["return_10d", "return_20d"]`), and returns a DataFrame (same index as input, one row per input row) with 4 new columns per base column: `xs_rank_{col}`, `xs_zscore_{col}`, `xs_sector_rank_{col}`, `xs_sector_zscore_{col}` — used by `scripts/build_extended_feature_matrix.py`.

- [ ] **Step 1: Write the failing tests**

Create `tests/features/test_cross_sectional.py`:

```python
import numpy as np
import pandas as pd
import pytest
from src.features.cross_sectional import compute_cross_sectional_features


def _panel(n_days=5, tickers=("A", "B", "C", "D"), sector_map=None):
    idx = pd.date_range("2024-01-02", periods=n_days, freq="B")
    rows = []
    for day_i, day in enumerate(idx):
        for ticker_i, ticker in enumerate(tickers):
            # deterministic, ticker-and-day-dependent value so ranks are unambiguous
            value = day_i * 10 + ticker_i
            rows.append({"date": day, "ticker": ticker, "return_10d": value})
    df = pd.DataFrame(rows).set_index("date")
    return df


def test_universe_rank_is_0_to_1_and_orders_correctly():
    df = _panel()
    result = compute_cross_sectional_features(df, base_columns=["return_10d"], sector_map={"A": "S1", "B": "S1", "C": "S2", "D": "S2"})
    # within each day, ticker D always has the highest return_10d value (ticker_i=3, highest offset)
    # so its universe rank should be the maximum (1.0) every day.
    d_ranks = result[df["ticker"] == "D"]["xs_rank_return_10d"]
    assert (d_ranks == 1.0).all()
    a_ranks = result[df["ticker"] == "A"]["xs_rank_return_10d"]
    assert (a_ranks == a_ranks.min()).all()
    assert a_ranks.iloc[0] < d_ranks.iloc[0]


def test_universe_zscore_has_zero_mean_within_each_day():
    df = _panel()
    result = compute_cross_sectional_features(df, base_columns=["return_10d"], sector_map={"A": "S1", "B": "S1", "C": "S2", "D": "S2"})
    for day in df.index.unique():
        day_zscores = result.loc[result.index == day, "xs_zscore_return_10d"]
        assert day_zscores.mean() == pytest.approx(0.0, abs=1e-9)


def test_sector_rank_only_compares_within_same_sector():
    df = _panel(tickers=("A", "B", "C", "D"))
    sector_map = {"A": "S1", "B": "S1", "C": "S2", "D": "S2"}
    result = compute_cross_sectional_features(df, base_columns=["return_10d"], sector_map=sector_map)
    # within S1 (A, B), B always has a higher return_10d value than A (ticker_i 1 > 0)
    # so B's sector rank should be 1.0 (max within S1), regardless of C/D's values.
    b_sector_ranks = result[df["ticker"] == "B"]["xs_sector_rank_return_10d"]
    assert (b_sector_ranks == 1.0).all()
    a_sector_ranks = result[df["ticker"] == "A"]["xs_sector_rank_return_10d"]
    assert (a_sector_ranks == 0.0).all()


def test_returns_correct_shape_and_column_names():
    df = _panel()
    result = compute_cross_sectional_features(
        df, base_columns=["return_10d"],
        sector_map={"A": "S1", "B": "S1", "C": "S2", "D": "S2"},
    )
    expected_cols = {"xs_rank_return_10d", "xs_zscore_return_10d", "xs_sector_rank_return_10d", "xs_sector_zscore_return_10d"}
    assert set(result.columns) == expected_cols
    assert len(result) == len(df)
    assert result.index.equals(df.index)


def test_cross_sectional_features_are_causal_no_lookahead():
    # Perturbing one ticker's value on a LATER day must not change another
    # ticker's cross-sectional rank/zscore on an EARLIER day.
    df = _panel(n_days=5)
    sector_map = {"A": "S1", "B": "S1", "C": "S2", "D": "S2"}
    result_before = compute_cross_sectional_features(df, base_columns=["return_10d"], sector_map=sector_map)

    df_modified = df.copy()
    last_day = df.index.unique()[-1]
    mask = (df_modified.index == last_day) & (df_modified["ticker"] == "D")
    df_modified.loc[mask, "return_10d"] = 99999.0

    result_after = compute_cross_sectional_features(df_modified, base_columns=["return_10d"], sector_map=sector_map)

    earlier_days_mask = df.index != last_day
    pd.testing.assert_frame_equal(
        result_before.loc[earlier_days_mask],
        result_after.loc[earlier_days_mask],
    )
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `PYTHONPATH=. pytest tests/features/test_cross_sectional.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.features.cross_sectional'`.

- [ ] **Step 3: Write `src/features/cross_sectional.py`**

```python
import pandas as pd


def compute_cross_sectional_features(df: pd.DataFrame, base_columns: list, sector_map: dict) -> pd.DataFrame:
    working = df.copy()
    working["_sector"] = working["ticker"].map(sector_map)

    result = pd.DataFrame(index=df.index)

    for col in base_columns:
        universe_grouped = working.groupby(working.index)[col]
        result[f"xs_rank_{col}"] = universe_grouped.rank(pct=True)
        universe_mean = universe_grouped.transform("mean")
        universe_std = universe_grouped.transform("std")
        result[f"xs_zscore_{col}"] = (working[col] - universe_mean) / universe_std

        sector_grouped = working.groupby([working.index, "_sector"])[col]
        result[f"xs_sector_rank_{col}"] = sector_grouped.rank(pct=True)
        sector_mean = sector_grouped.transform("mean")
        sector_std = sector_grouped.transform("std")
        result[f"xs_sector_zscore_{col}"] = (working[col] - sector_mean) / sector_std

    return result
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `PYTHONPATH=. pytest tests/features/test_cross_sectional.py -v`
Expected: PASS (5 passed).

- [ ] **Step 5: Commit**

```bash
git add src/features/cross_sectional.py tests/features/test_cross_sectional.py
git commit -m "feat: add cross-sectional rank/zscore features (universe and sector relative)"
```

---

### Task 2: Build the extended feature matrix

**Files:**
- Create: `scripts/build_extended_feature_matrix.py`

**Interfaces:**
- Consumes: `src.data.storage.load_parquet`/`save_parquet`, `src.data.tickers.SECTOR_MAP`, `src.features.cross_sectional.compute_cross_sectional_features`, `src.features.labels.generate_labels` (already approved, unchanged).
- Produces: `data/processed/feature_matrix_extended.parquet` — the existing `feature_matrix.parquet`'s 28 columns, PLUS 8 new cross-sectional columns, PLUS one new `label_20d_4pct` column (float, 1.0/0.0/NaN — NaN rows for tickers/dates where the longer horizon extends past available price history, matching `generate_labels`'s existing tail-drop behavior) = 37 columns. Does not modify or delete `feature_matrix.parquet`.

- [ ] **Step 1: Write `scripts/build_extended_feature_matrix.py`**

```python
import pandas as pd

from src import config
from src.data.storage import load_parquet, save_parquet
from src.data.tickers import SECTOR_MAP
from src.features.cross_sectional import compute_cross_sectional_features
from src.features.labels import generate_labels

RETURN_COLUMNS = ["return_10d", "return_20d"]


def main():
    matrix = load_parquet("feature_matrix", config.DATA_PROCESSED_DIR)
    panel = load_parquet("combined_price_panel", config.DATA_PROCESSED_DIR)

    xs_features = compute_cross_sectional_features(matrix, base_columns=RETURN_COLUMNS, sector_map=SECTOR_MAP)
    extended = matrix.join(xs_features, how="left")

    alt_labels = []
    for ticker, group in panel.groupby("ticker"):
        group = group.sort_index()
        labels = generate_labels(group["Close"], horizon=20, up_threshold=0.04, down_threshold=0.04)
        df = labels.rename("label_20d_4pct").to_frame()
        df["ticker"] = ticker
        alt_labels.append(df)
    alt_labels = pd.concat(alt_labels)

    extended_reset = extended.reset_index()
    alt_labels_reset = alt_labels.reset_index()
    merged = extended_reset.merge(
        alt_labels_reset, on=[extended_reset.columns[0], "ticker"], how="left"
    )
    merged = merged.set_index(extended_reset.columns[0])
    merged.index.name = matrix.index.name

    print(f"Extended matrix shape: {merged.shape} (original: {matrix.shape})")
    print(f"Rows with non-null label_20d_4pct: {merged['label_20d_4pct'].notna().sum()} / {len(merged)}")
    xs_cols = [c for c in merged.columns if c.startswith("xs_")]
    print(f"Cross-sectional columns: {xs_cols}")
    print(f"NaN count in cross-sectional columns: {merged[xs_cols].isna().sum().sum()}")

    save_parquet(merged, "feature_matrix_extended", config.DATA_PROCESSED_DIR)
    print(f"Saved to {config.DATA_PROCESSED_DIR / 'feature_matrix_extended.parquet'}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the live pipeline**

Run: `PYTHONPATH=. python scripts/build_extended_feature_matrix.py`

- [ ] **Step 3: Manually verify the deliverable**

Check:
1. `Extended matrix shape` has 38,685 rows (same as original — no rows added or dropped) and 37 columns (28 original + 8 cross-sectional + 1 new label).
2. `NaN count in cross-sectional columns` should be low/zero for the universe-level columns (every day has all 28 tickers present) but MAY be non-zero for sector-level columns if any sector has only 1 ticker on some day (std of a single value is NaN) — Industrials has only 3 tickers (fewest of any sector, per Phase 1's `SECTOR_MAP`), so check this isn't producing an unexpectedly large NaN count; a small number is expected and acceptable (these rows will simply be excluded when this specific column is used as a model feature, consistent with how NaN has been handled throughout this project — do NOT silently zero-fill without checking the count first).
3. `Rows with non-null label_20d_4pct` should be somewhat lower than the original 38,685 (the 20-day horizon drops more tail rows per ticker than the original 10-day one) but still the large majority.

- [ ] **Step 4: Commit the script (not the generated data file, which is gitignored)**

```bash
git add scripts/build_extended_feature_matrix.py
git commit -m "feat: add script to build extended feature matrix (cross-sectional features + 20d/4pct label)"
```

---

### Task 3: Cross-sectional baseline orchestrator (three configurations)

**Files:**
- Create: `scripts/run_phase5_cross_sectional_baselines.py`

**Interfaces:**
- Consumes: `src.data.storage.load_parquet`, `src.models.splits.time_split`, `src.models.train.fit_lightgbm`/`predict_proba` (LightGBM only for this phase — it was the most consistently well-behaved of the four classical algorithms across Phase 3/4, and this phase is testing FEATURE VALUE, not re-running the full algorithm sweep again), `src.models.evaluate.majority_baseline`/`accuracy_at_threshold` (using `local_majority_baseline` for all "beats baseline" comparisons, per the bug fixed in Phase 4), `src.models.ablation.TECHNICAL_COLUMNS`/`SENTIMENT_COLUMNS`.
- Produces: `docs/phase5_cross_sectional_results.md` reporting three configurations on the SAME full-history walk-forward split (2019-2022 train / 2023 validate / 2024-2025 test) used throughout this project: (1) original `label` + original features (the existing baseline, for direct comparison), (2) original `label` + original features + new cross-sectional features, (3) new `label_20d_4pct` + original features + new cross-sectional features.

**IMPORTANT — this script touches the test set for the first time with this new feature/label combination. It must be reviewed before its first execution, exactly like every other test-set-touching orchestrator in this project. Do not run it as part of implementing this task — implement, verify syntax, run the (unrelated) test suite, commit, and STOP.**

- [ ] **Step 1: Write `scripts/run_phase5_cross_sectional_baselines.py`**

```python
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
```

- [ ] **Step 2: Verify syntax**

Run: `python -m py_compile scripts/run_phase5_cross_sectional_baselines.py`
Expected: no errors.

- [ ] **Step 3: Run the full test suite**

Run: `PYTHONPATH=. pytest tests/ -v`
Expected: all tests pass (this script has no dedicated tests, matching the established pattern for one-shot live-data orchestrators).

- [ ] **Step 4: Commit**

```bash
git add scripts/run_phase5_cross_sectional_baselines.py
git commit -m "feat: add Phase 5 cross-sectional feature orchestrator (3 configs, walk-forward)"
```

- [ ] **Step 5: STOP — do not run this script**

This script touches the full-history test set for the first time with the new feature/label combinations. It must be reviewed (tracing test-set access ordering for EACH of the three configurations, confirming the model is fit once per configuration on train only, confirming `chosen_threshold` is fixed before any test access, confirming `local_majority_baseline` — not a stale full-set baseline — is used for every "beats baseline" claim) before execution. This review and the subsequent live run happen as a separate step after this task is complete.

---

## Self-Review Notes

- **Spec coverage:** Cross-sectional rank/z-score, universe and sector level, for the two return horizons the diagnostic flagged as most promising (Task 1, directly tested including a dedicated causality test following this project's established pattern). Extended-horizon label generated via already-approved code with no modification (Task 2). Three-way comparison isolating feature-value from label-horizon-value (Task 3's `CONFIGS` dict: config 1 is the pure existing baseline for reference, config 2 isolates "do cross-sectional features help at the SAME horizon," config 3 tests the combined best bet) — this design directly answers the causal question motivating this phase, not just "try everything at once and see."
- **Placeholder scan:** No TBD/TODO markers. Task 3's script is the one manual, non-pytest task, matching every other live-data orchestrator in this project — concrete expected behavior and an explicit stop-before-execution instruction instead of asserted output.
- **Type consistency:** `compute_cross_sectional_features`'s return columns (Task 1) are consumed by name (`XS_COLUMNS` list) in Task 3, matching exactly. `build_extended_feature_matrix.py`'s output schema (Task 2: 28 original + 8 cross-sectional + 1 new label = 37 columns) is what Task 3's `load_parquet("feature_matrix_extended", ...)` call expects to find. `_split_xy`'s three-tuple return (Task 3) is consumed consistently across all three configurations in `run_config`.

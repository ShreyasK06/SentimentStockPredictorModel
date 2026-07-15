# Phase 4: Feature Engineering & Additional Model Algorithms Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Neither baseline model (logistic regression, LightGBM) beat the majority-class baseline in Phase 3, on either the full 2019-2025 walk-forward split or the sentiment-covered recent-window split. This phase expands the technical feature set (9 new causal indicators using High/Low data not previously used) and adds two more model algorithms (Random Forest, XGBoost), then reruns both Phase 3 walk-forward experiments with the expanded feature set and all four algorithms, honestly reporting whether any combination beats baseline. If all four still plateau below the majority baseline on both experiments, a small LSTM is added as a final attempt, per the parent spec's explicit "if trees plateau, try LSTM — keep it small" fallback clause.

**Architecture:** Extend `src/features/technical.py` with 9 new indicator functions (Bollinger Bands, ATR, Stochastic Oscillator, normalized OBV, 52-week high/low distance, day-of-week cyclical encoding) — all causal by construction (backward-looking only). Extend `src/models/train.py` with Random Forest and XGBoost wrappers matching the existing `fit_*`/`predict_proba` interface. Update `src/models/ablation.py`'s `TECHNICAL_COLUMNS` to include the new features. Rebuild Phase 2's feature matrix (re-running the already-tested Phase 2 pipeline picks up the new technical features automatically — sentiment scores are cached and don't need re-scoring). Rerun both Phase 3 orchestrator scripts with all four algorithms.

**Tech Stack:** Python, pandas, numpy, scikit-learn (RandomForestClassifier), xgboost, lightgbm, pytest.

## Global Constraints

- **Hard rule, unchanged from Phase 2/3:** every new feature for day t must use only information available before market close on day t. All 9 new indicators are backward-looking rolling/cumulative computations ending at day t — verified by extending the existing causality-test pattern (Phase 2 Task 1's `test_technical_features_are_causal_no_lookahead`).
- New technical features (9, added to the existing 11 — 20 total):
  - `bollinger_pctb`: 20-day, 2-std Bollinger %B — `(close - lower_band) / (upper_band - lower_band)`.
  - `bollinger_bandwidth`: `(upper_band - lower_band) / middle_band`.
  - `atr_14_norm`: 14-day Average True Range, normalized by close (`ATR / close`) for cross-ticker comparability.
  - `stochastic_k`: 14-day %K — `(close - low_14) / (high_14 - low_14) * 100`.
  - `stochastic_d`: 3-day SMA of `stochastic_k`.
  - `obv_zscore`: On-Balance Volume, z-scored over a 20-day rolling window (raw OBV is an unbounded cumulative sum, not directly usable as a feature — normalize it).
  - `dist_from_52w_high`: `(close - rolling_max_252) / rolling_max_252`.
  - `dist_from_52w_low`: `(close - rolling_min_252) / rolling_min_252`.
  - `day_of_week_sin`, `day_of_week_cos`: cyclical encoding of the trading day's weekday (Monday=0..Friday=4, mapped to a 5-point cycle) — deterministic from the date, trivially causal.
- Warmup requirement grows: `dist_from_52w_high`/`dist_from_52w_low` need 252 rows of history (up from the existing 50-row `dist_from_ma50` requirement). This is fine for the full 2019-2025 feature matrix (still ~43,000+ rows after dropping warmup) and does NOT break the recent-window experiment, because `compute_technical_features` runs on each ticker's FULL price history before the recent-window script filters by date — by August 2025 there is ample prior history for a 252-day rolling window to be non-NaN.
- New model algorithms: `RandomForestClassifier` (scikit-learn, already a dependency) and `XGBClassifier` (`xgboost`, new dependency) — added alongside the existing `LogisticRegression`/`LGBMClassifier`, matching the existing `fit_*(X_train, y_train) -> model` / `predict_proba(model, X) -> np.ndarray` interface pattern from `src/models/train.py`. Tree-based models (RF, XGBoost) do not need the scaler.
- No leakage via scaling: unchanged from Phase 3 — any scaler stays fit-on-train-only (Random Forest/XGBoost/LightGBM don't need scaling; only if a new scaled model were added would this apply — none is in this plan).
- Both Phase 3 orchestrator scripts (`scripts/run_phase3_baselines.py`, `scripts/run_phase3_recent_window_baseline.py`) get rerun with all four algorithms — the test set in each is touched exactly once per script, per the existing walk-forward discipline already reviewed and approved in Phase 3. Re-running an already-approved script's logic with additional algorithms appended is not a new "first touch" of a previously-untouched test set (both test sets were already touched once each in Phase 3) — but the SAME discipline (banner before test access, fixed threshold, no retraining after validation) must still hold in the updated scripts, and the updated scripts must be reviewed before execution, exactly as Phase 3's Task 6 was.
- If all four algorithms plateau below the majority baseline on BOTH experiments after this phase's feature engineering, implement one small LSTM (per the parent spec's explicit "if and only if trees plateau... keep it small, financial data is low signal-to-noise and big models memorize noise" guidance) as a final attempt — scoped in Task 6, conditional on Task 5's results.

---

## File Structure

```
SentimentStockPredictorModel/
  requirements.txt                    # + xgboost
  src/
    features/
      technical.py                     # + 9 new indicator functions, extended compute_technical_features
    models/
      train.py                          # + fit_random_forest, fit_xgboost
      ablation.py                        # TECHNICAL_COLUMNS updated (20 columns)
  scripts/
    run_phase2_pipeline.py              # unchanged code, rerun to rebuild feature matrix with new features
    run_phase3_baselines.py              # updated: train/evaluate all 4 algorithms
    run_phase3_recent_window_baseline.py  # updated: train/evaluate all 4 algorithms
    run_phase4_lstm.py                    # created only if Task 5 shows all 4 algorithms still plateau
  tests/
    features/
      test_technical.py                 # + tests for 9 new indicators
    models/
      test_train.py                      # + tests for fit_random_forest, fit_xgboost
      test_ablation.py                    # updated column-count assertions
  docs/
    phase4_results.md                    # created at runtime: honest comparison of old vs. new results
```

---

### Task 1: Extended technical indicators

**Files:**
- Modify: `src/features/technical.py`
- Modify: `tests/features/test_technical.py`

**Interfaces:**
- Consumes: nothing new (pure pandas/numpy, now also uses `High`/`Low` columns from the price DataFrame, which were already present in Phase 1's schema but previously unused).
- Produces: 9 new functions matching the existing module's pattern — `compute_bollinger_bands(close, window=20, num_std=2) -> pd.DataFrame` (columns `bollinger_pctb`, `bollinger_bandwidth`), `compute_atr(high, low, close, window=14) -> pd.Series` (named `atr_14_norm`, already normalized by close), `compute_stochastic(high, low, close, k_window=14, d_window=3) -> pd.DataFrame` (columns `stochastic_k`, `stochastic_d`), `compute_obv_zscore(close, volume, window=20) -> pd.Series` (named `obv_zscore`), `compute_distance_from_52w_extremes(close, window=252) -> pd.DataFrame` (columns `dist_from_52w_high`, `dist_from_52w_low`), `compute_day_of_week_cyclical(index) -> pd.DataFrame` (columns `day_of_week_sin`, `day_of_week_cos`). `compute_technical_features(df)` is extended to call all of these and concat them alongside the existing 11 columns, for 20 total.

- [ ] **Step 1: Write the failing tests**

Append to `tests/features/test_technical.py` (the existing `_price_df` helper is reused; add a variant with High/Low):

```python
def _ohlcv_df(n=280, start=100.0, step=1.0):
    idx = pd.date_range("2024-01-02", periods=n, freq="B")
    close = pd.Series(start + step * np.arange(n), index=idx)
    high = close + 1.0
    low = close - 1.0
    volume = pd.Series(1_000_000 + 1000 * np.arange(n), index=idx)
    return pd.DataFrame({"Close": close, "High": high, "Low": low, "Volume": volume})


def test_compute_technical_features_returns_20_columns():
    df = _ohlcv_df()
    features = compute_technical_features(df)
    assert len(features.columns) == 20
    expected_new = [
        "bollinger_pctb", "bollinger_bandwidth", "atr_14_norm",
        "stochastic_k", "stochastic_d", "obv_zscore",
        "dist_from_52w_high", "dist_from_52w_low",
        "day_of_week_sin", "day_of_week_cos",
    ]
    for col in expected_new:
        assert col in features.columns


def test_bollinger_pctb_matches_manual_calculation():
    df = _ohlcv_df(n=40)
    features = compute_technical_features(df)
    close = df["Close"]
    ma20 = close.iloc[0:20].mean()
    std20 = close.iloc[0:20].std()
    upper = ma20 + 2 * std20
    lower = ma20 - 2 * std20
    expected_pctb = (close.iloc[19] - lower) / (upper - lower)
    assert features["bollinger_pctb"].iloc[19] == pytest.approx(expected_pctb)


def test_atr_is_normalized_by_close_and_nonnegative():
    df = _ohlcv_df(n=30)
    features = compute_technical_features(df)
    valid_atr = features["atr_14_norm"].dropna()
    assert len(valid_atr) > 0
    assert (valid_atr >= 0).all()


def test_stochastic_k_matches_manual_calculation():
    df = _ohlcv_df(n=30)
    features = compute_technical_features(df)
    high14 = df["High"].iloc[0:14].max()
    low14 = df["Low"].iloc[0:14].min()
    close14 = df["Close"].iloc[13]
    expected_k = (close14 - low14) / (high14 - low14) * 100
    assert features["stochastic_k"].iloc[13] == pytest.approx(expected_k)


def test_stochastic_k_bounded_0_to_100():
    df = _ohlcv_df(n=40)
    features = compute_technical_features(df)
    valid_k = features["stochastic_k"].dropna()
    assert (valid_k >= 0).all() and (valid_k <= 100).all()


def test_dist_from_52w_high_is_never_positive():
    # close can never exceed its own trailing rolling max, so this is always <= 0
    df = _ohlcv_df(n=260)
    features = compute_technical_features(df)
    valid = features["dist_from_52w_high"].dropna()
    assert len(valid) > 0
    assert (valid <= 1e-9).all()


def test_dist_from_52w_low_is_never_negative():
    df = _ohlcv_df(n=260)
    features = compute_technical_features(df)
    valid = features["dist_from_52w_low"].dropna()
    assert len(valid) > 0
    assert (valid >= -1e-9).all()


def test_day_of_week_cyclical_values_bounded():
    df = _ohlcv_df(n=30)
    features = compute_technical_features(df)
    assert (features["day_of_week_sin"].abs() <= 1.0 + 1e-9).all()
    assert (features["day_of_week_cos"].abs() <= 1.0 + 1e-9).all()


def test_day_of_week_cyclical_distinguishes_monday_and_friday():
    df = _ohlcv_df(n=10)
    features = compute_technical_features(df)
    weekdays = df.index.dayofweek
    monday_rows = features.index[weekdays == 0]
    friday_rows = features.index[weekdays == 4]
    assert len(monday_rows) > 0 and len(friday_rows) > 0
    mon_val = (features.loc[monday_rows[0], "day_of_week_sin"], features.loc[monday_rows[0], "day_of_week_cos"])
    fri_val = (features.loc[friday_rows[0], "day_of_week_sin"], features.loc[friday_rows[0], "day_of_week_cos"])
    assert mon_val != fri_val


def test_new_indicators_are_causal_no_lookahead():
    df = _ohlcv_df(n=280)
    features_before = compute_technical_features(df)
    df_modified = df.copy()
    df_modified.iloc[260:] = df_modified.iloc[260:] * 10
    features_after = compute_technical_features(df_modified)
    new_cols = [
        "bollinger_pctb", "bollinger_bandwidth", "atr_14_norm",
        "stochastic_k", "stochastic_d", "obv_zscore",
        "dist_from_52w_high", "dist_from_52w_low",
    ]
    pd.testing.assert_frame_equal(
        features_before.loc[features_before.index < df.index[260], new_cols],
        features_after.loc[features_after.index < df.index[260], new_cols],
    )
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `PYTHONPATH=. pytest tests/features/test_technical.py -v -k "20_columns or bollinger or atr or stochastic or dist_from_52w or day_of_week or new_indicators"`
Expected: FAIL (functions/columns don't exist yet).

- [ ] **Step 3: Write the new indicator functions in `src/features/technical.py`**

Append these functions before `compute_technical_features`:

```python
def compute_bollinger_bands(close: pd.Series, window: int = 20, num_std: int = 2) -> pd.DataFrame:
    ma = close.rolling(window=window, min_periods=window).mean()
    std = close.rolling(window=window, min_periods=window).std()
    upper = ma + num_std * std
    lower = ma - num_std * std
    pctb = (close - lower) / (upper - lower)
    bandwidth = (upper - lower) / ma
    return pd.DataFrame({"bollinger_pctb": pctb, "bollinger_bandwidth": bandwidth})


def compute_atr(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
    prev_close = close.shift(1)
    true_range = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    atr = true_range.rolling(window=window, min_periods=window).mean()
    return (atr / close).rename("atr_14_norm")


def compute_stochastic(high: pd.Series, low: pd.Series, close: pd.Series, k_window: int = 14, d_window: int = 3) -> pd.DataFrame:
    low_k = low.rolling(window=k_window, min_periods=k_window).min()
    high_k = high.rolling(window=k_window, min_periods=k_window).max()
    k = (close - low_k) / (high_k - low_k) * 100
    d = k.rolling(window=d_window, min_periods=d_window).mean()
    return pd.DataFrame({"stochastic_k": k, "stochastic_d": d})


def compute_obv_zscore(close: pd.Series, volume: pd.Series, window: int = 20) -> pd.Series:
    direction = close.diff().apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))
    obv = (direction * volume).cumsum()
    obv_mean = obv.rolling(window=window, min_periods=window).mean()
    obv_std = obv.rolling(window=window, min_periods=window).std()
    return ((obv - obv_mean) / obv_std).rename("obv_zscore")


def compute_distance_from_52w_extremes(close: pd.Series, window: int = 252) -> pd.DataFrame:
    rolling_max = close.rolling(window=window, min_periods=window).max()
    rolling_min = close.rolling(window=window, min_periods=window).min()
    dist_high = (close - rolling_max) / rolling_max
    dist_low = (close - rolling_min) / rolling_min
    return pd.DataFrame({"dist_from_52w_high": dist_high, "dist_from_52w_low": dist_low})


def compute_day_of_week_cyclical(index: pd.DatetimeIndex) -> pd.DataFrame:
    weekday = pd.Series(index.dayofweek, index=index)
    angle = 2 * 3.14159265358979 * weekday / 5
    return pd.DataFrame({
        "day_of_week_sin": angle.apply(lambda a: __import__("math").sin(a)),
        "day_of_week_cos": angle.apply(lambda a: __import__("math").cos(a)),
    }, index=index)
```

- [ ] **Step 4: Extend `compute_technical_features` to include the new indicators**

Replace the existing `compute_technical_features` function:

```python
def compute_technical_features(df: pd.DataFrame) -> pd.DataFrame:
    close = df["Close"]
    volume = df["Volume"]
    high = df["High"]
    low = df["Low"]
    return pd.concat([
        compute_returns(close),
        compute_rsi(close),
        compute_macd(close),
        compute_volatility(close),
        compute_volume_ratio(volume),
        compute_distance_from_ma(close),
        compute_bollinger_bands(close),
        compute_atr(high, low, close),
        compute_stochastic(high, low, close),
        compute_obv_zscore(close, volume),
        compute_distance_from_52w_extremes(close),
        compute_day_of_week_cyclical(df.index),
    ], axis=1)
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `PYTHONPATH=. pytest tests/features/test_technical.py -v`
Expected: PASS (all new tests plus all previously-existing tests in this file — the original 11-column tests must still pass since those columns/functions are unchanged, only more columns were added).

- [ ] **Step 6: Commit**

```bash
git add src/features/technical.py tests/features/test_technical.py
git commit -m "feat: add 9 causal technical indicators (Bollinger, ATR, Stochastic, OBV, 52w extremes, day-of-week)"
```

---

### Task 2: Random Forest and XGBoost training wrappers

**Files:**
- Modify: `requirements.txt` (add `xgboost`)
- Modify: `src/models/train.py`
- Modify: `tests/models/test_train.py`

**Interfaces:**
- Consumes: nothing new (matches the existing module's pattern).
- Produces: `train.fit_random_forest(X_train: pd.DataFrame, y_train: pd.Series) -> RandomForestClassifier`, `train.fit_xgboost(X_train: pd.DataFrame, y_train: pd.Series) -> XGBClassifier` — both consumed identically to `fit_lightgbm` (no scaler needed) via the existing `predict_proba(model, X) -> np.ndarray` function, which already works generically for any scikit-learn-compatible classifier.

- [ ] **Step 1: Add xgboost dependency**

Append to `requirements.txt`:
```
xgboost
```

Run: `pip install -r requirements.txt`
Expected: installs without error.

- [ ] **Step 2: Write the failing tests**

Append to `tests/models/test_train.py`:

```python
from src.models.train import fit_random_forest, fit_xgboost


def test_fit_random_forest_returns_fitted_model():
    X, y = _separable_data()
    model = fit_random_forest(X, y)
    assert hasattr(model, "predict_proba")


def test_random_forest_learns_separable_pattern():
    X, y = _separable_data()
    model = fit_random_forest(X, y)
    proba = predict_proba(model, X)
    predicted = (proba >= 0.5).astype(float)
    accuracy = (predicted == y.values).mean()
    assert accuracy > 0.85


def test_fit_xgboost_returns_fitted_model():
    X, y = _separable_data()
    model = fit_xgboost(X, y)
    assert hasattr(model, "predict_proba")


def test_xgboost_learns_separable_pattern():
    X, y = _separable_data()
    model = fit_xgboost(X, y)
    proba = predict_proba(model, X)
    predicted = (proba >= 0.5).astype(float)
    accuracy = (predicted == y.values).mean()
    assert accuracy > 0.85
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `PYTHONPATH=. pytest tests/models/test_train.py -v -k "random_forest or xgboost"`
Expected: FAIL with `ImportError: cannot import name 'fit_random_forest'`.

- [ ] **Step 4: Write the wrappers in `src/models/train.py`**

Add these imports at the top:

```python
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
```

Append these functions:

```python
def fit_random_forest(X_train: pd.DataFrame, y_train: pd.Series) -> RandomForestClassifier:
    model = RandomForestClassifier(n_estimators=200, max_depth=6, random_state=0)
    model.fit(X_train, y_train)
    return model


def fit_xgboost(X_train: pd.DataFrame, y_train: pd.Series) -> XGBClassifier:
    model = XGBClassifier(n_estimators=200, max_depth=4, eval_metric="logloss", use_label_encoder=False)
    model.fit(X_train, y_train)
    return model
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `PYTHONPATH=. pytest tests/models/test_train.py -v`
Expected: PASS (all new tests plus all previously-existing tests in this file).

- [ ] **Step 6: Commit**

```bash
git add requirements.txt src/models/train.py tests/models/test_train.py
git commit -m "feat: add Random Forest and XGBoost training wrappers"
```

---

### Task 3: Update ablation column list for the expanded feature set

**Files:**
- Modify: `src/models/ablation.py`
- Modify: `tests/models/test_ablation.py`

**Interfaces:**
- Consumes: nothing new.
- Produces: `TECHNICAL_COLUMNS` updated from 11 to 20 entries (the 9 new indicator names appended). `SENTIMENT_COLUMNS` and `run_ablation`'s logic are unchanged — this task only updates the column list constant.

- [ ] **Step 1: Update the failing test**

In `tests/models/test_ablation.py`, the existing `test_technical_and_sentiment_column_lists_are_disjoint_and_nonempty` test already asserts disjointness generically — it does not need to change. Add one new test:

```python
def test_technical_columns_includes_new_indicators():
    new_indicators = [
        "bollinger_pctb", "bollinger_bandwidth", "atr_14_norm",
        "stochastic_k", "stochastic_d", "obv_zscore",
        "dist_from_52w_high", "dist_from_52w_low",
        "day_of_week_sin", "day_of_week_cos",
    ]
    for col in new_indicators:
        assert col in TECHNICAL_COLUMNS
    assert len(TECHNICAL_COLUMNS) == 20
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=. pytest tests/models/test_ablation.py -v -k "new_indicators"`
Expected: FAIL (`TECHNICAL_COLUMNS` still has 11 entries).

- [ ] **Step 3: Update `TECHNICAL_COLUMNS` in `src/models/ablation.py`**

```python
TECHNICAL_COLUMNS = [
    "return_1d", "return_5d", "return_10d", "return_20d",
    "rsi_14", "macd", "macd_signal", "macd_hist",
    "volatility_20d", "volume_ratio_20d", "dist_from_ma50",
    "bollinger_pctb", "bollinger_bandwidth", "atr_14_norm",
    "stochastic_k", "stochastic_d", "obv_zscore",
    "dist_from_52w_high", "dist_from_52w_low",
    "day_of_week_sin", "day_of_week_cos",
]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `PYTHONPATH=. pytest tests/models/test_ablation.py -v`
Expected: PASS (all tests in this file, including the pre-existing ones — `run_ablation`'s own logic doesn't change, only the column list it selects).

- [ ] **Step 5: Run the full test suite**

Run: `PYTHONPATH=. pytest tests/ -v`
Expected: all tests pass (Phase 1-3's existing tests plus Phase 4 Tasks 1-3).

- [ ] **Step 6: Commit**

```bash
git add src/models/ablation.py tests/models/test_ablation.py
git commit -m "feat: expand TECHNICAL_COLUMNS to include new indicators (11 -> 20)"
```

---

### Task 4: Rebuild the feature matrix with the expanded technical feature set

**Files:**
- No file changes — this task re-runs the already-tested, already-approved `scripts/run_phase2_pipeline.py` unchanged. Its logic automatically picks up Task 1's extended `compute_technical_features`; Task 3's updated `TECHNICAL_COLUMNS` is not directly used by this script (the script builds ALL technical columns via `compute_technical_features` regardless of `TECHNICAL_COLUMNS`'s contents — that constant is only consumed by `ablation.py` and downstream orchestrator scripts).

**Interfaces:**
- Consumes: `src.features.build_features.build_feature_matrix` (unchanged), which now internally calls Task 1's extended `compute_technical_features`.
- Produces: an updated `data/processed/feature_matrix.parquet` with 20 technical + 5 sentiment + label + ticker = 27 columns (up from 18), same row-count logic (warmup-drop now uses the largest window among all technical indicators — 252 rows for the 52-week features, up from 50 — so slightly fewer warmup rows survive per ticker, still tens of thousands of total rows).

- [ ] **Step 1: Run the live pipeline**

Run: `PYTHONPATH=. python scripts/run_phase2_pipeline.py`

This reuses the cached sentiment scores in `data/raw/sentiment/` (already scored in Phase 2, no re-scoring needed) and recomputes the full feature matrix with the new technical indicators.

Expected console output pattern: same as the original Phase 2 live run (per-ticker sentiment loading, feature matrix row count, label balance) — the row count will be somewhat lower than the original 43,505 (larger warmup window drops more early rows), but should still be tens of thousands.

- [ ] **Step 2: Manually verify the deliverable**

Check:
1. No `WARNING: failed to process sentiment for...` lines (would indicate a regression in cached-sentiment loading).
2. `Tickers represented` still includes all 28 tickers.
3. Load the new `feature_matrix.parquet` and confirm it has 27 columns (20 technical + 5 sentiment + label + ticker) and zero NaN cells anywhere — the same invariant Phase 2 established, now re-verified for the expanded feature set.

- [ ] **Step 3: No commit needed for this task**

`data/processed/feature_matrix.parquet` is gitignored (per Phase 1's `.gitignore`) — it's a generated artifact, not tracked in version control. Nothing to commit here; proceed directly to Task 5.

---

### Task 5: Rerun both Phase 3 baselines with the expanded feature set and all four algorithms

**Files:**
- Modify: `scripts/run_phase3_baselines.py`
- Modify: `scripts/run_phase3_recent_window_baseline.py`

**Interfaces:**
- Consumes: `src.models.train.fit_random_forest`/`fit_xgboost` (Task 2), the updated `src.models.ablation.TECHNICAL_COLUMNS` (Task 3, automatically picked up via the existing `from src.models.ablation import ... TECHNICAL_COLUMNS` import both scripts already have).
- Produces: updated `docs/phase3_results.md` and `docs/phase3_recent_window_results.md`, now reporting all four algorithms (logistic regression, LightGBM, Random Forest, XGBoost) instead of two.

**IMPORTANT — these scripts touch their respective test sets. Even though each test set was already touched once in the original Phase 3 run, the MODIFIED scripts must be reviewed before re-execution, exactly as Phase 3's Task 6 was — a code change here could reintroduce the exact "early test access" bug that was caught and fixed in Phase 3. Do not run these scripts without review after modifying them.**

- [ ] **Step 1: Modify `scripts/run_phase3_baselines.py`**

Add the import:
```python
from src.models.train import fit_logistic_regression, fit_lightgbm, fit_random_forest, fit_xgboost, predict_proba
```

After the existing `lr_model, scaler = fit_logistic_regression(X_train, y_train)` and `lgbm_model = fit_lightgbm(X_train, y_train)` lines, add:
```python
    rf_model = fit_random_forest(X_train, y_train)
    xgb_model = fit_xgboost(X_train, y_train)
```

After the existing `lr_val_proba`/`lgbm_val_proba` lines, add:
```python
    rf_val_proba = predict_proba(rf_model, X_validate)
    xgb_val_proba = predict_proba(xgb_model, X_validate)
```

In the validation results section, add two more report lines analogous to the existing LightGBM one:
```python
    lines.append(f"Random Forest accuracy @0.5: {accuracy_at_threshold(y_validate, rf_val_proba, 0.5)}\n")
    lines.append(f"XGBoost accuracy @0.5: {accuracy_at_threshold(y_validate, xgb_val_proba, 0.5)}\n")
```

In the confidence-threshold sweep section, add:
```python
    lines.append("\nRandom Forest:\n")
    lines.append(sweep_confidence_threshold(y_validate, rf_val_proba).to_string())
    lines.append("\nXGBoost:\n")
    lines.append(sweep_confidence_threshold(y_validate, xgb_val_proba).to_string())
```

After `X_test, y_test = _split_xy(test_df)` and the existing `lr_test_proba`/`lgbm_test_proba` lines (this must remain AFTER the console banner and AFTER `chosen_threshold` is set — do not move it earlier), add:
```python
    rf_test_proba = predict_proba(rf_model, X_test)
    xgb_test_proba = predict_proba(xgb_model, X_test)
```

In the test results section, add:
```python
    lines.append(f"Random Forest: {accuracy_at_threshold(y_test, rf_test_proba, chosen_threshold)}\n")
    lines.append(f"XGBoost: {accuracy_at_threshold(y_test, xgb_test_proba, chosen_threshold)}\n")
```

- [ ] **Step 2: Apply the equivalent changes to `scripts/run_phase3_recent_window_baseline.py`**

Mirror the same pattern: import `fit_random_forest`/`fit_xgboost`, fit both models once on `X_train` (alongside the existing `lr_model`/`lgbm_model` fits, in the same block, before the console banner), compute validation probabilities and report them, compute test probabilities ONLY after the existing banner/`chosen_threshold` lines (in the same position where `lr_test_proba`/`lgbm_test_proba` already are), and report test results for both new models. Also extend the "Ablation on test set" section's spirit by leaving `run_ablation` as-is (it already uses LightGBM internally per its Task 5/Phase 3 design — this task does not change `run_ablation`'s algorithm, only adds RF/XGBoost as additional top-level model comparisons alongside it).

- [ ] **Step 3: Verify both scripts are syntactically valid**

Run: `python -m py_compile scripts/run_phase3_baselines.py scripts/run_phase3_recent_window_baseline.py`
Expected: no errors.

- [ ] **Step 4: Run the full test suite**

Run: `PYTHONPATH=. pytest tests/ -v`
Expected: all tests pass (these scripts have no dedicated tests, so this just confirms nothing else broke).

- [ ] **Step 5: Commit the script changes (NOT the results yet — review must happen first)**

```bash
git add scripts/run_phase3_baselines.py scripts/run_phase3_recent_window_baseline.py
git commit -m "feat: add Random Forest and XGBoost to both Phase 3 walk-forward baselines"
```

- [ ] **Step 6: STOP — do not run either script yet**

Both scripts touch their respective test sets. Per this task's opening note, they must be reviewed (tracing test-set access ordering, scaler/model-fit lineage, threshold discipline — the same checklist used for the original Phase 3 Task 6 review) before execution. This review and the subsequent live runs happen as a separate step after this plan's tasks are implemented and reviewed — do not execute `scripts/run_phase3_baselines.py` or `scripts/run_phase3_recent_window_baseline.py` as part of implementing this task.

---

### Task 6 (conditional): Small LSTM, only if Task 5 shows all four algorithms still plateau

**This task is CONDITIONAL.** Only implement it if, after Task 5's scripts are reviewed and run (as a separate step, not part of this task's implementation), all four algorithms (logistic regression, LightGBM, Random Forest, XGBoost) fail to beat the majority-class baseline on BOTH the full-history test set and the recent-window test set. If any algorithm beats baseline on either experiment, skip this task and proceed directly to the final honest write-up.

**Files (only if triggered):**
- Create: `src/models/sequences.py` (sliding-window sequence construction for LSTM input)
- Create: `src/models/train_lstm.py` (small LSTM wrapper, PyTorch — already a dependency from Phase 2's FinBERT scoring)
- Create: `scripts/run_phase4_lstm.py`
- Create: `tests/models/test_sequences.py`, `tests/models/test_train_lstm.py`

**Interfaces (only if triggered):**
- `sequences.build_sequences(X: pd.DataFrame, y: pd.Series, ticker: pd.Series, lookback: int = 10) -> tuple[np.ndarray, np.ndarray]` — builds per-ticker sliding windows of the last `lookback` days' features to predict the current day's label, never crossing ticker boundaries (a sequence must not mix rows from two different tickers) and never using data from before a ticker's own history begins.
- `train_lstm.fit_small_lstm(X_seq: np.ndarray, y: np.ndarray, hidden_size: int = 16, epochs: int = 20) -> nn.Module` — deliberately small (single LSTM layer, small hidden size) per the spec's explicit "keep it small" instruction, to avoid memorizing noise on this low-signal-to-noise dataset.

Because this task is conditional and its exact shape depends on Task 5's results (which inform what "plateau" means numerically and which test set(s) to target), do not write further implementation detail here — if triggered, treat this task as requiring its own brief follow-up plan (or an inline design discussion) before implementation, rather than following a pre-written spec that can't yet account for Task 5's actual findings.

---

## Self-Review Notes

- **Spec coverage:** 9 new causal technical indicators (Task 1) matching the parent spec's general "expand technical features" intent and this plan's own stated list. Two new algorithms (Task 2), matching the parent spec's implicit permission to try alternatives beyond the original two baselines. Ablation column list kept in sync (Task 3) so the existing, already-reviewed `run_ablation` machinery works unchanged with the larger feature set. Feature matrix rebuild (Task 4) reuses already-tested Phase 2 code with no new code risk. Both Phase 3 orchestrators updated (Task 5) with an explicit, repeated warning not to skip the pre-execution review this project has consistently required for any test-set-touching script. Conditional LSTM (Task 6) matches the parent spec's explicit fallback clause and its explicit sizing guidance ("keep it small").
- **Placeholder scan:** No TBD/TODO markers in Tasks 1-5. Task 6 is intentionally left under-specified because its design depends on Task 5's not-yet-known results — this is flagged explicitly as a scope decision, not an oversight, and instructs the implementer to pause and get a fresh design pass rather than guessing.
- **Type consistency:** `compute_bollinger_bands`/`compute_atr`/`compute_stochastic`/`compute_obv_zscore`/`compute_distance_from_52w_extremes`/`compute_day_of_week_cyclical` (Task 1) are all called with matching signatures inside the extended `compute_technical_features`. `fit_random_forest`/`fit_xgboost` (Task 2) return objects with `.predict_proba()`, matching the existing generic `predict_proba(model, X, scaler=None)` helper's assumption (already used identically for `fit_lightgbm`'s output) — no new wrapper logic needed for these two functions beyond the fit call itself. `TECHNICAL_COLUMNS` (Task 3) is consumed identically by `run_ablation` (unchanged) and by both orchestrator scripts' `FEATURE_COLUMNS = TECHNICAL_COLUMNS + SENTIMENT_COLUMNS` (unchanged pattern, just a longer list).

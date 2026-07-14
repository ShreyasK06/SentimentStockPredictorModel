# Phase 2: Features & Labels Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn Phase 1's raw price panel and news archive into a single, leak-free per-(ticker, day) feature matrix: ~11 technical indicators, FinBERT-scored daily sentiment (with 3-day/7-day rolling averages), and ±2%/10-day triple-barrier labels — ready for Phase 3's baseline models.

**Architecture:** Five independent, pure-logic modules under `src/features/` (technical indicators, point-in-time trading-day assignment, FinBERT scoring, sentiment aggregation, label generation), each unit-tested without needing the real FinBERT model or live data. One orchestrator (`build_features.py`) composes them per ticker and concatenates. A manual live script scores real headlines (needs `transformers`/`torch`, downloads the FinBERT model) and builds the final matrix.

**Tech Stack:** Python, pandas, numpy, transformers (ProsusAI/finbert), torch, pytest.

## Global Constraints

- **Hard rule, no exceptions:** every feature for day t must use only information available before market close on day t. Technical indicators are naturally causal (pandas `.rolling`/`.pct_change`/`.ewm` only look backward) — verified by an explicit causality test. Sentiment is the risk: a headline is only usable for day t's features if it was published before that day's 16:00 US/Eastern close (using `published_at`, the UTC-aware exact timestamp Phase 1 already preserves) — headlines published after close, or on a non-trading day, roll forward to the next trading day.
- Technical features (~10-15): returns over 1/5/10/20 days, RSI(14), MACD(12,26,9), 20-day rolling volatility, 20-day volume ratio, distance from 50-day MA. (11 total.)
- Sentiment features: FinBERT mean sentiment score, sentiment std, headline count per (ticker, effective trading day), plus 3-day and 7-day rolling sentiment averages.
- Labels: for each (ticker, day), label 1 if close reaches entry × 1.02 before entry × 0.98 within the next 10 trading days; label 0 if the reverse; drop the row (NaN) if neither threshold is hit within the horizon, or if fewer than 10 future trading days remain (tail rows).
- Phase 1's real output schema (verified against the actual live-pulled data, not assumed):
  - Price panel (`data/processed/combined_price_panel.parquet`): `DatetimeIndex` named `date` (tz-naive), columns `Open, High, Low, Close (float64), Volume (int64), ticker (object)`. Zero gaps guaranteed by Phase 1.
  - Per-ticker news (`data/raw/news/<TICKER>.parquet`): columns `date` (tz-aware `America/New_York`, midnight-bucketed — descriptive only, NOT used for point-in-time logic), `ticker`, `headline`, `source`, `url`, `published_at` (tz-aware UTC, exact publish moment — this is what Phase 2 uses for the point-in-time cutoff).
- No new SQLite. Reuse Phase 1's `storage.save_parquet`/`load_parquet` for any new cached artifacts (sentiment scoring cache), keyed the same way (per ticker, one parquet each).

---

## File Structure

```
SentimentStockPredictorModel/
  requirements.txt                # + transformers, torch
  src/
    config.py                     # + SENTIMENT_DIR
    features/
      __init__.py
      technical.py                  # returns, RSI, MACD, volatility, volume ratio, dist-from-MA
      trading_day.py                 # point-in-time effective-trading-day assignment
      sentiment.py                    # FinBERT scoring + daily aggregation + rolling windows
      labels.py                        # triple-barrier-style ±2%/10-day label generation
      build_features.py                 # orchestrator: per-ticker + full feature matrix
  scripts/
    run_phase2_pipeline.py          # manual live-run entry point (needs real FinBERT model)
  tests/
    features/
      __init__.py
      test_technical.py
      test_trading_day.py
      test_sentiment.py
      test_labels.py
      test_build_features.py
  data/
    raw/
      sentiment/                   # created at runtime: cached per-ticker scored headlines
    processed/
      feature_matrix.parquet        # created at runtime: final Phase 2 deliverable
```

---

### Task 1: Technical indicators module

**Files:**
- Create: `src/features/__init__.py`
- Create: `src/features/technical.py`
- Create: `tests/features/__init__.py`
- Create: `tests/features/test_technical.py`

**Interfaces:**
- Consumes: nothing (pure pandas/numpy on a single-ticker price DataFrame with `Close`/`Volume` columns, matching Phase 1's `combined_price_panel.parquet` schema).
- Produces: `technical.compute_technical_features(df: pd.DataFrame) -> pd.DataFrame` with columns `["return_1d", "return_5d", "return_10d", "return_20d", "rsi_14", "macd", "macd_signal", "macd_hist", "volatility_20d", "volume_ratio_20d", "dist_from_ma50"]`, same index as `df` — used by `build_features.build_ticker_feature_matrix`.

- [ ] **Step 1: Write the failing tests**

Create `tests/features/__init__.py` (empty) and `tests/features/test_technical.py`:

```python
import numpy as np
import pandas as pd
from src.features.technical import compute_technical_features


def _price_df(n=80, start=100.0, step=1.0):
    idx = pd.date_range("2024-01-02", periods=n, freq="B")
    close = pd.Series(start + step * np.arange(n), index=idx)
    volume = pd.Series(1_000_000 + 1000 * np.arange(n), index=idx)
    return pd.DataFrame({"Close": close, "Volume": volume})


def test_returns_expected_columns():
    df = _price_df()
    features = compute_technical_features(df)
    assert list(features.columns) == [
        "return_1d", "return_5d", "return_10d", "return_20d",
        "rsi_14", "macd", "macd_signal", "macd_hist",
        "volatility_20d", "volume_ratio_20d", "dist_from_ma50",
    ]
    assert features.index.equals(df.index)


def test_rsi_is_100_for_strictly_increasing_series():
    df = _price_df(n=30, start=100.0, step=1.0)
    features = compute_technical_features(df)
    assert features["rsi_14"].iloc[-1] == 100.0


def test_rsi_is_0_for_strictly_decreasing_series():
    df = _price_df(n=30, start=200.0, step=-1.0)
    features = compute_technical_features(df)
    assert features["rsi_14"].iloc[-1] == 0.0


def test_returns_match_pct_change():
    df = _price_df()
    features = compute_technical_features(df)
    expected_5d = df["Close"].pct_change(5)
    pd.testing.assert_series_equal(features["return_5d"], expected_5d, check_names=False)


def test_macd_hist_equals_macd_minus_signal():
    df = _price_df()
    features = compute_technical_features(df)
    diff = features["macd"] - features["macd_signal"]
    pd.testing.assert_series_equal(features["macd_hist"], diff, check_names=False)


def test_technical_features_are_causal_no_lookahead():
    # The single most important property in this project: a feature at day t
    # must not change if future data (day t+1 onward) is altered.
    df = _price_df(n=80)
    features_before = compute_technical_features(df)
    df_modified = df.copy()
    df_modified.iloc[40:] = df_modified.iloc[40:] * 10
    features_after = compute_technical_features(df_modified)
    pd.testing.assert_frame_equal(features_before.iloc[:40], features_after.iloc[:40])
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `PYTHONPATH=. pytest tests/features/test_technical.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.features'`.

- [ ] **Step 3: Create `src/features/__init__.py`** (empty file)

- [ ] **Step 4: Write `src/features/technical.py`**

```python
import pandas as pd


def compute_returns(close: pd.Series, windows=(1, 5, 10, 20)) -> pd.DataFrame:
    return pd.DataFrame({f"return_{w}d": close.pct_change(w) for w in windows})


def compute_rsi(close: pd.Series, window: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=window, min_periods=window).mean()
    avg_loss = loss.rolling(window=window, min_periods=window).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi.rename("rsi_14")


def compute_macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    hist = macd_line - signal_line
    return pd.DataFrame({"macd": macd_line, "macd_signal": signal_line, "macd_hist": hist})


def compute_volatility(close: pd.Series, window: int = 20) -> pd.Series:
    daily_returns = close.pct_change()
    return daily_returns.rolling(window=window, min_periods=window).std().rename("volatility_20d")


def compute_volume_ratio(volume: pd.Series, window: int = 20) -> pd.Series:
    avg_volume = volume.rolling(window=window, min_periods=window).mean()
    return (volume / avg_volume).rename("volume_ratio_20d")


def compute_distance_from_ma(close: pd.Series, window: int = 50) -> pd.Series:
    ma = close.rolling(window=window, min_periods=window).mean()
    return ((close - ma) / ma).rename("dist_from_ma50")


def compute_technical_features(df: pd.DataFrame) -> pd.DataFrame:
    close = df["Close"]
    volume = df["Volume"]
    return pd.concat([
        compute_returns(close),
        compute_rsi(close),
        compute_macd(close),
        compute_volatility(close),
        compute_volume_ratio(volume),
        compute_distance_from_ma(close),
    ], axis=1)
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `PYTHONPATH=. pytest tests/features/test_technical.py -v`
Expected: PASS (6 passed). Note: RSI/volatility/volume-ratio/dist-from-MA will be `NaN` for the first `window - 1` rows in every test — that's expected (insufficient warmup history), not a bug; the causality and column-shape tests are unaffected by this.

- [ ] **Step 6: Commit**

```bash
git add src/features/__init__.py src/features/technical.py tests/features/__init__.py tests/features/test_technical.py
git commit -m "feat: add technical indicators module (returns, RSI, MACD, volatility, volume ratio, dist-from-MA)"
```

---

### Task 2: Point-in-time effective-trading-day assignment

**Files:**
- Create: `src/features/trading_day.py`
- Create: `tests/features/test_trading_day.py`

**Interfaces:**
- Consumes: nothing (pure pandas datetime logic).
- Produces: `trading_day.assign_effective_trading_day(published_at: pd.Series, trading_days: pd.DatetimeIndex) -> pd.Series` (tz-naive `Timestamp` per row, named `"effective_trading_day"`) — used by `build_features.build_ticker_feature_matrix` to enforce the point-in-time hard rule on news.

- [ ] **Step 1: Write the failing tests**

Create `tests/features/test_trading_day.py`:

```python
import pandas as pd
from src.features.trading_day import assign_effective_trading_day


def test_before_close_on_trading_day_stays_same_day():
    trading_days = pd.DatetimeIndex(["2024-01-16", "2024-01-17", "2024-01-18"])
    # 2024-01-17 15:59 US/Eastern (winter, EST = UTC-5) = 20:59 UTC
    published = pd.Series([pd.Timestamp("2024-01-17 20:59:00", tz="UTC")])
    result = assign_effective_trading_day(published, trading_days)
    assert result.iloc[0] == pd.Timestamp("2024-01-17")


def test_after_close_on_trading_day_rolls_to_next_trading_day():
    trading_days = pd.DatetimeIndex(["2024-01-16", "2024-01-17", "2024-01-18"])
    # 2024-01-17 16:01 US/Eastern = 21:01 UTC
    published = pd.Series([pd.Timestamp("2024-01-17 21:01:00", tz="UTC")])
    result = assign_effective_trading_day(published, trading_days)
    assert result.iloc[0] == pd.Timestamp("2024-01-18")


def test_published_on_non_trading_day_rolls_to_next_trading_day():
    trading_days = pd.DatetimeIndex(["2024-01-16", "2024-01-17", "2024-01-22"])  # 18-21 = weekend/holiday gap
    # Saturday 2024-01-20, any time of day
    published = pd.Series([pd.Timestamp("2024-01-20 15:00:00", tz="UTC")])
    result = assign_effective_trading_day(published, trading_days)
    assert result.iloc[0] == pd.Timestamp("2024-01-22")


def test_handles_multiple_rows_independently():
    trading_days = pd.DatetimeIndex(["2024-01-16", "2024-01-17", "2024-01-18"])
    published = pd.Series([
        pd.Timestamp("2024-01-17 20:59:00", tz="UTC"),  # before close -> same day
        pd.Timestamp("2024-01-17 21:01:00", tz="UTC"),  # after close -> next day
    ])
    result = assign_effective_trading_day(published, trading_days)
    assert result.iloc[0] == pd.Timestamp("2024-01-17")
    assert result.iloc[1] == pd.Timestamp("2024-01-18")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `PYTHONPATH=. pytest tests/features/test_trading_day.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.features.trading_day'`.

- [ ] **Step 3: Write `src/features/trading_day.py`**

```python
import numpy as np
import pandas as pd

MARKET_CLOSE_ET = pd.Timestamp("16:00").time()


def assign_effective_trading_day(published_at: pd.Series, trading_days: pd.DatetimeIndex) -> pd.Series:
    published_et = pd.to_datetime(published_at).dt.tz_convert("America/New_York")
    calendar_day_naive = published_et.dt.tz_localize(None).dt.normalize()
    before_close = published_et.dt.time < MARKET_CLOSE_ET

    sorted_days = pd.DatetimeIndex(
        sorted(pd.DatetimeIndex(trading_days).tz_localize(None).normalize().unique())
    )
    is_trading_day = calendar_day_naive.isin(sorted_days)
    same_day_eligible = before_close & is_trading_day

    positions_next = sorted_days.searchsorted(calendar_day_naive.values, side="right")
    positions_next = np.clip(positions_next, 0, len(sorted_days) - 1)
    next_trading_day = pd.Series(sorted_days[positions_next], index=calendar_day_naive.index)

    effective = calendar_day_naive.where(same_day_eligible, next_trading_day)
    return effective.rename("effective_trading_day")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `PYTHONPATH=. pytest tests/features/test_trading_day.py -v`
Expected: PASS (4 passed).

- [ ] **Step 5: Commit**

```bash
git add src/features/trading_day.py tests/features/test_trading_day.py
git commit -m "feat: add point-in-time effective-trading-day assignment for news"
```

---

### Task 3: FinBERT sentiment scoring module

**Files:**
- Modify: `requirements.txt` (add `transformers`, `torch`)
- Modify: `src/config.py` (add `SENTIMENT_DIR`)
- Create: `src/features/sentiment.py` (scoring part only this task — aggregation is Task 4)
- Create: `tests/features/test_sentiment.py` (scoring tests only this task)

**Interfaces:**
- Consumes: nothing new for scoring (wraps `transformers.pipeline`).
- Produces: `sentiment.score_headlines(headlines: list) -> pd.DataFrame` with columns `["headline", "positive", "negative", "neutral", "sentiment_score"]` — used by `build_features.build_ticker_feature_matrix` (via the live pipeline script, which scores real headlines).

- [ ] **Step 1: Add dependencies**

Append to `requirements.txt`:
```
transformers
torch
```

Run: `pip install -r requirements.txt`
Expected: installs without error (this is a larger download — `torch` in particular; allow a few minutes).

- [ ] **Step 2: Add `SENTIMENT_DIR` to config**

In `src/config.py`, add after the existing `NEWS_DIR = DATA_RAW_DIR / "news"` line:

```python
SENTIMENT_DIR = DATA_RAW_DIR / "sentiment"
```

- [ ] **Step 3: Write the failing tests**

Create `tests/features/test_sentiment.py`:

```python
import pytest
from unittest.mock import patch, MagicMock
from src.features.sentiment import score_headlines


@patch("src.features.sentiment._get_classifier")
def test_score_headlines_computes_sentiment_score(mock_get_classifier):
    mock_classifier = MagicMock()
    mock_classifier.return_value = [
        [{"label": "positive", "score": 0.7}, {"label": "negative", "score": 0.1}, {"label": "neutral", "score": 0.2}],
        [{"label": "positive", "score": 0.05}, {"label": "negative", "score": 0.85}, {"label": "neutral", "score": 0.1}],
    ]
    mock_get_classifier.return_value = mock_classifier

    df = score_headlines(["Great earnings beat", "Massive layoffs announced"])

    assert list(df.columns) == ["headline", "positive", "negative", "neutral", "sentiment_score"]
    assert df.iloc[0]["sentiment_score"] == pytest.approx(0.6)
    assert df.iloc[1]["sentiment_score"] == pytest.approx(-0.8)
    mock_classifier.assert_called_once_with(
        ["Great earnings beat", "Massive layoffs announced"], batch_size=16, truncation=True
    )


def test_score_headlines_empty_list():
    df = score_headlines([])
    assert len(df) == 0
    assert list(df.columns) == ["headline", "positive", "negative", "neutral", "sentiment_score"]
```

- [ ] **Step 4: Run tests to verify they fail**

Run: `PYTHONPATH=. pytest tests/features/test_sentiment.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.features.sentiment'`.

- [ ] **Step 5: Write `src/features/sentiment.py`** (scoring portion — Task 4 appends the aggregation functions to this same file)

```python
import pandas as pd
from transformers import pipeline

_FINBERT_MODEL = "ProsusAI/finbert"
_classifier = None


def _get_classifier():
    global _classifier
    if _classifier is None:
        _classifier = pipeline("text-classification", model=_FINBERT_MODEL, top_k=None)
    return _classifier


def score_headlines(headlines: list) -> pd.DataFrame:
    columns = ["headline", "positive", "negative", "neutral", "sentiment_score"]
    if not headlines:
        return pd.DataFrame(columns=columns)
    classifier = _get_classifier()
    results = classifier(list(headlines), batch_size=16, truncation=True)
    rows = []
    for headline, scores in zip(headlines, results):
        score_map = {s["label"]: s["score"] for s in scores}
        positive = score_map.get("positive", 0.0)
        negative = score_map.get("negative", 0.0)
        neutral = score_map.get("neutral", 0.0)
        rows.append({
            "headline": headline,
            "positive": positive,
            "negative": negative,
            "neutral": neutral,
            "sentiment_score": positive - negative,
        })
    return pd.DataFrame(rows, columns=columns)
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `PYTHONPATH=. pytest tests/features/test_sentiment.py -v`
Expected: PASS (2 passed). Note: this test never downloads or runs the real FinBERT model — `_get_classifier` is fully mocked.

- [ ] **Step 7: Commit**

```bash
git add requirements.txt src/config.py src/features/sentiment.py tests/features/test_sentiment.py
git commit -m "feat: add FinBERT headline sentiment scoring"
```

---

### Task 4: Daily sentiment aggregation + rolling windows

**Files:**
- Modify: `src/features/sentiment.py` (append aggregation functions)
- Modify: `tests/features/test_sentiment.py` (append aggregation tests)

**Interfaces:**
- Consumes: `trading_day.assign_effective_trading_day` (Task 2, called by the caller, not internally — this task's functions take `effective_day` as an already-computed input, keeping `sentiment.py` decoupled from `trading_day.py`).
- Produces: `sentiment.aggregate_daily_sentiment(scored_news: pd.DataFrame, effective_day: pd.Series) -> pd.DataFrame` (indexed by date, columns `["sentiment_mean", "sentiment_std", "headline_count"]`), `sentiment.add_rolling_sentiment(daily_sentiment: pd.DataFrame, trading_days: pd.DatetimeIndex, windows=(3, 7)) -> pd.DataFrame` (adds `["sentiment_roll3", "sentiment_roll7"]`, reindexed to the full trading-day calendar) — both used by `build_features.build_ticker_feature_matrix`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/features/test_sentiment.py`:

```python
import pandas as pd
from src.features.sentiment import aggregate_daily_sentiment, add_rolling_sentiment


def test_aggregate_daily_sentiment_computes_mean_std_count():
    scored_news = pd.DataFrame({
        "headline": ["a", "b", "c"],
        "sentiment_score": [0.5, -0.5, 0.2],
    })
    effective_day = pd.Series([
        pd.Timestamp("2024-01-16"), pd.Timestamp("2024-01-16"), pd.Timestamp("2024-01-17"),
    ])
    daily = aggregate_daily_sentiment(scored_news, effective_day)
    assert daily.loc[pd.Timestamp("2024-01-16"), "sentiment_mean"] == pytest.approx(0.0)
    assert daily.loc[pd.Timestamp("2024-01-16"), "headline_count"] == 2
    assert daily.loc[pd.Timestamp("2024-01-17"), "sentiment_mean"] == pytest.approx(0.2)
    assert daily.loc[pd.Timestamp("2024-01-17"), "headline_count"] == 1


def test_add_rolling_sentiment_fills_no_news_days_with_zero_count():
    trading_days = pd.DatetimeIndex(["2024-01-16", "2024-01-17", "2024-01-18"])
    daily = pd.DataFrame(
        {"sentiment_mean": [0.5], "sentiment_std": [float("nan")], "headline_count": [1]},
        index=pd.DatetimeIndex(["2024-01-16"]),
    )
    result = add_rolling_sentiment(daily, trading_days, windows=(3, 7))
    assert result.loc[pd.Timestamp("2024-01-17"), "headline_count"] == 0
    assert list(result.index) == list(pd.DatetimeIndex(trading_days))


def test_add_rolling_sentiment_rolling_average_skips_no_news_days():
    trading_days = pd.DatetimeIndex(["2024-01-16", "2024-01-17", "2024-01-18"])
    daily = pd.DataFrame(
        {"sentiment_mean": [0.6, 0.0], "sentiment_std": [0.0, 0.0], "headline_count": [1, 1]},
        index=pd.DatetimeIndex(["2024-01-16", "2024-01-18"]),
    )
    result = add_rolling_sentiment(daily, trading_days, windows=(3,))
    # 2024-01-17 has no news (NaN sentiment_mean); the 3-day rolling average on
    # 2024-01-18 should average only the two real observations (0.6, 0.0) -> 0.3,
    # not treat the missing day as 0.
    assert result.loc[pd.Timestamp("2024-01-18"), "sentiment_roll3"] == pytest.approx(0.3)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `PYTHONPATH=. pytest tests/features/test_sentiment.py -v -k "aggregate or rolling"`
Expected: FAIL with `ImportError: cannot import name 'aggregate_daily_sentiment'`.

- [ ] **Step 3: Append to `src/features/sentiment.py`**

```python
def aggregate_daily_sentiment(scored_news: pd.DataFrame, effective_day: pd.Series) -> pd.DataFrame:
    df = scored_news.copy()
    df["effective_trading_day"] = effective_day.values
    grouped = df.groupby("effective_trading_day")["sentiment_score"]
    daily = grouped.agg(sentiment_mean="mean", sentiment_std="std", headline_count="count")
    daily.index.name = "date"
    return daily


def add_rolling_sentiment(daily_sentiment: pd.DataFrame, trading_days: pd.DatetimeIndex, windows=(3, 7)) -> pd.DataFrame:
    calendar = pd.DatetimeIndex(
        sorted(pd.DatetimeIndex(trading_days).tz_localize(None).normalize().unique())
    )
    reindexed = daily_sentiment.reindex(calendar)
    reindexed["headline_count"] = reindexed["headline_count"].fillna(0)
    for w in windows:
        reindexed[f"sentiment_roll{w}"] = reindexed["sentiment_mean"].rolling(window=w, min_periods=1).mean()
    return reindexed
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `PYTHONPATH=. pytest tests/features/test_sentiment.py -v`
Expected: PASS (5 passed — 2 from Task 3, 3 new).

- [ ] **Step 5: Commit**

```bash
git add src/features/sentiment.py tests/features/test_sentiment.py
git commit -m "feat: add daily sentiment aggregation and rolling windows"
```

---

### Task 5: Label generation (±2%/10-day triple-barrier)

**Files:**
- Create: `src/features/labels.py`
- Create: `tests/features/test_labels.py`

**Interfaces:**
- Consumes: nothing (pure numpy/pandas on a single-ticker `Close` series).
- Produces: `labels.generate_labels(close: pd.Series, horizon: int = 10, up_threshold: float = 0.02, down_threshold: float = 0.02) -> pd.Series` (named `"label"`, values `1.0`/`0.0`/`NaN`, same index as `close`) — used by `build_features.build_ticker_feature_matrix`.

- [ ] **Step 1: Write the failing tests**

Create `tests/features/test_labels.py`:

```python
import numpy as np
import pandas as pd
from src.features.labels import generate_labels


def test_label_1_when_up_threshold_hit_first():
    # entry=100 at index0; future = [101, 103, 104, 105, 106] (horizon=5)
    # up_target=102 hit at future index1 (103); down_target=98 never hit -> label 1
    idx = pd.date_range("2024-01-02", periods=6, freq="B")
    close = pd.Series([100, 101, 103, 104, 105, 106], index=idx, dtype=float)
    labels = generate_labels(close, horizon=5, up_threshold=0.02, down_threshold=0.02)
    assert labels.iloc[0] == 1.0


def test_label_0_when_down_threshold_hit_first():
    # entry=100; future = [99, 97, 96, 95, 94] (horizon=5)
    # down_target=98 hit at future index1 (97); up_target=102 never hit -> label 0
    idx = pd.date_range("2024-01-02", periods=6, freq="B")
    close = pd.Series([100, 99, 97, 96, 95, 94], index=idx, dtype=float)
    labels = generate_labels(close, horizon=5, up_threshold=0.02, down_threshold=0.02)
    assert labels.iloc[0] == 0.0


def test_label_nan_when_neither_threshold_hit():
    # entry=100; future stays within [98, 102] the whole horizon -> ambiguous, drop
    idx = pd.date_range("2024-01-02", periods=6, freq="B")
    close = pd.Series([100, 100.5, 101, 100.8, 101.2, 100.9], index=idx, dtype=float)
    labels = generate_labels(close, horizon=5, up_threshold=0.02, down_threshold=0.02)
    assert np.isnan(labels.iloc[0])


def test_label_nan_for_tail_rows_with_insufficient_future_data():
    # 6 rows total, horizon=5 -> only index0 has a full 5-day future window;
    # indices 1-5 must all be NaN regardless of their values.
    idx = pd.date_range("2024-01-02", periods=6, freq="B")
    close = pd.Series([100, 106, 106, 106, 106, 106], index=idx, dtype=float)
    labels = generate_labels(close, horizon=5, up_threshold=0.02, down_threshold=0.02)
    assert labels.iloc[0] == 1.0
    assert labels.iloc[1:].isna().all()


def test_labels_preserve_index():
    idx = pd.date_range("2024-01-02", periods=6, freq="B")
    close = pd.Series([100, 101, 103, 104, 105, 106], index=idx, dtype=float)
    labels = generate_labels(close, horizon=5)
    assert labels.index.equals(idx)
    assert labels.name == "label"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `PYTHONPATH=. pytest tests/features/test_labels.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.features.labels'`.

- [ ] **Step 3: Write `src/features/labels.py`**

```python
import numpy as np
import pandas as pd


def generate_labels(close: pd.Series, horizon: int = 10, up_threshold: float = 0.02, down_threshold: float = 0.02) -> pd.Series:
    values = close.values
    n = len(values)
    labels = np.full(n, np.nan)

    for t in range(n):
        entry = values[t]
        future = values[t + 1: t + 1 + horizon]
        if len(future) < horizon:
            continue  # insufficient future data - drop (tail row)

        up_target = entry * (1 + up_threshold)
        down_target = entry * (1 - down_threshold)
        up_hits = future >= up_target
        down_hits = future <= down_target
        hit_up = int(np.argmax(up_hits)) if up_hits.any() else None
        hit_down = int(np.argmax(down_hits)) if down_hits.any() else None

        if hit_up is None and hit_down is None:
            continue  # neither threshold hit within horizon - ambiguous, drop
        elif hit_down is None:
            labels[t] = 1.0
        elif hit_up is None:
            labels[t] = 0.0
        else:
            labels[t] = 1.0 if hit_up <= hit_down else 0.0

    return pd.Series(labels, index=close.index, name="label")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `PYTHONPATH=. pytest tests/features/test_labels.py -v`
Expected: PASS (5 passed).

- [ ] **Step 5: Commit**

```bash
git add src/features/labels.py tests/features/test_labels.py
git commit -m "feat: add triple-barrier label generation (+-2%%/10-day horizon)"
```

---

### Task 6: Feature matrix orchestrator

**Files:**
- Create: `src/features/build_features.py`
- Create: `tests/features/test_build_features.py`

**Interfaces:**
- Consumes: `technical.compute_technical_features`, `trading_day.assign_effective_trading_day`, `sentiment.aggregate_daily_sentiment`, `sentiment.add_rolling_sentiment`, `labels.generate_labels`.
- Produces: `build_features.build_ticker_feature_matrix(price_df: pd.DataFrame, scored_news: pd.DataFrame, trading_days: pd.DatetimeIndex) -> pd.DataFrame` (single ticker, technical + sentiment + label columns + `"ticker"`, warmup/ambiguous/tail rows dropped), `build_features.build_feature_matrix(price_panel: pd.DataFrame, scored_news_by_ticker: dict, trading_days: pd.DatetimeIndex) -> pd.DataFrame` (all tickers concatenated) — used by `scripts/run_phase2_pipeline.py`.

- [ ] **Step 1: Write the failing tests**

Create `tests/features/test_build_features.py`:

```python
import numpy as np
import pandas as pd
from src.features.build_features import build_ticker_feature_matrix, build_feature_matrix


def _price_df(ticker="AAPL", n=80, start=100.0, step=0.5):
    idx = pd.date_range("2024-01-02", periods=n, freq="B")
    close = pd.Series(start + step * np.arange(n), index=idx)
    volume = pd.Series(1_000_000 + 1000 * np.arange(n), index=idx)
    df = pd.DataFrame({
        "Open": close, "High": close, "Low": close, "Close": close, "Volume": volume,
    })
    df.index.name = "date"
    df["ticker"] = ticker
    return df


def _scored_news(ticker="AAPL"):
    idx = pd.date_range("2024-01-02", periods=80, freq="B")
    # one headline published well before close on each of the first 5 trading days
    published = [pd.Timestamp(d, tz="America/New_York").replace(hour=9).tz_convert("UTC") for d in idx[:5]]
    return pd.DataFrame({
        "headline": [f"headline {i}" for i in range(5)],
        "ticker": ticker,
        "sentiment_score": [0.3, -0.1, 0.5, 0.0, 0.2],
        "published_at": published,
    })


def test_build_ticker_feature_matrix_drops_warmup_and_tail_rows():
    price_df = _price_df()
    scored_news = _scored_news()
    trading_days = price_df.index
    matrix = build_ticker_feature_matrix(price_df, scored_news, trading_days)
    # warmup: dist_from_ma50 needs 50 rows; tail: label needs 10 future rows.
    assert matrix.index.min() >= price_df.index[49]
    assert matrix.index.max() <= price_df.index[-11]
    assert matrix["label"].isna().sum() == 0
    assert "ticker" in matrix.columns
    assert (matrix["ticker"] == "AAPL").all()


def test_build_ticker_feature_matrix_handles_ticker_with_no_news():
    price_df = _price_df(ticker="MSFT")
    empty_news = pd.DataFrame(columns=["headline", "ticker", "sentiment_score", "published_at"])
    trading_days = price_df.index
    matrix = build_ticker_feature_matrix(price_df, empty_news, trading_days)
    assert len(matrix) > 0
    assert matrix["headline_count"].fillna(0).eq(0).all()


def test_build_feature_matrix_concatenates_all_tickers():
    price_panel = pd.concat([_price_df("AAPL"), _price_df("MSFT")])
    scored_news_by_ticker = {"AAPL": _scored_news("AAPL")}
    trading_days = _price_df().index
    matrix = build_feature_matrix(price_panel, scored_news_by_ticker, trading_days)
    assert set(matrix["ticker"].unique()) == {"AAPL", "MSFT"}


def test_full_pipeline_is_causal_no_lookahead():
    # End-to-end causality check across technical + sentiment + label joins:
    # altering price and news data from day 60 onward must not change any
    # feature (not label - labels are intentionally forward-looking) at day < 60.
    price_df = _price_df(n=80)
    scored_news = _scored_news()
    trading_days = price_df.index
    feature_cols = [
        "return_1d", "return_5d", "return_10d", "return_20d", "rsi_14",
        "macd", "macd_signal", "macd_hist", "volatility_20d", "volume_ratio_20d",
        "dist_from_ma50", "sentiment_mean", "sentiment_std", "headline_count",
        "sentiment_roll3", "sentiment_roll7",
    ]
    matrix_before = build_ticker_feature_matrix(price_df, scored_news, trading_days)

    price_df_modified = price_df.copy()
    price_df_modified.loc[price_df_modified.index[60]:, "Close"] *= 5
    price_df_modified.loc[price_df_modified.index[60]:, "Volume"] *= 5

    matrix_after = build_ticker_feature_matrix(price_df_modified, scored_news, trading_days)

    common_index = matrix_before.index[matrix_before.index < price_df.index[60]]
    common_index = common_index.intersection(matrix_after.index)
    assert len(common_index) > 0
    pd.testing.assert_frame_equal(
        matrix_before.loc[common_index, feature_cols],
        matrix_after.loc[common_index, feature_cols],
    )
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `PYTHONPATH=. pytest tests/features/test_build_features.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.features.build_features'`.

- [ ] **Step 3: Write `src/features/build_features.py`**

```python
import pandas as pd

from src.features.technical import compute_technical_features
from src.features.trading_day import assign_effective_trading_day
from src.features.sentiment import aggregate_daily_sentiment, add_rolling_sentiment
from src.features.labels import generate_labels


def build_ticker_feature_matrix(price_df: pd.DataFrame, scored_news: pd.DataFrame, trading_days: pd.DatetimeIndex) -> pd.DataFrame:
    ticker = price_df["ticker"].iloc[0]
    technical = compute_technical_features(price_df)

    if not scored_news.empty:
        effective_day = assign_effective_trading_day(scored_news["published_at"], trading_days)
        daily_sentiment = aggregate_daily_sentiment(scored_news, effective_day)
    else:
        daily_sentiment = pd.DataFrame(columns=["sentiment_mean", "sentiment_std", "headline_count"])
    sentiment = add_rolling_sentiment(daily_sentiment, trading_days)

    labels = generate_labels(price_df["Close"])

    matrix = technical.join(sentiment, how="left").join(labels, how="left")
    matrix["ticker"] = ticker
    matrix = matrix.dropna(subset=technical.columns)
    matrix = matrix.dropna(subset=["label"])
    return matrix


def build_feature_matrix(price_panel: pd.DataFrame, scored_news_by_ticker: dict, trading_days: pd.DatetimeIndex) -> pd.DataFrame:
    empty_news = pd.DataFrame(columns=["headline", "ticker", "sentiment_score", "published_at"])
    frames = []
    for ticker, ticker_price_df in price_panel.groupby("ticker"):
        scored_news = scored_news_by_ticker.get(ticker, empty_news)
        frames.append(build_ticker_feature_matrix(ticker_price_df, scored_news, trading_days))
    return pd.concat(frames)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `PYTHONPATH=. pytest tests/features/test_build_features.py -v`
Expected: PASS (4 passed).

- [ ] **Step 5: Run the full test suite**

Run: `PYTHONPATH=. pytest tests/ -v`
Expected: all tests pass (Phase 1's 33 + Phase 2's Tasks 1-6).

- [ ] **Step 6: Commit**

```bash
git add src/features/build_features.py tests/features/test_build_features.py
git commit -m "feat: add feature matrix orchestrator joining technical, sentiment, and labels"
```

---

### Task 7: Live pipeline run script + manual verification

**Files:**
- Create: `scripts/run_phase2_pipeline.py`

**Interfaces:**
- Consumes: `config.*`, `src.data.tickers.UNIVERSE`, `src.data.calendar.get_nyse_trading_days`, `src.data.storage.load_parquet`/`save_parquet`, `src.features.sentiment.score_headlines`, `src.features.build_features.build_feature_matrix`.
- Produces: `data/raw/sentiment/<TICKER>.parquet` (cached scored headlines, so re-runs don't re-score with the real FinBERT model), `data/processed/feature_matrix.parquet` (the Phase 2 deliverable). This script is not covered by automated tests — it downloads and runs the real FinBERT model.

- [ ] **Step 1: Write `scripts/run_phase2_pipeline.py`**

```python
from src import config
from src.data.tickers import UNIVERSE
from src.data.calendar import get_nyse_trading_days
from src.data.storage import load_parquet, save_parquet
from src.features.sentiment import score_headlines
from src.features.build_features import build_feature_matrix


def _load_or_score_sentiment(ticker: str) -> None:
    cache_path = config.SENTIMENT_DIR / f"{ticker}.parquet"
    if cache_path.exists():
        return
    news = load_parquet(ticker, config.NEWS_DIR)
    scored = score_headlines(list(news["headline"]))
    scored["published_at"] = news["published_at"].values
    scored["ticker"] = ticker
    save_parquet(scored, ticker, config.SENTIMENT_DIR)


def main():
    trading_days = get_nyse_trading_days(config.START_DATE, config.END_DATE)
    price_panel = load_parquet("combined_price_panel", config.DATA_PROCESSED_DIR)

    scored_news_by_ticker = {}
    for ticker in UNIVERSE:
        print(f"Scoring sentiment for {ticker}...")
        _load_or_score_sentiment(ticker)
        scored_news_by_ticker[ticker] = load_parquet(ticker, config.SENTIMENT_DIR)

    matrix = build_feature_matrix(price_panel, scored_news_by_ticker, trading_days)

    print(f"Feature matrix rows: {len(matrix)}")
    print(f"Tickers represented: {sorted(matrix['ticker'].unique())}")
    label_counts = matrix["label"].value_counts(normalize=True)
    print(f"Label balance: {dict(label_counts)}")
    majority_baseline = label_counts.max()
    print(f"Majority-class baseline accuracy: {majority_baseline:.3f}")
    if majority_baseline > 0.65:
        print("WARNING: label balance is skewed (>65% majority class) - accuracy targets need to beat this baseline, not 50%.")

    config.DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    save_parquet(matrix, "feature_matrix", config.DATA_PROCESSED_DIR)
    print(f"Saved feature matrix to {config.DATA_PROCESSED_DIR / 'feature_matrix.parquet'}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the live pipeline**

Run: `PYTHONPATH=. python scripts/run_phase2_pipeline.py`

This downloads the FinBERT model on first run (~440MB) and scores every cached headline from Phase 1 (~6,700 headlines across 28 tickers) — expect several minutes on CPU. Subsequent runs skip already-scored tickers via the `data/raw/sentiment/` cache.

Expected console output pattern:
```
Scoring sentiment for AAPL...
...
Feature matrix rows: <tens of thousands>
Tickers represented: [... 28 tickers ...]
Label balance: {1.0: <~0.4-0.6>, 0.0: <~0.4-0.6>}
Majority-class baseline accuracy: <~0.5-0.6>
Saved feature matrix to .../data/processed/feature_matrix.parquet
```

- [ ] **Step 3: Manually verify the deliverable**

Check:
1. `Tickers represented` includes all 28 tickers from `UNIVERSE` — if any are missing, that ticker's news or price data may be malformed; investigate before Phase 3.
2. `Label balance` is roughly 40-60% per class (per the project's success metric: "50%-ish class balance verified"). If one class exceeds ~65%, note the majority-class baseline explicitly — Phase 3's accuracy target must beat this number, not a naive 50%.
3. `data/raw/sentiment/` has one parquet per ticker, and `data/processed/feature_matrix.parquet` exists and is non-empty.
4. Spot-check: load the feature matrix and confirm no `NaN` values remain in any feature or label column (the warmup/ambiguous/tail dropping in `build_ticker_feature_matrix` should have removed them all).

- [ ] **Step 4: Commit**

```bash
git add scripts/run_phase2_pipeline.py
git commit -m "feat: add live Phase 2 pipeline runner script (FinBERT scoring + feature matrix)"
```

---

## Self-Review Notes

- **Spec coverage:** Technical features (Task 1, exact list: returns 1/5/10/20d, RSI, MACD, volatility, volume ratio, dist-from-MA — 11 columns matching the spec's "~10-15"). Sentiment features (Tasks 3-4: FinBERT mean/std/count + 3-day/7-day rolling). Labels (Task 5: ±2%/10-day triple-barrier, ambiguous rows dropped). The point-in-time hard rule is explicitly addressed by Task 2 (news) and verified by causality tests in Tasks 1 and 6 (technical + full pipeline) — this is the project's single most emphasized constraint and now has direct automated regression coverage, not just a design intention.
- **Placeholder scan:** No TBD/TODO markers. Task 7 (live FinBERT run) is the one manual, non-pytest task, matching Phase 1's Task 8 pattern — it has concrete expected output and a concrete verification checklist instead of asserted output, because it needs the real ~440MB FinBERT model download and live compute, neither available in an automated test run.
- **Type consistency:** `compute_technical_features` (Task 1), `assign_effective_trading_day` (Task 2), `score_headlines`/`aggregate_daily_sentiment`/`add_rolling_sentiment` (Tasks 3-4), and `generate_labels` (Task 5) are all called with matching signatures in Task 6's `build_ticker_feature_matrix` — verified against each module's own interface declaration above. Phase 1's `storage.save_parquet`/`load_parquet` are reused as-is in Task 7 for the new `SENTIMENT_DIR` cache, consistent with their existing generic (ticker, out_dir) signature.

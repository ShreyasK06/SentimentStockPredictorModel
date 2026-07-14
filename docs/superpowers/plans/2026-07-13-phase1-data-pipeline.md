# Phase 1: Data Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a rock-solid pipeline that pulls 5+ years of daily OHLCV price data and financial news headlines for a 20-30 stock universe, stores everything in parquet keyed by (ticker, date), and validates there are no gaps or misaligned dates — the foundation the rest of the sentiment-stock-predictor project builds on.

**Architecture:** Small, single-responsibility modules under `src/data/`: a ticker universe constant, an NYSE trading-calendar helper (for gap detection against real market days, not just weekdays), a yfinance-backed price fetcher, a Finnhub-backed news fetcher, generic parquet storage helpers, and an orchestrator that ties them together and produces a validation report. A manual (non-pytest) script runs the live pull against real APIs.

**Tech Stack:** Python, pandas, yfinance, pandas_market_calendars, requests (Finnhub API), pyarrow (parquet), python-dotenv, pytest + pytest-mock.

## Global Constraints

- Universe: 20-30 large-cap, high-liquidity US stocks spanning multiple sectors (not just tech), to avoid overfitting to one sector's regime.
- Price data: daily OHLCV, minimum 5 years, must include the 2022 bear market drawdown.
- Sentiment data source: financial news headlines aggregated to daily per-ticker (this plan fetches and stores raw headlines only; FinBERT scoring is Phase 2).
- Storage: parquet files keyed by (ticker, date). No SQLite — parquet is simpler for this access pattern and plays natively with pandas.
- Hard rule (project-wide, applies to all future phases): every feature for day t must use only information available before market close on day t. This plan stores raw timestamped data only and does no feature computation, so it cannot violate this rule yet — but date/timestamp fields must be preserved precisely so Phase 2 can enforce it.
- Deliverable: a combined price panel of ~20-30 tickers × ~1,500 trading days with zero gaps against the real NYSE trading calendar.

---

## File Structure

```
SentimentStockPredictorModel/
  requirements.txt
  .env.example
  src/
    __init__.py
    config.py                  # paths, date range, API key loading
    data/
      __init__.py
      tickers.py                # UNIVERSE list + SECTOR_MAP
      calendar.py                # NYSE trading-day calendar helper
      prices.py                  # yfinance OHLCV fetcher
      news.py                    # Finnhub news fetcher + normalizer
      storage.py                  # generic parquet save/load + gap detection
      build_dataset.py            # orchestrator: build panel, validate, archive news
  scripts/
    run_phase1_pipeline.py       # manual live-run entry point (not unit tested)
  tests/
    __init__.py
    test_config.py
    data/
      __init__.py
      test_tickers.py
      test_calendar.py
      test_prices.py
      test_storage.py
      test_news.py
      test_build_dataset.py
  data/
    raw/
      prices/                   # created at runtime
      news/                     # created at runtime
    processed/                  # created at runtime
```

---

### Task 1: Project scaffolding & config

**Files:**
- Create: `requirements.txt`
- Create: `.env.example`
- Create: `src/__init__.py`
- Create: `src/config.py`
- Create: `tests/__init__.py`
- Create: `tests/test_config.py`

**Interfaces:**
- Produces: `config.PROJECT_ROOT: Path`, `config.DATA_RAW_DIR: Path`, `config.DATA_PROCESSED_DIR: Path`, `config.PRICES_DIR: Path`, `config.NEWS_DIR: Path`, `config.FINNHUB_API_KEY: str`, `config.START_DATE: str`, `config.END_DATE: str` — every later task imports these.

- [ ] **Step 1: Create `requirements.txt`**

```
pandas
numpy
yfinance
pandas_market_calendars
requests
pyarrow
python-dotenv
pytest
pytest-mock
```

- [ ] **Step 2: Create `.env.example`**

```
FINNHUB_API_KEY=your_finnhub_api_key_here
```

- [ ] **Step 3: Install dependencies**

Run: `pip install -r requirements.txt`
Expected: all packages install without error.

- [ ] **Step 4: Write the failing test**

Create `tests/__init__.py` (empty file) and `tests/test_config.py`:

```python
from datetime import date
from src import config


def test_directories_are_paths():
    assert config.PRICES_DIR.name == "prices"
    assert config.NEWS_DIR.name == "news"
    assert config.DATA_PROCESSED_DIR.name == "processed"


def test_date_range_covers_at_least_five_years():
    start = date.fromisoformat(config.START_DATE)
    end = date.fromisoformat(config.END_DATE)
    assert (end - start).days >= 5 * 365


def test_date_range_includes_2022_bear_market():
    start = date.fromisoformat(config.START_DATE)
    end = date.fromisoformat(config.END_DATE)
    assert start <= date(2022, 1, 1)
    assert end >= date(2022, 12, 31)
```

- [ ] **Step 5: Run test to verify it fails**

Run: `pytest tests/test_config.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.config'` (or `src` has no attribute `config`).

- [ ] **Step 6: Create `src/__init__.py`** (empty file)

- [ ] **Step 7: Write `src/config.py`**

```python
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
PRICES_DIR = DATA_RAW_DIR / "prices"
NEWS_DIR = DATA_RAW_DIR / "news"

FINNHUB_API_KEY = os.environ.get("FINNHUB_API_KEY", "")

START_DATE = "2019-01-01"
END_DATE = "2025-12-31"
```

- [ ] **Step 8: Run test to verify it passes**

Run: `pytest tests/test_config.py -v`
Expected: PASS (3 passed).

- [ ] **Step 9: Commit**

```bash
git add requirements.txt .env.example src/__init__.py src/config.py tests/__init__.py tests/test_config.py
git commit -m "chore: scaffold project config and dependencies"
```

---

### Task 2: Ticker universe module

**Files:**
- Create: `src/data/__init__.py`
- Create: `src/data/tickers.py`
- Create: `tests/data/__init__.py`
- Create: `tests/data/test_tickers.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `tickers.UNIVERSE: list[str]`, `tickers.SECTOR_MAP: dict[str, str]` — used by `build_dataset.py` and `scripts/run_phase1_pipeline.py`.

- [ ] **Step 1: Write the failing test**

Create `tests/data/__init__.py` (empty) and `tests/data/test_tickers.py`:

```python
from src.data.tickers import UNIVERSE, SECTOR_MAP


def test_universe_size_in_range():
    assert 20 <= len(UNIVERSE) <= 30


def test_universe_has_no_duplicates():
    assert len(UNIVERSE) == len(set(UNIVERSE))


def test_every_ticker_has_a_sector():
    for ticker in UNIVERSE:
        assert ticker in SECTOR_MAP


def test_at_least_four_sectors_represented():
    sectors = {SECTOR_MAP[t] for t in UNIVERSE}
    assert len(sectors) >= 4
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/data/test_tickers.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.data'`.

- [ ] **Step 3: Create `src/data/__init__.py`** (empty file)

- [ ] **Step 4: Write `src/data/tickers.py`**

```python
UNIVERSE = [
    # Tech
    "AAPL", "MSFT", "GOOGL", "AMZN", "META",
    # Finance
    "JPM", "BAC", "GS", "MS", "WFC",
    # Healthcare
    "UNH", "JNJ", "PFE", "ABBV", "MRK",
    # Energy
    "XOM", "CVX", "COP", "SLB", "EOG",
    # Consumer
    "PG", "KO", "WMT", "HD", "MCD",
    # Industrials
    "CAT", "BA", "HON",
]

SECTOR_MAP = {
    "AAPL": "Tech", "MSFT": "Tech", "GOOGL": "Tech", "AMZN": "Tech", "META": "Tech",
    "JPM": "Finance", "BAC": "Finance", "GS": "Finance", "MS": "Finance", "WFC": "Finance",
    "UNH": "Healthcare", "JNJ": "Healthcare", "PFE": "Healthcare", "ABBV": "Healthcare", "MRK": "Healthcare",
    "XOM": "Energy", "CVX": "Energy", "COP": "Energy", "SLB": "Energy", "EOG": "Energy",
    "PG": "Consumer", "KO": "Consumer", "WMT": "Consumer", "HD": "Consumer", "MCD": "Consumer",
    "CAT": "Industrials", "BA": "Industrials", "HON": "Industrials",
}
```

- [ ] **Step 5: Run test to verify it passes**

Run: `pytest tests/data/test_tickers.py -v`
Expected: PASS (4 passed).

- [ ] **Step 6: Commit**

```bash
git add src/data/__init__.py src/data/tickers.py tests/data/__init__.py tests/data/test_tickers.py
git commit -m "feat: add 28-ticker universe across 6 sectors"
```

---

### Task 3: NYSE trading calendar helper

**Files:**
- Create: `src/data/calendar.py`
- Create: `tests/data/test_calendar.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `calendar.get_nyse_trading_days(start: str, end: str) -> pd.DatetimeIndex` — used by `storage.find_missing_trading_days` callers in `build_dataset.py`.

- [ ] **Step 1: Write the failing test**

Create `tests/data/test_calendar.py`:

```python
import pandas as pd
from src.data.calendar import get_nyse_trading_days


def test_returns_datetimeindex():
    days = get_nyse_trading_days("2024-01-01", "2024-01-10")
    assert isinstance(days, pd.DatetimeIndex)
    assert len(days) > 0


def test_excludes_weekends():
    days = get_nyse_trading_days("2024-01-01", "2024-01-07")
    assert all(pd.Timestamp(d).weekday() < 5 for d in days)


def test_excludes_known_holiday():
    days = get_nyse_trading_days("2024-01-01", "2024-01-02")
    assert pd.Timestamp("2024-01-01").normalize() not in pd.DatetimeIndex(days).normalize()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/data/test_calendar.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.data.calendar'`.

- [ ] **Step 3: Write `src/data/calendar.py`**

```python
import pandas as pd
import pandas_market_calendars as mcal


def get_nyse_trading_days(start: str, end: str) -> pd.DatetimeIndex:
    nyse = mcal.get_calendar("NYSE")
    schedule = nyse.schedule(start_date=start, end_date=end)
    return pd.DatetimeIndex(schedule.index.date)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/data/test_calendar.py -v`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
git add src/data/calendar.py tests/data/test_calendar.py
git commit -m "feat: add NYSE trading calendar helper for gap detection"
```

---

### Task 4: Price fetcher (yfinance)

**Files:**
- Create: `src/data/prices.py`
- Create: `tests/data/test_prices.py`

**Interfaces:**
- Consumes: nothing (wraps `yfinance.download` directly).
- Produces: `prices.fetch_ohlcv(ticker: str, start: str, end: str) -> pd.DataFrame` with columns `["Open", "High", "Low", "Close", "Volume", "ticker"]` and a DatetimeIndex named `"date"` — used by `build_dataset.build_price_panel`.

- [ ] **Step 1: Write the failing test**

Create `tests/data/test_prices.py`:

```python
import pandas as pd
import pytest
from unittest.mock import patch
from src.data.prices import fetch_ohlcv


def _fake_yf_frame():
    idx = pd.date_range("2024-01-02", periods=3, freq="B")
    return pd.DataFrame({
        "Open": [100.0, 101.0, 102.0],
        "High": [101.0, 102.0, 103.0],
        "Low": [99.0, 100.0, 101.0],
        "Close": [100.5, 101.5, 102.5],
        "Adj Close": [100.5, 101.5, 102.5],
        "Volume": [1000, 1100, 1200],
    }, index=idx)


@patch("src.data.prices.yf.download")
def test_fetch_ohlcv_returns_expected_columns(mock_download):
    mock_download.return_value = _fake_yf_frame()
    df = fetch_ohlcv("AAPL", "2024-01-01", "2024-01-05")
    assert list(df.columns) == ["Open", "High", "Low", "Close", "Volume", "ticker"]
    assert (df["ticker"] == "AAPL").all()
    assert len(df) == 3
    assert df.index.name == "date"


@patch("src.data.prices.yf.download")
def test_fetch_ohlcv_raises_on_empty(mock_download):
    mock_download.return_value = pd.DataFrame()
    with pytest.raises(ValueError):
        fetch_ohlcv("BADTICKER", "2024-01-01", "2024-01-05")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/data/test_prices.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.data.prices'`.

- [ ] **Step 3: Write `src/data/prices.py`**

```python
import pandas as pd
import yfinance as yf

REQUIRED_COLUMNS = ["Open", "High", "Low", "Close", "Volume"]


def fetch_ohlcv(ticker: str, start: str, end: str) -> pd.DataFrame:
    df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
    if df.empty:
        raise ValueError(f"No price data returned for {ticker} between {start} and {end}")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df[REQUIRED_COLUMNS].copy()
    df.index.name = "date"
    df["ticker"] = ticker
    return df
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/data/test_prices.py -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add src/data/prices.py tests/data/test_prices.py
git commit -m "feat: add yfinance OHLCV fetcher"
```

---

### Task 5: Parquet storage + gap detection

**Files:**
- Create: `src/data/storage.py`
- Create: `tests/data/test_storage.py`

**Interfaces:**
- Consumes: nothing (generic pandas/parquet I/O).
- Produces: `storage.save_parquet(df: pd.DataFrame, ticker: str, out_dir: Path) -> Path`, `storage.load_parquet(ticker: str, out_dir: Path) -> pd.DataFrame`, `storage.find_missing_trading_days(df: pd.DataFrame, trading_days: pd.DatetimeIndex) -> list[str]` — used by `build_dataset.py` for both prices and news, keyed on the DataFrame's index.

- [ ] **Step 1: Write the failing test**

Create `tests/data/test_storage.py`:

```python
import pandas as pd
from src.data.storage import save_parquet, load_parquet, find_missing_trading_days


def test_save_and_load_roundtrip(tmp_path):
    idx = pd.date_range("2024-01-02", periods=3, freq="B")
    df = pd.DataFrame({"Open": [1, 2, 3], "Close": [1, 2, 3]}, index=idx)
    save_parquet(df, "AAPL", tmp_path)
    loaded = load_parquet("AAPL", tmp_path)
    assert list(loaded["Close"]) == [1, 2, 3]


def test_find_missing_trading_days_detects_gap():
    idx = pd.DatetimeIndex(["2024-01-02", "2024-01-04"])
    df = pd.DataFrame({"Close": [1, 2]}, index=idx)
    trading_days = pd.DatetimeIndex(["2024-01-02", "2024-01-03", "2024-01-04"])
    missing = find_missing_trading_days(df, trading_days)
    assert missing == ["2024-01-03"]


def test_find_missing_trading_days_empty_when_complete():
    idx = pd.DatetimeIndex(["2024-01-02", "2024-01-03"])
    df = pd.DataFrame({"Close": [1, 2]}, index=idx)
    trading_days = pd.DatetimeIndex(["2024-01-02", "2024-01-03"])
    missing = find_missing_trading_days(df, trading_days)
    assert missing == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/data/test_storage.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.data.storage'`.

- [ ] **Step 3: Write `src/data/storage.py`**

```python
from pathlib import Path
import pandas as pd


def save_parquet(df: pd.DataFrame, ticker: str, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{ticker}.parquet"
    df.to_parquet(path)
    return path


def load_parquet(ticker: str, out_dir: Path) -> pd.DataFrame:
    path = out_dir / f"{ticker}.parquet"
    return pd.read_parquet(path)


def find_missing_trading_days(df: pd.DataFrame, trading_days: pd.DatetimeIndex) -> list:
    have = set(pd.DatetimeIndex(df.index).normalize())
    want = set(pd.DatetimeIndex(trading_days).normalize())
    missing = sorted(want - have)
    return [d.strftime("%Y-%m-%d") for d in missing]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/data/test_storage.py -v`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
git add src/data/storage.py tests/data/test_storage.py
git commit -m "feat: add generic parquet storage and trading-day gap detection"
```

---

### Task 6: Finnhub news fetcher

**Files:**
- Create: `src/data/news.py`
- Create: `tests/data/test_news.py`

**Interfaces:**
- Consumes: nothing (wraps `requests.get` against the Finnhub company-news endpoint).
- Produces: `news.fetch_company_news(ticker: str, start: str, end: str, api_key: str, sleep_seconds: float = 1.0) -> list[dict]`, `news.articles_to_dataframe(articles: list[dict], ticker: str) -> pd.DataFrame` with columns `["date", "ticker", "headline", "source", "url"]` — used by `build_dataset.build_news_archive`.

- [ ] **Step 1: Write the failing test**

Create `tests/data/test_news.py`:

```python
from unittest.mock import patch, MagicMock
from src.data.news import fetch_company_news, articles_to_dataframe, _year_chunks


def test_year_chunks_splits_by_calendar_year():
    chunks = list(_year_chunks("2019-06-01", "2021-02-15"))
    assert chunks == [
        ("2019-06-01", "2019-12-31"),
        ("2020-01-01", "2020-12-31"),
        ("2021-01-01", "2021-02-15"),
    ]


@patch("src.data.news.time.sleep", return_value=None)
@patch("src.data.news.requests.get")
def test_fetch_company_news_aggregates_chunks(mock_get, mock_sleep):
    mock_resp = MagicMock()
    mock_resp.json.return_value = [{"headline": "x", "datetime": 1704153600, "source": "Reuters", "url": "http://x"}]
    mock_resp.raise_for_status.return_value = None
    mock_get.return_value = mock_resp

    articles = fetch_company_news("AAPL", "2024-01-01", "2024-01-05", api_key="fake")
    assert len(articles) == 1
    assert mock_get.call_count == 1
    call_kwargs = mock_get.call_args.kwargs
    assert call_kwargs["params"]["symbol"] == "AAPL"
    assert call_kwargs["params"]["token"] == "fake"


def test_articles_to_dataframe_normalizes_schema():
    articles = [{"headline": "AAPL rises", "datetime": 1704153600, "source": "Reuters", "url": "http://x"}]
    df = articles_to_dataframe(articles, "AAPL")
    assert list(df.columns) == ["date", "ticker", "headline", "source", "url"]
    assert df.iloc[0]["ticker"] == "AAPL"


def test_articles_to_dataframe_empty_list():
    df = articles_to_dataframe([], "AAPL")
    assert len(df) == 0
    assert list(df.columns) == ["date", "ticker", "headline", "source", "url"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/data/test_news.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.data.news'`.

- [ ] **Step 3: Write `src/data/news.py`**

```python
import time
from datetime import date, timedelta
import pandas as pd
import requests

FINNHUB_URL = "https://finnhub.io/api/v1/company-news"


def _year_chunks(start: str, end: str):
    start_d = date.fromisoformat(start)
    end_d = date.fromisoformat(end)
    chunk_start = start_d
    while chunk_start <= end_d:
        chunk_end = min(date(chunk_start.year, 12, 31), end_d)
        yield chunk_start.isoformat(), chunk_end.isoformat()
        chunk_start = chunk_end + timedelta(days=1)


def fetch_company_news(ticker: str, start: str, end: str, api_key: str, sleep_seconds: float = 1.0) -> list:
    articles = []
    for chunk_start, chunk_end in _year_chunks(start, end):
        resp = requests.get(FINNHUB_URL, params={
            "symbol": ticker, "from": chunk_start, "to": chunk_end, "token": api_key,
        }, timeout=30)
        resp.raise_for_status()
        articles.extend(resp.json())
        time.sleep(sleep_seconds)
    return articles


def articles_to_dataframe(articles: list, ticker: str) -> pd.DataFrame:
    columns = ["date", "ticker", "headline", "source", "url"]
    if not articles:
        return pd.DataFrame(columns=columns)
    df = pd.DataFrame(articles)
    df["date"] = pd.to_datetime(df["datetime"], unit="s").dt.normalize()
    df["ticker"] = ticker
    return df[columns]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/data/test_news.py -v`
Expected: PASS (4 passed).

- [ ] **Step 5: Commit**

```bash
git add src/data/news.py tests/data/test_news.py
git commit -m "feat: add Finnhub company-news fetcher with year-chunked pagination"
```

---

### Task 7: Dataset build orchestrator

**Files:**
- Create: `src/data/build_dataset.py`
- Create: `tests/data/test_build_dataset.py`

**Interfaces:**
- Consumes: `tickers.UNIVERSE`, `calendar.get_nyse_trading_days`, `prices.fetch_ohlcv`, `news.fetch_company_news`, `news.articles_to_dataframe`, `storage.save_parquet`, `storage.find_missing_trading_days`.
- Produces: `build_dataset.build_price_panel(tickers: list[str], start: str, end: str, prices_dir: Path) -> pd.DataFrame` (long panel with a `"date"` column, saves one parquet per ticker as a side effect), `build_dataset.validate_price_panel(panel: pd.DataFrame, tickers: list[str], trading_days: pd.DatetimeIndex) -> dict[str, list[str]]`, `build_dataset.build_news_archive(tickers: list[str], start: str, end: str, news_dir: Path, api_key: str) -> dict[str, int]` — used by `scripts/run_phase1_pipeline.py`.

- [ ] **Step 1: Write the failing test**

Create `tests/data/test_build_dataset.py`:

```python
import pandas as pd
from unittest.mock import patch
from src.data.build_dataset import build_price_panel, validate_price_panel


def _fake_prices(ticker, start, end):
    idx = pd.date_range("2024-01-02", periods=3, freq="B")
    df = pd.DataFrame({
        "Open": [1.0] * 3, "High": [1.0] * 3, "Low": [1.0] * 3, "Close": [1.0] * 3, "Volume": [100] * 3,
    }, index=idx)
    df.index.name = "date"
    df["ticker"] = ticker
    return df


@patch("src.data.build_dataset.save_parquet")
@patch("src.data.build_dataset.fetch_ohlcv")
def test_build_price_panel_concatenates_all_tickers(mock_fetch, mock_save, tmp_path):
    mock_fetch.side_effect = _fake_prices
    panel = build_price_panel(["AAPL", "MSFT"], "2024-01-01", "2024-01-05", tmp_path)
    assert set(panel["ticker"]) == {"AAPL", "MSFT"}
    assert len(panel) == 6
    assert "date" in panel.columns
    assert mock_save.call_count == 2


def test_validate_price_panel_flags_gap():
    panel = pd.DataFrame({
        "date": pd.DatetimeIndex(["2024-01-02", "2024-01-04"]),
        "ticker": ["AAPL", "AAPL"],
        "Close": [1, 2],
    })
    trading_days = pd.DatetimeIndex(["2024-01-02", "2024-01-03", "2024-01-04"])
    report = validate_price_panel(panel, ["AAPL"], trading_days)
    assert report["AAPL"] == ["2024-01-03"]


def test_validate_price_panel_empty_when_complete():
    panel = pd.DataFrame({
        "date": pd.DatetimeIndex(["2024-01-02", "2024-01-03"]),
        "ticker": ["AAPL", "AAPL"],
        "Close": [1, 2],
    })
    trading_days = pd.DatetimeIndex(["2024-01-02", "2024-01-03"])
    report = validate_price_panel(panel, ["AAPL"], trading_days)
    assert report["AAPL"] == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/data/test_build_dataset.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.data.build_dataset'`.

- [ ] **Step 3: Write `src/data/build_dataset.py`**

```python
from pathlib import Path
import pandas as pd

from src.data.prices import fetch_ohlcv
from src.data.news import fetch_company_news, articles_to_dataframe
from src.data.storage import save_parquet, find_missing_trading_days


def build_price_panel(tickers: list, start: str, end: str, prices_dir: Path) -> pd.DataFrame:
    frames = []
    for ticker in tickers:
        df = fetch_ohlcv(ticker, start, end)
        save_parquet(df, ticker, prices_dir)
        frames.append(df.reset_index())
    return pd.concat(frames, ignore_index=True)


def validate_price_panel(panel: pd.DataFrame, tickers: list, trading_days: pd.DatetimeIndex) -> dict:
    report = {}
    for ticker in tickers:
        sub = panel[panel["ticker"] == ticker].set_index("date")
        report[ticker] = find_missing_trading_days(sub, trading_days)
    return report


def build_news_archive(tickers: list, start: str, end: str, news_dir: Path, api_key: str) -> dict:
    counts = {}
    for ticker in tickers:
        articles = fetch_company_news(ticker, start, end, api_key)
        df = articles_to_dataframe(articles, ticker)
        save_parquet(df, ticker, news_dir)
        counts[ticker] = len(df)
    return counts
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/data/test_build_dataset.py -v`
Expected: PASS (3 passed).

- [ ] **Step 5: Run the full test suite**

Run: `pytest tests/ -v`
Expected: all tests pass (Tasks 1-7 combined).

- [ ] **Step 6: Commit**

```bash
git add src/data/build_dataset.py tests/data/test_build_dataset.py
git commit -m "feat: add dataset build orchestrator with gap validation and news archive"
```

---

### Task 8: Live pipeline run script + manual verification

**Files:**
- Create: `scripts/run_phase1_pipeline.py`

**Interfaces:**
- Consumes: `config.*`, `tickers.UNIVERSE`, `calendar.get_nyse_trading_days`, `build_dataset.build_price_panel`, `build_dataset.validate_price_panel`, `build_dataset.build_news_archive`, `storage.save_parquet`.
- Produces: `data/raw/prices/<TICKER>.parquet` and `data/raw/news/<TICKER>.parquet` per ticker, plus `data/processed/combined_price_panel.parquet`. This script is not covered by automated tests — it hits live network APIs (yfinance, Finnhub) and needs a real `FINNHUB_API_KEY`.

- [ ] **Step 1: Write `scripts/run_phase1_pipeline.py`**

```python
from src import config
from src.data.tickers import UNIVERSE
from src.data.calendar import get_nyse_trading_days
from src.data.build_dataset import build_price_panel, validate_price_panel, build_news_archive
from src.data.storage import save_parquet


def main():
    trading_days = get_nyse_trading_days(config.START_DATE, config.END_DATE)

    panel = build_price_panel(UNIVERSE, config.START_DATE, config.END_DATE, config.PRICES_DIR)
    gap_report = validate_price_panel(panel, UNIVERSE, trading_days)
    total_missing = sum(len(v) for v in gap_report.values())

    print(f"Trading days expected: {len(trading_days)}")
    print(f"Tickers pulled: {len(UNIVERSE)}")
    print(f"Total price rows: {len(panel)}")
    print(f"Tickers with gaps: {[t for t, v in gap_report.items() if v]}")
    print(f"Total missing (ticker, day) cells: {total_missing}")

    if not config.FINNHUB_API_KEY:
        print("FINNHUB_API_KEY not set - skipping news pull. Set it in .env to fetch headlines.")
    else:
        news_counts = build_news_archive(UNIVERSE, config.START_DATE, config.END_DATE, config.NEWS_DIR, config.FINNHUB_API_KEY)
        print(f"Headline counts per ticker: {news_counts}")
        low_coverage = [t for t, c in news_counts.items() if c < 50]
        if low_coverage:
            print(f"WARNING: low news coverage (<50 headlines over 5y) for: {low_coverage}")

    config.DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    save_parquet(panel.set_index("date"), "combined_price_panel", config.DATA_PROCESSED_DIR)
    print(f"Saved combined panel to {config.DATA_PROCESSED_DIR / 'combined_price_panel.parquet'}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Copy `.env.example` to `.env` and add a real Finnhub key**

Run: `cp .env.example .env`
Then edit `.env` and set `FINNHUB_API_KEY=<your real key>` (sign up free at finnhub.io if you don't have one).

- [ ] **Step 3: Run the live pipeline**

Run: `python scripts/run_phase1_pipeline.py`
Expected console output pattern:
```
Trading days expected: <~1750, since 2019-01-01 to 2025-12-31 spans ~7 years>
Tickers pulled: 28
Total price rows: <Tickers pulled * Trading days expected, approximately>
Tickers with gaps: []
Total missing (ticker, day) cells: 0
Headline counts per ticker: {...}
Saved combined panel to .../data/processed/combined_price_panel.parquet
```

- [ ] **Step 4: Manually verify the deliverable**

Check:
1. `Tickers with gaps: []` — if any ticker appears here, investigate that ticker's yfinance history (may be a recent IPO, delisting, or ticker symbol change) before proceeding to Phase 2.
2. `Total missing (ticker, day) cells: 0` — required for the "no gaps or misaligned dates" deliverable.
3. No ticker in the `low_coverage` warning list — if one appears, its sentiment signal in Phase 2 will be noise per the spec's "sentiment on 2 articles is noise" rule; consider swapping that ticker out of `UNIVERSE`.
4. `data/raw/prices/`, `data/raw/news/`, and `data/processed/combined_price_panel.parquet` all exist and are non-empty.

- [ ] **Step 5: Commit**

```bash
git add scripts/run_phase1_pipeline.py
git commit -m "feat: add live Phase 1 pipeline runner script"
```

(Do not commit `.env` or the contents of `data/` — add `.gitignore` entries if this repo doesn't already have one, per Step 6 below.)

- [ ] **Step 6: Add `.gitignore`**

Create `.gitignore`:

```
.env
data/raw/
data/processed/
__pycache__/
*.pyc
.pytest_cache/
```

Run:
```bash
git add .gitignore
git commit -m "chore: ignore env file and generated data artifacts"
```

---

## Self-Review Notes

- **Spec coverage:** Universe (Task 2), 5-year OHLCV incl. 2022 bear market (Task 1 config + Task 4 fetcher), news headlines aggregated per ticker (Task 6), parquet storage keyed by (ticker, date) (Task 5), no-gaps validation deliverable (Task 3 calendar + Task 7 validate_price_panel + Task 8 manual check). Point-in-time correctness is explicitly deferred to Phase 2 (noted in Global Constraints) since no features are computed here.
- **Placeholder scan:** No TBD/TODO markers; the one manual (non-pytest) task (Task 8) has concrete expected output and a concrete verification checklist instead of asserted output, because it depends on live network + a real API key neither of which are available in an automated test run.
- **Type consistency:** `save_parquet`/`load_parquet`/`find_missing_trading_days` (Task 5) are reused as-is by both prices and news in Task 7 — verified same signature used in both `build_price_panel` and `build_news_archive`. `fetch_ohlcv` (Task 4) and `fetch_company_news`/`articles_to_dataframe` (Task 6) signatures match their call sites in Task 7 exactly.

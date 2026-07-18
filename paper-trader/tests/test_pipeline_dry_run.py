"""
Full pipeline integration test on a fixture day, per Phase 7: no network calls
(news and SPY fetches are mocked to deterministic empty results - a "quiet
day" for every ticker, one of the explicitly-tested degradation paths), no
orders submitted (--dry-run), and no writes outside pytest's tmp_path.
"""
import pandas as pd
import pytest

from paper_trader import config
from paper_trader import pipeline as pipeline_mod
from paper_trader.data.run_log import read_run_log
from paper_trader.evaluation import report as report_mod
from paper_trader.execution.ledger import load_trades
from paper_trader.models import feature_store
from paper_trader.models.prediction_log import load_predictions

# Within the frozen historical matrix's raw price range (data/raw/prices goes
# through 2025-12-31), so price lookups are satisfied entirely from the
# on-disk cache with no live fetch.
FIXTURE_DATE = "2025-12-15"
FIXTURE_TICKERS = ["AAPL", "MSFT"]


@pytest.fixture
def offline_pipeline(isolated_logs, monkeypatch, tmp_path):
    # Quiet day for every ticker - no network call to the news API.
    monkeypatch.setattr(
        feature_store, "fetch_universe_news",
        lambda tickers, start, end, api_key: ({}, {t: "empty" for t in tickers}),
    )
    # No network call for the report's SPY equity-curve comparison.
    monkeypatch.setattr(report_mod, "_fetch_spy_close", lambda start, end: pd.Series(dtype=float))
    # Small, fast universe instead of all 28 tickers.
    monkeypatch.setattr(config, "get_universe", lambda: list(FIXTURE_TICKERS))
    # Keep every write inside tmp_path - never touch the real repo's data/ or reports/.
    monkeypatch.setattr(feature_store, "LIVE_PRICES_DIR", tmp_path / "live_prices")
    monkeypatch.setattr(feature_store, "LIVE_SENTIMENT_DIR", tmp_path / "live_sentiment")
    reports_dir = tmp_path / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(config, "REPORTS_DIR", reports_dir)
    return tmp_path


def test_dry_run_completes_ok_without_submitting_orders(offline_pipeline):
    result = pipeline_mod.run_pipeline(as_of_date=FIXTURE_DATE, dry_run=True)

    assert result["status"] == "ok"
    assert result["dry_run"] is True
    assert len(result["strategies"]) == len(config.STRATEGIES_CFG)


def test_dry_run_logs_predictions_for_every_universe_ticker(offline_pipeline):
    pipeline_mod.run_pipeline(as_of_date=FIXTURE_DATE, dry_run=True)

    preds = load_predictions("direction-daily")
    assert len(preds) > 0
    assert {p["ticker"] for p in preds}.issubset(set(FIXTURE_TICKERS))
    for p in preds:
        assert p["direction"] in ("up", "down")
        assert 0.5 <= p["confidence"] <= 1.0


def test_dry_run_rerun_same_date_is_idempotent(offline_pipeline):
    pipeline_mod.run_pipeline(as_of_date=FIXTURE_DATE, dry_run=True)
    pipeline_mod.run_pipeline(as_of_date=FIXTURE_DATE, dry_run=True)

    entries = [t for t in load_trades("direction-daily") if t["event"] == "entry"]
    keys = [(e["ticker"], e["entry_date"]) for e in entries]
    assert len(keys) == len(set(keys))


def test_dry_run_writes_a_run_log_even_though_nothing_traded(offline_pipeline):
    pipeline_mod.run_pipeline(as_of_date=FIXTURE_DATE, dry_run=True)

    record = read_run_log(FIXTURE_DATE)
    assert record is not None
    assert record["status"] == "ok"
    assert record["dry_run"] is True
    assert "timings" in record


def test_dry_run_generates_a_report(offline_pipeline):
    pipeline_mod.run_pipeline(as_of_date=FIXTURE_DATE, dry_run=True)
    assert (config.REPORTS_DIR / "latest.md").exists()

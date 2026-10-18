"""Fast tests for scripts/predict_news_direction.py and src/models/news_direction.decide (no network)."""
import importlib.util
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

from src.models.news_direction import decide

_SPEC = importlib.util.spec_from_file_location(
    "predict_news_direction", Path(__file__).resolve().parents[2] / "scripts" / "predict_news_direction.py")
pnd = importlib.util.module_from_spec(_SPEC)
sys.modules["predict_news_direction"] = pnd
_SPEC.loader.exec_module(pnd)


def test_decide():
    assert decide(0.70, 0.6) == "UP"
    assert decide(0.30, 0.6) == "DOWN"
    assert decide(0.55, 0.6) == "ABSTAIN"
    assert decide(float("nan"), 0.6) == "ABSTAIN"


def test_drop_incomplete_session():
    px = pd.DataFrame({"date": pd.to_datetime(["2026-10-15", "2026-10-16"]), "ticker": "A", "Close": [1, 2]})
    et = ZoneInfo("America/New_York")
    assert len(pnd.drop_incomplete_session(px, datetime(2026, 10, 16, 12, 0, tzinfo=et))) == 1
    assert len(pnd.drop_incomplete_session(px, datetime(2026, 10, 16, 17, 0, tzinfo=et))) == 2


def test_decisions_abstain_without_news_today():
    px = pd.DataFrame({"date": pd.to_datetime(["2026-10-15", "2026-10-16"] * 2), "ticker": ["A", "A", "B", "B"]})
    df = pd.DataFrame({"ticker": ["A", "B"], "date": pd.to_datetime(["2026-10-16", "2026-10-15"]),
                       "p_up": [0.9, 0.1], "news_count": [3, 1]})
    rows = {r["ticker"]: r for r in pnd.decisions(["A", "B", "C"], px, df, 0.6)}
    assert rows["A"]["decision"] == "UP"
    assert rows["B"]["decision"] == "ABSTAIN"  # its only news row is from a previous session
    assert rows["C"]["reason"] == "no price data"


def test_main_without_model_returns_2(tmp_path, monkeypatch):
    monkeypatch.setattr(pnd, "load_bundle", lambda path=None: None)
    assert pnd.main(["--offline", "--out-dir", str(tmp_path)]) == 2

import sys
from pathlib import Path

import pytest

PAPER_TRADER_ROOT = Path(__file__).resolve().parent.parent
if str(PAPER_TRADER_ROOT) not in sys.path:
    sys.path.insert(0, str(PAPER_TRADER_ROOT))

from paper_trader import config  # noqa: E402


@pytest.fixture
def isolated_logs(tmp_path, monkeypatch):
    """Redirect every logs/* path to a throwaway tmp_path so tests never touch real logs."""
    trades_dir = tmp_path / "trades"
    predictions_dir = tmp_path / "predictions"
    outcomes_dir = tmp_path / "outcomes"
    runs_dir = tmp_path / "runs"
    for d in (trades_dir, predictions_dir, outcomes_dir, runs_dir):
        d.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(config, "TRADES_DIR", trades_dir)
    monkeypatch.setattr(config, "PREDICTIONS_DIR", predictions_dir)
    monkeypatch.setattr(config, "OUTCOMES_DIR", outcomes_dir)
    monkeypatch.setattr(config, "RUNS_DIR", runs_dir)
    return tmp_path

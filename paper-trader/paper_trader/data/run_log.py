import json
from datetime import datetime, timezone

from paper_trader import config


def write_run_log(run_date: str, payload: dict) -> None:
    """
    Write logs/runs/YYYY-MM-DD.json. Callers must invoke this in a finally block
    (or equivalent) so a run always leaves a record, even on failure - silent
    failure is prohibited by design.
    """
    record = dict(payload)
    record.setdefault("date", run_date)
    record.setdefault("written_at_utc", datetime.now(timezone.utc).isoformat())
    path = config.RUNS_DIR / f"{run_date}.json"
    path.write_text(json.dumps(record, indent=2, default=str))


def read_run_log(run_date: str):
    path = config.RUNS_DIR / f"{run_date}.json"
    if not path.exists():
        return None
    return json.loads(path.read_text())


def last_run_date():
    files = sorted(config.RUNS_DIR.glob("*.json"))
    if not files:
        return None
    return files[-1].stem

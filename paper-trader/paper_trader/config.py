import os
import sys
from pathlib import Path

import yaml
from dotenv import load_dotenv

PAPER_TRADER_ROOT = Path(__file__).resolve().parent.parent
_CONFIG_PATH = PAPER_TRADER_ROOT / "config.yaml"

load_dotenv(PAPER_TRADER_ROOT.parent / ".env")

with open(_CONFIG_PATH) as f:
    _RAW = yaml.safe_load(f)

RESEARCH_REPO_ROOT = (PAPER_TRADER_ROOT / _RAW["paths"]["research_repo_root"]).resolve()
LOGS_DIR = PAPER_TRADER_ROOT / _RAW["paths"]["logs_dir"]
RUNS_DIR = LOGS_DIR / "runs"
REPORTS_DIR = PAPER_TRADER_ROOT / _RAW["paths"]["reports_dir"]
DATA_DIR = PAPER_TRADER_ROOT / _RAW["paths"]["data_dir"]

PREDICTIONS_DIR = LOGS_DIR / "predictions"
TRADES_DIR = LOGS_DIR / "trades"
OUTCOMES_DIR = LOGS_DIR / "outcomes"

# Back-compat single-file paths for the control strategy (direction-daily),
# matching the plan's literal "logs/trades.jsonl" / "logs/outcomes.jsonl" -
# every other named strategy (Phase 6) gets its own file under the *_DIR above.
TRADES_LOG = TRADES_DIR / "direction-daily.jsonl"
OUTCOMES_LOG = OUTCOMES_DIR / "direction-daily.jsonl"

for _d in (LOGS_DIR, RUNS_DIR, REPORTS_DIR, DATA_DIR, PREDICTIONS_DIR, TRADES_DIR, OUTCOMES_DIR):
    _d.mkdir(parents=True, exist_ok=True)


def predictions_log_path(strategy_name: str) -> Path:
    return PREDICTIONS_DIR / f"{strategy_name}.jsonl"


def trades_log_path(strategy_name: str) -> Path:
    return TRADES_DIR / f"{strategy_name}.jsonl"


def outcomes_log_path(strategy_name: str) -> Path:
    return OUTCOMES_DIR / f"{strategy_name}.jsonl"

# Make the research repo's `src` package importable so the paper-trader can
# wrap the existing predictor/sentiment/data code instead of duplicating it.
if str(RESEARCH_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(RESEARCH_REPO_ROOT))

MODEL_CFG = _RAW["model"]
UNIVERSE_CFG = _RAW["universe"]
STRATEGIES_CFG = _RAW["strategies"]
EXECUTION_CFG = _RAW["execution"]
MARKET_HOURS_CFG = _RAW["market_hours"]
EVALUATION_CFG = _RAW["evaluation"]

ALPACA_KEY_ID = os.environ.get("ALPACA_KEY_ID", "")
ALPACA_SECRET_KEY = os.environ.get("ALPACA_SECRET_KEY", "")
ALPACA_PAPER_BASE_URL = os.environ.get("ALPACA_PAPER_BASE_URL", "https://paper-api.alpaca.markets")
FINNHUB_API_KEY = os.environ.get("FINNHUB_API_KEY", "")


def get_universe() -> list:
    if UNIVERSE_CFG["source"] == "explicit":
        return list(UNIVERSE_CFG["explicit_tickers"])
    from src.data.tickers import UNIVERSE
    return list(UNIVERSE)


def get_sector_map() -> dict:
    from src.data.tickers import SECTOR_MAP
    return dict(SECTOR_MAP)


def get_strategy_config(name: str) -> dict:
    for cfg in STRATEGIES_CFG:
        if cfg["name"] == name:
            return cfg
    raise KeyError(f"Unknown strategy '{name}' - not present in config.yaml strategies list")


def strategy_names() -> list:
    return [cfg["name"] for cfg in STRATEGIES_CFG]


def trade_mode_strategy_names() -> list:
    return [cfg["name"] for cfg in STRATEGIES_CFG if cfg["mode"] == "trade"]

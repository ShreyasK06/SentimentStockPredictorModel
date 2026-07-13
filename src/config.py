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

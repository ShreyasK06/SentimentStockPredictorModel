"""FNSPID headline ingest (Phase 8 WP1a).

Streams the 5.7 GB ``All_external.csv`` from Hugging Face, keeps only headline-level
columns for S&P 500 tickers, and writes filtered parquet parts. The downloaded content
is treated as untrusted data: it is only parsed, never executed.

Also holds the pure helpers (alignment, dedupe, labels) used by
``scripts/phase8_build_fnspid_dataset.py`` so they can be unit tested offline.
"""
from __future__ import annotations

import hashlib
import io
import json
import re
import time
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

FNSPID_URL = (
    "https://huggingface.co/datasets/Zihan1004/FNSPID/resolve/main/Stock_news/All_external.csv"
)
KEEP_COLUMNS = ["Date", "Article_title", "Stock_symbol", "Url", "Publisher"]
SYMBOL_ALIASES = {"GOOG": "GOOGL", "FB": "META"}

PHASE8_TICKERS = [
    "AAPL", "ABBV", "AMZN", "BA", "BAC", "CAT", "COP", "CVX", "EOG", "GOOGL", "GS", "HD",
    "HON", "JNJ", "JPM", "KO", "MCD", "META", "MRK", "MS", "MSFT", "PFE", "PG", "SLB",
    "UNH", "WFC", "WMT", "XOM",
]


# ----------------------------------------------------------------------------- filter
def build_symbol_map(sp500_symbols: Iterable[str]) -> dict[str, str]:
    """FNSPID symbol -> normalized ticker (aliases GOOG->GOOGL, FB->META)."""
    mapping = {s: s for s in sp500_symbols}
    for alias, target in SYMBOL_ALIASES.items():
        mapping[alias] = target
    return mapping


def filter_chunk(chunk: pd.DataFrame, symbol_map: dict[str, str]) -> pd.DataFrame:
    """Keep headline columns for rows whose Stock_symbol is in symbol_map."""
    sym = chunk["Stock_symbol"].astype("string").str.strip().str.upper()
    keep = sym.isin(list(symbol_map)).fillna(False).to_numpy(dtype=bool)
    sub = chunk.loc[keep, KEEP_COLUMNS].copy()
    sub["fnspid_symbol"] = sym[keep].astype(str).values
    sub["ticker"] = sub["fnspid_symbol"].map(symbol_map)
    sub = sub.rename(
        columns={"Date": "raw_date", "Article_title": "headline", "Url": "url", "Publisher": "publisher"}
    )
    return sub[["raw_date", "headline", "url", "publisher", "fnspid_symbol", "ticker"]].reset_index(drop=True)


class _CountingRaw(io.RawIOBase):
    """Adapts a urllib3 response to a RawIOBase and counts decoded bytes read."""

    def __init__(self, raw):
        self._raw = raw
        self.bytes_read = 0

    def readable(self):
        return True

    def readinto(self, b):
        data = self._raw.read(len(b))
        n = len(data)
        b[:n] = data
        self.bytes_read += n
        return n


def _log(log_path: Path, msg: str) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}"
    with open(log_path, "a", encoding="utf-8") as fh:
        fh.write(line + "\n")
    print(line, flush=True)


def stream_filter_fnspid(
    out_dir: Path,
    sp500_symbols: Iterable[str],
    url: str = FNSPID_URL,
    chunksize: int = 200_000,
    chunks_per_part: int = 5,
    max_retries: int = 20,
) -> int:
    """Stream + filter. Resumable: chunks already represented in written parts are skipped
    (the stream is re-read from the start, but not re-written). Returns total rows kept."""
    import requests

    out_dir = Path(out_dir)
    parts_dir = out_dir / "parts"
    parts_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "download.log"
    progress_path = out_dir / "progress.json"
    symbol_map = build_symbol_map(sp500_symbols)

    if (out_dir / "DONE").exists():
        _log(log_path, "DONE marker present; nothing to do")
        return json.loads((out_dir / "DONE").read_text())["rows_kept"]

    for attempt in range(max_retries):
        progress = json.loads(progress_path.read_text()) if progress_path.exists() else {
            "chunks_done": 0, "rows_scanned": 0, "rows_kept": 0, "parts": 0,
        }
        _log(log_path, f"attempt {attempt}: resuming after chunk {progress['chunks_done']}")
        try:
            with requests.get(url, stream=True, timeout=(30, 120)) as resp:
                resp.raise_for_status()
                resp.raw.decode_content = True
                counting = _CountingRaw(resp.raw)
                text = io.TextIOWrapper(
                    io.BufferedReader(counting, 1 << 20), encoding="utf-8", errors="replace", newline=""
                )
                reader = pd.read_csv(
                    text, usecols=KEEP_COLUMNS, dtype=str, chunksize=chunksize,
                    engine="c", on_bad_lines="skip",
                )
                buffer: list[pd.DataFrame] = []
                chunk_idx = 0
                t0 = time.time()
                for chunk in reader:
                    chunk_idx += 1
                    if chunk_idx <= progress["chunks_done"]:
                        continue
                    progress["rows_scanned"] += len(chunk)
                    kept = filter_chunk(chunk, symbol_map)
                    progress["rows_kept"] += len(kept)
                    buffer.append(kept)
                    if len(buffer) >= chunks_per_part:
                        _flush(buffer, parts_dir, progress, chunk_idx, progress_path)
                        buffer = []
                        mb = counting.bytes_read / 1e6
                        _log(log_path, f"chunk {chunk_idx} bytes={mb:.0f}MB ({mb / max(time.time() - t0, 1):.1f} MB/s) "
                                       f"scanned={progress['rows_scanned']} kept={progress['rows_kept']}")
                if buffer:
                    _flush(buffer, parts_dir, progress, chunk_idx, progress_path)
                (out_dir / "DONE").write_text(json.dumps(progress))
                _log(log_path, f"finished: scanned={progress['rows_scanned']} kept={progress['rows_kept']}")
                return progress["rows_kept"]
        except Exception as exc:  # network drop etc.; resume
            _log(log_path, f"error: {type(exc).__name__}: {exc}; retrying")
            time.sleep(10)
    raise RuntimeError("FNSPID download failed after retries")


def _flush(buffer, parts_dir: Path, progress: dict, chunk_idx: int, progress_path: Path) -> None:
    df = pd.concat(buffer, ignore_index=True)
    part = parts_dir / f"part_{progress['parts']:04d}.parquet"
    tmp = part.with_suffix(".tmp")
    df.to_parquet(tmp, index=False)
    tmp.replace(part)
    progress["parts"] += 1
    progress["chunks_done"] = chunk_idx
    progress_path.write_text(json.dumps(progress))


def consolidate_parts(parts_dir: Path, out_path: Path) -> pd.DataFrame:
    files = sorted(Path(parts_dir).glob("part_*.parquet"))
    if not files:
        raise FileNotFoundError(f"no parts in {parts_dir}")
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    df.to_parquet(out_path, index=False)
    return df


# ------------------------------------------------------------------------ alignment
def stamped_date(raw_date: pd.Series) -> pd.Series:
    """Calendar date D as written in the Date string (first 10 chars), ignoring clock/timezone."""
    return pd.to_datetime(raw_date.astype(str).str.slice(0, 10), errors="coerce")


def conservative_effective_day(stamped: pd.Series, trading_days: pd.DatetimeIndex) -> pd.Series:
    """First NYSE trading day strictly after stamped date D."""
    days = pd.DatetimeIndex(sorted(pd.DatetimeIndex(trading_days).tz_localize(None).normalize().unique()))
    d = pd.to_datetime(stamped)
    pos = days.searchsorted(d.values, side="right")
    oob = pos >= len(days)
    out = pd.Series(days[np.where(oob, 0, pos)], index=d.index)
    return out.mask(oob | d.isna(), pd.NaT)


def naive_effective_day(raw_date: pd.Series, trading_days: pd.DatetimeIndex) -> pd.Series:
    """Diagnostic only: trust the stamped clock as UTC and use the Finnhub-style rule."""
    from src.features.trading_day import assign_effective_trading_day

    s = raw_date.astype(str).str.replace(" UTC", "", regex=False)
    ts = pd.to_datetime(s, errors="coerce")
    ok = ts.notna()
    out = pd.Series(pd.NaT, index=raw_date.index, dtype="datetime64[ns]")
    if ok.any():
        out.loc[ok] = assign_effective_trading_day(ts[ok], trading_days).values
    return out


# -------------------------------------------------------------------------- dedupe
_PUNCT = re.compile(r"[^\w\s]", flags=re.UNICODE)
_WS = re.compile(r"\s+")


def normalize_headline(text) -> str:
    if text is None or (isinstance(text, float) and np.isnan(text)):
        return ""
    return _WS.sub(" ", _PUNCT.sub("", str(text).lower())).strip()


def make_article_id(ticker: str, url, headline: str, stamped: str) -> str:
    key = f"{ticker}|{url}" if isinstance(url, str) and url else f"{ticker}|{headline}|{stamped}"
    return hashlib.sha1(key.encode("utf-8")).hexdigest()[:16]


def dedupe_articles(df: pd.DataFrame, window_days: int = 3) -> tuple[pd.DataFrame, dict]:
    """Needs columns ticker, url, headline, stamped_date. Drops (1) exact URL duplicates per
    ticker, (2) normalized-headline duplicates per ticker within `window_days` calendar days
    of the last KEPT copy (earliest kept). Returns (deduped, counts)."""
    df = df.copy()
    df["_norm"] = df["headline"].map(normalize_headline)
    df = df.sort_values(["ticker", "stamped_date"], kind="stable").reset_index(drop=True)
    n0 = len(df)

    has_url = df["url"].notna() & (df["url"].astype(str).str.len() > 0)
    dup_url = has_url & df.duplicated(["ticker", "url"], keep="first")
    df = df.loc[~dup_url].reset_index(drop=True)
    n_url = int(dup_url.sum())

    drop = np.zeros(len(df), dtype=bool)
    last_kept: dict = {}
    tickers = df["ticker"].values
    norms = df["_norm"].values
    dates = df["stamped_date"].values
    delta = np.timedelta64(window_days, "D")
    for i in range(len(df)):
        if norms[i] == "":
            continue
        key = (tickers[i], norms[i])
        prev = last_kept.get(key)
        if prev is not None and dates[i] - prev <= delta:
            drop[i] = True
        else:
            last_kept[key] = dates[i]
    n_head = int(drop.sum())
    out = df.loc[~drop].drop(columns=["_norm"]).reset_index(drop=True)
    return out, {"input": n0, "url_duplicates_removed": n_url,
                 "headline_duplicates_removed": n_head, "output": len(out)}


# --------------------------------------------------------------------------- labels
def build_labels(prices: pd.DataFrame) -> pd.DataFrame:
    """prices: columns ticker, date, Close (one row per ticker session).
    Label for date E uses only Close[E] and Close[next session]."""
    out = []
    for ticker, g in prices.sort_values(["ticker", "date"]).groupby("ticker"):
        g = g[["date", "Close"]].dropna().reset_index(drop=True)
        lab = pd.DataFrame({"ticker": ticker, "date": g["date"], "close": g["Close"]})
        lab["next_close"] = g["Close"].shift(-1)
        lab["ret_next"] = lab["next_close"] / lab["close"] - 1
        lab["ret_1d"] = lab["close"] / lab["close"].shift(1) - 1
        y = (lab["ret_next"] > 0).astype(float)
        y[lab["ret_next"].isna() | (lab["ret_next"] == 0)] = np.nan
        lab["y"] = y
        out.append(lab)
    return pd.concat(out, ignore_index=True)

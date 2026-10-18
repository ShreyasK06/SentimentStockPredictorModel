"""Phase 8 WP2: FinBERT scoring + daily news features + causal controls + labels.

    python scripts/phase8_build_features.py score --source fnspid --max-seconds 540   # repeat until done
    python scripts/phase8_build_features.py build --source fnspid      # also writes the *_naive diagnostic
    python scripts/phase8_build_features.py score --source finnhub
    python scripts/phase8_build_features.py build --source finnhub

Outputs (data/processed/phase8/):
    daily_features_{source}.parquet  ticker, date, year, y, ret_next, news_*, ctl_*
    daily_emb_{source}.npy           float32 [rows, 768] mean FinBERT [CLS] embedding, same row order
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.data.calendar import get_nyse_trading_days  # noqa: E402
from src.features.finbert_scoring import load_cache, score_cached, text_key  # noqa: E402
from src.features.market_controls import build_market_controls  # noqa: E402
from src.features.news_daily import build_daily_news_features, daily_mean_embeddings  # noqa: E402

P8 = ROOT / "data" / "processed" / "phase8"
CACHE = P8 / "finbert_cache"
PRICES_DIR = ROOT / "data" / "raw" / "prices_phase8"


def _naive_dt(s: pd.Series) -> pd.Series:
    s = pd.to_datetime(s)
    if s.dt.tz is not None:
        s = s.dt.tz_localize(None)
    return s.dt.normalize()


def load_articles(source: str) -> pd.DataFrame:
    a = pd.read_parquet(P8 / f"articles_{source}.parquet")
    a["headline"] = a["headline"].astype(str).str.strip()
    a = a[a["headline"] != ""].copy()
    for c in ("effective_day", "naive_effective_day"):
        a[c] = _naive_dt(a[c])
    a["text_key"] = a["headline"].map(text_key)
    return a


def step_score(source: str, max_seconds: float, batch_size: int):
    a = load_articles(source)
    res = score_cached(a["headline"].tolist(), CACHE, batch_size=batch_size, max_length=64,
                       max_seconds=max_seconds)
    print(res)


def load_prices() -> pd.DataFrame:
    frames = []
    for f in sorted(PRICES_DIR.glob("*.parquet")):
        d = pd.read_parquet(f).reset_index()
        d["date"] = pd.to_datetime(d["date"]).dt.tz_localize(None).dt.normalize()
        d["ticker"] = f.stem
        frames.append(d)
    return pd.concat(frames, ignore_index=True)


def _build(a, day_col, scores, key_index, emb, controls, labels, td):
    art = a[["ticker", day_col, "publisher", "text_key"]].rename(columns={day_col: "effective_day"})
    art = art.dropna(subset=["effective_day"]).merge(scores, on="text_key", how="inner")
    feats = build_daily_news_features(art, td)
    df = feats.merge(controls, on=["ticker", "date"], how="left")
    df = df.merge(labels[["ticker", "date", "y", "ret_next"]], on=["ticker", "date"], how="inner")
    df = df.dropna(subset=["y"]).sort_values(["date", "ticker"]).reset_index(drop=True)
    df["year"] = df["date"].dt.year
    e = daily_mean_embeddings(art[["ticker", "effective_day", "text_key"]], key_index, emb, df[["ticker", "date"]])
    return df, e


def step_build(source: str):
    a = load_articles(source)
    scores, emb = load_cache(CACHE, with_embeddings=True)
    keep = ~scores["text_key"].duplicated()
    scores, emb = scores[keep].reset_index(drop=True), emb[keep.to_numpy()]
    missing = int((~a["text_key"].isin(scores["text_key"])).sum())
    if missing:
        print(f"WARNING: {missing} articles not yet scored; run the score step first")
    key_index = pd.Series(np.arange(len(scores)), index=scores["text_key"]).to_dict()
    labels = pd.read_parquet(P8 / "labels.parquet")
    labels["date"] = pd.to_datetime(labels["date"])
    controls = build_market_controls(load_prices())
    td = get_nyse_trading_days("2009-01-01", (pd.Timestamp.today() + pd.Timedelta(days=30)).date().isoformat())
    variants = [("effective_day", "")] + ([("naive_effective_day", "_naive")] if source == "fnspid" else [])
    for col, suffix in variants:
        df, e = _build(a, col, scores, key_index, emb, controls, labels, td)
        df.to_parquet(P8 / f"daily_features_{source}{suffix}.parquet", index=False)
        np.save(P8 / f"daily_emb_{source}{suffix}.npy", e)
        print(source + suffix, df.shape, df.groupby("year").size().to_dict())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["score", "build"])
    ap.add_argument("--source", choices=["fnspid", "finnhub"], required=True)
    ap.add_argument("--max-seconds", type=float, default=540)
    ap.add_argument("--batch-size", type=int, default=64)
    args = ap.parse_args()
    if args.step == "score":
        step_score(args.source, args.max_seconds, args.batch_size)
    else:
        step_build(args.source)


if __name__ == "__main__":
    main()

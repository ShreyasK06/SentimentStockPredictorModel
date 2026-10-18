"""Phase 8 WP1a: FNSPID articles + next-day labels + coverage/timestamp reports.

Steps (run in order, each idempotent):
    python scripts/phase8_build_fnspid_dataset.py prices     # yfinance -> data/raw/prices_phase8
    python scripts/phase8_build_fnspid_dataset.py articles   # FNSPID parts -> articles_fnspid.parquet
    python scripts/phase8_build_fnspid_dataset.py labels     # prices -> labels.parquet
    python scripts/phase8_build_fnspid_dataset.py reports    # coverage + timestamp audit

FNSPID inputs are the filtered parquet parts written by src.data.fnspid.stream_filter_fnspid
from BOTH Stock_news/All_external.csv (data/raw/fnspid/parts) and
Stock_news/nasdaq_exteral_data.csv (data/raw/fnspid_nasdaq/parts).
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.data.calendar import get_nyse_trading_days  # noqa: E402
from src.data.fnspid import (  # noqa: E402
    PHASE8_TICKERS, build_labels, conservative_effective_day, dedupe_articles, make_article_id,
    naive_effective_day, normalize_headline, stamped_date,
)

PRICES_DIR = ROOT / "data" / "raw" / "prices_phase8"
OUT_DIR = ROOT / "data" / "processed" / "phase8"
REPORT_DIR = ROOT / "reports" / "phase8"
PART_DIRS = [ROOT / "data" / "raw" / "fnspid" / "parts", ROOT / "data" / "raw" / "fnspid_nasdaq" / "parts"]
PRICE_START = "2009-01-01"
TEST_START = "2022-01-01"


def _md(df) -> str:
    """Minimal markdown table (avoids the optional `tabulate` dependency)."""
    df = df.reset_index()
    head = "| " + " | ".join(map(str, df.columns)) + " |"
    sep = "|" + "---|" * len(df.columns)
    body = ["| " + " | ".join(map(str, r)) + " |" for r in df.itertuples(index=False)]
    return "\n".join([head, sep, *body])


def step_prices():
    from src.data.prices import fetch_ohlcv

    PRICES_DIR.mkdir(parents=True, exist_ok=True)
    end = date.today().isoformat()
    for t in PHASE8_TICKERS + ["SPY"]:
        df = fetch_ohlcv(t, PRICE_START, end)
        df.to_parquet(PRICES_DIR / f"{t}.parquet")
        print(t, len(df), df.index.min().date(), df.index.max().date())


def load_prices() -> pd.DataFrame:
    frames = []
    for t in PHASE8_TICKERS + ["SPY"]:
        d = pd.read_parquet(PRICES_DIR / f"{t}.parquet").reset_index()
        d["date"] = pd.to_datetime(d["date"]).dt.tz_localize(None).dt.normalize()
        d["ticker"] = t
        frames.append(d)
    return pd.concat(frames, ignore_index=True)


def trading_days() -> pd.DatetimeIndex:
    return get_nyse_trading_days(PRICE_START, (pd.Timestamp.today() + pd.Timedelta(days=30)).date().isoformat())


def load_parts() -> pd.DataFrame:
    frames = []
    for d in PART_DIRS:
        files = sorted(d.glob("part_*.parquet"))
        if not files:
            print(f"WARNING: no parts in {d}")
            continue
        df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
        df["fnspid_file"] = d.parent.name
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def step_articles():
    raw = load_parts()
    raw = raw[raw["ticker"].isin(PHASE8_TICKERS)].copy()
    raw = raw[raw["headline"].fillna("").astype(str).str.strip() != ""]
    raw["stamped_date"] = stamped_date(raw["raw_date"])
    raw = raw[raw["stamped_date"].notna()]
    n_raw = len(raw)
    dedup, counts = dedupe_articles(raw, window_days=3)
    td = trading_days()
    dedup["effective_day"] = conservative_effective_day(dedup["stamped_date"], td).values
    dedup["naive_effective_day"] = naive_effective_day(dedup["raw_date"], td).values
    dedup["article_id"] = [make_article_id(t, u, h, str(s)) for t, u, h, s in
                           zip(dedup["ticker"], dedup["url"], dedup["headline"], dedup["stamped_date"])]
    dedup["source"] = "fnspid"
    dedup["alignment"] = "conservative_date"
    cols = ["article_id", "ticker", "headline", "publisher", "url", "source", "raw_date", "stamped_date",
            "effective_day", "naive_effective_day", "alignment", "fnspid_file"]
    out = dedup[cols].dropna(subset=["effective_day"]).reset_index(drop=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUT_DIR / "articles_fnspid.parquet", index=False)
    counts["raw_28_tickers"] = n_raw
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / "wp1_dedupe_counts.json").write_text(json.dumps(counts, indent=1))
    print(counts, "->", len(out))


def step_labels():
    prices = load_prices()
    lab = build_labels(prices[["ticker", "date", "Close"]])
    lab.to_parquet(OUT_DIR / "labels.parquet", index=False)
    print(lab.groupby("ticker").size().describe())


def step_reports():
    art = pd.read_parquet(OUT_DIR / "articles_fnspid.parquet")
    lab = pd.read_parquet(OUT_DIR / "labels.parquet")
    lab = lab[lab["ticker"].isin(PHASE8_TICKERS)]
    art["year"] = art["effective_day"].dt.year
    fn_end = art["stamped_date"].max()
    fn_start = art["stamped_date"].min()

    days = art.drop_duplicates(["ticker", "effective_day"])
    lab_in = lab[(lab["date"] >= "2009-01-01") & (lab["date"] <= fn_end)].copy()
    lab_in["year"] = lab_in["date"].dt.year
    tot = lab_in.groupby("year").size()
    newsdays = days.groupby("year").size().reindex(tot.index).fillna(0)
    by_year = pd.DataFrame({"articles": art.groupby("year").size().reindex(tot.index).fillna(0).astype(int),
                            "ticker_days_with_news": newsdays.astype(int),
                            "ticker_trading_days": tot,
                            "pct_ticker_days_with_news": (100 * newsdays / tot).round(1)})
    yt = art.groupby(["year", "ticker"]).size().unstack(fill_value=0)
    yt.to_csv(REPORT_DIR / "wp1_coverage_by_year_ticker.csv")

    # frozen rule: dev_start = 2012 unless a year from 2012 has < 20% ticker-days with news
    dev_start_year = 2012
    for y in range(2012, 2022):
        if y in by_year.index and by_year.loc[y, "pct_ticker_days_with_news"] < 20:
            dev_start_year = y + 1
    cfg = {"dev_start": f"{dev_start_year}-01-01", "test_start": TEST_START,
           "fnspid_start": str(fn_start.date()), "fnspid_end": str(fn_end.date()), "tickers": PHASE8_TICKERS,
           "rule": "dev_start = 2012 unless a year >= 2012 has < 20% ticker-days with news; then the year after "
                   "the last such year before 2022"}
    (REPORT_DIR / "wp1_dataset_config.json").write_text(json.dumps(cfg, indent=1))

    # cross-period duplicate headlines (normalized text)
    art["norm"] = art["headline"].map(normalize_headline)
    art["period"] = np.where(art["effective_day"] < "2021-01-01", "train",
                             np.where(art["effective_day"] < TEST_START, "val", "test"))
    per = art.groupby("norm")["period"].nunique()
    multi = per[per > 1]
    top = art[art["norm"].isin(multi.index)].groupby("norm").size().sort_values(ascending=False).head(15)

    # timestamp audit
    hhmmss = art["raw_date"].astype(str).str.slice(11, 19)
    midnight = float((hhmmss == "00:00:00").mean())
    hour = pd.to_numeric(art["raw_date"].astype(str).str.slice(11, 13), errors="coerce")
    hist = hour[hhmmss != "00:00:00"].value_counts().sort_index()
    naive_same = float((art["naive_effective_day"] == art["effective_day"]).mean())

    lines = ["# Phase 8 WP1: FNSPID coverage (28 fixed tickers)", "",
             f"FNSPID stamped dates: {fn_start.date()} .. {fn_end.date()}. "
             f"Dedupe: {json.loads((REPORT_DIR / 'wp1_dedupe_counts.json').read_text())}", "",
             f"Chosen dev_start: **{cfg['dev_start']}** (rule: {cfg['rule']}). Test starts {TEST_START}.", "",
             "## Per year (effective trading day)", "", _md(by_year), "",
             "## Articles per ticker (all years)", "",
             _md(art.groupby("ticker").size().sort_values(ascending=False).to_frame("articles").T), "",
             "## Source file mix per year", "",
             _md(art.groupby(["year", "fnspid_file"]).size().unstack(fill_value=0)), "",
             "## Top publishers", "", _md(art["publisher"].fillna("(none)").value_counts().head(10).to_frame()), "",
             "## Headlines (normalized) appearing in more than one period (train<2021, val 2021, test>=2022)", "",
             f"{len(multi)} distinct normalized headlines span >1 period. Top repeated:", "",
             _md(top.to_frame("rows")), ""]
    (REPORT_DIR / "wp1_coverage.md").write_text("\n".join(lines), encoding="utf-8")

    audit = ["# Phase 8 WP1: FNSPID timestamp audit", "",
             f"Share of articles stamped exactly 00:00:00: **{midnight:.1%}** (date-only stamps).", "",
             "Hour-of-day histogram of non-midnight stamps (as written, labelled 'UTC'):", "",
             _md(hist.to_frame("articles").T), "",
             f"Share where naive (trust-the-clock) effective day equals the conservative one: {naive_same:.1%}.", "",
             "Conclusion: clock times are unreliable (most rows are date-only, and the planning probe found "
             "intraday-specific Benzinga headlines stamped at impossible ET times), so headline numbers use the "
             "frozen conservative rule: article stamped on date D -> effective trading day = first NYSE "
             "session strictly after D; label = sign(Close[E+1]/Close[E]-1)."]
    (REPORT_DIR / "wp1_timestamp_audit.md").write_text("\n".join(audit), encoding="utf-8")
    print(by_year.to_string())
    print(cfg)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("steps", nargs="+", choices=["prices", "articles", "labels", "reports"])
    args = ap.parse_args()
    for s in args.steps:
        {"prices": step_prices, "articles": step_articles, "labels": step_labels, "reports": step_reports}[s]()


if __name__ == "__main__":
    main()

"""Export the data the site's dashboard shows, one entry per ticker:

  * the Phase 8 next-day direction call (UP / DOWN / ABSTAIN) and its calibrated probability of a rise,
  * recent headlines, each scored by the fine-tuned sentiment model, with an overall news tone,
  * recent closing prices.

    python scripts/export_dashboard_data.py                  # cached news + a live tail from Finnhub, live prices
    python scripts/export_dashboard_data.py --offline        # cached articles and prices only
    python scripts/export_dashboard_data.py --tickers AAPL,JPM --news-days 14

Writes site/public/data/dashboard.json. The model, feature building and decision rule come from
scripts/predict_news_direction.py, so the dashboard and that script always agree. The Finnhub key is read from .env
and never printed. Re-run it, commit the JSON, and the site shows the new numbers.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import predict_news_direction as pnd  # noqa: E402

OUT = ROOT / "site" / "public" / "data" / "dashboard.json"
HEADLINE_DAYS = 7  # news window for the tone and the headline list
SCORED_PER_TICKER = 60  # most recent unique headlines scored per ticker
PRICE_SESSIONS = 90

COMPANIES = {
    "AAPL": ("Apple", "Technology"), "MSFT": ("Microsoft", "Technology"), "GOOGL": ("Alphabet", "Technology"),
    "META": ("Meta Platforms", "Technology"), "AMZN": ("Amazon", "Consumer"),
    "JPM": ("JPMorgan Chase", "Financials"), "BAC": ("Bank of America", "Financials"),
    "GS": ("Goldman Sachs", "Financials"), "MS": ("Morgan Stanley", "Financials"), "WFC": ("Wells Fargo", "Financials"),
    "XOM": ("Exxon Mobil", "Energy"), "CVX": ("Chevron", "Energy"), "COP": ("ConocoPhillips", "Energy"),
    "EOG": ("EOG Resources", "Energy"), "SLB": ("Schlumberger", "Energy"),
    "JNJ": ("Johnson & Johnson", "Health care"), "PFE": ("Pfizer", "Health care"), "MRK": ("Merck", "Health care"),
    "ABBV": ("AbbVie", "Health care"), "UNH": ("UnitedHealth", "Health care"),
    "BA": ("Boeing", "Industrials"), "CAT": ("Caterpillar", "Industrials"), "HON": ("Honeywell", "Industrials"),
    "WMT": ("Walmart", "Consumer"), "HD": ("Home Depot", "Consumer"), "MCD": ("McDonald's", "Consumer"),
    "KO": ("Coca-Cola", "Consumer"), "PG": ("Procter & Gamble", "Consumer"),
}


def clean_headline(text: str) -> str:
    """Cached headlines lost some punctuation to encoding errors ("isn�t"); restore apostrophes, drop the rest."""
    text = re.sub(r"(?<=\w)�(?=\w)", "'", text).replace("�", "")
    return re.sub(r"\s+", " ", text).strip()


def fetch_live(tickers: list[str], start: str, end: str) -> pd.DataFrame:
    """Recent Finnhub company news in the project's article schema (raises if there is no key or no network)."""
    from dotenv import load_dotenv

    from src.data.calendar import get_nyse_trading_days
    from src.data.finnhub_chunked import fetch_company_news_chunked, to_articles_frame

    load_dotenv(ROOT / ".env")
    key = os.getenv("FINNHUB_API_KEY", "")
    days = get_nyse_trading_days((date.fromisoformat(start) - timedelta(days=10)).isoformat(),
                                 (date.fromisoformat(end) + timedelta(days=14)).isoformat())
    frames = []
    for t in tickers:
        raw = fetch_company_news_chunked(t, start, end, key, chunk_days=1, sleep_seconds=1.1)
        frames.append(to_articles_frame(raw, t, days))
    frames = [f for f in frames if len(f)]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def load_articles(tickers: list[str], start: str, offline: bool) -> tuple[pd.DataFrame, str]:
    """Cached articles inside the window, plus a live tail when online. Unlike predict_news_direction's offline
    path this never scores the whole cached year."""
    notes, cached = [], pd.DataFrame()
    if pnd.CACHED_ARTICLES.exists():
        cached = pd.read_parquet(pnd.CACHED_ARTICLES)
        cached = cached[cached["ticker"].isin(tickers)].copy()
        cached["published_at"] = pd.to_datetime(cached["published_at"], utc=True)
        cached = cached[cached["published_at"] >= pd.Timestamp(start, tz="UTC")]
        if len(cached):
            notes.append(f"cached news through {cached['published_at'].max().date()}")
    live = pd.DataFrame()
    if not offline:
        from_day = start if cached.empty else (cached["published_at"].max().date() - timedelta(days=1)).isoformat()
        try:
            live = fetch_live(tickers, from_day, date.today().isoformat())
            notes.append(f"live Finnhub news from {from_day}")
        except Exception as exc:  # no key, network, rate limit
            notes.append(f"live news unavailable ({type(exc).__name__})")
    frames = [f for f in (cached, live) if len(f)]
    if not frames:
        return pd.DataFrame(), "; ".join(notes) or "no news"
    art = pd.concat(frames, ignore_index=True)
    art["published_at"] = pd.to_datetime(art["published_at"], utc=True)
    art = art.drop_duplicates(subset=["ticker", "headline", "published_at"]).reset_index(drop=True)
    return art, "; ".join(notes)


def json_safe(obj):
    """NaN and infinity are not valid JSON (browsers reject the file); write them as null."""
    if isinstance(obj, float):
        return obj if np.isfinite(obj) else None
    if isinstance(obj, dict):
        return {k: json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_safe(v) for v in obj]
    if isinstance(obj, np.generic):
        return json_safe(obj.item())
    return obj


def next_session(session: str) -> str | None:
    from src.data.calendar import get_nyse_trading_days

    days = get_nyse_trading_days(session, (date.fromisoformat(session) + timedelta(days=10)).isoformat())
    later = [d for d in days if d > pd.Timestamp(session)]
    return str(later[0].date()) if later else None


def price_block(px: pd.DataFrame) -> dict | None:
    if px is None or px.empty:
        return None
    px = px.sort_values("date")
    col = "Close" if "Close" in px.columns else "close"
    close = px[col].astype(float).to_numpy()
    dates = px["date"].dt.strftime("%Y-%m-%d").tolist()

    def change(n):
        return round(float(close[-1] / close[-1 - n] - 1), 4) if len(close) > n else None

    tail = slice(-PRICE_SESSIONS, None)
    return {"last": round(float(close[-1]), 2), "as_of": dates[-1], "change_1d": change(1), "change_5d": change(5),
            "series": [[d, round(float(c), 2)] for d, c in zip(dates[tail], close[tail])]}


def score_headlines(texts: list[str]) -> np.ndarray:
    """(n, 3) probabilities in [negative, neutral, positive] order from the fine-tuned sentiment model."""
    from src.models import sentiment_ft

    probs = sentiment_ft.predict_proba(texts)
    return probs[["negative", "neutral", "positive"]].to_numpy(dtype=float)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tickers", default=",".join(pnd.TICKERS))
    ap.add_argument("--news-days", type=int, default=30)
    ap.add_argument("--headlines", type=int, default=8, help="headlines listed per ticker")
    ap.add_argument("--offline", action="store_true")
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args(argv)
    tickers = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]

    bundle = pnd.load_bundle()
    if bundle is None:
        print(f"No model at {pnd.MODEL_PATH}. Run `python scripts/run_phase8_models.py cv` then `final` first.")
        return 2

    start = (date.today() - timedelta(days=args.news_days)).isoformat()
    articles, news_note = load_articles(tickers, start, args.offline)
    prices = pnd.drop_incomplete_session(pnd.get_prices(tickers, args.offline))
    # Yahoo returns today's row with a volume but blank prices until the close is final. Treating that row as a
    # session would feed the model empty price features, so keep only sessions that have a real close.
    if len(prices):
        prices = prices.dropna(subset=["Close"])
    print(f"news: {news_note} | {len(articles)} articles | prices for {prices['ticker'].nunique() if len(prices) else 0} tickers")

    primary, thr = bundle["selection"]["primary_family"], float("nan")
    df = pd.DataFrame()
    if len(articles) and len(prices):
        df, emb = pnd.build_rows(articles, prices)
        if len(df):
            p, thr, primary = pnd.predict(bundle, df, emb)
            df = df.assign(p_up=p)
    decisions = {r["ticker"]: r for r in pnd.decisions(tickers, prices, df, thr)}

    # Headlines: the most recent unique ones per ticker inside the news window, scored by the fine-tuned model.
    cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=HEADLINE_DAYS)
    picked: dict[str, pd.DataFrame] = {}
    if len(articles):
        recent = articles[articles["published_at"] >= cutoff].copy()
        recent["headline"] = recent["headline"].astype(str).map(clean_headline)
        recent = recent[recent["headline"] != ""].sort_values("published_at", ascending=False)
        for t in tickers:
            picked[t] = recent[recent["ticker"] == t].drop_duplicates("headline").head(SCORED_PER_TICKER)
    texts = sorted({h for d in picked.values() for h in d["headline"]})
    probs_by_text: dict[str, np.ndarray] = {}
    if texts:
        print(f"scoring {len(texts)} headlines with the fine-tuned sentiment model")
        probs_by_text = dict(zip(texts, score_headlines(texts)))

    entries = []
    for t in tickers:
        name, sector = COMPANIES.get(t, (t, "Other"))
        d = decisions.get(t, {"decision": "ABSTAIN", "reason": "no data"})
        outlook = {"call": d.get("decision", "ABSTAIN"), "p_up": d.get("p_up_calibrated"), "confidence": d.get("confidence"),
                   "threshold": d.get("threshold"), "session": d.get("session"), "articles_on_session": d.get("n_articles"),
                   "reason": d.get("reason", "")}
        outlook["target_session"] = next_session(outlook["session"]) if outlook["session"] else None

        news_df = picked.get(t, pd.DataFrame())
        counts = {"negative": 0, "neutral": 0, "positive": 0}
        tone, items = None, []
        if len(news_df):
            pr = np.stack([probs_by_text[h] for h in news_df["headline"]])
            top = pr.argmax(1)
            for i, label in enumerate(("negative", "neutral", "positive")):
                counts[label] = int((top == i).sum())
            tone = round(float((pr[:, 2] - pr[:, 0]).mean()), 4)
            for (_, row), probs in list(zip(news_df.iterrows(), pr))[: args.headlines]:
                items.append({"title": row["headline"], "source": str(row.get("publisher") or ""),
                              "published": row["published_at"].isoformat(), "url": str(row.get("url") or ""),
                              "probs": [round(float(x), 4) for x in probs]})
        entries.append({
            "symbol": t, "name": name, "sector": sector,
            "price": price_block(prices[prices["ticker"] == t]) if len(prices) else None,
            "outlook": outlook,
            "news": {"window_days": HEADLINE_DAYS, "scored": int(len(news_df)), "tone": tone, "counts": counts, "headlines": items},
        })

    result = {"generated_at": datetime.now(timezone.utc).isoformat(), "model": primary, "news_source": news_note,
              "target": "sign(Close[next session] / Close[session] - 1)", "tickers": entries}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(json_safe(result), separators=(",", ":"), allow_nan=False), encoding="utf-8")
    calls = pd.Series([e["outlook"]["call"] for e in entries]).value_counts().to_dict()
    print(f"wrote {out} ({out.stat().st_size / 1024:.0f} KB) | calls: {calls}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

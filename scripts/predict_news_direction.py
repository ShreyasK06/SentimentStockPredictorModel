"""Phase 8 live inference: latest Finnhub news -> FinBERT -> next-day UP / DOWN / ABSTAIN per ticker.

    python scripts/predict_news_direction.py                       # 28 tickers, last 30 days of news
    python scripts/predict_news_direction.py --tickers AAPL,JPM --news-days 20
    python scripts/predict_news_direction.py --offline             # cached Finnhub articles + local prices only

Decision row = the last COMPLETED NYSE session E with a close price. Features use only news whose
effective trading day <= E (Finnhub epoch; before 16:00 ET -> same session) and prices up to Close[E].
A ticker with no article whose effective day is E is ABSTAIN (no news that session = not an eligible
day, exactly as in training). The prediction is for Close[next session] vs Close[E].

Graceful degradation: no FINNHUB_API_KEY / network error -> uses data/processed/phase8/articles_finnhub.parquet
if present, else every ticker ABSTAIN with a reason; no model file -> exit code 2 with instructions.
Research output only, not financial advice.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.models.news_direction import decide  # noqa: E402

MODEL_PATH = ROOT / "models" / "phase8" / "final_models.joblib"
CACHED_ARTICLES = ROOT / "data" / "processed" / "phase8" / "articles_finnhub.parquet"
PRICES_DIR = ROOT / "data" / "raw" / "prices_phase8"
OUT_DIR = ROOT / "reports" / "phase8" / "latest"
TICKERS = ("AAPL ABBV AMZN BA BAC CAT COP CVX EOG GOOGL GS HD HON JNJ JPM KO MCD META MRK MS "
           "MSFT PFE PG SLB UNH WFC WMT XOM").split()
DISCLAIMER = ("Research output only, not financial advice. See docs/phase8_results.md for the measured "
              "held-out accuracy, coverage and baselines of this model.")


def load_bundle(path: Path = MODEL_PATH):
    if not path.exists():
        return None
    import joblib
    return joblib.load(path)


def get_articles(tickers, start, end, offline):
    """Return (articles DataFrame in the Finnhub article schema, source note)."""
    from src.data.calendar import get_nyse_trading_days
    td = get_nyse_trading_days((date.fromisoformat(start) - timedelta(days=10)).isoformat(),
                               (date.fromisoformat(end) + timedelta(days=14)).isoformat())
    note = "offline"
    if not offline:
        try:
            from dotenv import load_dotenv
            from src.data.finnhub_chunked import FinnhubUnavailable, fetch_company_news_chunked, to_articles_frame
            load_dotenv(ROOT / ".env")
            key = os.getenv("FINNHUB_API_KEY", "")
            try:
                frames = []
                for t in tickers:
                    raw = fetch_company_news_chunked(t, start, end, key, chunk_days=1, sleep_seconds=1.1)
                    frames.append(to_articles_frame(raw, t, td))
                frames = [f for f in frames if len(f)]
                df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
                return df, "finnhub_live"
            except FinnhubUnavailable as exc:
                note = f"finnhub unavailable ({exc})"
        except Exception as exc:  # network, import, etc.
            note = f"finnhub error ({type(exc).__name__})"
    if CACHED_ARTICLES.exists():
        df = pd.read_parquet(CACHED_ARTICLES)
        return df[df["ticker"].isin(tickers)], f"{note}; using cached {CACHED_ARTICLES.name}"
    return pd.DataFrame(), f"{note}; no cached articles"


def get_prices(tickers, offline):
    frames = []
    for t in list(tickers) + ["SPY"]:
        d = None
        if not offline:
            try:
                from src.data.prices import fetch_ohlcv
                d = fetch_ohlcv(t, (date.today() - timedelta(days=200)).isoformat(), date.today().isoformat())
            except Exception:
                d = None
        if d is None and (PRICES_DIR / f"{t}.parquet").exists():
            d = pd.read_parquet(PRICES_DIR / f"{t}.parquet")
        if d is None:
            continue
        d = d.reset_index()
        d["date"] = pd.to_datetime(d["date"]).dt.tz_localize(None).dt.normalize()
        d["ticker"] = t
        frames.append(d)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def drop_incomplete_session(prices: pd.DataFrame, now_et: datetime | None = None) -> pd.DataFrame:
    """Drop today's bar if the session has not closed yet (before 16:15 ET)."""
    if prices.empty:
        return prices
    now_et = now_et or datetime.now(timezone.utc).astimezone(ZoneInfo("America/New_York"))
    if (now_et.hour, now_et.minute) < (16, 15):
        prices = prices[prices["date"] < pd.Timestamp(now_et.date())]
    return prices


def build_rows(articles, prices):
    from src.data.calendar import get_nyse_trading_days
    from src.features.finbert_scoring import load_finbert, score_and_embed, text_key
    from src.features.market_controls import build_market_controls
    from src.features.news_daily import build_daily_news_features, daily_mean_embeddings

    a = articles.copy()
    a["headline"] = a["headline"].astype(str).str.strip()
    a = a[a["headline"] != ""]
    eff = pd.to_datetime(a["effective_day"])
    a["effective_day"] = (eff.dt.tz_localize(None) if eff.dt.tz is not None else eff).dt.normalize()
    a["text_key"] = a["headline"].map(text_key)
    uniq = a.drop_duplicates("text_key")
    tok, model = load_finbert()
    probs, emb = score_and_embed(uniq["headline"].tolist(), 64, 64, tok, model)
    sc = pd.DataFrame({"text_key": uniq["text_key"].values, "positive": probs[:, 0], "negative": probs[:, 1],
                       "neutral": probs[:, 2]})
    sc["score"] = sc["positive"] - sc["negative"]
    art = a[["ticker", "effective_day", "publisher", "text_key"]].merge(sc, on="text_key")
    td = get_nyse_trading_days("2000-01-01", (date.today() + timedelta(days=30)).isoformat())
    feats = build_daily_news_features(art, td)
    ctl = build_market_controls(prices)
    df = feats.merge(ctl, on=["ticker", "date"], how="inner")
    key_index = {k: i for i, k in enumerate(sc["text_key"])}
    e = daily_mean_embeddings(art[["ticker", "effective_day", "text_key"]], key_index, emb, df[["ticker", "date"]])
    return df, e


def predict(bundle, df, emb):
    primary = bundle["selection"]["primary_family"]
    arts = bundle["artifacts"]
    if primary == "c_ensemble":
        p = np.mean([arts[f + "__cal"].transform(arts[f].predict(df, emb)) for f in ("a_logit", "a_lgbm", "b_emb")],
                    axis=0)
    else:
        p = arts[primary].predict(df, emb)
    p = arts[primary + "__cal"].transform(p)
    return p, float(arts[primary + "__thr"]["threshold"]), primary


def decisions(tickers, prices, df, thr):
    rows = []
    for t in tickers:
        px = prices[prices["ticker"] == t] if len(prices) else pd.DataFrame()
        if px.empty:
            rows.append({"ticker": t, "decision": "ABSTAIN", "reason": "no price data"})
            continue
        last = px["date"].max()
        row = df[(df["ticker"] == t) & (df["date"] == last)] if len(df) else pd.DataFrame()
        if row.empty:
            rows.append({"ticker": t, "session": str(last.date()), "decision": "ABSTAIN",
                         "reason": "no news with effective day = last completed session"})
            continue
        pu = float(row["p_up"].iloc[0])
        dec = decide(pu, thr)
        rows.append({"ticker": t, "session": str(last.date()), "decision": dec, "p_up_calibrated": round(pu, 4),
                     "confidence": round(max(pu, 1 - pu), 4), "threshold": round(thr, 4),
                     "n_articles": int(row["news_count"].iloc[0]),
                     "reason": "confidence below validation threshold" if dec == "ABSTAIN" else ""})
    return rows


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tickers", default=",".join(TICKERS))
    ap.add_argument("--news-days", type=int, default=30)
    ap.add_argument("--offline", action="store_true")
    ap.add_argument("--out-dir", default=str(OUT_DIR))
    args = ap.parse_args(argv)
    tickers = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]
    bundle = load_bundle()
    if bundle is None:
        print(f"No model at {MODEL_PATH}. Run `python scripts/run_phase8_models.py cv` then `final` first.")
        return 2
    end = date.today().isoformat()
    start = (date.today() - timedelta(days=args.news_days)).isoformat()
    articles, news_note = get_articles(tickers, start, end, args.offline)
    prices = drop_incomplete_session(get_prices(tickers, args.offline))
    primary = bundle["selection"]["primary_family"]
    df, thr = pd.DataFrame(), float("nan")
    if len(articles) and len(prices):
        df, emb = build_rows(articles, prices)
        if len(df):
            p, thr, primary = predict(bundle, df, emb)
            df = df.assign(p_up=p)
    rows = decisions(tickers, prices, df, thr)
    result = {"generated_at": datetime.now(timezone.utc).isoformat(), "model": primary, "news_source": news_note,
              "target": "sign(Close[next session] / Close[session] - 1)", "disclaimer": DISCLAIMER,
              "predictions": rows}
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "predictions.json").write_text(json.dumps(result, indent=1))
    pd.DataFrame(rows).to_csv(out / "predictions.csv", index=False)
    print(f"model={primary} news={news_note}")
    print(pd.DataFrame(rows).to_string(index=False))
    print(DISCLAIMER)
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Forecast the next 7 trading days from the latest news and prices (Phase 7 TFT).

Pipeline
    latest Finnhub news (last N days) -> FinBERT scores -> per-ticker panel with
    up-to-date prices -> TFT quantile forecast per ticker -> JSON + CSV.

Examples
    python scripts/predict_latest.py
    python scripts/predict_latest.py --tickers AAPL,JPM --news-days 7
    python scripts/predict_latest.py --ckpt models/phase7/smoke/ckpt/<key>/best.ckpt --offline

Checkpoint resolution (when --ckpt is not given)
    reports/phase7/frozen_d.json -> feature set + hparams -> models/phase7/full/*.json
    info with the same feature_set / hparams and fold F3 -> ckpt_path.
    Missing -> exit code 2 with instructions (no traceback).

Graceful degradation (never a traceback on missing keys / network)
    * no FINNHUB_API_KEY or a Finnhub error  -> neutral sentiment (source "none")
    * yfinance failure                        -> local data/raw/prices (price_as_of recorded)
    * FinBERT failure                         -> neutral sentiment
    * --offline                               -> only local prices and scored news

The quantile band for cumulative returns is an APPROXIMATION (see
``src/models/tft.py``): median = sum of daily q50, spread from a normal
assumption on independent daily steps, band = median +/- 1.2816 * sigma_cum.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.tft_dataset import (  # noqa: E402
    HORIZONS,
    TARGET_COLUMNS,
    build_calendar_features,
    build_ticker_panel,
    finalize_panel,
)
from src.models import tft  # noqa: E402

logger = logging.getLogger("predict_latest")

N_FUTURE = tft.MAX_PREDICTION_LENGTH
Z_80 = 1.2816  # one-sided z for the 10% / 90% band
NEWS_COLUMNS = ["published_at", "positive", "negative", "neutral", "sentiment_score"]

DISCLAIMER = (
    "Research output only, not financial advice. Held-out accuracy of this model did not "
    "demonstrably beat a majority-class baseline (see docs/phase7_tft_results.md). "
    "Forecast bands are approximate."
)
TRAINING_NOTE = (
    "Model trained on data through 2022-12-31 and early-stopped on 2023 (fold F3); it is not "
    "refit on newer data, so it may be stale for current market regimes."
)
CKPT_MISSING_MSG = (
    "No trained checkpoint found. Run `python scripts/run_phase7_tft.py --stage d --phase cv` "
    "first (this writes reports/phase7/frozen_d.json and the F3 checkpoints), or pass --ckpt "
    "<path/to/best.ckpt>."
)


class CheckpointNotFound(Exception):
    pass


# ----------------------------------------------------------------------------
# Checkpoint resolution
# ----------------------------------------------------------------------------

def _hp_equal(a: dict, b: dict) -> bool:
    a = {**tft.DEFAULT_HPARAMS, **a}
    b = {**tft.DEFAULT_HPARAMS, **b}
    return set(a) == set(b) and all(math.isclose(float(a[k]), float(b[k])) for k in a)


def resolve_checkpoint(reports_dir: Path | None = None, models_dir: Path | None = None) -> tuple[Path, dict]:
    """frozen_d.json -> matching F3 info JSON -> (ckpt_path, info). Raises CheckpointNotFound."""
    reports_dir = Path(reports_dir or tft.REPORTS_DIR)
    models_dir = Path(models_dir or (tft.MODELS_DIR / "full"))
    frozen_path = reports_dir / "frozen_d.json"
    if not frozen_path.exists():
        raise CheckpointNotFound(f"{frozen_path} not found. {CKPT_MISSING_MSG}")
    frozen = json.loads(frozen_path.read_text())
    fs, hp = frozen["feature_set"], frozen["hparams"]
    matches = []
    for p in sorted(models_dir.glob("*.json")):
        try:
            info = json.loads(p.read_text())
        except (OSError, ValueError):
            continue
        if info.get("fold") != "F3" or info.get("feature_set") != fs or "ckpt_path" not in info:
            continue
        if not _hp_equal(info.get("hparams", {}), hp):
            continue
        if Path(info["ckpt_path"]).exists():
            matches.append(info)
    if not matches:
        raise CheckpointNotFound(
            f"No F3 checkpoint matching feature_set={fs!r} and the frozen hparams in {models_dir}. "
            f"{CKPT_MISSING_MSG}"
        )
    matches.sort(key=lambda i: i.get("seed", 0))
    return Path(matches[0]["ckpt_path"]), matches[0]


def find_info_for_ckpt(ckpt: Path, search_dirs: list[Path]) -> dict | None:
    target = str(Path(ckpt).resolve()).lower()
    for d in search_dirs:
        for p in Path(d).glob("*.json"):
            try:
                info = json.loads(p.read_text())
                if "ckpt_path" in info and str(Path(info["ckpt_path"]).resolve()).lower() == target:
                    return info
            except (OSError, ValueError):
                continue
    return None


# ----------------------------------------------------------------------------
# Data acquisition (each step degrades gracefully)
# ----------------------------------------------------------------------------

def _clean_prices(df: pd.DataFrame) -> pd.DataFrame:
    df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
    df.index = pd.DatetimeIndex(df.index).tz_localize(None).normalize()
    df = df[~df.index.duplicated(keep="last")].sort_index()
    return df.dropna(subset=["Close"])


def load_local_prices(ticker: str) -> pd.DataFrame:
    from src.config import PRICES_DIR

    return _clean_prices(pd.read_parquet(PRICES_DIR / f"{ticker}.parquet"))


def fetch_live_prices(ticker: str, start: str, end: str) -> pd.DataFrame:
    from src.data.prices import fetch_ohlcv

    return _clean_prices(fetch_ohlcv(ticker, start, end))


def drop_incomplete_bar(px: pd.DataFrame, now: datetime | None = None) -> pd.DataFrame:
    """Drop today's bar if the NYSE session has not closed yet (partial bar)."""
    now_et = pd.Timestamp(now or datetime.now(timezone.utc)).tz_convert("America/New_York")
    today = now_et.normalize().tz_localize(None)
    if len(px) and px.index[-1] == today and now_et.hour * 60 + now_et.minute < 16 * 60 + 15:
        logger.info("Dropping incomplete bar for %s", px.index[-1].date())
        return px.iloc[:-1]
    return px


def get_prices(tickers: list[str], lookback_days: int, offline: bool) -> tuple[dict, dict]:
    """Return ({ticker: OHLCV}, {ticker: source}). Falls back to local parquet."""
    end = datetime.now(timezone.utc).date()
    start = (end - timedelta(days=lookback_days)).isoformat()
    prices, source = {}, {}
    for t in tickers:
        px = None
        if not offline:
            try:
                px = drop_incomplete_bar(fetch_live_prices(t, start, end.isoformat()))
                source[t] = "yfinance"
            except Exception as e:  # noqa: BLE001 - any network/parse failure falls back
                logger.warning("%s: yfinance failed (%s); falling back to local prices", t, e)
                px = None
        if px is None or px.empty:
            try:
                px = load_local_prices(t)
                source[t] = "local"
            except Exception as e:  # noqa: BLE001
                logger.warning("%s: no local prices either (%s); skipping ticker", t, e)
                continue
        cutoff = px.index[-1] - pd.Timedelta(days=lookback_days)
        prices[t] = px[px.index >= cutoff]
    return prices, source


def load_local_news(ticker: str) -> pd.DataFrame | None:
    from src.config import SENTIMENT_DIR

    p = SENTIMENT_DIR / f"{ticker}.parquet"
    if not p.exists():
        return None
    return pd.read_parquet(p)[NEWS_COLUMNS].copy()


def fetch_news_articles(ticker: str, start: str, end: str, api_key: str) -> list:
    from src.data.news import fetch_company_news

    return fetch_company_news(ticker, start, end, api_key, sleep_seconds=0.2)


def score_headlines(headlines: list) -> pd.DataFrame:
    from src.features.sentiment import score_headlines as _score

    return _score(headlines)


def get_scored_news(
    tickers: list[str], news_days: int, offline: bool, api_key: str | None
) -> tuple[dict, dict, dict]:
    """Return ({ticker: scored news | None}, {ticker: source}, {ticker: note}).

    Sources: "finnhub_finbert", "local_cached_finbert" (offline) or "none" (neutral).
    """
    from src.data.news import articles_to_dataframe

    scored: dict = {t: None for t in tickers}
    source = {t: "none" for t in tickers}
    note = {t: "" for t in tickers}

    if offline:
        for t in tickers:
            try:
                df = load_local_news(t)
            except Exception as e:  # noqa: BLE001
                df, note[t] = None, f"local news unreadable: {e}"
            if df is not None and len(df):
                scored[t], source[t] = df, "local_cached_finbert"
            elif not note[t]:
                note[t] = "offline: no local scored news"
        return scored, source, note

    if not api_key:
        logger.warning("FINNHUB_API_KEY not set; using neutral sentiment for all tickers")
        return scored, source, {t: "FINNHUB_API_KEY missing" for t in tickers}

    end = datetime.now(timezone.utc).date()
    start = (end - timedelta(days=news_days)).isoformat()
    raw: dict = {}
    for t in tickers:
        try:
            arts = fetch_news_articles(t, start, end.isoformat(), api_key)
            df = articles_to_dataframe(arts, t)
            if len(df):
                df = df.drop_duplicates(subset=["headline"]).reset_index(drop=True)
            raw[t] = df
        except Exception as e:  # noqa: BLE001 - HTTP / network / JSON errors
            logger.warning("%s: Finnhub failed (%s); neutral sentiment", t, type(e).__name__)
            note[t] = f"Finnhub error: {type(e).__name__}"

    all_headlines = [h for df in raw.values() for h in df["headline"].tolist()]
    scores = None
    if all_headlines:
        try:
            scores = score_headlines(all_headlines)
            if len(scores) != len(all_headlines):
                raise ValueError("FinBERT returned unexpected row count")
        except Exception as e:  # noqa: BLE001
            logger.warning("FinBERT scoring failed (%s); neutral sentiment", e)
            for t in raw:
                note[t] = f"FinBERT error: {type(e).__name__}"
            scores = None
    pos = 0
    for t, df in raw.items():
        n = len(df)
        if n == 0:
            source[t], note[t] = "finnhub_finbert", "no articles in window"
            continue
        if scores is not None:
            s = scores.iloc[pos:pos + n].reset_index(drop=True)
            out = pd.DataFrame({
                "published_at": pd.to_datetime(df["published_at"], utc=True).dt.tz_localize(None),
                "positive": s["positive"].to_numpy(),
                "negative": s["negative"].to_numpy(),
                "neutral": s["neutral"].to_numpy(),
                "sentiment_score": s["sentiment_score"].to_numpy(),
            })
            scored[t], source[t] = out, "finnhub_finbert"
        pos += n
    return scored, source, note


# ----------------------------------------------------------------------------
# Panel + prediction dataset
# ----------------------------------------------------------------------------

def future_trading_dates(origin: pd.Timestamp, n: int = N_FUTURE) -> pd.DatetimeIndex:
    """The next ``n`` NYSE trading days after ``origin`` (business days if no calendar lib)."""
    origin = pd.Timestamp(origin).normalize()
    try:
        import pandas_market_calendars as mcal

        cal = mcal.get_calendar("NYSE")
        days = cal.valid_days(start_date=origin + pd.Timedelta(days=1), end_date=origin + pd.Timedelta(days=n * 3 + 14))
        days = pd.DatetimeIndex(days).tz_localize(None).normalize()
        if len(days) >= n:
            return days[:n]
    except Exception as e:  # noqa: BLE001
        logger.warning("NYSE calendar unavailable (%s); using plain business days", e)
    return pd.bdate_range(origin + pd.Timedelta(days=1), periods=n)


def build_inference_panel(prices: dict, scored: dict) -> pd.DataFrame:
    """Per-ticker features on the shared calendar (warm-up dropped, forward targets NaN at the end)."""
    days = pd.DatetimeIndex(sorted(set().union(*(set(p.index) for p in prices.values()))))
    frames = [build_ticker_panel(px, scored.get(t), days, ticker=t) for t, px in prices.items()]
    return finalize_panel(pd.concat(frames, ignore_index=True))


def add_future_rows(panel: pd.DataFrame, n: int = N_FUTURE) -> tuple[pd.DataFrame, pd.DatetimeIndex]:
    """Append ``n`` future decoder rows per ticker with known calendar covariates.

    Calendar features are recomputed over history + future dates so the last
    history row has a real "next trading day" gap. Unknown reals and the target
    are filled with 0 in decoder rows (not used by the TFT decoder, but non-NaN).
    """
    last = panel["date"].max()
    fut = future_trading_dates(last, n + 1)  # one extra day so day n has a correct gap
    all_dates = pd.DatetimeIndex(sorted(panel["date"].unique())).append(fut)
    cal = build_calendar_features(all_dates)
    fut = fut[:n]
    last_idx = int(panel["time_idx"].max())
    cal_cols = list(cal.columns)

    out = panel.copy()
    out[cal_cols] = cal.reindex(out["date"]).to_numpy()
    rows = []
    for t in out["ticker"].unique():
        base = {c: 0.0 for c in out.columns if c not in ("date", "ticker", "time_idx")}
        for k, d in enumerate(fut, start=1):
            r = dict(base)
            r.update({"date": d, "ticker": t, "time_idx": last_idx + k})
            r.update(cal.loc[d].to_dict())
            rows.append(r)
    fut_df = pd.DataFrame(rows)
    for c in TARGET_COLUMNS:
        fut_df[c] = np.nan
    out = pd.concat([out, fut_df[out.columns]], ignore_index=True)
    return out.sort_values(["ticker", "time_idx"]).reset_index(drop=True), fut


def make_predict_dataset(model, full_df: pd.DataFrame):
    from pytorch_forecasting import TimeSeriesDataSet

    df = tft._prepare(full_df.drop(columns=TARGET_COLUMNS, errors="ignore"))
    return TimeSeriesDataSet.from_parameters(
        model.dataset_parameters, df, predict=True, stop_randomization=True
    )


def run_forecast(model, panel: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (step frame, directional frame). Future rows are label-free (NaN actuals)."""
    full_df, _ = add_future_rows(panel)
    ds = make_predict_dataset(model, full_df)
    label_panel = full_df.copy()
    last_idx = int(panel["time_idx"].max())
    label_panel.loc[label_panel["time_idx"] > last_idx, tft.TARGET] = np.nan
    pred = tft.predict_frame(model, ds, label_panel)
    dir_df = tft.directional_frame(pred, label_panel)
    return pred, dir_df


# ----------------------------------------------------------------------------
# Output
# ----------------------------------------------------------------------------

def _num(x):
    if x is None:
        return None
    x = float(x)
    return None if (math.isnan(x) or math.isinf(x)) else x


def sentiment_summary(scored, news_days: int, source: str, note: str, origin) -> dict:
    n, mean = 0, None
    if scored is not None and len(scored):
        pub = pd.to_datetime(scored["published_at"])
        recent = scored[pub >= (pd.Timestamp(origin) - pd.Timedelta(days=news_days))]
        n = int(len(recent))
        mean = _num(recent["sentiment_score"].mean()) if n else None
    out = {"source": source, "articles_in_window": n, "mean_score": mean, "window_days": news_days}
    if note:
        out["note"] = note
    return out


def build_output(
    pred: pd.DataFrame,
    dir_df: pd.DataFrame,
    panel: pd.DataFrame,
    scored: dict,
    news_source: dict,
    news_note: dict,
    price_source: dict,
    news_days: int,
    model_meta: dict,
    generated_at: str | None = None,
) -> dict:
    """Assemble the JSON-serialisable result from a prediction frame (no model needed)."""
    origin = pd.Timestamp(panel["date"].max())
    tickers = sorted(pred["ticker"].unique())
    sources = {news_source.get(t, "none") for t in tickers}
    overall_src = next(iter(sources)) if len(sources) == 1 else "mixed"
    out: dict = {
        "disclaimer": DISCLAIMER,
        "generated_at": generated_at or datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "origin_date": origin.strftime("%Y-%m-%d"),
        "sentiment_source": overall_src,
        "model": {**model_meta, "training_note": TRAINING_NOTE},
        "band_method": (
            "APPROXIMATE: cumulative median = sum of daily q50; sigma_cum = sqrt(sum_k sigma_k^2) with "
            "sigma_k=(q90-q10)/2.563 (normal, independent steps); band = median +/- 1.2816*sigma_cum; "
            "implied price = last_close * exp(cumulative log return); last_close is split/dividend-adjusted."
        ),
        "forecasts": {},
    }
    for t in tickers:
        px_t = panel[panel["ticker"] == t].sort_values("time_idx")
        last_close = float(np.exp(px_t["log_close"].iloc[-1]))
        steps = pred[pred["ticker"] == t].sort_values("step")
        d = dir_df[dir_df["ticker"] == t].set_index("horizon")
        entry = {
            "origin_date": origin.strftime("%Y-%m-%d"),
            "price_as_of": origin.strftime("%Y-%m-%d"),
            "price_source": price_source.get(t, "unknown"),
            "last_close": _num(last_close),
            "sentiment": sentiment_summary(scored.get(t), news_days, news_source.get(t, "none"), news_note.get(t, ""), origin),
            "steps": [
                {"k": int(r.step), "date": pd.Timestamp(r.step_date).strftime("%Y-%m-%d"),
                 **{c: _num(getattr(r, c)) for c in tft.QCOLS}}
                for r in steps.itertuples()
            ],
            "horizons": {},
        }
        for h in HORIZONS:
            if h not in d.index:
                continue
            med, sig = float(d.loc[h, "pred_cum_median"]), float(d.loc[h, "sigma_cum"])
            entry["horizons"][str(h)] = {
                "date": pd.Timestamp(steps[steps["step"] == h]["step_date"].iloc[0]).strftime("%Y-%m-%d"),
                "cum_median_return": _num(med),
                "implied_price_median": _num(last_close * math.exp(med)),
                "implied_price_q10": _num(last_close * math.exp(med - Z_80 * sig)),
                "implied_price_q90": _num(last_close * math.exp(med + Z_80 * sig)),
                "sigma_cum": _num(sig),
                "p_up": _num(d.loc[h, "p_up"]),
            }
        out["forecasts"][t] = entry
    return out


def to_csv_frame(result: dict) -> pd.DataFrame:
    rows = []
    for t, e in result["forecasts"].items():
        for s in e["steps"]:
            h = e["horizons"].get(str(s["k"]), {})
            rows.append({
                "ticker": t, "origin_date": e["origin_date"], "price_as_of": e["price_as_of"],
                "last_close": e["last_close"], "sentiment_source": e["sentiment"]["source"],
                "articles_in_window": e["sentiment"]["articles_in_window"],
                "sent_mean_score": e["sentiment"]["mean_score"],
                "k": s["k"], "date": s["date"], **{c: s[c] for c in tft.QCOLS},
                "cum_median_return": h.get("cum_median_return"),
                "implied_price_median": h.get("implied_price_median"),
                "implied_price_q10": h.get("implied_price_q10"),
                "implied_price_q90": h.get("implied_price_q90"),
                "p_up": h.get("p_up"),
            })
    return pd.DataFrame(rows)


def write_outputs(result: dict, out_dir: Path) -> list[Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    origin = result["origin_date"]
    text = json.dumps(result, indent=2)
    paths = [out_dir / f"forecasts_{origin}.json", out_dir / f"forecasts_{origin}.csv", out_dir / "latest.json"]
    paths[0].write_text(text)
    to_csv_frame(result).to_csv(paths[1], index=False)
    paths[2].write_text(text)
    return paths


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tickers", default=None, help="comma list (default: tickers the checkpoint was trained on)")
    p.add_argument("--ckpt", default=None, help="explicit checkpoint path (skips frozen_d.json resolution)")
    p.add_argument("--news-days", type=int, default=7)
    p.add_argument("--price-lookback-days", type=int, default=500,
                   help="calendar days of price history (252-day technical warm-up + encoder window)")
    p.add_argument("--out-dir", default=str(tft.REPORTS_DIR / "latest"))
    p.add_argument("--offline", action="store_true", help="local data/raw/prices and data/raw/sentiment only")
    p.add_argument("--smoke", action="store_true", help="resolve the checkpoint from the smoke reports/models dirs")
    return p


def train_tickers(model) -> list[str]:
    classes = model.dataset_parameters["categorical_encoders"]["ticker"].classes_
    names = classes.keys() if isinstance(classes, dict) else classes
    return [str(c) for c in names if str(c) not in ("nan", "None")]


def main(argv=None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = build_parser().parse_args(argv)

    info = None
    try:
        if args.ckpt:
            ckpt = Path(args.ckpt)
            if not ckpt.exists():
                raise CheckpointNotFound(f"Checkpoint {ckpt} does not exist. {CKPT_MISSING_MSG}")
            info = find_info_for_ckpt(ckpt, [tft.MODELS_DIR / "full", tft.MODELS_DIR / "smoke"])
        else:
            sub = "smoke" if args.smoke else "full"
            reports = tft.REPORTS_DIR / ("smoke" if args.smoke else "")
            ckpt, info = resolve_checkpoint(reports, tft.MODELS_DIR / sub)
    except CheckpointNotFound as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2

    if (info or {}).get("feature_set") == "sentiment" and (info or {}).get("fold") == "F3":
        # F3 checkpoints are trained on <=2022 rows, where every sentiment feature is the
        # neutral fill (news coverage starts 2025-08). Live sentiment values are therefore
        # out-of-distribution inputs the model never learned from.
        logger.warning(
            "Checkpoint uses the 'sentiment' feature set but was trained before any news "
            "coverage; live sentiment inputs are out-of-distribution for this model."
        )

    import torch

    torch.set_num_threads(int(os.environ.get("PREDICT_TORCH_THREADS", "1")))
    model = tft.load_model(ckpt)
    trained = train_tickers(model)
    if args.tickers:
        tickers = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]
        unknown = [t for t in tickers if t not in trained]
        if unknown:
            logger.warning("Tickers not seen in training are skipped (no scaler/embedding): %s", unknown)
        tickers = [t for t in tickers if t in trained]
    else:
        tickers = trained
    if not tickers:
        print("ERROR: no usable tickers (none overlap with the checkpoint's training tickers).", file=sys.stderr)
        return 2

    prices, price_source = get_prices(tickers, args.price_lookback_days, args.offline)
    if not prices:
        print("ERROR: no price data available (network and local data/raw/prices both failed).", file=sys.stderr)
        return 2
    tickers = [t for t in tickers if t in prices]

    from src.config import FINNHUB_API_KEY

    scored, news_source, news_note = get_scored_news(tickers, args.news_days, args.offline, FINNHUB_API_KEY)

    try:
        panel = build_inference_panel(prices, scored)
    except ValueError as e:
        print(f"ERROR: could not build the feature panel ({e}). Try a larger --price-lookback-days.", file=sys.stderr)
        return 2
    enc_len = int(model.dataset_parameters.get("max_encoder_length", 30))
    if panel.groupby("ticker").size().min() < enc_len:
        print(f"ERROR: fewer than {enc_len} usable rows after warm-up; raise --price-lookback-days.", file=sys.stderr)
        return 2

    pred, dir_df = run_forecast(model, panel)

    # The panel is on a shared calendar and forward-fills tickers with older data.
    asof = {t: prices[t].index.max() for t in tickers}
    origin = panel["date"].max()
    for t, d in asof.items():
        if d < origin:
            logger.warning("%s: last real close is %s, older than origin %s (forward-filled)", t, d.date(), origin.date())

    f3 = (info or {}).get("fold") == "F3"
    meta = {
        "checkpoint": str(ckpt),
        "feature_set": (info or {}).get("feature_set"),
        "hparams": (info or {}).get("hparams"),
        "fold": (info or {}).get("fold"),
        "trained_through": "2022-12-31" if f3 else "unknown (see training_note)",
        "early_stopped_on": "2023" if f3 else "unknown",
        "encoder_length": enc_len,
        "trained_tickers": trained,
    }
    result = build_output(pred, dir_df, panel, scored, news_source, news_note, price_source, args.news_days, meta)
    for t, e in result["forecasts"].items():
        e["price_as_of"] = asof[t].strftime("%Y-%m-%d")
    for p in write_outputs(result, Path(args.out_dir)):
        print(f"wrote {p}")
    def _fmt(v) -> str:
        return f"{v:.2f}" if isinstance(v, (int, float)) else "n/a"

    for t, e in result["forecasts"].items():
        h1, h7 = e["horizons"].get("1", {}), e["horizons"].get("7", {})
        print(f"{t}: close {_fmt(e.get('last_close'))} (as of {e['price_as_of']}) p_up 1d={_fmt(h1.get('p_up'))} "
              f"7d={_fmt(h7.get('p_up'))} sentiment={e['sentiment']['source']} n={e['sentiment']['articles_in_window']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

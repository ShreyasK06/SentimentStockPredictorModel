"""Daily per-(ticker, effective trading day) news features (Phase 8 WP2).

Input: scored articles with columns ticker, effective_day (naive datetime), publisher,
positive, negative, neutral, score (= positive - negative).
Row E only ever uses articles whose effective_day <= E (rolling windows look back
over the NYSE trading-day calendar, never forward).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

ROLL_WINDOWS = (3, 5, 10, 20)
DECAY_HALFLIFE = 3.0


def _same_day(art: pd.DataFrame) -> pd.DataFrame:
    a = art.copy()
    lab = a[["positive", "negative", "neutral"]].to_numpy().argmax(axis=1)
    a["_is_pos"] = (lab == 0).astype(float)
    a["_is_neg"] = (lab == 1).astype(float)
    g = a.groupby(["ticker", "effective_day"])
    out = pd.DataFrame({
        "news_count": g.size(),
        "news_score_mean": g["score"].mean(),
        "news_score_max": g["score"].max(),
        "news_score_min": g["score"].min(),
        "news_score_std": g["score"].std().fillna(0.0),
        "news_pos_mean": g["positive"].mean(),
        "news_neg_mean": g["negative"].mean(),
        "news_neu_mean": g["neutral"].mean(),
        "news_share_pos": g["_is_pos"].mean(),
        "news_share_neg": g["_is_neg"].mean(),
        "news_score_sum": g["score"].sum(),
        "news_n_publishers": g["publisher"].nunique(),
    })
    out["news_net_tone"] = out["news_share_pos"] - out["news_share_neg"]
    out["news_log_count"] = np.log1p(out["news_count"])
    out.index = out.index.set_names(["ticker", "date"])
    return out.reset_index()


def build_daily_news_features(art: pd.DataFrame, trading_days: pd.DatetimeIndex) -> pd.DataFrame:
    """One row per (ticker, E) with >= 1 article. Rolling features use trading days <= E."""
    days = pd.DatetimeIndex(sorted(pd.DatetimeIndex(trading_days).tz_localize(None).normalize().unique()))
    same = _same_day(art)
    frames = []
    for ticker, g in same.groupby("ticker"):
        g = g.set_index("date").sort_index()
        full = g.reindex(days[(days >= g.index.min()) & (days <= g.index.max())])
        cnt = full["news_count"].fillna(0.0)
        ssum = full["news_score_sum"].fillna(0.0)
        feats = pd.DataFrame(index=full.index)
        for w in ROLL_WINDOWS:
            c = cnt.rolling(w, min_periods=1).sum()
            s = ssum.rolling(w, min_periods=1).sum()
            feats[f"news_roll{w}_count"] = c
            feats[f"news_roll{w}_score"] = s / c.where(c > 0)
        alpha = 1 - 0.5 ** (1 / DECAY_HALFLIFE)
        ew_s = ssum.ewm(alpha=alpha, adjust=False).mean()
        ew_c = cnt.ewm(alpha=alpha, adjust=False).mean()
        feats["news_decay_score"] = ew_s / ew_c.where(ew_c > 0)
        feats["news_decay_count"] = ew_c
        # trailing 20d excluding today
        prev_c = cnt.shift(1).rolling(20, min_periods=1).sum()
        prev_s = ssum.shift(1).rolling(20, min_periods=1).sum()
        prev_mean = prev_s / prev_c.where(prev_c > 0)
        feats["news_surprise"] = full["news_score_mean"] - prev_mean.fillna(0.0)
        feats["news_count_vs_20d"] = cnt / (prev_c / 20.0 + 1.0)
        has = cnt > 0
        idx = pd.Series(np.arange(len(full)), index=full.index)
        last = idx.where(has).ffill().shift(1)
        feats["news_days_since_prev"] = (idx - last).fillna(60.0).clip(upper=60.0)
        feats = feats[has.values]
        merged = g.join(feats, how="left")
        merged["ticker"] = ticker
        frames.append(merged.reset_index().rename(columns={"index": "date"}))
    out = pd.concat(frames, ignore_index=True)
    out = out.drop(columns=["news_score_sum"])
    return out.sort_values(["date", "ticker"]).reset_index(drop=True)


def daily_mean_embeddings(art_keys: pd.DataFrame, key_index: dict, matrix: np.ndarray,
                          rows: pd.DataFrame) -> np.ndarray:
    """Mean embedding of each row's articles.

    art_keys: ticker, effective_day, text_key. key_index: text_key -> row in `matrix`.
    rows: ticker, date (the daily feature rows). Returns float32 [len(rows), dim];
    a row with no embedded article gets zeros."""
    pos = art_keys["text_key"].map(key_index)
    ok = pos.notna().to_numpy()
    ak = art_keys.loc[ok, ["ticker", "effective_day"]].reset_index(drop=True)
    pid = pos[ok].astype(np.int64).to_numpy()
    key_rows = pd.MultiIndex.from_frame(rows[["ticker", "date"]])
    key_art = pd.MultiIndex.from_arrays([ak["ticker"], ak["effective_day"]])
    rid = key_rows.get_indexer(key_art)
    sel = rid >= 0
    rid, pid = rid[sel], pid[sel]
    dim = matrix.shape[1]
    out = np.zeros((len(rows), dim), dtype=np.float64)
    cnt = np.zeros(len(rows), dtype=np.float64)
    for s in range(0, len(rid), 50000):
        r, p = rid[s:s + 50000], pid[s:s + 50000]
        np.add.at(out, r, matrix[p].astype(np.float64))
        np.add.at(cnt, r, 1.0)
    out /= np.maximum(cnt, 1.0)[:, None]
    return out.astype(np.float32)

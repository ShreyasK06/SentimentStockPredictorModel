import numpy as np
import pandas as pd

from src.features.market_controls import build_market_controls
from src.features.news_daily import build_daily_news_features, daily_mean_embeddings


def _prices(n=120, seed=0):
    rng = np.random.default_rng(seed)
    days = pd.bdate_range("2020-01-01", periods=n)
    frames = []
    for t in ["AAA", "SPY"]:
        c = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
        frames.append(pd.DataFrame({"date": days, "ticker": t, "Open": c * 1.001, "High": c * 1.01,
                                    "Low": c * 0.99, "Close": c, "Volume": rng.integers(1e5, 1e6, n)}))
    return pd.concat(frames, ignore_index=True)


def test_controls_do_not_use_future_prices():
    p = _prices()
    base = build_market_controls(p)
    e = p["date"].unique()[80]
    p2 = p.copy()
    fut = p2["date"] > e
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        p2.loc[fut, col] = p2.loc[fut, col] * 3
    pert = build_market_controls(p2)
    a = base[base["date"] <= e].drop(columns=["ticker", "date"]).to_numpy()
    b = pert[pert["date"] <= e].drop(columns=["ticker", "date"]).to_numpy()
    np.testing.assert_allclose(a, b, equal_nan=True)
    assert not np.allclose(base[base["date"] > e]["ctl_ret_1d"].fillna(0),
                           pert[pert["date"] > e]["ctl_ret_1d"].fillna(0))


def _articles():
    days = pd.bdate_range("2020-01-01", periods=30)
    rows = []
    for i, d in enumerate(days):
        if i % 3 == 0:
            for k in range(2):
                pos = 0.6 if k == 0 else 0.2
                rows.append({"ticker": "AAA", "effective_day": d, "publisher": f"p{k}", "positive": pos,
                             "negative": 0.1, "neutral": 0.9 - pos, "score": pos - 0.1, "text_key": f"{i}-{k}"})
    return pd.DataFrame(rows), days


def test_news_features_ignore_future_articles():
    art, days = _articles()
    base = build_daily_news_features(art, days)
    e = days[15]
    art2 = art.copy()
    art2.loc[art2["effective_day"] > e, ["positive", "score"]] = [0.99, 0.89]
    pert = build_daily_news_features(art2, days)
    a = base[base["date"] <= e].drop(columns=["ticker", "date"]).to_numpy(dtype=float)
    b = pert[pert["date"] <= e].drop(columns=["ticker", "date"]).to_numpy(dtype=float)
    np.testing.assert_allclose(a, b, equal_nan=True)


def test_news_features_values():
    art, days = _articles()
    f = build_daily_news_features(art, days)
    assert len(f) == 10  # one row per news day
    r0 = f.iloc[0]
    assert r0["news_count"] == 2
    np.testing.assert_allclose(r0["news_score_mean"], (0.5 + 0.1) / 2)
    assert r0["news_days_since_prev"] == 60.0
    assert f.iloc[1]["news_days_since_prev"] == 3.0
    # roll3 at a news day covers only that day (previous news 3 sessions back is outside)
    assert f.iloc[1]["news_roll3_count"] == 2
    assert f.iloc[1]["news_roll5_count"] == 4


def test_daily_mean_embeddings():
    art, days = _articles()
    f = build_daily_news_features(art, days)
    keys = art["text_key"].tolist()
    mat = np.arange(len(keys) * 4, dtype=np.float16).reshape(len(keys), 4)
    idx = {k: i for i, k in enumerate(keys)}
    emb = daily_mean_embeddings(art[["ticker", "effective_day", "text_key"]], idx, mat, f[["ticker", "date"]])
    np.testing.assert_allclose(emb[0], mat[:2].astype(float).mean(axis=0))

"""Fast tests for scripts/predict_latest.py (no network, no FinBERT download)."""

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.data.tft_dataset import TARGET_COLUMNS
from src.models import tft

_SPEC = importlib.util.spec_from_file_location(
    "predict_latest", Path(__file__).resolve().parents[2] / "scripts" / "predict_latest.py"
)
pl = importlib.util.module_from_spec(_SPEC)
sys.modules["predict_latest"] = pl
_SPEC.loader.exec_module(pl)


def synthetic_prices(seed: int, n: int = 400, start: str = "2022-01-03") -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(start, periods=n, name="date")
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    return pd.DataFrame(
        {
            "Open": close * (1 + rng.normal(0, 0.002, n)),
            "High": close * 1.01,
            "Low": close * 0.99,
            "Close": close,
            "Volume": rng.integers(1_000_000, 2_000_000, n).astype(float),
        },
        index=idx,
    )


# ---------------------------------------------------------------- sentiment degradation

def test_missing_key_gives_neutral_sentiment(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("must not call Finnhub without a key")

    monkeypatch.setattr(pl, "fetch_news_articles", boom)
    scored, source, note = pl.get_scored_news(["AAA", "BBB"], 7, offline=False, api_key="")
    assert scored == {"AAA": None, "BBB": None}
    assert set(source.values()) == {"none"}
    assert "FINNHUB_API_KEY" in note["AAA"]


def test_finnhub_error_per_ticker_is_neutral(monkeypatch):
    def fetch(ticker, start, end, key):
        if ticker == "BAD":
            raise RuntimeError("HTTP 429")
        return [{"datetime": 1_760_000_000, "headline": f"{ticker} up", "source": "x", "url": "u"}]

    def fake_score(headlines):
        return pd.DataFrame({"headline": headlines, "positive": 0.7, "negative": 0.1,
                             "neutral": 0.2, "sentiment_score": 0.6})

    monkeypatch.setattr(pl, "fetch_news_articles", fetch)
    monkeypatch.setattr(pl, "score_headlines", fake_score)
    scored, source, note = pl.get_scored_news(["GOOD", "BAD"], 7, offline=False, api_key="k")
    assert source == {"GOOD": "finnhub_finbert", "BAD": "none"}
    assert scored["BAD"] is None and "Finnhub error" in note["BAD"]
    assert len(scored["GOOD"]) == 1 and scored["GOOD"]["sentiment_score"].iloc[0] == pytest.approx(0.6)
    assert scored["GOOD"]["published_at"].dt.tz is None


def test_finbert_failure_is_neutral(monkeypatch):
    monkeypatch.setattr(pl, "fetch_news_articles",
                        lambda *a: [{"datetime": 1_760_000_000, "headline": "h", "source": "s", "url": "u"}])

    def broken(_):
        raise OSError("no model")

    monkeypatch.setattr(pl, "score_headlines", broken)
    scored, source, note = pl.get_scored_news(["AAA"], 7, offline=False, api_key="k")
    assert scored["AAA"] is None and source["AAA"] == "none" and "FinBERT" in note["AAA"]


def test_yfinance_failure_falls_back_to_local(monkeypatch):
    local = synthetic_prices(0, 50)
    monkeypatch.setattr(pl, "fetch_live_prices", lambda *a: (_ for _ in ()).throw(ConnectionError("down")))
    monkeypatch.setattr(pl, "load_local_prices", lambda t: local)
    prices, source = pl.get_prices(["AAA"], 450, offline=False)
    assert source["AAA"] == "local" and prices["AAA"].index.max() == local.index.max()


# ---------------------------------------------------------------- checkpoint resolution

def test_checkpoint_resolution_failure_exit_code_2(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(tft, "REPORTS_DIR", tmp_path / "reports")
    monkeypatch.setattr(tft, "MODELS_DIR", tmp_path / "models")
    code = pl.main(["--offline"])
    err = capsys.readouterr().err
    assert code == 2
    assert "run_phase7_tft.py --stage d --phase cv" in err and "Traceback" not in err


def test_checkpoint_resolution_matches_f3_info(tmp_path):
    reports, models = tmp_path / "r", tmp_path / "m"
    reports.mkdir(), models.mkdir()
    ckpt = tmp_path / "best.ckpt"
    ckpt.write_text("x")
    hp = {**tft.DEFAULT_HPARAMS, "hidden_size": 32}
    (reports / "frozen_d.json").write_text(json.dumps({"feature_set": "technical", "hparams": hp}))
    other = {"fold": "F2", "feature_set": "technical", "hparams": hp, "ckpt_path": str(ckpt)}
    wrong_hp = {"fold": "F3", "feature_set": "technical", "hparams": tft.DEFAULT_HPARAMS, "ckpt_path": str(ckpt)}
    good = {"fold": "F3", "feature_set": "technical", "hparams": hp, "ckpt_path": str(ckpt), "seed": 0}
    for i, d in enumerate([other, wrong_hp, good]):
        (models / f"{i}.json").write_text(json.dumps(d))
    path, info = pl.resolve_checkpoint(reports, models)
    assert path == ckpt and info["fold"] == "F3" and info["hparams"]["hidden_size"] == 32
    (models / "2.json").unlink()
    with pytest.raises(pl.CheckpointNotFound):
        pl.resolve_checkpoint(reports, models)


# ---------------------------------------------------------------- calendar

def test_future_trading_dates_skip_weekend_and_holiday():
    d = pl.future_trading_dates(pd.Timestamp("2025-11-26"), 7)  # Wed before Thanksgiving
    assert len(d) == 7 and (d.dayofweek < 5).all() and d.is_monotonic_increasing
    assert d[0] > pd.Timestamp("2025-11-26")
    try:
        import pandas_market_calendars  # noqa: F401

        assert pd.Timestamp("2025-11-27") not in d
    except ImportError:
        pass
    d2 = pl.future_trading_dates(pd.Timestamp("2025-10-17"), 7)  # Friday
    assert d2[0] == pd.Timestamp("2025-10-20") and d2[-1] == pd.Timestamp("2025-10-28")


def test_add_future_rows_continuous_time_idx_and_calendar():
    prices = {"AAA": synthetic_prices(1), "BBB": synthetic_prices(2)}
    panel = pl.build_inference_panel(prices, {})
    full, fut = pl.add_future_rows(panel)
    assert len(fut) == 7 and fut[0] > panel["date"].max()
    for t, g in full.groupby("ticker"):
        g = g.sort_values("time_idx")
        assert (np.diff(g["time_idx"]) == 1).all()
        tail = g.tail(7)
        assert list(tail["date"]) == list(fut)
        assert tail["log_ret_1d"].eq(0).all() and tail["rsi_14"].eq(0).all()
    assert not full.drop(columns=TARGET_COLUMNS).isna().any().any()


# ---------------------------------------------------------------- output schema

def fake_frames(origin="2025-12-31"):
    panel = pd.DataFrame({
        "date": [pd.Timestamp(origin)] * 2, "ticker": ["AAA", "BBB"],
        "time_idx": [10, 10], "log_close": [np.log(100.0), np.log(50.0)],
    })
    steps = pd.bdate_range(pd.Timestamp(origin) + pd.Timedelta(days=1), periods=7)
    rows = []
    for t in ["AAA", "BBB"]:
        for k, d in enumerate(steps, 1):
            rows.append({"ticker": t, "origin_idx": 10, "origin_date": pd.Timestamp(origin), "step": k,
                         "step_idx": 10 + k, "step_date": d, "q02": -0.03, "q10": -0.015, "q25": -0.005,
                         "q50": 0.001, "q75": 0.007, "q90": 0.016, "q98": 0.03, "actual": np.nan, "pinball": np.nan})
    pred = pd.DataFrame(rows)
    return panel, pred, tft.directional_frame(pred, panel.assign(**{c: np.nan for c in TARGET_COLUMNS}))


def test_output_schema_and_writer(tmp_path):
    panel, pred, dir_df = fake_frames()
    news = pd.DataFrame({"published_at": [pd.Timestamp("2025-12-30 15:00")], "positive": 0.5, "negative": 0.1,
                         "neutral": 0.4, "sentiment_score": 0.4})
    res = pl.build_output(pred, dir_df, panel, {"AAA": news, "BBB": None}, {"AAA": "finnhub_finbert", "BBB": "none"},
                          {"BBB": "FINNHUB_API_KEY missing"}, {"AAA": "yfinance", "BBB": "local"}, 7,
                          {"checkpoint": "c.ckpt", "fold": "F3"}, generated_at="2025-12-31T00:00:00+00:00")
    assert "not financial advice" in res["disclaimer"] and "majority-class baseline" in res["disclaimer"]
    assert res["origin_date"] == "2025-12-31" and res["sentiment_source"] == "mixed"
    assert "2022-12-31" in res["model"]["training_note"]
    a = res["forecasts"]["AAA"]
    assert a["last_close"] == pytest.approx(100.0) and a["sentiment"]["articles_in_window"] == 1
    assert a["sentiment"]["mean_score"] == pytest.approx(0.4)
    assert res["forecasts"]["BBB"]["sentiment"]["source"] == "none"
    assert [s["k"] for s in a["steps"]] == list(range(1, 8))
    assert {"q02", "q10", "q25", "q50", "q75", "q90", "q98", "date"} <= set(a["steps"][0])
    assert set(a["horizons"]) == {"1", "3", "5", "7"}
    h3 = a["horizons"]["3"]
    assert h3["cum_median_return"] == pytest.approx(0.003)
    assert h3["implied_price_median"] == pytest.approx(100 * np.exp(0.003))
    assert h3["implied_price_q10"] < h3["implied_price_median"] < h3["implied_price_q90"]
    assert 0.5 < h3["p_up"] < 1.0

    paths = pl.write_outputs(res, tmp_path / "out")
    assert [p.name for p in paths] == ["forecasts_2025-12-31.json", "forecasts_2025-12-31.csv", "latest.json"]
    assert json.loads(paths[2].read_text())["forecasts"].keys() == {"AAA", "BBB"}
    csv = pd.read_csv(paths[1])
    assert len(csv) == 14 and {"ticker", "k", "q50", "p_up", "implied_price_median"} <= set(csv.columns)


# ---------------------------------------------------------------- integration (tiny model)

def test_end_to_end_tiny_model(tmp_path):
    import torch

    torch.set_num_threads(1)
    prices = {t: synthetic_prices(i) for i, t in enumerate(["AAA", "BBB"])}
    panel = pl.build_inference_panel(prices, {})
    dates = sorted(panel["date"].unique())
    split = tft.Split("T", str(dates[80].date()), str(dates[81].date()), str(dates[110].date()))
    hp = {"hidden_size": 8, "attention_head_size": 1, "hidden_continuous_size": 4, "encoder_length": 10}
    train_ds, val_ds, _ = tft.make_datasets(panel, split, "technical", encoder_length=10)
    model = tft.train_tft(train_ds, val_ds, hp, max_epochs=1, seed=0, ckpt_dir=tmp_path / "ck",
                          limit_train_batches=2, batch_size=64)
    torch.set_num_threads(1)
    loaded = tft.load_model(model.best_ckpt_path)
    assert pl.train_tickers(loaded) == ["AAA", "BBB"]

    pred, dir_df = pl.run_forecast(loaded, panel)
    assert len(pred) == 14 and set(pred["step"]) == set(range(1, 8))
    origin = panel["date"].max()
    assert (pred["origin_date"] == origin).all()
    assert pred["step_date"].min() > origin and pred["step_date"].notna().all()
    assert pred[tft.QCOLS].notna().all().all()

    res = pl.build_output(pred, dir_df, panel, {}, {"AAA": "none", "BBB": "none"}, {}, {"AAA": "local", "BBB": "local"},
                          7, {"checkpoint": "x"})
    assert res["sentiment_source"] == "none"
    assert res["forecasts"]["AAA"]["horizons"]["7"]["p_up"] is not None
    pl.write_outputs(res, tmp_path / "latest")
    assert (tmp_path / "latest" / "latest.json").exists()

"""Fast tests for src/models/tft.py using a small synthetic panel."""

import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm

from src.data.tft_dataset import (
    CALENDAR_FEATURES,
    OBSERVED_FEATURES,
    TARGET_COLUMNS,
    build_calendar_features,
    build_forward_targets,
)
from src.models import tft
from src.models.evaluate import majority_baseline

N_DAYS = 150
TICKERS = ["AAA", "BBB", "CCC"]
SPLIT = tft.Split("T", "2022-04-29", "2022-05-02", "2022-06-15", "2022-06-16", "2022-07-29")
ENC = 10
HP = {"hidden_size": 8, "attention_head_size": 1, "hidden_continuous_size": 4, "encoder_length": ENC}


def make_panel(seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    days = pd.bdate_range("2022-01-03", periods=N_DAYS)
    cal = build_calendar_features(days).reset_index(drop=True)
    frames = []
    for t in TICKERS:
        log_ret = rng.normal(0, 0.01, N_DAYS)
        close = pd.Series(100 * np.exp(np.cumsum(log_ret)), index=days)
        df = pd.DataFrame({"date": days, "ticker": t, "time_idx": np.arange(N_DAYS)})
        df["log_close"] = np.log(close.to_numpy())
        for c in OBSERVED_FEATURES:
            df[c] = rng.normal(0, 1, N_DAYS)
        df["log_ret_1d"] = log_ret
        df = pd.concat([df, cal], axis=1)
        tg = build_forward_targets(close).reset_index(drop=True)
        df = pd.concat([df, tg], axis=1)
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


@pytest.fixture(scope="module")
def panel():
    return make_panel()


@pytest.fixture(scope="module")
def bounds(panel):
    return tft.resolve_split(panel, SPLIT)


@pytest.fixture(scope="module")
def datasets(panel):
    return tft.make_datasets(panel, SPLIT, "technical", encoder_length=ENC)


def _decoder_idx(ds):
    dl = ds.to_dataloader(train=False, batch_size=256, num_workers=0)
    return np.concatenate([x["decoder_time_idx"].numpy() for x, _ in dl])


def test_train_decoder_never_crosses_train_end(datasets, bounds):
    train_ds, _, _ = datasets
    assert _decoder_idx(train_ds).max() <= bounds["train_end_idx"]


def test_val_and_test_decoders_inside_windows(datasets, bounds):
    _, val_ds, test_ds = datasets
    v = _decoder_idx(val_ds)
    assert v.min() >= bounds["val_start_idx"] and v.max() <= bounds["val_end_idx"]
    t = _decoder_idx(test_ds)
    assert t.min() >= bounds["test_start_idx"] and t.max() <= bounds["test_end_idx"]
    # every origin of the window is evaluated (predict=False)
    n_origins = bounds["val_end_idx"] - bounds["val_start_idx"] + 1 - (tft.MAX_PREDICTION_LENGTH - 1)
    assert len(val_ds) == n_origins * len(TICKERS)


def _scaler_state(ds):
    state = {"norm": [ds.target_normalizer.get_parameters(np.array([t])).tolist() for t in TICKERS]}
    for name, sc in sorted(ds.get_parameters()["scalers"].items()):
        for attr in ("center_", "scale_", "mean_", "var_"):
            if hasattr(sc, attr):
                state[f"{name}.{attr}"] = np.asarray(getattr(sc, attr)).tolist()
    assert len(state) > 3
    return state


def test_scalers_fit_on_train_only(panel, bounds):
    train_a, _, _ = tft.make_datasets(panel, SPLIT, "technical", ENC)
    perturbed = panel.copy()
    after = perturbed["time_idx"] > bounds["train_end_idx"]
    cols = list(dict.fromkeys(["log_ret_1d"] + tft.feature_columns("technical")))
    perturbed.loc[after, cols] = perturbed.loc[after, cols] * 50 + 7
    train_b, val_b, _ = tft.make_datasets(perturbed, SPLIT, "technical", ENC)
    assert _scaler_state(train_a) == _scaler_state(train_b)
    # val/test reuse the train-fitted normalizer rather than refitting
    assert _scaler_state(val_b) == _scaler_state(train_b)


def test_feature_lists_exclude_targets(datasets):
    for fs in tft.FEATURE_SETS:
        assert not set(tft.feature_columns(fs)) & set(TARGET_COLUMNS)
    train_ds = datasets[0]
    used = set(train_ds.reals) | set(train_ds.categoricals)
    assert not used & set(TARGET_COLUMNS)
    assert set(CALENDAR_FEATURES) <= used
    with pytest.raises(ValueError):
        tft.feature_columns("nonexistent")


@pytest.fixture(scope="module")
def trained(datasets, tmp_path_factory):
    train_ds, val_ds, _ = datasets
    ckpt = tmp_path_factory.mktemp("ckpt")
    return tft.train_tft(
        train_ds, val_ds, HP, max_epochs=1, seed=0, ckpt_dir=ckpt, limit_train_batches=3, batch_size=64
    )


def test_train_runs_and_saves_checkpoint(trained):
    assert trained.epochs_trained == 1
    assert trained.best_ckpt_path.endswith(".ckpt")


def test_predict_frame_origin_and_step_mapping(trained, datasets, panel, bounds):
    _, val_ds, _ = datasets
    pf = tft.predict_frame(trained, val_ds, panel)
    assert set(pf["step"]) == set(range(1, 8))
    assert pf[tft.QCOLS].isna().sum().sum() == 0

    dates = sorted(panel["date"].unique())
    nxt = {dates[i]: dates[i + 1] for i in range(len(dates) - 1)}
    s1 = pf[pf["step"] == 1]
    assert (s1["step_date"].to_numpy() == s1["origin_date"].map(nxt).to_numpy()).all()
    # decoder step 1 of the first origin is val_start; the origin is the day before
    assert pf["origin_idx"].min() == bounds["val_start_idx"] - 1
    assert pf["step_idx"].max() == bounds["val_end_idx"]

    # `actual` must equal the target the network actually saw in the decoder
    res = trained.predict(
        val_ds, mode="quantiles", return_index=True, return_y=True, batch_size=256,
        trainer_kwargs={"accelerator": "cpu", "logger": False, "enable_progress_bar": False,
                        "enable_model_summary": False},
    )
    y = res.y[0].numpy()
    idx = res.index.reset_index(drop=True)
    for i in (0, len(idx) // 2, len(idx) - 1):
        sel = pf[(pf["ticker"] == idx.loc[i, "ticker"]) & (pf["origin_idx"] == idx.loc[i, "time_idx"] - 1)]
        np.testing.assert_allclose(sel["actual"].to_numpy(), y[i], rtol=1e-4, atol=1e-6)


def _pred_df():
    rows = []
    for k in range(1, 8):
        rows.append(
            {
                "ticker": "AAA", "origin_idx": 100, "origin_date": pd.Timestamp("2022-05-20"), "step": k,
                "q02": -0.02, "q10": 0.0, "q25": 0.005, "q50": 0.01, "q75": 0.015, "q90": 0.02563, "q98": 0.04,
                "actual": 0.012, "pinball": 0.001,
            }
        )
    return pd.DataFrame(rows)


def test_directional_frame_cumulative_math():
    labels = pd.DataFrame(
        {
            "ticker": ["AAA"], "time_idx": [100],
            **{f"fwd_ret_{h}d": [0.01 * h] for h in (1, 3, 5, 7)},
            **{f"fwd_dir_{h}d": [1.0] for h in (1, 3, 5, 7)},
        }
    )
    d = tft.directional_frame(_pred_df(), labels)
    sigma_k = 0.02563 / tft.Z_80_WIDTH  # = 0.01
    for h in (1, 3, 5, 7):
        r = d[d["horizon"] == h].iloc[0]
        assert r["pred_cum_median"] == pytest.approx(0.01 * h)
        assert r["sigma_cum"] == pytest.approx(sigma_k * np.sqrt(h))
        assert r["p_up"] == pytest.approx(norm.cdf(0.01 * h / (sigma_k * np.sqrt(h))))
        assert r["pred_dir"] == 1.0
        assert r["fwd_dir"] == 1.0 and r["fwd_ret"] == pytest.approx(0.01 * h)


def test_majority_baseline_uses_same_rows():
    dir_df = pd.DataFrame(
        {
            "horizon": [1] * 6,
            "pred_dir": [1.0, 1.0, 0.0, 0.0, 1.0, 1.0],
            "fwd_dir": [1.0, 0.0, 0.0, 0.0, 0.0, np.nan],  # last row has no label and is dropped
            "p_up": [0.6] * 6,
        }
    )
    m = tft.directional_metrics(dir_df)["1"]
    y = pd.Series([1.0, 0.0, 0.0, 0.0, 0.0])
    assert m["n"] == 5
    assert m["majority_baseline"] == majority_baseline(y) == pytest.approx(0.8)
    assert m["accuracy"] == pytest.approx(3 / 5)
    assert m["always_up_accuracy"] == pytest.approx(0.2)


def test_test_phase_overwrite_guard(tmp_path, capsys):
    p = tmp_path / "test_a.json"
    tft.check_test_overwrite(p, force=False)  # absent: fine
    p.write_text("{}")
    with pytest.raises(FileExistsError):
        tft.check_test_overwrite(p, force=False)
    tft.check_test_overwrite(p, force=True)
    assert "WARNING" in capsys.readouterr().out

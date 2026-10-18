"""Temporal Fusion Transformer (pytorch-forecasting) library for Phase 7.

Design notes
------------
Target and normalizer
    The TFT forecasts the daily log return ``log_ret_1d`` for steps t+1..t+7
    from an encoder window ending at the forecast origin t. The target is
    scaled with ``GroupNormalizer(groups=["ticker"])`` (per-ticker standard
    scaling, no transformation because log returns are signed). It is fitted on
    the training rows only and reused (via ``from_dataset``) for validation and
    test. Reason: per-ticker volatility differs a lot (e.g. KO vs META), the
    scale is estimated from a long training history rather than from a 30-day
    window, and the fit is trivially auditable as "train only". An
    ``EncoderNormalizer`` re-scales each window by its own short history,
    which is noisier for a near-zero-mean target.

Leakage rules
    * ``train_ds`` is built from rows with ``time_idx <= train_end_idx`` only,
      so no decoder target crosses into validation.
    * ``val_ds``/``test_ds`` come from ``TimeSeriesDataSet.from_dataset`` with
      ``min_prediction_idx`` = first index of the evaluation window, so every
      decoder step lies inside the window (encoder context may reach earlier,
      which is past information and legitimate). Scalers, encoders and the
      target normalizer are those fitted on train.
    * Calendar columns are known-future covariates; everything else observed is
      encoder-only. ``fwd_*`` columns are never inputs
      (``assert_no_target_leakage``). ``log_ret_1d`` is the target and also its
      own history (standard TFT practice: past target values are encoder input).

Forecast origin mapping
    ``model.predict(..., return_index=True)`` reports, per sample, the
    ``time_idx`` of the FIRST decoder step. The origin t (last encoder day) is
    that value minus 1, and decoder step k (1..7) is date(origin + k).

Directional probability (approximation)
    pred_cum_median_h = sum_{k<=h} q50_k (median of a sum is not the sum of
    medians; this is an approximation that is exact for symmetric,
    independent steps). sigma_k = (q90_k - q10_k) / 2.563 (normal assumption,
    2 * 1.2816), and sigma_cum_h = sqrt(sum_{k<=h} sigma_k^2) assumes
    independent daily returns. p_up = Phi(pred_cum_median / sigma_cum).

CPU budget / ticker subset
    Benchmark (4 torch threads, CPU, default hparams, encoder 30, batch 128):
    F3 on all 28 tickers = 20,188 training windows = 157 batches/epoch at
    ~0.31 s/batch, i.e. ~48 s training + ~9 s validation/checkpoint per epoch
    (~57 s). A full fit of up to 15 epochs (~14 min worst case) exceeds the
    ~8 min budget, and the plan needs ~20 fits (stages a-e + tuning), so a
    fixed 16-ticker subset (``DEFAULT_TICKER_SUBSET``) is used: ~33 s/epoch,
    ~8 min worst case at 15 epochs, less with early stopping. The subset was
    fixed BEFORE any model result by a sector rule: the two largest-market-cap
    tickers of each of the 8 sectors present in the panel (approximate
    late-2025 market caps). It does not depend on model performance.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from scipy.stats import norm

from src.data.tft_dataset import (
    HORIZONS,
    KNOWN_CATEGORICALS,
    KNOWN_REALS,
    PRICE_FEATURES,
    SENTIMENT_FEATURES,
    TARGET_COLUMNS,
    TECHNICAL_FEATURES,
    assert_no_target_leakage,
)
from src.models.evaluate import majority_baseline

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = PROJECT_ROOT / "models" / "phase7"
REPORTS_DIR = PROJECT_ROOT / "reports" / "phase7"

TARGET = "log_ret_1d"
MAX_PREDICTION_LENGTH = 7
QUANTILES: list[float] = [0.02, 0.1, 0.25, 0.5, 0.75, 0.9, 0.98]
QCOLS: list[str] = ["q02", "q10", "q25", "q50", "q75", "q90", "q98"]
Z_80_WIDTH = 2.563  # q90 - q10 = 2 * 1.2816 sigma under normality

DEFAULT_HPARAMS: dict = {
    "hidden_size": 16,
    "attention_head_size": 2,
    "dropout": 0.1,
    "hidden_continuous_size": 8,
    "learning_rate": 1e-3,
    "encoder_length": 30,
}

FEATURE_SETS: dict[str, list[str]] = {
    "price": list(PRICE_FEATURES),
    "technical": list(PRICE_FEATURES) + list(TECHNICAL_FEATURES),
    "sentiment": list(PRICE_FEATURES) + list(TECHNICAL_FEATURES) + list(SENTIMENT_FEATURES),
}

# Fixed ticker subset decided BEFORE seeing any results (see module docstring):
# top-2 market cap per sector. None would mean all tickers in the panel.
TICKER_SUBSET_RULE = "two largest market-cap tickers per sector (8 sectors), fixed before any results"
DEFAULT_TICKER_SUBSET: list[str] | None = [
    "AAPL", "MSFT",  # technology
    "AMZN", "HD",  # consumer discretionary
    "GOOGL", "META",  # communication services
    "JPM", "BAC",  # financials
    "JNJ", "ABBV",  # health care
    "XOM", "CVX",  # energy
    "CAT", "BA",  # industrials
    "WMT", "PG",  # consumer staples
]


@dataclass(frozen=True)
class Split:
    name: str
    train_end: str
    val_start: str
    val_end: str
    test_start: str | None = None
    test_end: str | None = None


SPLITS: dict[str, Split] = {
    "F1": Split("F1", "2020-12-31", "2021-01-01", "2021-12-31"),
    "F2": Split("F2", "2021-12-31", "2022-01-01", "2022-12-31"),
    "F3": Split("F3", "2022-12-31", "2023-01-01", "2023-12-31"),
    "test": Split("test", "2022-12-31", "2023-01-01", "2023-12-31", "2024-01-01", "2025-12-31"),
    "sentiment_window": Split(
        "sentiment_window", "2025-09-30", "2025-10-01", "2025-10-31", "2025-11-01", "2025-12-31"
    ),
}


# ----------------------------------------------------------------------------
# Split / feature helpers
# ----------------------------------------------------------------------------

def _date_to_idx(panel: pd.DataFrame, date: str, side: str) -> int:
    """time_idx of the last trading day <= date (side='end') or first >= date ('start')."""
    cal = panel.drop_duplicates("time_idx").sort_values("time_idx")
    d = pd.Timestamp(date)
    if side == "end":
        sel = cal[cal["date"] <= d]
        if sel.empty:
            raise ValueError(f"No trading day <= {date}")
        return int(sel["time_idx"].iloc[-1])
    sel = cal[cal["date"] >= d]
    if sel.empty:
        raise ValueError(f"No trading day >= {date}")
    return int(sel["time_idx"].iloc[0])


def resolve_split(panel: pd.DataFrame, split: str | Split) -> dict:
    """Convert a split's dates into inclusive time_idx boundaries."""
    sp = SPLITS[split] if isinstance(split, str) else split
    out = {
        "name": sp.name,
        "train_end_idx": _date_to_idx(panel, sp.train_end, "end"),
        "val_start_idx": _date_to_idx(panel, sp.val_start, "start"),
        "val_end_idx": _date_to_idx(panel, sp.val_end, "end"),
        "test_start_idx": None,
        "test_end_idx": None,
        "dates": {
            "train_end": sp.train_end,
            "val_start": sp.val_start,
            "val_end": sp.val_end,
            "test_start": sp.test_start,
            "test_end": sp.test_end,
        },
    }
    if sp.test_start is not None:
        out["test_start_idx"] = _date_to_idx(panel, sp.test_start, "start")
        out["test_end_idx"] = _date_to_idx(panel, sp.test_end, "end")
    if not (out["train_end_idx"] < out["val_start_idx"] <= out["val_end_idx"]):
        raise ValueError(f"Inconsistent split boundaries: {out}")
    return out


def feature_columns(feature_set: str) -> list[str]:
    """Observed (encoder-only) real columns for a feature set, leakage-checked."""
    if feature_set not in FEATURE_SETS:
        raise ValueError(f"Unknown feature set {feature_set!r}; choose from {list(FEATURE_SETS)}")
    cols = list(FEATURE_SETS[feature_set])
    assert_no_target_leakage(cols + KNOWN_REALS + KNOWN_CATEGORICALS)
    return cols


def select_tickers(panel: pd.DataFrame, tickers: Sequence[str] | None) -> pd.DataFrame:
    if not tickers:
        return panel
    missing = sorted(set(tickers) - set(panel["ticker"].unique()))
    if missing:
        raise ValueError(f"Tickers not in panel: {missing}")
    return panel[panel["ticker"].isin(list(tickers))]


def _prepare(panel: pd.DataFrame) -> pd.DataFrame:
    """Copy with categorical columns as strings (pytorch-forecasting requirement)."""
    df = panel.sort_values(["ticker", "time_idx"]).reset_index(drop=True).copy()
    for c in KNOWN_CATEGORICALS:
        df[c] = df[c].astype(int).astype(str)
    df["ticker"] = df["ticker"].astype(str)
    return df


def make_datasets(
    panel: pd.DataFrame,
    split: str | Split,
    feature_set: str,
    encoder_length: int = 30,
    max_prediction_length: int = MAX_PREDICTION_LENGTH,
):
    """Build (train_ds, val_ds, test_ds | None) with train-only fitted scalers."""
    from pytorch_forecasting import GroupNormalizer, TimeSeriesDataSet
    from pytorch_forecasting.data.encoders import NaNLabelEncoder

    obs = feature_columns(feature_set)
    bounds = resolve_split(panel, split)
    df = _prepare(panel)

    unknown_reals = list(obs)
    if TARGET not in unknown_reals:
        unknown_reals = [TARGET] + unknown_reals  # past target history (encoder only)
    assert_no_target_leakage(unknown_reals)

    train_df = df[df["time_idx"] <= bounds["train_end_idx"]]
    train_ds = TimeSeriesDataSet(
        train_df,
        time_idx="time_idx",
        target=TARGET,
        group_ids=["ticker"],
        min_encoder_length=encoder_length,
        max_encoder_length=encoder_length,
        min_prediction_length=max_prediction_length,
        max_prediction_length=max_prediction_length,
        static_categoricals=["ticker"],
        time_varying_known_categoricals=list(KNOWN_CATEGORICALS),
        time_varying_known_reals=list(KNOWN_REALS),
        time_varying_unknown_reals=unknown_reals,
        target_normalizer=GroupNormalizer(groups=["ticker"], transformation=None),
        categorical_encoders={
            c: NaNLabelEncoder(add_nan=True) for c in ["ticker"] + list(KNOWN_CATEGORICALS)
        },
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=False,
        allow_missing_timesteps=False,
    )

    val_df = df[df["time_idx"] <= bounds["val_end_idx"]]
    val_ds = TimeSeriesDataSet.from_dataset(
        train_ds,
        val_df,
        min_prediction_idx=bounds["val_start_idx"],
        stop_randomization=True,
        predict=False,
    )
    test_ds = None
    if bounds["test_start_idx"] is not None:
        test_df = df[df["time_idx"] <= bounds["test_end_idx"]]
        test_ds = TimeSeriesDataSet.from_dataset(
            train_ds,
            test_df,
            min_prediction_idx=bounds["test_start_idx"],
            stop_randomization=True,
            predict=False,
        )
    return train_ds, val_ds, test_ds


# ----------------------------------------------------------------------------
# Model / training
# ----------------------------------------------------------------------------

def build_model(train_ds, hparams: dict | None = None):
    from pytorch_forecasting import TemporalFusionTransformer
    from pytorch_forecasting.metrics import QuantileLoss

    hp = {**DEFAULT_HPARAMS, **(hparams or {})}
    return TemporalFusionTransformer.from_dataset(
        train_ds,
        learning_rate=hp["learning_rate"],
        hidden_size=hp["hidden_size"],
        attention_head_size=hp["attention_head_size"],
        dropout=hp["dropout"],
        hidden_continuous_size=hp["hidden_continuous_size"],
        loss=QuantileLoss(quantiles=QUANTILES),
        log_interval=-1,
        reduce_on_plateau_patience=2,
        optimizer="adam",
    )


def train_tft(
    train_ds,
    val_ds,
    hparams: dict | None,
    max_epochs: int,
    seed: int,
    ckpt_dir: str | Path,
    limit_train_batches: int | None = None,
    batch_size: int = 128,
    patience: int = 3,
):
    """Train on CPU; return the model restored from its best val_loss checkpoint.

    The returned model carries ``epochs_trained``, ``best_ckpt_path`` and
    ``best_val_loss`` attributes.
    """
    import lightning.pytorch as pl
    import torch
    from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
    from pytorch_forecasting import TemporalFusionTransformer

    torch.set_num_threads(4)
    pl.seed_everything(seed, workers=True)
    ckpt_dir = Path(ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    for old in ckpt_dir.glob("*.ckpt"):
        old.unlink()

    model = build_model(train_ds, hparams)
    train_dl = train_ds.to_dataloader(train=True, batch_size=batch_size, num_workers=0)
    val_dl = val_ds.to_dataloader(train=False, batch_size=batch_size * 2, num_workers=0)

    ckpt_cb = ModelCheckpoint(
        dirpath=str(ckpt_dir), filename="best", monitor="val_loss", mode="min", save_top_k=1
    )
    trainer = pl.Trainer(
        accelerator="cpu",
        devices=1,
        max_epochs=max_epochs,
        gradient_clip_val=0.1,
        limit_train_batches=limit_train_batches if limit_train_batches else 1.0,
        callbacks=[EarlyStopping(monitor="val_loss", mode="min", patience=patience), ckpt_cb],
        logger=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        deterministic=False,
    )
    trainer.fit(model, train_dataloaders=train_dl, val_dataloaders=val_dl)
    best_path = ckpt_cb.best_model_path
    best = TemporalFusionTransformer.load_from_checkpoint(best_path)
    best.epochs_trained = int(trainer.current_epoch)
    best.best_ckpt_path = str(best_path)
    best.best_val_loss = float(ckpt_cb.best_model_score) if ckpt_cb.best_model_score is not None else float("nan")
    return best


def load_model(ckpt_path: str | Path):
    from pytorch_forecasting import TemporalFusionTransformer

    return TemporalFusionTransformer.load_from_checkpoint(str(ckpt_path))


# ----------------------------------------------------------------------------
# Predictions
# ----------------------------------------------------------------------------

def _pinball(actual: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Mean pinball loss over QUANTILES per row. actual (n,), q (n, 7)."""
    qs = np.asarray(QUANTILES)[None, :]
    err = actual[:, None] - q
    return np.maximum(qs * err, (qs - 1) * err).mean(axis=1)


def predict_frame(model, ds, panel: pd.DataFrame, batch_size: int = 256) -> pd.DataFrame:
    """Long frame: ticker, origin_idx, origin_date, step, step_date, q02..q98, actual, pinball.

    ``origin`` is the last encoder day t; step k is date(t + k) (decoder step 1 = t+1).
    ``actual`` is the realised daily log return at t+k, taken from the panel.
    """
    res = model.predict(
        ds,
        mode="quantiles",
        return_index=True,
        batch_size=batch_size,
        num_workers=0,
        trainer_kwargs={
            "accelerator": "cpu",
            "logger": False,
            "enable_progress_bar": False,
            "enable_model_summary": False,
        },
    )
    out = res.output.detach().cpu().numpy() if hasattr(res.output, "detach") else np.asarray(res.output)
    idx = res.index.reset_index(drop=True)
    n, steps, nq = out.shape
    if nq != len(QCOLS):
        raise ValueError(f"Expected {len(QCOLS)} quantiles, got {nq}")

    origin_idx = idx["time_idx"].to_numpy() - 1  # time_idx is the first decoder step
    tick = idx["ticker"].astype(str).to_numpy()

    cal = panel.drop_duplicates("time_idx").set_index("time_idx")["date"]
    daily = panel.set_index(["ticker", "time_idx"])[TARGET]

    rep = lambda a: np.repeat(a, steps)  # noqa: E731
    frame = pd.DataFrame(
        {
            "ticker": rep(tick),
            "origin_idx": rep(origin_idx),
            "step": np.tile(np.arange(1, steps + 1), n),
        }
    )
    frame["origin_date"] = frame["origin_idx"].map(cal)
    frame["step_idx"] = frame["origin_idx"] + frame["step"]
    frame["step_date"] = frame["step_idx"].map(cal)
    qarr = out.reshape(n * steps, nq)
    for j, c in enumerate(QCOLS):
        frame[c] = qarr[:, j]
    keys = pd.MultiIndex.from_arrays([frame["ticker"], frame["step_idx"]])
    frame["actual"] = daily.reindex(keys).to_numpy()
    ok = frame["actual"].notna().to_numpy()
    pin = np.full(len(frame), np.nan)
    pin[ok] = _pinball(frame.loc[ok, "actual"].to_numpy(), qarr[ok])
    frame["pinball"] = pin
    return frame.sort_values(["ticker", "origin_idx", "step"]).reset_index(drop=True)


def average_pred_frames(frames: Sequence[pd.DataFrame]) -> pd.DataFrame:
    """Seed ensemble: average quantile columns across frames with identical keys."""
    if len(frames) == 1:
        return frames[0]
    base = frames[0].copy()
    key = ["ticker", "origin_idx", "step"]
    for f in frames[1:]:
        if len(f) != len(base) or not (f[key].to_numpy() == base[key].to_numpy()).all():
            raise ValueError("Cannot average prediction frames with different keys")
    for c in QCOLS:
        base[c] = np.mean([f[c].to_numpy() for f in frames], axis=0)
    ok = base["actual"].notna().to_numpy()
    pin = np.full(len(base), np.nan)
    pin[ok] = _pinball(base.loc[ok, "actual"].to_numpy(), base.loc[ok, QCOLS].to_numpy())
    base["pinball"] = pin
    return base


def directional_frame(
    pred_df: pd.DataFrame, panel: pd.DataFrame, horizons: Sequence[int] = HORIZONS
) -> pd.DataFrame:
    """Per (ticker, origin, horizon) directional probability plus realised labels."""
    df = pred_df.sort_values(["ticker", "origin_idx", "step"]).copy()
    df["sigma_k"] = (df["q90"] - df["q10"]) / Z_80_WIDTH
    df["sigma_sq"] = df["sigma_k"] ** 2
    g = df.groupby(["ticker", "origin_idx"], sort=False)
    df["pred_cum_median"] = g["q50"].cumsum()
    df["sigma_cum"] = np.sqrt(g["sigma_sq"].cumsum())

    parts = []
    for h in horizons:
        sub = df[df["step"] == h]
        parts.append(
            pd.DataFrame(
                {
                    "ticker": sub["ticker"].to_numpy(),
                    "origin_idx": sub["origin_idx"].to_numpy(),
                    "origin_date": sub["origin_date"].to_numpy(),
                    "horizon": h,
                    "pred_cum_median": sub["pred_cum_median"].to_numpy(),
                    "sigma_cum": sub["sigma_cum"].to_numpy(),
                    "step_actual": sub["actual"].to_numpy() if "actual" in sub else np.nan,
                    "step_q10": sub["q10"].to_numpy(),
                    "step_q50": sub["q50"].to_numpy(),
                    "step_q90": sub["q90"].to_numpy(),
                    "step_pinball": sub["pinball"].to_numpy() if "pinball" in sub else np.nan,
                }
            )
        )
    out = pd.concat(parts, ignore_index=True)
    sig = out["sigma_cum"].to_numpy()
    safe = sig > 1e-12
    z = np.where(safe, out["pred_cum_median"].to_numpy() / np.where(safe, sig, 1.0), 0.0)
    out["p_up"] = norm.cdf(z)
    out["pred_dir"] = (out["p_up"] >= 0.5).astype(float)

    lab = panel.set_index(["ticker", "time_idx"])
    keys = pd.MultiIndex.from_arrays([out["ticker"], out["origin_idx"]])
    out["fwd_ret"] = np.nan
    out["fwd_dir"] = np.nan
    for h in horizons:
        m = (out["horizon"] == h).to_numpy()
        k = keys[m]
        out.loc[m, "fwd_ret"] = lab[f"fwd_ret_{h}d"].reindex(k).to_numpy()
        out.loc[m, "fwd_dir"] = lab[f"fwd_dir_{h}d"].reindex(k).to_numpy()
    return out.sort_values(["horizon", "ticker", "origin_idx"]).reset_index(drop=True)


def directional_metrics(dir_df: pd.DataFrame) -> dict:
    """Per-horizon metrics, always next to the majority baseline of the same rows."""
    result: dict[str, dict] = {}
    for h, g in dir_df.groupby("horizon"):
        g = g.dropna(subset=["fwd_dir"])
        if g.empty:
            continue
        y = g["fwd_dir"]
        m = {
            "n": int(len(g)),
            "accuracy": float((g["pred_dir"] == y).mean()),
            "majority_baseline": float(majority_baseline(y)),
            "always_up_accuracy": float((y == 1).mean()),
            "frac_pred_up": float(g["pred_dir"].mean()),
        }
        m["accuracy_minus_majority"] = m["accuracy"] - m["majority_baseline"]
        if "step_actual" in g and g["step_actual"].notna().any():
            s = g.dropna(subset=["step_actual"])
            m["step_pinball_loss"] = float(s["step_pinball"].mean())
            m["step_median_mae"] = float((s["step_actual"] - s["step_q50"]).abs().mean())
            m["step_q10_q90_coverage"] = float(
                ((s["step_actual"] >= s["step_q10"]) & (s["step_actual"] <= s["step_q90"])).mean()
            )
        result[str(int(h))] = m
    return result


def summarize_metrics(metrics: dict) -> dict:
    """Scalar summary (mean over horizons) of a ``directional_metrics`` result."""
    accs = [m["accuracy"] for m in metrics.values()]
    bases = [m["majority_baseline"] for m in metrics.values()]
    return {
        "mean_accuracy": float(np.mean(accs)) if accs else float("nan"),
        "mean_majority_baseline": float(np.mean(bases)) if bases else float("nan"),
    }


# ----------------------------------------------------------------------------
# Misc helpers shared with the runner
# ----------------------------------------------------------------------------

def hparams_key(hparams: dict) -> str:
    hp = {**DEFAULT_HPARAMS, **hparams}
    return hashlib.md5(json.dumps(hp, sort_keys=True).encode()).hexdigest()[:8]


def check_test_overwrite(path: str | Path, force: bool = False) -> None:
    """Refuse to overwrite an existing held-out test result unless forced."""
    p = Path(path)
    if p.exists():
        if not force:
            raise FileExistsError(
                f"{p} already exists. The held-out test set must be evaluated ONCE; "
                "pass --force only if you knowingly accept re-evaluating it."
            )
        print(f"WARNING: --force given; OVERWRITING existing held-out test result {p}. "
              "Re-evaluating the test set invalidates it as an unbiased estimate.")


class Timer:
    def __enter__(self):
        self.t0 = time.time()
        return self

    def __exit__(self, *exc):
        self.seconds = time.time() - self.t0
        return False

"""Phase 7 runner: staged TFT experiments + LightGBM baseline.

Stages
    a      price-only TFT, default hyperparameters
    b      + technical indicators
    c      + FinBERT sentiment
    d      best feature set from a-c (mean CV validation accuracy) with
           hyperparameters tuned on F3 validation only (bounded random search)
    e      stage d model + per-horizon confidence threshold chosen on F3 validation
    lgbm   LightGBM classifier per horizon on the same targets/splits

Phases
    cv                 walk-forward folds F1-F3 (validation metrics)
    test               F3-trained model evaluated ONCE on 2024-01-01..2025-12-31
    sentiment_window   technical vs sentiment ablation on Nov-Dec 2025

Results go to reports/phase7/ (reports/phase7/smoke/ with --smoke). Model
checkpoints and cached predictions go to models/phase7/ (gitignored).

Examples
    python scripts/run_phase7_tft.py --stage a --phase cv
    python scripts/run_phase7_tft.py --stage a --phase cv --smoke
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.tft_dataset import (  # noqa: E402
    HORIZONS,
    KNOWN_CATEGORICALS,
    KNOWN_REALS,
    assert_no_target_leakage,
)
from src.models import tft  # noqa: E402
from src.models.evaluate import accuracy_at_threshold, majority_baseline  # noqa: E402

PANEL_PATH = ROOT / "data" / "processed" / "tft_panel.parquet"
CV_FOLDS = ["F1", "F2", "F3"]
SMOKE_TICKERS = ["AAPL", "JPM", "XOM"]
TUNING_GRID = {
    "hidden_size": [8, 16, 32],
    "dropout": [0.1, 0.3],
    "learning_rate": [1e-3, 3e-3],
    "encoder_length": [20, 60],
}
N_TUNING_CONFIGS = 6
THRESHOLD_GRID = [round(0.50 + 0.01 * i, 2) for i in range(21)]  # 0.50 .. 0.70
MIN_COVERAGE = 0.20
STAGE_FEATURES = {"a": "price", "b": "technical", "c": "sentiment"}


# ----------------------------------------------------------------------------
# Context / IO
# ----------------------------------------------------------------------------

class Ctx:
    def __init__(self, args):
        self.args = args
        self.smoke = args.smoke
        self.out_dir = tft.REPORTS_DIR / ("smoke" if args.smoke else "")
        self.cache_dir = tft.MODELS_DIR / ("smoke" if args.smoke else "full")
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.seeds = [int(s) for s in str(args.seeds).split(",")]
        if args.smoke:
            self.max_epochs = 1
            self.limit = args.limit_train_batches or 5
        else:
            self.max_epochs = args.max_epochs
            self.limit = args.limit_train_batches
        self.tickers = resolve_tickers(args)
        self.tick_hash = tft.hparams_key({"t": ",".join(self.tickers)})

    def path(self, name: str) -> Path:
        return self.out_dir / name


def resolve_tickers(args) -> list[str]:
    panel_tickers = sorted(pd.read_parquet(PANEL_PATH, columns=["ticker"])["ticker"].unique())
    if args.smoke:
        return [t for t in SMOKE_TICKERS if t in panel_tickers] or panel_tickers[:3]
    if args.tickers == "default":
        return list(tft.DEFAULT_TICKER_SUBSET or panel_tickers)
    if args.tickers == "all":
        return panel_tickers
    return [t.strip() for t in args.tickers.split(",")]


def _clean(o):
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return None if (math.isnan(o) or math.isinf(o)) else float(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, pd.Timestamp):
        return o.strftime("%Y-%m-%d")
    return o


def write_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_clean(obj), indent=2))
    print(f"wrote {path}")


def read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def load_panel(ctx: Ctx) -> pd.DataFrame:
    panel = pd.read_parquet(PANEL_PATH)
    panel = tft.select_tickers(panel, ctx.tickers)
    return panel.reset_index(drop=True)


# ----------------------------------------------------------------------------
# TFT fold fitting with caching
# ----------------------------------------------------------------------------

def cache_key(ctx: Ctx, fold: str, feature_set: str, hp: dict, seed: int) -> str:
    hp = {**tft.DEFAULT_HPARAMS, **hp}
    return (
        f"{feature_set}_{tft.hparams_key(hp)}_{fold}_s{seed}_{ctx.tick_hash}"
        f"_e{ctx.max_epochs}_l{ctx.limit}"
    )


def fit_fold(ctx: Ctx, panel, fold: str, feature_set: str, hp: dict, seed: int) -> dict:
    """Train (or load cached) TFT for ``fold``; return info + validation-window preds."""
    hp = {**tft.DEFAULT_HPARAMS, **hp}
    key = cache_key(ctx, fold, feature_set, hp, seed)
    info_path = ctx.cache_dir / f"{key}.json"
    pred_path = ctx.cache_dir / f"{key}.parquet"
    if info_path.exists() and pred_path.exists() and not ctx.args.refit:
        info = read_json(info_path)
        if Path(info["ckpt_path"]).exists():
            print(f"[cache] {key}")
            return {"info": info, "pred": pd.read_parquet(pred_path)}

    t0 = time.time()
    train_ds, val_ds, _ = tft.make_datasets(panel, fold, feature_set, hp["encoder_length"])
    model = tft.train_tft(
        train_ds, val_ds, hp, ctx.max_epochs, seed, ctx.cache_dir / "ckpt" / key, ctx.limit
    )
    pred = tft.predict_frame(model, val_ds, panel)
    info = {
        "key": key,
        "fold": fold,
        "feature_set": feature_set,
        "hparams": hp,
        "seed": seed,
        "epochs_trained": model.epochs_trained,
        "best_val_loss": model.best_val_loss,
        "n_train_samples": len(train_ds),
        "n_val_samples": len(val_ds),
        "ckpt_path": model.best_ckpt_path,
        "wall_time_s": time.time() - t0,
    }
    pred.to_parquet(pred_path)
    write_json(info_path, info)
    return {"info": info, "pred": pred}


def ensemble_fold(ctx: Ctx, panel, fold, feature_set, hp) -> dict:
    runs = [fit_fold(ctx, panel, fold, feature_set, hp, s) for s in ctx.seeds]
    pred = tft.average_pred_frames([r["pred"] for r in runs])
    dir_df = tft.directional_frame(pred, panel)
    return {"infos": [r["info"] for r in runs], "pred": pred, "dir": dir_df}


def fold_record(split_name: str, panel, res: dict) -> dict:
    bounds = tft.resolve_split(panel, split_name)
    metrics = tft.directional_metrics(res["dir"])
    rec = {
        "split": split_name,
        "dates": bounds["dates"],
        "metrics": metrics,
        "summary": tft.summarize_metrics(metrics),
        "pinball_loss_all_steps": float(res["pred"]["pinball"].mean()),
        "wall_time_s": sum(i["wall_time_s"] for i in res["infos"]),
        "epochs_trained": [i["epochs_trained"] for i in res["infos"]],
        "best_val_loss": [i["best_val_loss"] for i in res["infos"]],
        "n_train_samples": res["infos"][0]["n_train_samples"],
    }
    return rec


def common_header(ctx: Ctx, stage: str, phase: str, feature_set: str | None, hp: dict | None) -> dict:
    return {
        "stage": stage,
        "phase": phase,
        "smoke": ctx.smoke,
        "tickers": ctx.tickers,
        "n_tickers": len(ctx.tickers),
        "ticker_rule": ("smoke subset" if ctx.smoke else tft.TICKER_SUBSET_RULE if ctx.args.tickers == "default" else "user-specified"),
        "feature_set": feature_set,
        "features": {
            "observed_reals": tft.FEATURE_SETS.get(feature_set, []) if feature_set else [],
            "known_reals": KNOWN_REALS,
            "known_categoricals": KNOWN_CATEGORICALS,
            "static_categoricals": ["ticker"],
            "target": tft.TARGET,
        },
        "hparams": ({**tft.DEFAULT_HPARAMS, **hp} if hp is not None else None),
        "seeds": ctx.seeds,
        "max_epochs": ctx.max_epochs,
        "limit_train_batches": ctx.limit,
        "quantiles": tft.QUANTILES,
    }


# ----------------------------------------------------------------------------
# Stage configuration
# ----------------------------------------------------------------------------

def mean_horizon_acc(metrics: dict) -> float:
    return float(np.mean([m["accuracy"] for m in metrics.values()]))


def stage_config(ctx: Ctx, stage: str) -> tuple[str, dict]:
    if stage in STAGE_FEATURES:
        return STAGE_FEATURES[stage], dict(tft.DEFAULT_HPARAMS)
    frozen = ctx.path("frozen_d.json")
    if not frozen.exists():
        raise SystemExit(f"{frozen} missing: run '--stage d --phase cv' first.")
    f = read_json(frozen)
    return f["feature_set"], f["hparams"]


def pick_feature_set(ctx: Ctx) -> tuple[str, dict]:
    scores = {}
    for st, fs in STAGE_FEATURES.items():
        p = ctx.path(f"cv_{st}.json")
        if not p.exists():
            raise SystemExit(f"{p} missing: run stages a, b, c (cv) before stage d.")
        r = read_json(p)
        if not r.get("complete"):
            raise SystemExit(f"{p} is incomplete (not all folds present).")
        scores[fs] = r["cv_mean"]["mean_accuracy"]
    best = max(scores, key=scores.get)
    return best, scores


def tuning_configs() -> list[dict]:
    """Default config + (N-1) seeded random draws from the grid (deterministic)."""
    rng = random.Random(0)
    keys = sorted(TUNING_GRID)
    combos = [dict(zip(keys, v)) for v in itertools.product(*(TUNING_GRID[k] for k in keys))]
    default = {k: tft.DEFAULT_HPARAMS[k] for k in keys}
    pool = [c for c in combos if c != default]
    cfgs = [default] + rng.sample(pool, N_TUNING_CONFIGS - 1)
    for c in cfgs:
        c["attention_head_size"] = 2 if c["hidden_size"] >= 16 else 1
        c["hidden_continuous_size"] = min(8, c["hidden_size"])
    return cfgs


def run_tuning(ctx: Ctx, panel, feature_set: str, fs_scores: dict) -> dict:
    """Evaluate every config on F3 validation ONLY; pick the best; freeze."""
    log = []
    for i, cfg in enumerate(tuning_configs()):
        hp = {**tft.DEFAULT_HPARAMS, **cfg}
        res = ensemble_fold(ctx, panel, "F3", feature_set, hp)
        metrics = tft.directional_metrics(res["dir"])
        entry = {
            "config_id": i,
            "hparams": hp,
            "val_split": "F3 (2023)",
            "val_mean_accuracy": mean_horizon_acc(metrics),
            "val_mean_majority_baseline": tft.summarize_metrics(metrics)["mean_majority_baseline"],
            "val_pinball_loss": float(res["pred"]["pinball"].mean()),
            "epochs_trained": [x["epochs_trained"] for x in res["infos"]],
            "wall_time_s": sum(x["wall_time_s"] for x in res["infos"]),
            "per_horizon_accuracy": {h: m["accuracy"] for h, m in metrics.items()},
        }
        log.append(entry)
        print(f"[tuning] cfg {i}: acc={entry['val_mean_accuracy']:.4f} pinball={entry['val_pinball_loss']:.5f} {cfg}")
        write_json(ctx.path("tuning_d.json"), {"stage": "d", "partial": True, "configs_tried": log})
    # Select on F3 validation mean accuracy; ties -> lower pinball loss. Never on test.
    best = sorted(log, key=lambda e: (-round(e["val_mean_accuracy"], 6), e["val_pinball_loss"]))[0]
    write_json(
        ctx.path("tuning_d.json"),
        {
            "stage": "d",
            "partial": False,
            "selection_rule": "max mean directional accuracy over horizons on F3 validation; tie -> lower pinball loss",
            "feature_set": feature_set,
            "feature_set_cv_scores": fs_scores,
            "grid": TUNING_GRID,
            "configs_tried": log,
            "selected_config_id": best["config_id"],
        },
    )
    frozen = {
        "stage": "d",
        "feature_set": feature_set,
        "hparams": best["hparams"],
        "selected_config_id": best["config_id"],
        "selected_on": "F3 validation (2023)",
        "tickers": ctx.tickers,
        "seeds": ctx.seeds,
        "max_epochs": ctx.max_epochs,
        "limit_train_batches": ctx.limit,
    }
    write_json(ctx.path("frozen_d.json"), frozen)
    return frozen


# ----------------------------------------------------------------------------
# CV phase
# ----------------------------------------------------------------------------

def assemble_cv(ctx, stage, panel, feature_set, hp, folds_res: dict, extra: dict) -> dict:
    out = common_header(ctx, stage, "cv", feature_set, hp)
    out["split_type"] = "walk-forward expanding CV, validation folds"
    out["folds"] = {f: fold_record(f, panel, r) for f, r in folds_res.items()}
    out["complete"] = all(f in out["folds"] for f in CV_FOLDS)
    accs = {h: [] for h in map(str, HORIZONS)}
    bases = {h: [] for h in map(str, HORIZONS)}
    for rec in out["folds"].values():
        for h, m in rec["metrics"].items():
            accs[h].append(m["accuracy"])
            bases[h].append(m["majority_baseline"])
    per_h = {
        h: {"mean_accuracy": float(np.mean(a)), "mean_majority_baseline": float(np.mean(bases[h]))}
        for h, a in accs.items() if a
    }
    out["cv_mean"] = {
        "mean_accuracy": float(np.mean([v["mean_accuracy"] for v in per_h.values()])),
        "mean_majority_baseline": float(np.mean([v["mean_majority_baseline"] for v in per_h.values()])),
        "per_horizon": per_h,
        "folds_included": list(out["folds"]),
    }
    out["validation_F3"] = out["folds"].get("F3")
    out.update(extra)
    return out


def _all_cached(ctx, fold, feature_set, hp) -> bool:
    for s in ctx.seeds:
        key = cache_key(ctx, fold, feature_set, hp, s)
        if not ((ctx.cache_dir / f"{key}.json").exists() and (ctx.cache_dir / f"{key}.parquet").exists()):
            return False
    return True


def phase_cv(ctx: Ctx) -> None:
    stage = ctx.args.stage
    if stage == "lgbm":
        return phase_cv_lgbm(ctx)
    t0 = time.time()
    panel = load_panel(ctx)
    if stage == "e":
        return phase_cv_e(ctx, panel)

    extra: dict = {}
    if stage == "d":
        frozen_path = ctx.path("frozen_d.json")
        if frozen_path.exists() and not ctx.args.retune:
            frozen = read_json(frozen_path)
        else:
            fs, scores = pick_feature_set(ctx)
            frozen = run_tuning(ctx, panel, fs, scores)
        extra["tuning_log"] = str(ctx.path("tuning_d.json").relative_to(ROOT))
        extra["selected_on"] = frozen["selected_on"]
        feature_set, hp = frozen["feature_set"], frozen["hparams"]
    else:
        feature_set, hp = stage_config(ctx, stage)

    requested = ctx.args.folds.split(",")
    folds_res = {}
    for f in CV_FOLDS:
        # chunked runs: recompute requested folds, reuse any already-cached fold
        if f in requested or _all_cached(ctx, f, feature_set, hp):
            folds_res[f] = ensemble_fold(ctx, panel, f, feature_set, hp)
    out = assemble_cv(ctx, stage, panel, feature_set, hp, folds_res, extra)
    out["wall_time_s"] = time.time() - t0
    write_json(ctx.path(f"cv_{stage}.json"), out)
    print_summary(out)


def thresholded(dir_df: pd.DataFrame, thresholds: dict) -> dict:
    """accuracy_at_threshold per horizon (evaluate.py), plus all-rows majority baseline."""
    out = {}
    for h in map(str, HORIZONS):
        g = dir_df[dir_df["horizon"] == int(h)].dropna(subset=["fwd_dir"])
        if g.empty:
            continue
        t = thresholds[h]
        r = accuracy_at_threshold(g["fwd_dir"].reset_index(drop=True), g["p_up"].to_numpy(), t)
        r["threshold"] = t
        r["n_total"] = int(len(g))
        r["majority_baseline_all_rows"] = float(majority_baseline(g["fwd_dir"]))
        out[h] = r
    return out


def choose_thresholds(dir_df: pd.DataFrame) -> tuple[dict, dict]:
    """Per-horizon threshold maximising accuracy with coverage >= MIN_COVERAGE (validation only)."""
    chosen, grid_log = {}, {}
    for h in map(str, HORIZONS):
        g = dir_df[dir_df["horizon"] == int(h)].dropna(subset=["fwd_dir"])
        y = g["fwd_dir"].reset_index(drop=True)
        rows = [{"threshold": t, **accuracy_at_threshold(y, g["p_up"].to_numpy(), t)} for t in THRESHOLD_GRID]
        grid_log[h] = rows
        ok = [r for r in rows if r["coverage"] >= MIN_COVERAGE and not math.isnan(r["accuracy"])]
        if not ok:  # no threshold meets the coverage floor: fall back to no filtering
            chosen[h] = THRESHOLD_GRID[0]
            continue
        best = max(ok, key=lambda r: (round(r["accuracy"], 6), -r["threshold"]))
        chosen[h] = best["threshold"]
    return chosen, grid_log


def phase_cv_e(ctx: Ctx, panel) -> None:
    t0 = time.time()
    feature_set, hp = stage_config(ctx, "e")
    res = {f: ensemble_fold(ctx, panel, f, feature_set, hp) for f in CV_FOLDS}
    thresholds, grid_log = choose_thresholds(res["F3"]["dir"])
    out = common_header(ctx, "e", "cv", feature_set, hp)
    out["split_type"] = "walk-forward CV; threshold chosen on F3 validation only"
    out["threshold_rule"] = (
        f"max accuracy s.t. coverage >= {MIN_COVERAGE}, grid {THRESHOLD_GRID[0]}..{THRESHOLD_GRID[-1]}"
    )
    out["thresholds"] = thresholds
    out["threshold_grid_F3"] = grid_log
    out["folds"] = {}
    for f, r in res.items():
        out["folds"][f] = {
            "threshold_selected_on_this_fold": f == "F3",
            "metrics_no_threshold": tft.directional_metrics(r["dir"]),
            "thresholded": thresholded(r["dir"], thresholds),
        }
    out["complete"] = True
    out["wall_time_s"] = time.time() - t0
    write_json(ctx.path("cv_e.json"), out)
    write_json(
        ctx.path("frozen_e.json"),
        {"stage": "e", "thresholds": thresholds, "selected_on": "F3 validation (2023)", "min_coverage": MIN_COVERAGE},
    )
    for f, rec in out["folds"].items():
        for h, r in rec["thresholded"].items():
            print(
                f"e {f} h={h}: t={r['threshold']} acc={r['accuracy']:.3f} "
                f"cov={r['coverage']:.2f} local_base={r['local_majority_baseline']:.3f}"
            )


# ----------------------------------------------------------------------------
# Test phase
# ----------------------------------------------------------------------------

def test_predictions(ctx: Ctx, panel, feature_set: str, hp: dict) -> dict:
    """F3-trained model(s) -> predictions over the held-out window (evaluated here once)."""
    hp = {**tft.DEFAULT_HPARAMS, **hp}
    frames, infos = [], []
    for s in ctx.seeds:
        r = fit_fold(ctx, panel, "F3", feature_set, hp, s)
        model = tft.load_model(r["info"]["ckpt_path"])
        _, _, test_ds = tft.make_datasets(panel, "test", feature_set, hp["encoder_length"])
        frames.append(tft.predict_frame(model, test_ds, panel))
        infos.append(r["info"])
    pred = tft.average_pred_frames(frames)
    return {"infos": infos, "pred": pred, "dir": tft.directional_frame(pred, panel)}


def phase_test(ctx: Ctx) -> None:
    stage = ctx.args.stage
    out_path = ctx.path(f"test_{stage}.json")
    cv_path = ctx.path(f"cv_{stage}.json")
    if not cv_path.exists():
        raise SystemExit(f"{cv_path} missing: run '--stage {stage} --phase cv' before test.")
    cv = read_json(cv_path)
    if not cv.get("complete"):
        raise SystemExit(f"{cv_path} is not complete.")
    if stage in ("d", "e") and not ctx.path("frozen_d.json").exists():
        raise SystemExit("frozen_d.json missing (frozen config required for stage d/e test).")
    if stage == "e" and not ctx.path("frozen_e.json").exists():
        raise SystemExit("frozen_e.json missing (frozen threshold required for stage e test).")
    tft.check_test_overwrite(out_path, ctx.args.force)

    t0 = time.time()
    panel = load_panel(ctx)
    if stage == "lgbm":
        return phase_test_lgbm(ctx, panel, out_path, cv)
    feature_set, hp = stage_config(ctx, stage)
    res = test_predictions(ctx, panel, feature_set, hp)
    out = common_header(ctx, stage, "test", feature_set, hp)
    out["split_type"] = "held-out test (model trained <=2022-12-31, early stopping on 2023)"
    out["dates"] = tft.resolve_split(panel, "test")["dates"]
    out["metrics"] = tft.directional_metrics(res["dir"])
    out["summary"] = tft.summarize_metrics(out["metrics"])
    out["pinball_loss_all_steps"] = float(res["pred"]["pinball"].mean())
    out["epochs_trained"] = [i["epochs_trained"] for i in res["infos"]]
    out["n_train_samples"] = res["infos"][0]["n_train_samples"]
    if stage == "e":
        thr = read_json(ctx.path("frozen_e.json"))["thresholds"]
        out["thresholds"] = thr
        out["thresholded"] = thresholded(res["dir"], thr)
    out["validation_reference_F3"] = cv.get("validation_F3") or cv["folds"].get("F3")
    out["cv_mean_reference"] = cv.get("cv_mean")
    out["wall_time_s"] = time.time() - t0
    write_json(out_path, out)
    print_summary(out)


# ----------------------------------------------------------------------------
# Sentiment window phase
# ----------------------------------------------------------------------------

def phase_sentiment_window(ctx: Ctx) -> None:
    out_path = ctx.path("sentiment_window.json")
    tft.check_test_overwrite(out_path, ctx.args.force)
    t0 = time.time()
    panel = load_panel(ctx)
    frozen = ctx.path("frozen_d.json")
    hp = read_json(frozen)["hparams"] if frozen.exists() else dict(tft.DEFAULT_HPARAMS)
    hp = {**tft.DEFAULT_HPARAMS, **hp}
    bounds = tft.resolve_split(panel, "sentiment_window")
    in_test = (panel["time_idx"] >= bounds["test_start_idx"]) & (panel["time_idx"] <= bounds["test_end_idx"])
    in_train = panel["time_idx"] <= bounds["train_end_idx"]
    out = common_header(ctx, "sentiment_window", "sentiment_window", None, hp)
    out["hparams_source"] = "frozen_d.json" if frozen.exists() else "defaults"
    out["split_type"] = "sentiment-window ablation (held-out Nov-Dec 2025; evaluated once)"
    out["dates"] = bounds["dates"]
    out["test_rows_with_news_pct"] = float((panel.loc[in_test, "has_news"] > 0).mean() * 100)
    out["train_rows_with_news_pct"] = float((panel.loc[in_train, "has_news"] > 0).mean() * 100)
    out["feature_sets"] = {}
    for fs in ["technical", "sentiment"]:
        frames, infos = [], []
        for s in ctx.seeds:
            r = fit_fold(ctx, panel, "sentiment_window", fs, hp, s)
            model = tft.load_model(r["info"]["ckpt_path"])
            _, _, test_ds = tft.make_datasets(panel, "sentiment_window", fs, hp["encoder_length"])
            frames.append(tft.predict_frame(model, test_ds, panel))
            infos.append(r["info"])
        pred = tft.average_pred_frames(frames)
        m = tft.directional_metrics(tft.directional_frame(pred, panel))
        out["feature_sets"][fs] = {
            "features": tft.FEATURE_SETS[fs],
            "metrics": m,
            "summary": tft.summarize_metrics(m),
            "pinball_loss_all_steps": float(pred["pinball"].mean()),
            "epochs_trained": [i["epochs_trained"] for i in infos],
            "n_train_samples": infos[0]["n_train_samples"],
        }
        print(f"{fs}: {out['feature_sets'][fs]['summary']}")
    out["wall_time_s"] = time.time() - t0
    write_json(out_path, out)


# ----------------------------------------------------------------------------
# LightGBM baseline
# ----------------------------------------------------------------------------

LGBM_PARAMS = dict(
    n_estimators=200, learning_rate=0.03, num_leaves=15, min_child_samples=100,
    subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0, verbose=-1,
)


def lgbm_columns() -> list[str]:
    cols = tft.feature_columns("sentiment") + list(KNOWN_REALS) + list(KNOWN_CATEGORICALS)
    cols = list(dict.fromkeys(cols))
    assert_no_target_leakage(cols)
    return cols


def lgbm_window(panel, split_name: str, which: str, seed: int) -> pd.DataFrame:
    """Train one classifier per horizon; predict origins matching the TFT evaluation set.

    Training rows need t + h <= train_end (their labels use prices up to t+h), so
    no label overlaps the evaluation window. Evaluation origins are
    [start-1, end-7], identical to the TFT decoder windows.
    """
    import lightgbm as lgb

    b = tft.resolve_split(panel, split_name)
    train_end = b["train_end_idx"]
    if which == "val":
        lo, hi = b["val_start_idx"] - 1, b["val_end_idx"] - tft.MAX_PREDICTION_LENGTH
    else:
        lo, hi = b["test_start_idx"] - 1, b["test_end_idx"] - tft.MAX_PREDICTION_LENGTH
    df = panel.copy()
    df["ticker_code"] = df["ticker"].astype("category").cat.codes
    feats = lgbm_columns() + ["ticker_code"]
    parts = []
    for h in HORIZONS:
        tr = df[df["time_idx"] + h <= train_end].dropna(subset=[f"fwd_dir_{h}d"])
        ev = df[(df["time_idx"] >= lo) & (df["time_idx"] <= hi)]
        clf = lgb.LGBMClassifier(random_state=seed, **LGBM_PARAMS)
        clf.fit(tr[feats], tr[f"fwd_dir_{h}d"].astype(int))
        p = clf.predict_proba(ev[feats])[:, 1]
        parts.append(
            pd.DataFrame(
                {
                    "ticker": ev["ticker"].to_numpy(),
                    "origin_idx": ev["time_idx"].to_numpy(),
                    "origin_date": ev["date"].to_numpy(),
                    "horizon": h,
                    "p_up": p,
                    "pred_dir": (p >= 0.5).astype(float),
                    "fwd_ret": ev[f"fwd_ret_{h}d"].to_numpy(),
                    "fwd_dir": ev[f"fwd_dir_{h}d"].to_numpy(),
                }
            )
        )
    return pd.concat(parts, ignore_index=True)


def lgbm_ensemble(ctx, panel, split_name, which) -> pd.DataFrame:
    frames = [lgbm_window(panel, split_name, which, s) for s in ctx.seeds]
    base = frames[0].copy()
    base["p_up"] = np.mean([f["p_up"].to_numpy() for f in frames], axis=0)
    base["pred_dir"] = (base["p_up"] >= 0.5).astype(float)
    return base


def phase_cv_lgbm(ctx: Ctx) -> None:
    t0 = time.time()
    panel = load_panel(ctx)
    out = common_header(ctx, "lgbm", "cv", "sentiment", None)
    out["features"]["observed_reals"] = lgbm_columns()
    out["lgbm_params"] = LGBM_PARAMS
    out["split_type"] = "walk-forward expanding CV, validation folds"
    out["folds"] = {}
    for f in CV_FOLDS:
        m = tft.directional_metrics(lgbm_ensemble(ctx, panel, f, "val"))
        out["folds"][f] = {
            "split": f,
            "dates": tft.resolve_split(panel, f)["dates"],
            "metrics": m,
            "summary": tft.summarize_metrics(m),
        }
    out["complete"] = True
    per_h = {}
    for h in map(str, HORIZONS):
        a = [out["folds"][f]["metrics"][h]["accuracy"] for f in CV_FOLDS]
        b = [out["folds"][f]["metrics"][h]["majority_baseline"] for f in CV_FOLDS]
        per_h[h] = {"mean_accuracy": float(np.mean(a)), "mean_majority_baseline": float(np.mean(b))}
    out["cv_mean"] = {
        "mean_accuracy": float(np.mean([v["mean_accuracy"] for v in per_h.values()])),
        "mean_majority_baseline": float(np.mean([v["mean_majority_baseline"] for v in per_h.values()])),
        "per_horizon": per_h,
        "folds_included": CV_FOLDS,
    }
    out["validation_F3"] = out["folds"]["F3"]
    out["wall_time_s"] = time.time() - t0
    write_json(ctx.path("cv_lgbm.json"), out)
    print_summary(out)


def phase_test_lgbm(ctx: Ctx, panel, out_path: Path, cv: dict) -> None:
    t0 = time.time()
    m = tft.directional_metrics(lgbm_ensemble(ctx, panel, "test", "test"))
    out = common_header(ctx, "lgbm", "test", "sentiment", None)
    out["features"]["observed_reals"] = lgbm_columns()
    out["lgbm_params"] = LGBM_PARAMS
    out["split_type"] = "held-out test (trained <=2022-12-31 with t+h<=train_end)"
    out["dates"] = tft.resolve_split(panel, "test")["dates"]
    out["metrics"] = m
    out["summary"] = tft.summarize_metrics(m)
    out["validation_reference_F3"] = cv.get("validation_F3")
    out["cv_mean_reference"] = cv.get("cv_mean")
    out["wall_time_s"] = time.time() - t0
    write_json(out_path, out)
    print_summary(out)


# ----------------------------------------------------------------------------

def print_summary(out: dict) -> None:
    print(f"== stage {out['stage']} phase {out['phase']} ({out.get('split_type')}) ==")
    if out["phase"] == "cv" and "folds" in out:
        for f, rec in out["folds"].items():
            s = rec["summary"]
            print(f"  {f}: mean acc {s['mean_accuracy']:.4f} vs majority {s['mean_majority_baseline']:.4f}")
        cm = out["cv_mean"]
        print(f"  CV mean: {cm['mean_accuracy']:.4f} vs majority {cm['mean_majority_baseline']:.4f}")
    elif "summary" in out:
        s = out["summary"]
        print(f"  mean acc {s['mean_accuracy']:.4f} vs majority {s['mean_majority_baseline']:.4f}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--stage", choices=["a", "b", "c", "d", "e", "lgbm"])
    p.add_argument("--phase", required=True, choices=["cv", "test", "sentiment_window"])
    p.add_argument("--tickers", default="default", help="comma list, 'all', or 'default' (documented subset)")
    p.add_argument("--max-epochs", type=int, default=15)
    p.add_argument("--limit-train-batches", type=int, default=None)
    p.add_argument("--seeds", default="0", help="comma list, e.g. 0,1,2 (predictions are averaged)")
    p.add_argument("--folds", default="F1,F2,F3", help="cv folds to (re)compute")
    p.add_argument("--smoke", action="store_true", help="3 tickers, 1 epoch, few batches; writes to reports/phase7/smoke/")
    p.add_argument("--force", action="store_true", help="overwrite an existing held-out test result (discouraged)")
    p.add_argument("--refit", action="store_true", help="ignore cached models/predictions")
    p.add_argument("--retune", action="store_true", help="stage d: redo the hyperparameter search")
    return p


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    if args.phase != "sentiment_window" and not args.stage:
        raise SystemExit("--stage is required for cv/test phases")
    ctx = Ctx(args)
    print(f"tickers ({len(ctx.tickers)}): {ctx.tickers}; seeds {ctx.seeds}; epochs<={ctx.max_epochs}; limit {ctx.limit}")
    {"cv": phase_cv, "test": phase_test, "sentiment_window": phase_sentiment_window}[args.phase](ctx)


if __name__ == "__main__":
    main()

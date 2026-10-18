"""Block-bootstrap CIs for Phase 7 held-out test accuracy.

Test predictions are not stored by the runner, so this script regenerates them by
INFERENCE ONLY from the already-trained, cached F3 checkpoints (TFT) or by refitting
the deterministic seeded LightGBM, then verifies that the regenerated accuracies
match reports/phase7/test_<stage>.json to 1e-9 before reporting any CI. No model
is selected or tuned here; nothing is written back to the test result files.

Resampling unit: blocks of BLOCK consecutive trading-day origins (all tickers in a
block move together), which respects overlapping horizons and cross-ticker
correlation. Output: reports/phase7/bootstrap_ci.json

    python scripts/phase7_bootstrap_ci.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import run_phase7_tft as run  # noqa: E402
from src.data.tft_dataset import HORIZONS  # noqa: E402
from src.models import tft  # noqa: E402

BLOCK = 20
N_BOOT = 2000
SEED = 12345
STAGES = ["a", "b", "c", "d", "e", "lgbm"]


def get_dir_frame(ctx, panel, stage: str) -> pd.DataFrame:
    if stage == "lgbm":
        return run.lgbm_ensemble(ctx, panel, "test", "test")
    feature_set, hp = run.stage_config(ctx, stage)
    return run.test_predictions(ctx, panel, feature_set, hp)["dir"]


def _stats(s: np.ndarray) -> tuple[float, float, float]:
    """s columns: n, n_correct, n_up -> (accuracy, majority baseline, difference)."""
    n = s[:, 0].sum()
    if n == 0:
        return np.nan, np.nan, np.nan
    acc = s[:, 1].sum() / n
    up = s[:, 2].sum() / n
    base = max(up, 1 - up)
    return acc, base, acc - base


def _block_table(g: pd.DataFrame, mask: np.ndarray, origin_min: int, nb: int) -> np.ndarray:
    b = ((g["origin_idx"].to_numpy() - origin_min) // BLOCK).astype(int)[mask]
    y = g["fwd_dir"].to_numpy()[mask]
    c = (g["pred_dir"].to_numpy()[mask] == y).astype(float)
    s = np.zeros((nb, 3))
    np.add.at(s[:, 0], b, 1)
    np.add.at(s[:, 1], b, c)
    np.add.at(s[:, 2], b, y)
    return s


def _summ(point, draws, extra=None) -> dict:
    lo, hi = np.nanpercentile(draws, 2.5, axis=0), np.nanpercentile(draws, 97.5, axis=0)
    d = {
        "accuracy": float(point[0]), "accuracy_ci95": [float(lo[0]), float(hi[0])],
        "majority_baseline": float(point[1]),
        "acc_minus_majority": float(point[2]),
        "acc_minus_majority_ci95": [float(lo[2]), float(hi[2])],
        "share_of_bootstrap_draws_acc_minus_majority_le_0": float(np.nanmean(draws[:, 2] <= 0)),
    }
    d.update(extra or {})
    return d


def main() -> None:
    args = run.build_parser().parse_args(["--phase", "test", "--stage", "d"])
    ctx = run.Ctx(args)
    panel = run.load_panel(ctx)
    out = {"block_trading_days": BLOCK, "n_boot": N_BOOT, "seed": SEED, "stages": {}}
    for stage in STAGES:
        tp = ctx.path(f"test_{stage}.json")
        if not tp.exists():
            continue
        ref = run.read_json(tp)
        df = get_dir_frame(ctx, panel, stage)
        m = tft.directional_metrics(df)
        for h in map(str, HORIZONS):
            if abs(m[h]["accuracy"] - ref["metrics"][h]["accuracy"]) > 1e-9:
                raise SystemExit(
                    f"stage {stage} h={h}: regenerated acc {m[h]['accuracy']} != json {ref['metrics'][h]['accuracy']}"
                )
        thr = run.read_json(ctx.path("frozen_e.json"))["thresholds"] if stage == "e" else {}
        rng = np.random.default_rng(SEED)
        gs = {h: df[df["horizon"] == h].dropna(subset=["fwd_dir"]) for h in HORIZONS}
        omin = int(min(g["origin_idx"].min() for g in gs.values()))
        nb = int(max((g["origin_idx"].max() - omin) // BLOCK for g in gs.values())) + 1
        rec = {"verified_matches_test_json": True, "n_blocks": nb, "horizons": {}}
        tables = {}
        for h, g in gs.items():
            full = _block_table(g, np.ones(len(g), dtype=bool), omin, nb)
            tables[h] = full
            draws = np.array([_stats(full[rng.integers(0, nb, nb)]) for _ in range(N_BOOT)])
            hr = {"all": _summ(_stats(full), draws, {
                "n": int(full[:, 0].sum()),
                "naive_binomial_se": float(np.sqrt(_stats(full)[0] * (1 - _stats(full)[0]) / full[:, 0].sum())),
            })}
            if stage == "e":
                p = g["p_up"].to_numpy()
                t = thr[str(h)]
                conf = (p >= t) | (p <= 1 - t)
                ct = _block_table(g, conf, omin, nb)
                cd = np.array([_stats(ct[rng.integers(0, nb, nb)]) for _ in range(N_BOOT)])
                hr["confident"] = _summ(_stats(ct), cd, {"threshold": t, "n": int(ct[:, 0].sum()),
                                                           "coverage": float(ct[:, 0].sum() / full[:, 0].sum())})
            rec["horizons"][str(h)] = hr

        def mh(pick):
            st = [_stats(s[pick]) for s in tables.values()]
            a = np.mean([x[0] for x in st])
            bs = np.mean([x[1] for x in st])
            return a, bs, a - bs

        draws = np.array([mh(rng.integers(0, nb, nb)) for _ in range(N_BOOT)])
        rec["mean_over_horizons"] = _summ(mh(np.arange(nb)), draws)
        out["stages"][stage] = rec
        mo = rec["mean_over_horizons"]
        print(f"{stage}: mean acc {mo['accuracy']:.4f} CI {mo['accuracy_ci95']} diff {mo['acc_minus_majority']:.4f} CI {mo['acc_minus_majority_ci95']}")
    run.write_json(ctx.path("bootstrap_ci.json"), out)


if __name__ == "__main__":
    main()

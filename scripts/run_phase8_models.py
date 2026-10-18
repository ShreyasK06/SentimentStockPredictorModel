"""Phase 8 WP3/WP4: news-sentiment next-day direction models with selective prediction.

    python scripts/run_phase8_models.py cv      # walk-forward CV (dev years only) + diagnostics
    python scripts/run_phase8_models.py final   # ONE-TIME held-out test (2022+) and Finnhub recent window

Protocol (frozen in docs/superpowers/2026-10-17-phase8-news-classifier.md):
  CV fold Y in 2017..2021: train = dev_start..Y-2, validation = Y-1 (calibration + threshold), eval = Y.
  Final: train = dev_start..2020, validation = 2021, TEST = 2022..FNSPID end (evaluated once, lock file).
  Hyperparameters: per family, lowest mean CV validation log-loss (raw probabilities).
  Primary model: the family with the highest POOLED CV-eval confident-call accuracy among families
  whose pooled CV coverage >= 5% (decided in `cv`, written to reports/phase8/cv_selection.json,
  read by `final`).
  Threshold: validation only, max accuracy s.t. coverage >= 5% and n >= 300.
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import log_loss  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from src.models.news_direction import EmbModel, TabModel, cols  # noqa: E402
from src.models.selective import (  # noqa: E402
    TestOnceLock, accuracy_coverage_curve, choose_threshold, fit_calibrator, selective_report,
)

warnings.filterwarnings("ignore")
P8 = ROOT / "data" / "processed" / "phase8"
REP = ROOT / "reports" / "phase8"
MODELS = ROOT / "models" / "phase8"
CFG = json.loads((REP / "wp1_dataset_config.json").read_text())
DEV_START = int(CFG["dev_start"][:4])
FOLD_YEARS = [2017, 2018, 2019, 2020, 2021]
MIN_COV, MIN_N = 0.05, 300
CURVE_COVS = (1.0, 0.5, 0.2, 0.1, 0.05, 0.02, 0.01)
SEED = 0


# ----------------------------------------------------------------------- data
def load(source: str, suffix: str = ""):
    df = pd.read_parquet(P8 / f"daily_features_{source}{suffix}.parquet")
    emb = np.load(P8 / f"daily_emb_{source}{suffix}.npy")
    assert len(df) == len(emb)
    return df, emb


def families(df):
    both, ctl = cols(df, "both"), cols(df, "ctl")
    news = cols(df, "news")
    return {
        "a_logit": [(f"C={C}", lambda C=C: TabModel("logit", both, C=C)) for C in (0.01, 0.1, 1.0)],
        "a_lgbm": [(f"leaves={nl},trees={nt}", lambda nl=nl, nt=nt: TabModel("lgbm", both, num_leaves=nl,
                                                                              n_estimators=nt))
                   for nl in (7, 15) for nt in (150, 400)],
        "b_emb": [(f"pca={k},C={C}", lambda k=k, C=C: EmbModel(ctl, k, C)) for k in (16, 32) for C in (0.01, 0.1)],
        # ablations (not candidates for primary)
        "abl_news_only_logit": [(f"C={C}", lambda C=C: TabModel("logit", news, C=C)) for C in (0.01, 0.1, 1.0)],
        "abl_ctl_only_logit": [(f"C={C}", lambda C=C: TabModel("logit", ctl, C=C)) for C in (0.01, 0.1, 1.0)],
    }


PRIMARY_CANDIDATES = ["a_logit", "a_lgbm", "b_emb", "c_ensemble"]


def split_masks(df, train_end_year, val_year, eval_years):
    y = df["year"]
    return (y >= DEV_START) & (y <= train_end_year), y == val_year, y.isin(eval_years)


def fit_eval(make, df, emb, tr, va, ev):
    m = make().fit(df[tr], emb[tr.to_numpy()], df.loc[tr, "y"].astype(int).to_numpy())
    pv = m.predict(df[va], emb[va.to_numpy()])
    pe = m.predict(df[ev], emb[ev.to_numpy()]) if ev.any() else np.array([])
    return m, pv, pe


def selective_block(p_val_raw, y_val, p_eval_raw, y_eval, dates_eval, train_major):
    cal = fit_calibrator(p_val_raw, y_val)
    pv, pe = cal.transform(p_val_raw), cal.transform(p_eval_raw)
    thr = choose_threshold(pv, y_val, MIN_COV, MIN_N)
    rep = selective_report(pe, y_eval, thr["threshold"], dates_eval, train_major, n_boot=2000, block=20, seed=SEED)
    return cal, pv, pe, thr, rep


# ------------------------------------------------------------------------- CV
def run_cv(source="fnspid", suffix="", fams_filter=None, tag="cv"):
    df, emb = load(source, suffix)
    fams = families(df)
    if fams_filter:
        fams = {k: v for k, v in fams.items() if k in fams_filter}
    # 1) raw predictions for every config and fold
    raw = {}  # (fam, cfg, fold) -> (pv, pe)
    for Y in FOLD_YEARS:
        tr, va, ev = split_masks(df, Y - 2, Y - 1, [Y])
        print(f"[{tag}] fold {Y}: train={tr.sum()} val={va.sum()} eval={ev.sum()}", flush=True)
        for fam, cfgs in fams.items():
            for name, make in cfgs:
                _, pv, pe = fit_eval(make, df, emb, tr, va, ev)
                raw[(fam, name, Y)] = (pv, pe)
    # 2) hyperparameters by mean validation log-loss
    chosen = {}
    for fam, cfgs in fams.items():
        scores = {}
        for name, _ in cfgs:
            ll = [log_loss(df.loc[df.year == Y - 1, "y"].astype(int), np.clip(raw[(fam, name, Y)][0], 1e-6, 1 - 1e-6))
                  for Y in FOLD_YEARS]
            scores[name] = float(np.mean(ll))
        chosen[fam] = {"config": min(scores, key=scores.get), "mean_val_logloss": scores}
    # 3) per-fold calibration + validation threshold; ensemble = mean of calibrated a/a/b, recalibrated on val
    out_rows, pooled = [], {}
    fam_list = list(fams) + (["c_ensemble"] if {"a_logit", "a_lgbm", "b_emb"} <= set(fams) else [])
    for Y in FOLD_YEARS:
        tr, va, ev = split_masks(df, Y - 2, Y - 1, [Y])
        yv, ye = df.loc[va, "y"].astype(int).to_numpy(), df.loc[ev, "y"].astype(int).to_numpy()
        train_major = int(df.loc[tr, "y"].mean() >= 0.5)
        cal_cache = {}
        for fam in fam_list:
            if fam == "c_ensemble":
                pv_raw = np.mean([cal_cache[f][0] for f in ("a_logit", "a_lgbm", "b_emb")], axis=0)
                pe_raw = np.mean([cal_cache[f][1] for f in ("a_logit", "a_lgbm", "b_emb")], axis=0)
            else:
                pv_raw, pe_raw = raw[(fam, chosen[fam]["config"], Y)]
            cal, pv, pe, thr, rep = selective_block(pv_raw, yv, pe_raw, ye, df.loc[ev, "date"].to_numpy(), train_major)
            cal_cache[fam] = (pv, pe)
            curve = accuracy_coverage_curve(pe, ye, CURVE_COVS)
            out_rows.append({"family": fam, "fold": Y, **rep,
                             "val_threshold": thr["threshold"], "val_accuracy": thr["val_accuracy"],
                             "val_coverage": thr["val_coverage"], "val_n": thr["val_n"],
                             "feasible": thr["feasible"], "curve": curve.to_dict(orient="records")})
            pooled.setdefault(fam, []).append(pd.DataFrame({
                "date": df.loc[ev, "date"].to_numpy(), "ticker": df.loc[ev, "ticker"].to_numpy(),
                "y": ye, "p": pe, "thr": thr["threshold"], "train_major": train_major}))
    # 4) pooled CV summary per family (each fold uses its own validation threshold)
    summary = {}
    for fam, parts in pooled.items():
        P = pd.concat(parts, ignore_index=True)
        called = (np.maximum(P.p, 1 - P.p) >= P.thr).to_numpy()
        tm = int(round(P.train_major.mean()))
        rep_all = selective_report(P.p, P.y, 0.0, P.date, tm)
        rep_c = selective_report(P.p[called], P.y[called], 0.0, P.date[called], tm) if called.any() else None
        nan = float("nan")
        summary[fam] = {
            "n_rows": rep_all["n_rows"], "unrestricted_accuracy": rep_all["unrestricted_accuracy"],
            "unrestricted_accuracy_ci": rep_all["unrestricted_accuracy_ci"],
            "local_majority_rows": rep_all["local_majority_rows"],
            "train_majority_applied_rows": rep_all["train_majority_applied_rows"],
            "n_calls": int(called.sum()), "coverage": float(called.mean()),
            "confident_accuracy": rep_c["unrestricted_accuracy"] if rep_c else nan,
            "confident_accuracy_ci": rep_c["unrestricted_accuracy_ci"] if rep_c else (nan, nan),
            "local_majority_calls": rep_c["local_majority_rows"] if rep_c else nan,
            "train_majority_applied_calls": rep_c["train_majority_applied_rows"] if rep_c else nan,
            "curve": accuracy_coverage_curve(P.p, P.y, CURVE_COVS).to_dict(orient="records"),
        }
        P.to_parquet(REP / f"{tag}_pooled_{fam}.parquet", index=False)
    return df, emb, chosen, out_rows, summary


def permutation_check(df, emb, chosen):
    """Train labels shuffled (fold-2021 setup, a_logit chosen config): accuracy must be ~50%."""
    tr, va, ev = split_masks(df, 2019, 2020, [2021])
    make = dict(families(df)["a_logit"])[chosen["a_logit"]["config"]]
    rng = np.random.default_rng(SEED)
    d_tr = df[tr].copy()
    d_tr["y"] = rng.permutation(d_tr["y"].to_numpy())
    m = make().fit(d_tr, emb[tr.to_numpy()], d_tr["y"].astype(int).to_numpy())
    p = m.predict(df[ev], emb[ev.to_numpy()])
    ye = df.loc[ev, "y"].astype(int).to_numpy()
    conf = np.maximum(p, 1 - p)
    top = conf >= np.quantile(conf, 0.95)
    return {"eval_year": 2021, "unrestricted_accuracy": float(((p >= .5) == ye).mean()),
            "top5pct_accuracy": float(((p[top] >= .5) == ye[top]).mean()), "n_top": int(top.sum()),
            "local_majority_rows": float(max(ye.mean(), 1 - ye.mean()))}


def jsonable(o):
    if isinstance(o, dict):
        return {str(k): jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jsonable(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, (np.integer, np.bool_)):
        return o.item()
    if isinstance(o, pd.Timestamp):
        return str(o.date())
    return o


def print_summary(title, summary):
    print(f"\n== {title}")
    for f, s in summary.items():
        ci = tuple(round(x, 3) for x in s["confident_accuracy_ci"])
        print(f"{f:22s} rows={s['n_rows']:6d} acc_all={s['unrestricted_accuracy']:.3f} "
              f"maj_rows={s['local_majority_rows']:.3f} | calls={s['n_calls']:5d} cov={s['coverage']:.3f} "
              f"acc_conf={s['confident_accuracy']:.3f} CI={ci} maj_calls={s['local_majority_calls']:.3f}")


def cmd_cv():
    REP.mkdir(parents=True, exist_ok=True)
    df, emb, chosen, rows, summary = run_cv()
    perm = permutation_check(df, emb, chosen)
    naive = None
    if (P8 / "daily_features_fnspid_naive.parquet").exists():
        _, _, _, _, naive = run_cv("fnspid", "_naive", fams_filter=["a_logit", "a_lgbm"], tag="cv_naive")
    eligible = {f: s for f, s in summary.items() if f in PRIMARY_CANDIDATES and s["coverage"] >= MIN_COV
                and s["confident_accuracy"] == s["confident_accuracy"]}
    primary = max(eligible, key=lambda f: eligible[f]["confident_accuracy"]) if eligible else "a_logit"
    alarms = [f"{r['family']} fold {r['fold']} confident acc {r['confident_accuracy']:.3f} n={r['n_calls']}"
              for r in rows if r["confident_accuracy"] == r["confident_accuracy"]
              and r["confident_accuracy"] > 0.70 and r["n_calls"] >= 50]
    res = {"dev_start": DEV_START, "fold_years": FOLD_YEARS, "min_coverage": MIN_COV, "min_n": MIN_N,
           "chosen_hparams": chosen, "per_fold": rows, "pooled": summary, "permutation_check": perm,
           "naive_alignment_diagnostic": naive, "primary_family": primary, "leakage_alarms_gt70": alarms}
    (REP / "cv_results.json").write_text(json.dumps(jsonable(res), indent=1))
    (REP / "cv_selection.json").write_text(json.dumps(jsonable(
        {"primary_family": primary, "chosen_hparams": {k: v["config"] for k, v in chosen.items()}}), indent=1))
    print_summary("CV pooled (eval years 2017-2021)", summary)
    for r in rows:
        print(f"  fold {r['fold']} {r['family']:22s} val_thr={r['val_threshold']:.3f} val_acc={r['val_accuracy']:.3f} "
              f"val_cov={r['val_coverage']:.3f} | eval acc_all={r['unrestricted_accuracy']:.3f} "
              f"calls={r['n_calls']} cov={r['coverage']:.3f} acc_conf={r['confident_accuracy']:.3f}")
    if naive:
        print_summary("CV pooled, NAIVE clock alignment (diagnostic only)", naive)
    print("permutation:", perm)
    print("primary:", primary, "alarms:", alarms)


# ---------------------------------------------------------------------- final
def cmd_final():
    sel = json.loads((REP / "cv_selection.json").read_text())
    lock = TestOnceLock(REP / "TEST_LOCK.json")
    lock.acquire_or_raise()
    (REP / "TEST_STARTED.json").write_text(json.dumps({"started": pd.Timestamp.utcnow().isoformat()}))
    df, emb = load("fnspid")
    have_fin = (P8 / "daily_features_finnhub.parquet").exists()
    fin_df, fin_emb = load("finnhub") if have_fin else (None, None)
    fams = families(df)
    tr, va, te = split_masks(df, 2020, 2021, list(range(2022, 2030)))
    yv = df.loc[va, "y"].astype(int).to_numpy()
    yt = df.loc[te, "y"].astype(int).to_numpy()
    train_major = int(df.loc[tr, "y"].mean() >= 0.5)
    MODELS.mkdir(parents=True, exist_ok=True)
    results, cal_probs, artifacts = {}, {}, {}
    order = ["a_logit", "a_lgbm", "b_emb", "abl_news_only_logit", "abl_ctl_only_logit", "c_ensemble"]
    for fam in order:
        if fam == "c_ensemble":
            parts = ("a_logit", "a_lgbm", "b_emb")
            pv_raw = np.mean([cal_probs[f]["val"] for f in parts], axis=0)
            pt_raw = np.mean([cal_probs[f]["test"] for f in parts], axis=0)
            pf_raw = np.mean([cal_probs[f]["fin"] for f in parts], axis=0) if have_fin else None
        else:
            make = dict(fams[fam])[sel["chosen_hparams"][fam]]
            m, pv_raw, pt_raw = fit_eval(make, df, emb, tr, va, te)
            pf_raw = m.predict(fin_df, fin_emb) if have_fin else None
            artifacts[fam] = m
        cal, pv, pt, thr, rep = selective_block(pv_raw, yv, pt_raw, yt, df.loc[te, "date"].to_numpy(), train_major)
        pf = cal.transform(pf_raw) if pf_raw is not None else None
        cal_probs[fam] = {"val": pv, "test": pt, "fin": pf}
        artifacts[fam + "__cal"] = cal
        artifacts[fam + "__thr"] = thr
        entry = {"threshold_from_2021_val": thr, "test": rep,
                 "test_curve": accuracy_coverage_curve(pt, yt, CURVE_COVS).to_dict(orient="records")}
        if pf is not None:
            yf = fin_df["y"].astype(int).to_numpy()
            fr = selective_report(pf, yf, thr["threshold"], fin_df["date"].to_numpy(), train_major)
            fr["period"] = [str(fin_df.date.min().date()), str(fin_df.date.max().date())]
            entry["finnhub_recent"] = fr
            entry["finnhub_curve"] = accuracy_coverage_curve(pf, yf, CURVE_COVS).to_dict(orient="records")
        results[fam] = entry
        pd.DataFrame({"date": df.loc[te, "date"].to_numpy(), "ticker": df.loc[te, "ticker"].to_numpy(),
                      "y": yt, "p": pt}).to_parquet(REP / f"test_preds_{fam}.parquet", index=False)
    import joblib
    joblib.dump({"artifacts": artifacts, "selection": sel,
                 "feature_cols": {k: cols(df, k) for k in ("news", "ctl", "both")},
                 "train_years": [DEV_START, 2020], "val_year": 2021}, MODELS / "final_models.joblib")
    out = {"primary_family": sel["primary_family"], "train": f"{DEV_START}-2020", "validation": "2021",
           "test": f"2022-01-01..{CFG['fnspid_end']}", "n_train": int(tr.sum()), "n_val": int(va.sum()),
           "n_test": int(te.sum()), "train_majority_class": train_major, "results": results}
    (REP / "final_test_results.json").write_text(json.dumps(jsonable(out), indent=1))
    lock.commit({"primary": sel["primary_family"]})
    for fam, e in results.items():
        r = e["test"]
        print(f"TEST {fam:22s} acc_all={r['unrestricted_accuracy']:.3f} maj={r['local_majority_rows']:.3f} "
              f"calls={r['n_calls']} cov={r['coverage']:.3f} acc_conf={r['confident_accuracy']:.3f} "
              f"CI={r['confident_accuracy_ci']} maj_calls={r['local_majority_calls']}")
        if "finnhub_recent" in e:
            f = e["finnhub_recent"]
            print(f"  FINNHUB acc_all={f['unrestricted_accuracy']:.3f} calls={f['n_calls']} cov={f['coverage']:.3f} "
                  f"acc_conf={f['confident_accuracy']} maj_calls={f['local_majority_calls']}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["cv", "final"])
    a = ap.parse_args()
    cmd_cv() if a.cmd == "cv" else cmd_final()

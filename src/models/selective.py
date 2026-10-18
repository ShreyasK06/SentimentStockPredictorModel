"""Selective prediction utilities (Phase 8 WP4).

Pure functions for calibration, confidence-threshold selection on VALIDATION
data only, selective (abstaining) reporting with moving-block bootstrap CIs,
accuracy-vs-coverage curves, and a run-once lock for the held-out test.
Only ``TestOnceLock`` touches the filesystem.
"""

from __future__ import annotations

import json
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

_EPS = 1e-6
MIN_CALLS_FLAG = 300


def _clip(p) -> np.ndarray:
    return np.clip(np.asarray(p, dtype=float), _EPS, 1 - _EPS)


def _logit(p) -> np.ndarray:
    p = _clip(p)
    return np.log(p / (1 - p))


# ---------------------------------------------------------------- calibration
class Calibrator:
    """Wraps a fitted calibration map. ``degenerate`` True => identity fallback."""

    def __init__(self, method: str, model=None, degenerate: bool = False):
        self.method = method
        self.model = model
        self.degenerate = degenerate

    def transform(self, p) -> np.ndarray:
        p = np.asarray(p, dtype=float)
        if self.method == "identity":
            return p.copy()
        if self.method == "isotonic":
            return np.asarray(self.model.predict(p), dtype=float)
        z = _logit(p).reshape(-1, 1)
        return self.model.predict_proba(z)[:, 1]


def fit_calibrator(p_val, y_val, method: str = "auto") -> Calibrator:
    """Fit on validation only. auto = isotonic if n >= 1000 else Platt."""
    p = np.asarray(p_val, dtype=float)
    y = np.asarray(y_val).astype(int)
    if method not in ("auto", "isotonic", "platt", "identity"):
        raise ValueError(f"unknown method {method!r}")
    if method == "identity":
        return Calibrator("identity")
    if len(p) == 0 or len(np.unique(y)) < 2:
        warnings.warn("calibration data empty or single-class; using identity", stacklevel=2)
        return Calibrator("identity", degenerate=True)
    if method == "auto":
        method = "isotonic" if len(p) >= 1000 else "platt"
    if method == "isotonic":
        iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        iso.fit(p, y)
        return Calibrator("isotonic", iso)
    lr = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000)
    lr.fit(_logit(p).reshape(-1, 1), y)
    return Calibrator("platt", lr)


# ----------------------------------------------------------------- confidence
def confidence(p) -> np.ndarray:
    p = np.asarray(p, dtype=float)
    return np.maximum(p, 1.0 - p)


def predicted_class(p) -> np.ndarray:
    return (np.asarray(p, dtype=float) >= 0.5).astype(int)


def _correct(p, y) -> np.ndarray:
    return (predicted_class(p) == np.asarray(y).astype(int)).astype(float)


# ------------------------------------------------------------------ threshold
def choose_threshold(p_val, y_val, min_coverage: float = 0.05, min_n: int = 300,
                     grid=None) -> dict:
    """Pick the confidence threshold on validation data only.

    Maximizes accuracy of called rows (confidence >= t) subject to
    coverage >= min_coverage and n_calls >= min_n; ties -> larger coverage.
    Infeasible -> feasible=False, threshold=0.5 (predict everything).
    """
    p = np.asarray(p_val, dtype=float)
    n = len(p)
    out = {"threshold": 0.5, "val_accuracy": float("nan"), "val_coverage": float("nan"),
           "val_n": 0, "feasible": False}
    if n == 0:
        return out
    conf = confidence(p)
    corr = _correct(p, y_val)
    cands = np.unique(conf) if grid is None else np.unique(np.asarray(grid, dtype=float))
    order = np.argsort(conf)
    conf_s, corr_s = conf[order], corr[order]
    # suffix sums: rows with conf >= t
    suffix_corr = np.concatenate([np.cumsum(corr_s[::-1])[::-1], [0.0]])
    idx = np.searchsorted(conf_s, cands, side="left")
    n_calls = n - idx
    cov = n_calls / n
    ok = (n_calls >= min_n) & (cov >= min_coverage) & (n_calls > 0)
    if not ok.any():
        out.update(val_accuracy=float(corr.mean()), val_coverage=1.0, val_n=n)
        return out
    acc = np.where(n_calls > 0, suffix_corr[idx] / np.maximum(n_calls, 1), -1.0)
    acc_ok = np.where(ok, acc, -1.0)
    best = acc_ok.max()
    tied = np.flatnonzero(ok & np.isclose(acc_ok, best, rtol=0, atol=1e-12))
    j = tied[np.argmax(n_calls[tied])]
    return {"threshold": float(cands[j]), "val_accuracy": float(acc[j]),
            "val_coverage": float(cov[j]), "val_n": int(n_calls[j]), "feasible": True}


# ------------------------------------------------------------------ bootstrap
def moving_block_bootstrap_ci(correct, dates, block: int = 20, n_boot: int = 2000,
                              seed: int = 0, alpha: float = 0.05):
    """Circular moving-block bootstrap over UNIQUE sorted dates.

    All rows of a date move together. Returns (lo, hi) for mean(correct).
    """
    c = np.asarray(correct, dtype=float)
    if c.size == 0:
        return (float("nan"), float("nan"))
    d = pd.to_datetime(pd.Series(np.asarray(dates)).reset_index(drop=True))
    codes, uniq = pd.factorize(d, sort=True)
    nd = len(uniq)
    s = np.bincount(codes, weights=c, minlength=nd)
    cnt = np.bincount(codes, minlength=nd).astype(float)
    b = max(1, min(int(block), nd))
    n_blocks = int(np.ceil(nd / b))
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, nd, size=(n_boot, n_blocks))
    idx = ((starts[:, :, None] + np.arange(b)[None, None, :]) % nd).reshape(n_boot, -1)[:, :nd]
    accs = s[idx].sum(axis=1) / cnt[idx].sum(axis=1)
    lo, hi = np.quantile(accs, [alpha / 2, 1 - alpha / 2])
    return (float(lo), float(hi))


# --------------------------------------------------------------------- report
def selective_report(p, y, threshold: float, dates, train_majority_class: int,
                     n_boot: int = 2000, block: int = 20, seed: int = 0) -> dict:
    p = np.asarray(p, dtype=float)
    y = np.asarray(y).astype(int)
    dates = np.asarray(dates)
    n = len(p)
    corr = _correct(p, y)
    called = confidence(p) >= threshold
    nc = int(called.sum())
    nan = float("nan")
    rep = {
        "n_rows": n,
        "threshold": float(threshold),
        "unrestricted_accuracy": float(corr.mean()) if n else nan,
        "n_calls": nc,
        "coverage": nc / n if n else nan,
    }
    rep["unrestricted_accuracy_ci"] = moving_block_bootstrap_ci(corr, dates, block, n_boot, seed)
    rep["confident_accuracy"] = float(corr[called].mean()) if nc else nan
    rep["confident_accuracy_ci"] = (moving_block_bootstrap_ci(corr[called], dates[called], block, n_boot, seed)
                                    if nc else (nan, nan))
    up_all = float(y.mean()) if n else nan
    up_calls = float(y[called].mean()) if nc else nan
    rep["local_majority_rows"] = max(up_all, 1 - up_all) if n else nan
    rep["local_majority_calls"] = max(up_calls, 1 - up_calls) if nc else nan
    tm = int(train_majority_class)
    rep["train_majority_applied_rows"] = float((y == tm).mean()) if n else nan
    rep["train_majority_applied_calls"] = float((y[called] == tm).mean()) if nc else nan
    rep["up_rate_calls"] = up_calls
    rep["pred_up_share_calls"] = float(predicted_class(p)[called].mean()) if nc else nan
    flags = []
    if nc < MIN_CALLS_FLAG:
        flags.append(f"n_calls<{MIN_CALLS_FLAG}")
    rep["flags"] = flags
    return rep


# ---------------------------------------------------------------------- curve
def accuracy_coverage_points(p, y) -> pd.DataFrame:
    """Full curve: one point per k = 1..n most-confident rows (stable order)."""
    p = np.asarray(p, dtype=float)
    y = np.asarray(y).astype(int)
    n = len(p)
    if n == 0:
        return pd.DataFrame(columns=["n_calls", "coverage", "threshold", "accuracy", "local_majority"])
    order = np.argsort(-confidence(p), kind="stable")
    k = np.arange(1, n + 1)
    corr = np.cumsum(_correct(p, y)[order])
    ups = np.cumsum(y[order])
    um = ups / k
    return pd.DataFrame({"n_calls": k, "coverage": k / n,
                         "threshold": confidence(p)[order],
                         "accuracy": corr / k, "local_majority": np.maximum(um, 1 - um)})


def accuracy_coverage_curve(p, y, coverages=(1.0, 0.5, 0.2, 0.1, 0.05, 0.02, 0.01)) -> pd.DataFrame:
    pts = accuracy_coverage_points(p, y)
    n = len(pts)
    rows = []
    for c in coverages:
        k = max(1, int(round(c * n))) if n else 0
        if k == 0:
            rows.append((c, np.nan, 0, 0.0, np.nan, np.nan))
            continue
        r = pts.iloc[k - 1]
        rows.append((c, r["threshold"], k, k / n, r["accuracy"], r["local_majority"]))
    return pd.DataFrame(rows, columns=["target_coverage", "threshold", "n_calls",
                                       "actual_coverage", "accuracy", "local_majority_calls"])


# ----------------------------------------------------------------------- lock
class TestOnceLock:
    """Refuses to let the held-out test be evaluated twice."""

    __test__ = False  # not a pytest class

    def __init__(self, path):
        self.path = Path(path)

    def acquire_or_raise(self) -> None:
        if self.path.exists():
            try:
                ts = json.loads(self.path.read_text()).get("timestamp", "unknown time")
            except Exception:
                ts = "unknown time"
            raise RuntimeError(
                f"held-out test already evaluated at {ts} ({self.path}); delete lock "
                "deliberately and report it to re-run")

    def commit(self, payload=None) -> None:
        """Write the lock; call only after a successful evaluation."""
        self.acquire_or_raise()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        body = {"timestamp": datetime.now(timezone.utc).isoformat(), "payload": payload}
        self.path.write_text(json.dumps(body, indent=2, default=str))


# ---------------------------------------------------------------- permutation
def permutation_sanity(fit_predict_fn: Callable, X_train, y_train, X_eval, y_eval,
                       seed: int = 0) -> float:
    """Shuffle train labels, fit, return eval accuracy (expect ~0.5).

    ``fit_predict_fn(X_train, y_train, X_eval) -> P(up)`` for X_eval.
    """
    rng = np.random.default_rng(seed)
    y_perm = rng.permutation(np.asarray(y_train))
    p = fit_predict_fn(X_train, y_perm, X_eval)
    return float(_correct(p, y_eval).mean())

import numpy as np
import pandas as pd
import pytest

from src.models import selective as S


def _synthetic(n, seed=0, signal=1.5):
    rng = np.random.default_rng(seed)
    z = rng.normal(size=n)
    y = (rng.random(n) < 1 / (1 + np.exp(-signal * z))).astype(int)
    p = 1 / (1 + np.exp(-signal * z))
    return p, y


def test_confidence_and_class():
    p = np.array([0.9, 0.5, 0.2])
    assert np.allclose(S.confidence(p), [0.9, 0.5, 0.8])
    assert list(S.predicted_class(p)) == [1, 1, 0]


def test_calibrator_isotonic_monotone_and_auto_choice():
    p, y = _synthetic(1500)
    cal = S.fit_calibrator(p, y)
    assert cal.method == "isotonic" and not cal.degenerate
    grid = np.linspace(-0.2, 1.2, 50)  # out of bounds are clipped
    out = cal.transform(grid)
    assert np.all(np.diff(out) >= -1e-12)
    assert out.min() >= 0 and out.max() <= 1


def test_calibrator_platt_small_n_monotone():
    p, y = _synthetic(200)
    cal = S.fit_calibrator(p, y)
    assert cal.method == "platt"
    out = cal.transform(np.linspace(0.01, 0.99, 30))
    assert np.all(np.diff(out) >= -1e-12)


def test_calibrator_single_class_identity_with_warning():
    p = np.array([0.2, 0.6, 0.8])
    with pytest.warns(UserWarning):
        cal = S.fit_calibrator(p, np.ones(3))
    assert cal.degenerate and cal.method == "identity"
    assert np.allclose(cal.transform(p), p)


def test_threshold_respects_min_n_and_coverage():
    p, y = _synthetic(4000, seed=1)
    r = S.choose_threshold(p, y, min_coverage=0.05, min_n=300)
    assert r["feasible"]
    assert r["val_n"] >= 300 and r["val_coverage"] >= 0.05
    conf = S.confidence(p)
    corr = (S.predicted_class(p) == y)
    best = max(corr[conf >= t].mean() for t in np.unique(conf)
               if (conf >= t).sum() >= 300 and (conf >= t).mean() >= 0.05)
    assert r["val_accuracy"] == pytest.approx(best)


def test_threshold_min_n_binds_over_coverage():
    p, y = _synthetic(2000, seed=2)
    r = S.choose_threshold(p, y, min_coverage=0.01, min_n=1000)
    assert r["val_n"] >= 1000


def test_threshold_ties_prefer_larger_coverage():
    p = np.array([0.9, 0.8, 0.7, 0.1, 0.2, 0.3])
    y = np.array([1, 1, 1, 0, 0, 0])
    r = S.choose_threshold(p, y, min_coverage=0.1, min_n=1)
    assert r["val_coverage"] == 1.0 and r["threshold"] == pytest.approx(0.7)


def test_threshold_infeasible():
    p, y = _synthetic(100)
    r = S.choose_threshold(p, y, min_coverage=0.05, min_n=300)
    assert r["feasible"] is False and r["threshold"] == 0.5
    assert r["val_coverage"] == 1.0
    e = S.choose_threshold(np.array([]), np.array([]))
    assert e["feasible"] is False


def test_bootstrap_contains_point_and_shrinks_with_n():
    rng = np.random.default_rng(3)

    def ci(n_days):
        dates = np.repeat(pd.bdate_range("2020-01-01", periods=n_days), 5)
        c = (rng.random(len(dates)) < 0.55).astype(float)
        return c.mean(), S.moving_block_bootstrap_ci(c, dates, block=10, n_boot=500, seed=0)

    pt1, (lo1, hi1) = ci(100)
    pt2, (lo2, hi2) = ci(1000)
    assert lo1 <= pt1 <= hi1 and lo2 <= pt2 <= hi2
    assert (hi2 - lo2) < (hi1 - lo1)


def test_bootstrap_groups_dates_and_empty():
    # Two dates (all-correct, all-wrong), block=1: whole dates are resampled,
    # so a bootstrap accuracy is a weighted mix of 1.0 and 0.0 with weights 3:1.
    dates = np.array(["2020-01-01"] * 3 + ["2020-01-02"] * 1)
    c = np.array([1, 1, 1, 0.0])
    lo, hi = S.moving_block_bootstrap_ci(c, dates, block=1, n_boot=500, seed=0)
    assert lo == 0.0 and hi == 1.0
    lo, hi = S.moving_block_bootstrap_ci(np.array([]), np.array([]))
    assert np.isnan(lo) and np.isnan(hi)


def test_selective_report_hand_computed():
    p = np.array([0.9, 0.8, 0.7, 0.6, 0.55, 0.45, 0.3, 0.2, 0.1, 0.5])
    y = np.array([1, 1, 0, 1, 0, 0, 0, 1, 0, 1])
    dates = pd.bdate_range("2021-01-01", periods=10)
    r = S.selective_report(p, y, 0.69, dates, train_majority_class=1, n_boot=50, block=2)
    # called rows: idx 0,1,2,6,7,8
    assert r["n_rows"] == 10 and r["n_calls"] == 6
    assert r["unrestricted_accuracy"] == pytest.approx(0.7)
    assert r["coverage"] == pytest.approx(0.6)
    assert r["confident_accuracy"] == pytest.approx(4 / 6)
    assert r["local_majority_rows"] == pytest.approx(0.5)
    assert r["local_majority_calls"] == pytest.approx(0.5)
    assert r["train_majority_applied_rows"] == pytest.approx(0.5)
    assert r["train_majority_applied_calls"] == pytest.approx(0.5)
    assert r["up_rate_calls"] == pytest.approx(0.5)
    assert r["pred_up_share_calls"] == pytest.approx(0.5)
    assert "n_calls<300" in r["flags"]
    lo, hi = r["confident_accuracy_ci"]
    assert 0 <= lo <= hi <= 1


def test_selective_report_no_calls():
    r = S.selective_report(np.array([0.6, 0.4]), np.array([1, 0]), 0.99,
                           pd.bdate_range("2021-01-01", periods=2), 1, n_boot=10)
    assert r["n_calls"] == 0 and np.isnan(r["confident_accuracy"])


def test_curve_monotone_coverage_and_accuracy_gain():
    p, y = _synthetic(3000, seed=4)
    df = S.accuracy_coverage_curve(p, y)
    assert list(df["target_coverage"]) == [1.0, 0.5, 0.2, 0.1, 0.05, 0.02, 0.01]
    assert np.all(np.diff(df["actual_coverage"]) < 0)
    assert np.all(np.diff(df["threshold"]) >= 0)
    assert df["accuracy"].iloc[-1] > df["accuracy"].iloc[0]
    pts = S.accuracy_coverage_points(p, y)
    assert len(pts) == 3000 and np.all(np.diff(pts["coverage"]) > 0)
    assert pts["accuracy"].iloc[-1] == pytest.approx(df["accuracy"].iloc[0])


def test_lock_refuses_second_run(tmp_path):
    lock = S.TestOnceLock(tmp_path / "sub" / "test.lock")
    lock.acquire_or_raise()
    assert not lock.path.exists()
    lock.commit({"desc": "final test"})
    assert lock.path.exists()
    with pytest.raises(RuntimeError, match="already evaluated"):
        S.TestOnceLock(lock.path).acquire_or_raise()
    with pytest.raises(RuntimeError):
        lock.commit({})


def test_permutation_sanity_near_chance():
    from sklearn.linear_model import LogisticRegression

    rng = np.random.default_rng(5)
    X = rng.normal(size=(2000, 3))
    y = (X[:, 0] > 0).astype(int)

    def fp(Xt, yt, Xe):
        return LogisticRegression().fit(Xt, yt).predict_proba(Xe)[:, 1]

    # a single shuffled fit gives a random decision direction, so average seeds
    accs = [S.permutation_sanity(fp, X[:1000], y[:1000], X[1000:], y[1000:], seed=s)
            for s in range(30)]
    assert 0.45 < float(np.mean(accs)) < 0.55

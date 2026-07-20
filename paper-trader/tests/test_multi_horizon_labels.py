"""
Covers the multi-horizon labeling work: predict_from_state's new
`label_column` parameter (weekly/twenty-day strategies train on their own
real label instead of secretly reusing direction-daily's 10d/2% label), and
build_universe_feature_state now generating all three label columns
(label, label_7d, label_20d_4pct) on the live-appended-rows path, not just
carrying them through from the frozen historical matrix.
"""
import numpy as np
import pandas as pd
import pytest

from paper_trader.models import predictor as predictor_mod
from paper_trader.models import feature_store as fs
from src.models.ablation import TECHNICAL_COLUMNS, SENTIMENT_COLUMNS

FEATURE_COLUMNS = TECHNICAL_COLUMNS + SENTIMENT_COLUMNS


def _synthetic_training_state():
    """
    A minimal but valid state dict: complete feature columns (no NaNs, so
    nothing gets dropna'd away) plus two distinguishable label columns -
    "label" is all 0.0, "label_alt" is all 1.0 - so whichever column actually
    got passed to the fit function is unambiguous from the captured y values.
    """
    n = 6
    dates = pd.bdate_range("2024-01-02", periods=n)
    training = pd.DataFrame({c: np.arange(n, dtype=float) for c in FEATURE_COLUMNS}, index=dates)
    training["label"] = 0.0
    training["label_alt"] = 1.0
    training["ticker"] = "SYN"

    predict_rows = pd.DataFrame({c: [0.0] for c in FEATURE_COLUMNS})
    predict_rows["ticker"] = "SYN"

    return {
        "training_matrix": training,
        "predict_rows": predict_rows,
        "data_window_hash": "deadbeef",
        "as_of_date": "2024-01-10",
    }


@pytest.fixture
def fake_fit_and_predict(monkeypatch):
    """Replace the real lightgbm fit/predict with instrumented fakes so the
    test asserts on *which label series was trained against*, not on model
    internals."""
    captured = {}

    def fake_fit(X, y):
        captured["y"] = y.copy()
        return "fake-model", None

    def fake_predict_proba(model, X, scaler=None):
        return np.full(len(X), 0.5)

    monkeypatch.setitem(predictor_mod._FIT_FUNCS, "lightgbm", fake_fit)
    monkeypatch.setattr(predictor_mod, "predict_proba", fake_predict_proba)
    return captured


class TestPredictFromStateLabelColumn:
    def test_explicit_label_column_trains_on_that_column_not_label(self, fake_fit_and_predict):
        state = _synthetic_training_state()
        model_cfg = {"feature_set": "base", "algorithm": "lightgbm", "horizon_days": 7}

        result = predictor_mod.predict_from_state(state, model_cfg=model_cfg, label_column="label_alt")

        assert fake_fit_and_predict["y"].tolist() == state["training_matrix"]["label_alt"].tolist()
        assert fake_fit_and_predict["y"].tolist() != state["training_matrix"]["label"].tolist()
        assert result, "expected a prediction for the synthetic ticker"
        assert result["SYN"]["horizon"] == 7

    def test_default_label_column_omitted_trains_on_label_unchanged(self, fake_fit_and_predict):
        state = _synthetic_training_state()
        model_cfg = {"feature_set": "base", "algorithm": "lightgbm", "horizon_days": 10}

        result = predictor_mod.predict_from_state(state, model_cfg=model_cfg)  # label_column omitted

        assert fake_fit_and_predict["y"].tolist() == state["training_matrix"]["label"].tolist()
        assert result["SYN"]["horizon"] == 10


class TestBuildUniverseFeatureStateAllLabels:
    @pytest.fixture
    def synthetic_price_history(self):
        """
        >=252 trading days (technical.py's 52-week window needs min_periods
        252) plus enough runway past that for a 20-trading-day label to
        resolve, so every one of the three label columns can actually
        produce non-null values somewhere in this series - not achievable
        with the repo's real fixture dates, whose raw price cache ends
        2025-12-31 and is too close to any candidate as_of_date for
        label_20d_4pct (needs 20 future trading days) to ever resolve.
        """
        rng = np.random.default_rng(42)
        n = 320
        dates = pd.bdate_range("2023-01-02", periods=n)
        close = np.maximum(100 + np.cumsum(rng.normal(0, 1, n)), 1.0)
        frame = pd.DataFrame({
            "Open": close + rng.normal(0, 0.5, n),
            "High": close + rng.uniform(0, 1, n),
            "Low": close - rng.uniform(0, 1, n),
            "Close": close,
            "Volume": rng.integers(1_000_000, 5_000_000, n).astype(float),
        }, index=dates)
        frame.index.name = "date"
        return frame

    @pytest.fixture
    def offline_live_path(self, monkeypatch, tmp_path, synthetic_price_history):
        # Force every row through the live-computed path (not the frozen
        # historical matrix) by making the "historical" matrix empty.
        monkeypatch.setattr(fs, "_load_historical_matrix", lambda: pd.DataFrame(index=pd.DatetimeIndex([], name="date")))
        monkeypatch.setattr(fs, "_combined_price_series", lambda ticker, as_of_date: synthetic_price_history.copy())
        monkeypatch.setattr(fs, "fetch_universe_news", lambda tickers, start, end, api_key: ({}, {t: "empty" for t in tickers}))
        monkeypatch.setattr(fs, "LIVE_SENTIMENT_DIR", tmp_path / "live_sentiment")
        return synthetic_price_history

    def test_all_three_label_columns_present_and_resolved_on_live_rows(self, offline_live_path):
        as_of_date = offline_live_path.index.max().date().isoformat()

        state = fs.build_universe_feature_state(["SYN"], as_of_date, feature_set="base")
        training = state["training_matrix"]

        for col in fs.ALL_LABEL_COLUMNS:
            assert col in training.columns, f"{col} missing from training_matrix"

        # Everything here came through the live path (historical was forced
        # empty), so any non-null value proves the live path itself computes
        # the label, not just carries a pre-existing column through unchanged.
        for col in fs.ALL_LABEL_COLUMNS:
            assert training[col].notna().any(), f"{col} never resolved on any live-computed row"

        assert set(training[fs.ALL_LABEL_COLUMNS].stack().unique()) <= {0.0, 1.0}

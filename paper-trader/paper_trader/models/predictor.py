import subprocess

from paper_trader import config
from paper_trader.models.feature_store import build_universe_feature_state, feature_columns

from src.models.train import fit_lightgbm, fit_logistic_regression, fit_random_forest, fit_xgboost, predict_proba

_FIT_FUNCS = {
    "lightgbm": lambda X, y: (fit_lightgbm(X, y), None),
    "random_forest": lambda X, y: (fit_random_forest(X, y), None),
    "xgboost": lambda X, y: (fit_xgboost(X, y), None),
    "logistic_regression": fit_logistic_regression,
}

_MODEL_VERSION = None


def _get_model_version() -> str:
    global _MODEL_VERSION
    if _MODEL_VERSION is None:
        try:
            out = subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"],
                cwd=config.RESEARCH_REPO_ROOT, stderr=subprocess.DEVNULL,
            )
            _MODEL_VERSION = out.decode().strip()
        except Exception:
            _MODEL_VERSION = "unknown"
    return _MODEL_VERSION


def predict_from_state(state: dict, model_cfg: dict = None) -> dict:
    """
    Confidence-gated direction predictions for every ticker present in a
    pre-built feature state (see feature_store.build_universe_feature_state).
    Returns {ticker: {direction, confidence, proba_up, horizon, model_version,
    data_window_hash, as_of_date}}. Tickers whose predict-row was dropped for
    missing features (e.g. too little price history) are simply absent.
    """
    model_cfg = model_cfg or config.MODEL_CFG
    cols = feature_columns(model_cfg["feature_set"])

    training = state["training_matrix"].dropna(subset=cols + ["label"])
    predict_rows = state["predict_rows"]
    if training.empty or predict_rows.empty:
        return {}

    predict_rows = predict_rows.dropna(subset=cols)
    if predict_rows.empty:
        return {}

    fit_func = _FIT_FUNCS[model_cfg["algorithm"]]
    model, scaler = fit_func(training[cols], training["label"])
    proba_up = predict_proba(model, predict_rows[cols], scaler=scaler)

    model_version = _get_model_version()
    horizon = model_cfg["horizon_days"]
    as_of_date = state.get("as_of_date")

    results = {}
    for i, (_, row) in enumerate(predict_rows.iterrows()):
        p = float(proba_up[i])
        results[row["ticker"]] = {
            "direction": "up" if p >= 0.5 else "down",
            "confidence": max(p, 1 - p),
            "proba_up": p,
            "horizon": horizon,
            "model_version": model_version,
            "data_window_hash": state["data_window_hash"],
            "as_of_date": as_of_date,
        }
    return results


def predict_universe(tickers: list, as_of_date: str, model_cfg: dict = None) -> dict:
    """Builds its own feature state - convenience entrypoint for tests/standalone use."""
    model_cfg = model_cfg or config.MODEL_CFG
    state = build_universe_feature_state(tickers, as_of_date, feature_set=model_cfg["feature_set"])
    state["as_of_date"] = as_of_date
    return predict_from_state(state, model_cfg=model_cfg)


def predict(ticker: str, as_of_date: str, model_cfg: dict = None) -> dict:
    """predict(ticker, as_of_date) -> {direction, confidence, horizon} - the Phase 2 stable interface."""
    return predict_universe([ticker], as_of_date, model_cfg=model_cfg).get(ticker)

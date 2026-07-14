import pandas as pd

from src.models.train import fit_lightgbm, predict_proba
from src.models.evaluate import accuracy_at_threshold

TECHNICAL_COLUMNS = [
    "return_1d", "return_5d", "return_10d", "return_20d",
    "rsi_14", "macd", "macd_signal", "macd_hist",
    "volatility_20d", "volume_ratio_20d", "dist_from_ma50",
    "bollinger_pctb", "bollinger_bandwidth", "atr_14_norm",
    "stochastic_k", "stochastic_d", "obv_zscore",
    "dist_from_52w_high", "dist_from_52w_low",
    "day_of_week_sin", "day_of_week_cos",
]

SENTIMENT_COLUMNS = [
    "sentiment_mean", "sentiment_std", "headline_count",
    "sentiment_roll3", "sentiment_roll7",
]


def run_ablation(X_train: pd.DataFrame, y_train: pd.Series, X_eval: pd.DataFrame, y_eval: pd.Series) -> pd.DataFrame:
    variants = {
        "technical_only": TECHNICAL_COLUMNS,
        "sentiment_only": SENTIMENT_COLUMNS,
        "both": TECHNICAL_COLUMNS + SENTIMENT_COLUMNS,
    }

    rows = {}
    for name, columns in variants.items():
        model = fit_lightgbm(X_train[columns], y_train)
        proba = predict_proba(model, X_eval[columns])
        result = accuracy_at_threshold(y_eval, proba, threshold=0.5)
        rows[name] = {"accuracy": result["accuracy"], "coverage": result["coverage"]}

    return pd.DataFrame.from_dict(rows, orient="index")

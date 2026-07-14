import numpy as np
import pandas as pd
from src.models.ablation import run_ablation, TECHNICAL_COLUMNS, SENTIMENT_COLUMNS


def _fixture(n=150, seed=0):
    rng = np.random.RandomState(seed)
    data = {col: rng.normal(size=n) for col in TECHNICAL_COLUMNS + SENTIMENT_COLUMNS}
    X = pd.DataFrame(data)
    # label driven only by a technical column, so technical_only should clearly outperform sentiment_only
    y = pd.Series((X[TECHNICAL_COLUMNS[0]] > 0).astype(float), name="label")
    return X, y


def test_technical_and_sentiment_column_lists_are_disjoint_and_nonempty():
    assert len(TECHNICAL_COLUMNS) > 0
    assert len(SENTIMENT_COLUMNS) > 0
    assert set(TECHNICAL_COLUMNS).isdisjoint(set(SENTIMENT_COLUMNS))


def test_run_ablation_returns_three_rows():
    X, y = _fixture()
    X_train, y_train = X.iloc[:100], y.iloc[:100]
    X_eval, y_eval = X.iloc[100:], y.iloc[100:]
    result = run_ablation(X_train, y_train, X_eval, y_eval)
    assert set(result.index) == {"technical_only", "sentiment_only", "both"}
    assert "accuracy" in result.columns


def test_technical_only_outperforms_sentiment_only_on_technical_driven_label():
    X, y = _fixture()
    X_train, y_train = X.iloc[:100], y.iloc[:100]
    X_eval, y_eval = X.iloc[100:], y.iloc[100:]
    result = run_ablation(X_train, y_train, X_eval, y_eval)
    assert result.loc["technical_only", "accuracy"] > result.loc["sentiment_only", "accuracy"]

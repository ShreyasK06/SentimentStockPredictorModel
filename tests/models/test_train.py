import numpy as np
import pandas as pd
import pytest
from src.models.train import fit_logistic_regression, fit_lightgbm, predict_proba


def _separable_data(n=200, seed=0):
    rng = np.random.RandomState(seed)
    X = pd.DataFrame({
        "f1": rng.normal(size=n),
        "f2": rng.normal(size=n),
    })
    # y is a clean linear function of f1 so both models can learn it easily
    y = pd.Series((X["f1"] > 0).astype(float), name="label")
    return X, y


def test_fit_logistic_regression_returns_model_and_scaler():
    X, y = _separable_data()
    model, scaler = fit_logistic_regression(X, y)
    assert hasattr(model, "predict_proba")
    assert hasattr(scaler, "transform")


def test_logistic_regression_learns_separable_pattern():
    X, y = _separable_data()
    model, scaler = fit_logistic_regression(X, y)
    proba = predict_proba(model, X, scaler=scaler)
    predicted = (proba >= 0.5).astype(float)
    accuracy = (predicted == y.values).mean()
    assert accuracy > 0.85


def test_fit_lightgbm_returns_fitted_model():
    X, y = _separable_data()
    model = fit_lightgbm(X, y)
    assert hasattr(model, "predict_proba")


def test_lightgbm_learns_separable_pattern():
    X, y = _separable_data()
    model = fit_lightgbm(X, y)
    proba = predict_proba(model, X)
    predicted = (proba >= 0.5).astype(float)
    accuracy = (predicted == y.values).mean()
    assert accuracy > 0.85


def test_predict_proba_requires_scaler_for_logistic_but_not_lightgbm():
    X, y = _separable_data()
    lr_model, scaler = fit_logistic_regression(X, y)
    lgbm_model = fit_lightgbm(X, y)
    lr_proba = predict_proba(lr_model, X, scaler=scaler)
    lgbm_proba = predict_proba(lgbm_model, X)
    assert lr_proba.shape == (len(X),)
    assert lgbm_proba.shape == (len(X),)

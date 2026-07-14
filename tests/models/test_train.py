import numpy as np
import pandas as pd
import pytest
from src.models.train import fit_logistic_regression, fit_lightgbm, fit_random_forest, fit_xgboost, predict_proba


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


def test_predict_proba_does_not_refit_scaler_on_different_distribution():
    """Regression test for data leakage prevention.

    Verifies that predict_proba() calls scaler.transform() (not fit_transform())
    when a scaler is provided. This prevents scaler refitting on validation/test data,
    which would cause data leakage across train/validate/test splits.

    The test works by:
    1. Fitting a scaler on one distribution (mean~0, std~1)
    2. Creating a second, DIFFERENT distribution (mean~500, std~100)
    3. Calling predict_proba with the original scaler on the new data
    4. Asserting that scaler.mean_ and scaler.scale_ are UNCHANGED

    If someone changes scaler.transform() to scaler.fit_transform() in predict_proba,
    the scaler's fitted parameters would change to match the new distribution,
    causing this test to fail.
    """
    # Step 1: Fit a scaler on one distribution
    X1, y1 = _separable_data(n=200, seed=0)
    model, scaler = fit_logistic_regression(X1, y1)

    # Capture the original scaler's fitted state after fitting on X1
    # X1 has mean ~ 0, std ~ 1, so:
    # scaler.mean_ should be close to [0, 0]
    # scaler.scale_ should be close to [1, 1]
    original_mean = scaler.mean_.copy()
    original_scale = scaler.scale_.copy()

    # Step 2: Create a SECOND, differently-distributed dataset
    # Shift and scale to have very different statistics:
    # X2 = X1 * 100 + 500  =>  mean ~ 500, std ~ 100
    X2 = pd.DataFrame({
        "f1": X1["f1"] * 100 + 500,
        "f2": X1["f2"] * 100 + 500,
    })

    # Step 3: Call predict_proba with the original scaler on the different data
    # The scaler MUST use its original fitted parameters (mean~0, scale~1),
    # NOT refit on X2 (which would give mean~500, scale~100)
    proba = predict_proba(model, X2, scaler=scaler)

    # Step 4: Verify the scaler's fitted state was NOT mutated
    # If scaler.fit_transform() was called instead of scaler.transform(),
    # these parameters would have changed to match X2's very different distribution
    np.testing.assert_array_equal(scaler.mean_, original_mean,
                                   err_msg="Scaler mean was unexpectedly changed during predict_proba")
    np.testing.assert_array_equal(scaler.scale_, original_scale,
                                   err_msg="Scaler scale was unexpectedly changed during predict_proba")

    # Also verify that predict_proba returned valid output
    assert proba.shape == (len(X2),)
    assert np.all((proba >= 0) & (proba <= 1)), "Probabilities should be in [0, 1]"


def test_fit_random_forest_returns_fitted_model():
    X, y = _separable_data()
    model = fit_random_forest(X, y)
    assert hasattr(model, "predict_proba")


def test_random_forest_learns_separable_pattern():
    X, y = _separable_data()
    model = fit_random_forest(X, y)
    proba = predict_proba(model, X)
    predicted = (proba >= 0.5).astype(float)
    accuracy = (predicted == y.values).mean()
    assert accuracy > 0.85


def test_fit_xgboost_returns_fitted_model():
    X, y = _separable_data()
    model = fit_xgboost(X, y)
    assert hasattr(model, "predict_proba")


def test_xgboost_learns_separable_pattern():
    X, y = _separable_data()
    model = fit_xgboost(X, y)
    proba = predict_proba(model, X)
    predicted = (proba >= 0.5).astype(float)
    accuracy = (predicted == y.values).mean()
    assert accuracy > 0.85

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from lightgbm import LGBMClassifier


def fit_logistic_regression(X_train: pd.DataFrame, y_train: pd.Series, scaler=None):
    if scaler is None:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_train)
    else:
        X_scaled = scaler.transform(X_train)
    model = LogisticRegression(max_iter=1000)
    model.fit(X_scaled, y_train)
    return model, scaler


def fit_lightgbm(X_train: pd.DataFrame, y_train: pd.Series):
    model = LGBMClassifier(verbosity=-1)
    model.fit(X_train, y_train)
    return model


def predict_proba(model, X: pd.DataFrame, scaler=None) -> np.ndarray:
    if scaler is not None:
        X = scaler.transform(X)
    return model.predict_proba(X)[:, 1]

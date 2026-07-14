import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier


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


def fit_random_forest(X_train: pd.DataFrame, y_train: pd.Series) -> RandomForestClassifier:
    model = RandomForestClassifier(n_estimators=200, max_depth=6, random_state=0)
    model.fit(X_train, y_train)
    return model


def fit_xgboost(X_train: pd.DataFrame, y_train: pd.Series) -> XGBClassifier:
    model = XGBClassifier(n_estimators=200, max_depth=4, eval_metric="logloss")
    model.fit(X_train, y_train)
    return model

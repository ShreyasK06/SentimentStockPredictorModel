"""Phase 8 next-day direction model classes + live decision helper.

Kept in src/ (not in the script) so pickled models in models/phase8/final_models.joblib
can be loaded by scripts/predict_news_direction.py.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

SEED = 0


def cols(df, kind):
    news = [c for c in df.columns if c.startswith("news_")]
    ctl = [c for c in df.columns if c.startswith("ctl_")]
    return {"news": news, "ctl": ctl, "both": news + ctl}[kind]


# --------------------------------------------------------------------- models
class TabModel:
    """Logistic or LightGBM on tabular features (NaNs median-imputed from train)."""

    def __init__(self, kind, feats, **hp):
        self.kind, self.feats, self.hp = kind, feats, hp

    def _X(self, df):
        return df[self.feats].astype(float).fillna(self.med).to_numpy()

    def fit(self, df, emb, y):
        self.med = df[self.feats].astype(float).median()
        X = self._X(df)
        if self.kind == "logit":
            self.m = make_pipeline(StandardScaler(), LogisticRegression(C=self.hp["C"], max_iter=2000))
        else:
            import lightgbm as lgb
            self.m = lgb.LGBMClassifier(n_estimators=self.hp["n_estimators"], learning_rate=0.03,
                                        num_leaves=self.hp["num_leaves"], min_child_samples=200,
                                        subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                                        reg_lambda=1.0, random_state=SEED, verbose=-1)
        self.m.fit(X, y)
        return self

    def predict(self, df, emb):
        return self.m.predict_proba(self._X(df))[:, 1]


class EmbModel:
    """Frozen FinBERT embedding head: PCA (fit on train) + controls -> logistic."""

    def __init__(self, ctl_feats, n_comp=32, C=0.1):
        self.ctl, self.n_comp, self.C = ctl_feats, n_comp, C

    def _X(self, df, emb):
        z = self.pca.transform(emb)
        c = df[self.ctl].astype(float).fillna(self.med).to_numpy()
        return np.hstack([z, c])

    def fit(self, df, emb, y):
        self.pca = PCA(n_components=self.n_comp, random_state=SEED).fit(emb)
        self.med = df[self.ctl].astype(float).median()
        self.m = make_pipeline(StandardScaler(), LogisticRegression(C=self.C, max_iter=2000))
        self.m.fit(self._X(df, emb), y)
        return self

    def predict(self, df, emb):
        return self.m.predict_proba(self._X(df, emb))[:, 1]


def decide(p_cal: float, threshold: float) -> str:
    """UP / DOWN when calibrated confidence max(p, 1-p) >= validation threshold, else ABSTAIN."""
    if p_cal is None or not np.isfinite(p_cal):
        return "ABSTAIN"
    conf = max(p_cal, 1.0 - p_cal)
    if conf < threshold:
        return "ABSTAIN"
    return "UP" if p_cal >= 0.5 else "DOWN"

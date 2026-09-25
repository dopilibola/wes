"""Shared paths, data loading and the repeated-CV evaluation used by every experiment."""
import os
import warnings

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import QuantileTransformer

from base_features import build_features, extra

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.environ.get("WES_DATA", os.path.join(HERE, ".."))   # folder with the competition files
CACHE = os.path.join(HERE, "cache")
os.makedirs(CACHE, exist_ok=True)


def load_signals(split="train"):
    return pd.read_csv(os.path.join(DATA, f"{split}_signals.csv"), parse_dates=["signal_sanasi"])


def load_tx(split="train"):
    return pd.read_parquet(os.path.join(DATA, f"{split}_transactions.parquet"))


def cached(name, fn):
    """Compute a feature table once and keep it in research/cache/."""
    path = os.path.join(CACHE, name + ".parquet")
    if not os.path.exists(path):
        if fn is None:
            raise FileNotFoundError(f"{name} is not cached yet; run the earlier script that builds it")
        fn().to_parquet(path)
    return pd.read_parquet(path)


y = load_signals().eskalatsiya.values


def base():
    """The 284 features of the submitted model."""
    def make():
        s, t = load_signals(), load_tx()
        return build_features(t, s).join(extra(t, s))
    return cached("base_train", make).replace([np.inf, -np.inf], np.nan)


P = dict(n_estimators=600, learning_rate=0.01, num_leaves=7, min_child_samples=100, subsample=0.7,
         subsample_freq=1, colsample_bytree=0.3, reg_lambda=10, verbose=-1, n_jobs=-1)
rk = lambda a: pd.Series(a).rank(pct=True).values


def lgbm(p=P):
    return lambda sd: lgb.LGBMClassifier(**p, random_state=sd)


def logreg(C=0.005):
    return lambda sd: make_pipeline(SimpleImputer(strategy="median"), QuantileTransformer(output_distribution="normal"),
                                    LogisticRegression(C=C, max_iter=3000))


def oof(X, mk, seeds=range(5)):
    """Out-of-fold predictions for 5-fold stratified CV, one row per seed."""
    out = []
    for sd in seeds:
        o = np.zeros(len(y))
        for tr, va in StratifiedKFold(5, shuffle=True, random_state=sd).split(X, y):
            o[va] = mk(sd).fit(X.iloc[tr], y[tr]).predict_proba(X.iloc[va])[:, 1]
        out.append(o)
    return np.array(out)


def score(O):
    """Mean and std of AUC across seeds."""
    a = [roc_auc_score(y, o) for o in O]
    return np.mean(a), np.std(a)


def blend(A, B, w=0.4):
    return np.array([(1 - w) * rk(a) + w * rk(b) for a, b in zip(A, B)])


def save(name, arr):
    np.save(os.path.join(CACHE, name + ".npy"), arr)


def load(name):
    return np.load(os.path.join(CACHE, name + ".npy"))


def report(label, G, L, w=0.4):
    print(f"{label}: LightGBM {score(G)[0]:.4f} | LogReg {score(L)[0]:.4f} | blend {score(blend(G, L, w))[0]:.4f} ± {score(blend(G, L, w))[1]:.4f}", flush=True)

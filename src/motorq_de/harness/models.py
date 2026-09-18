"""Model factories and split strategies. Fixed hyper-parameters, seeded, deterministic."""

from __future__ import annotations

from collections.abc import Iterator

import lightgbm as lgb
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

LGBM_PARAMS = dict(
    n_estimators=200,
    learning_rate=0.05,
    num_leaves=31,
    min_child_samples=50,
    subsample=0.8,
    subsample_freq=1,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    n_jobs=4,
    verbose=-1,
    deterministic=True,
    force_row_wise=True,
)


def make_model(name: str, seed: int):
    if name == "lightgbm":
        return lgb.LGBMClassifier(random_state=seed, **LGBM_PARAMS)
    if name == "logistic":
        return Pipeline(
            [
                ("impute", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
                ("clf", LogisticRegression(max_iter=500, C=0.5, random_state=seed)),
            ]
        )
    raise ValueError(name)


def fit_predict(name: str, seed: int, X_tr, y_tr, w_tr, X_te) -> np.ndarray:
    m = make_model(name, seed)
    if name == "lightgbm":
        m.fit(X_tr, y_tr, sample_weight=w_tr)
    else:
        m.fit(X_tr, y_tr, clf__sample_weight=w_tr)
    return m.predict_proba(X_te)[:, 1]


def grouped_folds(
    y: np.ndarray, groups: np.ndarray, n_splits: int, seed: int
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    skf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    yield from skf.split(np.zeros(len(y)), y, groups)


def oof_predictions(
    name: str,
    X: np.ndarray,
    y: np.ndarray,
    w: np.ndarray,
    groups: np.ndarray,
    n_splits: int,
    seed: int,
) -> np.ndarray:
    """Out-of-fold predictions for every row, grouped by vehicle."""
    oof = np.full(len(y), np.nan)
    for tr, te in grouped_folds(y, groups, n_splits, seed):
        oof[te] = fit_predict(name, seed, X[tr], y[tr], w[tr], X[te])
    return oof


def temporal_split(
    dates: np.ndarray, train_frac: float, gap_days: int
) -> tuple[np.ndarray, np.ndarray]:
    """Train on the earliest `train_frac` of days, skip `gap_days`, test on the rest."""
    uniq = np.unique(dates)
    cut = uniq[int(len(uniq) * train_frac)]
    train = dates < cut
    test = dates >= cut + np.timedelta64(gap_days, "D")
    return np.flatnonzero(train), np.flatnonzero(test)

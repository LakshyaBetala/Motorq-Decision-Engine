"""Model factories and split strategies. Fixed hyper-parameters, seeded, deterministic."""

from __future__ import annotations

import os
from collections.abc import Callable, Iterator, Sequence
from typing import Any

import lightgbm as lgb
import numpy as np
from joblib import Parallel, delayed
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


def cpu_budget() -> int:
    """Cores this process may use: MDE_CPUS (set it to the container's CPU limit, since
    os.cpu_count() reports the host inside a container), else the machine's core count."""
    env = os.environ.get("MDE_CPUS")
    return max(1, int(env) if env else (os.cpu_count() or 1))


def cv_parallelism(n_folds: int) -> tuple[int, int]:
    """(worker processes, LightGBM threads per fit) for fold-parallel cross-validation.

    MDE_FOLD_JOBS overrides the worker count; otherwise one worker per three cores. Threads x
    workers never exceeds the budget. LightGBM's deterministic mode makes a fit identical for
    any thread count (pinned by tests/harness/test_models.py), so this only changes wall time.
    """
    budget = cpu_budget()
    env = os.environ.get("MDE_FOLD_JOBS")
    workers = max(1, min(n_folds, int(env) if env else budget // 3))
    return workers, max(1, budget // workers)


def make_model(name: str, seed: int, n_threads: int | None = None):
    if name == "lightgbm":
        params = dict(LGBM_PARAMS)
        if n_threads is not None:
            params["n_jobs"] = n_threads
        return lgb.LGBMClassifier(random_state=seed, **params)
    if name == "logistic":
        return Pipeline(
            [
                ("impute", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
                ("clf", LogisticRegression(max_iter=500, C=0.5, random_state=seed)),
            ]
        )
    raise ValueError(name)


def fit_predict(
    name: str, seed: int, X_tr, y_tr, w_tr, X_te, n_threads: int | None = None
) -> np.ndarray:
    m = make_model(name, seed, n_threads)
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


def run_folds(
    fn: Callable[..., Any], folds: Sequence[tuple[np.ndarray, np.ndarray]], *args: Any
) -> list[Any]:
    """Evaluate fn(*args, tr, te, n_threads) for every fold, in parallel processes when the
    core budget allows.

    Folds are independent seeded fits, so the result is identical to the sequential loop;
    only wall time changes. Large arrays in *args are memory-mapped to the workers once.
    """
    workers, threads = cv_parallelism(len(folds))
    if workers == 1:
        return [fn(*args, tr, te, threads) for tr, te in folds]
    return list(Parallel(n_jobs=workers)(delayed(fn)(*args, tr, te, threads) for tr, te in folds))


def _fit_fold(name: str, seed: int, X, y, w, tr, te, n_threads: int) -> np.ndarray:
    return fit_predict(name, seed, X[tr], y[tr], w[tr], X[te], n_threads)


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
    folds = list(grouped_folds(y, groups, n_splits, seed))
    preds = run_folds(_fit_fold, folds, name, seed, X, y, w)
    oof = np.full(len(y), np.nan)
    for (_, te), p in zip(folds, preds, strict=True):
        oof[te] = p
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

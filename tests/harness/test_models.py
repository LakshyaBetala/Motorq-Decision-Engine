"""Fold-parallel cross-validation must change wall time only, never a number."""

from __future__ import annotations

import numpy as np
import pytest

from motorq_de.harness import models


def _data(seed=0, n=6000, n_groups=300):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 6))
    y = (X[:, 0] + 0.5 * X[:, 1] + rng.normal(scale=1.5, size=n) > 1.5).astype(int)
    w = rng.uniform(0.5, 2.0, n)
    groups = rng.integers(0, n_groups, n)
    return X, y, w, groups


def test_lightgbm_fit_is_identical_for_any_thread_count():
    X, y, w, _ = _data()
    ref = models.fit_predict("lightgbm", 7, X[:4000], y[:4000], w[:4000], X[4000:], n_threads=1)
    for threads in (2, 4, 8):
        p = models.fit_predict("lightgbm", 7, X[:4000], y[:4000], w[:4000], X[4000:], threads)
        assert np.array_equal(p, ref), threads


def test_parallel_oof_equals_sequential(monkeypatch: pytest.MonkeyPatch):
    X, y, w, groups = _data()
    monkeypatch.setenv("MDE_FOLD_JOBS", "1")
    seq = models.oof_predictions("lightgbm", X, y, w, groups, 5, 3)
    monkeypatch.setenv("MDE_FOLD_JOBS", "3")
    monkeypatch.setenv("MDE_CPUS", "6")
    assert models.cv_parallelism(5) == (3, 2)
    par = models.oof_predictions("lightgbm", X, y, w, groups, 5, 3)
    assert not np.isnan(seq).any()
    assert np.array_equal(seq, par)


def test_cv_parallelism_respects_budget(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("MDE_FOLD_JOBS", raising=False)
    monkeypatch.setenv("MDE_CPUS", "2")  # a small container: sequential folds, 2 threads
    assert models.cv_parallelism(5) == (1, 2)
    monkeypatch.setenv("MDE_CPUS", "16")
    workers, threads = models.cv_parallelism(5)
    assert workers == 5 and threads == 3 and workers * threads <= 16
    monkeypatch.setenv("MDE_FOLD_JOBS", "2")
    assert models.cv_parallelism(5) == (2, 8)

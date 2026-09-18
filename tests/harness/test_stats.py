from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score

from motorq_de.harness import stats


def _data(seed=0, n=4000, sep=1.0):
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < 0.1).astype(int)
    s = np.round(rng.normal(0, 1, n) + y * sep, 1)
    w = rng.uniform(0.5, 3, n)
    g = rng.integers(0, 200, n)
    return y, s, w, g


def test_fast_auc_matches_sklearn_with_weights_and_ties():
    for seed in range(4):
        y, s, w, _ = _data(seed)
        assert abs(stats.fast_auc(y, s, w) - roc_auc_score(y, s, sample_weight=w)) < 1e-9
        assert abs(stats.fast_auc(y, s) - roc_auc_score(y, s)) < 1e-9


def test_cluster_bootstrap_is_deterministic_and_covers_point():
    y, s, w, g = _data()
    a = stats.cluster_bootstrap_auc(y, s, g, w, B=100, seed=1)
    b = stats.cluster_bootstrap_auc(y, s, g, w, B=100, seed=1)
    assert a == b
    assert a.lo <= a.point <= a.hi


def test_paired_delta_detects_real_difference():
    y, s, w, g = _data(sep=1.0)
    noise = np.random.default_rng(9).normal(0, 1.5, len(s))
    worse = s + noise
    d = stats.paired_bootstrap_delta_auc(y, worse, s, g, w, B=200, seed=2)
    assert d.point < 0 and d.hi < 0  # worse model is significantly worse
    assert d.prob_greater_than(-0.005) < 0.2


def test_paired_delta_null_when_identical():
    y, s, w, g = _data()
    d = stats.paired_bootstrap_delta_auc(y, s, s, g, w, B=50, seed=3)
    assert d.point == 0 and d.lo == 0 and d.hi == 0
    assert d.prob_greater_than(-0.005) == 1.0


def test_recall_at_top_fraction_monotone():
    y, s, w, _ = _data()
    r2 = stats.recall_at_top_fraction(y, s, 0.02, w)
    r5 = stats.recall_at_top_fraction(y, s, 0.05, w)
    r20 = stats.recall_at_top_fraction(y, s, 0.20, w)
    assert 0 <= r2 <= r5 <= r20 <= 1


def test_univariate_auc_ignores_nans_and_direction():
    y, s, w, _ = _data()
    x = s.copy()
    x[::7] = np.nan
    a = stats.univariate_auc(y, x, w)
    b = stats.univariate_auc(y, -x, w)
    assert abs(a - b) < 1e-9 and a > 0.6

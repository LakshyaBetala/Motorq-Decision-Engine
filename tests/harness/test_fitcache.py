"""The fit cache changes wall time only: a hit is the array a fresh fit would produce."""

from __future__ import annotations

import numpy as np

from motorq_de.harness.experiments import learning_curve
from motorq_de.harness.fitcache import FitCache
from motorq_de.harness.frames import FeatureStore

SIGS = ["brake_pad_wear_pct", "dtc_brake_family", "trip_distance_mi", "harsh_brake_count"]


def test_cache_hit_is_bit_identical_and_survives_a_new_store(source, brake_spec, tmp_path):
    cache = FitCache(directory=tmp_path)
    store = FeatureStore(source, fits=cache)
    M = store.matrix(brake_spec, SIGS)
    a = store.oof("lightgbm", M, brake_spec.seed, 5)
    assert cache.stats() == {"cache_hits": 0, "cache_misses": 1}
    b = store.oof("lightgbm", M, brake_spec.seed, 5)
    assert cache.stats() == {"cache_hits": 1, "cache_misses": 1}
    assert np.array_equal(a, b)
    # a sub-selection from a bigger matrix has the same rows: same key, same array
    big = store.matrix(brake_spec, SIGS + ["odometer_mi"]).select(SIGS)
    assert big.row_key == M.row_key and big.signals == M.signals
    c = store.oof("lightgbm", big, brake_spec.seed, 5)
    assert np.array_equal(a, c) and cache.hits == 2
    # a fresh process with the same directory reads it from disk
    store2 = FeatureStore(source, fits=FitCache(directory=tmp_path))
    d = store2.oof("lightgbm", store2.matrix(brake_spec, SIGS), brake_spec.seed, 5)
    assert np.array_equal(a, d) and store2.fits.hits == 1
    # disabled: recompute, still identical
    store2.fits.enabled = False
    e = store2.oof("lightgbm", store2.matrix(brake_spec, SIGS), brake_spec.seed, 5)
    assert np.array_equal(a, e) and store2.fits.hits == 1


def test_key_depends_on_signal_order_seed_and_environment():
    k1 = FitCache.key("rows", ["a", "b"], "lightgbm", 5, 1)
    assert k1 == FitCache.key("rows", ["a", "b"], "lightgbm", 5, 1)
    assert k1 != FitCache.key("rows", ["b", "a"], "lightgbm", 5, 1)
    assert k1 != FitCache.key("rows", ["a", "b"], "lightgbm", 5, 2)
    assert k1 != FitCache.key("other", ["a", "b"], "lightgbm", 5, 1)


def test_learning_curve_is_nested_and_full_fraction_hits_the_cache(source, brake_spec, tmp_path):
    store = FeatureStore(source, fits=FitCache(directory=tmp_path))
    M = store.matrix(brake_spec, SIGS)
    store.oof("lightgbm", M, brake_spec.seed, 5)  # what ablation would have done
    lc = learning_curve(store, brake_spec, SIGS, fractions=(0.5, 1.0))
    pts = lc["points"]
    assert [p["fraction"] for p in pts] == [0.5, 1.0]
    assert pts[0]["n_vehicles"] < pts[1]["n_vehicles"] and pts[0]["n_rows"] < pts[1]["n_rows"]
    assert pts[1]["n_rows"] == len(M.y)
    assert store.fits.hits == 1  # the 100% point reused the existing fit
    assert lc["auc_gain_half_to_full"] is not None
    assert isinstance(lc["still_improving"], bool)

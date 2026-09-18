from __future__ import annotations

import numpy as np

from motorq_de.harness.frames import SUFFIXES, FeatureStore, build_matrix


def test_matrix_shapes_and_weights(source, brake_spec):
    M = build_matrix(
        source, brake_spec, ["brake_pad_wear_pct", "harsh_brake_count"], max_rows=5000, neg_ratio=4
    )
    assert M.X.shape[1] == 2 * len(SUFFIXES)
    assert M.X.dtype == np.float32
    assert set(M.signal_of) == {"brake_pad_wear_pct", "harsh_brake_count"}
    assert len(M.y) == len(M.w) == len(M.groups) == len(M.oem) == len(M.dates) == M.X.shape[0]
    assert (M.y >= 0).all()
    # weights reconstruct the true base rate
    est = (M.w * (M.y == 1)).sum() / M.w.sum()
    assert abs(est - M.base_rate) / M.base_rate < 0.15


def test_matrix_is_deterministic(source, brake_spec):
    a = build_matrix(source, brake_spec, ["odometer_delta_mi"], 3000, 3)
    b = build_matrix(source, brake_spec, ["odometer_delta_mi"], 3000, 3)
    np.testing.assert_array_equal(a.X, b.X)
    np.testing.assert_array_equal(a.y, b.y)


def test_rolling_features_are_trailing_means(source, brake_spec):
    M = build_matrix(source, brake_spec, ["trip_distance_mi"], 200_000, 50)
    i_raw = M.feature_names.index("trip_distance_mi")
    i_m7 = M.feature_names.index("trip_distance_mi__m7")
    # trailing mean of a non-negative signal is non-negative and bounded by the running max
    x = M.X[:, i_m7]
    assert np.nanmin(x) >= -1e-6
    assert np.nanmax(x) <= np.nanmax(M.X[:, i_raw]) + 1e-3


def test_select_keeps_column_alignment(source, brake_spec):
    store = FeatureStore(source)
    M = store.matrix(brake_spec, ["brake_pad_wear_pct", "harsh_brake_count", "co2_kg"], 5000, 3)
    S = M.select(["co2_kg"])
    assert S.signals == ["co2_kg"] and S.X.shape[1] == len(SUFFIXES)
    np.testing.assert_array_equal(S.X[:, 0], M.X[:, M.feature_names.index("co2_kg")])


def test_store_caches(source, brake_spec):
    store = FeatureStore(source)
    a = store.matrix(brake_spec, ["co2_kg"], 2000, 2)
    b = store.matrix(brake_spec, ["co2_kg"], 2000, 2)
    assert a is b

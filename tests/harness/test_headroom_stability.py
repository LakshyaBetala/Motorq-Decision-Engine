"""Tuning headroom and seed stability on the fixture fleet."""

from __future__ import annotations

from motorq_de.harness.experiments import seed_stability, tuning_headroom
from motorq_de.harness.frames import FeatureStore
from motorq_de.harness.models import LGBM_VARIANTS, lgbm_params, make_model

SIGNALS = [
    "brake_pad_wear_pct",
    "brake_pad_wear_rear_pct",
    "trip_distance_mi",
    "harsh_brake_events",
]


def test_variant_names_resolve_to_distinct_parameter_sets():
    base = lgbm_params("lightgbm")
    for v in LGBM_VARIANTS:
        p = lgbm_params(f"lightgbm:{v}")
        assert p != base and p["deterministic"] and p["force_row_wise"]
        assert make_model(f"lightgbm:{v}", 1).get_params()["num_leaves"] == p["num_leaves"]


def test_tuning_headroom_reports_every_variant_against_the_default(source, brake_spec):
    th = tuning_headroom(FeatureStore(source), brake_spec, SIGNALS)
    assert set(th["variants"]) == set(LGBM_VARIANTS)
    for v, row in th["variants"].items():
        assert row["params"] == LGBM_VARIANTS[v]
        d = row["delta_vs_default"]
        assert d["lo"] <= d["point"] <= d["hi"]
        assert 0.5 < row["auc"]["point"] < 1.0
    assert th["best_variant"] in th["variants"]
    assert th["headroom"] == th["variants"][th["best_variant"]]["delta_vs_default"]["point"]
    assert th["loose_lower_bound"] == (th["headroom_lo"] > th["headroom_threshold"])
    # nothing in the fixed grid should move a saturated brake model by a large margin
    assert abs(th["headroom"]) < 0.05


def test_seed_stability_uses_distinct_fold_seeds_and_is_deterministic(source, brake_spec):
    a = seed_stability(FeatureStore(source), brake_spec, SIGNALS)
    assert len({r["seed"] for r in a["per_seed"]}) == 3
    assert a["auc_spread"] >= 0 and a["spread_threshold"] == 0.005
    assert a["seed_sensitive"] == (a["auc_spread"] > 0.005)
    b = seed_stability(FeatureStore(source), brake_spec, SIGNALS)
    assert a == b

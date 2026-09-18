"""Oracle tests: on synthetic data with known structure the harness must recover it.

Fast checks run on the small session fixture. Power-hungry checks (theft, battery,
strict ablation) run only when MDE_ORACLE_DATASET points at a default-profile dataset:

    MDE_ORACLE_DATASET=data/synthetic/<hash> uv run pytest tests/oracle -q
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from motorq_de.data.synthetic import SyntheticSource
from motorq_de.harness.experiments import (
    ablation,
    cross_oem_validation,
    feature_analysis,
    temporal_validation,
)
from motorq_de.harness.frames import FeatureStore
from motorq_de.quality.checks import leakage_check, quality_report, usable_signals
from motorq_de.world import registry as reg

ORACLE = os.environ.get("MDE_ORACLE_DATASET")
needs_big = pytest.mark.skipif(
    not ORACLE, reason="set MDE_ORACLE_DATASET to a default-profile dataset"
)


def _usable(source, spec):
    lk = leakage_check(source, spec, reg.signal_ids())
    q = quality_report(source, reg.signal_ids())
    return usable_signals(reg.signal_ids(), q, lk)["usable"]


@pytest.fixture(scope="module")
def big():
    return SyntheticSource(Path(ORACLE)) if ORACLE else None


# ------------------------------------------------------------------ small fixture (fast)


def test_brake_feature_analysis_ranks_sensor_first_and_gps_last(source, brake_spec, truth):
    usable = _usable(source, brake_spec)
    fa = feature_analysis(FeatureStore(source), brake_spec, usable, n_repeats=1)
    rank = [r["signal"] for r in fa["ranking"]]
    imp = {r["signal"]: r["perm_importance"] for r in fa["ranking"]}
    assert rank[0] == "brake_pad_wear_pct"
    assert "dtc_brake_family" in rank[:5]
    for sid in truth["noise"]:
        assert imp[sid] < 0.003, (sid, imp[sid])
    for sid in (
        "gps_lat_mean",
        "gps_lon_mean",
        "gps_fix_count",
        "night_park_share",
        "dwell_location_entropy",
    ):
        assert imp[sid] < 0.003, (sid, imp[sid])
    assert fa["auc"]["point"] > 0.8 and fa["auc"]["lo"] < fa["auc"]["point"] < fa["auc"]["hi"]


def test_brake_ablation_structure_on_small(source, brake_spec):
    usable = _usable(source, brake_spec)
    store = FeatureStore(source)
    fa = feature_analysis(store, brake_spec, usable, n_repeats=1)
    ab = ablation(
        store,
        brake_spec,
        usable,
        order_least_to_most=fa["order_least_to_most_important"],
        screen_k=12,
    )
    assert "brake_pad_wear_pct" in ab["sufficient_set"]
    assert len(ab["sufficient_set"]) < len(usable)
    assert ab["sufficient_auc"]["point"] > ab["full_auc"]["point"] - 0.01
    assert ab["trace"][0]["removed"] is None and ab["trace"][0]["accepted"]
    # the small fixture cannot resolve 0.0025 AUC: the tool must say so
    assert ab["underpowered"] is True


def test_cross_oem_reflects_sensor_gap(source, brake_spec):
    res = cross_oem_validation(
        FeatureStore(source),
        brake_spec,
        ["brake_pad_wear_pct", "dtc_brake_family", "trip_distance_mi", "harsh_brake_count"],
    )
    per = res["per_oem"]
    evaluated = {k: v for k, v in per.items() if v.get("auc")}
    assert len(evaluated) >= 3
    # OEMs without the sensor have a much higher share of missing features
    with_sensor = [
        k for k in evaluated if k in ("oem_a", "oem_b", "oem_d", "oem_f", "oem_i", "oem_j")
    ]
    without = [k for k in evaluated if k in ("oem_c", "oem_e", "oem_g", "oem_h")]
    if with_sensor and without:
        assert min(per[k]["feature_missing_share"] for k in without) > max(
            per[k]["feature_missing_share"] for k in with_sensor
        )


def test_temporal_validation_runs_forward(source, brake_spec):
    res = temporal_validation(
        FeatureStore(source),
        brake_spec,
        ["brake_pad_wear_pct", "dtc_brake_family", "trip_distance_mi"],
    )
    assert res["test_start"] > res["train_end"]
    assert res["forward_auc"]["point"] > 0.7
    assert -0.1 < res["degradation"] < 0.1


# ------------------------------------------------------------------ default dataset (power)


@needs_big
def test_brake_ablation_recovers_drivers_rejects_decoys_and_noise(big, brake_spec, truth):
    usable = _usable(big, brake_spec)
    store = FeatureStore(big)
    fa = feature_analysis(store, brake_spec, usable, n_repeats=1)
    ab = ablation(
        store, brake_spec, usable, order_least_to_most=fa["order_least_to_most_important"]
    )
    suff = set(ab["sufficient_set"])
    assert "brake_pad_wear_pct" in suff
    assert not (suff & set(truth["noise"])), suff & set(truth["noise"])
    assert not (suff & set(truth["decoys"])), suff & set(truth["decoys"])
    assert not any(s.startswith("gps_") for s in suff)
    assert len(suff) <= 12
    assert ab["underpowered"] is False


@needs_big
def test_theft_uses_location_not_wear(big, theft_spec, truth):
    usable = _usable(big, theft_spec)
    fa = feature_analysis(FeatureStore(big), theft_spec, usable, n_repeats=1)
    rank = [r["signal"] for r in fa["ranking"]]
    theft_drivers = set(truth["targets"]["theft_event"]["drivers"])
    assert len(theft_drivers & set(rank[:6])) >= 2, rank[:6]
    assert "brake_pad_wear_pct" not in rank[:10]


@needs_big
def test_battery_uses_ev_signals(big, battery_spec, truth):
    usable = _usable(big, battery_spec)
    fa = feature_analysis(FeatureStore(big), battery_spec, usable, n_repeats=1)
    rank = [r["signal"] for r in fa["ranking"]]
    drivers = set(truth["targets"]["battery_degradation_event"]["drivers"])
    assert len(drivers & set(rank[:8])) >= 2, rank[:8]


@needs_big
def test_cross_oem_worst_is_a_gapped_oem(big, brake_spec):
    res = cross_oem_validation(
        FeatureStore(big),
        brake_spec,
        [
            "brake_pad_wear_pct",
            "dtc_brake_family",
            "trip_distance_mi",
            "harsh_brake_count",
            "odometer_delta_mi",
        ],
    )
    assert res["worst_oem"] in {"oem_c", "oem_e", "oem_g", "oem_h"}
    assert res["min_auc"] < res["mean_auc"] - 0.02

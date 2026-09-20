"""Redundancy analysis recovers the planted structure: decoys are copies of their driver."""

from __future__ import annotations

from motorq_de.harness.experiments import redundancy
from motorq_de.harness.frames import FeatureStore


def test_decoys_cluster_with_their_driver_and_noise_stays_independent(source, brake_spec, truth):
    signals = [
        "brake_pad_wear_pct",
        "brake_pad_wear_rear_pct",
        "brake_wear_est_rear_pct",
        "trip_distance_mi",
        "odometer_delta_mi",
        "radio_volume_mean",
        "wiper_activations",
        "gps_lat_mean",
    ]
    r = redundancy(FeatureStore(source), brake_spec, signals)
    assert r["method"] == "spearman" and r["n_signals"] == len(signals)
    groups = {frozenset(g) for g in r["clusters"]}
    assert any({"brake_pad_wear_pct", "brake_pad_wear_rear_pct"} <= g for g in groups)
    assert any({"trip_distance_mi", "odometer_delta_mi"} <= g for g in groups)
    clustered = set().union(*groups) if groups else set()
    for noise in ("radio_volume_mean", "wiper_activations", "gps_lat_mean"):
        assert noise not in clustered, noise
    assert r["n_independent_groups"] < len(signals)
    assert all(abs(p["spearman"]) >= r["threshold"] for p in r["pairs"])

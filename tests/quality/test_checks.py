"""Quality engine must catch the planted leaks, keep the real sensors, and compute the
whole-set coverage the data gate depends on."""

from __future__ import annotations

from motorq_de.quality.checks import coverage_report, leakage_check, quality_report, usable_signals
from motorq_de.world import registry as reg


def test_leakage_detector_catches_planted_leaks_and_keeps_drivers(source, brake_spec, truth):
    lk = leakage_check(source, brake_spec, reg.signal_ids())
    planted = set(truth["targets"]["brake_service_event"]["leakage"])
    assert planted <= set(lk["flagged"]), lk["flagged"]
    drivers = set(truth["targets"]["brake_service_event"]["drivers"])
    assert not (drivers & set(lk["flagged"])), drivers & set(lk["flagged"])
    # reasons are specific, not generic
    assert "flag_lift" in lk["per_signal"]["service_appointment_scheduled"]["reasons"]
    assert "monotone_to_event" in lk["per_signal"]["service_interval_remaining_days"]["reasons"]


def test_leakage_detector_no_false_positives_on_noise(source, brake_spec, truth):
    lk = leakage_check(source, brake_spec, reg.noise_signals())
    assert lk["flagged"] == []


def test_coverage_full_set_reflects_oem_gaps(source):
    cov = coverage_report(source, ["brake_pad_wear_pct", "harsh_brake_count"])
    by = cov["fleet_share_full_set_by_oem"]
    assert by["oem_c"] == 0.0 and by["oem_g"] == 0.0
    assert by["oem_f"] > 0.9 and by["oem_i"] > 0.9
    assert 0.4 <= cov["fleet_share_full_set"] <= 0.8
    assert {"oem_c", "oem_g"} <= set(cov["oems_missing_any_signal"])


def test_coverage_without_gapped_signal_is_near_full(source):
    cov = coverage_report(source, ["harsh_brake_count", "odometer_delta_mi"])
    assert cov["fleet_share_full_set"] > 0.6  # limited only by model-year connectivity


def test_quality_report_flags_sparse_and_stale(source):
    q = quality_report(source, ["brake_pad_wear_pct", "odometer_delta_mi", "phone_usage_proxy"])
    assert q["per_signal"]["brake_pad_wear_pct"]["nonnull_rate"] > 0.4
    assert "sparse" not in q["flags"].get("odometer_delta_mi", [])


def test_usable_signals_drops_leaks_and_sparse():
    quality = {"flags": {"a": ["sparse"], "b": ["drift"]}}
    leakage = {"flagged": ["c"]}
    out = usable_signals(["a", "b", "c", "d"], quality, leakage)
    assert out["usable"] == ["b", "d"]
    assert out["dropped"] == {"a": "sparse", "c": "leakage"}

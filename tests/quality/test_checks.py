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


def _series(vehicle: str, values, start="2025-01-01"):
    import pandas as pd

    return pd.DataFrame(
        {
            "vehicle_id": vehicle,
            "date": pd.date_range(start, periods=len(values), freq="D"),
            "s": values,
        }
    )


def test_frozen_share_detects_stuck_sensor_but_not_bound_plateaus():
    import pandas as pd

    from motorq_de.data.frame_source import _frozen_share

    stuck = _series("stuck", [37.2] * 20)  # identical for 20 days: a stuck sensor
    live = _series("live", [30 + i * 0.5 for i in range(20)])
    full = _series("full", [100.0] * 20)  # battery charged to 100% every day: a set-point
    gap = _series("gap", [5.0] * 10, "2025-01-01")
    gap2 = _series("gap", [5.0] * 10, "2025-02-01")  # two 10-day runs separated by a gap
    df = pd.concat([stuck, live, full, gap, gap2], ignore_index=True)
    assert _frozen_share(df, "s", "pct") == 0.25  # only 'stuck' out of 4 vehicles
    assert _frozen_share(df, "s", "pct", run_days=8) == 0.5  # 'gap' now qualifies


def test_implausible_share_uses_unit_ranges():
    import pandas as pd

    from motorq_de.data.frame_source import _implausible_share

    v = pd.Series([50.0, 101.0, -1.0, 99.0])
    assert _implausible_share(v, "pct") == 0.5
    assert _implausible_share(pd.Series([-5.0, 3.0]), "kWh") == 0.5
    assert _implausible_share(pd.Series([-5.0, 3.0]), "index") == 0.0  # unranged unit


def test_generated_fleet_has_no_frozen_or_implausible_signals(source):
    q = quality_report(source, reg.signal_ids())
    bad = {k: v for k, v in q["flags"].items() if "frozen" in v or "implausible" in v}
    assert bad == {}, bad
    assert all(
        "frozen_share" in row and "implausible_share" in row for row in q["per_signal"].values()
    )

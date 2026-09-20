"""The signal registry's COVESA VSS mapping: well-formed, attached to metadata, and honest
about which signals are Motorq-derived (no VSS counterpart)."""

from __future__ import annotations

from motorq_de.world.registry import BY_ID, VSS


def test_vss_paths_are_well_formed_and_exposed_on_metadata():
    assert set(VSS) <= set(BY_ID), set(VSS) - set(BY_ID)
    for sid, path in VSS.items():
        assert path.startswith("Vehicle."), (sid, path)
        assert all(seg and seg[0].isupper() for seg in path.split(".")), (sid, path)
        assert BY_ID[sid].meta().vss == path
    assert VSS["brake_pad_wear_pct"].endswith("Brake.PadWear")
    assert VSS["service_interval_remaining_days"] == "Vehicle.Service.TimeToService"


def test_noise_and_derived_scores_have_no_vss_counterpart():
    for sid, d in BY_ID.items():
        if d.role == "noise":
            assert sid not in VSS, sid
        if d.category == "derived" and d.role not in ("leakage", "decoy"):
            assert sid not in VSS, sid
    # every raw health sensor with a physical unit is mapped, except the one VSS lacks
    no_vss_counterpart = {"air_filter_life_pct"}
    for sid, d in BY_ID.items():
        if d.category == "health" and d.unit in ("pct", "kPa", "V", "hr", "mi", "C"):
            assert (sid in VSS) != (sid in no_vss_counterpart), sid

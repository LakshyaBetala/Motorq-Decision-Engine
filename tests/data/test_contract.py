"""The canonical contract catches every ingestion defect that would silently corrupt a study."""

from __future__ import annotations

import pandas as pd

from motorq_de.data.contract import validate, validate_source

TODAY = pd.Timestamp("2026-09-20")


def _frames():
    veh = pd.DataFrame(
        {
            "vehicle_id": ["a", "b"],
            "oem": ["oem_a", "oem_b"],
            "model_year": [2022, 2023],
            "powertrain": ["ice", "ev"],
        }
    )
    days = pd.date_range("2025-01-01", periods=200, freq="D")
    sig = pd.DataFrame(
        {
            "vehicle_id": ["a"] * 200 + ["b"] * 200,
            "date": list(days) * 2,
            "oem": ["oem_a"] * 200 + ["oem_b"] * 200,
            "model_year": [2022] * 200 + [2023] * 200,
            "active": [True] * 400,
            "s1": [1.0] * 400,
            "s2": [2.0] * 400,
        }
    )
    ev = pd.DataFrame(
        {"vehicle_id": ["a"], "date": [pd.Timestamp("2025-03-01")], "event_type": ["brake"]}
    )
    return veh, ev, sig


def test_clean_frames_pass():
    veh, ev, sig = _frames()
    rep = validate(veh, ev, sig, ["s1", "s2"], today=TODAY)
    assert rep.ok and rep.warnings == [], rep.as_dict()
    assert rep.stats["grid_completeness"] == 1.0 and rep.stats["n_days"] == 200


def test_missing_column_and_bad_type_are_errors():
    veh, ev, sig = _frames()
    rep = validate(veh.drop(columns=["powertrain"]), ev, sig.assign(s1="x"), ["s1"], today=TODAY)
    assert any("missing required column 'powertrain'" in e for e in rep.errors)
    assert any("non-numeric signal columns" in e for e in rep.errors)


def test_duplicate_grain_orphans_and_future_dates_are_errors():
    veh, ev, sig = _frames()
    dup = pd.concat([sig, sig.iloc[:1]])
    orphan_ev = pd.concat([ev, ev.assign(vehicle_id="zzz")])
    future_ev = ev.assign(date=[TODAY + pd.Timedelta(days=3)])
    rep = validate(veh, pd.concat([orphan_ev, future_ev]), dup, ["s1"], today=TODAY)
    msgs = " | ".join(rep.errors)
    assert "duplicate (vehicle_id, date)" in msgs
    assert "not in vehicles" in msgs
    assert "dated in the future" in msgs


def test_unknown_powertrain_and_model_year_range():
    veh, ev, sig = _frames()
    veh = veh.assign(powertrain=["diesel", "ev"], model_year=[1975, 2023])
    rep = validate(veh, ev, sig, ["s1"], today=TODAY)
    assert any("unknown values ['diesel']" in e for e in rep.errors)
    assert any("model_year" in e for e in rep.errors)


def test_short_history_and_grid_gaps_are_warnings():
    veh, ev, sig = _frames()
    short = sig[sig["date"] < "2025-03-01"]
    gappy = short.iloc[:-5]
    rep = validate(veh, ev[ev["date"] < "2025-03-01"], gappy, ["s1"], today=TODAY)
    assert rep.ok
    assert any("days of history" in w for w in rep.warnings)
    assert any("missing from the full grid" in w for w in rep.warnings)


def test_values_on_inactive_days_warn_unless_allowed():
    veh, ev, sig = _frames()
    sig.loc[sig.index[:3], "active"] = False
    rep = validate(veh, ev, sig, ["s1", "s2"], today=TODAY)
    assert any("inactive vehicle-days" in w for w in rep.warnings)
    rep2 = validate(veh, ev, sig, ["s1", "s2"], today=TODAY, inactive_ok=frozenset({"s1", "s2"}))
    assert rep2.warnings == []


def test_synthetic_source_satisfies_the_contract(source):
    rep = validate_source(source)
    assert rep.ok, rep.errors
    assert rep.warnings == [], rep.warnings
    assert rep.stats["n_signals_catalogued"] == 92

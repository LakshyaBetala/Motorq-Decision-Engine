"""The synthetic world must be deterministic, reproduce the sourced real-world rates it
claims to encode, and expose the planted structure in truth.json."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from motorq_de.world import registry as reg
from motorq_de.world.config import load_coverage, load_params
from motorq_de.world.generator import WorldGenerator


def test_dataset_hash_is_deterministic_and_seed_sensitive():
    a = WorldGenerator(profile="small", seed=1).dataset_hash
    b = WorldGenerator(profile="small", seed=1).dataset_hash
    c = WorldGenerator(profile="small", seed=2).dataset_hash
    assert a == b
    assert a != c


def test_generation_is_byte_deterministic(tmp_path: Path):
    g1 = WorldGenerator(profile="small", seed=3)
    g2 = WorldGenerator(profile="small", seed=3)
    d1 = g1.generate(tmp_path / "a")
    d2 = g2.generate(tmp_path / "b")
    s1 = pd.read_parquet(d1 / "signals_daily.parquet")
    s2 = pd.read_parquet(d2 / "signals_daily.parquet")
    pd.testing.assert_frame_equal(s1, s2)
    e1 = pd.read_parquet(d1 / "events.parquet")
    e2 = pd.read_parquet(d2 / "events.parquet")
    pd.testing.assert_frame_equal(e1, e2)


def test_brake_rate_consistent_with_sourced_pad_life(truth: dict):
    # 50k-mile base pad life (30k-70k sourced), mixed duty 30-110 mi/day, heavy urban vans
    # much shorter -> ~0.6-1.6 services per vehicle-year is the plausible band.
    r = truth["achieved_rates"]["brake_service_events_per_vehicle_year"]
    assert 0.6 <= r <= 1.6, r


def test_theft_rate_matches_configured_synthetic_rate(truth: dict):
    p = load_params()
    target = p["theft"]["synthetic_annual_rate"]
    r = truth["achieved_rates"]["theft_events_per_vehicle_year"]
    # small fleet -> wide tolerance, but must be the right order of magnitude
    assert 0.4 * target <= r <= 2.5 * target, (r, target)
    # and the disclosure that it is elevated over NICB must be present
    assert "elevated" in truth["disclosures"]["theft_rate_elevated"].lower()
    assert p["theft"]["real_world_annual_rate"] == pytest.approx(0.0025)


def test_battery_rate_in_modeled_band(truth: dict):
    r = truth["achieved_rates"]["battery_events_per_ev_year"]
    assert 0.1 <= r <= 0.6, r


def test_positive_rate_is_rare_event(source, brake_spec):
    lf = source.training_frame(brake_spec, reg.drivers_for("brake_service_event"))
    assert 0.005 <= lf.positive_rate <= 0.05, lf.positive_rate


def test_truth_lists_planted_structure(truth: dict):
    t = truth["targets"]["brake_service_event"]
    assert "brake_pad_wear_pct" in t["drivers"]
    assert "harsh_brake_count" in t["drivers"]
    assert set(t["leakage"]) == {"service_appointment_scheduled", "service_interval_remaining_days"}
    assert "gps_lat_mean" not in t["drivers"]
    assert "night_park_share" in truth["targets"]["theft_event"]["drivers"]
    assert truth["decoys"]["brake_wear_est_rear_pct"] == "brake_pad_wear_pct"
    assert len(truth["noise"]) >= 5


def test_coverage_gaps_reflect_matrix(truth: dict):
    cov = load_coverage()
    missing = truth["coverage_gaps_for_drivers"]["brake_pad_wear_pct"]
    expected = sorted(o for o in cov.oem_ids() if not cov.coverage(o, "brake_pad_wear_pct").emits)
    assert sorted(missing) == expected
    assert {"oem_c", "oem_g"} <= set(missing)


def test_oems_without_sensor_emit_nothing(source):
    df = source.read_signals(["brake_pad_wear_pct"])
    for oem in ("oem_c", "oem_g", "oem_h"):
        assert df.loc[df.oem == oem, "brake_pad_wear_pct"].isna().all(), oem


def test_ev_only_signals_absent_on_ice(source):
    df = source.read_signals(["soc_min_daily", "oil_life_pct"])
    veh = source.vehicles().set_index("vehicle_id")
    pt = veh.loc[df.vehicle_id, "powertrain"].to_numpy()
    assert df.loc[pt == "ice", "soc_min_daily"].isna().all()
    assert df.loc[pt == "ev", "oil_life_pct"].isna().all()


def test_weekly_signals_have_weekly_gaps(source):
    q = source.quality("air_filter_life_pct")
    assert q.declared_gap_days == 7.0
    assert 5.0 <= q.median_gap_days <= 9.0


def test_params_json_matches_loaded_params(dataset_dir: Path):
    saved = json.loads((dataset_dir / "params.json").read_text(encoding="utf-8"))
    assert saved["population"]["n_vehicles"] == 600
    assert saved["brakes"]["pad_life_miles_base"] == 50000

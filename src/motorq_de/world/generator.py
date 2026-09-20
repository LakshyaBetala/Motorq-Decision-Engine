"""Latent-degradation synthetic fleet generator.

Each vehicle carries hidden wear states that evolve with usage, climate and OEM.
Observable signals are noisy, OEM-dependent views of those states; events are
hazard-driven from the latents. Ground truth is written alongside the data.

Output (under <out_dir>/<dataset_hash>/):
    vehicles.parquet        one row per vehicle
    signals_daily.parquet   one row per vehicle-day, wide, NaN where not emitted
    events.parquet          one row per event
    truth.json              drivers / decoys / noise / leakage per target, gaps, rates
    params.json             the exact parameters used
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from motorq_de.hashing import canonical_json, sha256_hex
from motorq_de.world import registry as reg
from motorq_de.world.config import HERE, CoverageMatrix, load_coverage, load_params

PROFILES: dict[str, dict[str, Any]] = {
    "small": {"population": {"n_vehicles": 600, "n_days": 200}},
    "default": {},
    "full": {"population": {"n_vehicles": 10000, "n_days": 730}},
}

B, T, V = "brake_service_event", "theft_event", "battery_degradation_event"


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


# decoy_id -> f(observed_driver_values, rng, n). Each is a noisy transform of its driver.
DECOY_FN = {
    "brake_pad_wear_rear_pct": lambda d, rng, n: np.clip(d * 0.85 + rng.normal(0, 9, n), 0, 100),
    "brake_wear_est_rear_pct": lambda d, rng, n: np.clip(d + rng.normal(0, 12, n), 0, 100),
    "pad_thickness_proxy_mm": lambda d, rng, n: np.clip(12 - d / 10 + rng.normal(0, 2.5, n), 0, 14),
    "mileage_band": lambda d, rng, n: np.floor(d / 25.0) + rng.integers(-1, 2, n),
    "aggressive_driving_index": lambda d, rng, n: np.clip(d * 10 + rng.normal(0, 25, n), 0, None),
    "driver_score": lambda d, rng, n: np.clip(90 - 4 * d + rng.normal(0, 8, n), 0, 100),
    "brake_dtc_history_30d": lambda d, rng, n: d + rng.poisson(0.3, n),
    "overnight_risk_score": lambda d, rng, n: np.clip(d * 100 + rng.normal(0, 25, n), 0, 100),
    "parking_variability_score": lambda d, rng, n: np.clip(d * 30 + rng.normal(0, 20, n), 0, None),
    "soc_stress_index": lambda d, rng, n: np.clip((100 - d) + rng.normal(0, 25, n), 0, None),
    "fast_charge_flag_30d": lambda d, rng, n: (d > 0.15).astype(float),
    "thermal_stress_index": lambda d, rng, n: np.clip(d - 20 + rng.normal(0, 8, n), 0, None),
    "battery_range_mi": lambda d, rng, n: np.clip(400 - d * 0.6 + rng.normal(0, 15, n), 50, None),
}

if set(DECOY_FN) != set(reg.decoys()):  # every registry decoy must have a generator
    raise RuntimeError(f"DECOY_FN/registry mismatch: {set(DECOY_FN) ^ set(reg.decoys())}")


@dataclass
class Vehicles:
    n: int
    vehicle_id: np.ndarray
    oem: np.ndarray
    model_year: np.ndarray
    segment: np.ndarray
    climate: np.ndarray
    duty: np.ndarray
    is_ev: np.ndarray
    aggressiveness: np.ndarray
    night_park_base: np.ndarray
    dwell_entropy_base: np.ndarray
    region_theft_index: np.ndarray
    urban_share_base: np.ndarray
    dcfc_propensity: np.ndarray
    pad_life_miles: np.ndarray
    ev_range_mi: np.ndarray
    lat: np.ndarray
    lon: np.ndarray
    service_threshold: np.ndarray
    wear_sensor_bias: np.ndarray

    def frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "vehicle_id": self.vehicle_id,
                "oem": self.oem,
                "model_year": self.model_year,
                "segment": self.segment,
                "climate": self.climate,
                "duty": self.duty,
                "powertrain": np.where(self.is_ev, "ev", "ice"),
            }
        )


class WorldGenerator:
    def __init__(
        self,
        params: dict[str, Any] | None = None,
        coverage: CoverageMatrix | None = None,
        seed: int = 42,
        profile: str = "default",
    ):
        self.params = load_params(overrides=PROFILES[profile]) if params is None else params
        self.coverage = coverage or load_coverage()
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.version = self.params["generator_version"]
        self.dataset_hash = self._hash()

    # ------------------------------------------------------------------ identity
    def _hash(self) -> str:
        cov_text = (HERE / "coverage.yaml").read_text(encoding="utf-8")
        reg_text = ",".join(reg.signal_ids())
        return sha256_hex(
            canonical_json(self.params) + cov_text + reg_text + f"|{self.seed}|{self.version}"
        )[:16]

    # ------------------------------------------------------------------ population
    def _sample_vehicles(self) -> Vehicles:
        p = self.params["population"]
        rng = self.rng
        n = int(p["n_vehicles"])
        oem_ids = self.coverage.oem_ids()
        oem = rng.choice(oem_ids, size=n, p=self.coverage.fleet_weights())
        model_year = rng.choice(p["model_years"], size=n, p=p["model_year_weights"])
        ev_share = np.array([self.coverage.oems[o].ev_share for o in oem])
        is_ev = rng.random(n) < ev_share
        ice_segments = [s for s in p["segments"] if s != "ev"]
        segment = rng.choice(ice_segments, size=n, p=[0.25, 0.25, 0.30, 0.20])
        segment = np.where(is_ev, "ev", segment)
        climate = rng.choice(p["climate_zones"], size=n, p=p["climate_weights"])
        duty = rng.choice(p["duty_classes"], size=n, p=p["duty_weights"])

        br = self.params["brakes"]
        pad_life = (
            br["pad_life_miles_base"]
            * np.array([br["pad_life_miles_by_segment"][s] for s in segment])
            * np.array([br["pad_life_miles_by_duty"][d] for d in duty])
            / np.array([br["oem_wear_multiplier"][o] for o in oem])
            / np.array([br["climate_wear_multiplier"][c] for c in climate])
            * rng.lognormal(0, 0.12, n)
        )
        return Vehicles(
            n=n,
            vehicle_id=np.array([f"V{i:06d}" for i in range(n)]),
            oem=oem,
            model_year=model_year.astype(int),
            segment=segment,
            climate=climate,
            duty=duty,
            is_ev=is_ev,
            aggressiveness=rng.lognormal(0, 0.45, n),
            night_park_base=rng.beta(1.6, 3.0, n),
            dwell_entropy_base=rng.gamma(2.0, 0.6, n),
            region_theft_index=rng.lognormal(0, 0.5, n),
            urban_share_base=np.clip(
                np.array([self.params["usage"]["urban_share_by_segment"][s] for s in segment])
                + rng.normal(0, 0.12, n),
                0.05,
                0.95,
            ),
            dcfc_propensity=rng.beta(1.2, 3.0, n),
            pad_life_miles=pad_life,
            ev_range_mi=rng.normal(260, 40, n).clip(150, 400),
            lat=rng.normal(38.0, 4.0, n),
            lon=rng.normal(-95.0, 12.0, n),
            service_threshold=np.clip(
                rng.normal(br["hazard_threshold"], br["threshold_sd_between_vehicles"], n),
                0.78,
                1.08,
            ),
            wear_sensor_bias=rng.normal(0, br["sensor_bias_sd"], n),
        )

    # ------------------------------------------------------------------ main loop
    def generate(self, out_dir: Path) -> Path:
        out = out_dir / self.dataset_hash
        out.mkdir(parents=True, exist_ok=True)
        veh = self._sample_vehicles()
        n = veh.n
        rng = self.rng
        P = self.params
        n_days = int(P["population"]["n_days"])
        start = date.fromisoformat(P["population"]["start_date"])
        usage, br, bat, th = P["usage"], P["brakes"], P["battery"], P["theft"]

        # latent state
        age_years = np.clip(start.year - veh.model_year + 0.5, 0.25, 8)
        brake_wear = rng.uniform(0.02, 0.85, n)
        soh = np.where(
            veh.is_ev,
            1.0
            - bat["soh_loss_per_year_base"]
            * age_years
            * (1 + 0.3 * veh.dcfc_propensity)
            * rng.lognormal(0, 0.15, n),
            np.nan,
        )
        odometer = (
            age_years
            * np.array([usage["miles_per_day_by_duty"][d] for d in veh.duty])
            * 365
            * rng.lognormal(0, 0.2, n)
        )
        engine_hours = odometer / 32.0
        oil_life = rng.uniform(20, 100, n)
        air_filter = rng.uniform(30, 100, n)
        charge_cycles = np.where(veh.is_ev, odometer / veh.ev_range_mi, np.nan)
        dcfc_share_30d = veh.dcfc_propensity.copy()
        miles_ring = np.zeros((7, n))
        pending_service_in = np.full(n, -1)  # days until scheduled brake service, -1 = none
        stolen_days_left = np.zeros(n, dtype=int)
        tire_base = rng.normal(240, 6, (n, 4))

        # coverage masks per signal: emits & model-year eligible & powertrain match
        emits = {}
        miss_rate = {}
        weekly = {}
        for sid in reg.signal_ids():
            sdef = reg.BY_ID[sid]
            e = np.zeros(n, dtype=bool)
            m = np.zeros(n)
            w = np.zeros(n, dtype=bool)
            for o in self.coverage.oem_ids():
                cov = self.coverage.coverage(o, sid)
                idx = veh.oem == o
                e[idx] = cov.emits & (veh.model_year[idx] >= cov.from_model_year)
                m[idx] = cov.missing_rate
                w[idx] = cov.frequency == "weekly"
            if sdef.powertrain == "ev":
                e &= veh.is_ev
            elif sdef.powertrain == "ice":
                e &= ~veh.is_ev
            emits[sid], miss_rate[sid], weekly[sid] = e, m, w
        weekly_dow = rng.integers(0, 7, n)

        events: list[dict[str, Any]] = []
        writer: pq.ParquetWriter | None = None
        chunk: dict[str, list[np.ndarray]] = {}
        chunk_days = 30
        pad_life = veh.pad_life_miles
        hb_mult = br["harsh_brake_wear_multiplier"]
        seg_theft = np.array([th["segment_multiplier"][s] for s in veh.segment])
        is_hot = veh.climate == "hot"
        clim_amp = np.where(veh.climate == "hot", 14, np.where(veh.climate == "cold", 16, 12))
        clim_mean = np.where(veh.climate == "hot", 24, np.where(veh.climate == "cold", 6, 15))
        miles_mean = np.array([usage["miles_per_day_by_duty"][d] for d in veh.duty])
        sigma = np.sqrt(np.log(1 + usage["miles_per_day_cv"] ** 2))
        # normalise theft shape multipliers so the fleet-mean daily hazard equals the configured rate
        _shape = (
            seg_theft
            * veh.region_theft_index
            * (0.3 + 1.4 * veh.night_park_base)
            * (0.5 + 0.5 * np.clip(veh.dwell_entropy_base / 2.0, 0, 2))
        )
        theft_scale = (th["synthetic_annual_rate"] / 365.0) / float(_shape.mean())

        for d in range(n_days):
            today = start + timedelta(days=d)
            dow = today.weekday()
            weekend = dow >= 5
            season = np.sin(2 * np.pi * (today.timetuple().tm_yday - 100) / 365.0)
            ambient_max = clim_mean + clim_amp * season + rng.normal(0, 3, n)
            ambient_min = ambient_max - rng.uniform(6, 12, n)

            active = stolen_days_left == 0
            # -------------------------------------------------- usage
            mean_today = miles_mean * (usage["weekend_factor"] if weekend else 1.0)
            miles = rng.lognormal(np.log(mean_today) - sigma**2 / 2, sigma) * active
            miles = np.where(rng.random(n) < (0.35 if weekend else 0.06), 0.0, miles)
            urban = np.clip(veh.urban_share_base + rng.normal(0, 0.05, n), 0, 1)
            trips = rng.poisson(miles * usage["trips_per_100mi"] / 100.0)
            avg_speed = np.where(miles > 0, 58 - 32 * urban + rng.normal(0, 4, n), np.nan)
            harsh_brake = rng.poisson(veh.aggressiveness * (miles / 30.0) * (0.5 + urban))
            harsh_accel = rng.poisson(veh.aggressiveness * (miles / 30.0) * 0.6)
            harsh_corner = rng.poisson(veh.aggressiveness * (miles / 30.0) * 0.3)
            speeding = rng.poisson(veh.aggressiveness * (miles / 30.0) * (1.2 - urban))
            idle_min = rng.gamma(2.0, 8.0 + 20 * urban, n) * (miles > 0)
            miles_ring[d % 7] = miles
            odometer += miles
            engine_hours += np.where(
                veh.is_ev, 0, miles / np.clip(np.nan_to_num(avg_speed, nan=35), 10, 80)
            )
            oil_life = np.clip(oil_life - miles / 60.0, 0, 100)
            oil_life = np.where(oil_life <= 2, 100.0, oil_life)
            air_filter = np.clip(air_filter - miles / 300.0, 0, 100)
            air_filter = np.where(air_filter <= 2, 100.0, air_filter)

            # -------------------------------------------------- brakes
            wear_inc = (miles / pad_life) * (
                1 + hb_mult * harsh_brake * 30.0 / np.maximum(miles, 1.0)
            )
            brake_wear = np.clip(brake_wear + wear_inc, 0, 1.2)
            hazard = _sigmoid(br["hazard_steepness"] * (brake_wear - veh.service_threshold))
            newly_flagged = (pending_service_in < 0) & (rng.random(n) < hazard) & active
            pending_service_in = np.where(
                newly_flagged,
                rng.integers(0, br["service_lag_days_max"] + 1, n),
                pending_service_in,
            )
            service_today = pending_service_in == 0
            for i in np.flatnonzero(service_today):
                events.append(
                    {
                        "vehicle_id": veh.vehicle_id[i],
                        "date": today,
                        "event_type": B,
                        "detail": float(brake_wear[i]),
                    }
                )
            brake_wear = np.where(service_today, br["reset_wear_after_service"], brake_wear)
            pending_service_in = np.where(
                pending_service_in > 0,
                pending_service_in - 1,
                np.where(service_today, -1, pending_service_in),
            )
            dtc_brake = rng.poisson(0.01 + 0.6 * np.clip(brake_wear - 0.8, 0, None) / 0.2)

            # -------------------------------------------------- battery (EV)
            dcfc_today = (rng.random(n) < veh.dcfc_propensity) & (miles > 0)
            dcfc_share_30d = np.clip(dcfc_share_30d * (29 / 30) + dcfc_today / 30.0, 0, 1)
            soc_min = np.clip(95 - (miles / veh.ev_range_mi) * 100 - rng.uniform(0, 15, n), 3, 95)
            charge_cycles = charge_cycles + np.where(veh.is_ev, miles / veh.ev_range_mi, 0)
            batt_temp = (
                ambient_max + 6 + 14 * dcfc_today + 3 * (miles / 100.0) + rng.normal(0, 2, n)
            )
            daily_loss = (
                bat["soh_loss_per_year_base"]
                * (1 + (bat["soh_loss_dcfc_multiplier_at_full_share"] - 1) * dcfc_share_30d)
                + bat["soh_loss_hot_climate_add_per_year"] * is_hot
            ) / 365.0
            soh = np.where(veh.is_ev, soh - daily_loss * rng.lognormal(0, 0.3, n), np.nan)
            b_hazard = (
                bat["event_base_daily_hazard"]
                * (
                    1
                    + 40
                    * _sigmoid(
                        bat["event_hazard_steepness"]
                        * (bat["event_soh_threshold"] - np.nan_to_num(soh, nan=1.0))
                    )
                )
                * (1 + 1.5 * dcfc_share_30d)
                * (1 + 0.5 * is_hot)
                * (1 + 2.0 * np.clip((batt_temp - 45) / 10, 0, None))
            )
            b_event = veh.is_ev & active & (rng.random(n) < b_hazard)
            for i in np.flatnonzero(b_event):
                events.append(
                    {
                        "vehicle_id": veh.vehicle_id[i],
                        "date": today,
                        "event_type": V,
                        "detail": float(soh[i]),
                    }
                )
            soh = np.where(b_event, np.minimum(soh + 0.06, 1.0), soh)
            dtc_batt = rng.poisson(
                np.where(
                    veh.is_ev,
                    0.01 + 0.5 * np.clip(0.88 - np.nan_to_num(soh, nan=1.0), 0, None) / 0.1,
                    0,
                )
            )

            # -------------------------------------------------- theft
            night_park = np.clip(veh.night_park_base + rng.normal(0, 0.08, n), 0, 1)
            dwell_ent = np.clip(veh.dwell_entropy_base + rng.normal(0, 0.25, n), 0, None)
            region_idx = veh.region_theft_index * rng.lognormal(0, 0.05, n)
            t_hazard = (
                theft_scale
                * seg_theft
                * region_idx
                * (0.3 + 1.4 * night_park)
                * (0.5 + 0.5 * np.clip(dwell_ent / 2.0, 0, 2))
            )
            t_event = active & (rng.random(n) < t_hazard)
            for i in np.flatnonzero(t_event):
                u = rng.random()
                if u < th["recovery_same_day"]:
                    rec = 0
                elif u < th["recovery_within_2_days"]:
                    rec = int(rng.integers(1, 3))
                elif u < th["recovery_probability"]:
                    rec = int(rng.integers(3, 31))
                else:
                    rec = 60
                events.append(
                    {
                        "vehicle_id": veh.vehicle_id[i],
                        "date": today,
                        "event_type": T,
                        "detail": float(rec),
                    }
                )
                stolen_days_left[i] = rec
            stolen_days_left = np.where(stolen_days_left > 0, stolen_days_left - 1, 0)

            # -------------------------------------------------- observable signals
            ev = veh.is_ev
            nan = np.full(n, np.nan)
            tp = tire_base + rng.normal(0, 4, (n, 4)) - 2 * (ambient_max < 5)[:, None]
            weekly_delta = miles_ring.sum(axis=0)
            row: dict[str, np.ndarray] = {
                "gps_lat_mean": veh.lat + rng.normal(0, 0.05, n),
                "gps_lon_mean": veh.lon + rng.normal(0, 0.05, n),
                "gps_fix_count": (trips * 40 + rng.poisson(20, n)).astype(float),
                "trip_count": trips.astype(float),
                "trip_distance_mi": miles,
                "trip_duration_min": np.where(
                    miles > 0, miles / np.nan_to_num(avg_speed, nan=35) * 60 + idle_min * 0.3, 0
                ),
                "avg_speed_mph": avg_speed,
                "max_speed_mph": np.where(
                    miles > 0,
                    np.nan_to_num(avg_speed, nan=35) + 20 + speeding * 3 + rng.normal(0, 5, n),
                    np.nan,
                ),
                "night_park_share": night_park,
                "dwell_location_entropy": dwell_ent,
                "geofence_exits": rng.poisson(0.2 + 2 * night_park, n).astype(float),
                "ignition_off_duration_hr": np.clip(
                    10 + 6 * night_park + rng.normal(0, 2, n), 1, 24
                ),
                "first_ignition_hour": np.clip(rng.normal(7, 1.5, n), 3, 12),
                "last_ignition_hour": np.clip(rng.normal(18, 2, n), 12, 23.9),
                "idle_minutes": idle_min,
                "highway_share": 1 - urban,
                "urban_share": urban,
                "region_theft_index": region_idx,
                "odometer_mi": odometer,
                "odometer_delta_mi": np.where(weekly["odometer_delta_mi"], weekly_delta, miles),
                "brake_pad_wear_pct": np.clip(
                    (brake_wear + veh.wear_sensor_bias) * 100 + rng.normal(0, 3, n), 0, 100
                ),
                "brake_pad_wear_rear_pct": nan,
                "brake_fluid_level_low": (rng.random(n) < 0.005 + 0.02 * (brake_wear > 0.9)).astype(
                    float
                ),
                "oil_life_pct": np.where(ev, nan, oil_life),
                "engine_hours": np.where(ev, nan, engine_hours),
                "engine_hours_delta": np.where(
                    ev, nan, miles / np.clip(np.nan_to_num(avg_speed, nan=35), 10, 80)
                ),
                "coolant_temp_max_c": np.where(
                    ev, nan, 88 + rng.normal(0, 4, n) + 0.2 * ambient_max
                ),
                "dtc_count_active": (dtc_brake + dtc_batt + rng.poisson(0.05, n)).astype(float),
                "dtc_brake_family": dtc_brake.astype(float),
                "dtc_powertrain_family": rng.poisson(0.03, n).astype(float),
                "dtc_battery_family": np.where(ev, dtc_batt.astype(float), nan),
                "dtc_emissions_family": np.where(ev, nan, rng.poisson(0.02, n).astype(float)),
                "tire_pressure_fl_kpa": tp[:, 0],
                "tire_pressure_fr_kpa": tp[:, 1],
                "tire_pressure_rl_kpa": tp[:, 2],
                "tire_pressure_rr_kpa": tp[:, 3],
                "tire_pressure_min_kpa": tp.min(axis=1),
                "tpms_warning": (tp.min(axis=1) < 220).astype(float),
                "air_filter_life_pct": np.where(ev, nan, air_filter),
                "washer_fluid_low": (rng.random(n) < 0.02).astype(float),
                "battery_12v_voltage": rng.normal(12.6, 0.15, n) - 0.01 * (ambient_min < 0),
                "harsh_brake_count": harsh_brake.astype(float),
                "harsh_accel_count": harsh_accel.astype(float),
                "harsh_corner_count": harsh_corner.astype(float),
                "speeding_events": speeding.astype(float),
                "seatbelt_unbuckled_events": rng.poisson(0.3 * trips / 5, n).astype(float),
                "fcw_events": rng.poisson(0.05 * veh.aggressiveness * miles / 30, n).astype(float),
                "ldw_events": rng.poisson(0.08 * miles / 30, n).astype(float),
                "crash_detected": (rng.random(n) < 0.00005 * veh.aggressiveness).astype(float),
                "driver_score": np.clip(
                    90 - 4 * harsh_brake - 2 * harsh_accel - speeding + rng.normal(0, 8, n), 0, 100
                ),
                "max_decel_g": np.where(
                    miles > 0, 0.25 + 0.08 * (harsh_brake > 0) + rng.normal(0, 0.05, n), np.nan
                ),
                "phone_usage_proxy": rng.poisson(1.0, n).astype(float),
                "fatigue_events": rng.poisson(0.05, n).astype(float),
                "fuel_level_pct": np.where(ev, nan, rng.uniform(15, 95, n)),
                "fuel_consumed_gal": np.where(ev, nan, miles / rng.normal(19, 3, n).clip(8, 40)),
                "fuel_economy_mpg": np.where(
                    ev | (miles == 0), nan, rng.normal(19, 3, n).clip(8, 40)
                ),
                "fuel_theft_flag": np.where(ev, nan, (rng.random(n) < 0.002).astype(float)),
                "co2_kg": np.where(ev, miles * 0.12, miles / 19.0 * 8.9),
                "soc_min_daily": np.where(ev, soc_min, nan),
                "soc_max_daily": np.where(
                    ev, np.clip(soc_min + rng.uniform(20, 60, n), 0, 100), nan
                ),
                "soc_mean": np.where(ev, np.clip(soc_min + rng.uniform(10, 30, n), 0, 100), nan),
                "dc_fast_charge_share": np.where(ev, dcfc_share_30d, nan),
                "charge_cycles": np.where(ev, charge_cycles, nan),
                "charge_energy_kwh": np.where(
                    ev, np.maximum(miles * 0.32 + rng.normal(0, 2, n), 0), nan
                ),
                "battery_temp_max_c": np.where(ev, batt_temp, nan),
                "battery_range_mi": np.where(
                    ev, veh.ev_range_mi * np.nan_to_num(soh, nan=1.0) + rng.normal(0, 15, n), nan
                ),
                "cabin_temp_max_c": np.where(ev, ambient_max + 8 + rng.normal(0, 3, n), nan),
                "regen_energy_kwh": np.where(
                    ev, np.maximum(miles * 0.06 * (0.5 + urban) + rng.normal(0, 0.5, n), 0), nan
                ),
                "charge_events_count": np.where(
                    ev, rng.poisson(0.6 + dcfc_today, n).astype(float), nan
                ),
                "ambient_temp_max_c": ambient_max,
                "ambient_temp_min_c": ambient_min,
                "precipitation_mm": rng.gamma(0.4, 6, n) * (rng.random(n) < 0.3),
                "altitude_mean_m": rng.normal(300, 50, n),
                "day_of_week": np.full(n, float(dow)),
                "is_holiday": np.full(n, float(today.month == 12 and today.day == 25)),
                # decoys are computed after coverage masking from the OBSERVED driver (see below)
                "brake_wear_est_rear_pct": nan,
                "pad_thickness_proxy_mm": nan,
                "mileage_band": nan,
                "aggressive_driving_index": nan,
                "brake_dtc_history_30d": nan,
                "overnight_risk_score": nan,
                "parking_variability_score": nan,
                "soc_stress_index": nan,
                "fast_charge_flag_30d": nan,
                "thermal_stress_index": nan,
                "infotainment_reboots": rng.poisson(0.1, n).astype(float),
                "bluetooth_pairings": rng.poisson(0.3, n).astype(float),
                "wiper_activations": rng.poisson(3, n).astype(float),
                "door_open_count": rng.poisson(8, n).astype(float),
                "radio_volume_mean": rng.normal(12, 3, n),
                # leakage: DMS knows the appointment; interval computed post-hoc (see _add_retro_leak)
                "service_appointment_scheduled": (pending_service_in >= 0).astype(float),
                "service_interval_remaining_days": nan,
            }

            # -------------------------------------------------- coverage, missingness, theft blackout
            for sid, vals in row.items():
                vals = np.asarray(vals, dtype=np.float32)
                keep = emits[sid] & active
                keep &= rng.random(n) >= miss_rate[sid]
                w = weekly[sid]
                keep &= ~w | (weekly_dow == dow)
                row[sid] = np.where(keep, vals, np.nan).astype(np.float32)
            # decoys: noisy functions of the OBSERVED driver, NaN wherever the driver is NaN, so
            # they are correlated with the target but carry no information the driver lacks
            for sid, fn in DECOY_FN.items():
                drv = row[reg.BY_ID[sid].decoy_of]
                vals = fn(drv.astype(np.float64), rng, n).astype(np.float32)
                keep = np.isfinite(drv) & (rng.random(n) >= miss_rate[sid]) & emits[sid]
                row[sid] = np.where(keep, vals, np.nan).astype(np.float32)
            # theft blackout: everything dark except a few GPS fixes while stolen (if tracked)
            stolen = ~active
            if stolen.any():
                row["gps_fix_count"] = np.where(
                    stolen & emits["gps_fix_count"],
                    rng.poisson(3, n).astype(np.float32),
                    row["gps_fix_count"],
                )

            chunk.setdefault("vehicle_id", []).append(veh.vehicle_id)
            chunk.setdefault("date", []).append(np.full(n, np.datetime64(today, "D")))
            chunk.setdefault("oem", []).append(veh.oem)
            chunk.setdefault("model_year", []).append(veh.model_year.astype(np.int16))
            chunk.setdefault("active", []).append(active.copy())
            for sid in reg.signal_ids():
                chunk.setdefault(sid, []).append(row[sid])

            if (d + 1) % chunk_days == 0 or d == n_days - 1:
                table = pa.table({k: np.concatenate(v) for k, v in chunk.items()})
                if writer is None:
                    writer = pq.ParquetWriter(
                        out / "signals_daily.parquet", table.schema, compression="zstd"
                    )
                writer.write_table(table)
                chunk = {}
        assert writer is not None
        writer.close()

        veh_df = veh.frame()
        veh_df.to_parquet(out / "vehicles.parquet", index=False)
        ev_df = pd.DataFrame(events, columns=["vehicle_id", "date", "event_type", "detail"])
        ev_df["date"] = pd.to_datetime(ev_df["date"])
        ev_df.to_parquet(out / "events.parquet", index=False)
        self._add_retro_leak(out, ev_df)
        self._write_truth(out, veh_df, ev_df, n_days)
        (out / "params.json").write_text(canonical_json(self.params), encoding="utf-8")
        return out

    # ------------------------------------------------------------------ post-processing
    def _add_retro_leak(self, out: Path, ev_df: pd.DataFrame) -> None:
        """`service_interval_remaining_days` = days to the next brake service, as a dealer
        system would export it retroactively. This is a deliberate leak the quality engine
        must catch."""
        sig = pd.read_parquet(out / "signals_daily.parquet")
        brake = ev_df[ev_df.event_type == B][["vehicle_id", "date"]].rename(
            columns={"date": "next_service"}
        )
        brake = brake.sort_values("next_service")
        sig = sig.sort_values("date")
        sig["date"] = pd.to_datetime(sig["date"])
        merged = pd.merge_asof(
            sig[["vehicle_id", "date"]],
            brake,
            left_on="date",
            right_on="next_service",
            by="vehicle_id",
            direction="forward",
        )
        days = (merged["next_service"] - merged["date"]).dt.days.astype("float32")
        sig["service_interval_remaining_days"] = days.clip(upper=365).to_numpy(dtype=np.float32)
        sig = sig.sort_values(["date", "vehicle_id"]).reset_index(drop=True)
        sig.to_parquet(out / "signals_daily.parquet", index=False, compression="zstd")

    def _write_truth(
        self, out: Path, veh_df: pd.DataFrame, ev_df: pd.DataFrame, n_days: int
    ) -> None:
        veh_years = len(veh_df) * n_days / 365.0
        ev_years = int(veh_df.powertrain.eq("ev").sum()) * n_days / 365.0
        counts = ev_df.event_type.value_counts().to_dict()
        gaps = {}
        for target in (B, T, V):
            for sid in reg.drivers_for(target):
                missing = [
                    o for o in self.coverage.oem_ids() if not self.coverage.coverage(o, sid).emits
                ]
                if missing:
                    gaps[sid] = missing
        truth = {
            "generator_version": self.version,
            "dataset_hash": self.dataset_hash,
            "seed": self.seed,
            "targets": {
                t: {
                    "drivers": reg.drivers_for(t),
                    "context": reg.context_for(t),
                    "leakage": reg.leakage_for(t),
                }
                for t in (B, T, V)
            },
            "decoys": reg.decoys(),
            "noise": reg.noise_signals(),
            "coverage_gaps_for_drivers": gaps,
            "achieved_rates": {
                "brake_service_events_per_vehicle_year": counts.get(B, 0) / veh_years,
                "theft_events_per_vehicle_year": counts.get(T, 0) / veh_years,
                "battery_events_per_ev_year": (counts.get(V, 0) / ev_years) if ev_years else None,
            },
            "disclosures": {
                "theft_rate_elevated": "Synthetic theft rate is elevated ~12x over NICB real-world rate for learnability; see params.yaml.",
                "battery_event_modeled": "Battery service event is a modeled construct calibrated to Geotab degradation rates; not a real-world event rate.",
            },
        }
        (out / "truth.json").write_text(json.dumps(truth, indent=2), encoding="utf-8")

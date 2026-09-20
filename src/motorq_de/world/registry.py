"""Signal registry for the synthetic world.

Signals follow Motorq's public taxonomy (location/trips, health, driver behavior,
fuel/energy, plus context). Each signal declares its *role* in the ground truth:

- driver:   causally informative for one or more targets
- context:  informative only via interaction (kept honest: may or may not survive ablation)
- decoy:    noisy copy of a driver — correlated but carries no incremental information
- noise:    pure noise
- leakage:  computed with post-event knowledge; must be caught by the quality engine

The registry is the single source of truth for `truth.json`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from motorq_de.schemas import SignalMeta

Role = Literal["driver", "context", "decoy", "noise", "leakage", "behaviour"]


@dataclass(frozen=True)
class SignalDef:
    signal_id: str
    category: str
    unit: str
    description: str
    frequency: Literal["realtime", "daily", "weekly"] = "daily"
    powertrain: Literal["any", "ev", "ice"] = "any"
    role: Role = "behaviour"
    drives: tuple[str, ...] = field(default_factory=tuple)
    decoy_of: str | None = None
    leaks: str | None = None

    def meta(self) -> SignalMeta:
        return SignalMeta(
            signal_id=self.signal_id,
            category=self.category,  # type: ignore[arg-type]
            unit=self.unit,
            description=self.description,
            declared_frequency=self.frequency,
            powertrain=self.powertrain,
            vss=VSS.get(self.signal_id),
        )


# COVESA Vehicle Signal Specification (VSS) source signal for each daily aggregate. Paths were
# verified against the spec sources (github.com/COVESA/vehicle_signal_specification, master,
# 2026-09). A daily value is an aggregate (mean/max/min/delta/count) of the VSS signal; units
# differ where noted (VSS uses m, km/h, seconds). Signals without a VSS counterpart (behaviour
# event counts, DTC families, decoys, noise) are absent: they are Motorq-derived.
VSS: dict[str, str] = {
    "gps_lat_mean": "Vehicle.CurrentLocation.Latitude",
    "gps_lon_mean": "Vehicle.CurrentLocation.Longitude",
    "gps_fix_count": "Vehicle.CurrentLocation.Timestamp",
    "trip_distance_mi": "Vehicle.TraveledDistance",  # daily delta; VSS unit m
    "avg_speed_mph": "Vehicle.Speed",  # VSS unit km/h
    "max_speed_mph": "Vehicle.Speed",
    "altitude_mean_m": "Vehicle.CurrentLocation.Altitude",
    "odometer_mi": "Vehicle.TraveledDistance",
    "odometer_delta_mi": "Vehicle.TraveledDistance",
    "brake_pad_wear_pct": "Vehicle.Chassis.Axle.Row1.Wheel.Left.Brake.PadWear",
    "brake_pad_wear_rear_pct": "Vehicle.Chassis.Axle.Row2.Wheel.Left.Brake.PadWear",
    "brake_fluid_level_low": "Vehicle.Chassis.Axle.Row1.Wheel.Left.Brake.IsFluidLevelLow",
    "oil_life_pct": "Vehicle.Powertrain.CombustionEngine.EngineOil.LifeRemaining",  # VSS: s
    "engine_hours": "Vehicle.Powertrain.CombustionEngine.EngineHours",
    "engine_hours_delta": "Vehicle.Powertrain.CombustionEngine.EngineHours",
    "coolant_temp_max_c": "Vehicle.Powertrain.CombustionEngine.EngineCoolant.Temperature",
    "dtc_count_active": "Vehicle.Diagnostics.DTCCount",
    "dtc_brake_family": "Vehicle.Diagnostics.DTCList",  # SAE J2012 family prefix
    "dtc_powertrain_family": "Vehicle.Diagnostics.DTCList",
    "dtc_battery_family": "Vehicle.Diagnostics.DTCList",
    "dtc_emissions_family": "Vehicle.Diagnostics.DTCList",
    "tire_pressure_fl_kpa": "Vehicle.Chassis.Axle.Row1.Wheel.Left.Tire.Pressure",
    "tire_pressure_fr_kpa": "Vehicle.Chassis.Axle.Row1.Wheel.Right.Tire.Pressure",
    "tire_pressure_rl_kpa": "Vehicle.Chassis.Axle.Row2.Wheel.Left.Tire.Pressure",
    "tire_pressure_rr_kpa": "Vehicle.Chassis.Axle.Row2.Wheel.Right.Tire.Pressure",
    "tire_pressure_min_kpa": "Vehicle.Chassis.Axle.Row1.Wheel.Left.Tire.Pressure",  # min of 4
    "tpms_warning": "Vehicle.Chassis.Axle.Row1.Wheel.Left.Tire.IsPressureLow",  # any of 4
    "washer_fluid_low": "Vehicle.Body.Windshield.Front.WasherFluid.IsLevelLow",
    "battery_12v_voltage": "Vehicle.LowVoltageBattery.CurrentVoltage",
    "seatbelt_unbuckled_events": "Vehicle.Cabin.Seat.Row1.DriverSide.IsBelted",
    "fuel_level_pct": "Vehicle.Powertrain.FuelSystem.RelativeLevel",
    "soc_min_daily": "Vehicle.Powertrain.TractionBattery.StateOfCharge.Current",
    "soc_max_daily": "Vehicle.Powertrain.TractionBattery.StateOfCharge.Current",
    "soc_mean": "Vehicle.Powertrain.TractionBattery.StateOfCharge.Current",
    "charge_cycles": "Vehicle.Powertrain.TractionBattery.Charging.IsCharging",  # transitions
    "charge_events_count": "Vehicle.Powertrain.TractionBattery.Charging.IsCharging",
    "charge_energy_kwh": "Vehicle.Powertrain.TractionBattery.AccumulatedChargedEnergy",  # delta
    "dc_fast_charge_share": "Vehicle.Powertrain.TractionBattery.Charging.ChargeCurrent.DC",
    "battery_temp_max_c": "Vehicle.Powertrain.TractionBattery.Temperature.Max",
    "battery_range_mi": "Vehicle.Powertrain.TractionBattery.Range",  # VSS unit m
    "cabin_temp_max_c": "Vehicle.Cabin.HVAC.AmbientAirTemperature",
    "ambient_temp_max_c": "Vehicle.Exterior.AirTemperature",
    "ambient_temp_min_c": "Vehicle.Exterior.AirTemperature",
    "precipitation_mm": "Vehicle.Exterior.PrecipitationIntensity",  # mm/h integrated
    # the two leakage signals are VSS *service scheduling* state - known only once service is
    # planned, which is exactly why the quality engine must drop them
    "service_appointment_scheduled": "Vehicle.Service.IsServiceDue",
    "service_interval_remaining_days": "Vehicle.Service.TimeToService",  # VSS unit s
}


B, T, V = "brake_service_event", "theft_event", "battery_degradation_event"

SIGNALS: list[SignalDef] = [
    # ------------------------------------------------------------ location / trips
    SignalDef("gps_lat_mean", "location_trips", "deg", "Mean latitude of fixes", "realtime"),
    SignalDef("gps_lon_mean", "location_trips", "deg", "Mean longitude of fixes", "realtime"),
    SignalDef("gps_fix_count", "location_trips", "count", "GPS fixes received", "realtime"),
    SignalDef("trip_count", "location_trips", "count", "Trips completed", "realtime"),
    SignalDef(
        "trip_distance_mi",
        "location_trips",
        "mi",
        "Distance driven",
        "realtime",
        role="driver",
        drives=(B,),
    ),
    SignalDef("trip_duration_min", "location_trips", "min", "Driving time", "realtime"),
    SignalDef(
        "avg_speed_mph",
        "location_trips",
        "mph",
        "Average moving speed",
        "realtime",
        role="context",
        drives=(B,),
    ),
    SignalDef("max_speed_mph", "location_trips", "mph", "Max speed", "realtime"),
    SignalDef(
        "night_park_share",
        "location_trips",
        "ratio",
        "Share of overnight hours parked outside depot",
        "realtime",
        role="driver",
        drives=(T,),
    ),
    SignalDef(
        "dwell_location_entropy",
        "location_trips",
        "nats",
        "Entropy of parking locations (7d)",
        "realtime",
        role="driver",
        drives=(T,),
    ),
    SignalDef("geofence_exits", "location_trips", "count", "Geofence exit events", "realtime"),
    SignalDef(
        "ignition_off_duration_hr",
        "location_trips",
        "hr",
        "Longest ignition-off dwell",
        "realtime",
        role="context",
        drives=(T,),
    ),
    SignalDef(
        "first_ignition_hour", "location_trips", "hour", "Hour of first ignition", "realtime"
    ),
    SignalDef(
        "last_ignition_hour", "location_trips", "hour", "Hour of last ignition off", "realtime"
    ),
    SignalDef("idle_minutes", "location_trips", "min", "Engine-on idle time", "realtime"),
    SignalDef("highway_share", "location_trips", "ratio", "Share of miles on highway", "realtime"),
    SignalDef(
        "urban_share",
        "location_trips",
        "ratio",
        "Share of miles urban",
        "realtime",
        role="context",
        drives=(B,),
    ),
    SignalDef(
        "region_theft_index",
        "location_trips",
        "index",
        "Regional theft index of overnight location",
        "daily",
        role="driver",
        drives=(T,),
    ),
    # ------------------------------------------------------------ health
    SignalDef("odometer_mi", "health", "mi", "Verified odometer", "daily"),
    SignalDef(
        "odometer_delta_mi",
        "health",
        "mi",
        "Odometer change since last read",
        "daily",
        role="driver",
        drives=(B,),
    ),
    SignalDef(
        "brake_pad_wear_pct",
        "health",
        "pct",
        "Front brake pad wear (OEM-estimated)",
        "daily",
        role="driver",
        drives=(B,),
    ),
    SignalDef(
        "brake_pad_wear_rear_pct",
        "health",
        "pct",
        "Rear brake pad wear (OEM-estimated)",
        "daily",
        role="decoy",
        decoy_of="brake_pad_wear_pct",
    ),
    SignalDef("brake_fluid_level_low", "health", "flag", "Brake fluid low warning", "daily"),
    SignalDef("oil_life_pct", "health", "pct", "Remaining oil life", "daily", powertrain="ice"),
    SignalDef("engine_hours", "health", "hr", "Cumulative engine hours", "daily", powertrain="ice"),
    SignalDef(
        "engine_hours_delta", "health", "hr", "Engine hours today", "daily", powertrain="ice"
    ),
    SignalDef(
        "coolant_temp_max_c", "health", "C", "Max coolant temperature", "realtime", powertrain="ice"
    ),
    SignalDef("dtc_count_active", "health", "count", "Active diagnostic trouble codes", "realtime"),
    SignalDef(
        "dtc_brake_family",
        "health",
        "count",
        "Active DTCs in brake/ABS family",
        "realtime",
        role="driver",
        drives=(B,),
    ),
    SignalDef(
        "dtc_powertrain_family", "health", "count", "Active DTCs in powertrain family", "realtime"
    ),
    SignalDef(
        "dtc_battery_family",
        "health",
        "count",
        "Active DTCs in HV battery family",
        "realtime",
        powertrain="ev",
        role="driver",
        drives=(V,),
    ),
    SignalDef(
        "dtc_emissions_family",
        "health",
        "count",
        "Active DTCs in emissions family",
        "realtime",
        powertrain="ice",
    ),
    SignalDef("tire_pressure_fl_kpa", "health", "kPa", "Tire pressure FL", "realtime"),
    SignalDef("tire_pressure_fr_kpa", "health", "kPa", "Tire pressure FR", "realtime"),
    SignalDef("tire_pressure_rl_kpa", "health", "kPa", "Tire pressure RL", "realtime"),
    SignalDef("tire_pressure_rr_kpa", "health", "kPa", "Tire pressure RR", "realtime"),
    SignalDef(
        "tire_pressure_min_kpa", "health", "kPa", "Min tire pressure across wheels", "realtime"
    ),
    SignalDef("tpms_warning", "health", "flag", "TPMS warning active", "realtime"),
    SignalDef(
        "air_filter_life_pct",
        "health",
        "pct",
        "Remaining air filter life",
        "weekly",
        powertrain="ice",
    ),
    SignalDef("washer_fluid_low", "health", "flag", "Washer fluid low", "daily"),
    SignalDef("battery_12v_voltage", "health", "V", "12V battery voltage at start", "daily"),
    # ------------------------------------------------------------ driver behaviour
    SignalDef(
        "harsh_brake_count",
        "driver_behavior",
        "count",
        "Harsh braking events",
        "realtime",
        role="driver",
        drives=(B,),
    ),
    SignalDef(
        "harsh_accel_count", "driver_behavior", "count", "Harsh acceleration events", "realtime"
    ),
    SignalDef(
        "harsh_corner_count", "driver_behavior", "count", "Harsh cornering events", "realtime"
    ),
    SignalDef("speeding_events", "driver_behavior", "count", "Speeding events", "realtime"),
    SignalDef(
        "seatbelt_unbuckled_events",
        "driver_behavior",
        "count",
        "Seatbelt unbuckled while moving",
        "realtime",
    ),
    SignalDef("fcw_events", "driver_behavior", "count", "Forward collision warnings", "realtime"),
    SignalDef("ldw_events", "driver_behavior", "count", "Lane departure warnings", "realtime"),
    SignalDef("crash_detected", "driver_behavior", "flag", "Crash detected", "realtime"),
    SignalDef(
        "driver_score",
        "driver_behavior",
        "score",
        "Composite driver score",
        "daily",
        role="decoy",
        decoy_of="harsh_brake_count",
    ),
    SignalDef("max_decel_g", "driver_behavior", "g", "Max deceleration", "realtime"),
    SignalDef(
        "phone_usage_proxy",
        "driver_behavior",
        "count",
        "Phone usage proxy events",
        "realtime",
        role="noise",
    ),
    SignalDef(
        "fatigue_events", "driver_behavior", "count", "Fatigue alerts", "realtime", role="noise"
    ),
    # ------------------------------------------------------------ fuel / energy
    SignalDef("fuel_level_pct", "fuel_energy", "pct", "Fuel level", "realtime", powertrain="ice"),
    SignalDef(
        "fuel_consumed_gal", "fuel_energy", "gal", "Fuel consumed", "daily", powertrain="ice"
    ),
    SignalDef("fuel_economy_mpg", "fuel_energy", "mpg", "Fuel economy", "daily", powertrain="ice"),
    SignalDef(
        "fuel_theft_flag",
        "fuel_energy",
        "flag",
        "Fuel level drop while parked",
        "daily",
        powertrain="ice",
    ),
    SignalDef("co2_kg", "fuel_energy", "kg", "CO2 emitted", "daily"),
    SignalDef(
        "soc_min_daily",
        "fuel_energy",
        "pct",
        "Minimum state of charge",
        "realtime",
        powertrain="ev",
        role="driver",
        drives=(V,),
    ),
    SignalDef(
        "soc_max_daily",
        "fuel_energy",
        "pct",
        "Maximum state of charge",
        "realtime",
        powertrain="ev",
    ),
    SignalDef(
        "soc_mean", "fuel_energy", "pct", "Mean state of charge", "realtime", powertrain="ev"
    ),
    SignalDef(
        "dc_fast_charge_share",
        "fuel_energy",
        "ratio",
        "Share of charge energy from DCFC (30d)",
        "daily",
        powertrain="ev",
        role="driver",
        drives=(V,),
    ),
    SignalDef(
        "charge_cycles",
        "fuel_energy",
        "count",
        "Cumulative equivalent full cycles",
        "daily",
        powertrain="ev",
        role="driver",
        drives=(V,),
    ),
    SignalDef(
        "charge_energy_kwh", "fuel_energy", "kWh", "Energy charged", "daily", powertrain="ev"
    ),
    SignalDef(
        "battery_temp_max_c",
        "fuel_energy",
        "C",
        "Max HV battery temperature",
        "realtime",
        powertrain="ev",
        role="driver",
        drives=(V,),
    ),
    SignalDef(
        "battery_range_mi",
        "fuel_energy",
        "mi",
        "Estimated range at full",
        "daily",
        powertrain="ev",
        role="decoy",
        decoy_of="charge_cycles",
    ),
    SignalDef(
        "cabin_temp_max_c",
        "fuel_energy",
        "C",
        "Max cabin temperature",
        "realtime",
        powertrain="ev",
        role="context",
        drives=(V,),
    ),
    SignalDef(
        "regen_energy_kwh",
        "fuel_energy",
        "kWh",
        "Regenerative energy recovered",
        "daily",
        powertrain="ev",
    ),
    SignalDef(
        "charge_events_count", "fuel_energy", "count", "Charging sessions", "daily", powertrain="ev"
    ),
    # ------------------------------------------------------------ context
    SignalDef(
        "ambient_temp_max_c",
        "context",
        "C",
        "Max ambient temperature",
        "daily",
        role="context",
        drives=(V,),
    ),
    SignalDef("ambient_temp_min_c", "context", "C", "Min ambient temperature", "daily"),
    SignalDef("precipitation_mm", "context", "mm", "Precipitation", "daily"),
    SignalDef("altitude_mean_m", "context", "m", "Mean altitude", "daily"),
    SignalDef("day_of_week", "context", "index", "Day of week", "daily"),
    SignalDef("is_holiday", "context", "flag", "Public holiday", "daily"),
    # ------------------------------------------------------------ derived / decoys / noise / leakage
    SignalDef(
        "brake_wear_est_rear_pct",
        "derived",
        "pct",
        "Dealer-system rear wear estimate",
        "weekly",
        role="decoy",
        decoy_of="brake_pad_wear_pct",
    ),
    SignalDef(
        "pad_thickness_proxy_mm",
        "derived",
        "mm",
        "Pad thickness proxy from dealer DMS",
        "weekly",
        role="decoy",
        decoy_of="brake_pad_wear_pct",
    ),
    SignalDef(
        "mileage_band",
        "derived",
        "band",
        "Coarse mileage band",
        "daily",
        role="decoy",
        decoy_of="odometer_delta_mi",
    ),
    SignalDef(
        "aggressive_driving_index",
        "derived",
        "index",
        "Vendor aggressive-driving index",
        "daily",
        role="decoy",
        decoy_of="harsh_brake_count",
    ),
    SignalDef(
        "brake_dtc_history_30d",
        "derived",
        "count",
        "Brake DTC count, 30d, from DMS",
        "weekly",
        role="decoy",
        decoy_of="dtc_brake_family",
    ),
    SignalDef(
        "overnight_risk_score",
        "derived",
        "score",
        "Vendor overnight risk score",
        "daily",
        role="decoy",
        decoy_of="night_park_share",
    ),
    SignalDef(
        "parking_variability_score",
        "derived",
        "score",
        "Vendor parking variability",
        "daily",
        role="decoy",
        decoy_of="dwell_location_entropy",
    ),
    SignalDef(
        "soc_stress_index",
        "derived",
        "index",
        "Vendor SoC stress index",
        "daily",
        powertrain="ev",
        role="decoy",
        decoy_of="soc_min_daily",
    ),
    SignalDef(
        "fast_charge_flag_30d",
        "derived",
        "flag",
        "Any DCFC in 30d",
        "daily",
        powertrain="ev",
        role="decoy",
        decoy_of="dc_fast_charge_share",
    ),
    SignalDef(
        "thermal_stress_index",
        "derived",
        "index",
        "Vendor thermal stress index",
        "daily",
        powertrain="ev",
        role="decoy",
        decoy_of="battery_temp_max_c",
    ),
    SignalDef(
        "infotainment_reboots", "derived", "count", "Infotainment reboots", "daily", role="noise"
    ),
    SignalDef(
        "bluetooth_pairings", "derived", "count", "Bluetooth pairings", "daily", role="noise"
    ),
    SignalDef("wiper_activations", "derived", "count", "Wiper activations", "daily", role="noise"),
    SignalDef("door_open_count", "derived", "count", "Door open events", "daily", role="noise"),
    SignalDef("radio_volume_mean", "derived", "level", "Mean radio volume", "daily", role="noise"),
    SignalDef(
        "service_appointment_scheduled",
        "derived",
        "flag",
        "Service appointment scheduled (DMS export)",
        "daily",
        role="leakage",
        leaks=B,
    ),
    SignalDef(
        "service_interval_remaining_days",
        "derived",
        "days",
        "Days to next service (DMS, retroactive)",
        "daily",
        role="leakage",
        leaks=B,
    ),
]

BY_ID: dict[str, SignalDef] = {s.signal_id: s for s in SIGNALS}


def drivers_for(target: str) -> list[str]:
    return [s.signal_id for s in SIGNALS if s.role == "driver" and target in s.drives]


def context_for(target: str) -> list[str]:
    return [s.signal_id for s in SIGNALS if s.role == "context" and target in s.drives]


def decoys() -> dict[str, str]:
    return {s.signal_id: s.decoy_of for s in SIGNALS if s.role == "decoy" and s.decoy_of}


def noise_signals() -> list[str]:
    return [s.signal_id for s in SIGNALS if s.role == "noise"]


def leakage_for(target: str) -> list[str]:
    return [s.signal_id for s in SIGNALS if s.role == "leakage" and s.leaks == target]


def signal_ids() -> list[str]:
    return [s.signal_id for s in SIGNALS]

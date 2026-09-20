"""The canonical data contract every DataSource must satisfy.

Three tables, one grain each. The engine never reads anything else, so this is the whole
integration surface between Motorq's platform and the decision engine:

    vehicles       one row per vehicle      vehicle_id, oem, model_year, powertrain (+ optional segment, climate, duty)
    events         one row per event        vehicle_id, date, event_type (+ optional detail)
    signals_daily  one row per vehicle-day  vehicle_id, date, oem, model_year, active, <one column per signal>

`validate()` checks structure, keys, types, referential integrity and date sanity, and returns
a report instead of raising: ingestion problems are findings, and the CLI (`mde data check`)
prints them. Anything listed under `errors` makes the source unusable; `warnings` are reported
in the brief's data section.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

VEHICLE_REQUIRED = {
    "vehicle_id": "string",
    "oem": "string",
    "model_year": "integer",
    "powertrain": "string",
}
VEHICLE_OPTIONAL = {"segment": "string", "climate": "string", "duty": "string"}
EVENT_REQUIRED = {"vehicle_id": "string", "date": "datetime", "event_type": "string"}
SIGNAL_KEYS = {
    "vehicle_id": "string",
    "date": "datetime",
    "oem": "string",
    "model_year": "integer",
    "active": "boolean",
}
POWERTRAINS = {"ice", "ev", "hybrid", "phev", "any"}
MODEL_YEAR_RANGE = (1990, 2035)


@dataclass
class ContractReport:
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    stats: dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not self.errors

    def as_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "errors": self.errors,
            "warnings": self.warnings,
            "stats": self.stats,
        }


def _kind(s: pd.Series) -> str:
    if pd.api.types.is_bool_dtype(s):
        return "boolean"
    if pd.api.types.is_datetime64_any_dtype(s):
        return "datetime"
    if pd.api.types.is_integer_dtype(s):
        return "integer"
    if pd.api.types.is_float_dtype(s):
        return "float"
    return "string"


def _check_columns(
    df: pd.DataFrame, required: dict[str, str], table: str, rep: ContractReport
) -> None:
    for col, kind in required.items():
        if col not in df.columns:
            rep.errors.append(f"{table}: missing required column '{col}'")
            continue
        got = _kind(df[col])
        ok = got == kind or (
            kind == "integer" and got == "float" and df[col].dropna().mod(1).eq(0).all()
        )
        if not ok:
            rep.errors.append(f"{table}.{col}: expected {kind}, got {got}")
        if df[col].isna().any() and col != "detail":
            rep.errors.append(
                f"{table}.{col}: {int(df[col].isna().sum())} null values in a key/required column"
            )


def validate(
    vehicles: pd.DataFrame,
    events: pd.DataFrame,
    signals: pd.DataFrame,
    signal_ids: list[str],
    today: pd.Timestamp | None = None,
    inactive_ok: frozenset[str] = frozenset(),
) -> ContractReport:
    """Validate the three canonical frames. `signals` may be a subset of signal columns.
    `inactive_ok` names signals allowed to report on inactive vehicle-days (location pings
    from a stolen, otherwise dark vehicle)."""
    rep = ContractReport()
    today = today or pd.Timestamp.utcnow().tz_localize(None).normalize()

    # ---- vehicles
    _check_columns(vehicles, VEHICLE_REQUIRED, "vehicles", rep)
    if "vehicle_id" in vehicles:
        dup = int(vehicles["vehicle_id"].duplicated().sum())
        if dup:
            rep.errors.append(f"vehicles: {dup} duplicate vehicle_id rows")
    if "powertrain" in vehicles:
        bad = sorted(set(vehicles["powertrain"].dropna().astype(str).str.lower()) - POWERTRAINS)
        if bad:
            rep.errors.append(
                f"vehicles.powertrain: unknown values {bad}; expected one of {sorted(POWERTRAINS)}"
            )
    if "model_year" in vehicles and pd.api.types.is_numeric_dtype(vehicles["model_year"]):
        lo, hi = MODEL_YEAR_RANGE
        out = int(((vehicles["model_year"] < lo) | (vehicles["model_year"] > hi)).sum())
        if out:
            rep.errors.append(f"vehicles.model_year: {out} values outside {lo}-{hi}")
    if "oem" in vehicles:
        n_oem = int(vehicles["oem"].nunique())
        rep.stats["n_oems"] = n_oem
        if n_oem < 2:
            rep.warnings.append("vehicles: a single OEM; leave-one-OEM-out validation cannot run")
    rep.stats["n_vehicles"] = int(len(vehicles))

    # ---- events
    _check_columns(events, EVENT_REQUIRED, "events", rep)
    ids = set(vehicles["vehicle_id"]) if "vehicle_id" in vehicles else set()
    if "vehicle_id" in events:
        orphan = int((~events["vehicle_id"].isin(ids)).sum())
        if orphan:
            rep.errors.append(f"events: {orphan} rows reference a vehicle_id not in vehicles")
    if {"vehicle_id", "date", "event_type"} <= set(events.columns):
        dup = int(events.duplicated(["vehicle_id", "date", "event_type"]).sum())
        if dup:
            rep.warnings.append(
                f"events: {dup} duplicate (vehicle_id, date, event_type) rows; the engine keeps one"
            )
        future = int((pd.to_datetime(events["date"]) > today).sum())
        if future:
            rep.errors.append(f"events: {future} rows dated in the future")
        rep.stats["events_by_type"] = {
            k: int(v) for k, v in events["event_type"].value_counts().items()
        }

    # ---- signals_daily
    _check_columns(signals, SIGNAL_KEYS, "signals_daily", rep)
    missing_sig = [s for s in signal_ids if s not in signals.columns]
    if missing_sig:
        rep.errors.append(
            f"signals_daily: {len(missing_sig)} catalogued signals absent: {missing_sig[:8]}{' ...' if len(missing_sig) > 8 else ''}"
        )
    present_sig = [s for s in signal_ids if s in signals.columns]
    non_numeric = [
        s
        for s in present_sig
        if not (pd.api.types.is_numeric_dtype(signals[s]) or pd.api.types.is_bool_dtype(signals[s]))
    ]
    if non_numeric:
        rep.errors.append(f"signals_daily: non-numeric signal columns {non_numeric[:8]}")
    if {"vehicle_id", "date"} <= set(signals.columns):
        dup = int(signals.duplicated(["vehicle_id", "date"]).sum())
        if dup:
            rep.errors.append(
                f"signals_daily: {dup} duplicate (vehicle_id, date) rows; the grain is one row per vehicle-day"
            )
        orphan = int((~signals["vehicle_id"].isin(ids)).sum())
        if orphan:
            rep.errors.append(
                f"signals_daily: {orphan} rows reference a vehicle_id not in vehicles"
            )
        d = pd.to_datetime(signals["date"])
        if len(d):
            span_days = int((d.max() - d.min()).days) + 1
            rep.stats["date_range"] = [str(d.min().date()), str(d.max().date())]
            rep.stats["n_days"] = span_days
            expected = len(ids) * span_days
            rep.stats["grid_completeness"] = round(len(signals) / expected, 4) if expected else None
            if expected and len(signals) < expected:
                rep.warnings.append(
                    f"signals_daily: {expected - len(signals)} vehicle-days missing from the full grid (treated as inactive)"
                )
            if span_days < 180:
                rep.warnings.append(
                    f"signals_daily: only {span_days} days of history; forward validation needs >= 180"
                )
            if (d > today).any():
                rep.errors.append("signals_daily: rows dated in the future")
        if "date" in events and len(events):
            e = pd.to_datetime(events["date"])
            if len(d) and (e.min() < d.min() or e.max() > d.max()):
                rep.warnings.append(
                    "events: some events fall outside the signal date range and cannot be labelled"
                )
    if "active" in signals and pd.api.types.is_bool_dtype(signals["active"]):
        rep.stats["active_share"] = round(float(signals["active"].mean()), 4)
        if present_sig:
            # an inactive vehicle-day may carry a zero count; any other value is a pipeline bug
            inactive = signals.loc[~signals["active"], present_sig]
            leak = [
                s
                for s in present_sig
                if s not in inactive_ok and (inactive[s].fillna(0) != 0).any()
            ]
            if leak:
                rep.warnings.append(
                    f"signals_daily: {len(leak)} signals carry non-zero values on inactive vehicle-days (e.g. {leak[:4]}); expected null"
                )
    rep.stats["n_signal_rows"] = int(len(signals))
    rep.stats["n_signals_checked"] = len(present_sig)
    return rep


def validate_source(
    source: Any, signal_ids: list[str] | None = None, sample_signals: int = 16
) -> ContractReport:
    """Validate a DataSource end-to-end. Reads a sample of signal columns to keep it cheap."""
    metas = source.list_signals()
    ids = signal_ids or [m.signal_id for m in metas]
    sample = ids[:sample_signals]
    signals = source.read_signals(sample)
    location = frozenset(m.signal_id for m in metas if m.category == "location_trips")
    rep = validate(source.vehicles(), source.events(), signals, sample, inactive_ok=location)
    rep.stats["n_signals_catalogued"] = len(ids)
    rep.stats["dataset_hash"] = getattr(source, "dataset_hash", None)
    return rep


def numeric_signal_columns(df: pd.DataFrame, signal_ids: list[str]) -> list[str]:
    return [s for s in signal_ids if s in df.columns and np.issubdtype(df[s].dtype, np.number)]

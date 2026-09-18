"""Load and validate world parameters and the OEM coverage matrix."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from motorq_de.world.registry import BY_ID, signal_ids

HERE = Path(__file__).parent


def load_params(
    path: Path | None = None, overrides: dict[str, Any] | None = None
) -> dict[str, Any]:
    with open(path or HERE / "params.yaml", encoding="utf-8") as fh:
        params = yaml.safe_load(fh)
    if overrides:
        params = _deep_merge(params, overrides)
    return params


def _deep_merge(base: dict, over: dict) -> dict:
    out = dict(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


@dataclass(frozen=True)
class SignalCoverage:
    emits: bool
    from_model_year: int
    frequency: str
    missing_rate: float


@dataclass(frozen=True)
class OemProfile:
    oem: str
    fleet_weight: float
    ev_share: float
    connectivity_from_model_year: int
    signals: dict[str, SignalCoverage]


@dataclass(frozen=True)
class CoverageMatrix:
    oems: dict[str, OemProfile]
    vendor_signals: frozenset[str]

    def coverage(self, oem: str, signal_id: str) -> SignalCoverage:
        return self.oems[oem].signals[signal_id]

    def oem_ids(self) -> list[str]:
        return list(self.oems)

    def fleet_weights(self) -> list[float]:
        return [p.fleet_weight for p in self.oems.values()]


def load_coverage(path: Path | None = None) -> CoverageMatrix:
    with open(path or HERE / "coverage.yaml", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    defaults = raw["defaults"]
    vendor = frozenset(raw.get("vendor_signals", []))
    oems: dict[str, OemProfile] = {}
    for oem, prof in raw["oems"].items():
        conn_from = int(
            prof.get("connectivity_from_model_year", defaults["connectivity_from_model_year"])
        )
        overrides = prof.get("signals", {}) or {}
        sigs: dict[str, SignalCoverage] = {}
        for sid in signal_ids():
            sdef = BY_ID[sid]
            o = overrides.get(sid, {})
            base_missing = (
                defaults["realtime_missing_rate"]
                if sdef.frequency == "realtime"
                else defaults["missing_rate"]
            )
            # vendor signals are available for every OEM regardless of connectivity year
            from_my = 0 if sid in vendor else int(o.get("from_model_year", conn_from))
            sigs[sid] = SignalCoverage(
                emits=bool(o.get("emits", True)),
                from_model_year=from_my,
                frequency=str(o.get("frequency", sdef.frequency)),
                missing_rate=float(o.get("missing_rate", base_missing)),
            )
        oems[oem] = OemProfile(
            oem=oem,
            fleet_weight=float(prof["fleet_weight"]),
            ev_share=float(prof["ev_share"]),
            connectivity_from_model_year=conn_from,
            signals=sigs,
        )
    total = sum(p.fleet_weight for p in oems.values())
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"OEM fleet weights must sum to 1, got {total}")
    return CoverageMatrix(oems=oems, vendor_signals=vendor)

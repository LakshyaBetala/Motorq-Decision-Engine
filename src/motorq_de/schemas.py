"""Shared, frozen contracts used across every layer.

Nothing in here computes anything. These are the shapes that tools accept and
emit; the reliability contract (determinism, provenance) is enforced on them.
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

TargetEvent = Literal["brake_service_event", "theft_event", "battery_degradation_event"]
DeliveryMode = Literal["batch_daily", "batch_hourly", "streaming"]
Consumer = Literal["fuse_action_hub", "fuse_assistant", "api", "internal"]
Decision = Literal["BUILD_READY", "PILOT", "NOT_FEASIBLE"]


class Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


# --------------------------------------------------------------------------- spec


class Range(Frozen):
    """A three-point estimate. Sampled as a PERT distribution in economics."""

    low: float
    base: float
    high: float

    @model_validator(mode="after")
    def _ordered(self) -> Range:
        if not (self.low <= self.base <= self.high):
            raise ValueError(f"Range must satisfy low <= base <= high, got {self}")
        return self


class ValueAssumptions(Frozen):
    """Human-supplied business inputs. The engine propagates their uncertainty;
    it never invents them. The target event *rate* is NOT here: the harness measures it
    from the data. Humans supply what data cannot know:

    value_bearing_fraction   share of target events that would have caused the costly
                             outcome (e.g. a brake service that would have been an unplanned
                             roadside failure rather than a scheduled visit)
    preventable_fraction     given a correct early warning, share of that cost avoided
    usd_per_avoided_event    cost of one such outcome
    fleet_size               vehicles the capability would run on
    """

    value_bearing_fraction: Range
    preventable_fraction: Range
    usd_per_avoided_event: Range
    fleet_size: Range
    validated_by_pilot: bool = False
    source_note: str = ""


class Constraints(Frozen):
    min_oem_coverage: float = Field(0.6, ge=0.0, le=1.0)
    max_latency_s: int | None = None
    max_run_cost_usd_month: float | None = None
    min_signal_history_months: int = 6


class ProblemSpec(Frozen):
    capability_name: str
    target_event: TargetEvent
    horizon_days: int = Field(7, ge=1, le=90)
    decision_unit: Literal["vehicle_day"] = "vehicle_day"
    # the decision population; "auto" scopes EV-only targets to EVs
    population_powertrain: Literal["auto", "any", "ev", "ice"] = "auto"
    delivery_mode: DeliveryMode = "batch_daily"
    consumer: Consumer = "fuse_action_hub"
    value: ValueAssumptions
    constraints: Constraints = Constraints()
    seed: int = 42

    @property
    def powertrain_scope(self) -> str:
        if self.population_powertrain != "auto":
            return self.population_powertrain
        return "ev" if self.target_event == "battery_degradation_event" else "any"


# --------------------------------------------------------------------------- data


class SignalMeta(Frozen):
    signal_id: str
    category: Literal[
        "location_trips", "health", "driver_behavior", "fuel_energy", "context", "derived"
    ]
    unit: str
    description: str
    declared_frequency: Literal["realtime", "daily", "weekly"]
    powertrain: Literal["any", "ev", "ice"] = "any"


class CoverageStat(Frozen):
    oem: str
    emits: bool
    from_model_year: int | None
    vehicles_total: int
    vehicles_covered: int
    observed_nonnull_rate: float

    @property
    def coverage(self) -> float:
        return self.vehicles_covered / self.vehicles_total if self.vehicles_total else 0.0


class QualityStat(Frozen):
    signal_id: str
    nonnull_rate: float
    median_gap_days: float
    declared_gap_days: float
    psi_first_last_quarter: float
    history_months: float


class DateWindow(Frozen):
    start: date
    end: date


# --------------------------------------------------------------------------- evidence


class Evidence(Frozen):
    evidence_id: str
    run_id: str
    step: str
    tool: str
    inputs: dict[str, Any]
    inputs_hash: str
    dataset_hash: str
    seed: int
    outputs: dict[str, Any]
    created_at: datetime


class Cited(Frozen):
    """A number that can appear in a brief. Renderer fails closed if evidence_ids is empty."""

    value: float | int | str
    evidence_ids: tuple[str, ...]
    label: str = ""


# --------------------------------------------------------------------------- verdict


class GateResult(Frozen):
    name: str
    passed: bool
    value: float | None
    threshold: float | None
    evidence_ids: tuple[str, ...]
    note: str = ""


class FlagResult(Frozen):
    name: str
    tripped: bool
    value: float | None
    threshold: float | None
    evidence_ids: tuple[str, ...]
    note: str = ""


class Verdict(Frozen):
    decision: Decision
    gates: tuple[GateResult, ...]
    flags: tuple[FlagResult, ...]
    policy_version: str

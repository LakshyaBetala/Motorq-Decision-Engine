from __future__ import annotations

import json
from pathlib import Path

import pytest

from motorq_de.data.synthetic import SyntheticSource
from motorq_de.schemas import ProblemSpec, Range, ValueAssumptions
from motorq_de.world.generator import WorldGenerator


def R(low: float, base: float, high: float) -> Range:
    return Range(low=low, base=base, high=high)


@pytest.fixture(scope="session")
def dataset_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("synthetic")
    return WorldGenerator(profile="small", seed=7).generate(root)


@pytest.fixture(scope="session")
def truth(dataset_dir: Path) -> dict:
    return json.loads((dataset_dir / "truth.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="session")
def source(dataset_dir: Path) -> SyntheticSource:
    return SyntheticSource(dataset_dir)


@pytest.fixture(scope="session")
def brake_spec() -> ProblemSpec:
    # Value ranges sourced: downtime $448-760/day and $3.5k-6.5k/incident (oxmaint, fleetrabbit);
    # predictive maintenance cuts unplanned downtime 30-50%.
    return ProblemSpec(
        capability_name="brake_service_7d",
        target_event="brake_service_event",
        horizon_days=7,
        value=ValueAssumptions(
            value_bearing_fraction=R(0.05, 0.1, 0.2),
            preventable_fraction=R(0.3, 0.4, 0.5),
            usd_per_avoided_event=R(1500, 3500, 6500),
            fleet_size=R(5000, 10000, 20000),
            source_note="downtime cost per incident from public fleet studies; see params.yaml",
        ),
    )


@pytest.fixture(scope="session")
def theft_spec() -> ProblemSpec:
    return ProblemSpec(
        capability_name="theft_risk_30d",
        target_event="theft_event",
        horizon_days=30,
        value=ValueAssumptions(
            value_bearing_fraction=R(0.6, 0.8, 1.0),
            preventable_fraction=R(0.1, 0.2, 0.3),
            usd_per_avoided_event=R(8000, 15000, 30000),
            fleet_size=R(5000, 10000, 20000),
        ),
    )


@pytest.fixture(scope="session")
def battery_spec() -> ProblemSpec:
    return ProblemSpec(
        capability_name="battery_service_14d",
        target_event="battery_degradation_event",
        horizon_days=14,
        value=ValueAssumptions(
            value_bearing_fraction=R(0.3, 0.5, 0.8),
            preventable_fraction=R(0.2, 0.3, 0.5),
            usd_per_avoided_event=R(2000, 5000, 12000),
            fleet_size=R(500, 1000, 3000),
        ),
    )

"""Deployment-fit rubric. Motorq has an architecture (Kafka + Snowpipe Streaming into managed
Iceberg tables, REST/Kafka/Snowflake delivery); this only decides which existing pattern a
capability fits and whether the spec's constraints are met. It does not propose stacks.
"""

from __future__ import annotations

from typing import Any

from motorq_de.schemas import ProblemSpec

PATTERNS = {
    "batch_daily": {
        "description": "Daily feature job on Snowflake, scores written to a table exposed via API / Fuse",
        "typical_latency_s": 24 * 3600,
        "min_horizon_days": 1,
    },
    "batch_hourly": {
        "description": "Hourly incremental feature job on Snowflake dynamic tables, scores via API",
        "typical_latency_s": 3600,
        "min_horizon_days": 0,
    },
    "streaming": {
        "description": "Kafka consumer computes features and scores per event; results to Kafka + API",
        "typical_latency_s": 60,
        "min_horizon_days": 0,
    },
}


def deployment_fit(spec: ProblemSpec) -> dict[str, Any]:
    c = spec.constraints
    candidates = []
    for name, pat in PATTERNS.items():
        meets_latency = c.max_latency_s is None or pat["typical_latency_s"] <= c.max_latency_s
        meets_horizon = spec.horizon_days >= pat["min_horizon_days"]
        candidates.append(
            {
                "pattern": name,
                "description": pat["description"],
                "typical_latency_s": pat["typical_latency_s"],
                "meets_latency": meets_latency,
                "meets_horizon": meets_horizon,
                "feasible": meets_latency and meets_horizon,
            }
        )
    feasible = [x for x in candidates if x["feasible"]]
    # recommend the cheapest feasible pattern that matches the requested delivery mode, else cheapest feasible
    order = ["batch_daily", "batch_hourly", "streaming"]
    feasible.sort(key=lambda x: order.index(x["pattern"]))
    requested = next((x for x in feasible if x["pattern"] == spec.delivery_mode), None)
    recommended = requested or (feasible[0] if feasible else None)
    note = ""
    if spec.horizon_days >= 1 and spec.delivery_mode == "streaming":
        note = f"A {spec.horizon_days}-day horizon does not need streaming; batch_daily is sufficient and cheaper."
    return {
        "requested_delivery_mode": spec.delivery_mode,
        "horizon_days": spec.horizon_days,
        "max_latency_s": c.max_latency_s,
        "candidates": candidates,
        "recommended_pattern": recommended["pattern"] if recommended else None,
        "any_feasible": bool(feasible),
        "note": note,
    }

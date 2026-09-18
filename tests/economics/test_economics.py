from __future__ import annotations

import numpy as np

from motorq_de.economics.cost import compare_costs, load_price_sheet, run_cost
from motorq_de.economics.deployment import deployment_fit
from motorq_de.economics.value import choose_operating_point, pert, roi_distribution, tornado
from motorq_de.schemas import ProblemSpec, Range, ValueAssumptions
from motorq_de.world import registry as reg

R = lambda a, b, c: Range(low=a, base=b, high=c)  # noqa: E731
METAS = {s.signal_id: s.meta() for s in reg.SIGNALS}


def test_price_sheet_every_section_sourced_or_placeholder():
    p = load_price_sheet()
    text = open(
        __import__("motorq_de.economics.cost", fromlist=["HERE"]).HERE / "price_sheet.yaml",
        encoding="utf-8",
    ).read()
    for section in (
        "snowflake",
        "kafka",
        "oem_data",
        "telemetry",
        "inference",
        "engineering",
        "operations",
    ):
        body = p[section]
        sourced = "source:" in text.split(f"\n{section}:")[1].split("\n\n")[0]
        assert sourced or body.get("placeholder"), section


def test_cost_scales_with_volume_and_reports_levers():
    full = run_cost([METAS[s] for s in reg.signal_ids()[:60]], 10_000, "batch_daily")
    small = run_cost(
        [METAS["brake_pad_wear_pct"], METAS["odometer_delta_mi"]], 10_000, "batch_daily"
    )
    assert full["volumes"]["gb_per_month"] > small["volumes"]["gb_per_month"] * 5
    assert full["monthly"]["infra_total"] > small["monthly"]["infra_total"]
    # the honest message: infra is small; build and OEM dominate
    assert full["monthly"]["infra_total"] < full["monthly"]["build_amortized"]
    assert "lever" in full["lever_note"].lower()
    cmp = compare_costs(full, small)
    assert cmp["infra_change_pct"] < -0.5


def test_polling_cost_driven_by_highest_cadence_not_count():
    rt = run_cost([METAS["harsh_brake_count"]], 10_000, "batch_daily")  # realtime
    daily_many = run_cost(
        [METAS[s] for s in ("brake_pad_wear_pct", "odometer_delta_mi", "oil_life_pct")],
        10_000,
        "batch_daily",
    )
    assert rt["monthly"]["oem_api_calls"] > daily_many["monthly"]["oem_api_calls"] * 50
    assert rt["polling"]["highest_cadence_signals"] == ["harsh_brake_count"]


def test_oem_marginal_zero_when_packages_already_bought():
    c = run_cost([METAS["brake_pad_wear_pct"]], 10_000, "batch_daily")
    assert c["monthly"]["oem_marginal"] == 0
    assert c["monthly"]["oem_attributed"] > 0
    assert "oem_data" in c["placeholders"]


def test_pert_respects_bounds_and_mode():
    rng = np.random.default_rng(0)
    x = pert(rng, R(1, 2, 10), 20_000)
    assert x.min() >= 1 and x.max() <= 10
    assert 2 < np.median(x) < 4  # right-skewed, mode near 2


def _va():
    return ValueAssumptions(
        value_bearing_fraction=R(0.05, 0.1, 0.2),
        preventable_fraction=R(0.3, 0.4, 0.5),
        usd_per_avoided_event=R(1500, 3500, 6500),
        fleet_size=R(5000, 10000, 20000),
    )


def test_roi_distribution_is_deterministic_and_coherent():
    a = roi_distribution(
        _va(), R(0.8, 1.0, 1.2), R(0.4, 0.5, 0.6), 0.02, 7, R(40, 75, 120), 2000.0, 1500.0, seed=1
    )
    b = roi_distribution(
        _va(), R(0.8, 1.0, 1.2), R(0.4, 0.5, 0.6), 0.02, 7, R(40, 75, 120), 2000.0, 1500.0, seed=1
    )
    assert a == b
    assert a["roi"]["p5"] <= a["roi"]["p50"] <= a["roi"]["p95"]
    assert 0 <= a["p_roi_positive"] <= 1


def test_roi_negative_when_value_tiny():
    va = ValueAssumptions(
        value_bearing_fraction=R(0.01, 0.02, 0.03),
        preventable_fraction=R(0.05, 0.1, 0.15),
        usd_per_avoided_event=R(50, 100, 150),
        fleet_size=R(500, 1000, 1500),
    )
    r = roi_distribution(
        va, R(0.8, 1.0, 1.2), R(0.3, 0.4, 0.5), 0.05, 7, R(40, 75, 120), 5000.0, 1500.0
    )
    assert r["p_roi_positive"] < 0.05


def test_tornado_ranks_dollar_per_event_high():
    t = tornado(_va(), R(0.8, 1.0, 1.2), R(0.4, 0.5, 0.6), 0.02, 7, R(40, 75, 120), 2000.0, 1500.0)
    names = [r["input"] for r in t["ranked"]]
    assert names.index("usd_per_avoided_event") < 3
    assert all(r["swing"] >= 0 for r in t["ranked"])


def test_operating_point_trades_alerts_against_value():
    ops = [
        {"alert_rate": 0.005, "recall": 0.15, "precision": 0.5},
        {"alert_rate": 0.02, "recall": 0.40, "precision": 0.3},
        {"alert_rate": 0.10, "recall": 0.80, "precision": 0.1},
    ]
    cheap_alerts = choose_operating_point(
        _va(), R(0.8, 1.0, 1.2), ops, 0.05, 7, R(1, 2, 3), 2000.0, 1500.0
    )
    dear_alerts = choose_operating_point(
        _va(), R(0.8, 1.0, 1.2), ops, 0.05, 7, R(400, 750, 1200), 2000.0, 1500.0
    )
    assert cheap_alerts["chosen"]["alert_rate"] >= dear_alerts["chosen"]["alert_rate"]


def test_deployment_rubric():
    va = _va()
    daily = ProblemSpec(
        capability_name="x", target_event="brake_service_event", horizon_days=7, value=va
    )
    fit = deployment_fit(daily)
    assert fit["recommended_pattern"] == "batch_daily" and fit["any_feasible"]
    tight = daily.model_copy(
        update={"constraints": daily.constraints.model_copy(update={"max_latency_s": 120})}
    )
    assert deployment_fit(tight)["recommended_pattern"] == "streaming"
    over = daily.model_copy(update={"delivery_mode": "streaming"})
    assert "does not need streaming" in deployment_fit(over)["note"]

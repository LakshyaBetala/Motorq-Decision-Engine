"""Policy boundaries. Every gate and flag is exercised at its threshold."""

from __future__ import annotations

import pytest

from motorq_de.policy.verdict import THRESHOLDS, EvidenceBundle, decide
from motorq_de.schemas import ProblemSpec, Range, ValueAssumptions

R = lambda a, b, c: Range(low=a, base=b, high=c)  # noqa: E731


def spec(**over):
    va = ValueAssumptions(
        value_bearing_fraction=R(0.05, 0.1, 0.2),
        preventable_fraction=R(0.3, 0.4, 0.5),
        usd_per_avoided_event=R(1500, 3500, 6500),
        fleet_size=R(5000, 10000, 20000),
        validated_by_pilot=over.pop("validated", False),
    )
    return ProblemSpec(capability_name="t", target_event="brake_service_event", value=va, **over)


def bundle(
    share=0.8,
    auc_lo=0.85,
    base=0.7,
    p_pos=0.9,
    feasible=True,
    xo_std=0.01,
    xo_gap=0.01,
    deg_upper=0.01,
    roi_p5=0.5,
    history=12.0,
    underpowered=False,
    suspicious=(),
    placeholders=(),
    lc_gain=0.002,
):
    ev = EvidenceBundle()
    ev.put(
        "learning_curve",
        {"auc_gain_half_to_full": lc_gain, "still_improving": lc_gain > 0.01},
        "ev_lc",
    )
    ev.put("coverage_sufficient", {"fleet_share_full_set": share}, "ev_cov")
    ev.put(
        "model_comparison",
        {
            "sets": {
                "sufficient": {
                    "models": {
                        "lightgbm": {
                            "auc": {"point": auc_lo + 0.02, "lo": auc_lo, "hi": auc_lo + 0.04}
                        }
                    },
                    "baselines": {"best_single_signal_auc": base},
                }
            }
        },
        "ev_cmp",
    )
    ev.put(
        "roi", {"p_roi_positive": p_pos, "roi": {"p5": roi_p5, "p50": 1.0, "p95": 2.0}}, "ev_roi"
    )
    ev.put(
        "deployment",
        {"any_feasible": feasible, "recommended_pattern": "batch_daily" if feasible else None},
        "ev_dep",
    )
    ev.put(
        "cross_oem",
        {"std_auc": xo_std, "mean_auc": 0.85, "min_auc": 0.85 - xo_gap, "worst_oem": "oem_c"},
        "ev_xo",
    )
    ev.put("temporal", {"degradation_upper": deg_upper}, "ev_tv")
    ev.put("quality_sufficient", {"per_signal": {"a": {"history_months": history}}}, "ev_q")
    ev.put(
        "ablation",
        {
            "underpowered": underpowered,
            "resolution": 0.001,
            "tolerance": 0.005,
            "sufficient_set": ["a", "b"],
        },
        "ev_ab",
    )
    ev.put("leakage", {"suspicious": list(suspicious)}, "ev_lk")
    ev.put(
        "cost_sufficient",
        {"placeholders": list(placeholders), "monthly": {"marginal_total": 1500.0}},
        "ev_cost",
    )
    return ev


def test_build_ready_when_everything_clean():
    v = decide(spec(validated=True), bundle())
    assert v.decision == "BUILD_READY"
    assert all(g.passed for g in v.gates) and not any(f.tripped for f in v.flags)
    assert all(g.evidence_ids for g in v.gates)


def test_default_spec_is_pilot_because_value_unvalidated():
    v = decide(spec(), bundle())
    assert v.decision == "PILOT"
    assert [f.name for f in v.flags if f.tripped] == ["value_unvalidated"]


@pytest.mark.parametrize(
    "kw",
    [
        {"share": 0.59},
        {"auc_lo": 0.70, "base": 0.70},
        {"p_pos": 0.49},
        {"feasible": False},
    ],
)
def test_each_gate_fails_to_not_feasible(kw):
    v = decide(spec(validated=True), bundle(**kw))
    assert v.decision == "NOT_FEASIBLE"
    assert sum(not g.passed for g in v.gates) == 1


def test_gate_boundaries_inclusive_exclusive():
    assert decide(spec(validated=True), bundle(share=0.6)).gates[0].passed  # >= threshold passes
    assert decide(spec(validated=True), bundle(p_pos=0.5)).gates[2].passed
    assert (
        not decide(spec(validated=True), bundle(auc_lo=0.7, base=0.7)).gates[1].passed
    )  # lift must be > 0


@pytest.mark.parametrize(
    "kw,flag",
    [
        ({"xo_std": 0.031}, "cross_oem_variance"),
        ({"xo_gap": 0.06}, "cross_oem_variance"),
        ({"deg_upper": 0.031}, "temporal_degradation"),
        ({"roi_p5": -0.1}, "roi_spans_negative"),
        ({"history": 3.0}, "short_history"),
        ({"underpowered": True}, "ablation_underpowered"),
        ({"suspicious": ("a",)}, "suspicious_signals"),
        ({"placeholders": ("oem_data",)}, "cost_placeholders"),
        ({"lc_gain": 0.02}, "data_still_improving"),
    ],
)
def test_each_flag_trips_to_pilot(kw, flag):
    v = decide(spec(validated=True), bundle(**kw))
    assert v.decision == "PILOT"
    assert [f.name for f in v.flags if f.tripped] == [flag]


def test_suspicious_only_counts_signals_in_sufficient_set():
    v = decide(spec(validated=True), bundle(suspicious=("zzz",)))
    assert v.decision == "BUILD_READY"


def test_missing_evidence_fails_closed():
    ev = EvidenceBundle()
    v = decide(spec(validated=True), ev)
    assert v.decision == "NOT_FEASIBLE"
    assert not any(g.passed for g in v.gates)


def test_thresholds_are_reported_on_results():
    v = decide(spec(validated=True), bundle())
    g = {x.name: x for x in v.gates}
    assert g["economics"].threshold == THRESHOLDS["p_roi_positive"]
    assert g["data"].threshold == 0.6


def test_policy_thresholds_come_from_yaml():
    from motorq_de.policy.verdict import POLICY_VERSION

    assert POLICY_VERSION == "1.3"
    assert THRESHOLDS["false_alerts_per_100_vehicle_months_max"] == 25.0


def test_alert_burden_flag():
    ev = bundle()
    ev.put("operating_point", {"chosen": {"false_alerts_per_100_vehicle_months": 40.0}}, "ev_op")
    v = decide(spec(validated=True), ev)
    assert v.decision == "PILOT"
    assert [f.name for f in v.flags if f.tripped] == ["alert_burden"]
    ev.put("operating_point", {"chosen": {"false_alerts_per_100_vehicle_months": 5.0}}, "ev_op")
    assert decide(spec(validated=True), ev).decision == "BUILD_READY"


def test_cost_gate_only_exists_when_a_ceiling_is_requested():
    v = decide(spec(validated=True), bundle())
    assert "cost" not in [g.name for g in v.gates]
    over = spec(validated=True, constraints={"max_run_cost_usd_month": 2000.0})
    v = decide(over, bundle())
    gate = next(g for g in v.gates if g.name == "cost")
    assert gate.passed and gate.value == 1500.0 and gate.threshold == 2000.0
    assert gate.evidence_ids == ("ev_cost",)
    assert v.decision == "BUILD_READY"
    tight = spec(validated=True, constraints={"max_run_cost_usd_month": 1000.0})
    v = decide(tight, bundle())
    assert not next(g for g in v.gates if g.name == "cost").passed
    assert v.decision == "NOT_FEASIBLE"

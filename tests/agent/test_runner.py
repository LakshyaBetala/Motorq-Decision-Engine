"""Integration: the headless default plan runs end to end on the small fixture and produces a
fully cited brief whose every gate and flag resolves to stored evidence. Determinism: two
runs on the same dataset/spec produce identical evidence outputs."""

from __future__ import annotations

import json
import re

import pytest

from motorq_de.agent.runner import Runner
from motorq_de.ledger.store import Ledger, memory_engine
from motorq_de.report.render import check_citations

pytestmark = pytest.mark.slow


def _ledger():
    return Ledger(memory_engine())


@pytest.fixture(scope="module")
def result(source, brake_spec):
    led = _ledger()
    res = Runner(source, led).run(brake_spec)
    return res, led


def test_run_completes_with_verdict_and_brief(result):
    res, led = result
    assert res.verdict.decision in ("BUILD_READY", "PILOT", "NOT_FEASIBLE")
    run = led.get_run(res.run_id)
    assert run["status"] == "done"
    assert [s["name"] for s in run["steps"]] == [
        "DEFINE",
        "FEASIBILITY",
        "EXPERIMENT",
        "ECONOMICS",
        "DELIVERY",
        "POLICY",
        "REPORT",
    ]
    assert all(s["status"] == "done" for s in run["steps"])


def test_every_citation_in_brief_resolves_to_stored_evidence(result):
    res, led = result
    check_citations(res.brief)
    ids = set(re.findall(r"ev_[0-9a-f]{12}", res.brief_md))
    assert ids, "brief cites nothing"
    for eid in ids:
        assert led.get_evidence(eid) is not None, eid


def test_gates_and_flags_cite_evidence(result):
    res, _ = result
    for g in res.verdict.gates:
        assert g.evidence_ids, g.name
    for f in res.verdict.flags:
        if f.tripped and f.name != "value_unvalidated":
            assert f.evidence_ids, f.name


def test_leaks_never_reach_the_model(result):
    res, _ = result
    ab = res.evidence["ablation"].outputs
    fa = res.evidence["feature_analysis"].outputs
    leaks = set(res.evidence["leakage"].outputs["flagged"])
    assert leaks == {"service_appointment_scheduled", "service_interval_remaining_days"}
    assert not (leaks & set(ab["candidate_set"]))
    assert not (leaks & {r["signal"] for r in fa["ranking"]})


def test_economics_uses_measured_event_rate(result):
    res, _ = result
    er = res.evidence["event_rate"].outputs
    roi = res.evidence["roi"].outputs
    assert roi["inputs"]["event_rate"]["base"] == er["rate_per_vehicle_year"]
    assert 0.6 <= er["rate_per_vehicle_year"] <= 1.6


def test_headless_is_deterministic(source, brake_spec, result):
    res1, _ = result
    res2 = Runner(source, _ledger()).run(brake_spec)
    for name in ("feature_analysis", "ablation", "cross_oem", "temporal", "roi", "cost_sufficient"):
        assert res1.evidence[name].outputs == res2.evidence[name].outputs, name

    # evidence ids embed the run id by design; the verdict's substance must match exactly
    def strip(v):
        return [(x.name, x.passed if hasattr(x, "passed") else x.tripped, x.value) for x in v]

    assert res1.verdict.decision == res2.verdict.decision
    assert strip(res1.verdict.gates) == strip(res2.verdict.gates)
    assert strip(res1.verdict.flags) == strip(res2.verdict.flags)


def test_whatif_reuses_harness_evidence_and_recomputes_economics(source, brake_spec, result):
    res, led = result
    runner = Runner(source, led)
    new_value = brake_spec.value.model_copy(
        update={
            "value_bearing_fraction": brake_spec.value.value_bearing_fraction.model_copy(
                update={"low": 0.5, "base": 0.7, "high": 0.9}
            )
        }
    )
    w = runner.whatif(res.run_id, value=new_value)
    run = led.get_run(w.run_id)
    assert run["kind"] == "whatif" and run["derived_from"] == res.run_id
    # harness evidence is the parent's (same ids); economics is new
    assert w.evidence["ablation"].evidence_id == res.evidence["ablation"].evidence_id
    assert w.evidence["roi"].evidence_id != res.evidence["roi"].evidence_id
    assert (
        w.evidence["roi"].outputs["p_roi_positive"] > res.evidence["roi"].outputs["p_roi_positive"]
    )
    check_citations(w.brief)
    # what-if runs in seconds: no harness steps recorded
    assert [s["name"] for s in run["steps"]] == [
        "DEFINE",
        "ECONOMICS",
        "DELIVERY",
        "POLICY",
        "REPORT",
    ]


def test_replay_is_identical(source, brake_spec, result):
    res, led = result
    rep = Runner(source, led).replay(res.run_id)
    assert rep.diff["environment_identical"] is True
    statuses = {r["name"]: r["status"] for r in rep.diff["records"]}
    assert (
        statuses["compute"] == "informational"
    )  # cache bypassed on replay: reported, not compared
    assert rep.diff is not None and rep.diff["identical"] is True, [
        r for r in rep.diff["records"] if r["status"] != "identical"
    ]
    assert led.get_run(rep.run_id)["kind"] == "replay"


def test_no_vehicle_ids_leak_into_evidence_or_brief(source, result):
    res, led = result
    vins = set(source.vehicles()["vehicle_id"].head(50))
    blob = json.dumps([e["outputs"] for e in led.evidence_for_run(res.run_id)]) + res.brief_md
    assert not any(v in blob for v in vins)


def test_event_level_operating_point_feeds_economics(result):
    res, _ = result
    ch = res.evidence["operating_point"].outputs["chosen"]
    assert ch["recall_basis"] == "event"
    assert ch["false_alerts_per_100_vehicle_months"] is not None
    roi_in = res.evidence["roi"].outputs["inputs"]
    assert (
        roi_in["false_alerts_per_100_vehicle_months"] == ch["false_alerts_per_100_vehicle_months"]
    )
    assert "ablation_daily_cadence" in res.evidence and "cost_daily_cadence" in res.evidence


def test_narrative_failure_does_not_fail_the_study(source, brake_spec, result):
    """A language-model outage while writing the optional narrative ships the brief headless
    and finishes the run; the REPORT step note says why."""
    res, _ = result
    led = _ledger()
    run_id = "narrative0503"
    led.create_run(run_id, brake_spec, source.dataset_hash, True, "predict brakes")

    def outage(ev, verdict):
        raise RuntimeError("HTTP Error 503: Service Unavailable")

    brief, md = Runner(source, led)._report(
        run_id, brake_spec, res.evidence, res.verdict, True, outage
    )
    assert brief.narrative == () and brief.llm_used is False
    assert "headless" in md
    run = led.get_run(run_id)
    assert run["status"] == "done" and run["error"] is None
    report = next(s for s in run["steps"] if s["name"] == "REPORT")
    assert report["status"] == "done" and "narrative skipped" in (report["note"] or "")

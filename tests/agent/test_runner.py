"""Integration: the headless default plan runs end to end on the small fixture and produces a
fully cited brief whose every gate and flag resolves to stored evidence. Determinism: two
runs on the same dataset/spec produce identical evidence outputs."""

from __future__ import annotations

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

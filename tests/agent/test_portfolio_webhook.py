from __future__ import annotations

import json
from types import SimpleNamespace

from motorq_de.agent import webhook
from motorq_de.agent.portfolio import portfolio
from motorq_de.ledger.store import Ledger, memory_engine
from motorq_de.schemas import (
    FlagResult,
    GateResult,
    ProblemSpec,
    Range,
    ValueAssumptions,
    Verdict,
)

R = lambda a, b, c: Range(low=a, base=b, high=c)  # noqa: E731


def _spec(name, target="brake_service_event"):
    return ProblemSpec(
        capability_name=name,
        target_event=target,
        value=ValueAssumptions(
            value_bearing_fraction=R(0.05, 0.1, 0.2),
            preventable_fraction=R(0.3, 0.4, 0.5),
            usd_per_avoided_event=R(1, 2, 3),
            fleet_size=R(1, 2, 3),
        ),
    )


def _verdict(decision):
    return Verdict(
        decision=decision,
        gates=(
            GateResult(
                name="data",
                passed=True,
                value=0.7,
                threshold=0.6,
                evidence_ids=("ev_aaaaaaaaaaaa",),
            ),
        ),
        flags=(
            FlagResult(
                name="value_unvalidated",
                tripped=decision == "PILOT",
                value=None,
                threshold=None,
                evidence_ids=(),
            ),
        ),
        policy_version="1.3",
    )


def _study(led, run_id, name, decision, suff, cost=100.0):
    led.create_run(run_id, _spec(name), "ds", False, None)
    led.record(
        run_id,
        "E",
        "ablation",
        {"r": run_id},
        "ds",
        1,
        lambda: {"sufficient_set": suff},
        name="ablation",
    )
    led.record(
        run_id,
        "E",
        "coverage_report",
        {"r": run_id},
        "ds",
        1,
        lambda: {"fleet_share_full_set": 0.7},
        name="coverage_sufficient",
    )
    led.record(
        run_id,
        "X",
        "run_cost",
        {"r": run_id},
        "ds",
        1,
        lambda: {
            "monthly": {"marginal_total": cost},
            "polling": {"highest_cadence_signals": [s for s in suff if s.startswith("rt_")]},
        },
        name="cost_sufficient",
    )
    led.record(
        run_id,
        "X",
        "roi_distribution",
        {"r": run_id},
        "ds",
        1,
        lambda: {"p_roi_positive": 0.8, "net_value_year": {"p50": 1000.0}},
        name="roi",
    )
    led.record(
        run_id,
        "X",
        "tornado",
        {"r": run_id},
        "ds",
        1,
        lambda: {"dominant_input": "usd_per_avoided_event"},
        name="tornado",
    )
    led.finish_run(run_id, _verdict(decision), "md", {})


def test_portfolio_rows_and_cogs():
    led = Ledger(memory_engine())
    _study(led, "r1", "brake", "PILOT", ["a", "b", "rt_x"])
    _study(led, "r2", "theft", "NOT_FEASIBLE", ["c", "d"])
    _study(led, "r3", "brake", "BUILD_READY", ["a", "e"], cost=50.0)  # newer brake study wins
    catalog = [
        {
            "signal_id": s,
            "category": "health" if s in "abe" else "location_trips",
            "declared_frequency": "realtime" if s.startswith("rt_") else "daily",
        }
        for s in ("a", "b", "c", "d", "e", "rt_x", "unused1")
    ]
    p = portfolio(led, catalog)
    caps = {r["capability_name"]: r for r in p["capabilities"]}
    assert set(caps) == {"brake", "theft"}
    assert caps["brake"]["run_id"] == "r3" and caps["brake"]["decision"] == "BUILD_READY"
    assert p["capabilities"][0]["capability_name"] == "brake"  # BUILD_READY sorts first
    cogs = p["cogs"]
    assert cogs["required"] == ["a", "e"]  # only viable capabilities count; theft is not feasible
    assert "unused1" in cogs["unused_signals"] and "c" in cogs["unused_signals"]
    assert cogs["realtime_signals_required"] == []
    assert "daily" in cogs["cadence_lever"]


def test_whatif_runs_are_not_capabilities():
    led = Ledger(memory_engine())
    _study(led, "r1", "brake", "PILOT", ["a"])
    led.create_run("w1", _spec("brake"), "ds", False, None, derived_from="r1", kind="whatif")
    led.finish_run("w1", _verdict("BUILD_READY"), "md", {})
    p = portfolio(led, [])
    assert [r["run_id"] for r in p["capabilities"]] == ["r1"]


def test_webhook_payload_and_post(monkeypatch):
    led = Ledger(memory_engine())
    _study(led, "r1", "brake", "PILOT", ["a", "b"])
    ev = led.evidence_objects("r1")
    res = SimpleNamespace(
        run_id="r1",
        verdict=_verdict("PILOT"),
        brief=SimpleNamespace(
            capability_name="brake", target_event="brake_service_event", horizon_days=7
        ),
        evidence=ev,
    )
    payload = webhook.summary_payload(res, "ds")
    assert "PILOT" in payload["text"] and ev["roi"].evidence_id in payload["text"]
    assert "vehicle_id" not in payload["text"]

    sent = {}

    class _Resp:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    def fake_urlopen(req, timeout):
        sent["url"] = req.full_url
        sent["body"] = json.loads(req.data)
        return _Resp()

    monkeypatch.setattr(webhook.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setenv("MDE_WEBHOOK_URL", "https://hooks.example/abc")
    assert webhook.notify_run(res, "ds") is True
    assert sent["url"] == "https://hooks.example/abc" and "brake" in sent["body"]["text"]
    monkeypatch.delenv("MDE_WEBHOOK_URL")
    assert webhook.notify_run(res, "ds") is False

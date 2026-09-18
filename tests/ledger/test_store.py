from __future__ import annotations

from motorq_de.ledger.store import Ledger, memory_engine
from motorq_de.schemas import ProblemSpec, Range, ValueAssumptions

R = lambda a, b, c: Range(low=a, base=b, high=c)  # noqa: E731


def _ledger():
    return Ledger(memory_engine())


def _spec():
    return ProblemSpec(
        capability_name="t",
        target_event="brake_service_event",
        value=ValueAssumptions(
            value_bearing_fraction=R(0.05, 0.1, 0.2),
            preventable_fraction=R(0.3, 0.4, 0.5),
            usd_per_avoided_event=R(1500, 3500, 6500),
            fleet_size=R(5000, 10000, 20000),
        ),
    )


def test_record_is_deterministic_in_id_and_outputs():
    led = _ledger()
    led.create_run("run1", _spec(), "ds", False, None)
    calls = {"n": 0}

    def fn():
        calls["n"] += 1
        return {"auc": 0.8765432109, "list": [1.00000001, 2.0], "nested": {"x": float("nan")}}

    a = led.record("run1", "S", "tool", {"signals": ["b", "a"]}, "ds", 42, fn)
    b = led.record("run1", "S", "tool", {"signals": ["b", "a"]}, "ds", 42, fn)
    assert a.evidence_id == b.evidence_id
    assert a.outputs == b.outputs
    assert a.outputs["auc"] == 0.876543  # rounded to 6 dp
    assert a.outputs["nested"]["x"] is None  # NaN canonicalised
    assert a.evidence_id.startswith("ev_") and len(a.evidence_id) == 15


def test_evidence_id_changes_with_inputs_seed_dataset():
    led = _ledger()
    led.create_run("run1", _spec(), "ds", False, None)
    base = led.record("run1", "S", "tool", {"k": 1}, "ds", 1, lambda: {"v": 1})
    assert (
        led.record("run1", "S", "tool", {"k": 2}, "ds", 1, lambda: {"v": 1}).evidence_id
        != base.evidence_id
    )
    assert (
        led.record("run1", "S", "tool", {"k": 1}, "ds", 2, lambda: {"v": 1}).evidence_id
        != base.evidence_id
    )
    assert (
        led.record("run1", "S", "tool", {"k": 1}, "ds2", 1, lambda: {"v": 1}).evidence_id
        != base.evidence_id
    )


def test_run_lifecycle_and_lookup():
    led = _ledger()
    led.create_run("run1", _spec(), "ds", False, "text")
    sid = led.start_step("run1", "DEFINE")
    e = led.record("run1", "DEFINE", "tool", {}, "ds", 1, lambda: {"v": 3})
    led.end_step(sid, note="ok")
    led.add_message("run1", "user", "why?", [e.evidence_id])
    led.finish_run("run1", None, "# brief", {"x": 1})
    run = led.get_run("run1")
    assert run["status"] == "done" and run["steps"][0]["note"] == "ok"
    assert run["evidence"][0]["evidence_id"] == e.evidence_id
    assert led.get_evidence(e.evidence_id)["outputs"] == {"v": 3}
    assert led.messages("run1")[0]["evidence_ids"] == [e.evidence_id]
    assert led.list_runs()[0]["run_id"] == "run1"


def test_failed_run_records_error():
    led = _ledger()
    led.create_run("run1", _spec(), "ds", False, None)
    led.finish_run("run1", None, None, None, error="boom")
    assert led.get_run("run1")["status"] == "failed"
    assert "boom" in led.get_run("run1")["error"]

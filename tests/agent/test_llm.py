"""LLM layer tests with a stub client: no network, no credentials. They verify the contract
that matters - the model cannot inject numbers, cannot change the verdict, and the parser
never invents value assumptions."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from motorq_de.agent import llm
from motorq_de.agent.llm import SpecDraft, answer, narrative, parse_spec
from motorq_de.ledger.store import Ledger, memory_engine
from motorq_de.schemas import (
    Evidence,
    FlagResult,
    GateResult,
    ProblemSpec,
    Range,
    ValueAssumptions,
    Verdict,
)

DEFAULTS = {
    "brake_service_event": {
        "value_bearing_fraction": {"low": 0.05, "base": 0.1, "high": 0.2},
        "preventable_fraction": {"low": 0.3, "base": 0.4, "high": 0.5},
        "usd_per_avoided_event": {"low": 1500, "base": 3500, "high": 6500},
        "fleet_size": {"low": 5000, "base": 10000, "high": 20000},
        "source_note": "public downtime studies",
    }
}


class _Msgs:
    def __init__(self, parsed=None, texts=None, tool_turns=None):
        self._parsed = parsed
        self._texts = list(texts or [])
        self._tool_turns = list(tool_turns or [])
        self.calls = []

    def parse(self, **kw):
        self.calls.append(kw)
        return SimpleNamespace(parsed_output=self._parsed)

    def create(self, **kw):
        self.calls.append(kw)
        if self._tool_turns:
            name, args = self._tool_turns.pop(0)
            block = SimpleNamespace(type="tool_use", id="tu1", name=name, input=args)
            return SimpleNamespace(stop_reason="tool_use", content=[block])
        return SimpleNamespace(
            stop_reason="end_turn", content=[SimpleNamespace(type="text", text=self._texts.pop(0))]
        )


def _stub(monkeypatch, msgs):
    monkeypatch.setattr(llm, "_client", lambda: SimpleNamespace(messages=msgs))


def test_parse_spec_defaults_value_when_not_supplied(monkeypatch):
    draft = SpecDraft(
        capability_name="brake_7d",
        target_event="brake_service_event",
        horizon_days=7,
        interpretation_notes="maintenance -> brake",
    )
    _stub(monkeypatch, _Msgs(parsed=draft))
    spec, notes = parse_spec("Should Fuse predict brake service a week out?", DEFAULTS)
    assert isinstance(spec, ProblemSpec) and spec.horizon_days == 7
    assert spec.value.source_note.startswith("DEFAULTS")
    assert "defaulted" in notes.lower()


def test_parse_spec_uses_supplied_value(monkeypatch):
    draft = SpecDraft(
        capability_name="x",
        target_event="theft_event",
        horizon_days=30,
        value={
            "value_bearing_fraction": {"low": 0.5, "base": 0.8, "high": 1.0},
            "preventable_fraction": {"low": 0.1, "base": 0.2, "high": 0.3},
            "usd_per_avoided_event": {"low": 8000, "base": 15000, "high": 30000},
            "fleet_size": {"low": 1000, "base": 2000, "high": 3000},
        },
    )
    _stub(monkeypatch, _Msgs(parsed=draft))
    spec, _ = parse_spec("...", DEFAULTS)
    assert spec.value.fleet_size.base == 2000 and spec.value.source_note == "supplied in request"


def test_parse_spec_repairs_once_then_fails(monkeypatch):
    bad = SpecDraft(
        capability_name="x",
        target_event="brake_service_event",
        horizon_days=7,
        value={
            "value_bearing_fraction": {"low": 0.9, "base": 0.1, "high": 0.2},  # invalid range
            "preventable_fraction": {"low": 0.3, "base": 0.4, "high": 0.5},
            "usd_per_avoided_event": {"low": 1, "base": 2, "high": 3},
            "fleet_size": {"low": 1, "base": 2, "high": 3},
        },
    )
    msgs = _Msgs(parsed=bad)
    _stub(monkeypatch, msgs)
    with pytest.raises(ValueError):
        parse_spec("...", DEFAULTS)
    assert len(msgs.calls) == 2 and "Previous attempt failed" in msgs.calls[1]["system"]


def _ev(name, eid, outputs):
    return Evidence(
        evidence_id=eid,
        run_id="r",
        step="S",
        tool=name,
        inputs={},
        inputs_hash="h",
        dataset_hash="d",
        seed=1,
        outputs=outputs,
        created_at=__import__("datetime").datetime(2026, 1, 1),
    )


def _verdict():
    return Verdict(
        decision="PILOT",
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
                name="value_unvalidated", tripped=True, value=None, threshold=None, evidence_ids=()
            ),
        ),
        policy_version="1.0",
    )


def test_narrative_drops_uncited_numbers(monkeypatch):
    ev = {
        "roi": _ev(
            "roi",
            "ev_aaaaaaaaaaaa",
            {"p_roi_positive": 0.4, "roi": {"p5": -1, "p50": 0.2, "p95": 1}, "net_value_year": {}},
        )
    }
    text = "The evidence supports a pilot. P(ROI>0) is 0.40 [ev_aaaaaaaaaaaa]. The model reaches AUC 0.99 easily. Coverage is limited."
    _stub(monkeypatch, _Msgs(texts=[text]))
    lines = narrative(ev, _verdict())
    assert [ln.text for ln in lines] == [
        "The evidence supports a pilot.",
        "P(ROI>0) is 0.40 [ev_aaaaaaaaaaaa].",
        "Coverage is limited.",
    ]


def test_answer_uses_tools_and_cites_only_known_evidence(monkeypatch):
    led = Ledger(memory_engine())
    spec = ProblemSpec(
        capability_name="t",
        target_event="brake_service_event",
        value=ValueAssumptions(
            value_bearing_fraction=Range(low=0.05, base=0.1, high=0.2),
            preventable_fraction=Range(low=0.3, base=0.4, high=0.5),
            usd_per_avoided_event=Range(low=1, base=2, high=3),
            fleet_size=Range(low=1, base=2, high=3),
        ),
    )
    led.create_run("run1", spec, "ds", True, "q")
    e = led.record(
        "run1",
        "EXPERIMENT",
        "ablation",
        {"k": 1},
        "ds",
        1,
        lambda: {"sufficient_set": ["a", "b"], "removed_in_order": ["gps_lat_mean"]},
    )
    led.finish_run("run1", _verdict(), "md", {})
    reply = f"GPS was removed in ablation because it added nothing [{e.evidence_id}]. It would have cost $9,999 more. It matters for theft."
    msgs = _Msgs(
        texts=[reply],
        tool_turns=[
            ("list_evidence", {}),
            ("get_evidence", {"evidence_id": e.evidence_id, "key": None}),
        ],
    )
    _stub(monkeypatch, msgs)
    text, cited = answer(led, "run1", "why was GPS removed?")
    assert cited == [e.evidence_id]
    assert "$9,999" not in text and "It matters for theft." in text
    # the tool results actually came from the ledger
    tool_result_msgs = [
        m
        for m in msgs.calls[-1]["messages"]
        if m["role"] == "user" and isinstance(m["content"], list)
    ]
    payload = json.loads(tool_result_msgs[-1]["content"][0]["content"])
    assert payload["sufficient_set"] == ["a", "b"]
    assert led.messages("run1")[-1]["evidence_ids"] == [e.evidence_id]


def test_llm_available_requires_credentials(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
    assert llm.llm_available() is False

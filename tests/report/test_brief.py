from __future__ import annotations

import pytest

from motorq_de.report.brief import Brief, Line, Section, UncitedClaimError, sanitize_narrative
from motorq_de.report.render import to_json, to_markdown


def _brief(lines):
    return Brief(
        run_id="r1",
        dataset_hash="d1",
        capability_name="x",
        target_event="brake_service_event",
        horizon_days=7,
        decision="PILOT",
        policy_version="1.0",
        sections=(Section(title="S", lines=tuple(lines)),),
        generated_at="2026-09-19T00:00:00Z",
    )


def test_renderer_fails_closed_on_uncited_number():
    with pytest.raises(UncitedClaimError):
        to_markdown(_brief([Line("AUC is 0.87")]))
    with pytest.raises(UncitedClaimError):
        to_json(_brief([Line("costs $3,500")]))


def test_renderer_accepts_cited_numbers_and_uncited_prose():
    md = to_markdown(_brief([Line("AUC is 0.87", ("ev_0123456789ab",)), Line("No numbers here")]))
    assert "`[ev_0123456789ab]`" in md
    assert "No numbers here" in md


def test_sanitize_narrative_drops_uncited_numeric_sentences():
    known = {"ev_0123456789ab"}
    text = "The model works well. AUC reached 0.87 [ev_0123456789ab]. It costs about $500 a month. Coverage is broad."
    kept, dropped = sanitize_narrative(text, known)
    assert [k.text for k in kept] == [
        "The model works well.",
        "AUC reached 0.87 [ev_0123456789ab].",
        "Coverage is broad.",
    ]
    assert dropped == ["It costs about $500 a month."]
    assert kept[1].evidence_ids == ("ev_0123456789ab",)


def test_sanitize_narrative_rejects_unknown_citations():
    kept, dropped = sanitize_narrative("AUC 0.9 [ev_ffffffffffff].", {"ev_0123456789ab"})
    assert kept == [] and len(dropped) == 1

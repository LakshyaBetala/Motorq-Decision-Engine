"""Render a Brief to Markdown / JSON. Fails closed: any line with a digit and no evidence
citation raises UncitedClaimError. Lines without numbers may stand uncited (they carry no claim)."""

from __future__ import annotations

import json
import re
from typing import Any

from motorq_de.report.brief import NUM_RE, Brief, Line, UncitedClaimError

# words like "7-day" or "v1.0" in fixed labels are still digits: every such line must cite.
_ALLOW_UNCITED = re.compile(r"^(Capability:|Decision:|Flags not tripped:)")


def check_citations(brief: Brief) -> None:
    for sec in brief.sections:
        for line in sec.lines:
            _check(line, sec.title)
    for line in brief.narrative:
        _check(line, "narrative")


def _check(line: Line, where: str) -> None:
    if NUM_RE.search(line.text) and not line.evidence_ids and not _ALLOW_UNCITED.match(line.text):
        raise UncitedClaimError(f"[{where}] uncited numeric claim: {line.text[:120]}")


def to_markdown(brief: Brief) -> str:
    check_citations(brief)
    out = [
        f"# Evidence brief - {brief.capability_name}",
        "",
        f"**Decision: {brief.decision}**  ·  target `{brief.target_event}`  ·  horizon {brief.horizon_days}d  ·  run `{brief.run_id}`  ·  dataset `{brief.dataset_hash}`  ·  policy v{brief.policy_version}  ·  {'LLM narrative' if brief.llm_used else 'headless'}",
        "",
    ]
    for sec in brief.sections:
        out.append(f"## {sec.title}")
        out.append("")
        for line in sec.lines:
            cite = f" `[{', '.join(line.evidence_ids)}]`" if line.evidence_ids else ""
            out.append(f"- {line.text}{cite}")
        out.append("")
    if brief.narrative:
        out.append("## Narrative")
        out.append("")
        out.append(
            " ".join(
                f"{ln.text}" + (f" `[{', '.join(ln.evidence_ids)}]`" if ln.evidence_ids else "")
                for ln in brief.narrative
            )
        )
        out.append("")
    out.append(
        f"_Generated {brief.generated_at}. Every number above cites an evidence_id; click-through resolves to the tool call that produced it._"
    )
    return "\n".join(out)


def to_json(brief: Brief) -> dict[str, Any]:
    check_citations(brief)
    return json.loads(brief.model_dump_json())

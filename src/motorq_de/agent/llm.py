"""The LLM interface - the only module that talks to a model provider.

Providers: `anthropic` (default), `bedrock` (Claude on AWS), `gemini` (Google, REST via urllib).
Three jobs, all optional (the engine runs headless without credentials):

    parse_spec   free text -> ProblemSpec (structured output, one repair attempt)
    narrative    short prose over the evidence; every numeric sentence must cite an
                 evidence id or it is dropped by `sanitize_narrative`
    answer       Q&A over a finished run using two tools that read the ledger

The LLM never computes a number and never touches the verdict.
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from typing import Any, Literal

from pydantic import BaseModel, ValidationError

from motorq_de.ledger.store import Ledger
from motorq_de.report.brief import Line, sanitize_narrative
from motorq_de.schemas import Evidence, ProblemSpec, Range, ValueAssumptions, Verdict

DEFAULT_MODEL = "claude-opus-5"
GEMINI_DEFAULT_MODEL = "gemini-3.6-flash"
GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"


def model_id() -> str:
    if provider() == "gemini":
        return os.environ.get("MDE_LLM_MODEL", GEMINI_DEFAULT_MODEL)
    m = os.environ.get("MDE_LLM_MODEL", DEFAULT_MODEL)
    if provider() == "bedrock" and not m.startswith("anthropic."):
        m = "anthropic." + m  # Bedrock model ids carry the vendor prefix
    return m


def provider() -> str:
    """anthropic (default), bedrock (AWS credentials + AWS_REGION) or gemini (GEMINI_API_KEY)."""
    return os.environ.get("MDE_LLM_PROVIDER", "anthropic").lower()


def llm_available() -> bool:
    if provider() == "gemini":
        return bool(os.environ.get("GEMINI_API_KEY"))
    try:
        import anthropic  # noqa: F401
    except ImportError:
        return False
    if provider() == "bedrock":
        return bool(os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION"))
    return bool(os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN"))


# --------------------------------------------------------------------------- gemini transport


def _gemini_call(
    system: str,
    contents: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None = None,
    json_schema: dict[str, Any] | None = None,
    max_tokens: int = 8192,
) -> dict[str, Any]:
    """One generateContent request. Returns the first candidate's content (its parts).

    Gemini 3 counts its thinking against maxOutputTokens, so thinking is kept low and the
    budget generous; a MAX_TOKENS finish is surfaced instead of returning a truncated tail."""
    payload: dict[str, Any] = {
        "systemInstruction": {"parts": [{"text": system}]},
        "contents": contents,
        "generationConfig": {
            "maxOutputTokens": max_tokens,
            "temperature": 0,
            "thinkingConfig": {"thinkingLevel": os.environ.get("MDE_GEMINI_THINKING", "low")},
        },
    }
    if json_schema is not None:
        payload["generationConfig"]["responseMimeType"] = "application/json"
        payload["generationConfig"]["responseJsonSchema"] = json_schema
    if tools:
        payload["tools"] = [{"functionDeclarations": tools}]
    req = urllib.request.Request(
        GEMINI_URL.format(model=model_id()),
        data=json.dumps(payload).encode(),
        headers={
            "content-type": "application/json",
            "x-goog-api-key": os.environ.get("GEMINI_API_KEY", ""),
        },
    )
    body = _http_json(req)
    cands = body.get("candidates") or []
    if not cands:
        raise RuntimeError(f"gemini returned no candidates: {json.dumps(body)[:300]}")
    if cands[0].get("finishReason") == "MAX_TOKENS":
        raise RuntimeError("gemini hit maxOutputTokens before finishing; raise max_tokens")
    return cands[0].get("content") or {"role": "model", "parts": []}


GEMINI_RETRY_STATUS = {429, 500, 503}


def _http_json(req: urllib.request.Request, attempts: int = 5) -> dict[str, Any]:
    """POST and decode; retries transient statuses. 429 (free-tier requests-per-minute) backs
    off 8/16/32/64 s; 5xx backs off 2/4/8/16 s."""
    for i in range(attempts):
        try:
            with urllib.request.urlopen(req, timeout=120) as r:  # noqa: S310
                return json.loads(r.read())
        except urllib.error.HTTPError as exc:
            if exc.code not in GEMINI_RETRY_STATUS or i == attempts - 1:
                raise
            time.sleep((8 if exc.code == 429 else 2) * 2**i)
    raise RuntimeError("unreachable")


def _gemini_text(content: dict[str, Any]) -> str:
    return "".join(p.get("text", "") for p in content.get("parts", []) if "text" in p)


def _client():
    import anthropic

    if provider() == "bedrock":
        region = os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION")
        return anthropic.AnthropicBedrockMantle(aws_region=region)
    return anthropic.Anthropic()


# --------------------------------------------------------------------------- spec parsing


class RangeDraft(BaseModel):
    low: float
    base: float
    high: float


class ValueDraft(BaseModel):
    value_bearing_fraction: RangeDraft
    preventable_fraction: RangeDraft
    usd_per_avoided_event: RangeDraft
    fleet_size: RangeDraft
    source_note: str = ""


class SpecDraft(BaseModel):
    """What the model extracts from a request. Value assumptions are only filled when the
    request states them; otherwise the engine substitutes documented defaults and says so."""

    capability_name: str
    target_event: Literal["brake_service_event", "theft_event", "battery_degradation_event"]
    horizon_days: int
    delivery_mode: Literal["batch_daily", "batch_hourly", "streaming"] = "batch_daily"
    consumer: Literal["fuse_action_hub", "fuse_assistant", "api", "internal"] = "fuse_action_hub"
    # Constraints stay None unless the request states them; the engine's defaults apply.
    # A model that "helpfully" tightens coverage or latency would otherwise move the verdict.
    min_oem_coverage: float | None = None
    max_latency_s: int | None = None
    value: ValueDraft | None = None
    interpretation_notes: str = ""


PARSE_SYSTEM = """You turn a product request about a connected-vehicle capability into a structured
ProblemSpec for a feasibility engine. Choose the closest supported target_event. Set horizon_days
from the request (default 7 for maintenance, 30 for theft, 14 for battery). Only fill `value`
if the request states business numbers; otherwise leave it null. Leave min_oem_coverage and
max_latency_s null unless the request states a coverage or latency requirement. Put any
judgement calls in interpretation_notes. Do not invent numbers."""


def _draft(text: str, last_err: str) -> SpecDraft:
    system = PARSE_SYSTEM + (
        f"\n\nPrevious attempt failed validation: {last_err}" if last_err else ""
    )
    if provider() == "gemini":
        content = _gemini_call(
            system,
            [{"role": "user", "parts": [{"text": text}]}],
            json_schema=SpecDraft.model_json_schema(),
        )
        return SpecDraft.model_validate_json(_gemini_text(content))
    resp = _client().messages.parse(
        model=model_id(),
        max_tokens=4096,
        system=system,
        messages=[{"role": "user", "content": text}],
        output_format=SpecDraft,
    )
    return resp.parsed_output


def parse_spec(text: str, defaults: dict[str, dict[str, Any]]) -> tuple[ProblemSpec, str]:
    """Free text -> validated ProblemSpec. `defaults` maps target_event -> value dict used when
    the request supplies no business numbers. Returns (spec, notes)."""
    last_err = ""
    for _attempt in range(2):
        try:
            draft = _draft(text, last_err)
            if draft.value is not None:
                value = ValueAssumptions(
                    value_bearing_fraction=Range(**draft.value.value_bearing_fraction.model_dump()),
                    preventable_fraction=Range(**draft.value.preventable_fraction.model_dump()),
                    usd_per_avoided_event=Range(**draft.value.usd_per_avoided_event.model_dump()),
                    fleet_size=Range(**draft.value.fleet_size.model_dump()),
                    source_note=draft.value.source_note or "supplied in request",
                )
                notes = draft.interpretation_notes
            else:
                d = defaults[draft.target_event]
                value = ValueAssumptions.model_validate(
                    {
                        **d,
                        "source_note": "DEFAULTS - not supplied in the request; "
                        + d.get("source_note", ""),
                    }
                )
                notes = (
                    draft.interpretation_notes + " " if draft.interpretation_notes else ""
                ) + "Value assumptions defaulted."
            spec = ProblemSpec(
                capability_name=draft.capability_name,
                target_event=draft.target_event,
                horizon_days=draft.horizon_days,
                delivery_mode=draft.delivery_mode,
                consumer=draft.consumer,
                value=value,
                constraints={
                    k: v
                    for k, v in {
                        "min_oem_coverage": draft.min_oem_coverage,
                        "max_latency_s": draft.max_latency_s,
                    }.items()
                    if v is not None
                },
            )
            return spec, notes.strip()
        except (ValidationError, KeyError) as exc:
            last_err = str(exc)[:500]
    raise ValueError(f"could not produce a valid ProblemSpec: {last_err}")


# --------------------------------------------------------------------------- narrative


def _digest(ev: dict[str, Evidence], verdict: Verdict) -> str:
    """Compact, citable summary of the evidence for the model. Each entry carries its id."""
    keep = {
        "event_rate": ["rate_per_vehicle_year", "rate_ci_lo", "rate_ci_hi", "n_events"],
        "usable": ["n_in", "n_usable", "dropped"],
        "leakage": ["flagged", "suspicious"],
        "coverage_sufficient": ["fleet_share_full_set", "oems_missing_any_signal"],
        "feature_analysis": ["auc"],
        "ablation": [
            "sufficient_set",
            "full_auc",
            "sufficient_auc",
            "underpowered",
            "removed_in_order",
        ],
        "model_comparison": ["paired"],
        "redundancy": ["n_pairs_redundant", "n_independent_groups", "clusters"],
        "temporal": ["cv_auc", "forward_auc", "degradation", "degradation_upper"],
        "cross_oem": ["mean_auc", "std_auc", "min_auc", "worst_oem"],
        "cross_oem_gap": ["oem", "missing_signals"],
        "cost_sufficient": ["monthly", "lever_note", "placeholders"],
        "cost_compare": [
            "gb_per_month_full",
            "gb_per_month_reduced",
            "infra_change_pct",
            "marginal_change_pct",
        ],
        "operating_point": ["chosen"],
        "roi": ["p_roi_positive", "roi", "net_value_year"],
        "tornado": ["dominant_input", "ranked"],
        "deployment": ["recommended_pattern", "note"],
    }
    out: dict[str, Any] = {}
    for name, fields in keep.items():
        if name in ev:
            o = ev[name].outputs
            out[name] = {
                "evidence_id": ev[name].evidence_id,
                **{f: o.get(f) for f in fields if f in o},
            }
    if "feature_analysis" in ev:
        out["feature_analysis"]["top_signals"] = [
            r["signal"] for r in ev["feature_analysis"].outputs["ranking"][:6]
        ]
    out["verdict"] = json.loads(verdict.model_dump_json())
    return json.dumps(out, default=str)


NARRATIVE_SYSTEM = """You write the narrative section of an internal feasibility brief for Motorq's
product and engineering leads. You are given evidence as JSON; each block has an evidence_id.
Write 5-8 plain sentences: what the evidence supports, the strongest reasons, the biggest
uncertainties, and what a pilot would need to resolve. Every sentence that contains a number
MUST end with a citation like [ev_0123456789ab] using the evidence_id of the block it came from.
Never introduce a number that is not in the evidence. Do not restate the verdict as your own
decision; it is computed by policy and the build decision is a human call."""


def narrative(ev: dict[str, Evidence], verdict: Verdict) -> list[Line]:
    if provider() == "gemini":
        content = _gemini_call(
            NARRATIVE_SYSTEM
            + "\n\nOutput only the final sentences: no headings, no working notes.",
            [{"role": "user", "parts": [{"text": _digest(ev, verdict)}]}],
            max_tokens=16384,
        )
        text = _gemini_text(content)
    else:
        resp = _client().messages.create(
            model=model_id(),
            max_tokens=2048,
            system=NARRATIVE_SYSTEM,
            messages=[{"role": "user", "content": _digest(ev, verdict)}],
        )
        text = "".join(b.text for b in resp.content if b.type == "text")
    known = {e.evidence_id for e in ev.values()}
    kept, _dropped = sanitize_narrative(text, known)
    return kept


# --------------------------------------------------------------------------- Q&A


QA_SYSTEM = """You answer questions about a finished feasibility run. Use the tools to read the
stored evidence; answer only from it. Every sentence containing a number must cite the
evidence_id it came from as [ev_...]. If the evidence does not contain the answer, say so.
Keep answers under 200 words."""

QA_TOOLS = [
    {
        "name": "list_evidence",
        "description": "List the evidence records of the run: id, step, tool, and the top-level keys of the outputs.",
        "input_schema": {"type": "object", "properties": {}, "additionalProperties": False},
        "strict": True,
    },
    {
        "name": "get_evidence",
        "description": "Return one evidence record's full outputs by evidence_id. Optionally restrict to one top-level key.",
        "input_schema": {
            "type": "object",
            "properties": {"evidence_id": {"type": "string"}, "key": {"type": ["string", "null"]}},
            "required": ["evidence_id", "key"],
            "additionalProperties": False,
        },
        "strict": True,
    },
]


# Gemini function declarations use the OpenAPI subset: no type unions, no additionalProperties.
QA_TOOLS_GEMINI = [
    {
        "name": "list_evidence",
        "description": QA_TOOLS[0]["description"],
        "parameters": {"type": "object", "properties": {}},
    },
    {
        "name": "get_evidence",
        "description": QA_TOOLS[1]["description"],
        "parameters": {
            "type": "object",
            "properties": {"evidence_id": {"type": "string"}, "key": {"type": "string"}},
            "required": ["evidence_id"],
        },
    },
]


def _answer_gemini(ledger: Ledger, run: dict[str, Any], context: str, max_turns: int) -> str:
    contents: list[dict[str, Any]] = [{"role": "user", "parts": [{"text": context}]}]
    text = ""
    for _ in range(max_turns):
        content = _gemini_call(QA_SYSTEM, contents, tools=QA_TOOLS_GEMINI)
        text = _gemini_text(content)
        calls = [p["functionCall"] for p in content.get("parts", []) if "functionCall" in p]
        if not calls:
            break
        contents.append(content)  # echoed verbatim so any thought signatures survive
        contents.append(
            {
                "role": "user",
                "parts": [
                    {
                        "functionResponse": {
                            "name": c["name"],
                            "response": {
                                "result": _run_tool(ledger, run, c["name"], c.get("args") or {})
                            },
                        }
                    }
                    for c in calls
                ],
            }
        )
    return text


def answer(ledger: Ledger, run_id: str, question: str, max_turns: int = 6) -> tuple[str, list[str]]:
    run = ledger.get_run(run_id)
    if run is None:
        raise KeyError(run_id)
    known = {e["evidence_id"] for e in run["evidence"]}
    context = f"Run {run_id}: capability {run['spec'].get('capability_name')}, target {run['spec'].get('target_event')}, decision {(run['verdict'] or {}).get('decision')}.\n\nQuestion: {question}"
    if provider() == "gemini":
        text = _answer_gemini(ledger, run, context, max_turns)
        return _finish_answer(ledger, run_id, question, text, known)
    client = _client()
    messages: list[dict[str, Any]] = [{"role": "user", "content": context}]
    text = ""
    for _ in range(max_turns):
        resp = client.messages.create(
            model=model_id(), max_tokens=4096, system=QA_SYSTEM, tools=QA_TOOLS, messages=messages
        )
        if resp.stop_reason == "refusal":
            return "The model declined to answer this question.", []
        text = "".join(b.text for b in resp.content if b.type == "text")
        if resp.stop_reason != "tool_use":
            break
        messages.append({"role": "assistant", "content": resp.content})
        results = []
        for block in resp.content:
            if block.type != "tool_use":
                continue
            args = block.input if isinstance(block.input, dict) else json.loads(block.input)
            results.append(
                {
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": _run_tool(ledger, run, block.name, args),
                }
            )
        messages.append({"role": "user", "content": results})
    return _finish_answer(ledger, run_id, question, text, known)


def _finish_answer(
    ledger: Ledger, run_id: str, question: str, text: str, known: set[str]
) -> tuple[str, list[str]]:
    """The same guardrail for every provider: keep only sentences whose citations exist."""
    kept, _ = sanitize_narrative(text, known)
    cited = sorted({i for ln in kept for i in ln.evidence_ids})
    clean = (
        " ".join(ln.text for ln in kept) or "No citable answer could be produced from the evidence."
    )
    ledger.add_message(run_id, "user", question, [])
    ledger.add_message(run_id, "assistant", clean, cited)
    return clean, cited


def _run_tool(ledger: Ledger, run: dict[str, Any], name: str, args: dict[str, Any]) -> str:
    if name == "list_evidence":
        rows = []
        for e in run["evidence"]:
            full = ledger.get_evidence(e["evidence_id"]) or {}
            rows.append(
                {
                    "evidence_id": e["evidence_id"],
                    "step": e["step"],
                    "tool": e["tool"],
                    "keys": sorted((full.get("outputs") or {}).keys())[:25],
                }
            )
        return json.dumps(rows)
    if name == "get_evidence":
        e = ledger.get_evidence(str(args.get("evidence_id", "")))
        if e is None or e["run_id"] != run["run_id"]:
            return json.dumps({"error": "unknown evidence_id for this run"})
        out = e["outputs"]
        key = args.get("key")
        if key:
            out = out.get(key, {"error": f"no key {key}"})
        s = json.dumps(out, default=str)
        return s if len(s) < 60_000 else s[:60_000] + " ...[truncated]"
    return json.dumps({"error": f"unknown tool {name}"})

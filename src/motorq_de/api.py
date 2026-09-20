"""FastAPI surface over the Decision Engine.

GET  /health
GET  /datasets
GET  /runs                      list runs
POST /runs                      {spec} | {text} (+ use_llm) -> ticket
GET  /tickets/{ticket}          queue status -> run_id when done
GET  /runs/{run_id}             status, steps, verdict, evidence index
GET  /runs/{run_id}/brief       markdown + json
GET  /runs/{run_id}/evidence    all evidence records
GET  /runs/{run_id}/evidence/{evidence_id}
POST /runs/{run_id}/ask         {question} -> cited answer (LLM)
GET  /runs/{run_id}/messages

Access: when MDE_API_KEY is set every route except /health requires
`Authorization: Bearer <key>` (or `X-API-Key: <key>`); the dashboard's server-side proxy adds
it, so the key never reaches a browser. Unset, the API is open, for local use only. Network
placement (VPC, SPCS service, SSO in front of the dashboard) is the real control; this is the
belt under it. MDE_CORS_ORIGINS is a comma-separated allow-list (default: any origin).
"""

from __future__ import annotations

import asyncio
import hmac
import json
import os
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, PlainTextResponse, StreamingResponse
from pydantic import BaseModel

from motorq_de.agent.service import EXAMPLE_SPECS, Service
from motorq_de.schemas import ProblemSpec

app = FastAPI(title="Motorq Decision Engine", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in os.environ.get("MDE_CORS_ORIGINS", "*").split(",")],
    allow_methods=["*"],
    allow_headers=["*"],
)
_service: Service | None = None

OPEN_PATHS = {"/health", "/docs", "/openapi.json", "/redoc"}


def api_key() -> str | None:
    return os.environ.get("MDE_API_KEY") or None


def presented_key(request: Request) -> str | None:
    auth = request.headers.get("authorization", "")
    if auth.lower().startswith("bearer "):
        return auth[7:].strip()
    return request.headers.get("x-api-key")


@app.middleware("http")
async def require_key(request: Request, call_next):
    key = api_key()
    if key and request.url.path not in OPEN_PATHS and request.method != "OPTIONS":
        given = presented_key(request) or ""
        if not hmac.compare_digest(given.encode(), key.encode()):
            return JSONResponse({"detail": "missing or invalid API key"}, status_code=401)
    return await call_next(request)


def service() -> Service:
    global _service
    if _service is None:
        _service = Service()
    return _service


class RunRequest(BaseModel):
    spec: ProblemSpec | None = None
    text: str | None = None
    example: str | None = None
    use_llm: bool = False


class AskRequest(BaseModel):
    question: str


class WhatIfRequest(BaseModel):
    value: dict[str, Any] | None = None
    price_overrides: dict[str, Any] | None = None
    constraints: dict[str, Any] | None = None


@app.get("/health")
def health() -> dict[str, Any]:
    s = service()
    return {
        "ok": True,
        "dataset_hash": s.source.dataset_hash,
        "llm_available": s.llm_available(),
        "auth_required": api_key() is not None,
    }


@app.get("/datasets")
def datasets() -> list[dict[str, Any]]:
    return Service.datasets()


@app.get("/examples")
def examples() -> dict[str, Any]:
    return EXAMPLE_SPECS


@app.get("/runs")
def runs() -> list[dict[str, Any]]:
    return service().ledger.list_runs()


@app.post("/runs")
def create_run(req: RunRequest) -> dict[str, Any]:
    s = service()
    notes = ""
    if req.spec is not None:
        spec = req.spec
    elif req.example:
        if req.example not in EXAMPLE_SPECS:
            raise HTTPException(400, f"unknown example {req.example}")
        spec = ProblemSpec.model_validate(EXAMPLE_SPECS[req.example])
    elif req.text:
        if not s.llm_available():
            raise HTTPException(
                400,
                "free-text requests need the LLM interface; set ANTHROPIC_API_KEY or send a structured spec",
            )
        spec, notes = s.spec_from_text(req.text)
    else:
        raise HTTPException(400, "send spec, example or text")
    ticket = s.submit(spec, req.text, req.use_llm)
    return {"ticket": ticket, "spec": spec.model_dump(), "notes": notes}


@app.get("/tickets/{ticket}")
def ticket(ticket: str) -> dict[str, Any]:
    return service().ticket_status(ticket)


@app.get("/runs/{run_id}")
def get_run(run_id: str) -> dict[str, Any]:
    run = service().ledger.get_run(run_id)
    if run is None:
        raise HTTPException(404, "run not found")
    run.pop("brief_md", None)
    run.pop("brief_json", None)
    return run


@app.get("/runs/{run_id}/brief")
def brief(run_id: str) -> dict[str, Any]:
    run = service().ledger.get_run(run_id)
    if run is None:
        raise HTTPException(404, "run not found")
    return {
        "run_id": run_id,
        "decision": (run["verdict"] or {}).get("decision"),
        "markdown": run["brief_md"],
        "json": run["brief_json"],
    }


@app.get("/runs/{run_id}/evidence")
def evidence(run_id: str) -> list[dict[str, Any]]:
    return service().ledger.evidence_for_run(run_id)


@app.get("/runs/{run_id}/evidence/{evidence_id}")
def evidence_one(run_id: str, evidence_id: str) -> dict[str, Any]:
    e = service().ledger.get_evidence(evidence_id)
    if e is None or e["run_id"] != run_id:
        raise HTTPException(404, "evidence not found")
    return e


@app.post("/runs/{run_id}/ask")
def ask(run_id: str, req: AskRequest) -> dict[str, Any]:
    s = service()
    if not s.llm_available():
        raise HTTPException(400, "Q&A needs the LLM interface; set ANTHROPIC_API_KEY")
    if s.ledger.get_run(run_id) is None:
        raise HTTPException(404, "run not found")
    text, cited = s.ask(run_id, req.question)
    return {"answer": text, "evidence_ids": cited}


@app.get("/runs/{run_id}/messages")
def messages(run_id: str) -> list[dict[str, Any]]:
    return service().ledger.messages(run_id)


@app.post("/runs/{run_id}/whatif")
def whatif(run_id: str, req: WhatIfRequest) -> dict[str, Any]:
    """Re-run economics -> policy -> brief with new human inputs; seconds, reuses harness evidence."""
    s = service()
    if s.ledger.get_run(run_id) is None:
        raise HTTPException(404, "run not found")
    try:
        res = s.whatif(run_id, req.value, req.price_overrides, req.constraints)
    except (KeyError, ValueError) as exc:
        raise HTTPException(400, str(exc)) from exc
    return {"run_id": res.run_id, "decision": res.verdict.decision, "derived_from": run_id}


@app.post("/runs/{run_id}/replay")
def replay(run_id: str) -> dict[str, Any]:
    """Queue a full replay of a study; the new run stores an evidence-by-evidence diff."""
    s = service()
    if s.ledger.get_run(run_id) is None:
        raise HTTPException(404, "run not found")
    ticket = "t_" + __import__("uuid").uuid4().hex[:10]
    fut = s._pool.submit(s.replay, run_id)
    with s._lock:
        s._futures[ticket] = fut
    return {"ticket": ticket, "replay_of": run_id}


@app.get("/runs/{run_id}/brief.md", response_class=PlainTextResponse)
def brief_markdown(run_id: str) -> str:
    run = service().ledger.get_run(run_id)
    if run is None or not run["brief_md"]:
        raise HTTPException(404, "brief not found")
    return run["brief_md"]


@app.get("/runs/{run_id}/events")
async def events(run_id: str):
    """Server-sent events with step progress until the run finishes."""
    s = service()

    async def gen():
        last = None
        while True:
            run = s.ledger.get_run(run_id)
            if run is None:
                yield "event: error\ndata: {}\n\n"
                return
            snap = json.dumps(
                {
                    "status": run["status"],
                    "steps": run["steps"],
                    "n_evidence": len(run["evidence"]),
                    "decision": (run["verdict"] or {}).get("decision"),
                }
            )
            if snap != last:
                yield f"data: {snap}\n\n"
                last = snap
            if run["status"] in ("done", "failed"):
                return
            await asyncio.sleep(2)

    return StreamingResponse(gen(), media_type="text/event-stream")


@app.get("/portfolio")
def portfolio_view() -> dict[str, Any]:
    return service().portfolio()

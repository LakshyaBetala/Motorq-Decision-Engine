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
"""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from motorq_de.agent.service import EXAMPLE_SPECS, Service
from motorq_de.schemas import ProblemSpec

app = FastAPI(title="Motorq Decision Engine", version="0.1.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
_service: Service | None = None


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


@app.get("/health")
def health() -> dict[str, Any]:
    s = service()
    return {"ok": True, "dataset_hash": s.source.dataset_hash, "llm_available": s.llm_available()}


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

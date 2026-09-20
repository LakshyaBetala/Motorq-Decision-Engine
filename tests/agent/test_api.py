"""API contract tests with the small fixture; runs are not executed here (that is the slow
integration test) - we verify routing, validation and ledger read paths."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from motorq_de import api
from motorq_de.agent.service import Service
from motorq_de.ledger.store import Ledger, memory_engine
from motorq_de.schemas import ProblemSpec, Range, ValueAssumptions


@pytest.fixture
def client(source, monkeypatch):
    svc = Service(source=source, ledger=Ledger(memory_engine()))
    monkeypatch.setattr(api, "_service", svc)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
    return TestClient(api.app), svc


def test_health_and_examples(client):
    c, svc = client
    h = c.get("/health").json()
    assert h["ok"] and h["dataset_hash"] == svc.source.dataset_hash and h["llm_available"] is False
    assert set(c.get("/examples").json()) == {"brake", "theft", "battery"}


def test_create_run_validates_input(client):
    c, _ = client
    assert c.post("/runs", json={}).status_code == 400
    assert c.post("/runs", json={"example": "nope"}).status_code == 400
    r = c.post("/runs", json={"text": "predict brakes"})
    assert r.status_code == 400 and "LLM" in r.json()["detail"]


def test_ledger_read_paths(client):
    c, svc = client
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
    svc.ledger.create_run("r1", spec, "ds", False, None)
    e = svc.ledger.record("r1", "S", "tool", {}, "ds", 1, lambda: {"v": 1})
    svc.ledger.finish_run("r1", None, "# brief", {"x": 1})
    assert c.get("/runs").json()[0]["run_id"] == "r1"
    assert c.get("/runs/r1").json()["status"] == "done"
    assert c.get("/runs/r1/brief").json()["markdown"] == "# brief"
    assert c.get(f"/runs/r1/evidence/{e.evidence_id}").json()["outputs"] == {"v": 1}
    assert c.get("/runs/r1/evidence/ev_000000000000").status_code == 404
    assert c.get("/runs/nope").status_code == 404
    assert c.post("/runs/r1/ask", json={"question": "why?"}).status_code == 400


def test_new_endpoints_validate(client):
    c, _ = client
    assert c.post("/runs/nope/whatif", json={}).status_code == 404
    assert c.post("/runs/nope/replay").status_code == 404
    assert c.get("/runs/nope/brief.md").status_code == 404
    p = c.get("/portfolio").json()
    assert p["n_capabilities"] == 0 and p["cogs"]["n_catalog"] >= 85


def test_api_key_guards_every_route_but_health(client, monkeypatch):
    c, _ = client
    monkeypatch.setenv("MDE_API_KEY", "s3cret")
    assert c.get("/health").status_code == 200
    assert c.get("/health").json()["auth_required"] is True
    assert c.get("/runs").status_code == 401
    assert c.get("/runs", headers={"authorization": "Bearer wrong"}).status_code == 401
    assert c.get("/runs", headers={"authorization": "Bearer s3cret"}).status_code == 200
    assert c.get("/runs", headers={"x-api-key": "s3cret"}).status_code == 200
    assert c.post("/runs", json={}, headers={"authorization": "Bearer s3cret"}).status_code == 400


def test_api_is_open_when_no_key_is_configured(client, monkeypatch):
    c, _ = client
    monkeypatch.delenv("MDE_API_KEY", raising=False)
    assert c.get("/runs").status_code == 200
    assert c.get("/health").json()["auth_required"] is False

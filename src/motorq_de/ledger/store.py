"""Run ledger: runs, steps, tool calls, evidence, messages.

Postgres via DATABASE_URL; SQLite file fallback for local work. `record()` is the one way a
tool result becomes Evidence: it hashes the inputs, mints the evidence_id, stores the
rounded outputs, and returns the Evidence object.
"""

from __future__ import annotations

import json
import os
import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from sqlalchemy import JSON, Float, ForeignKey, Integer, String, Text, create_engine, select
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column
from sqlalchemy.pool import StaticPool

from motorq_de.hashing import canonical_json, evidence_id, hash_inputs, round_floats
from motorq_de.schemas import Evidence, ProblemSpec, Verdict


class Base(DeclarativeBase):
    pass


class RunRow(Base):
    __tablename__ = "runs"
    run_id: Mapped[str] = mapped_column(String(32), primary_key=True)
    spec_json: Mapped[dict] = mapped_column(JSON)
    dataset_hash: Mapped[str] = mapped_column(String(64))
    status: Mapped[str] = mapped_column(String(16), default="running")
    created_at: Mapped[str] = mapped_column(String(32))
    finished_at: Mapped[str | None] = mapped_column(String(32), nullable=True)
    verdict_json: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    brief_md: Mapped[str | None] = mapped_column(Text, nullable=True)
    brief_json: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    llm_used: Mapped[int] = mapped_column(Integer, default=0)
    request_text: Mapped[str | None] = mapped_column(Text, nullable=True)
    derived_from: Mapped[str | None] = mapped_column(
        String(32), nullable=True
    )  # what-if / replay parent
    kind: Mapped[str] = mapped_column(String(16), default="study")  # study | whatif | replay


class StepRow(Base):
    __tablename__ = "steps"
    step_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[str] = mapped_column(ForeignKey("runs.run_id"), index=True)
    name: Mapped[str] = mapped_column(String(64))
    status: Mapped[str] = mapped_column(String(16))
    started_at: Mapped[str] = mapped_column(String(32))
    ended_at: Mapped[str | None] = mapped_column(String(32), nullable=True)
    note: Mapped[str | None] = mapped_column(Text, nullable=True)


class EvidenceRow(Base):
    __tablename__ = "evidence"
    evidence_id: Mapped[str] = mapped_column(String(32), primary_key=True)
    run_id: Mapped[str] = mapped_column(ForeignKey("runs.run_id"), index=True)
    step: Mapped[str] = mapped_column(String(64))
    tool: Mapped[str] = mapped_column(String(64))
    name: Mapped[str] = mapped_column(
        String(64), default=""
    )  # logical name in the run (e.g. coverage_sufficient)
    inputs_json: Mapped[dict] = mapped_column(JSON)
    inputs_hash: Mapped[str] = mapped_column(String(64))
    dataset_hash: Mapped[str] = mapped_column(String(64))
    seed: Mapped[int] = mapped_column(Integer)
    outputs_json: Mapped[dict] = mapped_column(JSON)
    duration_ms: Mapped[float] = mapped_column(Float)
    created_at: Mapped[str] = mapped_column(String(32))


class MessageRow(Base):
    __tablename__ = "messages"
    message_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[str] = mapped_column(ForeignKey("runs.run_id"), index=True)
    role: Mapped[str] = mapped_column(String(16))
    content: Mapped[str] = mapped_column(Text)
    evidence_ids_json: Mapped[list] = mapped_column(JSON)
    created_at: Mapped[str] = mapped_column(String(32))


def _now() -> str:
    return datetime.now(UTC).isoformat()


def make_engine(url: str | None = None):
    url = (
        url
        or os.environ.get("DATABASE_URL")
        or f"sqlite:///{Path('data/ledger.sqlite').as_posix()}"
    )
    if url.startswith("sqlite"):
        Path(url.split("///", 1)[1]).parent.mkdir(parents=True, exist_ok=True)
    engine = create_engine(url, future=True)
    Base.metadata.create_all(engine)
    _migrate(engine)
    return engine


_ADDED_COLUMNS = {
    "runs": {"derived_from": "VARCHAR(32)", "kind": "VARCHAR(16) DEFAULT 'study'"},
    "evidence": {"name": "VARCHAR(64) DEFAULT ''"},
}


def _migrate(engine) -> None:
    """Additive migrations: add columns introduced after the first release."""
    from sqlalchemy import inspect, text

    insp = inspect(engine)
    with engine.begin() as conn:
        for table, cols in _ADDED_COLUMNS.items():
            if table not in insp.get_table_names():
                continue
            existing = {c["name"] for c in insp.get_columns(table)}
            for col, ddl in cols.items():
                if col not in existing:
                    conn.execute(text(f"ALTER TABLE {table} ADD COLUMN {col} {ddl}"))


def memory_engine():
    """A shared in-memory SQLite engine (one connection across threads) for tests and demos."""
    engine = create_engine(
        "sqlite://", future=True, connect_args={"check_same_thread": False}, poolclass=StaticPool
    )
    Base.metadata.create_all(engine)
    return engine


class Ledger:
    def __init__(self, engine=None):
        self.engine = engine or make_engine()

    # ------------------------------------------------------------------ runs
    def create_run(
        self,
        run_id: str,
        spec: ProblemSpec,
        dataset_hash: str,
        llm_used: bool,
        request_text: str | None,
        derived_from: str | None = None,
        kind: str = "study",
    ) -> None:
        with Session(self.engine) as s:
            s.add(
                RunRow(
                    run_id=run_id,
                    spec_json=json.loads(spec.model_dump_json()),
                    dataset_hash=dataset_hash,
                    status="running",
                    created_at=_now(),
                    llm_used=int(llm_used),
                    request_text=request_text,
                    derived_from=derived_from,
                    kind=kind,
                )
            )
            s.commit()

    def finish_run(
        self,
        run_id: str,
        verdict: Verdict | None,
        brief_md: str | None,
        brief_json: dict | None,
        error: str | None = None,
    ) -> None:
        with Session(self.engine) as s:
            row = s.get(RunRow, run_id)
            assert row is not None
            row.status = "failed" if error else "done"
            row.finished_at = _now()
            row.verdict_json = json.loads(verdict.model_dump_json()) if verdict else None
            row.brief_md = brief_md
            row.brief_json = brief_json
            row.error = error
            s.commit()

    def get_run(self, run_id: str) -> dict[str, Any] | None:
        with Session(self.engine) as s:
            row = s.get(RunRow, run_id)
            if row is None:
                return None
            steps = s.scalars(
                select(StepRow).where(StepRow.run_id == run_id).order_by(StepRow.step_id)
            ).all()
            ev = s.scalars(
                select(EvidenceRow)
                .where(EvidenceRow.run_id == run_id)
                .order_by(EvidenceRow.created_at)
            ).all()
            return {
                "run_id": row.run_id,
                "spec": row.spec_json,
                "dataset_hash": row.dataset_hash,
                "status": row.status,
                "created_at": row.created_at,
                "finished_at": row.finished_at,
                "verdict": row.verdict_json,
                "brief_md": row.brief_md,
                "brief_json": row.brief_json,
                "error": row.error,
                "llm_used": bool(row.llm_used),
                "request_text": row.request_text,
                "derived_from": row.derived_from,
                "kind": row.kind,
                "steps": [
                    {
                        "name": x.name,
                        "status": x.status,
                        "started_at": x.started_at,
                        "ended_at": x.ended_at,
                        "note": x.note,
                    }
                    for x in steps
                ],
                "evidence": [
                    {
                        "evidence_id": e.evidence_id,
                        "step": e.step,
                        "tool": e.tool,
                        "name": e.name,
                        "duration_ms": e.duration_ms,
                        "created_at": e.created_at,
                    }
                    for e in ev
                ],
            }

    def list_runs(self, limit: int = 50) -> list[dict[str, Any]]:
        with Session(self.engine) as s:
            rows = s.scalars(select(RunRow).order_by(RunRow.created_at.desc()).limit(limit)).all()
            return [
                {
                    "run_id": r.run_id,
                    "capability_name": r.spec_json.get("capability_name"),
                    "target_event": r.spec_json.get("target_event"),
                    "status": r.status,
                    "decision": (r.verdict_json or {}).get("decision"),
                    "created_at": r.created_at,
                    "dataset_hash": r.dataset_hash,
                    "kind": r.kind,
                    "derived_from": r.derived_from,
                    "horizon_days": r.spec_json.get("horizon_days"),
                }
                for r in rows
            ]

    # ------------------------------------------------------------------ steps
    def start_step(self, run_id: str, name: str, note: str | None = None) -> int:
        with Session(self.engine) as s:
            row = StepRow(run_id=run_id, name=name, status="running", started_at=_now(), note=note)
            s.add(row)
            s.commit()
            return int(row.step_id)

    def end_step(self, step_id: int, status: str = "done", note: str | None = None) -> None:
        with Session(self.engine) as s:
            row = s.get(StepRow, step_id)
            assert row is not None
            row.status = status
            row.ended_at = _now()
            if note:
                row.note = note
            s.commit()

    # ------------------------------------------------------------------ evidence
    def record(
        self,
        run_id: str,
        step: str,
        tool: str,
        inputs: dict[str, Any],
        dataset_hash: str,
        seed: int,
        fn: Callable[[], dict[str, Any]],
        name: str = "",
    ) -> Evidence:
        """Execute `fn`, store its rounded outputs as Evidence, return the Evidence."""
        ih = hash_inputs(
            {"tool": tool, "inputs": inputs, "dataset_hash": dataset_hash, "seed": seed}
        )
        eid = evidence_id(run_id, tool, ih)
        t0 = time.perf_counter()
        outputs = round_floats(fn())
        ms = (time.perf_counter() - t0) * 1000
        ev = Evidence(
            evidence_id=eid,
            run_id=run_id,
            step=step,
            tool=tool,
            inputs=round_floats(inputs),
            inputs_hash=ih,
            dataset_hash=dataset_hash,
            seed=seed,
            outputs=outputs,
            created_at=datetime.now(UTC),
        )
        with Session(self.engine) as s:
            if s.get(EvidenceRow, eid) is None:
                s.add(
                    EvidenceRow(
                        evidence_id=eid,
                        run_id=run_id,
                        step=step,
                        tool=tool,
                        name=name or tool,
                        inputs_json=ev.inputs,
                        inputs_hash=ih,
                        dataset_hash=dataset_hash,
                        seed=seed,
                        outputs_json=json.loads(canonical_json(outputs)),
                        duration_ms=ms,
                        created_at=ev.created_at.isoformat(),
                    )
                )
                s.commit()
        return ev

    def get_evidence(self, evidence_id: str) -> dict[str, Any] | None:
        with Session(self.engine) as s:
            e = s.get(EvidenceRow, evidence_id)
            if e is None:
                return None
            return {
                "evidence_id": e.evidence_id,
                "run_id": e.run_id,
                "step": e.step,
                "tool": e.tool,
                "name": e.name,
                "inputs": e.inputs_json,
                "inputs_hash": e.inputs_hash,
                "dataset_hash": e.dataset_hash,
                "seed": e.seed,
                "outputs": e.outputs_json,
                "duration_ms": e.duration_ms,
                "created_at": e.created_at,
            }

    def evidence_for_run(self, run_id: str) -> list[dict[str, Any]]:
        with Session(self.engine) as s:
            rows = s.scalars(
                select(EvidenceRow)
                .where(EvidenceRow.run_id == run_id)
                .order_by(EvidenceRow.created_at)
            ).all()
            return [self.get_evidence(r.evidence_id) for r in rows]  # type: ignore[misc]

    def evidence_objects(self, run_id: str) -> dict[str, Evidence]:
        """Rebuild the runner's {name: Evidence} map for a finished run (what-if / replay)."""
        out: dict[str, Evidence] = {}
        for e in self.evidence_for_run(run_id):
            out[e["name"] or e["tool"]] = Evidence(
                evidence_id=e["evidence_id"],
                run_id=e["run_id"],
                step=e["step"],
                tool=e["tool"],
                inputs=e["inputs"],
                inputs_hash=e["inputs_hash"],
                dataset_hash=e["dataset_hash"],
                seed=e["seed"],
                outputs=e["outputs"],
                created_at=datetime.fromisoformat(e["created_at"]),
            )
        return out

    # ------------------------------------------------------------------ messages
    def add_message(self, run_id: str, role: str, content: str, evidence_ids: list[str]) -> None:
        with Session(self.engine) as s:
            s.add(
                MessageRow(
                    run_id=run_id,
                    role=role,
                    content=content,
                    evidence_ids_json=evidence_ids,
                    created_at=_now(),
                )
            )
            s.commit()

    def messages(self, run_id: str) -> list[dict[str, Any]]:
        with Session(self.engine) as s:
            rows = s.scalars(
                select(MessageRow)
                .where(MessageRow.run_id == run_id)
                .order_by(MessageRow.message_id)
            ).all()
            return [
                {
                    "role": r.role,
                    "content": r.content,
                    "evidence_ids": r.evidence_ids_json,
                    "created_at": r.created_at,
                }
                for r in rows
            ]

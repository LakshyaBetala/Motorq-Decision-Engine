"""Service facade used by the CLI and the API: owns the data source, ledger and runner, and
decides whether the LLM interface is in play."""

from __future__ import annotations

import os
import threading
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Any

from motorq_de.agent.portfolio import portfolio as _portfolio
from motorq_de.agent.runner import Runner, RunResult
from motorq_de.agent.webhook import notify_run
from motorq_de.data.source import DataSource
from motorq_de.data.synthetic import SyntheticSource, dataset_dirs, latest_dataset
from motorq_de.ledger.store import Ledger
from motorq_de.schemas import ProblemSpec

DATA_ROOT = Path(os.environ.get("MDE_DATA_ROOT", "data/synthetic"))


def _load_value_assumptions() -> dict[str, dict[str, Any]]:
    import yaml

    from motorq_de.economics.cost import HERE as ECON

    raw = yaml.safe_load((ECON / "value_assumptions.yaml").read_text(encoding="utf-8"))
    out = {}
    for target, body in raw["targets"].items():
        out[target] = {k: v for k, v in body.items() if k != "owner"}
    return out


EXAMPLE_VALUES: dict[str, dict[str, Any]] = _load_value_assumptions()

EXAMPLE_SPECS: dict[str, dict[str, Any]] = {
    "brake": {
        "capability_name": "brake_service_7d",
        "target_event": "brake_service_event",
        "horizon_days": 7,
        "value": EXAMPLE_VALUES["brake_service_event"],
    },
    "theft": {
        "capability_name": "theft_risk_30d",
        "target_event": "theft_event",
        "horizon_days": 30,
        "value": EXAMPLE_VALUES["theft_event"],
    },
    "battery": {
        "capability_name": "battery_service_14d",
        "target_event": "battery_degradation_event",
        "horizon_days": 14,
        "value": EXAMPLE_VALUES["battery_degradation_event"],
    },
}


def snowflake_source(config_path: Path):
    """Build a SnowflakeSource from a YAML config and SNOWFLAKE_* environment credentials."""
    from motorq_de.data.snowflake import SnowflakeConfig, SnowflakeSource, connector_query

    params = {
        "account": os.environ["SNOWFLAKE_ACCOUNT"],
        "user": os.environ["SNOWFLAKE_USER"],
        "password": os.environ.get("SNOWFLAKE_PASSWORD"),
        "warehouse": os.environ.get("SNOWFLAKE_WAREHOUSE"),
        "database": os.environ.get("SNOWFLAKE_DATABASE"),
        "schema": os.environ.get("SNOWFLAKE_SCHEMA"),
        "role": os.environ.get("SNOWFLAKE_ROLE"),
    }
    params = {k: v for k, v in params.items() if v}
    cfg = SnowflakeConfig.from_yaml(config_path)
    return SnowflakeSource(connector_query(params), cfg, label=str(config_path))


class Service:
    def __init__(
        self,
        source: DataSource | None = None,
        ledger: Ledger | None = None,
        dataset: Path | None = None,
    ):
        if source is None and os.environ.get("MDE_SNOWFLAKE_CONFIG"):
            source = snowflake_source(Path(os.environ["MDE_SNOWFLAKE_CONFIG"]))
        if source is None:
            ds = dataset or (
                Path(os.environ["MDE_DATASET"])
                if os.environ.get("MDE_DATASET")
                else latest_dataset(DATA_ROOT)
            )
            source = SyntheticSource(ds)
        self.source = source
        self.ledger = ledger or Ledger()
        self._runners: dict[str, Runner] = {}
        self._pool = ThreadPoolExecutor(max_workers=1)
        self._futures: dict[str, Future] = {}
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ llm
    @staticmethod
    def llm_available() -> bool:
        from motorq_de.agent.llm import llm_available

        return llm_available()

    def spec_from_text(self, text: str) -> tuple[ProblemSpec, str]:
        from motorq_de.agent.llm import parse_spec

        return parse_spec(text, EXAMPLE_VALUES)

    # ------------------------------------------------------------------ runs
    def runner(self, progress=None) -> Runner:
        key = self.source.dataset_hash
        if key not in self._runners:
            self._runners[key] = Runner(self.source, self.ledger, progress)
        return self._runners[key]

    def run(
        self,
        spec: ProblemSpec,
        request_text: str | None = None,
        use_llm: bool = False,
        progress=None,
    ) -> RunResult:
        narrative = None
        if use_llm and self.llm_available():
            from motorq_de.agent.llm import narrative as _narr

            narrative = _narr
        res = self.runner(progress).run(
            spec, request_text=request_text, llm_used=narrative is not None, narrative=narrative
        )
        notify_run(res, self.source.dataset_hash)
        return res

    def whatif(
        self,
        run_id: str,
        value: dict[str, Any] | None = None,
        price_overrides: dict[str, Any] | None = None,
        constraints: dict[str, Any] | None = None,
    ) -> RunResult:
        from motorq_de.schemas import ValueAssumptions

        v = ValueAssumptions.model_validate(value) if value else None
        return self.runner().whatif(run_id, v, price_overrides, constraints)

    def replay(self, run_id: str, progress=None) -> RunResult:
        return self.runner(progress).replay(run_id)

    def portfolio(self) -> dict[str, Any]:
        catalog = [m.model_dump() for m in self.source.list_signals()]
        return _portfolio(self.ledger, catalog)

    def submit(
        self, spec: ProblemSpec, request_text: str | None = None, use_llm: bool = False
    ) -> str:
        """Queue a run; returns a ticket the API can poll via the ledger once the run_id is known."""
        ticket = "t_" + uuid.uuid4().hex[:10]
        fut = self._pool.submit(self.run, spec, request_text, use_llm)
        with self._lock:
            self._futures[ticket] = fut
        return ticket

    def ticket_status(self, ticket: str) -> dict[str, Any]:
        fut = self._futures.get(ticket)
        if fut is None:
            return {"ticket": ticket, "status": "unknown"}
        if not fut.done():
            return {"ticket": ticket, "status": "running"}
        exc = fut.exception()
        if exc:
            return {"ticket": ticket, "status": "failed", "error": f"{type(exc).__name__}: {exc}"}
        res: RunResult = fut.result()
        return {
            "ticket": ticket,
            "status": "done",
            "run_id": res.run_id,
            "decision": res.verdict.decision,
        }

    def ask(self, run_id: str, question: str) -> tuple[str, list[str]]:
        from motorq_de.agent.llm import answer

        return answer(self.ledger, run_id, question)

    # ------------------------------------------------------------------ datasets
    @staticmethod
    def datasets() -> list[dict[str, Any]]:
        import json

        out = []
        for d in dataset_dirs(DATA_ROOT):
            t = json.loads((d / "truth.json").read_text(encoding="utf-8"))
            out.append(
                {
                    "dataset_hash": d.name,
                    "generator_version": t.get("generator_version"),
                    "seed": t.get("seed"),
                    "achieved_rates": t.get("achieved_rates"),
                }
            )
        return out

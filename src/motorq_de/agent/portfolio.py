"""Portfolio view: every capability side by side, and the COGS report - which signals and
polling cadences no capability needs.

Reads only the ledger. A capability is represented by its latest finished study run
(what-if runs are listed as variants of their parent, not as capabilities).
"""

from __future__ import annotations

from typing import Any

from motorq_de.ledger.store import Ledger

DECISION_ORDER = {"BUILD_READY": 0, "PILOT": 1, "NOT_FEASIBLE": 2}


def portfolio(ledger: Ledger, catalog: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    runs = [r for r in ledger.list_runs(limit=500) if r["status"] == "done"]
    latest: dict[str, dict[str, Any]] = {}
    for r in runs:  # list_runs is newest-first
        if r.get("kind", "study") != "study":
            continue
        key = r["capability_name"] or r["run_id"]
        latest.setdefault(key, r)

    rows: list[dict[str, Any]] = []
    required: dict[str, set[str]] = {}
    cadence_needed: dict[str, set[str]] = {}
    for cap, r in latest.items():
        ev = {e["name"] or e["tool"]: e for e in ledger.evidence_for_run(r["run_id"])}
        out = _outputs_of(ev)
        verdict = ledger.get_run(r["run_id"])["verdict"] or {}
        roi = out("roi")
        cs = out("cost_sufficient")
        suff = out("ablation", "sufficient_set") or []
        cov = out("coverage_sufficient", "fleet_share_full_set")
        tripped = [f["name"] for f in verdict.get("flags", []) if f.get("tripped")]
        failed = [g["name"] for g in verdict.get("gates", []) if not g.get("passed")]
        rows.append(
            {
                "capability_name": cap,
                "run_id": r["run_id"],
                "target_event": r["target_event"],
                "horizon_days": r.get("horizon_days"),
                "decision": verdict.get("decision"),
                "gates_failed": failed,
                "flags_tripped": tripped,
                "p_roi_positive": (roi or {}).get("p_roi_positive"),
                "net_value_p50": ((roi or {}).get("net_value_year") or {}).get("p50"),
                "coverage": cov,
                "n_sufficient_signals": len(suff),
                "sufficient_set": suff,
                "run_cost_marginal_month": ((cs or {}).get("monthly") or {}).get("marginal_total"),
                "dominant_input": out("tornado", "dominant_input"),
                "created_at": r["created_at"],
                "dataset_hash": r["dataset_hash"],
            }
        )
        if verdict.get("decision") in ("BUILD_READY", "PILOT"):
            required[cap] = set(suff)
            cadence_needed[cap] = set(
                ((cs or {}).get("polling") or {}).get("highest_cadence_signals") or []
            )
    rows.sort(
        key=lambda x: (DECISION_ORDER.get(x["decision"] or "", 9), -(x["p_roi_positive"] or 0))
    )

    union = set().union(*required.values()) if required else set()
    cogs = None
    if catalog is not None:
        by_id = {m["signal_id"]: m for m in catalog}
        unused = sorted(s for s in by_id if s not in union)
        realtime_needed = sorted(
            s for s in union if by_id.get(s, {}).get("declared_frequency") == "realtime"
        )
        cogs = {
            "n_catalog": len(by_id),
            "n_required_by_viable_capabilities": len(union),
            "required": sorted(union),
            "required_by": {
                s: sorted(c for c, req in required.items() if s in req) for s in sorted(union)
            },
            "unused_signals": unused,
            "unused_by_category": _group(unused, by_id),
            "realtime_signals_required": realtime_needed,
            "cadence_lever": (
                "no viable capability needs realtime cadence - polling could drop to daily"
                if union and not realtime_needed
                else f"{len(realtime_needed)} realtime signals are required by viable capabilities"
                if union
                else "no viable capability yet"
            ),
        }
    return {"capabilities": rows, "cogs": cogs, "n_capabilities": len(rows)}


def _outputs_of(ev: dict[str, dict[str, Any]]):
    def out(name: str, key: str | None = None):
        if name not in ev:
            return {} if key is None else None
        o = ev[name]["outputs"]
        return o if key is None else o.get(key)

    return out


def _group(signals: list[str], by_id: dict[str, dict[str, Any]]) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for s in signals:
        out.setdefault(by_id[s].get("category", "?"), []).append(s)
    return out

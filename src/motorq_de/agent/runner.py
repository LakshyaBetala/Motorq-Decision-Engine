"""Headless runner: the default plan, end to end, with every tool call recorded as Evidence.

This is what `--no-llm` executes and what the LLM agent extends. The stages mirror the
README: DEFINE -> FEASIBILITY -> EXPERIMENT -> ECONOMICS -> DELIVERY -> POLICY -> REPORT,
with code-level replan predicates (no LLM involved) that add evidence when a result is
borderline.
"""

from __future__ import annotations

import json
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from motorq_de.data.source import DataSource
from motorq_de.economics.cost import compare_costs, run_cost
from motorq_de.economics.deployment import deployment_fit
from motorq_de.economics.value import choose_operating_point, roi_distribution, tornado
from motorq_de.harness.experiments import (
    ablation,
    cross_oem_validation,
    feature_analysis,
    model_comparison,
    temporal_validation,
)
from motorq_de.harness.frames import FeatureStore
from motorq_de.ledger.store import Ledger
from motorq_de.policy.verdict import EvidenceBundle, decide
from motorq_de.quality.checks import (
    coverage_report,
    event_rate,
    leakage_check,
    quality_report,
    usable_signals,
)
from motorq_de.report.brief import Brief, Line, build_brief
from motorq_de.report.render import to_json, to_markdown
from motorq_de.schemas import Evidence, ProblemSpec, Range, Verdict

STAGES = ("DEFINE", "FEASIBILITY", "EXPERIMENT", "ECONOMICS", "DELIVERY", "POLICY", "REPORT")


@dataclass
class RunResult:
    run_id: str
    verdict: Verdict
    brief: Brief
    brief_md: str
    evidence: dict[str, Evidence]
    replans: list[str] = field(default_factory=list)


class Runner:
    def __init__(
        self, source: DataSource, ledger: Ledger, progress: Callable[[str], None] | None = None
    ):
        self.source = source
        self.ledger = ledger
        self.store = FeatureStore(source)
        self.progress = progress or (lambda _m: None)

    # ------------------------------------------------------------------ helpers
    def _rec(
        self,
        run_id: str,
        step: str,
        name: str,
        tool: str,
        inputs: dict[str, Any],
        seed: int,
        fn: Callable[[], dict[str, Any]],
        ev: dict[str, Evidence],
    ) -> dict[str, Any]:
        self.progress(f"{step}: {tool}")
        e = self.ledger.record(run_id, step, tool, inputs, self.source.dataset_hash, seed, fn)
        ev[name] = e
        return e.outputs

    def candidate_signals(self, spec: ProblemSpec) -> list[str]:
        metas = self.source.list_signals()
        if spec.target_event == "battery_degradation_event":
            return [m.signal_id for m in metas if m.powertrain in ("any", "ev")]
        return [m.signal_id for m in metas]

    # ------------------------------------------------------------------ run
    def run(
        self,
        spec: ProblemSpec,
        request_text: str | None = None,
        llm_used: bool = False,
        narrative: Callable[[dict[str, Evidence], Verdict], list[Line]] | None = None,
    ) -> RunResult:
        run_id = uuid.uuid4().hex[:12]
        self.ledger.create_run(run_id, spec, self.source.dataset_hash, llm_used, request_text)
        ev: dict[str, Evidence] = {}
        replans: list[str] = []
        seed = spec.seed
        try:
            # ---------------------------------------------------------- DEFINE
            sid = self.ledger.start_step(run_id, "DEFINE")
            self._rec(
                run_id,
                "DEFINE",
                "spec",
                "problem_spec",
                {"spec": json.loads(spec.model_dump_json())},
                seed,
                lambda: json.loads(spec.model_dump_json()),
                ev,
            )
            truth = getattr(self.source, "truth", lambda: {})()
            if truth:
                veh = self.source.vehicles()
                dr = self.source.date_range()
                t = dict(truth)
                t["n_vehicles"] = int(len(veh))
                t["n_days"] = int((dr.end - dr.start).days + 1)
                self._rec(
                    run_id,
                    "DEFINE",
                    "dataset_truth",
                    "dataset_truth",
                    {"dataset_hash": self.source.dataset_hash},
                    seed,
                    lambda: t,
                    ev,
                )
            self.ledger.end_step(sid)

            # ---------------------------------------------------------- FEASIBILITY
            sid = self.ledger.start_step(run_id, "FEASIBILITY")
            cands = self.candidate_signals(spec)
            er = self._rec(
                run_id,
                "FEASIBILITY",
                "event_rate",
                "event_rate",
                {"target": spec.target_event, "population": spec.powertrain_scope},
                seed,
                lambda: event_rate(self.source, spec),
                ev,
            )
            rate = Range(
                low=er["rate_ci_lo"],
                base=er["rate_per_vehicle_year"],
                high=max(er["rate_ci_hi"], er["rate_per_vehicle_year"]),
            )
            q_all = self._rec(
                run_id,
                "FEASIBILITY",
                "quality_all",
                "quality_report",
                {"signals": cands},
                seed,
                lambda: quality_report(self.source, cands),
                ev,
            )
            lk = self._rec(
                run_id,
                "FEASIBILITY",
                "leakage",
                "leakage_check",
                {"signals": cands, "horizon_days": spec.horizon_days, "target": spec.target_event},
                seed,
                lambda: leakage_check(self.source, spec, cands),
                ev,
            )
            us = self._rec(
                run_id,
                "FEASIBILITY",
                "usable",
                "usable_signals",
                {"signals": cands},
                seed,
                lambda: usable_signals(cands, q_all, lk),
                ev,
            )
            usable = us["usable"]
            self._rec(
                run_id,
                "FEASIBILITY",
                "coverage_all",
                "coverage_report",
                {"signals": usable},
                seed,
                lambda: coverage_report(self.source, usable),
                ev,
            )
            self.ledger.end_step(sid)

            # ---------------------------------------------------------- EXPERIMENT
            sid = self.ledger.start_step(run_id, "EXPERIMENT")
            fa = self._rec(
                run_id,
                "EXPERIMENT",
                "feature_analysis",
                "feature_analysis",
                {"signals": usable, "n_repeats": 1},
                seed,
                lambda: feature_analysis(self.store, spec, usable, n_repeats=1),
                ev,
            )
            order = fa["order_least_to_most_important"]
            ab = self._rec(
                run_id,
                "EXPERIMENT",
                "ablation",
                "ablation",
                {"signals": usable, "order": order},
                seed,
                lambda: ablation(self.store, spec, usable, order_least_to_most=order),
                ev,
            )
            suff = ab["sufficient_set"]
            self._rec(
                run_id,
                "EXPERIMENT",
                "coverage_sufficient",
                "coverage_report",
                {"signals": suff},
                seed,
                lambda: coverage_report(self.source, suff),
                ev,
            )
            self._rec(
                run_id,
                "EXPERIMENT",
                "quality_sufficient",
                "quality_report",
                {"signals": suff},
                seed,
                lambda: quality_report(self.source, suff),
                ev,
            )
            cmp = self._rec(
                run_id,
                "EXPERIMENT",
                "model_comparison",
                "model_comparison",
                {"sets": {"sufficient": suff, "full_usable": usable}},
                seed,
                lambda: model_comparison(
                    self.store, spec, {"sufficient": suff, "full_usable": usable}
                ),
                ev,
            )
            self._rec(
                run_id,
                "EXPERIMENT",
                "temporal",
                "temporal_validation",
                {"signals": suff},
                seed,
                lambda: temporal_validation(self.store, spec, suff),
                ev,
            )
            xo = self._rec(
                run_id,
                "EXPERIMENT",
                "cross_oem",
                "cross_oem_validation",
                {"signals": suff},
                seed,
                lambda: cross_oem_validation(self.store, spec, suff),
                ev,
            )
            # replan predicate: cross-OEM gap -> explain which required signals the worst OEM lacks
            if (
                xo.get("worst_oem")
                and xo.get("std_auc") is not None
                and (xo["std_auc"] > 0.03 or (xo["mean_auc"] - xo["min_auc"]) > 0.05)
            ):
                replans.append("cross_oem_gap")
                worst = xo["worst_oem"]
                cov_s = ev["coverage_sufficient"].outputs
                self._rec(
                    run_id,
                    "EXPERIMENT",
                    "cross_oem_gap",
                    "cross_oem_gap_analysis",
                    {"oem": worst, "signals": suff},
                    seed,
                    lambda: {
                        "oem": worst,
                        "missing_signals": [
                            s for s in suff if not cov_s["per_signal"][s][worst]["emits"]
                        ],
                        "coverage_by_signal": {
                            s: cov_s["per_signal"][s][worst]["coverage"] for s in suff
                        },
                        "auc_worst": xo["per_oem"][worst]["auc"],
                        "auc_mean": xo["mean_auc"],
                    },
                    ev,
                )
            self.ledger.end_step(sid, note=";".join(replans) or None)

            # ---------------------------------------------------------- ECONOMICS
            sid = self.ledger.start_step(run_id, "ECONOMICS")
            metas = {m.signal_id: m for m in self.source.list_signals()}
            fleet = spec.value.fleet_size.base
            cf = self._rec(
                run_id,
                "ECONOMICS",
                "cost_full",
                "run_cost",
                {"signals": usable, "fleet_size": fleet, "delivery_mode": spec.delivery_mode},
                seed,
                lambda: run_cost([metas[s] for s in usable], fleet, spec.delivery_mode),
                ev,
            )
            cs = self._rec(
                run_id,
                "ECONOMICS",
                "cost_sufficient",
                "run_cost",
                {"signals": suff, "fleet_size": fleet, "delivery_mode": spec.delivery_mode},
                seed,
                lambda: run_cost([metas[s] for s in suff], fleet, spec.delivery_mode),
                ev,
            )
            self._rec(
                run_id,
                "ECONOMICS",
                "cost_compare",
                "compare_costs",
                {"full": ev["cost_full"].evidence_id, "reduced": ev["cost_sufficient"].evidence_id},
                seed,
                lambda: compare_costs(cf, cs),
                ev,
            )
            lg = cmp["sets"]["sufficient"]["models"]["lightgbm"]
            # recall uncertainty proxy: AUC CI half-width scaled; a direct recall bootstrap is a later refinement
            rec_hw = (lg["auc"]["hi"] - lg["auc"]["lo"]) / 2 * 1.5
            insp = Range(**self._price_insp())
            op = self._rec(
                run_id,
                "ECONOMICS",
                "operating_point",
                "choose_operating_point",
                {"operating_points": lg["operating_points"], "recall_ci_halfwidth": rec_hw},
                seed,
                lambda: choose_operating_point(
                    spec.value,
                    rate,
                    lg["operating_points"],
                    rec_hw,
                    spec.horizon_days,
                    insp,
                    cs["monthly"]["marginal_total"],
                    cs["monthly"]["build_amortized"],
                    seed,
                ),
                ev,
            )
            ch = op["chosen"]
            recall = Range(
                low=max(0.0, ch["recall"] - rec_hw),
                base=ch["recall"],
                high=min(1.0, ch["recall"] + rec_hw),
            )
            roi = self._rec(
                run_id,
                "ECONOMICS",
                "roi",
                "roi_distribution",
                {
                    "event_rate": rate.model_dump(),
                    "recall": recall.model_dump(),
                    "alert_rate": ch["alert_rate"],
                    "run_cost_month": cs["monthly"]["marginal_total"],
                },
                seed,
                lambda: roi_distribution(
                    spec.value,
                    rate,
                    recall,
                    ch["alert_rate"],
                    spec.horizon_days,
                    insp,
                    cs["monthly"]["marginal_total"],
                    cs["monthly"]["build_amortized"],
                    seed,
                ),
                ev,
            )
            self._rec(
                run_id,
                "ECONOMICS",
                "tornado",
                "tornado",
                {"recall": recall.model_dump(), "alert_rate": ch["alert_rate"]},
                seed,
                lambda: tornado(
                    spec.value,
                    rate,
                    recall,
                    ch["alert_rate"],
                    spec.horizon_days,
                    insp,
                    cs["monthly"]["marginal_total"],
                    cs["monthly"]["build_amortized"],
                    seed,
                ),
                ev,
            )
            # replan predicate: borderline economics -> widen value ranges and re-run ROI
            if roi["p_roi_positive"] is not None and 0.4 <= roi["p_roi_positive"] <= 0.6:
                replans.append("roi_borderline")
                wide = spec.value.model_copy(
                    update={
                        "usd_per_avoided_event": Range(
                            low=spec.value.usd_per_avoided_event.low * 0.5,
                            base=spec.value.usd_per_avoided_event.base,
                            high=spec.value.usd_per_avoided_event.high * 1.5,
                        ),
                        "preventable_fraction": Range(
                            low=max(0.0, spec.value.preventable_fraction.low * 0.5),
                            base=spec.value.preventable_fraction.base,
                            high=min(1.0, spec.value.preventable_fraction.high * 1.5),
                        ),
                    }
                )
                self._rec(
                    run_id,
                    "ECONOMICS",
                    "roi_wide",
                    "roi_distribution",
                    {"widened": True, "recall": recall.model_dump()},
                    seed,
                    lambda: roi_distribution(
                        wide,
                        rate,
                        recall,
                        ch["alert_rate"],
                        spec.horizon_days,
                        insp,
                        cs["monthly"]["marginal_total"],
                        cs["monthly"]["build_amortized"],
                        seed,
                    ),
                    ev,
                )
            self.ledger.end_step(
                sid, note=";".join(r for r in replans if r.startswith("roi")) or None
            )

            # ---------------------------------------------------------- DELIVERY
            sid = self.ledger.start_step(run_id, "DELIVERY")
            self._rec(
                run_id,
                "DELIVERY",
                "deployment",
                "deployment_fit",
                {"delivery_mode": spec.delivery_mode, "horizon_days": spec.horizon_days},
                seed,
                lambda: deployment_fit(spec),
                ev,
            )
            self.ledger.end_step(sid)

            # ---------------------------------------------------------- POLICY
            sid = self.ledger.start_step(run_id, "POLICY")
            bundle = EvidenceBundle()
            for name in (
                "coverage_sufficient",
                "model_comparison",
                "roi",
                "deployment",
                "cross_oem",
                "temporal",
                "quality_sufficient",
                "ablation",
                "leakage",
                "cost_sufficient",
            ):
                if name in ev:
                    bundle.put(name, ev[name].outputs, ev[name].evidence_id)
            verdict = decide(spec, bundle)
            self._rec(
                run_id,
                "POLICY",
                "verdict",
                "decision_policy",
                {"policy_version": verdict.policy_version},
                seed,
                lambda: json.loads(verdict.model_dump_json()),
                ev,
            )
            self.ledger.end_step(sid, note=verdict.decision)

            # ---------------------------------------------------------- REPORT
            sid = self.ledger.start_step(run_id, "REPORT")
            brief = build_brief(
                spec, run_id, self.source.dataset_hash, ev, verdict, llm_used=llm_used
            )
            if narrative is not None:
                brief = brief.model_copy(update={"narrative": tuple(narrative(ev, verdict))})
            md = to_markdown(brief)
            self.ledger.finish_run(run_id, verdict, md, to_json(brief))
            self.ledger.end_step(sid)
            return RunResult(
                run_id=run_id,
                verdict=verdict,
                brief=brief,
                brief_md=md,
                evidence=ev,
                replans=replans,
            )
        except Exception as exc:  # record the failure, then re-raise
            self.ledger.finish_run(run_id, None, None, None, error=f"{type(exc).__name__}: {exc}")
            raise

    @staticmethod
    def _price_insp() -> dict[str, float]:
        from motorq_de.economics.cost import load_price_sheet

        return dict(load_price_sheet()["operations"]["inspection_cost_per_alert"])

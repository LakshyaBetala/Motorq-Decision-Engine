"""Headless runner: the default plan, end to end, with every tool call recorded as Evidence.

This is what `mde run headless` executes and what the LLM agent extends. Stages mirror the
README: DEFINE -> FEASIBILITY -> EXPERIMENT -> ECONOMICS -> DELIVERY -> POLICY -> REPORT,
with code-level replan predicates (no LLM involved) that add evidence when a result is
borderline.

Two derived runs reuse a finished study's evidence:
    whatif   re-runs ECONOMICS -> POLICY -> REPORT with new value assumptions or price
             overrides in seconds, citing the parent's harness evidence
    replay   re-runs the whole study from the stored spec and diffs every evidence record
"""

from __future__ import annotations

import json
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from motorq_de.data.contract import validate_source
from motorq_de.data.source import DataSource
from motorq_de.economics.cost import compare_costs, load_price_sheet, run_cost
from motorq_de.economics.deployment import deployment_fit
from motorq_de.economics.value import choose_operating_point, roi_distribution, tornado
from motorq_de.harness import runtime
from motorq_de.harness.experiments import (
    ablation,
    cross_oem_validation,
    feature_analysis,
    learning_curve,
    model_comparison,
    redundancy,
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
from motorq_de.schemas import Evidence, ProblemSpec, Range, ValueAssumptions, Verdict

STAGES = ("DEFINE", "FEASIBILITY", "EXPERIMENT", "ECONOMICS", "DELIVERY", "POLICY", "REPORT")
POLICY_INPUTS = (
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
    "operating_point",
)


@dataclass
class RunResult:
    run_id: str
    verdict: Verdict
    brief: Brief
    brief_md: str
    evidence: dict[str, Evidence]
    replans: list[str] = field(default_factory=list)
    diff: dict[str, Any] | None = None


class Runner:
    def __init__(
        self,
        source: DataSource,
        ledger: Ledger,
        progress: Callable[[str], None] | None = None,
    ):
        self.source = source
        self.ledger = ledger
        self.store = FeatureStore(source)
        self.progress = progress or (lambda _m: None)

    # ------------------------------------------------------------------ helpers
    def _rec(self, run_id, step, name, tool, inputs, seed, fn, ev) -> dict[str, Any]:
        self.progress(f"{step}: {tool}")
        e = self.ledger.record(
            run_id, step, tool, inputs, self.source.dataset_hash, seed, fn, name=name
        )
        ev[name] = e
        return e.outputs

    def candidate_signals(self, spec: ProblemSpec) -> list[str]:
        metas = self.source.list_signals()
        if spec.powertrain_scope == "ev":
            return [m.signal_id for m in metas if m.powertrain in ("any", "ev")]
        if spec.powertrain_scope == "ice":
            return [m.signal_id for m in metas if m.powertrain in ("any", "ice")]
        return [m.signal_id for m in metas]

    def _metas(self) -> dict[str, Any]:
        return {m.signal_id: m for m in self.source.list_signals()}

    def _signal_costs(self, signals: list[str], spec: ProblemSpec, prices) -> dict[str, float]:
        """Marginal monthly cost of each signal on its own (drives cost-aware ablation)."""
        metas = self._metas()
        fleet = spec.value.fleet_size.base
        out = {}
        for s in signals:
            c = run_cost([metas[s]], fleet, spec.delivery_mode, prices)["monthly"]
            out[s] = c["infra_total"] + c["oem_api_calls"] + c["oem_marginal"]
        return out

    # ------------------------------------------------------------------ study
    def run(
        self,
        spec: ProblemSpec,
        request_text: str | None = None,
        llm_used: bool = False,
        narrative: Callable[[dict[str, Evidence], Verdict], list[Line]] | None = None,
        kind: str = "study",
        derived_from: str | None = None,
    ) -> RunResult:
        run_id = uuid.uuid4().hex[:12]
        self.ledger.create_run(
            run_id,
            spec,
            self.source.dataset_hash,
            llm_used,
            request_text,
            derived_from=derived_from,
            kind=kind,
        )
        ev: dict[str, Evidence] = {}
        replans: list[str] = []
        prices = load_price_sheet()
        try:
            self._define(run_id, spec, ev)
            usable, rate = self._feasibility(run_id, spec, ev)
            suff = self._experiment(run_id, spec, ev, usable, replans, prices)
            self._economics(run_id, spec, ev, usable, suff, rate, prices, replans)
            self._delivery(run_id, spec, ev)
            verdict = self._policy(run_id, spec, ev)
            brief, md = self._report(run_id, spec, ev, verdict, llm_used, narrative)
            return RunResult(run_id, verdict, brief, md, ev, replans)
        except Exception as exc:  # record the failure, then re-raise
            self.ledger.finish_run(run_id, None, None, None, error=f"{type(exc).__name__}: {exc}")
            raise

    # ------------------------------------------------------------------ stages
    def _define(self, run_id, spec, ev) -> None:
        sid = self.ledger.start_step(run_id, "DEFINE")
        seed = spec.seed
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
        # the source must satisfy the canonical contract before anything is computed on it
        contract = self._rec(
            run_id,
            "DEFINE",
            "data_contract",
            "data_contract",
            {"dataset_hash": self.source.dataset_hash},
            seed,
            lambda: validate_source(self.source).as_dict(),
            ev,
        )
        if not contract["ok"]:
            raise RuntimeError(
                "source violates the canonical data contract: " + "; ".join(contract["errors"])
            )
        fp = runtime.fingerprint()
        self._rec(
            run_id,
            "DEFINE",
            "runtime",
            "runtime_fingerprint",
            {"dataset_hash": self.source.dataset_hash},
            seed,
            lambda: fp,
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

    def _feasibility(self, run_id, spec, ev) -> tuple[list[str], Range]:
        sid = self.ledger.start_step(run_id, "FEASIBILITY")
        seed = spec.seed
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
        return usable, rate

    def _experiment(self, run_id, spec, ev, usable, replans, prices) -> list[str]:
        sid = self.ledger.start_step(run_id, "EXPERIMENT")
        seed = spec.seed
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
        importance = {r["signal"]: r["perm_importance"] for r in fa["ranking"]}
        costs = self._signal_costs(usable, spec, prices)
        ab = self._rec(
            run_id,
            "EXPERIMENT",
            "ablation",
            "ablation",
            {"signals": usable, "order": order, "cost_aware": True},
            seed,
            lambda: ablation(
                self.store,
                spec,
                usable,
                order_least_to_most=order,
                signal_cost=costs,
                importance=importance,
            ),
            ev,
        )
        suff = ab["sufficient_set"]
        # how the screened candidates relate to each other: redundancy groups explain why
        # ablation could drop a signal without losing information
        self._rec(
            run_id,
            "EXPERIMENT",
            "redundancy",
            "redundancy",
            {"signals": ab["candidate_set"], "sufficient_set": suff},
            seed,
            lambda: redundancy(self.store, spec, ab["candidate_set"]),
            ev,
        )
        # cadence ablation: what does the capability lose if it only had daily/weekly signals?
        metas = self._metas()
        daily_only = [s for s in usable if metas[s].declared_frequency != "realtime"]
        if 0 < len(daily_only) < len(usable):
            order_daily = [s for s in order if s in set(daily_only)]
            self._rec(
                run_id,
                "EXPERIMENT",
                "ablation_daily_cadence",
                "ablation",
                {"signals": daily_only, "order": order_daily, "cadence": "daily_or_weekly"},
                seed,
                lambda: ablation(
                    self.store,
                    spec,
                    daily_only,
                    order_least_to_most=order_daily,
                    signal_cost=costs,
                    importance=importance,
                ),
                ev,
            )
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
        sets = {"sufficient": suff, "full_usable": usable}
        if "ablation_daily_cadence" in ev:
            sets["daily_cadence"] = ev["ablation_daily_cadence"].outputs["sufficient_set"]
        self._rec(
            run_id,
            "EXPERIMENT",
            "model_comparison",
            "model_comparison",
            {"sets": sets},
            seed,
            lambda: model_comparison(self.store, spec, sets),
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
        # replan predicate: cross-OEM gap -> which required signals does the worst OEM lack?
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
                    "within_oem_auc": xo["per_oem"][worst].get("within_oem_auc"),
                    "diagnosis": xo["per_oem"][worst].get("diagnosis"),
                },
                ev,
            )
        # was this much data needed: AUC of the sufficient set on nested vehicle subsets
        self._rec(
            run_id,
            "EXPERIMENT",
            "learning_curve",
            "learning_curve",
            {"signals": suff},
            seed,
            lambda: learning_curve(self.store, spec, suff),
            ev,
        )
        # what the study cost to compute, and how much of it was served from the fit cache
        self._rec(
            run_id,
            "EXPERIMENT",
            "compute",
            "compute",
            {"cache_enabled": self.store.fits.enabled},
            seed,
            lambda: {
                **self.store.fits.stats(),
                "cache_enabled": self.store.fits.enabled,
                "fold_workers": runtime.cv_parallelism(5)[0],
                "lgbm_threads": runtime.cv_parallelism(5)[1],
            },
            ev,
        )
        self.ledger.end_step(sid, note=";".join(replans) or None)
        return suff

    def _economics(self, run_id, spec, ev, usable, suff, rate, prices, replans) -> None:
        sid = self.ledger.start_step(run_id, "ECONOMICS")
        seed = spec.seed
        metas = self._metas()
        fleet = spec.value.fleet_size.base
        cmp = ev["model_comparison"].outputs
        cf = self._rec(
            run_id,
            "ECONOMICS",
            "cost_full",
            "run_cost",
            {"signals": usable, "fleet_size": fleet, "delivery_mode": spec.delivery_mode},
            seed,
            lambda: run_cost([metas[s] for s in usable], fleet, spec.delivery_mode, prices),
            ev,
        )
        cs = self._rec(
            run_id,
            "ECONOMICS",
            "cost_sufficient",
            "run_cost",
            {"signals": suff, "fleet_size": fleet, "delivery_mode": spec.delivery_mode},
            seed,
            lambda: run_cost([metas[s] for s in suff], fleet, spec.delivery_mode, prices),
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
        if "daily_cadence" in cmp["sets"]:
            daily = cmp["sets"]["daily_cadence"]["signals"]
            self._rec(
                run_id,
                "ECONOMICS",
                "cost_daily_cadence",
                "run_cost",
                {"signals": daily, "fleet_size": fleet, "delivery_mode": spec.delivery_mode},
                seed,
                lambda: run_cost([metas[s] for s in daily], fleet, spec.delivery_mode, prices),
                ev,
            )
        lg = cmp["sets"]["sufficient"]["models"]["lightgbm"]
        # recall uncertainty proxy: AUC CI half-width scaled; a direct recall bootstrap is a
        # later refinement
        rec_hw = (lg["auc"]["hi"] - lg["auc"]["lo"]) / 2 * 1.5
        insp = Range(**prices["operations"]["inspection_cost_per_alert"])
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
        fa_rate = ch.get("false_alerts_per_100_vehicle_months")
        args = (
            spec.horizon_days,
            insp,
            cs["monthly"]["marginal_total"],
            cs["monthly"]["build_amortized"],
            seed,
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
                "false_alerts_per_100_vehicle_months": fa_rate,
                "run_cost_month": cs["monthly"]["marginal_total"],
            },
            seed,
            lambda: roi_distribution(
                spec.value, rate, recall, ch["alert_rate"], *args, false_alerts_per_100vm=fa_rate
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
                spec.value, rate, recall, ch["alert_rate"], *args, false_alerts_per_100vm=fa_rate
            ),
            ev,
        )
        # replan predicate: borderline economics -> widen value ranges and re-run ROI
        if roi["p_roi_positive"] is not None and 0.4 <= roi["p_roi_positive"] <= 0.6:
            replans.append("roi_borderline")
            v = spec.value
            wide = v.model_copy(
                update={
                    "usd_per_avoided_event": Range(
                        low=v.usd_per_avoided_event.low * 0.5,
                        base=v.usd_per_avoided_event.base,
                        high=v.usd_per_avoided_event.high * 1.5,
                    ),
                    "preventable_fraction": Range(
                        low=max(0.0, v.preventable_fraction.low * 0.5),
                        base=v.preventable_fraction.base,
                        high=min(1.0, v.preventable_fraction.high * 1.5),
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
                    wide, rate, recall, ch["alert_rate"], *args, false_alerts_per_100vm=fa_rate
                ),
                ev,
            )
        self.ledger.end_step(sid, note=";".join(r for r in replans if r.startswith("roi")) or None)

    def _delivery(self, run_id, spec, ev) -> None:
        sid = self.ledger.start_step(run_id, "DELIVERY")
        self._rec(
            run_id,
            "DELIVERY",
            "deployment",
            "deployment_fit",
            {"delivery_mode": spec.delivery_mode, "horizon_days": spec.horizon_days},
            spec.seed,
            lambda: deployment_fit(spec),
            ev,
        )
        self.ledger.end_step(sid)

    def _policy(self, run_id, spec, ev) -> Verdict:
        sid = self.ledger.start_step(run_id, "POLICY")
        bundle = EvidenceBundle()
        for name in POLICY_INPUTS:
            if name in ev:
                bundle.put(name, ev[name].outputs, ev[name].evidence_id)
        verdict = decide(spec, bundle)
        self._rec(
            run_id,
            "POLICY",
            "verdict",
            "decision_policy",
            {"policy_version": verdict.policy_version},
            spec.seed,
            lambda: json.loads(verdict.model_dump_json()),
            ev,
        )
        self.ledger.end_step(sid, note=verdict.decision)
        return verdict

    def _report(self, run_id, spec, ev, verdict, llm_used, narrative) -> tuple[Brief, str]:
        sid = self.ledger.start_step(run_id, "REPORT")
        brief = build_brief(spec, run_id, self.source.dataset_hash, ev, verdict, llm_used=llm_used)
        if narrative is not None:
            brief = brief.model_copy(update={"narrative": tuple(narrative(ev, verdict))})
        md = to_markdown(brief)
        self.ledger.finish_run(run_id, verdict, md, to_json(brief))
        self.ledger.end_step(sid)
        return brief, md

    # ------------------------------------------------------------------ derived runs
    def whatif(
        self,
        parent_run_id: str,
        value: ValueAssumptions | None = None,
        price_overrides: dict[str, Any] | None = None,
        constraints: dict[str, Any] | None = None,
    ) -> RunResult:
        """Re-run ECONOMICS -> POLICY -> REPORT on a finished study with new human inputs.
        Harness evidence is reused (and cited) from the parent; economics is recomputed and
        recorded under the new run. Seconds, not minutes."""
        parent = self.ledger.get_run(parent_run_id)
        if parent is None or parent["status"] != "done":
            raise KeyError(f"run {parent_run_id} not found or not finished")
        if parent["dataset_hash"] != self.source.dataset_hash:
            raise ValueError("what-if must run on the parent's dataset")
        spec = ProblemSpec.model_validate(parent["spec"])
        update: dict[str, Any] = {}
        if value is not None:
            update["value"] = value
        if constraints:
            update["constraints"] = spec.constraints.model_copy(update=constraints)
        spec = spec.model_copy(update=update)
        prices = load_price_sheet()
        if price_overrides:
            prices = _deep_merge(prices, price_overrides)
        ev = self.ledger.evidence_objects(parent_run_id)
        usable = ev["usable"].outputs["usable"]
        suff = ev["ablation"].outputs["sufficient_set"]
        er = ev["event_rate"].outputs
        rate = Range(
            low=er["rate_ci_lo"],
            base=er["rate_per_vehicle_year"],
            high=max(er["rate_ci_hi"], er["rate_per_vehicle_year"]),
        )
        run_id = uuid.uuid4().hex[:12]
        self.ledger.create_run(
            run_id,
            spec,
            self.source.dataset_hash,
            False,
            None,
            derived_from=parent_run_id,
            kind="whatif",
        )
        replans: list[str] = []
        try:
            sid = self.ledger.start_step(run_id, "DEFINE")
            self._rec(
                run_id,
                "DEFINE",
                "spec",
                "problem_spec",
                {"spec": json.loads(spec.model_dump_json()), "derived_from": parent_run_id},
                spec.seed,
                lambda: json.loads(spec.model_dump_json()),
                ev,
            )
            if price_overrides:
                self._rec(
                    run_id,
                    "DEFINE",
                    "price_overrides",
                    "price_overrides",
                    {"overrides": price_overrides},
                    spec.seed,
                    lambda: {"overrides": price_overrides},
                    ev,
                )
            self.ledger.end_step(sid)
            self._economics(run_id, spec, ev, usable, suff, rate, prices, replans)
            self._delivery(run_id, spec, ev)
            verdict = self._policy(run_id, spec, ev)
            brief, md = self._report(run_id, spec, ev, verdict, False, None)
            return RunResult(run_id, verdict, brief, md, ev, replans)
        except Exception as exc:
            self.ledger.finish_run(run_id, None, None, None, error=f"{type(exc).__name__}: {exc}")
            raise

    def replay(self, parent_run_id: str) -> RunResult:
        """Re-run a study from its stored spec and diff every evidence record against the
        original. Identical outputs prove the reliability contract on this dataset."""
        parent = self.ledger.get_run(parent_run_id)
        if parent is None:
            raise KeyError(parent_run_id)
        if parent["dataset_hash"] != self.source.dataset_hash:
            raise ValueError(
                f"replay needs dataset {parent['dataset_hash']}, runner has {self.source.dataset_hash}"
            )
        spec = ProblemSpec.model_validate(parent["spec"])
        # a replay must recompute: the fit cache is bypassed for its duration
        was_enabled = self.store.fits.enabled
        self.store.fits.enabled = False
        try:
            res = self.run(spec, kind="replay", derived_from=parent_run_id)
        finally:
            self.store.fits.enabled = was_enabled
        before = self.ledger.evidence_objects(parent_run_id)
        diff = diff_evidence(before, res.evidence)
        if "runtime" in before and "runtime" in res.evidence:
            same_env = runtime.same_numeric_environment(
                before["runtime"].outputs, res.evidence["runtime"].outputs
            )
            diff["environment_identical"] = same_env
            if not same_env:
                diff["note"] = (
                    "library or code versions differ from the original run; LightGBM does not "
                    "guarantee identical arithmetic across versions, so numeric differences are "
                    "expected and are not a determinism defect"
                )
        res.diff = diff
        self.ledger.add_message(
            res.run_id, "system", json.dumps({"replay_diff": diff}, default=str), []
        )
        return res


PROVENANCE_KEYS = frozenset({"evidence_ids", "evidence_id", "run_id"})
# operational records: how the study was computed, not what it found. They differ between a
# run and its replay by construction (cache bypassed, thread counts) and are reported, not
# compared; the numeric environment is compared separately via `environment_identical`
INFORMATIONAL_RECORDS = frozenset({"compute", "runtime"})


def _substance(x: Any) -> Any:
    """Outputs with provenance references removed: evidence ids embed the run id, so they
    differ between a run and its replay by construction and are not part of the finding."""
    if isinstance(x, dict):
        return {k: _substance(v) for k, v in x.items() if k not in PROVENANCE_KEYS}
    if isinstance(x, list):
        return [_substance(v) for v in x]
    return x


def diff_evidence(a: dict[str, Evidence], b: dict[str, Evidence]) -> dict[str, Any]:
    """Evidence-by-evidence comparison of two runs' outputs (ids are expected to differ)."""
    names = sorted(set(a) | set(b))
    rows = []
    identical = True
    for n in names:
        if n not in a or n not in b:
            rows.append(
                {"name": n, "status": "missing_in_" + ("replay" if n not in b else "original")}
            )
            identical = False
            continue
        oa, ob = _substance(a[n].outputs), _substance(b[n].outputs)
        same = oa == ob
        paths = [] if same else _first_diffs(oa, ob)
        if n in INFORMATIONAL_RECORDS:
            rows.append({"name": n, "status": "informational", "paths": paths})
            continue
        if not same:
            identical = False
        rows.append({"name": n, "status": "identical" if same else "differs", "paths": paths})
    return {"identical": identical, "records": rows}


def _first_diffs(x: Any, y: Any, path: str = "", out: list | None = None, limit: int = 5) -> list:
    out = [] if out is None else out
    if len(out) >= limit:
        return out
    if isinstance(x, dict) and isinstance(y, dict):
        for k in sorted(set(x) | set(y)):
            _first_diffs(x.get(k), y.get(k), f"{path}.{k}" if path else str(k), out, limit)
    elif isinstance(x, list) and isinstance(y, list):
        if len(x) != len(y):
            out.append({"path": path, "a": f"len {len(x)}", "b": f"len {len(y)}"})
        for i, (p, q) in enumerate(zip(x, y, strict=False)):
            _first_diffs(p, q, f"{path}[{i}]", out, limit)
    elif x != y:
        out.append({"path": path, "a": x, "b": y})
    return out


def _deep_merge(base: dict, over: dict) -> dict:
    out = dict(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out

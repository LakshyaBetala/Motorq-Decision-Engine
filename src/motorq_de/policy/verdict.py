"""Decision policy v1.0 — a pure function of evidence.

    HARD GATES         any failure -> NOT_FEASIBLE
      data       fleet share covered by the full sufficient set >= constraints.min_oem_coverage
      model      lower CI bound of (AUC_model - AUC_best_baseline) > 0
      economics  P(ROI > 0) >= 0.5
      delivery   at least one deployment pattern meets the constraints
      cost       (only when constraints.max_run_cost_usd_month is set) marginal monthly run cost
                 of the sufficient set <= the ceiling

    UNCERTAINTY FLAGS  any trip -> PILOT instead of BUILD_READY
      cross_oem_variance      std of leave-one-OEM-out AUC > 0.03, or min < mean - 0.05
      temporal_degradation    CV AUC - forward AUC (upper bound) > 0.03
      roi_spans_negative      5th percentile ROI < 0
      value_unvalidated       value assumptions not validated by a pilot
      short_history           a sufficient-set signal has < min_signal_history_months of data
      ablation_underpowered   an accepted removal rests on a bootstrap that cannot resolve tolerance/2
      suspicious_signals      quality flagged a sufficient-set signal as 'confirm availability'
      cost_placeholders       run cost rests on placeholder unit prices (always true until
                              Motorq's contracted rates replace the price sheet)
      alert_burden            false-alert episodes per 100 vehicle-months above the ceiling

Thresholds live in policy.yaml (versioned); the brief stamps the version.

The LLM never touches this. It can only add evidence.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from motorq_de.schemas import FlagResult, GateResult, ProblemSpec, Verdict

_POLICY = yaml.safe_load((Path(__file__).parent / "policy.yaml").read_text(encoding="utf-8"))
POLICY_VERSION = str(_POLICY["version"])
THRESHOLDS: dict[str, float] = {**_POLICY["gates"], **_POLICY["flags"]}


class EvidenceBundle:
    """The subset of tool outputs the policy reads, each tagged with its evidence_id."""

    def __init__(self) -> None:
        self.items: dict[str, tuple[dict[str, Any], str]] = {}

    def put(self, name: str, outputs: dict[str, Any], evidence_id: str) -> None:
        self.items[name] = (outputs, evidence_id)

    def get(self, name: str) -> tuple[dict[str, Any] | None, tuple[str, ...]]:
        if name not in self.items:
            return None, ()
        out, eid = self.items[name]
        return out, (eid,)


def decide(spec: ProblemSpec, ev: EvidenceBundle) -> Verdict:
    gates: list[GateResult] = []
    flags: list[FlagResult] = []
    c = spec.constraints

    # ------------------------------------------------------------------ gates
    cov, cov_ids = ev.get("coverage_sufficient")
    share = cov["fleet_share_full_set"] if cov else None
    gates.append(
        GateResult(
            name="data",
            passed=bool(share is not None and share >= c.min_oem_coverage),
            value=share,
            threshold=c.min_oem_coverage,
            evidence_ids=cov_ids,
            note="" if cov else "coverage evidence missing",
        )
    )

    cmp, cmp_ids = ev.get("model_comparison")
    lift_lo = None
    if cmp:
        # first set is the sufficient set by convention; compare to its best single-signal baseline
        first = next(iter(cmp["sets"].values()))
        auc = first["models"]["lightgbm"]["auc"]
        base = first["baselines"]["best_single_signal_auc"] or 0.5
        lift_lo = auc["lo"] - base
    gates.append(
        GateResult(
            name="model",
            passed=bool(lift_lo is not None and lift_lo > THRESHOLDS["model_lift_ci_lo"]),
            value=lift_lo,
            threshold=THRESHOLDS["model_lift_ci_lo"],
            evidence_ids=cmp_ids,
            note="lower CI bound of AUC minus best single-signal baseline",
        )
    )

    roi, roi_ids = ev.get("roi")
    p_pos = roi["p_roi_positive"] if roi else None
    gates.append(
        GateResult(
            name="economics",
            passed=bool(p_pos is not None and p_pos >= THRESHOLDS["p_roi_positive"]),
            value=p_pos,
            threshold=THRESHOLDS["p_roi_positive"],
            evidence_ids=roi_ids,
        )
    )

    # cost ceiling: only a gate when the request states one; the marginal monthly run cost of
    # the sufficient set must not exceed it
    cost, cost_ids = ev.get("cost_sufficient")
    marginal = ((cost or {}).get("monthly") or {}).get("marginal_total")
    if c.max_run_cost_usd_month is not None:
        gates.append(
            GateResult(
                name="cost",
                passed=bool(marginal is not None and marginal <= c.max_run_cost_usd_month),
                value=marginal,
                threshold=c.max_run_cost_usd_month,
                evidence_ids=cost_ids,
                note="marginal monthly run cost of the sufficient set vs the requested ceiling",
            )
        )

    dep, dep_ids = ev.get("deployment")
    gates.append(
        GateResult(
            name="delivery",
            passed=bool(dep and dep["any_feasible"]),
            value=None,
            threshold=None,
            evidence_ids=dep_ids,
            note=dep["recommended_pattern"]
            if dep and dep["recommended_pattern"]
            else "no feasible pattern",
        )
    )

    # ------------------------------------------------------------------ flags
    xo, xo_ids = ev.get("cross_oem")
    if xo and xo.get("std_auc") is not None:
        gap = (xo["mean_auc"] - xo["min_auc"]) if xo.get("min_auc") is not None else 0.0
        tripped = (
            xo["std_auc"] > THRESHOLDS["cross_oem_std"] or gap > THRESHOLDS["cross_oem_min_gap"]
        )
        flags.append(
            FlagResult(
                name="cross_oem_variance",
                tripped=bool(tripped),
                value=xo["std_auc"],
                threshold=THRESHOLDS["cross_oem_std"],
                evidence_ids=xo_ids,
                note=f"worst OEM {xo.get('worst_oem')} at AUC {xo.get('min_auc')}"
                if tripped
                else "",
            )
        )

    tv, tv_ids = ev.get("temporal")
    if tv and "degradation_upper" in tv:
        flags.append(
            FlagResult(
                name="temporal_degradation",
                tripped=bool(tv["degradation_upper"] > THRESHOLDS["temporal_degradation_upper"]),
                value=tv["degradation_upper"],
                threshold=THRESHOLDS["temporal_degradation_upper"],
                evidence_ids=tv_ids,
            )
        )

    if roi:
        flags.append(
            FlagResult(
                name="roi_spans_negative",
                tripped=bool(roi["roi"]["p5"] < THRESHOLDS["roi_p5"]),
                value=roi["roi"]["p5"],
                threshold=THRESHOLDS["roi_p5"],
                evidence_ids=roi_ids,
            )
        )

    flags.append(
        FlagResult(
            name="value_unvalidated",
            tripped=not spec.value.validated_by_pilot,
            value=None,
            threshold=None,
            evidence_ids=(),
            note="value assumptions are human-supplied and not yet validated against pilot outcomes",
        )
    )

    q, q_ids = ev.get("quality_sufficient")
    if q:
        short = [
            s
            for s, r in q["per_signal"].items()
            if r["history_months"] < c.min_signal_history_months
        ]
        flags.append(
            FlagResult(
                name="short_history",
                tripped=bool(short),
                value=float(
                    min((r["history_months"] for r in q["per_signal"].values()), default=0.0)
                ),
                threshold=float(c.min_signal_history_months),
                evidence_ids=q_ids,
                note=", ".join(short),
            )
        )

    ab, ab_ids = ev.get("ablation")
    if ab:
        flags.append(
            FlagResult(
                name="ablation_underpowered",
                tripped=bool(ab.get("underpowered")),
                value=ab.get("resolution"),
                threshold=ab.get("tolerance", 0.005) / 2,
                evidence_ids=ab_ids,
                note="an accepted removal could not be resolved to tolerance/2; more positives needed"
                if ab.get("underpowered")
                else "",
            )
        )

    lk, lk_ids = ev.get("leakage")
    if lk and ab:
        susp = sorted(set(lk.get("suspicious", [])) & set(ab["sufficient_set"]))
        flags.append(
            FlagResult(
                name="suspicious_signals",
                tripped=bool(susp),
                value=float(len(susp)),
                threshold=0.0,
                evidence_ids=lk_ids,
                note=", ".join(susp)
                + (" - confirm availability at prediction time" if susp else ""),
            )
        )

    op, op_ids = ev.get("operating_point")
    if op and op.get("chosen", {}).get("false_alerts_per_100_vehicle_months") is not None:
        fa = op["chosen"]["false_alerts_per_100_vehicle_months"]
        flags.append(
            FlagResult(
                name="alert_burden",
                tripped=bool(fa > THRESHOLDS["false_alerts_per_100_vehicle_months_max"]),
                value=fa,
                threshold=THRESHOLDS["false_alerts_per_100_vehicle_months_max"],
                evidence_ids=op_ids,
                note="false-alert episodes per 100 vehicle-months at the chosen operating point",
            )
        )

    cost, cost_ids = ev.get("cost_sufficient")
    if cost:
        flags.append(
            FlagResult(
                name="cost_placeholders",
                tripped=bool(cost.get("placeholders")),
                value=float(len(cost.get("placeholders", []))),
                threshold=0.0,
                evidence_ids=cost_ids,
                note=", ".join(cost.get("placeholders", [])),
            )
        )

    # ------------------------------------------------------------------ verdict
    if not all(g.passed for g in gates):
        decision = "NOT_FEASIBLE"
    elif any(f.tripped for f in flags):
        decision = "PILOT"
    else:
        decision = "BUILD_READY"
    return Verdict(
        decision=decision, gates=tuple(gates), flags=tuple(flags), policy_version=POLICY_VERSION
    )

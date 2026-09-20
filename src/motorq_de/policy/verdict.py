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
      data_still_improving    learning curve still rising from half to all vehicles (more data
                              would raise the AUC; the reported figure is a lower bound)
      seed_sensitive          sufficient-set AUC moves by more than the ablation tolerance
                              under a different fold assignment

Thresholds live in policy.yaml (versioned); the brief stamps the version. `sensitivity`
reports how far every gate and flag sits from its threshold, in units of the step sizes
policy.yaml declares, and whether the verdict survives every threshold moving one step
against it. The thresholds are not adaptive: they are fixed, versioned and reviewed; the
sensitivity record is what makes that review concrete.

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
STEPS: dict[str, float] = dict(_POLICY.get("sensitivity_steps", {}))
# threshold key and direction for every numeric gate/flag: "min" passes (or does not trip)
# when value >= threshold, "max" when value <= threshold. Keys that are request constraints
# live on spec.constraints rather than in THRESHOLDS.
GATE_KEYS: dict[str, tuple[str, str]] = {
    "data": ("min_oem_coverage", "min"),
    "model": ("model_lift_ci_lo", "min"),
    "economics": ("p_roi_positive", "min"),
    "cost": ("max_run_cost_usd_month", "max"),
}
FLAG_KEYS: dict[str, tuple[str, str]] = {
    "cross_oem_variance": ("cross_oem_std", "max"),
    "temporal_degradation": ("temporal_degradation_upper", "max"),
    "roi_spans_negative": ("roi_p5", "min"),
    "short_history": ("min_signal_history_months", "min"),
    "alert_burden": ("false_alerts_per_100_vehicle_months_max", "max"),
    "data_still_improving": ("learning_curve_gain_half_to_full", "max"),
    "seed_sensitive": ("seed_auc_spread_max", "max"),
}
CONSTRAINT_KEYS = ("min_oem_coverage", "max_run_cost_usd_month", "min_signal_history_months")


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


def decide(
    spec: ProblemSpec, ev: EvidenceBundle, thresholds: dict[str, float] | None = None
) -> Verdict:
    T = THRESHOLDS if thresholds is None else thresholds
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
            passed=bool(lift_lo is not None and lift_lo > T["model_lift_ci_lo"]),
            value=lift_lo,
            threshold=T["model_lift_ci_lo"],
            evidence_ids=cmp_ids,
            note="lower CI bound of AUC minus best single-signal baseline",
        )
    )

    roi, roi_ids = ev.get("roi")
    p_pos = roi["p_roi_positive"] if roi else None
    gates.append(
        GateResult(
            name="economics",
            passed=bool(p_pos is not None and p_pos >= T["p_roi_positive"]),
            value=p_pos,
            threshold=T["p_roi_positive"],
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
        tripped = xo["std_auc"] > T["cross_oem_std"] or gap > T["cross_oem_min_gap"]
        flags.append(
            FlagResult(
                name="cross_oem_variance",
                tripped=bool(tripped),
                value=xo["std_auc"],
                threshold=T["cross_oem_std"],
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
                tripped=bool(tv["degradation_upper"] > T["temporal_degradation_upper"]),
                value=tv["degradation_upper"],
                threshold=T["temporal_degradation_upper"],
                evidence_ids=tv_ids,
            )
        )

    if roi:
        flags.append(
            FlagResult(
                name="roi_spans_negative",
                tripped=bool(roi["roi"]["p5"] < T["roi_p5"]),
                value=roi["roi"]["p5"],
                threshold=T["roi_p5"],
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

    lc, lc_ids = ev.get("learning_curve")
    if lc:
        gain = lc.get("auc_gain_half_to_full")
        rising = gain is not None and gain > T["learning_curve_gain_half_to_full"]
        flags.append(
            FlagResult(
                name="data_still_improving",
                tripped=bool(rising),
                value=gain,
                threshold=T["learning_curve_gain_half_to_full"],
                evidence_ids=lc_ids,
                note="AUC still rising from half to all vehicles; the reported AUC is a lower bound"
                if rising
                else "",
            )
        )

    ss, ss_ids = ev.get("seed_stability")
    if ss and ss.get("auc_spread") is not None:
        wobbly = ss["auc_spread"] > T["seed_auc_spread_max"]
        flags.append(
            FlagResult(
                name="seed_sensitive",
                tripped=bool(wobbly),
                value=ss["auc_spread"],
                threshold=T["seed_auc_spread_max"],
                evidence_ids=ss_ids,
                note="sufficient-set AUC spread across fold assignments; the reported figure "
                "is fold-assignment noise to this degree"
                if wobbly
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
                tripped=bool(fa > T["false_alerts_per_100_vehicle_months_max"]),
                value=fa,
                threshold=T["false_alerts_per_100_vehicle_months_max"],
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


# ------------------------------------------------------------------ sensitivity


def _margin(value: float | None, threshold: float | None, direction: str) -> float | None:
    """Signed distance from the threshold; positive means on the passing (non-tripping) side."""
    if value is None or threshold is None:
        return None
    return float(value - threshold) if direction == "min" else float(threshold - value)


def _shift(spec: ProblemSpec, sign: int) -> tuple[ProblemSpec, dict[str, float]]:
    """Every threshold with a declared step moved one step: sign=+1 against the capability
    (harder to pass, easier to trip), sign=-1 in its favour."""
    T = dict(THRESHOLDS)
    cons: dict[str, Any] = {}
    for key, direction in {**GATE_KEYS, **FLAG_KEYS}.values():
        if key not in STEPS:
            continue
        # a "min" threshold gets harder by rising, a "max" one by falling
        delta = STEPS[key] * sign * (1 if direction == "min" else -1)
        if key in CONSTRAINT_KEYS:
            cur = getattr(spec.constraints, key)
            if cur is None:
                continue
            if key == "max_run_cost_usd_month":
                delta = -cur * STEPS[key] * sign  # the cost step is a fraction of the ceiling
            cons[key] = type(cur)(cur + delta)
        else:
            T[key] = T[key] + delta
    spec2 = spec.model_copy(update={"constraints": spec.constraints.model_copy(update=cons)})
    return spec2, T


def sensitivity(spec: ProblemSpec, ev: EvidenceBundle) -> dict[str, Any]:
    """How close the verdict is to a different one.

    For every gate and flag: the margin to its threshold and that margin in step units
    (policy.yaml `sensitivity_steps`), so a reviewer can see which gate binds. Then the
    verdict is recomputed with every stepped threshold moved one step against the
    capability and one step in its favour; `robust` is true when neither changes it."""
    base = decide(spec, ev)
    gates = []
    for g in base.gates:
        key, direction = GATE_KEYS.get(g.name, (None, "min"))
        m = _margin(g.value, g.threshold, direction)
        step = STEPS.get(key) if key else None
        if key == "max_run_cost_usd_month" and step and g.threshold:
            step = step * g.threshold
        gates.append(
            {
                "name": g.name,
                "passed": g.passed,
                "value": g.value,
                "threshold": g.threshold,
                "margin": None if m is None else round(m, 6),
                "margin_steps": None if m is None or not step else round(m / step, 3),
            }
        )
    flags = []
    for f in base.flags:
        key, direction = FLAG_KEYS.get(f.name, (None, "max"))
        m = _margin(f.value, f.threshold, direction)
        step = STEPS.get(key) if key else None
        flags.append(
            {
                "name": f.name,
                "tripped": f.tripped,
                "value": f.value,
                "threshold": f.threshold,
                "margin": None if m is None else round(m, 6),
                "margin_steps": None if m is None or not step else round(m / step, 3),
            }
        )
    stepped = [g for g in gates if g["passed"] and g["margin_steps"] is not None]
    binding = min(stepped, key=lambda g: g["margin_steps"])["name"] if stepped else None

    def under(sign: int) -> dict[str, Any]:
        spec2, T2 = _shift(spec, sign)
        v = decide(spec2, ev, T2)
        changed = [
            g.name for g, g0 in zip(v.gates, base.gates, strict=True) if g.passed != g0.passed
        ]
        changed += [
            f.name for f, f0 in zip(v.flags, base.flags, strict=True) if f.tripped != f0.tripped
        ]
        return {"decision": v.decision, "changed": changed}

    against, favour = under(+1), under(-1)
    return {
        "policy_version": POLICY_VERSION,
        "decision": base.decision,
        "gates": gates,
        "flags": flags,
        "binding_gate": binding,
        "failed_gates": [g["name"] for g in gates if not g["passed"]],
        "steps": STEPS,
        "one_step_against": against,
        "one_step_in_favour": favour,
        "robust": bool(against["decision"] == base.decision == favour["decision"]),
    }

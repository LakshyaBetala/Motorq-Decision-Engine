"""The evidence brief: typed, every numeric line cites evidence, rendered fail-closed.

`build_brief` turns the run's Evidence objects and Verdict into template lines. The LLM
may add narrative through `sanitize_narrative`, which drops any sentence that contains a
number but no valid `[ev_...]` citation.
"""

from __future__ import annotations

import re
from datetime import UTC, datetime
from typing import Any

from motorq_de.schemas import Evidence, Frozen, ProblemSpec, Verdict

CITE_RE = re.compile(r"\[(ev_[0-9a-f]{12})(?:,\s*ev_[0-9a-f]{12})*\]")
NUM_RE = re.compile(r"\d")


class Line(Frozen):
    text: str
    evidence_ids: tuple[str, ...] = ()

    def __init__(
        self, text: str = "", evidence_ids: tuple[str, ...] = (), **kw: Any
    ) -> None:  # positional-friendly
        ids = tuple(dict.fromkeys(kw.pop("evidence_ids", evidence_ids)))  # dedupe, keep order
        super().__init__(text=kw.pop("text", text), evidence_ids=ids, **kw)


class Section(Frozen):
    title: str
    lines: tuple[Line, ...]


class Brief(Frozen):
    run_id: str
    dataset_hash: str
    capability_name: str
    target_event: str
    horizon_days: int
    decision: str
    policy_version: str
    sections: tuple[Section, ...]
    narrative: tuple[Line, ...] = ()
    generated_at: str
    llm_used: bool = False


class UncitedClaimError(ValueError):
    pass


def _fmt(x: float | None, nd: int = 3) -> str:
    return "n/a" if x is None else f"{x:.{nd}f}"


def _ci(d: dict[str, Any] | None, nd: int = 3) -> str:
    if not d:
        return "n/a"
    return f"{d['point']:.{nd}f} [{d['lo']:.{nd}f}, {d['hi']:.{nd}f}]"


def _usd(x: float | None) -> str:
    return "n/a" if x is None else f"${x:,.0f}"


def build_brief(
    spec: ProblemSpec,
    run_id: str,
    dataset_hash: str,
    ev: dict[str, Evidence],
    verdict: Verdict,
    llm_used: bool = False,
) -> Brief:
    E = lambda k: ev[k].outputs if k in ev else None  # noqa: E731
    I = lambda k: (ev[k].evidence_id,) if k in ev else ()  # noqa: E731, E741
    sections: list[Section] = []

    # --------------------------------------------------------------- spec
    sections.append(
        Section(
            title="Specification",
            lines=(
                Line(f"Capability: {spec.capability_name}"),
                Line(
                    f"Target: {spec.target_event}; horizon {spec.horizon_days} days; unit {spec.decision_unit}; delivery {spec.delivery_mode}; consumer {spec.consumer}",
                    I("spec"),
                ),
                Line(
                    "Value assumptions (human-supplied): value-bearing fraction of target events "
                    f"[{spec.value.value_bearing_fraction.low}, {spec.value.value_bearing_fraction.base}, {spec.value.value_bearing_fraction.high}]; "
                    f"preventable fraction [{spec.value.preventable_fraction.low}, {spec.value.preventable_fraction.base}, {spec.value.preventable_fraction.high}]; "
                    f"$/avoided event [{spec.value.usd_per_avoided_event.low:,.0f}, {spec.value.usd_per_avoided_event.base:,.0f}, {spec.value.usd_per_avoided_event.high:,.0f}]; "
                    f"fleet [{spec.value.fleet_size.low:,.0f}, {spec.value.fleet_size.base:,.0f}, {spec.value.fleet_size.high:,.0f}]"
                    + (f". Source: {spec.value.source_note}" if spec.value.source_note else ""),
                    I("spec"),
                ),
            ),
        )
    )

    # --------------------------------------------------------------- data
    lk, cov_s, q, us, truth = (
        E("leakage"),
        E("coverage_sufficient"),
        E("quality_sufficient"),
        E("usable"),
        E("dataset_truth"),
    )
    lines = []
    if truth:
        r = truth["achieved_rates"]
        lines.append(
            Line(
                f"Dataset {dataset_hash}: synthetic, {truth.get('n_vehicles', 'n/a')} vehicles, {truth.get('n_days', 'n/a')} days; achieved rates brake {_fmt(r.get('brake_service_events_per_vehicle_year'))}/veh-yr, theft {_fmt(r.get('theft_events_per_vehicle_year'))}/veh-yr, battery {_fmt(r.get('battery_events_per_ev_year'))}/EV-yr",
                I("dataset_truth"),
            )
        )
        for d in truth.get("disclosures", {}).values():
            lines.append(Line(f"Disclosure: {d}", I("dataset_truth")))
    er = E("event_rate")
    if er:
        lines.append(
            Line(
                f"Measured target-event rate ({er['population_powertrain']} population, {er['n_vehicles']:,} vehicles, {er['vehicle_years']:,.0f} vehicle-years, {er['n_events']:,} events): {er['rate_per_vehicle_year']:.4f}/vehicle-year [{er['rate_ci_lo']:.4f}, {er['rate_ci_hi']:.4f}]; {er['events_per_vehicle_per_horizon']:.4f} per vehicle per {spec.horizon_days}-day horizon",
                I("event_rate"),
            )
        )
    if us:
        lines.append(
            Line(
                f"{us['n_in']} candidate signals; {us['n_usable']} usable after quality and leakage filters; dropped: "
                + (", ".join(f"{k} ({v})" for k, v in us["dropped"].items()) or "none"),
                I("usable"),
            )
        )
    if lk:
        if lk.get("flagged"):
            lines.append(
                Line(
                    "Leakage detected and excluded: "
                    + ", ".join(
                        f"{s} ({', '.join(lk['per_signal'][s]['reasons'])})" for s in lk["flagged"]
                    ),
                    I("leakage"),
                )
            )
        if lk.get("suspicious"):
            lines.append(
                Line(
                    "Unusually strong signals kept - confirm they are produced before, not because of, the event: "
                    + ", ".join(lk["suspicious"]),
                    I("leakage"),
                )
            )
        lines.append(
            Line(f"Positive rate at this horizon: {lk['positive_rate']:.4f}", I("leakage"))
        )
    if cov_s:
        lines.append(
            Line(
                f"Fleet share with the full sufficient signal set: {cov_s['fleet_share_full_set']:.3f} (threshold {spec.constraints.min_oem_coverage})",
                I("coverage_sufficient"),
            )
        )
        by = cov_s["fleet_share_full_set_by_oem"]
        lines.append(
            Line(
                "Coverage by OEM: " + ", ".join(f"{o} {v:.2f}" for o, v in sorted(by.items())),
                I("coverage_sufficient"),
            )
        )
        if cov_s.get("oems_missing_any_signal"):
            lines.append(
                Line(
                    "OEMs lacking at least one required signal: "
                    + ", ".join(cov_s["oems_missing_any_signal"]),
                    I("coverage_sufficient"),
                )
            )
    if q and q.get("flags"):
        lines.append(
            Line(
                "Quality flags on sufficient-set signals: "
                + "; ".join(f"{k}: {', '.join(v)}" for k, v in q["flags"].items()),
                I("quality_sufficient"),
            )
        )
    sections.append(Section(title="Data feasibility", lines=tuple(lines)))

    # --------------------------------------------------------------- model
    fa, ab, cmp = E("feature_analysis"), E("ablation"), E("model_comparison")
    lines = []
    if fa:
        top = fa["ranking"][:8]
        lines.append(
            Line(
                f"Full usable set ({fa['n_signals']} signals, {fa['n_features']} features): AUC {_ci(fa['auc'])} on {fa['n_rows']:,} sampled vehicle-days ({fa['n_pos']:,} positives; grouped {fa['n_splits']}-fold x {fa['n_repeats']})",
                I("feature_analysis"),
            )
        )
        lines.append(
            Line(
                "Top signals by permutation importance (AUC drop, 95% CI): "
                + "; ".join(
                    f"{r['signal']} {r['perm_importance']:+.4f} [{r['perm_ci_lo']:+.4f}, {r['perm_ci_hi']:+.4f}]"
                    for r in top
                ),
                I("feature_analysis"),
            )
        )
    if ab:
        lines.append(
            Line(
                f"Ablation (screen to top {ab['screen_k']}, then ordered elimination, tolerance {ab['tolerance']} AUC at {ab['noninferiority_confidence']:.0%} bootstrap confidence): sufficient set of {len(ab['sufficient_set'])} signals - "
                + ", ".join(ab["sufficient_set"]),
                I("ablation"),
            )
        )
        lines.append(
            Line(
                f"Full-candidate AUC {_ci(ab['full_auc'])} vs sufficient-set AUC {_ci(ab['sufficient_auc'])}; {len(ab['removed_in_order'])} signals removed; stop: {ab['stop_reason']}; {ab['n_fits']} model fits",
                I("ablation"),
            )
        )
        if ab.get("underpowered"):
            lines.append(
                Line(
                    f"Ablation is underpowered: bootstrap resolution {ab['resolution']:.4f} AUC exceeds tolerance/2; removals near the boundary rest on point estimates",
                    I("ablation"),
                )
            )
        if ab.get("kept_conservatively"):
            lines.append(
                Line(
                    f"Kept conservatively (drop not proven below tolerance, not proven above it): {', '.join(ab['kept_conservatively'])}",
                    I("ablation"),
                )
            )
    if cmp:
        for name, entry in cmp["sets"].items():
            lg = entry["models"]["lightgbm"]
            lo = entry["models"].get("logistic")
            b = entry["baselines"]
            lines.append(
                Line(
                    f"{name} ({entry['n_signals']} signals): LightGBM AUC {_ci(lg['auc'])}, PR-AUC {lg['pr_auc']:.3f}, Brier {lg['brier']:.4f}"
                    + (f"; logistic AUC {_ci(lo['auc'])}" if lo else "")
                    + f"; best single-signal model ({b['best_single_signal']}) AUC {_ci(b.get('best_single_signal_auc_ci'))} on the same population; majority 0.500",
                    I("model_comparison"),
                )
            )
        abd = E("ablation_daily_cadence")
        if abd and "daily_cadence" in cmp["sets"]:
            dc = cmp["sets"]["daily_cadence"]["models"]["lightgbm"]
            same = set(abd["sufficient_set"]) == set(ab["sufficient_set"]) if ab else False
            if same:
                lines.append(
                    Line(
                        "Cadence: the cost-aware sufficient set already needs no realtime polling - daily/weekly signals only",
                        I("ablation_daily_cadence") + I("ablation"),
                    )
                )
            else:
                lines.append(
                    Line(
                        f"Cadence ablation - daily/weekly signals only ({len(abd['sufficient_set'])} signals, no realtime polling): AUC {_ci(dc['auc'])} vs sufficient-set AUC {_ci(cmp['sets']['sufficient']['models']['lightgbm']['auc'])}; set: {', '.join(abd['sufficient_set'])}",
                        I("ablation_daily_cadence") + I("model_comparison"),
                    )
                )
        if cmp.get("paired"):
            p = cmp["paired"]
            lines.append(
                Line(
                    f"Paired delta AUC ({p['b']} minus {p['a']}): {_ci(p['delta_auc_b_minus_a'], 4)}",
                    I("model_comparison"),
                )
            )
    sections.append(Section(title="Model", lines=tuple(lines)))

    # --------------------------------------------------------------- robustness
    tv, xo = E("temporal"), E("cross_oem")
    lines = []
    if tv and "forward_auc" in tv:
        lines.append(
            Line(
                f"Forward split (train to {tv['train_end']}, {tv['gap_days']}-day gap, test from {tv['test_start']}): AUC {_ci(tv['forward_auc'])} vs CV {_ci(tv['cv_auc'])}; degradation {tv['degradation']:+.4f} (upper bound {tv['degradation_upper']:+.4f})",
                I("temporal"),
            )
        )
        if tv.get("forward_windows"):
            lines.append(
                Line(
                    "Forward windows: "
                    + "; ".join(
                        f"{w['start']}..{w['end']} AUC {w['auc']:.3f} ({w['n_pos']} pos)"
                        for w in tv["forward_windows"]
                    ),
                    I("temporal"),
                )
            )
    if xo and xo.get("mean_auc") is not None:
        per = xo["per_oem"]
        lines.append(
            Line(
                f"Leave-one-OEM-out over {xo['n_oems_evaluated']} OEMs: mean AUC {xo['mean_auc']:.3f}, std {_fmt(xo['std_auc'])}, min {xo['min_auc']:.3f} ({xo['worst_oem']})",
                I("cross_oem"),
            )
        )
        diag = {o: v.get("diagnosis") for o, v in per.items() if v.get("diagnosis")}
        if diag:
            lines.append(
                Line(
                    "Within-OEM benchmark (train and test on the OEM alone) vs held-out: "
                    + "; ".join(
                        f"{o} within {_ci(per[o]['within_oem_auc'])} / held-out {per[o]['auc']['point']:.3f} -> {d.replace('_', ' ')}"
                        for o, d in sorted(diag.items())
                    ),
                    I("cross_oem"),
                )
            )
        lines.append(
            Line(
                "Per OEM: "
                + "; ".join(
                    f"{o} {_ci(v['auc'])} (missing features {v['feature_missing_share']:.0%})"
                    for o, v in sorted(per.items())
                    if v.get("auc")
                ),
                I("cross_oem"),
            )
        )
    sections.append(Section(title="Robustness", lines=tuple(lines)))

    # --------------------------------------------------------------- economics
    cf, cs, cc, op, roi, tor = (
        E("cost_full"),
        E("cost_sufficient"),
        E("cost_compare"),
        E("operating_point"),
        E("roi"),
        E("tornado"),
    )
    lines = []
    if cs and cf and cc:
        m, mf = cs["monthly"], cf["monthly"]
        lines.append(
            Line(
                f"Run cost, sufficient set, {cs['fleet_size']:,.0f} vehicles, {cs['delivery_mode']}: infra {_usd(m['infra_total'])}/mo (ingest {_usd(m['ingest'])}, storage {_usd(m['storage'])}, compute {_usd(m['compute'])}, Kafka {_usd(m['kafka'])}), OEM API calls {_usd(m['oem_api_calls'])}/mo, build {_usd(cs['build_one_off_usd'])} one-off ({_usd(m['build_amortized'])}/mo amortised); marginal total {_usd(m['marginal_total'])}/mo; fully attributed (incl. OEM packages) {_usd(m['attributed_total'])}/mo",
                I("cost_sufficient"),
            )
        )
        lines.append(
            Line(
                f"Full usable set for comparison: {cc['gb_per_month_full']:.1f} GB/mo vs {cc['gb_per_month_reduced']:.1f} GB/mo ({(cc['infra_change_pct'] or 0):+.0%} infra); infra {_usd(mf['infra_total'])}/mo vs {_usd(m['infra_total'])}/mo; marginal {_usd(mf['marginal_total'])}/mo vs {_usd(m['marginal_total'])}/mo ({(cc['marginal_change_pct'] or 0):+.1%})",
                I("cost_compare"),
            )
        )
        lines.append(
            Line(
                cs["lever_note"]
                + f" Highest-cadence signals driving polling: {', '.join(cs['polling']['highest_cadence_signals'][:6])}",
                I("cost_sufficient"),
            )
        )
        if cs.get("placeholders"):
            lines.append(
                Line(
                    f"Price-sheet placeholders (replace with contracted rates): {', '.join(cs['placeholders'])}; sheet as of {cs['price_sheet_as_of']}",
                    I("cost_sufficient"),
                )
            )
    if op:
        ch = op["chosen"]
        basis = ch.get("recall_basis", "vehicle_day")
        ev_bits = ""
        if basis == "event":
            ev_bits = (
                f"; event-level: {ch['recall']:.3f} of events caught, median lead {_fmt(ch.get('median_lead_days'), 1)} days, "
                f"{_fmt(ch.get('false_alerts_per_100_vehicle_months'), 1)} false-alert episodes per 100 vehicle-months"
            )
        lines.append(
            Line(
                f"Operating point chosen to maximise median net value: alert on top {ch['alert_rate']:.2%} of vehicle-days -> precision {_fmt(ch.get('precision'))}{ev_bits}; median false-alert cost {_usd(ch['median_false_alert_cost_year'])}/yr",
                I("operating_point"),
            )
        )
    cd = E("cost_daily_cadence")
    if cd and cs and cd["monthly"]["marginal_total"] != cs["monthly"]["marginal_total"]:
        lines.append(
            Line(
                f"Daily-cadence-only set run cost: marginal {_usd(cd['monthly']['marginal_total'])}/mo vs {_usd(cs['monthly']['marginal_total'])}/mo for the sufficient set; OEM API calls {_usd(cd['monthly']['oem_api_calls'])}/mo vs {_usd(cs['monthly']['oem_api_calls'])}/mo",
                I("cost_daily_cadence") + I("cost_sufficient"),
            )
        )
    if roi:
        r = roi["roi"]
        lines.append(
            Line(
                f"ROI distribution ({roi['n_draws']:,} draws): P(ROI > 0) = {roi['p_roi_positive']:.2f}; ROI p5 {r['p5']:.2f}, p50 {r['p50']:.2f}, p95 {r['p95']:.2f}; net value/yr p5 {_usd(roi['net_value_year']['p5'])}, p50 {_usd(roi['net_value_year']['p50'])}, p95 {_usd(roi['net_value_year']['p95'])}; detected events/yr p50 {roi['detected_events_year']['p50']:,.0f}",
                I("roi"),
            )
        )
    if tor:
        lines.append(
            Line(
                f"Verdict is most sensitive to: {tor['dominant_input']}. Tornado (median ROI at input low -> high): "
                + "; ".join(
                    f"{x['input']} {x['roi_at_low']:.1f} -> {x['roi_at_high']:.1f}"
                    for x in tor["ranked"][:5]
                ),
                I("tornado"),
            )
        )
    sections.append(Section(title="Economics", lines=tuple(lines)))

    # --------------------------------------------------------------- delivery
    dep = E("deployment")
    lines = []
    if dep:
        lines.append(
            Line(
                f"Recommended pattern: {dep['recommended_pattern']} (requested {dep['requested_delivery_mode']}); feasible: {', '.join(c['pattern'] for c in dep['candidates'] if c['feasible'])}"
                + (f". {dep['note']}" if dep.get("note") else ""),
                I("deployment"),
            )
        )
    sections.append(Section(title="Delivery fit", lines=tuple(lines)))

    # --------------------------------------------------------------- verdict
    lines = [
        Line(
            f"Decision: {verdict.decision} (policy v{verdict.policy_version}). This states what the evidence supports; whether to build is a human call."
        )
    ]
    for g in verdict.gates:
        lines.append(
            Line(
                f"Gate {g.name}: {'PASS' if g.passed else 'FAIL'} (value {_fmt(g.value)}, threshold {_fmt(g.threshold)}){' - ' + g.note if g.note else ''}",
                g.evidence_ids or I("spec"),
            )
        )
    for f in verdict.flags:
        if f.tripped:
            lines.append(
                Line(
                    f"Flag {f.name}: TRIPPED (value {_fmt(f.value)}, threshold {_fmt(f.threshold)}){' - ' + f.note if f.note else ''}",
                    f.evidence_ids or I("spec"),
                )
            )
    clean = [f.name for f in verdict.flags if not f.tripped]
    if clean:
        lines.append(Line("Flags not tripped: " + ", ".join(clean)))
    sections.append(Section(title="Verdict", lines=tuple(lines)))

    return Brief(
        run_id=run_id,
        dataset_hash=dataset_hash,
        capability_name=spec.capability_name,
        target_event=spec.target_event,
        horizon_days=spec.horizon_days,
        decision=verdict.decision,
        policy_version=verdict.policy_version,
        sections=tuple(sections),
        generated_at=datetime.now(UTC).isoformat(),
        llm_used=llm_used,
    )


def sanitize_narrative(text: str, known_ids: set[str]) -> tuple[list[Line], list[str]]:
    """Split LLM narrative into sentences; keep those that either contain no digits or carry
    a valid citation to known evidence. Return kept lines and dropped sentences."""
    kept: list[Line] = []
    dropped: list[str] = []
    for sent in re.split(r"(?<=[.!?])\s+", text.strip()):
        if not sent:
            continue
        ids = tuple(i for i in re.findall(r"ev_[0-9a-f]{12}", sent) if i in known_ids)
        has_num = bool(NUM_RE.search(re.sub(r"\[ev_[0-9a-f, ]+\]", "", sent)))
        if has_num and not ids:
            dropped.append(sent)
        else:
            kept.append(Line(text=sent, evidence_ids=ids))
    return kept, dropped

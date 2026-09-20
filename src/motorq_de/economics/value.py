"""Value, ROI distribution and sensitivity.

The value chain, per year:

    events           = fleet_size x event_rate             (event_rate MEASURED from data, with CI)
    detected         = events x event_recall               (event-level recall from the harness)
    avoided_value    = detected x value_bearing_fraction x preventable_fraction x usd_per_avoided_event
    false_alerts     = fleet_size x 12 x false_alerts_per_100_vehicle_months / 100
    false_alert_cost = false_alerts x inspection_cost
    net              = avoided_value - false_alert_cost - run_cost - build_amortized
    roi              = net / (run_cost + build_amortized)

Every human-supplied term is a Range sampled as a PERT (modified beta) distribution.
Recall is sampled from its bootstrap CI. 10,000 draws, fixed seed. The output is a
distribution: P(ROI > 0), percentiles, and a one-at-a-time tornado.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from motorq_de.schemas import Range, ValueAssumptions

N_DRAWS = 10_000


def pert(rng: np.random.Generator, r: Range, n: int, lam: float = 4.0) -> np.ndarray:
    lo, mode, hi = r.low, r.base, r.high
    if hi <= lo:
        return np.full(n, mode)
    a = 1 + lam * (mode - lo) / (hi - lo)
    b = 1 + lam * (hi - mode) / (hi - lo)
    return lo + rng.beta(a, b, n) * (hi - lo)


def _terms(
    value: ValueAssumptions,
    event_rate: Range,
    recall: Range,
    alert_rate: float,
    horizon_days: int,
    inspection_cost: Range,
    run_cost_month: float,
    build_amortized_month: float,
    n: int,
    seed: int,
    false_alerts_per_100vm: float | None = None,
) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    fleet = pert(rng, value.fleet_size, n)
    epy = pert(rng, event_rate, n)
    vbf = pert(rng, value.value_bearing_fraction, n)
    prev = pert(rng, value.preventable_fraction, n)
    usd = pert(rng, value.usd_per_avoided_event, n)
    rec = np.clip(pert(rng, recall, n), 0, 1)
    insp = pert(rng, inspection_cost, n)
    events = fleet * epy
    detected = events * rec
    avoided = detected * vbf * prev * usd
    if false_alerts_per_100vm is not None and np.isfinite(false_alerts_per_100vm):
        false_alerts = fleet * 12.0 * false_alerts_per_100vm / 100.0
    else:  # fallback when event-level metrics are unavailable: alert episodes minus catches
        distinct_alerts = fleet * 365 * alert_rate / max(horizon_days, 1)
        false_alerts = np.clip(distinct_alerts - detected, 0, None)
    false_cost = false_alerts * insp
    # run cost scales with fleet relative to the fleet the cost model was priced at (base)
    scale = fleet / max(value.fleet_size.base, 1)
    cost_year = (run_cost_month + build_amortized_month) * 12 * scale
    net = avoided - false_cost - cost_year
    roi = np.where(cost_year > 0, net / cost_year, np.nan)
    return {
        "events": events,
        "detected": detected,
        "avoided_value": avoided,
        "false_alerts": false_alerts,
        "false_alert_cost": false_cost,
        "cost_year": cost_year,
        "net_value": net,
        "roi": roi,
    }


def roi_distribution(
    value: ValueAssumptions,
    event_rate: Range,
    recall: Range,
    alert_rate: float,
    horizon_days: int,
    inspection_cost: Range,
    run_cost_month: float,
    build_amortized_month: float,
    seed: int = 42,
    n: int = N_DRAWS,
    false_alerts_per_100vm: float | None = None,
) -> dict[str, Any]:
    t = _terms(
        value,
        event_rate,
        recall,
        alert_rate,
        horizon_days,
        inspection_cost,
        run_cost_month,
        build_amortized_month,
        n,
        seed,
        false_alerts_per_100vm,
    )
    roi = t["roi"][np.isfinite(t["roi"])]
    pct = lambda x, q: float(np.percentile(x, q))  # noqa: E731
    return {
        "n_draws": n,
        "inputs": {
            "event_rate": event_rate.model_dump(),
            "recall": recall.model_dump(),
            "alert_rate": alert_rate,
            "false_alerts_per_100_vehicle_months": false_alerts_per_100vm,
            "horizon_days": horizon_days,
            "run_cost_month": run_cost_month,
            "build_amortized_month": build_amortized_month,
            "value_assumptions": value.model_dump(),
            "inspection_cost": inspection_cost.model_dump(),
        },
        "p_roi_positive": float(np.mean(roi > 0)) if len(roi) else None,
        "roi": {
            "p5": pct(roi, 5),
            "p50": pct(roi, 50),
            "p95": pct(roi, 95),
            "mean": float(roi.mean()),
        },
        "net_value_year": {
            "p5": pct(t["net_value"], 5),
            "p50": pct(t["net_value"], 50),
            "p95": pct(t["net_value"], 95),
        },
        "avoided_value_year": {
            "p5": pct(t["avoided_value"], 5),
            "p50": pct(t["avoided_value"], 50),
            "p95": pct(t["avoided_value"], 95),
        },
        "false_alert_cost_year": {"p50": pct(t["false_alert_cost"], 50)},
        "cost_year": {"p50": pct(t["cost_year"], 50)},
        "detected_events_year": {"p50": pct(t["detected"], 50)},
    }


def tornado(
    value: ValueAssumptions,
    event_rate: Range,
    recall: Range,
    alert_rate: float,
    horizon_days: int,
    inspection_cost: Range,
    run_cost_month: float,
    build_amortized_month: float,
    seed: int = 42,
    false_alerts_per_100vm: float | None = None,
) -> dict[str, Any]:
    """One-at-a-time sensitivity of median ROI: swing each input to its low / high while
    holding the others at base. Ranked by swing width."""
    base_kwargs = dict(
        value=value,
        event_rate=event_rate,
        recall=recall,
        alert_rate=alert_rate,
        horizon_days=horizon_days,
        inspection_cost=inspection_cost,
        run_cost_month=run_cost_month,
        build_amortized_month=build_amortized_month,
    )

    def median_roi(**over: Any) -> float:
        kw = {**base_kwargs, **over}
        v: ValueAssumptions = kw["value"]
        t = _terms(
            v,
            kw["event_rate"],
            kw["recall"],
            kw["alert_rate"],
            kw["horizon_days"],
            kw["inspection_cost"],
            kw["run_cost_month"],
            kw["build_amortized_month"],
            2000,
            seed,
            false_alerts_per_100vm,
        )
        r = t["roi"][np.isfinite(t["roi"])]
        return float(np.median(r)) if len(r) else float("nan")

    def fixed(r: Range, which: str) -> Range:
        x = getattr(r, which)
        return Range(low=x, base=x, high=x)

    rows = []
    for name in (
        "value_bearing_fraction",
        "preventable_fraction",
        "usd_per_avoided_event",
        "fleet_size",
    ):
        r: Range = getattr(value, name)
        lo = median_roi(value=value.model_copy(update={name: fixed(r, "low")}))
        hi = median_roi(value=value.model_copy(update={name: fixed(r, "high")}))
        rows.append({"input": name, "roi_at_low": lo, "roi_at_high": hi, "swing": abs(hi - lo)})
    lo = median_roi(recall=fixed(recall, "low"))
    hi = median_roi(recall=fixed(recall, "high"))
    rows.append({"input": "recall", "roi_at_low": lo, "roi_at_high": hi, "swing": abs(hi - lo)})
    lo = median_roi(event_rate=fixed(event_rate, "low"))
    hi = median_roi(event_rate=fixed(event_rate, "high"))
    rows.append(
        {"input": "event_rate_measured", "roi_at_low": lo, "roi_at_high": hi, "swing": abs(hi - lo)}
    )
    lo = median_roi(inspection_cost=fixed(inspection_cost, "high"))
    hi = median_roi(inspection_cost=fixed(inspection_cost, "low"))
    rows.append(
        {"input": "inspection_cost", "roi_at_low": lo, "roi_at_high": hi, "swing": abs(hi - lo)}
    )
    lo = median_roi(run_cost_month=run_cost_month * 2.0)
    hi = median_roi(run_cost_month=run_cost_month * 0.5)
    rows.append(
        {"input": "run_cost_x0.5_x2", "roi_at_low": lo, "roi_at_high": hi, "swing": abs(hi - lo)}
    )
    rows.sort(key=lambda r: -r["swing"])
    return {
        "base_median_roi": median_roi(),
        "ranked": rows,
        "dominant_input": rows[0]["input"] if rows else None,
    }


def choose_operating_point(
    value: ValueAssumptions,
    event_rate: Range,
    operating_points: list[dict[str, float]],
    recall_ci_halfwidth: float,
    horizon_days: int,
    inspection_cost: Range,
    run_cost_month: float,
    build_amortized_month: float,
    seed: int = 42,
) -> dict[str, Any]:
    """Pick the alert rate that maximises median net value. The operating point is an
    economic decision (alert cost vs. avoided cost), not a modelling default. Uses event-level
    recall and false-alert rate when the harness provides them, and the harness's own
    cluster-bootstrap recall interval when it is present; `recall_ci_halfwidth` (a proxy
    derived from the AUC interval) is only the fallback for rows without one."""
    rows = []
    for op in operating_points:
        r = op.get("event_recall")
        r = op["recall"] if r is None else r
        fa = op.get("false_alerts_per_100_vehicle_months")
        ci = op.get("event_recall_ci") if op.get("event_recall") is not None else None
        if ci:
            rec = Range(low=max(0.0, min(ci["lo"], r)), base=r, high=min(1.0, max(ci["hi"], r)))
            basis_ci = "bootstrap"
        else:
            rec = Range(
                low=max(0.0, r - recall_ci_halfwidth),
                base=r,
                high=min(1.0, r + recall_ci_halfwidth),
            )
            basis_ci = "auc_proxy"
        t = _terms(
            value,
            event_rate,
            rec,
            op["alert_rate"],
            horizon_days,
            inspection_cost,
            run_cost_month,
            build_amortized_month,
            3000,
            seed,
            fa,
        )
        rows.append(
            {
                "alert_rate": op["alert_rate"],
                "recall": r,
                "recall_basis": "event" if op.get("event_recall") is not None else "vehicle_day",
                "recall_lo": rec.low,
                "recall_hi": rec.high,
                "recall_ci_basis": basis_ci,
                "precision": op.get("precision"),
                "median_lead_days": op.get("median_lead_days"),
                "false_alerts_per_100_vehicle_months": fa,
                "median_net_value_year": float(np.median(t["net_value"])),
                "median_false_alert_cost_year": float(np.median(t["false_alert_cost"])),
            }
        )
    best = max(rows, key=lambda r: r["median_net_value_year"])
    return {"candidates": rows, "chosen": best}

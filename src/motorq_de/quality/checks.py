"""Data feasibility checks. Pure functions over a DataSource; outputs are JSON-serialisable
dicts that the ledger wraps as Evidence.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from motorq_de.data.source import DataSource
from motorq_de.schemas import ProblemSpec

PSI_DRIFT_THRESHOLD = 0.20
GAP_TOLERANCE = 1.5
MIN_NONNULL = 0.05
LEAK_AUC_HARD = 0.99
LEAK_AUC_SUSPICIOUS = 0.95
LEAK_AVAILABILITY_RATIO = 2.0
LEAK_MONOTONE_RHO = 0.90
LEAK_FLAG_LIFT = 15.0
LEAK_FLAG_RECALL = 0.25


def coverage_report(source: DataSource, signals: list[str]) -> dict[str, Any]:
    """Per-signal OEM coverage plus the fleet share covered by the *whole set*."""
    per_signal: dict[str, dict[str, Any]] = {}
    for sid in signals:
        cov = source.coverage_by_oem(sid)
        per_signal[sid] = {
            oem: {
                "emits": c.emits,
                "from_model_year": c.from_model_year,
                "vehicles_total": c.vehicles_total,
                "vehicles_covered": c.vehicles_covered,
                "coverage": round(c.coverage, 6),
                "observed_nonnull_rate": round(c.observed_nonnull_rate, 6),
            }
            for oem, c in cov.items()
        }
    veh = source.vehicles()
    # fleet share with the full set: a vehicle is covered by the set if every signal it is
    # eligible for (by powertrain) has been emitted by that vehicle at least once.
    covered = np.ones(len(veh), dtype=bool)
    veh_index = veh.set_index("vehicle_id")
    for sid in signals:
        meta = source.signal_metadata(sid)
        eligible = np.ones(len(veh), dtype=bool)
        if meta.powertrain != "any":
            eligible = (veh_index["powertrain"] == meta.powertrain).to_numpy()
        cov = source.coverage_by_oem(sid)
        sig_ok = np.zeros(len(veh), dtype=bool)
        for oem, c in cov.items():
            idx = (veh_index["oem"] == oem).to_numpy()
            if c.emits and c.from_model_year is not None:
                sig_ok |= idx & (veh_index["model_year"].to_numpy() >= c.from_model_year)
        covered &= sig_ok | ~eligible
    by_oem = veh.assign(covered=covered).groupby("oem")["covered"].mean().round(6).to_dict()
    return {
        "signals": signals,
        "per_signal": per_signal,
        "fleet_share_full_set": round(float(covered.mean()), 6),
        "fleet_share_full_set_by_oem": by_oem,
        "oems_missing_any_signal": sorted([o for o, v in by_oem.items() if v < 0.5]),
    }


def quality_report(source: DataSource, signals: list[str]) -> dict[str, Any]:
    rows: dict[str, dict[str, Any]] = {}
    flags: dict[str, list[str]] = {}
    for sid in signals:
        q = source.quality(sid)
        f: list[str] = []
        if q.nonnull_rate < MIN_NONNULL:
            f.append("sparse")
        if (
            np.isfinite(q.median_gap_days)
            and q.median_gap_days > q.declared_gap_days * GAP_TOLERANCE
        ):
            f.append("stale_vs_declared")
        if q.psi_first_last_quarter > PSI_DRIFT_THRESHOLD:
            f.append("drift")
        rows[sid] = {
            "nonnull_rate": round(q.nonnull_rate, 6),
            "median_gap_days": None
            if not np.isfinite(q.median_gap_days)
            else round(q.median_gap_days, 3),
            "declared_gap_days": q.declared_gap_days,
            "psi_first_last_quarter": round(q.psi_first_last_quarter, 6),
            "history_months": round(q.history_months, 2),
        }
        if f:
            flags[sid] = f
    return {"signals": signals, "per_signal": rows, "flags": flags}


def leakage_check(source: DataSource, spec: ProblemSpec, signals: list[str]) -> dict[str, Any]:
    """Flag signals that know the future.

    Data alone cannot prove a signal is available at prediction time, so the detector is
    three-tier and says which tier fired:

    DROPPED (hard leak signatures)
      availability_jump   non-null rate in the pre-event window >= 2x baseline (DMS fields that
                          exist only once an appointment exists)
      monotone_to_event   |Spearman| between value and days-to-event among positives >= 0.9
                          (retroactively computed countdowns)
      flag_lift           near-binary signal whose active value carries >= 15x the base positive
                          rate while covering >= 25% of positives (an "appointment booked" flag)
      near_perfect_auc    univariate AUC >= 0.99

    SUSPICIOUS (kept, reported for human confirmation)
      strong_univariate   AUC in [0.95, 0.99) — plausibly a very good sensor; confirm it is
                          produced before, not because of, the event.
    """
    lf = source.training_frame(spec, signals)
    df = lf.frame[lf.frame["y"] >= 0].reset_index(drop=True)
    y = df["y"].to_numpy()
    pos = y == 1
    base_rate = float(pos.mean())
    out: dict[str, dict[str, Any]] = {}
    flagged: list[str] = []
    suspicious: list[str] = []
    if pos.sum() < 20 or (~pos).sum() < 20:
        return {
            "signals": signals,
            "per_signal": {},
            "flagged": [],
            "suspicious": [],
            "note": "too few positives to test",
        }
    days_to = _days_to_event(source, spec, df)
    for sid in lf.signal_columns:
        col = df[sid].to_numpy(dtype=float)
        avail = np.isfinite(col)
        base_avail = float(avail[~pos].mean())
        pos_avail = float(avail[pos].mean())
        ratio = (pos_avail / base_avail) if base_avail > 0 else (np.inf if pos_avail > 0 else 1.0)
        auc = None
        if avail.sum() > 100 and len(np.unique(y[avail])) == 2:
            a = roc_auc_score(y[avail], col[avail])
            auc = float(max(a, 1 - a))
        # monotone-to-event among positives
        rho = None
        m = pos & avail & np.isfinite(days_to)
        if m.sum() > 50 and np.nanstd(col[m]) > 0:
            rho = float(abs(pd.Series(col[m]).corr(pd.Series(days_to[m]), method="spearman")))
        # flag lift for near-binary signals
        lift, recall = None, None
        vals = col[avail]
        uniq = np.unique(vals)
        if 2 <= len(uniq) <= 3:
            mode = uniq[np.argmax([(vals == u).sum() for u in uniq])]
            active = avail & (col != mode)
            if active.sum() > 20:
                lift = float(y[active].mean() / base_rate) if base_rate > 0 else None
                recall = float((active & pos).sum() / pos.sum())
        reasons: list[str] = []
        if np.isfinite(ratio) and ratio >= LEAK_AVAILABILITY_RATIO and pos_avail > 0.2:
            reasons.append("availability_jump")
        if rho is not None and rho >= LEAK_MONOTONE_RHO:
            reasons.append("monotone_to_event")
        if (
            lift is not None
            and recall is not None
            and lift >= LEAK_FLAG_LIFT
            and recall >= LEAK_FLAG_RECALL
        ):
            reasons.append("flag_lift")
        if auc is not None and auc >= LEAK_AUC_HARD:
            reasons.append("near_perfect_auc")
        susp = bool(auc is not None and LEAK_AUC_SUSPICIOUS <= auc < LEAK_AUC_HARD and not reasons)
        out[sid] = {
            "univariate_auc": None if auc is None else round(auc, 6),
            "availability_ratio_pre_event": None
            if not np.isfinite(ratio)
            else round(float(ratio), 6),
            "spearman_days_to_event": None if rho is None else round(rho, 6),
            "flag_lift": None if lift is None else round(lift, 3),
            "flag_recall": None if recall is None else round(recall, 6),
            "reasons": reasons,
            "suspicious": susp,
        }
        if reasons:
            flagged.append(sid)
        elif susp:
            suspicious.append(sid)
    return {
        "signals": signals,
        "per_signal": out,
        "flagged": sorted(flagged),
        "suspicious": sorted(suspicious),
        "positive_rate": round(lf.positive_rate, 6),
    }


def _days_to_event(source: DataSource, spec: ProblemSpec, df: pd.DataFrame) -> np.ndarray:
    ev = source.events()
    ev = ev[ev["event_type"] == spec.target_event][["vehicle_id", "date"]].rename(
        columns={"date": "next_event"}
    )
    ev = ev.sort_values("next_event")
    left = df[["vehicle_id", "date"]].copy()
    left["date"] = pd.to_datetime(left["date"])
    left["_i"] = np.arange(len(left))
    left = left.sort_values("date")
    merged = pd.merge_asof(
        left,
        ev,
        left_on="date",
        right_on="next_event",
        by="vehicle_id",
        direction="forward",
        allow_exact_matches=False,
    )
    merged = merged.sort_values("_i")
    return (merged["next_event"] - merged["date"]).dt.days.to_numpy(dtype=float)


def usable_signals(
    all_signals: list[str],
    quality: dict[str, Any],
    leakage: dict[str, Any],
    powertrain_eligible: list[str] | None = None,
) -> dict[str, Any]:
    """Apply the filters: drop sparse, drop leaks. Drift and staleness are reported, not dropped."""
    dropped: dict[str, str] = {}
    keep: list[str] = []
    for sid in all_signals:
        if sid in leakage.get("flagged", []):
            dropped[sid] = "leakage"
        elif "sparse" in quality.get("flags", {}).get(sid, []):
            dropped[sid] = "sparse"
        elif powertrain_eligible is not None and sid not in powertrain_eligible:
            dropped[sid] = "powertrain_ineligible"
        else:
            keep.append(sid)
    return {"usable": keep, "dropped": dropped, "n_in": len(all_signals), "n_usable": len(keep)}


def event_rate(source: DataSource, spec: ProblemSpec) -> dict[str, Any]:
    """Measured target-event rate per vehicle-year for the decision population, with a
    Poisson 95% CI. This replaces a human guess with a number the data knows."""
    veh = source.vehicles()
    if spec.powertrain_scope != "any":
        veh = veh[veh["powertrain"] == spec.powertrain_scope]
    ids = set(veh["vehicle_id"])
    ev = source.events()
    ev = ev[(ev["event_type"] == spec.target_event) & ev["vehicle_id"].isin(ids)]
    dr = source.date_range()
    veh_years = len(veh) * ((dr.end - dr.start).days + 1) / 365.0
    n = int(len(ev))
    rate = n / veh_years if veh_years > 0 else float("nan")
    half = 1.96 * np.sqrt(max(n, 1)) / veh_years if veh_years > 0 else float("nan")
    return {
        "target_event": spec.target_event,
        "population_powertrain": spec.powertrain_scope,
        "n_vehicles": int(len(veh)),
        "vehicle_years": round(float(veh_years), 3),
        "n_events": n,
        "rate_per_vehicle_year": round(float(rate), 6),
        "rate_ci_lo": round(float(max(0.0, rate - half)), 6),
        "rate_ci_hi": round(float(rate + half), 6),
        "events_per_vehicle_per_horizon": round(float(rate * spec.horizon_days / 365.0), 6),
    }

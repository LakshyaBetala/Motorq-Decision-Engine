"""The experiment tools. Each is a pure function of (FeatureStore, spec, args) returning a
JSON-serialisable dict; the ledger wraps it as Evidence. Every metric carries a CI.

feature_analysis      signal-level permutation importance (all derived features of a signal
                      permuted jointly), mutual information, univariate AUC; full-model AUC
ablation              ordered backward elimination with a paired cluster-bootstrap
                      non-inferiority test at each step -> minimal sufficient set
temporal_validation   grouped CV vs. forward-in-time split with a horizon gap
cross_oem_validation  leave-one-OEM-out
model_comparison      named signal sets x {lightgbm, logistic} vs. baselines, with the
                      operating-point metrics the economics module consumes
redundancy            Spearman clusters among candidates -> independent information groups
learning_curve        AUC on nested vehicle subsets -> is more data still helping
tuning_headroom       fixed LightGBM grid on the sufficient set -> how loose the lower bound is
seed_stability        the sufficient-set AUC under different fold assignments
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif

from motorq_de.harness import stats
from motorq_de.harness.fitcache import FitCache
from motorq_de.harness.frames import FeatureStore, Matrix
from motorq_de.harness.models import (
    LGBM_VARIANTS,
    fit_predict,
    grouped_folds,
    make_model,
    oof_predictions,
    run_folds,
    temporal_split,
)
from motorq_de.schemas import ProblemSpec

N_SPLITS = 5
BOOT_B = 300
ABLATION_TOLERANCE = 0.005
NONINFERIORITY_CONFIDENCE = 0.80
MIN_POS_FOR_OEM = 25
ALERT_RATES = (0.0025, 0.005, 0.01, 0.02, 0.03, 0.05, 0.075, 0.10, 0.15, 0.20)
PERM_MAX_ROWS = 8_000  # rows per fold used for permutation importance
ABLATION_SCREEN_K = 20
TUNING_HEADROOM = 0.01  # best-of-grid AUC delta (lower CI bound) that marks a loose lower bound
SEED_STABILITY_SEEDS = 3
SINGLE_SIGNAL_CANDIDATES = 3  # signals kept after the importance screen before elimination


# --------------------------------------------------------------------------- helpers


def _summary(M: Matrix) -> dict[str, Any]:
    return {
        "n_rows": int(len(M.y)),
        "n_pos": int(M.y.sum()),
        "n_pos_total": M.n_pos_total,
        "n_neg_total": M.n_neg_total,
        "base_rate": round(M.base_rate, 6),
        "neg_sampling_fraction": round(M.neg_sampling_fraction, 6),
        "n_signals": len(M.signals),
        "n_features": len(M.feature_names),
    }


def _operating_metrics(
    y, s, w, M: Matrix | None = None, horizon_days: int = 7, seed: int = 0
) -> dict[str, Any]:
    r50 = stats.recall_at_precision(y, s, 0.5, w)
    r30 = stats.recall_at_precision(y, s, 0.3, w)
    points = []
    for f in ALERT_RATES:
        p = {
            "alert_rate": f,
            "recall": round(stats.recall_at_top_fraction(y, s, f, w), 6),
            "precision": round(stats.precision_at_top_fraction(y, s, f, w), 6),
        }
        if M is not None:
            ev = stats.event_level_metrics(
                y, s, w, M.groups, M.dates, M.days_to_event, horizon_days, f
            )
            rci = stats.event_recall_ci(
                y, s, w, M.groups, M.dates, M.days_to_event, f, BOOT_B, seed
            )
            p.update(
                {
                    "event_recall": None
                    if not np.isfinite(ev["event_recall"])
                    else round(ev["event_recall"], 6),
                    "event_recall_ci": None if not np.isfinite(rci.point) else rci.as_dict(),
                    "median_lead_days": None
                    if not np.isfinite(ev["median_lead_days"])
                    else round(ev["median_lead_days"], 3),
                    "false_alerts_per_100_vehicle_months": round(
                        ev["false_alerts_per_100_vehicle_months"], 6
                    ),
                    "n_events": ev["n_events"],
                }
            )
        points.append(p)
    return {
        "pr_auc": round(stats.pr_auc(y, s, w), 6),
        "brier": round(stats.brier(y, s, w), 6),
        "recall_at_precision_0_5": round(r50, 6),
        "recall_at_precision_0_3": round(r30, 6),
        "operating_points": points,
    }


def _signal_columns(M: Matrix, sid: str) -> list[int]:
    return [i for i, s in enumerate(M.signal_of) if s == sid]


def _perm_fold(
    X,
    y,
    w,
    seed: int,
    signals: list[str],
    signal_cols: dict[str, list[int]],
    tr,
    te,
    n_threads: int,
):
    """One CV fold: fit, predict the held-out rows, and permutation-drop every signal."""
    m = make_model("lightgbm", seed, n_threads)
    m.fit(X[tr], y[tr], sample_weight=w[tr])
    p = m.predict_proba(X[te])[:, 1]
    rng = np.random.default_rng(seed)
    # permutation importance on a capped, stratified subset of the fold
    sub = te if len(te) <= PERM_MAX_ROWS else np.sort(rng.choice(te, PERM_MAX_ROWS, replace=False))
    base = stats.fast_auc(y[sub], m.predict_proba(X[sub])[:, 1], w[sub])
    Xs = X[sub]
    drops: dict[str, float] = {}
    for sid in signals:
        cols = signal_cols[sid]
        Xp = Xs.copy()
        perm = rng.permutation(len(sub))
        Xp[:, cols] = Xp[perm][:, cols]
        pp = m.predict_proba(Xp)[:, 1]
        drops[sid] = base - stats.fast_auc(y[sub], pp, w[sub])
    return p, drops


# --------------------------------------------------------------------------- tools


def feature_analysis(
    store: FeatureStore,
    spec: ProblemSpec,
    signals: list[str],
    n_repeats: int = 2,
    model: str = "lightgbm",
) -> dict[str, Any]:
    M = store.matrix(spec, signals)
    seed = spec.seed
    signal_cols = {s: _signal_columns(M, s) for s in M.signals}
    perm_drops: dict[str, list[float]] = {s: [] for s in M.signals}
    oofs = []
    for r in range(n_repeats):
        folds = list(grouped_folds(M.y, M.groups, N_SPLITS, seed + r))
        results = run_folds(_perm_fold, folds, M.X, M.y, M.w, seed + r, M.signals, signal_cols)
        oof = np.full(len(M.y), np.nan)
        for (_, te), (p, drops) in zip(folds, results, strict=True):
            oof[te] = p
            for sid, d in drops.items():
                perm_drops[sid].append(d)
        oofs.append(oof)
        if r == 0:
            store.fits.put(
                FitCache.key(M.row_key, M.signals, "lightgbm", N_SPLITS, seed),
                oof,
                {"signals": list(M.signals), "model": "lightgbm", "seed": seed},
            )
    auc_ci = stats.cluster_bootstrap_auc(M.y, oofs[0], M.groups, M.w, BOOT_B, seed)
    repeat_aucs = [stats.fast_auc(M.y, o, M.w) for o in oofs]

    # mutual information on raw values (subsample for speed)
    rng = np.random.default_rng(seed)
    sub = rng.choice(len(M.y), size=min(40_000, len(M.y)), replace=False)
    raw_cols = [M.feature_names.index(s) for s in M.signals]
    Xraw = M.X[sub][:, raw_cols].astype(float)
    med = np.nanmedian(Xraw, axis=0)
    med = np.where(np.isfinite(med), med, 0.0)
    Xraw = np.where(np.isfinite(Xraw), Xraw, med)
    mi = mutual_info_classif(Xraw, M.y[sub], random_state=seed, n_neighbors=5)

    ranking = []
    for i, sid in enumerate(M.signals):
        d = np.array(perm_drops[sid])
        k = len(d)
        se = d.std(ddof=1) / np.sqrt(k) if k > 1 else 0.0
        ranking.append(
            {
                "signal": sid,
                "perm_importance": round(float(d.mean()), 6),
                "perm_ci_lo": round(float(d.mean() - 1.96 * se), 6),
                "perm_ci_hi": round(float(d.mean() + 1.96 * se), 6),
                "mutual_info": round(float(mi[i]), 6),
                "univariate_auc": round(stats.univariate_auc(M.y, M.X[:, raw_cols[i]], M.w), 6),
            }
        )
    ranking.sort(key=lambda r: -r["perm_importance"])
    return {
        **_summary(M),
        "model": model,
        "n_repeats": n_repeats,
        "n_splits": N_SPLITS,
        "auc": auc_ci.as_dict(),
        "auc_by_repeat": [round(a, 6) for a in repeat_aucs],
        **_operating_metrics(M.y, oofs[0], M.w, M, spec.horizon_days, seed),
        "ranking": ranking,
        "order_least_to_most_important": [r["signal"] for r in reversed(ranking)],
    }


def ablation(
    store: FeatureStore,
    spec: ProblemSpec,
    signals: list[str],
    order_least_to_most: list[str] | None = None,
    tolerance: float = ABLATION_TOLERANCE,
    min_signals: int = 1,
    screen_k: int = ABLATION_SCREEN_K,
    signal_cost: dict[str, float] | None = None,
    importance: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Two-stage: importance screen to the top `screen_k` signals, then ordered backward elimination.

    Remove the least important signal, refit (grouped 5-fold OOF), and test non-inferiority
    of the reduced set against the FULL set: accept if >= 80% of paired cluster-bootstrap
    replicates of AUC(reduced) - AUC(full) exceed -tolerance. A rejected signal is kept and
    elimination continues over the remaining candidates; the sufficient set is what survives.

    `underpowered` is reported when an ACCEPTED removal rests on a bootstrap that cannot
    resolve tolerance/2 - the only direction in which low power can mislead (a signal dropped
    that was actually needed). A rejected removal with a wide interval is not a power problem:
    the signal is kept, which is the conservative outcome. Rejections whose interval still
    reaches into the non-inferior region are listed as `kept_conservatively`.
    """
    seed = spec.seed
    if order_least_to_most is None:
        fa = feature_analysis(store, spec, signals, n_repeats=1)
        order_least_to_most = fa["order_least_to_most_important"]
        importance = importance or {r["signal"]: r["perm_importance"] for r in fa["ranking"]}
    order_all = [s for s in order_least_to_most if s in set(signals)]
    # cost-aware ordering: remove the least importance-per-dollar first, so the sufficient set
    # is the cheapest non-inferior set rather than merely a small one
    if signal_cost and importance:
        eps = 1e-9
        order_all = sorted(
            order_all,
            key=lambda sid: (
                max(importance.get(sid, 0.0), 0.0) / max(signal_cost.get(sid, 0.0), eps)
            ),
        )
    # stage 1 - importance screen: keep the top-K; the rest are removed without refitting
    screened_out = order_all[: max(0, len(order_all) - screen_k)]
    kept = [s for s in signals if s not in set(screened_out)]
    M_full = store.matrix(spec, signals).select(kept)
    order = [s for s in order_all if s in set(M_full.signals)]
    s_full = store.oof("lightgbm", M_full, seed, N_SPLITS)
    full_auc = stats.cluster_bootstrap_auc(M_full.y, s_full, M_full.groups, M_full.w, BOOT_B, seed)

    current = list(M_full.signals)
    accepted = list(current)
    accepted_pred = s_full
    trace: list[dict[str, Any]] = [
        {
            "step": 0,
            "removed": None,
            "n_signals": len(current),
            "auc": full_auc.as_dict(),
            "delta_vs_full": None,
            "accepted": True,
        }
    ]
    n_fits = N_SPLITS
    stop_reason = "all_candidates_tested"
    any_underpowered = False
    accepted_resolution = 0.0
    kept_needed: list[str] = []
    kept_conservatively: list[str] = []
    for step, sid in enumerate(order, start=1):
        if len(current) - 1 < min_signals:
            break
        candidate = [s for s in current if s != sid]
        M = M_full.select(candidate)
        s_red = store.oof("lightgbm", M, seed, N_SPLITS)
        n_fits += N_SPLITS
        auc_red = stats.cluster_bootstrap_auc(M.y, s_red, M.groups, M.w, BOOT_B, seed)
        delta = stats.paired_bootstrap_delta_auc(
            M.y, s_red, s_full, M.groups, M.w, BOOT_B, seed + step
        )
        resolution = delta.half_width
        # non-inferiority: accept the removal if at least NONINFERIORITY_CONFIDENCE of the
        # paired bootstrap replicates show a drop smaller than `tolerance`
        p_ok = delta.prob_greater_than(-tolerance)
        ok = p_ok >= NONINFERIORITY_CONFIDENCE
        # only an accepted removal can be misled by low power; a wide interval on a clear
        # drop (removing the main sensor) is expected, not a resolution problem
        underpowered = ok and resolution > tolerance / 2
        if ok:
            any_underpowered = any_underpowered or underpowered
            accepted_resolution = max(accepted_resolution, resolution)
        trace.append(
            {
                "step": step,
                "removed": sid,
                "n_signals": len(candidate),
                "auc": auc_red.as_dict(),
                "delta_vs_full": delta.as_dict(),
                "resolution": round(float(resolution), 6),
                "p_noninferior": round(float(p_ok), 4),
                "underpowered": bool(underpowered),
                "accepted": bool(ok),
            }
        )
        if ok:
            current, accepted, accepted_pred = candidate, candidate, s_red
        else:
            # the signal is needed: keep it and continue testing the remaining candidates, so a
            # cheap uninformative signal cannot hide behind an expensive necessary one
            kept_needed.append(sid)
            if delta.hi > -tolerance:
                kept_conservatively.append(sid)
    if kept_needed:
        stop_reason = "all_candidates_tested; kept as needed: " + ", ".join(kept_needed)
    elif len(current) <= min_signals:
        stop_reason = "reached_min_signals"
    else:
        stop_reason = "all_candidates_tested"
    suff_auc = stats.cluster_bootstrap_auc(
        M_full.y, accepted_pred, M_full.groups, M_full.w, BOOT_B, seed
    )
    removed = screened_out + [t["removed"] for t in trace if t["removed"] and t["accepted"]]
    return {
        **_summary(M_full),
        "tolerance": tolerance,
        "noninferiority_confidence": NONINFERIORITY_CONFIDENCE,
        "cost_aware": bool(signal_cost and importance),
        "screen_k": screen_k,
        "screened_out": screened_out,
        "candidate_set": list(M_full.signals),
        "full_auc": full_auc.as_dict(),
        "sufficient_set": accepted,
        "sufficient_auc": suff_auc.as_dict(),
        "removed_in_order": removed,
        "kept_as_needed": kept_needed,
        "kept_conservatively": kept_conservatively,
        "stop_reason": stop_reason,
        "underpowered": bool(any_underpowered),
        "resolution": round(float(accepted_resolution), 6),
        "trace": trace,
        "n_fits": n_fits,
        **{
            f"sufficient_{k}": v
            for k, v in _operating_metrics(M_full.y, accepted_pred, M_full.w).items()
        },
    }


def temporal_validation(
    store: FeatureStore, spec: ProblemSpec, signals: list[str], train_frac: float = 0.7
) -> dict[str, Any]:
    seed = spec.seed
    M = store.matrix(spec, signals)
    oof = store.oof("lightgbm", M, seed, N_SPLITS)  # same fit as model_comparison: a cache hit
    cv_auc = stats.cluster_bootstrap_auc(M.y, oof, M.groups, M.w, BOOT_B, seed)
    tr, te = temporal_split(M.dates, train_frac, spec.horizon_days)
    if M.y[te].sum() < 10 or M.y[tr].sum() < 10:
        return {
            **_summary(M),
            "note": "too few positives for a forward split",
            "cv_auc": cv_auc.as_dict(),
        }
    p = fit_predict("lightgbm", seed, M.X[tr], M.y[tr], M.w[tr], M.X[te])
    fwd_auc = stats.cluster_bootstrap_auc(M.y[te], p, M.groups[te], M.w[te], BOOT_B, seed)
    # rolling forward windows over the test period
    windows = []
    te_dates = M.dates[te]
    uniq = np.unique(te_dates)
    for chunk in np.array_split(uniq, 3):
        m = np.isin(te_dates, chunk)
        if M.y[te][m].sum() >= 10:
            windows.append(
                {
                    "start": str(chunk[0]),
                    "end": str(chunk[-1]),
                    "auc": round(stats.fast_auc(M.y[te][m], p[m], M.w[te][m]), 6),
                    "n_pos": int(M.y[te][m].sum()),
                }
            )
    degradation = cv_auc.point - fwd_auc.point
    return {
        **_summary(M),
        "train_frac": train_frac,
        "gap_days": spec.horizon_days,
        "train_end": str(M.dates[tr].max()),
        "test_start": str(M.dates[te].min()),
        "cv_auc": cv_auc.as_dict(),
        "forward_auc": fwd_auc.as_dict(),
        "degradation": round(float(degradation), 6),
        # conservative upper bound on the drop: CV point minus forward CI lower bound
        "degradation_upper": round(float(cv_auc.point - fwd_auc.lo), 6),
        "forward_windows": windows,
        **{f"forward_{k}": v for k, v in _operating_metrics(M.y[te], p, M.w[te]).items()},
    }


def cross_oem_validation(
    store: FeatureStore, spec: ProblemSpec, signals: list[str]
) -> dict[str, Any]:
    seed = spec.seed
    M = store.matrix(spec, signals)
    per_oem: dict[str, Any] = {}
    aucs = []
    for oem in sorted(np.unique(M.oem)):
        te = np.flatnonzero(M.oem == oem)
        tr = np.flatnonzero(M.oem != oem)
        n_pos = int(M.y[te].sum())
        if n_pos < MIN_POS_FOR_OEM:
            per_oem[oem] = {
                "n_rows": int(len(te)),
                "n_pos": n_pos,
                "auc": None,
                "note": "too few positives",
            }
            continue
        p = fit_predict("lightgbm", seed, M.X[tr], M.y[tr], M.w[tr], M.X[te])
        ci = stats.cluster_bootstrap_auc(M.y[te], p, M.groups[te], M.w[te], BOOT_B, seed)
        # how much of this OEM's feature matrix is missing (coverage gaps show up here)
        missing = float(np.mean(~np.isfinite(M.X[te])))
        # within-OEM benchmark: a model trained on this OEM alone. within ~ held-out means the
        # OEM lacks signal; within >> held-out means the model does not transfer.
        within = None
        if n_pos >= 2 * MIN_POS_FOR_OEM and len(np.unique(M.groups[te])) >= N_SPLITS:
            oof_w = oof_predictions(
                "lightgbm", M.X[te], M.y[te], M.w[te], M.groups[te], N_SPLITS, seed
            )
            within = stats.cluster_bootstrap_auc(
                M.y[te], oof_w, M.groups[te], M.w[te], BOOT_B, seed
            ).as_dict()
        per_oem[oem] = {
            "n_rows": int(len(te)),
            "n_pos": n_pos,
            "auc": ci.as_dict(),
            "feature_missing_share": round(missing, 6),
            "within_oem_auc": within,
            "transfer_gap": round(within["point"] - ci.point, 6) if within else None,
            "diagnosis": (
                None
                if within is None
                else (
                    "does_not_transfer"
                    if within["point"] - ci.point > 0.03
                    else "oem_lacks_signal"
                    if within["point"] < 0.75
                    else "ok"
                )
            ),
        }
        aucs.append(ci.point)
    arr = np.array(aucs)
    worst = min(
        (k for k, v in per_oem.items() if v.get("auc")),
        key=lambda k: per_oem[k]["auc"]["point"],
        default=None,
    )
    return {
        **_summary(M),
        "per_oem": per_oem,
        "n_oems_evaluated": int(len(arr)),
        "mean_auc": round(float(arr.mean()), 6) if len(arr) else None,
        "std_auc": round(float(arr.std(ddof=1)), 6) if len(arr) > 1 else None,
        "min_auc": round(float(arr.min()), 6) if len(arr) else None,
        "worst_oem": worst,
    }


REDUNDANCY_RHO = 0.9
REDUNDANCY_MAX_ROWS = 40_000


def redundancy(
    store: FeatureStore,
    spec: ProblemSpec,
    signals: list[str],
    threshold: float = REDUNDANCY_RHO,
) -> dict[str, Any]:
    """How the signals relate to each other: Spearman rank correlation between the raw daily
    values of every pair, on pairwise-complete rows. Pairs with |rho| >= threshold carry the
    same information; ablation should keep at most one of each, and the brief says which.
    Spearman (not Pearson) because telematics signals are skewed and unit-scaled arbitrarily;
    rank correlation is invariant to monotone transforms."""
    M = store.matrix(spec, signals)
    rng = np.random.default_rng(spec.seed)
    idx = np.arange(len(M.y))
    if len(idx) > REDUNDANCY_MAX_ROWS:
        idx = np.sort(rng.choice(idx, REDUNDANCY_MAX_ROWS, replace=False))
    raw_cols = [M.feature_names.index(sid) for sid in M.signals]
    df = pd.DataFrame(M.X[idx][:, raw_cols].astype(float), columns=list(M.signals))
    rho = df.corr(method="spearman", min_periods=200)
    pairs = []
    sig = list(M.signals)
    for i in range(len(sig)):
        for j in range(i + 1, len(sig)):
            r = rho.iat[i, j]
            if np.isfinite(r) and abs(r) >= threshold:
                pairs.append({"a": sig[i], "b": sig[j], "spearman": round(float(r), 4)})
    pairs.sort(key=lambda p: -abs(p["spearman"]))
    # connected components of the redundancy graph = groups that carry one piece of information
    parent = {s: s for s in sig}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for p_ in pairs:
        parent[find(p_["a"])] = find(p_["b"])
    groups: dict[str, list[str]] = {}
    for s_ in sig:
        groups.setdefault(find(s_), []).append(s_)
    clusters = sorted((sorted(g) for g in groups.values() if len(g) > 1), key=lambda g: -len(g))
    return {
        **_summary(M),
        "method": "spearman",
        "threshold": threshold,
        "n_rows_used": int(len(idx)),
        "n_pairs_redundant": len(pairs),
        "pairs": pairs[:50],
        "clusters": clusters,
        "n_independent_groups": len(sig) - sum(len(g) - 1 for g in clusters),
    }


LEARNING_CURVE_FRACTIONS = (0.25, 0.5, 0.75, 1.0)
LEARNING_CURVE_RISING = 0.01  # AUC gain from half the vehicles to all of them


def learning_curve(
    store: FeatureStore,
    spec: ProblemSpec,
    signals: list[str],
    fractions: tuple[float, ...] = LEARNING_CURVE_FRACTIONS,
) -> dict[str, Any]:
    """Was this much data needed, and would more help? Out-of-fold AUC of the sufficient set
    on nested subsets of VEHICLES (not rows: a vehicle's days are not independent). A curve
    that is still rising between half and all of the fleet means the reported AUC is a lower
    bound of what more history or more vehicles would give; a flat curve means the
    capability is data-saturated and the remaining uncertainty is about value, not signal."""
    seed = spec.seed
    M = store.matrix(spec, signals)
    rng = np.random.default_rng(seed)
    vehicles = np.unique(M.groups)
    order = rng.permutation(vehicles)
    points = []
    for f in fractions:
        take = set(order[: max(int(round(f * len(vehicles))), 1)].tolist())
        rows = np.flatnonzero(np.isin(M.groups, list(take)))
        sub = Matrix(
            X=M.X[rows],
            y=M.y[rows],
            w=M.w[rows],
            groups=M.groups[rows],
            oem=M.oem[rows],
            dates=M.dates[rows],
            days_to_event=M.days_to_event[rows],
            feature_names=M.feature_names,
            signal_of=M.signal_of,
            signals=M.signals,
            neg_sampling_fraction=M.neg_sampling_fraction,
            n_pos_total=M.n_pos_total,
            n_neg_total=M.n_neg_total,
            row_key=M.row_key if f == 1.0 else "",
        )
        n_pos = int(sub.y.sum())
        if n_pos < 20 or len(np.unique(sub.groups)) < N_SPLITS:
            points.append(
                {
                    "fraction": f,
                    "n_vehicles": len(take),
                    "n_rows": int(len(rows)),
                    "n_pos": n_pos,
                    "auc": None,
                }
            )
            continue
        oof = (
            store.oof("lightgbm", sub, seed, N_SPLITS)
            if f == 1.0
            else oof_predictions("lightgbm", sub.X, sub.y, sub.w, sub.groups, N_SPLITS, seed)
        )
        ci = stats.cluster_bootstrap_auc(sub.y, oof, sub.groups, sub.w, BOOT_B, seed)
        points.append(
            {
                "fraction": f,
                "n_vehicles": len(take),
                "n_rows": int(len(rows)),
                "n_pos": n_pos,
                "auc": ci.as_dict(),
            }
        )
    by_f = {p_["fraction"]: p_ for p_ in points if p_["auc"]}
    gain = (
        by_f[1.0]["auc"]["point"] - by_f[0.5]["auc"]["point"]
        if 1.0 in by_f and 0.5 in by_f
        else None
    )
    return {
        **_summary(M),
        "fractions": list(fractions),
        "points": points,
        "auc_gain_half_to_full": None if gain is None else round(float(gain), 6),
        "still_improving": bool(gain is not None and gain > LEARNING_CURVE_RISING),
        "rising_threshold": LEARNING_CURVE_RISING,
    }


def model_comparison(
    store: FeatureStore,
    spec: ProblemSpec,
    signal_sets: dict[str, list[str]],
    models: tuple[str, ...] = ("lightgbm", "logistic"),
) -> dict[str, Any]:
    seed = spec.seed
    all_signals = sorted({s for v in signal_sets.values() for s in v})
    M_all = store.matrix(spec, all_signals)
    results: dict[str, Any] = {}
    for name, sigs in signal_sets.items():
        M = M_all.select(sigs)
        entry: dict[str, Any] = {
            "signals": list(M.signals),
            "n_signals": len(M.signals),
            "models": {},
        }
        for model in models:
            oof = store.oof(model, M, seed, N_SPLITS)
            ci = stats.cluster_bootstrap_auc(M.y, oof, M.groups, M.w, BOOT_B, seed)
            entry["models"][model] = {
                "auc": ci.as_dict(),
                **_operating_metrics(M.y, oof, M.w, M, spec.horizon_days, seed),
            }
        # fair single-signal baseline: a one-signal model on the SAME population (univariate
        # AUC on available rows only would flatter signals with partial coverage)
        raw_cols = [M.feature_names.index(s) for s in M.signals]
        uni = {
            s: stats.univariate_auc(M.y, M.X[:, c], M.w)
            for s, c in zip(M.signals, raw_cols, strict=True)
        }
        cands = sorted((s for s in uni if np.isfinite(uni[s])), key=lambda s: -uni[s])[
            :SINGLE_SIGNAL_CANDIDATES
        ]
        single: dict[str, dict[str, Any]] = {}
        for s in cands:
            S = M.select([s])
            oof_s = store.oof("lightgbm", S, seed, N_SPLITS)
            single[s] = {
                "auc": stats.cluster_bootstrap_auc(
                    S.y, oof_s, S.groups, S.w, BOOT_B, seed
                ).as_dict(),
                "univariate_auc_available_rows": round(uni[s], 6),
            }
        best = max(single, key=lambda s: single[s]["auc"]["point"]) if single else None
        entry["baselines"] = {
            "majority_auc": 0.5,
            "best_single_signal": best,
            "best_single_signal_auc": single[best]["auc"]["point"] if best else None,
            "best_single_signal_auc_ci": single[best]["auc"] if best else None,
            "single_signal_models": single,
        }
        results[name] = entry
    # paired comparison between the first two sets (typically full vs sufficient) on lightgbm
    names = list(signal_sets)
    paired = None
    if len(names) >= 2:
        A, B_ = M_all.select(signal_sets[names[0]]), M_all.select(signal_sets[names[1]])
        sa = store.oof("lightgbm", A, seed, N_SPLITS)
        sb = store.oof("lightgbm", B_, seed, N_SPLITS)
        d = stats.paired_bootstrap_delta_auc(A.y, sb, sa, A.groups, A.w, BOOT_B, seed)
        paired = {"a": names[0], "b": names[1], "delta_auc_b_minus_a": d.as_dict()}
    return {**_summary(M_all), "sets": results, "paired": paired}


def tuning_headroom(
    store: FeatureStore,
    spec: ProblemSpec,
    signals: list[str],
    variants: tuple[str, ...] = tuple(LGBM_VARIANTS),
) -> dict[str, Any]:
    """How much a fixed hyper-parameter grid moves the sufficient-set AUC.

    The verdict is always computed on the one fixed configuration so studies stay
    comparable and no study is tuned to its own noise. This records what that costs: every
    variant's out-of-fold AUC and the paired delta of the best variant against the default.
    The best-of-grid delta is selected on the same folds it is measured on, so it is an
    optimistic (upper) bound on what tuning would gain; a delta whose lower CI bound is
    above `headroom_threshold` means the reported AUC is a loose lower bound."""
    seed = spec.seed
    M = store.matrix(spec, signals)
    base = store.oof("lightgbm", M, seed, N_SPLITS)
    base_ci = stats.cluster_bootstrap_auc(M.y, base, M.groups, M.w, BOOT_B, seed)
    rows: dict[str, dict[str, Any]] = {}
    for v in variants:
        name = f"lightgbm:{v}"
        oof = store.oof(name, M, seed, N_SPLITS)
        ci = stats.cluster_bootstrap_auc(M.y, oof, M.groups, M.w, BOOT_B, seed)
        d = stats.paired_bootstrap_delta_auc(M.y, oof, base, M.groups, M.w, BOOT_B, seed)
        rows[v] = {
            "params": LGBM_VARIANTS[v],
            "auc": ci.as_dict(),
            "delta_vs_default": d.as_dict(),
        }
    best = max(rows, key=lambda v: rows[v]["delta_vs_default"]["point"]) if rows else None
    delta = rows[best]["delta_vs_default"] if best else None
    return {
        **_summary(M),
        "default_auc": base_ci.as_dict(),
        "variants": rows,
        "best_variant": best,
        "headroom": None if delta is None else delta["point"],
        "headroom_lo": None if delta is None else delta["lo"],
        "headroom_threshold": TUNING_HEADROOM,
        "loose_lower_bound": bool(delta is not None and delta["lo"] > TUNING_HEADROOM),
        "note": "best-of-grid on the evaluation folds: an optimistic bound; the verdict "
        "uses the default configuration",
    }


def seed_stability(
    store: FeatureStore,
    spec: ProblemSpec,
    signals: list[str],
    n_seeds: int = SEED_STABILITY_SEEDS,
) -> dict[str, Any]:
    """Determinism says the same seed gives the same answer; this asks whether a different
    seed would give a materially different one. The sufficient-set AUC is recomputed under
    `n_seeds` fold assignments (the model seed follows the fold seed). A spread wider than
    the ablation tolerance means the reported AUC is fold-assignment noise to that degree
    and the sufficient set should be read as one of several equivalent choices."""
    seed = spec.seed
    M = store.matrix(spec, signals)
    per_seed = []
    for i in range(n_seeds):
        s = seed + 1000 * i
        oof = store.oof("lightgbm", M, s, N_SPLITS)
        per_seed.append({"seed": s, "auc": round(stats.fast_auc(M.y, oof, M.w), 6)})
    aucs = [r["auc"] for r in per_seed]
    spread = float(max(aucs) - min(aucs)) if aucs else float("nan")
    return {
        **_summary(M),
        "per_seed": per_seed,
        "auc_mean": round(float(np.mean(aucs)), 6),
        "auc_spread": round(spread, 6),
        "spread_threshold": ABLATION_TOLERANCE,
        "seed_sensitive": bool(spread > ABLATION_TOLERANCE),
    }

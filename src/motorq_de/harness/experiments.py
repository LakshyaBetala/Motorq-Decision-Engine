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
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.feature_selection import mutual_info_classif

from motorq_de.harness import stats
from motorq_de.harness.frames import FeatureStore, Matrix
from motorq_de.harness.models import (
    fit_predict,
    grouped_folds,
    make_model,
    oof_predictions,
    temporal_split,
)
from motorq_de.schemas import ProblemSpec

N_SPLITS = 5
BOOT_B = 300
ABLATION_TOLERANCE = 0.005
NONINFERIORITY_CONFIDENCE = 0.80
MIN_POS_FOR_OEM = 25
ALERT_RATES = (0.005, 0.01, 0.02, 0.05, 0.10)
PERM_MAX_ROWS = 8_000  # rows per fold used for permutation importance
ABLATION_SCREEN_K = 20
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


def _operating_metrics(y, s, w) -> dict[str, Any]:
    r50 = stats.recall_at_precision(y, s, 0.5, w)
    r30 = stats.recall_at_precision(y, s, 0.3, w)
    return {
        "pr_auc": round(stats.pr_auc(y, s, w), 6),
        "brier": round(stats.brier(y, s, w), 6),
        "recall_at_precision_0_5": round(r50, 6),
        "recall_at_precision_0_3": round(r30, 6),
        "operating_points": [
            {
                "alert_rate": f,
                "recall": round(stats.recall_at_top_fraction(y, s, f, w), 6),
                "precision": round(stats.precision_at_top_fraction(y, s, f, w), 6),
            }
            for f in ALERT_RATES
        ],
    }


def _signal_columns(M: Matrix, sid: str) -> list[int]:
    return [i for i, s in enumerate(M.signal_of) if s == sid]


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
    perm_drops: dict[str, list[float]] = {s: [] for s in M.signals}
    oofs = []
    for r in range(n_repeats):
        oof = np.full(len(M.y), np.nan)
        for tr, te in grouped_folds(M.y, M.groups, N_SPLITS, seed + r):
            m = make_model("lightgbm", seed + r)
            m.fit(M.X[tr], M.y[tr], sample_weight=M.w[tr])
            p = m.predict_proba(M.X[te])[:, 1]
            oof[te] = p
            rng = np.random.default_rng(seed + r)
            # permutation importance on a capped, stratified subset of the fold
            sub = (
                te
                if len(te) <= PERM_MAX_ROWS
                else np.sort(rng.choice(te, PERM_MAX_ROWS, replace=False))
            )
            base = stats.fast_auc(M.y[sub], m.predict_proba(M.X[sub])[:, 1], M.w[sub])
            Xs = M.X[sub]
            for sid in M.signals:
                cols = _signal_columns(M, sid)
                Xp = Xs.copy()
                perm = rng.permutation(len(sub))
                Xp[:, cols] = Xp[perm][:, cols]
                pp = m.predict_proba(Xp)[:, 1]
                perm_drops[sid].append(base - stats.fast_auc(M.y[sub], pp, M.w[sub]))
        oofs.append(oof)
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
        **_operating_metrics(M.y, oofs[0], M.w),
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
) -> dict[str, Any]:
    """Two-stage: importance screen to the top `screen_k` signals, then ordered backward elimination.

    Remove the least important signal, refit (grouped 5-fold OOF), and test non-inferiority
    of the reduced set against the FULL set: accept if >= 80% of paired cluster-bootstrap
    replicates of AUC(reduced) - AUC(full) exceed -tolerance. Stop at the first rejection; the
    sufficient set is the last accepted set. `underpowered` is reported when the bootstrap
    cannot resolve tolerance/2.
    """
    seed = spec.seed
    if order_least_to_most is None:
        fa = feature_analysis(store, spec, signals, n_repeats=1)
        order_least_to_most = fa["order_least_to_most_important"]
    order_all = [s for s in order_least_to_most if s in set(signals)]
    # stage 1 - importance screen: keep the top-K; the rest are removed without refitting
    screened_out = order_all[: max(0, len(order_all) - screen_k)]
    kept = [s for s in signals if s not in set(screened_out)]
    M_full = store.matrix(spec, signals).select(kept)
    order = [s for s in order_all if s in set(M_full.signals)]
    s_full = oof_predictions(
        "lightgbm", M_full.X, M_full.y, M_full.w, M_full.groups, N_SPLITS, seed
    )
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
    stop_reason = "reached_min_signals"
    any_underpowered = False
    for step, sid in enumerate(order, start=1):
        if len(current) - 1 < min_signals:
            break
        candidate = [s for s in current if s != sid]
        M = M_full.select(candidate)
        s_red = oof_predictions("lightgbm", M.X, M.y, M.w, M.groups, N_SPLITS, seed)
        n_fits += N_SPLITS
        auc_red = stats.cluster_bootstrap_auc(M.y, s_red, M.groups, M.w, BOOT_B, seed)
        delta = stats.paired_bootstrap_delta_auc(
            M.y, s_red, s_full, M.groups, M.w, BOOT_B, seed + step
        )
        resolution = delta.half_width
        underpowered = resolution > tolerance / 2
        # non-inferiority: accept the removal if at least NONINFERIORITY_CONFIDENCE of the
        # paired bootstrap replicates show a drop smaller than `tolerance`
        p_ok = delta.prob_greater_than(-tolerance)
        ok = p_ok >= NONINFERIORITY_CONFIDENCE
        any_underpowered = any_underpowered or underpowered
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
            stop_reason = f"removing_{sid}_is_materially_worse"
            break
    suff_auc = stats.cluster_bootstrap_auc(
        M_full.y, accepted_pred, M_full.groups, M_full.w, BOOT_B, seed
    )
    removed = screened_out + [t["removed"] for t in trace if t["removed"] and t["accepted"]]
    return {
        **_summary(M_full),
        "tolerance": tolerance,
        "noninferiority_confidence": NONINFERIORITY_CONFIDENCE,
        "screen_k": screen_k,
        "screened_out": screened_out,
        "candidate_set": list(M_full.signals),
        "full_auc": full_auc.as_dict(),
        "sufficient_set": accepted,
        "sufficient_auc": suff_auc.as_dict(),
        "removed_in_order": removed,
        "stop_reason": stop_reason,
        "underpowered": bool(any_underpowered),
        "resolution": round(float(max((t.get("resolution", 0.0) for t in trace), default=0.0)), 6),
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
    oof = oof_predictions("lightgbm", M.X, M.y, M.w, M.groups, N_SPLITS, seed)
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
        per_oem[oem] = {
            "n_rows": int(len(te)),
            "n_pos": n_pos,
            "auc": ci.as_dict(),
            "feature_missing_share": round(missing, 6),
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
            oof = oof_predictions(model, M.X, M.y, M.w, M.groups, N_SPLITS, seed)
            ci = stats.cluster_bootstrap_auc(M.y, oof, M.groups, M.w, BOOT_B, seed)
            entry["models"][model] = {"auc": ci.as_dict(), **_operating_metrics(M.y, oof, M.w)}
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
            oof_s = oof_predictions("lightgbm", S.X, S.y, S.w, S.groups, N_SPLITS, seed)
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
        sa = oof_predictions("lightgbm", A.X, A.y, A.w, A.groups, N_SPLITS, seed)
        sb = oof_predictions("lightgbm", B_.X, B_.y, B_.w, B_.groups, N_SPLITS, seed)
        d = stats.paired_bootstrap_delta_auc(A.y, sb, sa, A.groups, A.w, BOOT_B, seed)
        paired = {"a": names[0], "b": names[1], "delta_auc_b_minus_a": d.as_dict()}
    return {**_summary(M_all), "sets": results, "paired": paired}

"""Statistics for honest evidence: fast AUC, cluster bootstrap CIs, paired deltas.

Vehicle-days from the same vehicle are correlated, so every bootstrap resamples
*vehicles* (clusters), not rows. All bootstraps are seeded and therefore deterministic.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import rankdata
from sklearn.metrics import average_precision_score, brier_score_loss, precision_recall_curve


@dataclass(frozen=True)
class CI:
    point: float
    lo: float
    hi: float

    def as_dict(self) -> dict[str, float]:
        return {"point": round(self.point, 6), "lo": round(self.lo, 6), "hi": round(self.hi, 6)}


def fast_auc(y: np.ndarray, s: np.ndarray, w: np.ndarray | None = None) -> float:
    """Weighted Mann-Whitney AUC in O(n log n)."""
    if w is None:
        w = np.ones_like(s, dtype=float)
    pos, neg = y == 1, y == 0
    if not pos.any() or not neg.any():
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    s_sorted, w_sorted, y_sorted = s[order], w[order], y[order]
    # cumulative negative weight strictly below each score, with ties split evenly
    cw_neg = np.cumsum(np.where(y_sorted == 0, w_sorted, 0.0))
    # handle ties: for each unique score, compute neg weight below and neg weight at the tie
    uniq, starts = np.unique(s_sorted, return_index=True)
    ends = np.append(starts[1:], len(s_sorted))
    below = np.concatenate([[0.0], cw_neg[ends[:-1] - 1]]) if len(uniq) > 1 else np.array([0.0])
    at = cw_neg[ends - 1] - below
    contrib_per_group = below + 0.5 * at
    group_id = np.repeat(np.arange(len(uniq)), ends - starts)
    contrib = contrib_per_group[group_id]
    num = float(np.sum(np.where(y_sorted == 1, w_sorted * contrib, 0.0)))
    den = float(np.sum(w[pos]) * np.sum(w[neg]))
    return num / den if den > 0 else float("nan")


def pr_auc(y: np.ndarray, s: np.ndarray, w: np.ndarray | None = None) -> float:
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(average_precision_score(y, s, sample_weight=w))


def brier(y: np.ndarray, p: np.ndarray, w: np.ndarray | None = None) -> float:
    return float(brier_score_loss(y, np.clip(p, 0, 1), sample_weight=w))


def recall_at_precision(
    y: np.ndarray, s: np.ndarray, target_precision: float, w: np.ndarray | None = None
) -> float:
    """Highest recall achievable while precision >= target (0 if unattainable)."""
    if len(np.unique(y)) < 2:
        return float("nan")
    p, r, _ = precision_recall_curve(y, s, sample_weight=w)
    ok = p[:-1] >= target_precision
    return float(r[:-1][ok].max()) if ok.any() else 0.0


def recall_at_top_fraction(
    y: np.ndarray, s: np.ndarray, frac: float, w: np.ndarray | None = None
) -> float:
    """Recall when alerting on the top `frac` of the (weighted) population."""
    if w is None:
        w = np.ones_like(s, dtype=float)
    order = np.argsort(-s, kind="mergesort")
    cw = np.cumsum(w[order]) / w.sum()
    k = int(np.searchsorted(cw, frac, side="right"))
    top = order[: max(k, 1)]
    tp = float(np.sum(w[top] * (y[top] == 1)))
    total_pos = float(np.sum(w * (y == 1)))
    return tp / total_pos if total_pos > 0 else float("nan")


def precision_at_top_fraction(
    y: np.ndarray, s: np.ndarray, frac: float, w: np.ndarray | None = None
) -> float:
    if w is None:
        w = np.ones_like(s, dtype=float)
    order = np.argsort(-s, kind="mergesort")
    cw = np.cumsum(w[order]) / w.sum()
    k = int(np.searchsorted(cw, frac, side="right"))
    top = order[: max(k, 1)]
    return float(np.sum(w[top] * (y[top] == 1)) / np.sum(w[top]))


# ------------------------------------------------------------------ cluster bootstrap


def _cluster_index(groups: np.ndarray) -> tuple[np.ndarray, list[np.ndarray]]:
    order = np.argsort(groups, kind="mergesort")
    g_sorted = groups[order]
    uniq, starts = np.unique(g_sorted, return_index=True)
    splits = np.split(order, starts[1:])
    return uniq, splits


def cluster_bootstrap_auc(
    y: np.ndarray,
    s: np.ndarray,
    groups: np.ndarray,
    w: np.ndarray | None = None,
    B: int = 300,
    seed: int = 0,
) -> CI:
    point = fast_auc(y, s, w)
    _, splits = _cluster_index(groups)
    rng = np.random.default_rng(seed)
    G = len(splits)
    vals = np.empty(B)
    for b in range(B):
        pick = rng.integers(0, G, G)
        idx = np.concatenate([splits[i] for i in pick])
        vals[b] = fast_auc(y[idx], s[idx], None if w is None else w[idx])
    vals = vals[np.isfinite(vals)]
    lo, hi = np.percentile(vals, [2.5, 97.5]) if len(vals) else (point, point)
    return CI(point, float(lo), float(hi))


@dataclass(frozen=True)
class PairedDelta:
    point: float
    lo: float
    hi: float
    samples: np.ndarray

    def as_dict(self) -> dict[str, float]:
        return {"point": round(self.point, 6), "lo": round(self.lo, 6), "hi": round(self.hi, 6)}

    def prob_greater_than(self, threshold: float) -> float:
        return float(np.mean(self.samples > threshold)) if len(self.samples) else float("nan")

    @property
    def half_width(self) -> float:
        return (self.hi - self.lo) / 2


def paired_bootstrap_delta_auc(
    y: np.ndarray,
    s_a: np.ndarray,
    s_b: np.ndarray,
    groups: np.ndarray,
    w: np.ndarray | None = None,
    B: int = 300,
    seed: int = 0,
) -> PairedDelta:
    """Bootstrap distribution of AUC(a) - AUC(b) on the same rows, resampling vehicles."""
    point = fast_auc(y, s_a, w) - fast_auc(y, s_b, w)
    _, splits = _cluster_index(groups)
    rng = np.random.default_rng(seed)
    G = len(splits)
    vals = np.empty(B)
    for b in range(B):
        pick = rng.integers(0, G, G)
        idx = np.concatenate([splits[i] for i in pick])
        ww = None if w is None else w[idx]
        vals[b] = fast_auc(y[idx], s_a[idx], ww) - fast_auc(y[idx], s_b[idx], ww)
    vals = vals[np.isfinite(vals)]
    lo, hi = np.percentile(vals, [2.5, 97.5]) if len(vals) else (point, point)
    return PairedDelta(point, float(lo), float(hi), vals)


def univariate_auc(y: np.ndarray, x: np.ndarray, w: np.ndarray | None = None) -> float:
    """Direction-agnostic AUC of a raw feature on the rows where it is available."""
    m = np.isfinite(x)
    if m.sum() < 50 or len(np.unique(y[m])) < 2:
        return float("nan")
    a = fast_auc(y[m], rankdata(x[m], method="average"), None if w is None else w[m])
    return float(max(a, 1 - a)) if np.isfinite(a) else float("nan")

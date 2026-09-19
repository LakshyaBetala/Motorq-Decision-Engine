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
#
# Resampling vehicles with replacement is equivalent to giving each vehicle a multinomial
# count and every row of that vehicle the weight w * count: a positive row duplicated k
# times and a negative row duplicated m times contribute k*m pairs, exactly the weighted
# Mann-Whitney statistic. So the scores are sorted ONCE and each replicate is an O(n)
# weighted cumsum instead of a fresh O(n log n) sort. Ties are split evenly as in fast_auc.


class AucSorter:
    """Pre-sorted score structure: weighted AUC for any row-weight vector in O(n)."""

    def __init__(self, y: np.ndarray, s: np.ndarray):
        order = np.argsort(s, kind="mergesort")
        s_sorted = s[order]
        self.order = order
        self.y_sorted = y[order] == 1
        uniq, starts = np.unique(s_sorted, return_index=True)
        ends = np.append(starts[1:], len(s_sorted))
        self.group_id = np.repeat(np.arange(len(uniq)), ends - starts)
        self.n_groups = len(uniq)

    def auc(self, w: np.ndarray) -> float:
        ws = w[self.order]
        neg_w = np.where(self.y_sorted, 0.0, ws)
        pos_w = np.where(self.y_sorted, ws, 0.0)
        neg_g = np.bincount(self.group_id, weights=neg_w, minlength=self.n_groups)
        below = np.concatenate([[0.0], np.cumsum(neg_g)[:-1]])
        contrib = (below + 0.5 * neg_g)[self.group_id]
        den = pos_w.sum() * neg_w.sum()
        return float(np.sum(pos_w * contrib) / den) if den > 0 else float("nan")


def _replicate_weights(groups: np.ndarray, w: np.ndarray, B: int, seed: int):
    """Yield B row-weight vectors for cluster (vehicle) bootstrap replicates."""
    codes, uniq = _factorize(groups)
    G = len(uniq)
    rng = np.random.default_rng(seed)
    for _ in range(B):
        count = np.bincount(rng.integers(0, G, G), minlength=G).astype(float)
        yield w * count[codes]


def _factorize(groups: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    uniq, codes = np.unique(groups, return_inverse=True)
    return codes, uniq


def cluster_bootstrap_auc(
    y: np.ndarray,
    s: np.ndarray,
    groups: np.ndarray,
    w: np.ndarray | None = None,
    B: int = 300,
    seed: int = 0,
) -> CI:
    w = np.ones(len(s), dtype=float) if w is None else np.asarray(w, dtype=float)
    sorter = AucSorter(y, s)
    point = sorter.auc(w)
    vals = np.fromiter(
        (sorter.auc(wr) for wr in _replicate_weights(groups, w, B, seed)), dtype=float, count=B
    )
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
    w = np.ones(len(s_a), dtype=float) if w is None else np.asarray(w, dtype=float)
    sa, sb = AucSorter(y, s_a), AucSorter(y, s_b)
    point = sa.auc(w) - sb.auc(w)
    vals = np.fromiter(
        (sa.auc(wr) - sb.auc(wr) for wr in _replicate_weights(groups, w, B, seed)),
        dtype=float,
        count=B,
    )
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


# ------------------------------------------------------------------ event-level metrics


def event_level_metrics(
    y: np.ndarray,
    s: np.ndarray,
    w: np.ndarray,
    groups: np.ndarray,
    dates: np.ndarray,
    days_to_event: np.ndarray,
    horizon_days: int,
    alert_rate: float,
) -> dict[str, float]:
    """What a fleet manager experiences at a given alert rate.

    event_recall                        share of events with at least one flagged vehicle-day
                                        in the horizon window before the event
    median_lead_days                    median days from the first flag to the event (caught events)
    false_alerts_per_100_vehicle_months false-alert episodes per 100 vehicle-months; consecutive
                                        flagged negative days on one vehicle are one episode,
                                        approximated as flagged negative vehicle-days / horizon
                                        (weighted for negative downsampling)
    """
    order = np.argsort(-s, kind="mergesort")
    cw = np.cumsum(w[order]) / w.sum()
    k = int(np.searchsorted(cw, alert_rate, side="right"))
    flagged = np.zeros(len(s), dtype=bool)
    flagged[order[: max(k, 1)]] = True

    pos = (y == 1) & np.isfinite(days_to_event)
    if pos.any():
        event_date = dates[pos].astype("datetime64[D]") + days_to_event[pos].astype(int).astype(
            "timedelta64[D]"
        )
        keys = [(int(g), str(d)) for g, d in zip(groups[pos], event_date, strict=True)]
        caught: dict[tuple[int, str], bool] = {}
        lead: dict[tuple[int, str], float] = {}
        for key, f, d in zip(keys, flagged[pos], days_to_event[pos], strict=True):
            caught[key] = caught.get(key, False) or bool(f)
            if f:
                lead[key] = max(lead.get(key, 0.0), float(d))
        n_events = len(caught)
        n_caught = sum(caught.values())
        event_recall = n_caught / n_events if n_events else float("nan")
        median_lead = float(np.median(list(lead.values()))) if lead else float("nan")
    else:
        n_events, n_caught, event_recall, median_lead = 0, 0, float("nan"), float("nan")

    neg = y == 0
    fa_days = float(np.sum(w[neg & flagged]))
    fa_episodes = fa_days / max(horizon_days, 1)
    vehicle_months = float(np.sum(w)) / 30.44
    return {
        "alert_rate": float(np.sum(w[flagged]) / np.sum(w)),
        "n_events": int(n_events),
        "n_events_caught": int(n_caught),
        "event_recall": float(event_recall),
        "median_lead_days": median_lead,
        "false_alerts_per_100_vehicle_months": float(100.0 * fa_episodes / vehicle_months)
        if vehicle_months
        else float("nan"),
    }

"""Feature engineering and deterministic sampling for the experiment harness.

Features per signal (all computed on the complete vehicle-day grid, per vehicle):
    <sid>            value on day t
    <sid>__m7        trailing 7-day mean (min 1 obs)
    <sid>__m30       trailing 30-day mean (min 1 obs)
    <sid>__d30       value(t) minus trailing 30-day mean  (level shift; captures wear slope)

Ablation operates at the *signal* level: a signal is in or out with all its derived
features, because that is the unit the cost model prices.

Sampling: all positives (up to a cap) plus negatives at `neg_ratio`, drawn with a fixed
seed. Negatives carry weight 1/sampling_fraction so precision-type metrics are unbiased
for the true base rate.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from motorq_de.data.source import DataSource
from motorq_de.harness.fitcache import FitCache
from motorq_de.harness.models import oof_predictions
from motorq_de.hashing import hash_inputs
from motorq_de.schemas import ProblemSpec

SUFFIXES = ("", "__m7", "__m30", "__d30")


@dataclass
class Matrix:
    X: np.ndarray  # float32 (n, p)
    y: np.ndarray  # int8
    w: np.ndarray  # float64 sample weights correcting negative downsampling
    groups: np.ndarray  # vehicle index (int)
    oem: np.ndarray  # str
    dates: np.ndarray  # datetime64[D]
    days_to_event: np.ndarray  # float; days until the next target event (NaN if none)
    feature_names: list[str]
    signal_of: list[str]  # signal id for each feature column
    signals: list[str]
    neg_sampling_fraction: float
    n_pos_total: int
    n_neg_total: int
    # identity of the ROWS (dataset, label definition, sampling); independent of the signals,
    # so a fit on a signal subset is the same whichever parent matrix it was selected from
    row_key: str = ""

    def select(self, signals: list[str]) -> Matrix:
        keep = [i for i, s in enumerate(self.signal_of) if s in set(signals)]
        return Matrix(
            X=self.X[:, keep],
            y=self.y,
            w=self.w,
            groups=self.groups,
            oem=self.oem,
            dates=self.dates,
            days_to_event=self.days_to_event,
            feature_names=[self.feature_names[i] for i in keep],
            signal_of=[self.signal_of[i] for i in keep],
            signals=[s for s in self.signals if s in set(signals)],
            neg_sampling_fraction=self.neg_sampling_fraction,
            n_pos_total=self.n_pos_total,
            n_neg_total=self.n_neg_total,
            row_key=self.row_key,
        )

    @property
    def base_rate(self) -> float:
        return self.n_pos_total / (self.n_pos_total + self.n_neg_total)


class FeatureStore:
    """Builds and caches the feature matrix for (dataset, spec, signals). One build per tool
    chain; every tool selects columns from it, so all experiments see identical rows."""

    def __init__(self, source: DataSource, fits: FitCache | None = None):
        self.source = source
        self._cache: dict[str, Matrix] = {}
        self.fits = fits or FitCache()

    def row_key(self, spec: ProblemSpec, max_rows: int, neg_ratio: int) -> str:
        """What determines the rows of a matrix: dataset, label definition and sampling."""
        return hash_inputs(
            {
                "dataset": self.source.dataset_hash,
                "target_event": spec.target_event,
                "horizon_days": spec.horizon_days,
                "decision_unit": spec.decision_unit,
                "powertrain": spec.powertrain_scope,
                "seed": spec.seed,
                "max_rows": max_rows,
                "neg_ratio": neg_ratio,
            }
        )

    def oof(self, model: str, M: Matrix, seed: int, n_splits: int) -> np.ndarray:
        """Out-of-fold predictions for M, served from the fit cache when the identical fit
        has been done before (same rows, ordered signals, model, folds, seed, environment)."""
        key = FitCache.key(M.row_key, M.signals, model, n_splits, seed)
        hit = self.fits.get(key)
        if hit is not None:
            return hit
        oof = oof_predictions(model, M.X, M.y, M.w, M.groups, n_splits, seed)
        self.fits.put(key, oof, {"signals": list(M.signals), "model": model, "seed": seed})
        return oof

    def key(self, spec: ProblemSpec, signals: list[str], max_rows: int, neg_ratio: int) -> str:
        return hash_inputs(
            {
                "dataset": self.source.dataset_hash,
                "spec": spec.model_dump(),
                "signals": sorted(signals),
                "max_rows": max_rows,
                "neg_ratio": neg_ratio,
            }
        )

    def matrix(
        self, spec: ProblemSpec, signals: list[str], max_rows: int = 150_000, neg_ratio: int = 6
    ) -> Matrix:
        k = self.key(spec, signals, max_rows, neg_ratio)
        if k not in self._cache:
            m = build_matrix(self.source, spec, signals, max_rows, neg_ratio)
            m.row_key = self.row_key(spec, max_rows, neg_ratio)
            self._cache[k] = m
        return self._cache[k]


def build_matrix(
    source: DataSource, spec: ProblemSpec, signals: list[str], max_rows: int, neg_ratio: int
) -> Matrix:
    lf = source.training_frame(spec, signals)
    df = lf.frame  # sorted by vehicle then date; complete grid
    sig_cols = list(lf.signal_columns)
    n_veh = df["vehicle_id"].nunique()
    n_days = len(df) // n_veh
    if n_veh * n_days != len(df):
        raise ValueError("training_frame must be a complete vehicle x day grid")

    # (n_days, n_veh) layout per signal for fast rolling along time
    feats: dict[str, np.ndarray] = {}
    for sid in sig_cols:
        wide = df[sid].to_numpy(dtype=np.float32).reshape(n_veh, n_days).T  # days x vehicles
        w = pd.DataFrame(wide)
        m7 = w.rolling(7, min_periods=1).mean().to_numpy(dtype=np.float32)
        m30 = w.rolling(30, min_periods=1).mean().to_numpy(dtype=np.float32)
        feats[sid] = wide.T.reshape(-1)
        feats[sid + "__m7"] = m7.T.reshape(-1)
        feats[sid + "__m30"] = m30.T.reshape(-1)
        feats[sid + "__d30"] = (wide - m30).T.reshape(-1)

    y_all = df["y"].to_numpy()
    known = y_all >= 0
    pos_idx = np.flatnonzero(known & (y_all == 1))
    neg_idx = np.flatnonzero(known & (y_all == 0))
    n_pos_total, n_neg_total = len(pos_idx), len(neg_idx)

    rng = np.random.default_rng(spec.seed)
    max_pos = max(1000, max_rows // (1 + neg_ratio))
    if len(pos_idx) > max_pos:
        pos_idx = np.sort(rng.choice(pos_idx, size=max_pos, replace=False))
    n_neg = min(len(neg_idx), max(len(pos_idx) * neg_ratio, 1), max_rows - len(pos_idx))
    neg_frac = n_neg / max(len(neg_idx), 1)
    neg_take = np.sort(rng.choice(neg_idx, size=n_neg, replace=False))
    idx = np.sort(np.concatenate([pos_idx, neg_take]))

    feature_names = [c for sid in sig_cols for c in (sid + s for s in SUFFIXES)]
    signal_of = [sid for sid in sig_cols for _ in SUFFIXES]
    X = np.column_stack([feats[c][idx] for c in feature_names]).astype(np.float32)
    y = y_all[idx].astype(np.int8)
    pos_frac = len(pos_idx) / max(n_pos_total, 1)
    w = np.where(y == 1, 1.0 / pos_frac, 1.0 / neg_frac)
    veh_codes = pd.factorize(df["vehicle_id"])[0]
    days_to = _days_to_next_event(source, spec, df.iloc[idx][["vehicle_id", "date"]])
    return Matrix(
        X=X,
        y=y,
        w=w,
        groups=veh_codes[idx],
        oem=df["oem"].to_numpy()[idx],
        dates=df["date"].to_numpy().astype("datetime64[D]")[idx],
        days_to_event=days_to,
        feature_names=feature_names,
        signal_of=signal_of,
        signals=sig_cols,
        neg_sampling_fraction=float(neg_frac),
        n_pos_total=int(n_pos_total),
        n_neg_total=int(n_neg_total),
    )


def _days_to_next_event(source: DataSource, spec: ProblemSpec, rows: pd.DataFrame) -> np.ndarray:
    """Days from each sampled vehicle-day to that vehicle's next target event (NaN if none).
    Used for event-level metrics: rows sharing (vehicle, next event) belong to one event."""
    ev = source.events()
    ev = ev[ev["event_type"] == spec.target_event][["vehicle_id", "date"]].rename(
        columns={"date": "next_event"}
    )
    ev["next_event"] = pd.to_datetime(ev["next_event"])
    ev = ev.sort_values("next_event")
    left = rows.copy()
    left["date"] = pd.to_datetime(left["date"])
    left["_i"] = np.arange(len(left))
    left = left.sort_values("date")
    m = pd.merge_asof(
        left,
        ev,
        left_on="date",
        right_on="next_event",
        by="vehicle_id",
        direction="forward",
        allow_exact_matches=False,
    ).sort_values("_i")
    return (m["next_event"] - m["date"]).dt.days.to_numpy(dtype=float)

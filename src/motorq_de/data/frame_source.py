"""FrameSource: DataSource semantics over in-memory pandas frames.

Both SyntheticSource (parquet) and SnowflakeSource (SQL) reduce to three frames plus a
catalog, so labelling, coverage and quality live here once and behave identically on
synthetic and production data.

    vehicles      vehicle_id, oem, model_year, powertrain, ...
    signals_daily vehicle_id, date, oem, model_year, active, <one column per signal>
    events        vehicle_id, date, event_type, detail
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import date

import numpy as np
import pandas as pd

from motorq_de.data.source import LabeledFrame
from motorq_de.schemas import CoverageStat, DateWindow, ProblemSpec, QualityStat, SignalMeta

META_COLS = ("vehicle_id", "date", "oem", "model_year", "active")
DECLARED_GAP_DAYS = {"realtime": 1.0, "daily": 1.0, "weekly": 7.0}
EMITS_MIN_VEHICLE_SHARE = (
    0.20  # an OEM "emits" a signal if >= 20% of its eligible vehicles ever report it
)
MODEL_YEAR_COVERAGE = 0.50


class FrameSource(ABC):
    dataset_hash: str

    # ------------------------------------------------------------- to implement
    @abstractmethod
    def list_signals(self) -> list[SignalMeta]: ...

    @abstractmethod
    def vehicles(self) -> pd.DataFrame: ...

    @abstractmethod
    def events(self) -> pd.DataFrame: ...

    @abstractmethod
    def read_signals(self, signals: list[str]) -> pd.DataFrame:
        """Complete vehicle x day grid with META_COLS + the requested signal columns."""

    def declared_emits(self, oem: str, signal_id: str) -> tuple[bool, int] | None:
        """Optional declared coverage (emits, from_model_year). None -> infer from data."""
        return None

    # ------------------------------------------------------------- shared
    def signal_metadata(self, signal_id: str) -> SignalMeta:
        for m in self.list_signals():
            if m.signal_id == signal_id:
                return m
        raise KeyError(signal_id)

    def date_range(self) -> DateWindow:
        df = self.read_signals([])
        d = pd.to_datetime(df["date"])
        return DateWindow(start=d.min().date(), end=d.max().date())

    def _column(self, signal_id: str) -> pd.DataFrame:
        return self.read_signals([signal_id])

    def coverage_by_oem(self, signal_id: str) -> dict[str, CoverageStat]:
        df = self._column(signal_id)
        meta = self.signal_metadata(signal_id)
        veh = self.vehicles().set_index("vehicle_id")
        out: dict[str, CoverageStat] = {}
        for oem, g in df.groupby("oem", sort=True):
            per_vehicle = g.groupby("vehicle_id")[signal_id].apply(lambda s: bool(s.notna().any()))
            eligible = np.ones(len(per_vehicle), dtype=bool)
            if meta.powertrain != "any":
                eligible = (veh.loc[per_vehicle.index, "powertrain"] == meta.powertrain).to_numpy()
            total = int(eligible.sum())
            covered = int((per_vehicle.to_numpy() & eligible).sum())
            declared = self.declared_emits(str(oem), signal_id)
            if declared is not None:
                emits, from_my = declared
                emits = bool(emits and total > 0)
            else:
                emits = total > 0 and covered / total >= EMITS_MIN_VEHICLE_SHARE
                from_my = _infer_from_model_year(per_vehicle, veh, eligible) if emits else None
            out[str(oem)] = CoverageStat(
                oem=str(oem),
                emits=emits,
                from_model_year=from_my if emits else None,
                vehicles_total=total,
                vehicles_covered=covered,
                observed_nonnull_rate=float(g[signal_id].notna().mean()),
            )
        return out

    def quality(
        self, signal_id: str, window: DateWindow | None = None, frame: pd.DataFrame | None = None
    ) -> QualityStat:
        df = self._column(signal_id) if frame is None else frame[["vehicle_id", "date", signal_id]]
        df = df.copy()
        df["date"] = pd.to_datetime(df["date"])
        if window:
            df = df[
                (df["date"] >= pd.Timestamp(window.start))
                & (df["date"] <= pd.Timestamp(window.end))
            ]
        present = df[df[signal_id].notna()]
        nonnull = float(len(present) / max(len(df), 1))
        veh_ids = self.vehicles()["vehicle_id"]
        sample_ids = set(veh_ids.iloc[:: max(1, len(veh_ids) // 400)])
        sub = present[present["vehicle_id"].isin(sample_ids)]
        if len(sub) > 1:
            gaps = (
                sub.sort_values(["vehicle_id", "date"])
                .groupby("vehicle_id")["date"]
                .diff()
                .dt.days.dropna()
            )
            median_gap = float(gaps.median()) if len(gaps) else float("nan")
        else:
            median_gap = float("nan")
        declared = DECLARED_GAP_DAYS[self.signal_metadata(signal_id).declared_frequency]
        psi = _psi(present, signal_id) if len(present) > 200 else 0.0
        months = (df["date"].max() - df["date"].min()).days / 30.44 if len(df) else 0.0
        return QualityStat(
            signal_id=signal_id,
            nonnull_rate=nonnull,
            median_gap_days=median_gap,
            declared_gap_days=declared,
            psi_first_last_quarter=psi,
            history_months=float(months),
        )

    def training_frame(self, spec: ProblemSpec, signals: list[str] | None = None) -> LabeledFrame:
        sig_cols = signals or [m.signal_id for m in self.list_signals()]
        df = self.read_signals(sig_cols)
        df["date"] = pd.to_datetime(df["date"])
        if spec.powertrain_scope != "any":
            veh = self.vehicles()
            keep = veh.loc[veh["powertrain"] == spec.powertrain_scope, "vehicle_id"]
            df = df[df["vehicle_id"].isin(set(keep))]
        ev = self.events()
        ev = ev[ev["event_type"] == spec.target_event][["vehicle_id", "date"]].rename(
            columns={"date": "next_event"}
        )
        ev["next_event"] = pd.to_datetime(ev["next_event"])
        ev = ev.sort_values("next_event")
        df = df.sort_values("date")
        merged = pd.merge_asof(
            df[["vehicle_id", "date"]],
            ev,
            left_on="date",
            right_on="next_event",
            by="vehicle_id",
            direction="forward",
            allow_exact_matches=False,
        )
        days_to = (merged["next_event"] - merged["date"]).dt.days
        df["y"] = ((days_to > 0) & (days_to <= spec.horizon_days)).astype(np.int8).to_numpy()
        present = [c for c in sig_cols if c in df.columns]
        last = df["date"].max() - pd.Timedelta(days=spec.horizon_days)
        unknowable = (df["date"] > last) | ~df["active"].astype(bool)
        df.loc[unknowable, "y"] = -1
        df = df.sort_values(["vehicle_id", "date"]).reset_index(drop=True)
        known = df["y"] >= 0
        return LabeledFrame(
            frame=df,
            signal_columns=tuple(present),
            positive_rate=float(df.loc[known, "y"].mean()) if known.any() else float("nan"),
            n_vehicles=int(df["vehicle_id"].nunique()),
            n_days=int(df["date"].nunique()),
        )


def _infer_from_model_year(
    per_vehicle: pd.Series, veh: pd.DataFrame, eligible: np.ndarray
) -> int | None:
    my = veh.loc[per_vehicle.index, "model_year"].to_numpy()
    cov = per_vehicle.to_numpy() & eligible
    years = sorted(set(my[eligible]))
    for y in years:
        m = (my == y) & eligible
        if m.sum() and cov[m].mean() >= MODEL_YEAR_COVERAGE:
            return int(y)
    return None


def _psi(present: pd.DataFrame, col: str, bins: int = 10) -> float:
    present = present.sort_values("date")
    q = len(present) // 4
    a, b = present[col].iloc[:q].to_numpy(dtype=float), present[col].iloc[-q:].to_numpy(dtype=float)
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    if len(a) < 50 or len(b) < 50 or np.nanstd(a) == 0:
        return 0.0
    edges = np.quantile(a, np.linspace(0, 1, bins + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    pa_, _ = np.histogram(a, edges)
    pb_, _ = np.histogram(b, edges)
    pa_ = np.clip(pa_ / pa_.sum(), 1e-6, None)
    pb_ = np.clip(pb_ / pb_.sum(), 1e-6, None)
    return float(np.sum((pb_ - pa_) * np.log(pb_ / pa_)))


def as_date(s: str) -> date:
    return date.fromisoformat(s)

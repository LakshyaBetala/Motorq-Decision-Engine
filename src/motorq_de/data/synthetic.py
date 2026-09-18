"""SyntheticSource: DataSource over the generated parquet dataset."""

from __future__ import annotations

import json
from datetime import date
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from motorq_de.data.source import LabeledFrame
from motorq_de.schemas import CoverageStat, DateWindow, ProblemSpec, QualityStat, SignalMeta
from motorq_de.world import registry as reg
from motorq_de.world.config import load_coverage

META_COLS = ("vehicle_id", "date", "oem", "model_year", "active")
DECLARED_GAP_DAYS = {"realtime": 1.0, "daily": 1.0, "weekly": 7.0}


class SyntheticSource:
    def __init__(self, dataset_dir: Path):
        self.dir = Path(dataset_dir)
        self.dataset_hash = self.dir.name
        self._signals_path = self.dir / "signals_daily.parquet"
        self._coverage = load_coverage()
        self._truth = json.loads((self.dir / "truth.json").read_text(encoding="utf-8"))
        self._vehicles = pd.read_parquet(self.dir / "vehicles.parquet")
        self._events = pd.read_parquet(self.dir / "events.parquet")
        self._events["date"] = pd.to_datetime(self._events["date"])
        self._schema_cols = pq.read_schema(self._signals_path).names

    # ----------------------------------------------------------------- catalog
    def list_signals(self) -> list[SignalMeta]:
        return [s.meta() for s in reg.SIGNALS]

    def signal_metadata(self, signal_id: str) -> SignalMeta:
        return reg.BY_ID[signal_id].meta()

    def vehicles(self) -> pd.DataFrame:
        return self._vehicles.copy()

    def events(self) -> pd.DataFrame:
        return self._events.copy()

    def truth(self) -> dict:
        return self._truth

    def date_range(self) -> DateWindow:
        tbl = pq.read_table(self._signals_path, columns=["date"])
        d = tbl.column("date").to_pandas()
        return DateWindow(start=d.min().date(), end=d.max().date())

    # ----------------------------------------------------------------- column access
    @lru_cache(maxsize=256)
    def _column(self, signal_id: str) -> pd.DataFrame:
        cols = ["vehicle_id", "date", "oem", "model_year", signal_id]
        return pq.read_table(self._signals_path, columns=cols).to_pandas()

    def read_signals(self, signals: list[str]) -> pd.DataFrame:
        cols = list(META_COLS) + [s for s in signals if s in self._schema_cols]
        df = pq.read_table(self._signals_path, columns=cols).to_pandas()
        df["date"] = pd.to_datetime(df["date"])
        return df

    # ----------------------------------------------------------------- coverage / quality
    def coverage_by_oem(self, signal_id: str) -> dict[str, CoverageStat]:
        df = self._column(signal_id)
        out: dict[str, CoverageStat] = {}
        sdef = reg.BY_ID[signal_id]
        for oem, g in df.groupby("oem", sort=True):
            per_vehicle = g.groupby("vehicle_id")[signal_id].apply(lambda s: s.notna().any())
            cov = self._coverage.coverage(oem, signal_id)
            # a vehicle counts as covered if it ever emitted the signal
            vehicles_total = int(per_vehicle.size)
            vehicles_covered = int(per_vehicle.sum())
            eligible_pt = True
            if sdef.powertrain != "any":
                pts = self._vehicles.set_index("vehicle_id").loc[per_vehicle.index, "powertrain"]
                eligible = (pts == sdef.powertrain).to_numpy()
                vehicles_total = int(eligible.sum())
                vehicles_covered = int((per_vehicle.to_numpy() & eligible).sum())
                eligible_pt = vehicles_total > 0
            out[oem] = CoverageStat(
                oem=oem,
                emits=bool(cov.emits and eligible_pt),
                from_model_year=cov.from_model_year if cov.emits else None,
                vehicles_total=vehicles_total,
                vehicles_covered=vehicles_covered,
                observed_nonnull_rate=float(g[signal_id].notna().mean()),
            )
        return out

    def quality(self, signal_id: str, window: DateWindow | None = None) -> QualityStat:
        df = self._column(signal_id)
        df["date"] = pd.to_datetime(df["date"])
        if window:
            df = df[
                (df["date"] >= pd.Timestamp(window.start))
                & (df["date"] <= pd.Timestamp(window.end))
            ]
        present = df[df[signal_id].notna()]
        nonnull = float(len(present) / max(len(df), 1))
        # gap statistics on a fixed subsample of vehicles (deterministic, cheap at scale)
        sample_ids = self._vehicles["vehicle_id"].iloc[:: max(1, len(self._vehicles) // 400)]
        sub = present[present["vehicle_id"].isin(set(sample_ids))]
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
        sdef = reg.BY_ID[signal_id]
        declared = DECLARED_GAP_DAYS[sdef.frequency]
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

    # ----------------------------------------------------------------- labelled frame
    def training_frame(self, spec: ProblemSpec, signals: list[str] | None = None) -> LabeledFrame:
        sig_cols = signals or reg.signal_ids()
        df = self.read_signals(sig_cols)
        if spec.powertrain_scope != "any":
            keep = self._vehicles.loc[
                self._vehicles["powertrain"] == spec.powertrain_scope, "vehicle_id"
            ]
            df = df[df["vehicle_id"].isin(set(keep))]
        ev = self._events[self._events["event_type"] == spec.target_event][["vehicle_id", "date"]]
        ev = ev.rename(columns={"date": "next_event"}).sort_values("next_event")
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
        y = ((days_to > 0) & (days_to <= spec.horizon_days)).astype(np.int8).to_numpy()
        df["y"] = y
        # rows whose label window runs past the end of data are unknowable; a stolen /
        # blacked-out vehicle-day has no signals at all. Both are kept in the grid (rolling
        # features need the complete grid) but marked y = -1 for the harness to drop.
        present = [c for c in sig_cols if c in df.columns]
        last = df["date"].max() - pd.Timedelta(days=spec.horizon_days)
        unknowable = (df["date"] > last) | ~df["active"].astype(bool)
        df.loc[unknowable, "y"] = -1
        df = df.sort_values(["vehicle_id", "date"]).reset_index(drop=True)
        known = df["y"] >= 0
        return LabeledFrame(
            frame=df,
            signal_columns=tuple(present),
            positive_rate=float(df.loc[known, "y"].mean()),
            n_vehicles=int(df["vehicle_id"].nunique()),
            n_days=int(df["date"].nunique()),
        )


def _psi(present: pd.DataFrame, col: str, bins: int = 10) -> float:
    """Population stability index between first and last quarter of the history."""
    present = present.sort_values("date")
    q = len(present) // 4
    a, b = present[col].iloc[:q].to_numpy(), present[col].iloc[-q:].to_numpy()
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


def dataset_dirs(root: Path) -> list[Path]:
    return sorted([p for p in Path(root).glob("*") if (p / "truth.json").exists()])


def latest_dataset(root: Path) -> Path:
    dirs = dataset_dirs(root)
    if not dirs:
        raise FileNotFoundError(f"No datasets under {root}; run `mde world generate`")
    return max(dirs, key=lambda p: p.stat().st_mtime)


def as_date(s: str) -> date:
    return date.fromisoformat(s)

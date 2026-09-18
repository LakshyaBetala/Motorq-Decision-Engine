"""The DataSource protocol — the one seam between demonstration and finding.

Every tool upstream consumes this interface and nothing else about the data.
`SyntheticSource` implements it over the generated parquet; `SnowflakeSource`
documents how the same calls map onto Motorq's normalised tables.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import pandas as pd

from motorq_de.schemas import CoverageStat, DateWindow, ProblemSpec, QualityStat, SignalMeta


@dataclass(frozen=True)
class LabeledFrame:
    """One row per vehicle-day with raw signal values and the label for `spec`.

    Columns: vehicle_id, date, oem, model_year, y, <signal columns...>.
    The grid is complete (every vehicle, every day, sorted by vehicle then date). Rows whose
    label is unknowable carry y = -1 and must be dropped *after* feature engineering.
    Feature engineering (rolling windows, deltas) is the harness's job, not the source's,
    so that the same engineering applies identically to synthetic and Snowflake data.
    """

    frame: pd.DataFrame
    signal_columns: tuple[str, ...]
    positive_rate: float
    n_vehicles: int
    n_days: int


@runtime_checkable
class DataSource(Protocol):
    dataset_hash: str

    def list_signals(self) -> list[SignalMeta]: ...

    def signal_metadata(self, signal_id: str) -> SignalMeta: ...

    def coverage_by_oem(self, signal_id: str) -> dict[str, CoverageStat]: ...

    def quality(self, signal_id: str, window: DateWindow | None = None) -> QualityStat: ...

    def vehicles(self) -> pd.DataFrame: ...

    def events(self) -> pd.DataFrame: ...

    def training_frame(
        self, spec: ProblemSpec, signals: list[str] | None = None
    ) -> LabeledFrame: ...

    def date_range(self) -> DateWindow: ...

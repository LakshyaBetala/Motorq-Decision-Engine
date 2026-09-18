"""SyntheticSource: DataSource over the generated parquet dataset."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from motorq_de.data.frame_source import META_COLS, FrameSource
from motorq_de.schemas import SignalMeta
from motorq_de.world import registry as reg
from motorq_de.world.config import load_coverage


class SyntheticSource(FrameSource):
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
        self._col_cache: dict[str, pd.DataFrame] = {}

    def list_signals(self) -> list[SignalMeta]:
        return [s.meta() for s in reg.SIGNALS]

    def vehicles(self) -> pd.DataFrame:
        return self._vehicles.copy()

    def events(self) -> pd.DataFrame:
        return self._events.copy()

    def truth(self) -> dict:
        return self._truth

    def read_signals(self, signals: list[str]) -> pd.DataFrame:
        cols = list(META_COLS) + [s for s in signals if s in self._schema_cols]
        df = pq.read_table(self._signals_path, columns=cols).to_pandas()
        df["date"] = pd.to_datetime(df["date"])
        return df

    def _column(self, signal_id: str) -> pd.DataFrame:
        if signal_id not in self._col_cache:
            if len(self._col_cache) > 128:
                self._col_cache.clear()
            self._col_cache[signal_id] = self.read_signals([signal_id])
        return self._col_cache[signal_id].copy()

    def declared_emits(self, oem: str, signal_id: str) -> tuple[bool, int] | None:
        cov = self._coverage.coverage(oem, signal_id)
        return cov.emits, cov.from_model_year


def dataset_dirs(root: Path) -> list[Path]:
    return sorted([p for p in Path(root).glob("*") if (p / "truth.json").exists()])


def latest_dataset(root: Path) -> Path:
    dirs = dataset_dirs(root)
    if not dirs:
        raise FileNotFoundError(f"No datasets under {root}; run `mde world generate`")
    return max(dirs, key=lambda p: p.stat().st_mtime)

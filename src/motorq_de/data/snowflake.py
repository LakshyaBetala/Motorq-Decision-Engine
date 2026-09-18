"""SnowflakeSource: the adapter that turns a demonstration into a finding.

Reads Motorq's normalised tables into the three frames FrameSource needs and reuses the
exact labelling / coverage / quality logic that runs on synthetic data. Table and column
names are configuration, not code (see `SnowflakeConfig`), so pointing this at a real
schema is an edit to a YAML file.

Expected shapes (rename via config):

    VEHICLES        vin, oem, model_year, powertrain[, segment, ...]
    SIGNALS_DAILY   vin, day, <one column per signal>        (wide, one row per vin-day)
      - or -        vin, day, signal_id, value                (long; pivoted client-side)
    EVENTS          vin, event_date, event_type[, detail]
    SIGNAL_CATALOG  signal_id, category, unit, description, declared_frequency, powertrain

A wide daily table is the natural product of a Snowflake dynamic table over the streaming
Iceberg landing tables (Motorq lands data via Snowpipe Streaming into managed Iceberg,
see https://motorq.com/blog-news/motorq-connected-vehicle-data-snowflake).

Scoping: `max_vehicles` samples VINs deterministically (HASH(vin) modulo) so a study runs on
tens of thousands of vehicles rather than the whole platform; `date_from`/`date_to` bound the
history. The query function is injectable so the adapter is testable without Snowflake.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from motorq_de.data.frame_source import META_COLS, FrameSource
from motorq_de.schemas import SignalMeta

QueryFn = Callable[[str], pd.DataFrame]


@dataclass
class SnowflakeConfig:
    tables: dict[str, str] = field(
        default_factory=lambda: {
            "vehicles": "VEHICLES",
            "signals_daily": "SIGNALS_DAILY",
            "events": "EVENTS",
            "catalog": "SIGNAL_CATALOG",
        }
    )
    columns: dict[str, str] = field(
        default_factory=lambda: {
            "vin": "VIN",
            "day": "DAY",
            "oem": "OEM",
            "model_year": "MODEL_YEAR",
            "powertrain": "POWERTRAIN",
            "event_type": "EVENT_TYPE",
            "event_date": "EVENT_DATE",
            "detail": "DETAIL",
            "signal_id": "SIGNAL_ID",
            "value": "VALUE",
        }
    )
    signals_layout: str = "wide"  # wide | long
    max_vehicles: int | None = 20000
    vin_sample_modulus: int = 1000  # HASH(vin) % modulus < threshold selects the sample
    date_from: str | None = None
    date_to: str | None = None
    event_type_map: dict[str, str] = field(default_factory=dict)  # engine target -> table value

    def __post_init__(self) -> None:
        # partial overrides merge over the defaults
        d = SnowflakeConfig.__dataclass_fields__
        self.tables = {**d["tables"].default_factory(), **self.tables}  # type: ignore[misc]
        self.columns = {**d["columns"].default_factory(), **self.columns}  # type: ignore[misc]

    @classmethod
    def from_yaml(cls, path: Path) -> SnowflakeConfig:
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        cfg = cls()
        cfg.tables.update(raw.get("tables", {}))
        cfg.columns.update(raw.get("columns", {}))
        cfg.signals_layout = raw.get("signals_layout", cfg.signals_layout)
        cfg.max_vehicles = raw.get("max_vehicles", cfg.max_vehicles)
        cfg.date_from = raw.get("date_from")
        cfg.date_to = raw.get("date_to")
        cfg.event_type_map = raw.get("event_type_map", {})
        return cfg


def connector_query(connection_params: dict[str, Any]) -> QueryFn:
    """Build a query function on snowflake-connector-python (optional dependency)."""
    import snowflake.connector  # type: ignore[import-not-found]

    conn = snowflake.connector.connect(**connection_params)

    def run(sql: str) -> pd.DataFrame:
        cur = conn.cursor()
        try:
            cur.execute(sql)
            return cur.fetch_pandas_all()
        finally:
            cur.close()

    return run


class SnowflakeSource(FrameSource):
    def __init__(
        self, query: QueryFn, config: SnowflakeConfig | None = None, label: str = "snowflake"
    ):
        self.query = query
        self.cfg = config or SnowflakeConfig()
        self.dataset_hash = self._hash(label)
        self._catalog: list[SignalMeta] | None = None
        self._vehicles: pd.DataFrame | None = None
        self._events: pd.DataFrame | None = None
        self._grid_cache: dict[tuple[str, ...], pd.DataFrame] = {}

    # ------------------------------------------------------------- identity
    def _hash(self, label: str) -> str:
        key = f"{label}|{self.cfg.tables}|{self.cfg.max_vehicles}|{self.cfg.date_from}|{self.cfg.date_to}"
        return "sf_" + hashlib.sha256(key.encode()).hexdigest()[:13]

    # ------------------------------------------------------------- SQL fragments
    def _c(self, k: str) -> str:
        return self.cfg.columns[k]

    def _t(self, k: str) -> str:
        return self.cfg.tables[k]

    def _vin_filter(self) -> str:
        if not self.cfg.max_vehicles:
            return "TRUE"
        # deterministic sample: keep VINs whose hash bucket falls below the sampling threshold;
        # the threshold is refined client-side to the exact vehicle count
        return f"ABS(HASH({self._c('vin')})) % {self.cfg.vin_sample_modulus} < {self.cfg.vin_sample_modulus}"

    def _date_filter(self, col: str) -> str:
        parts = []
        if self.cfg.date_from:
            parts.append(f"{col} >= '{self.cfg.date_from}'")
        if self.cfg.date_to:
            parts.append(f"{col} <= '{self.cfg.date_to}'")
        return " AND ".join(parts) if parts else "TRUE"

    def sql_vehicles(self) -> str:
        c = self._c
        return (
            f"SELECT {c('vin')} AS vehicle_id, {c('oem')} AS oem, {c('model_year')} AS model_year, "
            f"{c('powertrain')} AS powertrain FROM {self._t('vehicles')} WHERE {self._vin_filter()}"
        )

    def sql_events(self) -> str:
        c = self._c
        return (
            f"SELECT {c('vin')} AS vehicle_id, {c('event_date')}::date AS date, {c('event_type')} AS event_type, "
            f"{c('detail')} AS detail FROM {self._t('events')} WHERE {self._vin_filter()} AND {self._date_filter(c('event_date'))}"
        )

    def sql_catalog(self) -> str:
        return f"SELECT signal_id, category, unit, description, declared_frequency, powertrain FROM {self._t('catalog')}"

    def sql_signals(self, signals: list[str]) -> str:
        c = self._c
        if self.cfg.signals_layout == "wide":
            cols = ", ".join(f"s.{s} AS {s}" for s in signals)
            sel = f", {cols}" if cols else ""
            return (
                f"SELECT s.{c('vin')} AS vehicle_id, s.{c('day')}::date AS date, v.{c('oem')} AS oem, "
                f"v.{c('model_year')} AS model_year{sel} FROM {self._t('signals_daily')} s "
                f"JOIN {self._t('vehicles')} v ON v.{c('vin')} = s.{c('vin')} "
                f"WHERE {self._vin_filter().replace(c('vin'), 's.' + c('vin'))} AND {self._date_filter('s.' + c('day'))}"
            )
        in_list = ", ".join(f"'{s}'" for s in signals) or "''"
        return (
            f"SELECT s.{c('vin')} AS vehicle_id, s.{c('day')}::date AS date, v.{c('oem')} AS oem, "
            f"v.{c('model_year')} AS model_year, s.{c('signal_id')} AS signal_id, s.{c('value')} AS value "
            f"FROM {self._t('signals_daily')} s JOIN {self._t('vehicles')} v ON v.{c('vin')} = s.{c('vin')} "
            f"WHERE s.{c('signal_id')} IN ({in_list}) AND {self._vin_filter().replace(c('vin'), 's.' + c('vin'))} "
            f"AND {self._date_filter('s.' + c('day'))}"
        )

    # ------------------------------------------------------------- frames
    def list_signals(self) -> list[SignalMeta]:
        if self._catalog is None:
            df = self.query(self.sql_catalog())
            df.columns = [c.lower() for c in df.columns]
            self._catalog = [
                SignalMeta(
                    signal_id=str(r.signal_id),
                    category=str(r.category),
                    unit=str(r.unit),
                    description=str(r.description),
                    declared_frequency=str(r.declared_frequency),
                    powertrain=str(getattr(r, "powertrain", "any") or "any"),
                )
                for r in df.itertuples(index=False)
            ]
        return list(self._catalog)

    def vehicles(self) -> pd.DataFrame:
        if self._vehicles is None:
            df = self.query(self.sql_vehicles())
            df.columns = [c.lower() for c in df.columns]
            if self.cfg.max_vehicles and len(df) > self.cfg.max_vehicles:
                # exact deterministic sample by vin hash
                h = df["vehicle_id"].map(
                    lambda v: int(hashlib.sha256(str(v).encode()).hexdigest()[:8], 16)
                )
                df = df.loc[np.argsort(h.to_numpy(), kind="mergesort")[: self.cfg.max_vehicles]]
            df["model_year"] = df["model_year"].astype(int)
            self._vehicles = df.sort_values("vehicle_id").reset_index(drop=True)
        return self._vehicles.copy()

    def events(self) -> pd.DataFrame:
        if self._events is None:
            df = self.query(self.sql_events())
            df.columns = [c.lower() for c in df.columns]
            df["date"] = pd.to_datetime(df["date"])
            inv = {v: k for k, v in self.cfg.event_type_map.items()}
            if inv:
                df["event_type"] = df["event_type"].map(lambda x: inv.get(x, x))
            ids = set(self.vehicles()["vehicle_id"])
            self._events = df[df["vehicle_id"].isin(ids)].reset_index(drop=True)
        return self._events.copy()

    def read_signals(self, signals: list[str]) -> pd.DataFrame:
        key = tuple(sorted(signals))
        if key in self._grid_cache:
            return self._grid_cache[key].copy()
        df = self.query(self.sql_signals(list(key)))
        df.columns = [c.lower() for c in df.columns]
        if self.cfg.signals_layout == "long":
            df = df.pivot_table(
                index=["vehicle_id", "date", "oem", "model_year"],
                columns="signal_id",
                values="value",
                aggfunc="mean",
            ).reset_index()
            df.columns.name = None
        veh = self.vehicles()
        df = df[df["vehicle_id"].isin(set(veh["vehicle_id"]))]
        df["date"] = pd.to_datetime(df["date"])
        grid = _complete_grid(df, veh, list(key))
        if len(self._grid_cache) > 8:
            self._grid_cache.clear()
        self._grid_cache[key] = grid
        return grid.copy()


def _complete_grid(df: pd.DataFrame, veh: pd.DataFrame, signals: list[str]) -> pd.DataFrame:
    """Production tables have rows only on days with data; the harness needs a complete
    vehicle x day grid. Missing days become rows with NaN signals; `active` marks days on
    which the vehicle reported anything at all."""
    days = pd.date_range(df["date"].min(), df["date"].max(), freq="D")
    idx = pd.MultiIndex.from_product(
        [veh["vehicle_id"].to_numpy(), days], names=["vehicle_id", "date"]
    )
    present = df.set_index(["vehicle_id", "date"])
    grid = present.reindex(idx)
    grid["active"] = present.reindex(idx).index.isin(present.index)
    grid = grid.reset_index()
    meta = veh.set_index("vehicle_id")
    grid["oem"] = meta.loc[grid["vehicle_id"], "oem"].to_numpy()
    grid["model_year"] = meta.loc[grid["vehicle_id"], "model_year"].to_numpy()
    cols = list(META_COLS) + [s for s in signals if s in grid.columns]
    for s in signals:
        if s not in grid.columns:
            grid[s] = np.nan
            cols.append(s)
    grid = grid[cols].sort_values(["vehicle_id", "date"]).reset_index(drop=True)
    for s in signals:
        grid[s] = grid[s].astype("float32")
    return grid

"""SnowflakeSource: the adapter that turns a demonstration into a finding.

NOT IMPLEMENTED in this cycle. Every method documents the SQL shape it would run
against Motorq's normalised tables so that wiring it is an adapter task, not a rewrite.

Assumed table shapes (to be replaced with Motorq's actual schema):

    VEHICLES(vin, oem, make, model, model_year, segment, powertrain, region)
    SIGNALS(vin, ts, signal_id, value_num, value_str)           -- long, normalised
    EVENTS(vin, ts, event_type, detail)                          -- maintenance, theft, ...
    SIGNAL_CATALOG(signal_id, category, unit, description, declared_frequency)

Delivery: Motorq lands data in Snowflake via Snowpipe Streaming into managed Iceberg
tables (source: https://motorq.com/blog-news/motorq-connected-vehicle-data-snowflake),
so daily aggregation below would be a scheduled task or dynamic table.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from motorq_de.data.source import LabeledFrame
from motorq_de.schemas import CoverageStat, DateWindow, ProblemSpec, QualityStat, SignalMeta


class SnowflakeSource:
    def __init__(self, connection_params: dict[str, Any], database: str, schema: str):
        self.dataset_hash = "snowflake:" + database + "." + schema
        self._conn = connection_params
        self._db, self._schema = database, schema

    def _not_ready(self, sql: str) -> None:
        raise NotImplementedError(
            "SnowflakeSource is a typed stub in this cycle. Intended query:\n" + sql
        )

    def list_signals(self) -> list[SignalMeta]:
        self._not_ready(
            "SELECT signal_id, category, unit, description, declared_frequency FROM SIGNAL_CATALOG"
        )
        return []

    def signal_metadata(self, signal_id: str) -> SignalMeta:
        self._not_ready("SELECT * FROM SIGNAL_CATALOG WHERE signal_id = :signal_id")
        raise AssertionError

    def coverage_by_oem(self, signal_id: str) -> dict[str, CoverageStat]:
        self._not_ready(
            """
            WITH per_vehicle AS (
              SELECT v.vin, v.oem, MAX(CASE WHEN s.signal_id = :signal_id THEN 1 ELSE 0 END) AS emitted
              FROM VEHICLES v LEFT JOIN SIGNALS s ON s.vin = v.vin AND s.ts >= DATEADD(day, -90, CURRENT_DATE)
              GROUP BY v.vin, v.oem)
            SELECT oem, COUNT(*) AS vehicles_total, SUM(emitted) AS vehicles_covered
            FROM per_vehicle GROUP BY oem
            """
        )
        return {}

    def quality(self, signal_id: str, window: DateWindow | None = None) -> QualityStat:
        self._not_ready(
            """
            SELECT vin, ts::date AS d, COUNT(*) AS n
            FROM SIGNALS WHERE signal_id = :signal_id AND ts BETWEEN :start AND :end
            GROUP BY vin, d  -- gaps, non-null rate and PSI computed client-side
            """
        )
        raise AssertionError

    def vehicles(self) -> pd.DataFrame:
        self._not_ready(
            "SELECT vin AS vehicle_id, oem, model_year, segment, powertrain FROM VEHICLES"
        )
        raise AssertionError

    def events(self) -> pd.DataFrame:
        self._not_ready(
            "SELECT vin AS vehicle_id, ts::date AS date, event_type, detail FROM EVENTS"
        )
        raise AssertionError

    def training_frame(self, spec: ProblemSpec, signals: list[str] | None = None) -> LabeledFrame:
        self._not_ready(
            """
            -- daily wide frame: one row per vin-day with one column per signal
            SELECT vin AS vehicle_id, ts::date AS date, oem, model_year,
                   {pivot of AVG/LAST value per signal_id in :signals}
            FROM SIGNALS s JOIN VEHICLES v USING (vin)
            WHERE ts BETWEEN :start AND :end
            GROUP BY vin, ts::date, oem, model_year
            -- label: EXISTS event of :target_event in (date, date + :horizon_days]
            """
        )
        raise AssertionError

    def date_range(self) -> DateWindow:
        self._not_ready("SELECT MIN(ts)::date, MAX(ts)::date FROM SIGNALS")
        raise AssertionError

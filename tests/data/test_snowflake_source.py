"""SnowflakeSource is exercised with a fake query function that serves the synthetic dataset
in production shape (uppercase columns, VIN naming, rows only on days with data). The adapter
must reproduce SyntheticSource's labels and infer coverage that matches the declared matrix."""

from __future__ import annotations

import re

import numpy as np
import pandas as pd
import pytest

from motorq_de.data.snowflake import SnowflakeConfig, SnowflakeSource
from motorq_de.data.source import DataSource
from motorq_de.world import registry as reg


def fake_query(source):
    """Serve our own SQL from the synthetic parquet. Dispatches on table name."""
    veh = source.vehicles()
    ev = source.events()

    def run(sql: str) -> pd.DataFrame:
        if "FROM SIGNAL_CATALOG" in sql:
            rows = [m.model_dump() for m in source.list_signals()]
            df = pd.DataFrame(rows)
            return df.rename(columns=str.upper)
        if "FROM VEHICLES" in sql:
            return veh[["vehicle_id", "oem", "model_year", "powertrain"]].rename(columns=str.upper)
        if "FROM EVENTS" in sql:
            e = ev.rename(columns={"date": "DATE"}).rename(columns=str.upper)
            return e[["VEHICLE_ID", "DATE", "EVENT_TYPE", "DETAIL"]]
        if "FROM SIGNALS_DAILY" in sql:
            sigs = re.findall(r"s\.(\w+) AS \1", sql)
            df = source.read_signals(sigs)
            df = df[df["active"].astype(bool)]  # production tables have rows only on reporting days
            df = df.drop(columns=["active"])
            return df.rename(columns=str.upper)
        raise AssertionError(sql)

    return run


@pytest.fixture(scope="module")
def sf(source):
    cfg = SnowflakeConfig(max_vehicles=None)
    return SnowflakeSource(fake_query(source), cfg, label="test")


def test_satisfies_protocol_and_catalog(sf):
    assert isinstance(sf, DataSource)
    assert {m.signal_id for m in sf.list_signals()} == set(reg.signal_ids())
    assert sf.dataset_hash.startswith("sf_")


def test_sql_uses_configured_names():
    cfg = SnowflakeConfig(
        tables={
            "vehicles": "DB.S.VEH",
            "signals_daily": "DB.S.SIG",
            "events": "DB.S.EV",
            "catalog": "DB.S.CAT",
        },
        columns={"vin": "VEHICLE_VIN", "day": "OBS_DAY"},
        max_vehicles=None,
    )
    s = SnowflakeSource(lambda q: pd.DataFrame(), cfg)
    assert "DB.S.VEH" in s.sql_vehicles() and "VEHICLE_VIN AS vehicle_id" in s.sql_vehicles()
    q = s.sql_signals(["brake_pad_wear_pct"])
    assert (
        "DB.S.SIG" in q
        and "s.OBS_DAY::date AS date" in q
        and "s.brake_pad_wear_pct AS brake_pad_wear_pct" in q
    )


def test_long_layout_sql_pivots():
    cfg = SnowflakeConfig(signals_layout="long", max_vehicles=None)
    s = SnowflakeSource(lambda q: pd.DataFrame(), cfg)
    assert "SIGNAL_ID IN ('a', 'b')" in s.sql_signals(["a", "b"])


def test_grid_is_complete_and_active_marks_reporting_days(sf, source):
    g = sf.read_signals(["odometer_delta_mi"])
    n_veh = g["vehicle_id"].nunique()
    assert len(g) == n_veh * g["date"].nunique()
    ref = source.read_signals(["odometer_delta_mi"])
    assert g["active"].mean() == pytest.approx(ref["active"].astype(bool).mean(), abs=0.01)


def test_labels_match_synthetic_source(sf, source, brake_spec):
    a = sf.training_frame(brake_spec, ["odometer_delta_mi", "brake_pad_wear_pct"]).frame
    b = source.training_frame(brake_spec, ["odometer_delta_mi", "brake_pad_wear_pct"]).frame
    m = a.merge(b, on=["vehicle_id", "date"], suffixes=("_sf", "_syn"))
    assert len(m) == len(a) == len(b)
    assert (m["y_sf"] == m["y_syn"]).all()
    np.testing.assert_allclose(
        m["brake_pad_wear_pct_sf"].to_numpy(),
        m["brake_pad_wear_pct_syn"].to_numpy(),
        rtol=1e-5,
        equal_nan=True,
    )


def test_inferred_coverage_matches_declared_matrix(sf, source):
    inferred = sf.coverage_by_oem("brake_pad_wear_pct")
    declared = source.coverage_by_oem("brake_pad_wear_pct")
    for oem in declared:
        assert inferred[oem].emits == declared[oem].emits, oem
        assert inferred[oem].vehicles_covered == declared[oem].vehicles_covered
    # model-year inference lands at the declared connectivity year for sensor OEMs
    assert (
        inferred["oem_f"].from_model_year is not None and inferred["oem_f"].from_model_year <= 2020
    )
    assert inferred["oem_c"].from_model_year is None


def test_event_rate_and_quality_work_on_adapter(sf, brake_spec):
    from motorq_de.quality.checks import event_rate, quality_report

    er = event_rate(sf, brake_spec)
    assert 0.6 <= er["rate_per_vehicle_year"] <= 1.6
    q = quality_report(sf, ["brake_pad_wear_pct", "air_filter_life_pct"])
    assert q["per_signal"]["air_filter_life_pct"]["declared_gap_days"] == 7.0


def test_max_vehicles_sampling_is_deterministic(source):
    cfg = SnowflakeConfig(max_vehicles=100)
    a = SnowflakeSource(fake_query(source), cfg, label="t").vehicles()
    b = SnowflakeSource(fake_query(source), cfg, label="t").vehicles()
    assert len(a) == 100 and list(a["vehicle_id"]) == list(b["vehicle_id"])

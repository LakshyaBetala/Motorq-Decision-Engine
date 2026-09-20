# Running against Motorq data (Snowflake adapter)

Everything the engine does on synthetic data it does identically on production data through
`SnowflakeSource`. The adapter reads three frames plus a catalog and hands them to the same
labelling, coverage, quality, harness, economics and policy code that the tests exercise.

## 1. Tables the adapter expects

Names are configuration; shapes are what matters.

| Frame | Columns (rename in config) | Notes |
|---|---|---|
| `VEHICLES` | `vin, oem, model_year, powertrain` | `powertrain` in `{ice, ev}`; add `segment` etc. freely |
| `SIGNALS_DAILY` (wide) | `vin, day, <one column per signal>` | one row per vin-day on days with data; the adapter completes the grid |
| `SIGNALS_DAILY` (long) | `vin, day, signal_id, value` | pivoted client-side; set `signals_layout: long` |
| `EVENTS` | `vin, event_date, event_type[, detail]` | target events: maintenance/service, theft, battery service |
| `SIGNAL_CATALOG` | `signal_id, category, unit, description, declared_frequency, powertrain` | `declared_frequency` in `{realtime, daily, weekly}` |

A wide daily table is the natural product of a Snowflake dynamic table over the Snowpipe
Streaming / managed Iceberg landing tables Motorq already runs.

## 1a. The contract, checked before every study

`mde data check` (with `MDE_SNOWFLAKE_CONFIG` set) validates the three tables against the
canonical contract in `data/contract.py`: required columns and types, one row per vehicle /
per event / per vehicle-day, referential integrity, no future dates, history length, grid
completeness, and values on inactive days. Errors make the source unusable and every study
re-runs the same validation at its DEFINE stage and fails closed; warnings appear in the
brief's data section. Each catalogued signal's COVESA VSS path is in
`world/registry.py::VSS`, which is the mapping to use when reconciling a Motorq signal name
with an OEM's native one.

## 2. Config

`snowflake.yaml`:

```yaml
tables:
  vehicles: MOTORQ.NORMALIZED.VEHICLES
  signals_daily: MOTORQ.NORMALIZED.SIGNALS_DAILY
  events: MOTORQ.NORMALIZED.EVENTS
  catalog: MOTORQ.NORMALIZED.SIGNAL_CATALOG
columns:
  vin: VIN
  day: OBS_DAY
  oem: MAKE_GROUP
  model_year: MODEL_YEAR
  powertrain: POWERTRAIN
  event_type: EVENT_TYPE
  event_date: EVENT_TS
signals_layout: wide
max_vehicles: 20000          # deterministic VIN sample (HASH(vin)); null = all
date_from: "2025-01-01"
date_to: "2026-06-30"
event_type_map:               # engine target -> value in EVENTS.event_type
  brake_service_event: BRAKE_SERVICE
  theft_event: STOLEN
  battery_degradation_event: HV_BATTERY_SERVICE
```

Credentials come from the environment (`SNOWFLAKE_ACCOUNT`, `SNOWFLAKE_USER`,
`SNOWFLAKE_PASSWORD`, `SNOWFLAKE_WAREHOUSE`, `SNOWFLAKE_DATABASE`, `SNOWFLAKE_SCHEMA`,
`SNOWFLAKE_ROLE`). Install the extra: `uv sync --extra snowflake`.

## 3. Run

```bash
mde run headless --example brake --snowflake snowflake.yaml --out brief.md
# or for the API / dashboard
MDE_SNOWFLAKE_CONFIG=snowflake.yaml mde serve
```

## 4. What changes on real data

- **Coverage is inferred, not declared.** On synthetic data the OEM coverage matrix is known;
  on production data an OEM "emits" a signal when at least 20% of its eligible vehicles ever
  report it, and `from_model_year` is the first model year with >= 50% coverage. Both
  thresholds are constants in `data/frame_source.py`.
- **The event rate is measured** from `EVENTS` for the scoped population, with a Poisson CI.
  Only the value-bearing fraction, preventable fraction, $/event and fleet size remain human
  inputs.
- **Leakage detection matters more.** Dealer/DMS fields that are populated retroactively will
  be caught by the availability-jump and monotone-to-event detectors; very strong sensors are
  reported as "confirm availability at prediction time", not dropped.
- **Price sheet.** Replace the placeholders in `economics/price_sheet.yaml` with contracted
  rates; the `cost_placeholders` flag stops tripping once none remain.
- **Scale.** Feature analysis and ablation run on a class-balanced sample (`max_rows`,
  default 150k vehicle-days) drawn deterministically from the scoped population; 20k vehicles
  over 18 months is a comfortable study size.

## 5. Testing the adapter without Snowflake

`tests/data/test_snowflake_source.py` injects a fake query function that serves the synthetic
dataset in production shape (uppercase columns, VIN naming, rows only on reporting days) and
asserts the adapter reproduces the synthetic source's labels and coverage exactly. Point the
same fake at a CSV export of a real schema to validate a configuration before connecting.

-- SIGNALS_DAILY: one row per VIN-day, one column per signal, built as a Dynamic Table over the
-- Snowpipe Streaming / managed Iceberg landing table. The engine completes the grid client-side,
-- so only days with data need rows. Aggregation rule per signal:
--   counts  -> SUM over the day      (harsh_brake_count, dtc_* ...)
--   levels  -> last value of the day (brake_pad_wear_pct, odometer_mi, soc_* ...)
--   maxima  -> MAX                   (battery_temp_max_c, ambient_temp_max_c ...)
-- Assumed landing shape: SIGNALS_RAW(VIN, TS, SIGNAL_ID, VALUE_NUM)
CREATE OR REPLACE DYNAMIC TABLE MOTORQ.NORMALIZED.SIGNALS_DAILY
  TARGET_LAG = '1 hour'
  WAREHOUSE  = MDE_WH
AS
WITH d AS (
  SELECT VIN, TS::DATE AS DAY, SIGNAL_ID, VALUE_NUM, TS
  FROM MOTORQ.RAW.SIGNALS_RAW
  WHERE TS >= DATEADD(month, -24, CURRENT_DATE)
),
lastv AS (  -- last value of the day for level-type signals
  SELECT VIN, DAY, SIGNAL_ID, VALUE_NUM
  FROM d
  QUALIFY ROW_NUMBER() OVER (PARTITION BY VIN, DAY, SIGNAL_ID ORDER BY TS DESC) = 1
)
SELECT
  d.VIN,
  d.DAY,
  -- levels (last value)
  MAX(IFF(l.SIGNAL_ID = 'brake_pad_wear_pct', l.VALUE_NUM, NULL))        AS brake_pad_wear_pct,
  MAX(IFF(l.SIGNAL_ID = 'odometer_mi',        l.VALUE_NUM, NULL))        AS odometer_mi,
  MAX(IFF(l.SIGNAL_ID = 'oil_life_pct',       l.VALUE_NUM, NULL))        AS oil_life_pct,
  MAX(IFF(l.SIGNAL_ID = 'soc_min_daily',      l.VALUE_NUM, NULL))        AS soc_min_daily,
  -- counts (sum)
  SUM(IFF(d.SIGNAL_ID = 'harsh_brake_count',  d.VALUE_NUM, 0))           AS harsh_brake_count,
  SUM(IFF(d.SIGNAL_ID = 'dtc_brake_family',   d.VALUE_NUM, 0))           AS dtc_brake_family,
  -- maxima
  MAX(IFF(d.SIGNAL_ID = 'battery_temp_max_c', d.VALUE_NUM, NULL))        AS battery_temp_max_c
  -- ... one expression per SIGNAL_CATALOG row; generate this list from the catalog.
FROM d
LEFT JOIN lastv l ON l.VIN = d.VIN AND l.DAY = d.DAY AND l.SIGNAL_ID = d.SIGNAL_ID
GROUP BY d.VIN, d.DAY;

-- Alternative: keep the landing table long and set `signals_layout: long` in snowflake.yaml;
-- the engine pivots client-side (simpler, more data transferred).

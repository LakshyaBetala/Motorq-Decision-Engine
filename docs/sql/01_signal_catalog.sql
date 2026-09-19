-- SIGNAL_CATALOG: one row per normalised signal the engine may consider.
-- declared_frequency in ('realtime','daily','weekly'); powertrain in ('any','ev','ice').
-- Seed from Motorq's signal registry; the engine reads this table as the candidate list.
CREATE TABLE IF NOT EXISTS MOTORQ.NORMALIZED.SIGNAL_CATALOG (
  SIGNAL_ID            VARCHAR      NOT NULL PRIMARY KEY,
  CATEGORY             VARCHAR      NOT NULL,   -- location_trips | health | driver_behavior | fuel_energy | context | derived
  UNIT                 VARCHAR,
  DESCRIPTION          VARCHAR,
  DECLARED_FREQUENCY   VARCHAR      NOT NULL,   -- realtime | daily | weekly
  POWERTRAIN           VARCHAR      DEFAULT 'any'
);

-- Example rows (replace with the registry export):
-- INSERT INTO MOTORQ.NORMALIZED.SIGNAL_CATALOG VALUES
--   ('brake_pad_wear_pct','health','pct','Front brake pad wear (OEM-estimated)','daily','any'),
--   ('harsh_brake_count','driver_behavior','count','Harsh braking events','realtime','any'),
--   ('soc_min_daily','fuel_energy','pct','Minimum state of charge','realtime','ev');

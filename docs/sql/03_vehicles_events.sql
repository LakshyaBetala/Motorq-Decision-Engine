-- VEHICLES: the decision population. powertrain in ('ice','ev').
CREATE OR REPLACE VIEW MOTORQ.NORMALIZED.VEHICLES AS
SELECT VIN,
       MAKE_GROUP                           AS OEM,        -- e.g. GM, Ford, Toyota (the engine groups by this)
       MODEL_YEAR,
       IFF(FUEL_TYPE = 'BEV', 'ev', 'ice')  AS POWERTRAIN,
       SEGMENT
FROM MOTORQ.REF.VEHICLE_MASTER
WHERE ENROLLED = TRUE;

-- EVENTS: target events, one row each. Either emit the engine's target names directly (below)
-- or map source values with `event_type_map` in snowflake.yaml.
CREATE OR REPLACE VIEW MOTORQ.NORMALIZED.EVENTS AS
SELECT VIN, SERVICE_DATE::DATE AS EVENT_DATE, 'brake_service_event' AS EVENT_TYPE, TO_VARCHAR(WORK_ORDER_ID) AS DETAIL
FROM MOTORQ.MAINT.SERVICE_RECORDS WHERE COMPONENT = 'BRAKES'
UNION ALL
SELECT VIN, REPORTED_AT::DATE, 'theft_event', TO_VARCHAR(RECOVERY_DAYS)
FROM MOTORQ.RECOVERY.STOLEN_VEHICLE_CASES
UNION ALL
SELECT VIN, SERVICE_DATE::DATE, 'battery_degradation_event', TO_VARCHAR(WORK_ORDER_ID)
FROM MOTORQ.MAINT.SERVICE_RECORDS WHERE COMPONENT = 'HV_BATTERY';

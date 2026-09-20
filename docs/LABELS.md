# Label definitions for real data

A feasibility verdict is only as good as its label. On the synthetic fleet the label is
known by construction; on Motorq's data each target event has to be tied to a source of
truth, an event date, and a statement of when that fact becomes *known*, because anything
recorded after the event but dated before it is leakage.

This document is the proposed definition for each of the three example capabilities. Every
row marked **confirm** is a decision Motorq's data owners make, not the engine; the engine
records the chosen definition in the study spec so the verdict is reproducible from it.

The contract the engine enforces on every source is in [SNOWFLAKE.md](SNOWFLAKE.md) §1a:
`EVENTS(vin, event_date, event_type[, detail])`. What follows is how each `event_type` row
should be produced.

## Common rules

| Rule | Why |
|---|---|
| The event date is when the event *happened*, not when it was recorded. | A work order opened on day 12 for a failure on day 9 must be dated day 9, or the last three days before it become false negatives. |
| Every label row carries `known_at`, the date the fact entered the system (confirm: column name). | The engine's leakage detector flags signals that only exist once an appointment exists; `known_at` lets a study exclude rows the fleet could not have acted on. |
| One event per vehicle per episode. | Two work orders for the same repair are one event; otherwise event recall is inflated. |
| Vehicles with no label source at all are excluded, not treated as negatives. | A vehicle whose maintenance is done outside the recording system has unknown labels; calling them negatives biases recall upward. |
| The label source must be independent of the feature signals. | If "brake service" is derived from a brake DTC, a model that reads the DTC is predicting its own label. |

## brake_service_event

| Item | Proposed definition | Status |
|---|---|---|
| Source of truth | Fleet maintenance system or dealer management system work orders with a brake line item (pads, rotors, calipers, brake fluid flush excluded). | confirm: which systems Motorq receives, and the line-item taxonomy |
| Event date | Vehicle-in date on the work order; if absent, the invoice date minus the recorded shop time. | confirm |
| Horizon | 7 days by default: the lead a fleet manager needs to schedule the vehicle. | product decision |
| Known at | Appointment creation date. | confirm |
| Exclusions | Scheduled interval services with no brake wear finding (a pad measurement above the shop's replace threshold). | confirm: whether pad thickness is recorded |
| Leakage to watch | Appointment flags, service-interval countdowns, "vehicle at dealer" location, DTCs raised by the wear sensor itself. The synthetic fleet plants the first two; the detector catches them. | tested on synthetic |
| Sanity checks | Rate per vehicle-year against the fleet's own maintenance history (the synthetic world uses 1.05/yr); rate by OEM and vehicle age. | the study prints the measured rate with a Poisson interval |

## theft_event

| Item | Proposed definition | Status |
|---|---|---|
| Source of truth | Insurance claim or police report with a theft cause code, or the fleet's own stolen-vehicle report; OEM "vehicle reported stolen" status as a secondary source. | confirm: which of these Motorq can join on VIN |
| Event date | Last authorised ignition-off before the reported theft window, if telematics shows it; otherwise the report date. | confirm |
| Horizon | 3 to 7 days: the window in which a risk score can change parking, immobiliser or geofence behaviour. | product decision |
| Known at | Report date. | confirm |
| Exclusions | Recovered-within-hours false reports; theft of contents without the vehicle. | confirm |
| Leakage to watch | Location signals *after* the event (the synthetic fleet models a post-theft GPS blackout); "stolen" status flags; insurer-side fields. Anything with a value on days the vehicle is inactive must be reviewed. | contract allows location signals on inactive days for this reason |
| Sanity checks | Rate per vehicle-year against national fleet theft statistics for the vehicle classes involved; rate by parking region. | the study prints the measured rate |

## battery_degradation_event

| Item | Proposed definition | Status |
|---|---|---|
| Source of truth | State-of-health reading crossing a threshold (proposed: 80 % of rated capacity, the common warranty and second-life boundary) measured by an independent method (dealer diagnostic, OEM warranty claim), or a battery replacement. | confirm: which SoH source is independent of the daily telemetry |
| Event date | Date of the diagnostic or claim. | confirm |
| Horizon | 30 to 90 days: capacity loss is slow and the decision is about residual value and route assignment. | product decision |
| Known at | Diagnostic date. | confirm |
| Exclusions | Vehicles whose SoH telemetry *is* the label source (then the study is a smoothing exercise, not a prediction). | confirm |
| Leakage to watch | The daily SoH signal itself when the label is derived from it; warranty-claim flags; "battery service scheduled". | detector tiers 1 and 2 |
| Sanity checks | Degradation rate against published fleet EV studies for the chemistry and climate; rate by model year. | the study prints the measured rate |

## What the engine does with a label definition

1. `data_contract` checks the `EVENTS` frame for type, date range and per-vehicle
   plausibility and fails the study if it is violated.
2. `event_rate` measures the rate with an interval and compares it to nothing: the human
   compares it to the sanity-check source above.
3. `leakage_check` tests every candidate signal against the label for the four hard leak
   signatures and the one soft one; a label derived from a feature signal shows up as
   `near_perfect_auc` on that signal.
4. The label definition (source, event-date rule, horizon, exclusions) belongs in the study
   request text and is recorded with the spec, so the brief states which definition the
   verdict is about.

A verdict computed on a different label definition is a different study; the ledger keeps
both.

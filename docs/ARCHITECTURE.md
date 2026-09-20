# Architecture and methodology

Written for a technical reviewer asking: how does real data get in, what exactly is computed,
what is deterministic, how is the decision made, and what is still missing. Every section
names the code that implements it.

---

## 1. Data: where it comes from and how it gets in

### 1.1 Today: synthetic, by design

Everything validated so far runs on a synthetic fleet (`world/`). That is deliberate, not a
shortcut: the harness has to be proven to recover known structure before it is trusted on data
whose structure is unknown. The oracle tests (`tests/oracle/`) plant drivers, decoys, noise and
leaks, then assert the engine ranks the drivers first, rejects the decoys and noise, drops the
leaks, and blames the right OEM for a coverage gap. On 5,000 vehicles x 18 months the brake
sufficient set contains the planted sensor and no decoy, noise or GPS signal, and the
ablation is fully powered.

### 1.2 The canonical structure (the whole integration surface)

The engine reads three tables and nothing else. This is the contract (`data/contract.py`):

| Table | Grain | Required columns | Notes |
|---|---|---|---|
| `vehicles` | one row per vehicle | `vehicle_id, oem, model_year, powertrain` | `vehicle_id` is a salted hash, never a VIN; optional `segment, climate, duty` |
| `events` | one row per event | `vehicle_id, date, event_type` | the labels; `detail` optional |
| `signals_daily` | one row per vehicle-day | `vehicle_id, date, oem, model_year, active` + one numeric column per signal | `active=false` marks days with no telemetry (grid completion) |

`validate()` checks structure, key uniqueness at the stated grain, types, referential
integrity, future dates, history length and grid completeness, and returns errors and
warnings. Every study validates its source at the DEFINE stage and **fails closed** on an
error; the result is evidence (`data_contract`) and the first line of the brief's data
section. `mde data check` runs the same validation from the command line against whichever
source is configured.

Signal names are Motorq's public taxonomy (location/trips, health, driver behaviour,
fuel/energy, context). Each raw sensor signal also carries its **COVESA Vehicle Signal
Specification** path (`world/registry.py::VSS`, verified against the spec sources, Sept 2026):
`brake_pad_wear_pct` is `Vehicle.Chassis.Axle.Row1.Wheel.Left.Brake.PadWear`,
`battery_12v_voltage` is `Vehicle.LowVoltageBattery.CurrentVoltage`, and so on. VSS is the
industry's vendor-neutral signal model, so the mapping is how a Motorq normalised signal and
an OEM's native signal are shown to mean the same thing. Two consequences worth noting:

- The leakage signals map to `Vehicle.Service.IsServiceDue` and `Vehicle.Service.TimeToService`.
  VSS classifies them as service-scheduling state, i.e. known only once a service is planned,
  which is exactly why the quality engine must drop them.
- Derived scores, behaviour event counts and DTC *families* have no VSS path: they are
  Motorq-computed, and the registry says so.

### 1.3 Production pipeline, end to end

```
OEM APIs / push feeds
   -> Motorq ingestion + normalisation (canonical model; ~250B points/month)
   -> Snowflake via Snowpipe Streaming + Managed Iceberg (seconds of latency)
   -> dynamic tables shaped to the contract   docs/sql/01..03 (signal catalog, signals_daily, vehicles/events)
   -> SnowflakeSource (data/snowflake.py)     role-scoped read of the three tables; VIN -> salted hash;
                                              deterministic vehicle sample; grid completion
   -> canonical contract check                fail closed
   -> harness / economics / policy            in Motorq's account, see 1.4
   -> ledger (Postgres)                       evidence only: aggregates, statistics, no vehicle ids
   -> dashboard, Markdown brief, Slack webhook, portfolio job
```

The only Motorq-specific work is the three dynamic-table definitions and a `snowflake.yaml`
naming them (`docs/SNOWFLAKE.md`). The `SnowflakeSource` is tested against a
production-shaped fake (`tests/data/test_snowflake_source.py`); a live run needs
credentials and is the single step that turns the demonstration into a finding.

### 1.4 Running inside Motorq's boundary (the data never leaves)

The engine is two containers and a Postgres. The deployment that keeps the data inside is:

- **Snowpark Container Services**: run the API container in a compute pool in Motorq's own
  Snowflake account. The container reads the three tables over the internal connection;
  no data egress, Snowflake's IAM, private networking and secrets apply. This is the
  recommended path.
- **ECS/EKS in Motorq's VPC** with a private link to Snowflake: equivalent isolation, more
  operations work.

Either way: VINs are hashed before they reach the engine, the ledger stores aggregates and
statistics only (asserted by `test_no_vehicle_ids_leak_into_evidence_or_brief`), and the
LLM, when enabled, sees only the evidence digest. An LLM on Bedrock keeps even that inside
AWS. Details: `docs/DEPLOY.md`.

### 1.5 Scoping the data a study needs

A study does not read everything. Scope is set in this order:

1. `ProblemSpec.population_powertrain` (`auto` picks EV-only for battery targets, ICE-only for
   fuel targets, all vehicles otherwise) restricts vehicles.
2. Candidate signals are those compatible with that powertrain scope.
3. Quality filters drop sparse signals (< 5 % non-null) and leaks; drift, staleness, frozen
   and implausible readings are reported, not dropped.
4. Ablation reduces the candidates to the sufficient set; the economics are computed on
   that set, and the portfolio job reports which signals no viable capability needs.

---

## 2. How the synthetic data is generated (and what it is not)

`world/generator.py`, parameters in `world/params.yaml` (every value sourced or marked
`modeled: true`; the brief discloses the modelled ones).

- **Population**: 5,000 vehicles across 10 OEMs, segments, climates, duty classes and model
  years, with an OEM coverage matrix (`coverage.yaml`) saying which OEM emits which signal
  and at what cadence. Some OEMs lack the brake-wear sensor on purpose.
- **Latent state**: each vehicle carries hidden brake wear, battery state of health and theft
  exposure. Wear accrues with miles (FHWA duty-class mileage), harsh braking, climate and OEM
  multipliers; SoH declines at Geotab's measured 2.3 %/yr with DC-fast-charge and heat
  adders; theft exposure follows NICB segment rates (elevated 12x for learnability and
  disclosed).
- **Events** are drawn from hazard functions of the latent state (a logistic hazard that hits
  50 %/day at a per-vehicle service threshold), with a 0-12 day inspection-to-service lag, so
  the label is noisy the way a real service record is.
- **Signals** are noisy observations of the latent state and of usage: a driver signal sees
  the latent with sensor noise and a per-vehicle bias; a decoy is a noisy function of the
  *observed* driver (so it carries no extra information); noise signals are pure noise;
  leakage signals are computed with post-event knowledge.
- **Refinements made in this round**: energy signals clipped at zero (additive noise had
  produced negative kWh in 8 % of rows: the new plausibility check found it); generator
  version 0.2.1.

What it is not: a claim about the real world. It is a test bench with known answers.

---

## 3. Feature engineering and cleaning

`harness/frames.py`, `quality/checks.py`, `data/frame_source.py`.

**Label**: for vehicle *v* on day *t*, `y = 1` if the next event of the target type occurs in
`(t, t + h]` with *h* the horizon. Days whose label cannot be known are excluded (`y = -1`):
the final *h* days of history (the window has not closed) and inactive days (no telemetry).

**Features per signal** (4): the day's value, its 7-day mean, its 30-day mean, and the 30-day
delta (today minus 30 days ago). Missing values stay missing (LightGBM handles them natively;
no imputation invents a reading). Nothing is scaled: trees are invariant to monotone
transforms, and it keeps the brief readable in the signal's own units.

**Sampling**: vehicle-days are class-weighted so the rare positive class is represented; weights
are carried through every statistic (AUC, bootstrap, metrics), so nothing is computed on a
distorted population.

**Cleaning** is a reported pipeline, not a silent one. Per signal:

| Check | Statistic | Action |
|---|---|---|
| Sparsity | non-null rate | drop if < 5 % |
| Staleness | median inter-observation gap vs declared cadence | report |
| Drift | population stability index, first vs last quarter | report if > 0.20 |
| Frozen sensor | share of vehicles with an identical value on >= 14 consecutive active days, plateaus at a physical bound excluded | report if > 10 % |
| Implausible | share of readings outside the unit's physical range (`PLAUSIBLE_RANGE`) | report if > 1 % |
| Leakage | three tiers, below | drop / report |

**Leakage** is the check that matters most, because a leaked signal produces a spectacular
model that predicts nothing. Three tiers: (dropped) availability jump before the event,
monotone countdown to the event (|Spearman| >= 0.9 among positives), a near-binary flag
whose active value carries >= 15x the base positive rate, univariate AUC >= 0.99; (reported)
univariate AUC in [0.95, 0.99) for human confirmation that the signal is produced before,
not because of, the event.

**Redundancy** (`experiments.redundancy`): Spearman rank correlation between every pair of
screened signals on pairwise-complete rows; pairs with |rho| >= 0.9 are joined into
information groups. On the brake study 20 candidates carry far fewer independent groups:
odometer ~ engine hours (0.999), distance ~ fuel ~ CO2 ~ odometer delta, front pad wear ~ rear
pad wear ~ pad-thickness proxy. That is why ablation can drop signals without losing AUC, and
the brief shows the groups.

---

## 4. The model, and how signal relationships are measured

**Model**: LightGBM gradient-boosted trees, fixed hyper-parameters (200 trees, learning rate
0.05, 31 leaves, min 50 samples per leaf, 0.8 row and column subsampling, L2 = 1), seeded.
Chosen because it handles missing values natively, is invariant to feature scaling, captures
interactions, and is the strongest widely-deployed tabular baseline. A logistic regression is
fitted alongside as a linear reference, and a **one-signal LightGBM** is the baseline the
model gate compares against, so "beats a coin flip" is never enough.

**Cross-validation**: 5 folds, stratified by label and **grouped by vehicle**
(`StratifiedGroupKFold`), so a vehicle never appears in both train and test. Out-of-fold
predictions give every row a score from a model that did not see its vehicle.

**How relationships are measured** (each is evidence in the brief):

| Question | Method |
|---|---|
| Does signal *s* carry information about the target? | univariate AUC; mutual information (k-NN estimator) |
| Does the model *use* it, beyond the others? | permutation importance: AUC drop when *s*'s four columns are shuffled in the held-out fold, with a bootstrap CI |
| Is it *needed*? | ablation: refit without it and test non-inferiority (Section 5) |
| Is it a copy of something else? | Spearman redundancy groups |
| Does it transfer across OEMs? | leave-one-OEM-out plus the within-OEM benchmark (Section 5) |

---

## 5. The mathematics behind the decision

Notation: rows *i* with label *y_i*, score *s_i*, weight *w_i*, vehicle *g_i*.

**AUC** is the weighted probability that a random positive outranks a random negative, ties
counted one half:

AUC = sum over pos i, neg j of w_i w_j [ s_i > s_j ] + 1/2 [ s_i = s_j ] , divided by (sum w_pos)(sum w_neg)

`stats.AucSorter` sorts once and evaluates any weight vector in O(n).

**Uncertainty: cluster bootstrap by vehicle.** Rows of one vehicle are not independent, so
resampling rows would understate variance. Vehicles are resampled with replacement; that is
equivalent to giving vehicle *g* a multinomial count *c_g* and every row the weight
*w_i c_{g_i}*, so each of the 300 replicates is an O(n) weighted AUC on the same sorted
scores. Percentile intervals at 2.5 / 97.5 %. Verified equal to explicit resampling to 1e-9.

**Non-inferiority for ablation.** Removing a signal is accepted if

P_boot( AUC_reduced - AUC_full > -tau ) >= 0.80, tau = 0.005,

where the probability is over *paired* replicates (the same vehicle resample applied to both
score vectors, so vehicle-level noise cancels). Signals are removed cheapest-per-importance
first; a rejected removal is kept and the search continues over the remaining candidates.
`underpowered` is reported when an *accepted* removal's interval half-width exceeds tau/2: the
data could not resolve the tolerance and the removal rests on a point estimate. A wide
interval on a *rejected* removal is not a power problem (the signal is kept, conservatively).

**Model gate.** lower CI bound of AUC(sufficient set) minus AUC(best single signal on the same
rows) > 0. The bound, not the point estimate, has to clear the baseline.

**Temporal robustness.** Train on the earliest 70 % of days, skip *h* days, test on the rest;
`degradation = AUC_cv - AUC_forward` with its CI; the flag trips on the upper bound.

**Cross-OEM.** Leave-one-OEM-out AUC per OEM; the flag trips on std > 0.03 or a worst OEM more
than 0.05 below the mean. The within-OEM benchmark trains and tests on the OEM alone: if
within ~ held-out, the OEM *lacks the signal*; if within >> held-out, the model *does not
transfer*. The brief prints the diagnosis per OEM.

**Event-level operating point.** For alert rate *a* (top *a* of vehicle-days), an event is
caught if any alert fires in the *h* days before it; lead time is days from first alert to
event; a false-alert episode is a maximal run of consecutive alert days not followed by an
event within *h*, counted per 100 vehicle-months. These are the quantities a fleet manager
experiences, and the ones the economics use.

**Economics.** Inputs are ranges (low, base, high) drawn from a PERT distribution
(Beta(1 + 4(b-l)/(h-l), 1 + 4(h-b)/(h-l)) scaled to [l, h]); 10,000 draws. Per draw:

value/yr = fleet x event rate x recall(a) x preventable x value-bearing x $/avoided event
cost/yr  = false-alert episodes(a) x inspection cost + 12 x (run cost + amortised build)
ROI      = (value - cost) / cost

Reported: P(ROI > 0), ROI and net-value percentiles, the alert rate that maximises median
net value, and a tornado (median ROI at each input's low vs high, others at base). Event rate
is *measured* on the source with a Poisson interval, never assumed.

**Cost.** Volumes (rows x bytes x cadence) times unit prices from `price_sheet.yaml`; marginal
cost (what this capability adds) and fully attributed cost (including OEM packages) are both
shown, and placeholders are flagged until contracted rates replace them.

---

## 6. The decision policy: deterministic, and how "dynamic"

`policy/verdict.py::decide(spec, evidence) -> Verdict` is a **pure function**: the same evidence
and spec always give the same verdict, and it uses no data, no model and no randomness.
Thresholds live in `policy/policy.yaml`, are versioned, and the brief stamps the version.

Gates (any failure -> NOT_FEASIBLE), flags (any trip -> PILOT instead of BUILD_READY):

| Gate | Rule | Where the threshold comes from |
|---|---|---|
| data | fleet share with the full sufficient set >= `min_oem_coverage` | the request (spec constraint, default 0.6) |
| model | AUC lower bound minus best single-signal AUC > 0 | policy.yaml |
| economics | P(ROI > 0) >= 0.5 | policy.yaml |
| cost | marginal monthly run cost <= `max_run_cost_usd_month` | the request; the gate exists only when a ceiling is stated |
| delivery | a deployment pattern meets `max_latency_s` | the request |

Nine flags: cross-OEM variance, temporal degradation, ROI tail negative, value assumptions
unvalidated by a pilot, short signal history, ablation underpowered, suspicious signals
(leakage tier 2 or quality flags on the sufficient set), cost placeholders, alert burden.

**Are the gates dynamic?** In three ways, and deliberately not in a fourth:

1. The *statistics* compared to a threshold adapt to the data: intervals widen with fewer
   vehicles, the baseline is the best single signal *on this population*, the event rate is
   measured.
2. Request-level constraints parameterise the gates (coverage floor, cost ceiling, latency).
3. Thresholds are configuration, versioned and owned, so a change is a reviewable diff.
4. They are **not** tuned by the data or the model. A threshold that adapts to the evidence is
   a verdict that grades itself; the reproducibility contract requires the rule to be fixed
   before the evidence is seen.

Accuracy of the verdict is only as good as the evidence: the oracle tests are what show the
evidence is right on data with known answers, and the flags are how the policy says what it
does not know.

---

## 7. Determinism: what is guaranteed, and how it is kept

**Guaranteed identical** (same dataset hash, same spec, same seed, same numeric environment):
every number in every evidence record, the brief, and the verdict.

How:

| Source of variation | Control |
|---|---|
| Data | content-hashed dataset (`dataset_hash`), deterministic vehicle sampling from the hash |
| Splits | `StratifiedGroupKFold(shuffle=True, random_state=seed)` |
| Model | LightGBM `deterministic=true` + `force_row_wise=true`, seeded; per its documentation, results are then stable across thread counts (verified: identical at 1/3/4/8 threads, `tests/harness/test_models.py`) |
| Fold parallelism | folds are independent seeded fits; parallel and sequential outputs are bit-identical (tested) |
| Bootstrap | `numpy.random.default_rng(seed + step)`; sort-once evaluation has no order dependence |
| Monte Carlo | seeded generator, fixed 10,000 draws |
| Iteration order | signals sorted; dictionaries built in fixed order; evidence hashed on sorted JSON |
| Policy | pure function |

**Not guaranteed, and said so**: identical arithmetic across LightGBM/NumPy versions, compilers
or CPU architectures (LightGBM documents this). Therefore each run records a **runtime
fingerprint** (`harness/runtime.py`): Python, platform, library versions, the LightGBM
parameter hash, the code version. `replay` compares fingerprints: identical fingerprint and
different numbers is a defect; different fingerprint is an expected difference and the diff
says so. In production the container image pins every version (`uv.lock`), so the fingerprint
only changes when the image does.

**Maintaining it**: `test_headless_is_deterministic` and `test_replay_is_identical` run in the
suite before every push; the runtime line is printed in every brief; `mde run replay` is the
operational check after any upgrade.

---

## 8. What is still needed

1. A run against Motorq's Snowflake through `SnowflakeSource` (the adapter and its contract
   validation are ready; credentials are not).
2. Contracted unit prices (`price_sheet.yaml`) and product-owned value ranges.
3. A pilot to validate predicted value against realised value (clears `value_unvalidated`).
4. Deployment in Snowpark Container Services with Motorq's identity layer in front.

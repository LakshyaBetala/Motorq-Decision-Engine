# Motorq Decision Engine — Design Spec

Date: 2026-09-18
Status: approved for implementation

## 1. Purpose

A deterministic, reproducible harness that, for a candidate Fuse capability, produces a fully cited evidence brief covering: data feasibility (signal existence, OEM coverage, quality), minimal sufficient signal set with confidence intervals, robustness (temporal, cross-OEM), run cost of that signal set, ROI distribution under explicit value ranges, and a computed verdict (BUILD-READY / PILOT / NOT-FEASIBLE). An LLM layer parses free text into a spec and answers questions over evidence; the pipeline runs headless without it.

Non-goals: making the build decision; proposing infrastructure; estimating business value from data; replacing Fuse.

## 2. Repository layout

```
motorq_de/                      Python package (src layout: src/motorq_de)
  world/        generator.py, registry.py, coverage.py, truth.py
  data/         source.py (Protocol), synthetic.py, snowflake.py (stub)
  quality/      coverage.py, freshness.py, missingness.py, drift.py, leakage.py
  harness/      frames.py, importance.py, ablation.py, temporal.py, cross_oem.py, compare.py, stats.py
  economics/    price_sheet.yaml, volume.py, cost.py, value.py, roi.py, sensitivity.py, deployment.py
  policy/       gates.py, flags.py, verdict.py
  ledger/       models.py, store.py, hashing.py
  report/       brief.py, render.py
  agent/        spec_parser.py, plan.py, machine.py, tools.py, llm.py, qa.py
  cli.py
api/            FastAPI app
web/            Next.js app
tests/          mirrors package; tests/oracle for harness-vs-truth
docs/
```

Dependency rule (enforced by an import-linter test): `world → data → quality → harness → economics → policy → ledger → report → agent → api`. No upward imports. `agent/` is the only module allowed to import `anthropic`.

## 3. Core schemas (Pydantic v2, frozen)

### ProblemSpec
```
capability_name: str
target_event: Literal["brake_service_event", "theft_event", "battery_degradation_event"]
horizon_days: int  (1..90)
decision_unit: Literal["vehicle_day"]
delivery_mode: Literal["batch_daily", "batch_hourly", "streaming"]
consumer: Literal["fuse_action_hub", "fuse_assistant", "api", "internal"]
value: ValueAssumptions
constraints: Constraints
seed: int = 42
```
ValueAssumptions — each a `Range(low, base, high)`, all required:
`events_per_vehicle_year`, `preventable_fraction`, `usd_per_avoided_event`, `fleet_size`.
Constraints: `min_oem_coverage: float = 0.6`, `max_latency_s: int | None`, `max_run_cost_usd_month: float | None`, `min_signal_history_months: int = 6`.

### Evidence
```
evidence_id: str        "ev_" + first 12 hex of sha256(run_id + tool + inputs_hash)
run_id: str
step: str
tool: str
inputs: dict            JSON-serialisable
inputs_hash: str        sha256 of canonical JSON
dataset_hash: str
seed: int
outputs: dict           JSON-serialisable; all floats rounded to 6 dp before hashing
created_at: datetime
```
Determinism test: two runs with identical (dataset_hash, inputs_hash, seed) produce identical `outputs`.

### Verdict
```
decision: Literal["BUILD_READY", "PILOT", "NOT_FEASIBLE"]
gates: list[GateResult(name, passed, value, threshold, evidence_ids)]
flags: list[FlagResult(name, tripped, value, threshold, evidence_ids)]
policy_version: str
```

## 4. Synthetic world

### Population
- 10 OEM groups (`oem_a`..`oem_j`), 10,000 vehicles, model years 2019–2025, 24 months of daily data.
- Vehicle attributes: oem, model_year, segment (sedan/suv/van/truck/ev), climate_zone (hot/temperate/cold), duty (light/medium/heavy), powertrain (ice/ev). EVs only in segments where the OEM ships EVs.

### Latent states (per vehicle-day, hidden)
`brake_wear, battery_soh, engine_wear, tire_wear, fuel_system_wear` in [0,1], plus `theft_risk_context` (derived from dwell location entropy, night parking share, region base rate).

Dynamics: each wear state increments daily by `base_rate[oem, segment] × duty_mult × climate_mult × noise`. Brake wear increments additionally with `harsh_brake_count`. Battery SoH degrades with `dc_fast_charge_share` and hot climate. Maintenance resets the relevant state.

### Events (hazard-driven)
- `brake_service_event`: daily hazard = sigmoid(k·(brake_wear − θ_oem)). Label for a vehicle-day = event within next `horizon_days`.
- `battery_degradation_event`: hazard on battery_soh below threshold with hot-climate multiplier; EV only.
- `theft_event`: hazard = base_rate × f(theft_risk_context) × g(vehicle segment); independent of wear states.
Target base rates chosen so positive rate at 7-day horizon is ~1–3% (class imbalance is deliberate).

### Observable signals (~90)
Grouped as in Motorq's public taxonomy: location/trips, health, driver behavior, fuel/energy, plus derived aggregates. Each signal is `view(latent or behaviour) + oem_bias + noise`, sampled at a signal-specific frequency, with signal- and OEM-specific missingness.

Planted structure (ground truth saved to `truth.json`):
- Drivers for brake: `brake_pad_wear_pct` (direct, only some OEMs emit), `harsh_brake_count`, `odometer_delta`, `avg_speed`, `dtc_brake_family`, `duty`.
- Drivers for battery: `soc_min_daily`, `dc_fast_charge_share`, `cabin_temp_max`, `battery_temp_max`, `charge_cycles`.
- Drivers for theft: `night_park_share`, `dwell_location_entropy`, `region_theft_index`, `ignition_off_duration`, `gps_*`.
- Decoys: 10 signals correlated with drivers but with no causal path (must be rejected by ablation), 5 pure-noise signals, 2 leakage traps (`service_appointment_scheduled`, `days_since_last_service` computed post-event) that quality/leakage checks must flag.

### Coverage matrix
`coverage.yaml`: for each (oem, signal) → `{emits: bool, from_model_year: int, frequency: str, missing_rate: float}`. Deliberate gaps: oem_c and oem_g do not emit `brake_pad_wear_pct`; oem_e emits odometer weekly; EV signals only on EV OEMs.

### Storage
Parquet under `data/synthetic/<dataset_hash>/`: `vehicles.parquet`, `signals_daily.parquet` (long → wide per vehicle-day), `events.parquet`, `truth.json`, `coverage.yaml`. `dataset_hash` = sha256 of generator config + seed + generator version.

## 5. DataSource protocol

```python
class DataSource(Protocol):
    dataset_hash: str
    def list_signals(self) -> list[SignalMeta]
    def signal_metadata(self, signal_id: str) -> SignalMeta
    def coverage_by_oem(self, signal_id: str) -> dict[str, CoverageStat]
    def quality(self, signal_id: str, window: DateWindow) -> QualityStat
    def training_frame(self, spec: ProblemSpec) -> TrainingFrame  # X, y, groups(oem), time(date), vehicle_id
```
`SyntheticSource` reads parquet. `SnowflakeSource` raises `NotImplementedError` with the SQL shape documented in docstrings against assumed tables `SIGNALS(vin, ts, signal_id, value)`, `VEHICLES(vin, oem, model_year, ...)`, `EVENTS(vin, ts, event_type)`.

## 6. Quality engine

Per signal (and per spec): coverage by OEM × model year; freshness (median gap vs. declared frequency); missingness by OEM; drift (PSI between first and last quarter); leakage check (signal availability aligned to event time; flags signals whose non-null rate jumps in the horizon window before the event). Outputs are Evidence.

## 7. Experiment harness

Label construction: for each vehicle-day t, `y = 1` if target event in (t, t+horizon]. Features are the signal values at day t plus rolling 7/30-day aggregates defined in `frames.py`. Rows after a vehicle's first event of that type within the horizon are excluded to avoid trivial leakage.

Model: LightGBM binary with fixed params; logistic regression as second model; baselines: majority, best single signal.

Splits:
- `repeated_stratified_cv`: 5 folds × 3 repeats, grouped by vehicle_id.
- `temporal`: train months 1–18, test 19–24; also rolling 3-month forward windows.
- `leave_one_oem_out`: 10 folds.

Metrics: ROC-AUC, PR-AUC, Brier, recall@precision≥0.5. Every metric reported with a 95% bootstrap CI (1000 resamples, fixed seed).

Tools (each returns Evidence):
- `feature_analysis(spec, candidate_signals)`: permutation importance with CIs, mutual information; returns ranked list.
- `ablation(spec, signals)`: greedy backward elimination. At each step remove the signal whose removal has the smallest ΔAUC; stop when the paired-bootstrap 95% CI of ΔAUC (full vs reduced) excludes a drop > 0.005. Returns the sufficient set, the elimination trace, and per-signal ΔAUC with CIs.
- `temporal_validation(spec, signals)`: forward split metrics and degradation vs CV.
- `cross_oem_validation(spec, signals)`: per-held-out-OEM AUC, mean, variance, min.
- `model_comparison(spec, signal_sets)`: metrics for each set × model, paired.

Oracle tests (`tests/oracle`): on the synthetic dataset, (a) the sufficient set for brake contains ≥4 of the 6 planted drivers and 0 decoys; (b) leakage traps are flagged by quality and excluded; (c) GPS signals are not in the brake sufficient set but at least two `gps_*`/dwell signals are in the theft sufficient set; (d) for oem_c held out, brake AUC drops by ≥ 0.03 relative to the mean (the planted coverage gap is detectable).

## 8. Economics

`price_sheet.yaml` — unit prices with `source` notes and `as_of` date. Categories: `oem_data_per_vehicle_month_by_package`, `ingest_per_million_events`, `kafka_per_gb`, `snowflake_storage_per_tb_month`, `snowflake_compute_per_credit`, `inference_per_1k`, `engineer_hour`. Values are placeholders marked as such; the point is auditability.

`volume.py`: from signal set × fleet_size × frequency → events/day, bytes/day, feature rows/day, inference calls/day.

`cost.py`: run cost/month = Σ category volumes × unit prices; also reports cost of the full signal set for the same spec so savings are explicit.

`value.py` + `roi.py`: value/year = fleet_size × events_per_vehicle_year × detection_rate (recall at operating point from the harness) × preventable_fraction × usd_per_avoided_event. Each Range is sampled as a PERT (beta) distribution; 10,000 draws with fixed seed → ROI distribution, `P(ROI > 0)`, 5th/50th/95th percentiles.

`sensitivity.py`: one-at-a-time low/high swings per assumption → tornado ordering.

`deployment.py`: rubric — horizon ≥ 1 day and no latency constraint → `batch_daily on Snowflake`; horizon < 1 day or latency ≤ 60 s → `streaming on Kafka`; between → `batch_hourly`. Reports fit and whether spec constraints are met.

## 9. Decision policy (policy_version "1.0")

Hard gates:
- `data`: all signals in the sufficient set exist AND fleet share with full-set coverage ≥ `min_oem_coverage`.
- `model`: lower bound of 95% CI of (AUC_model − AUC_best_baseline) > 0.
- `economics`: P(ROI > 0) ≥ 0.5.
- `delivery`: deployment rubric returns a pattern meeting constraints.

Flags:
- `cross_oem_variance`: std of leave-one-OEM-out AUC > 0.03 OR min AUC < mean − 0.05.
- `temporal_degradation`: forward-split AUC lower than CV AUC by > 0.03 (CI upper bound).
- `roi_spans_negative`: 5th percentile ROI < 0.
- `value_unvalidated`: always true unless spec marks `value.validated_by_pilot = True`.
- `short_history`: any sufficient-set signal has < `min_signal_history_months` months of data.

Verdict: any gate fails → NOT_FEASIBLE; else any flag → PILOT; else BUILD_READY.

## 10. Ledger

SQLAlchemy models: `runs(run_id, spec_json, dataset_hash, status, created_at, verdict_json)`, `steps(step_id, run_id, name, status, started_at, ended_at, replan_reason)`, `tool_calls(call_id, step_id, tool, inputs_json, inputs_hash, evidence_id, duration_ms)`, `evidence(evidence_id, run_id, tool, inputs_hash, dataset_hash, seed, outputs_json, created_at)`, `artifacts(artifact_id, run_id, kind, path)`, `messages(run_id, role, content, evidence_ids)`.
Postgres via `DATABASE_URL`; falls back to SQLite file when unset.

## 11. Report

`brief.py` builds a typed `Brief` (sections: spec, data, model, robustness, economics, delivery, verdict, follow-ups). Every numeric field is a `Cited(value, evidence_ids)`. `render.py` renders Markdown and JSON; it raises `UncitedClaimError` if any numeric field has empty evidence_ids. Narrative strings from the LLM are scanned for numbers; a sentence containing a number without an adjacent `[ev_…]` citation is dropped and logged.

## 12. Agent

State machine states: `DEFINE → FEASIBILITY → EXPERIMENT → ECONOMICS → DELIVERY → POLICY → REPORT → DONE`, plus `REPLAN` reachable from EXPERIMENT and ECONOMICS. Max 3 replans.

Default plan (headless): quality → feature_analysis → ablation → temporal → cross_oem → cost → roi → sensitivity → deployment → policy → report.

Replan predicates (code, not LLM): cross-OEM flag → per-OEM ablation for the worst OEM; cost rank vs AUC rank disagreement between candidate signal sets → model_comparison on the cheaper set; ROI P(>0) in [0.45, 0.55] → sensitivity on the top-2 assumptions with widened ranges.

LLM (Anthropic SDK, tool use): used in `spec_parser` (free text → ProblemSpec, validated; on validation failure, one repair attempt then error), in `plan` (may add optional tools from an allowlist; may not remove required ones), in narrative (cited), and in `qa` (answers from evidence store only; tool `search_evidence(run_id, query)`). Model id read from config; default per the claude-api skill at implementation time.

`--no-llm`: skips spec_parser (requires structured spec), uses default plan, narrative is template-generated.

## 13. API

`POST /runs` (body: ProblemSpec or `{text}`), `GET /runs/{id}` (status, steps, verdict), `GET /runs/{id}/brief` (markdown + json), `GET /runs/{id}/evidence/{evidence_id}`, `POST /runs/{id}/ask` ({question} → answer with citations), `GET /datasets` (hashes, generator config). Runs execute in a background task; status polled.

## 14. Web

Next.js app: run list; run page with step timeline, verdict card (gates/flags), brief sections with click-through citations to evidence JSON, ablation trace chart, cross-OEM bar chart, ROI histogram + tornado, ask box.

## 15. Testing

- Unit: schemas, hashing, price sheet math, policy at every boundary (parametrised), renderer fail-closed.
- Determinism: run harness twice on same dataset/seed → identical evidence outputs.
- Oracle: §7 tests.
- Integration: headless run on a small generator config (500 vehicles, 6 months) completes and produces a brief with zero uncited claims.
- Import-linter: dependency direction.

## 16. Out of scope for this cycle

Real Snowflake connection; auth on the API; multi-user; scheduled portfolio-wide COGS runs (design allows it; not built).

## 17. Revisions made during implementation (2026-09-19)

Each of these was forced by a result from the first runs, not by preference. The code and
tests reflect the revised design; the sections above are kept as the original contract.

| Area | Original | Revised | Why |
|---|---|---|---|
| Value assumptions (§3) | `events_per_vehicle_year` supplied by humans | Event rate is **measured** from the data with a Poisson CI (`quality.event_rate`); humans supply `value_bearing_fraction` instead | One fewer invented number; makes the target/value mismatch explicit |
| Population (§3) | implicit | `ProblemSpec.population_powertrain` (`auto` scopes EV-only targets to EVs) | ICE-only signals trivially identified EVs and dominated the battery ranking |
| Decoys (§4) | noisy copy of the latent for all OEMs | noisy function of the **observed** driver, NaN where the driver is NaN | A vendor signal built from the hidden state beat the real sensor because it had no OEM gaps |
| Brake realism (§4) | fixed service threshold | per-vehicle service threshold and sensor bias; 0-12 day service lag | Univariate AUC of the sensor was 0.98, indistinguishable from leakage |
| Leakage (§6) | AUC >= 0.95 dropped | Three-tier: hard signatures (availability jump, monotone-to-event, flag lift, AUC >= 0.99) dropped; AUC in [0.95, 0.99) kept and reported as *suspicious*; policy flag `suspicious_signals` | Data alone cannot distinguish a perfect sensor from a leak |
| Ablation rule (§7) | CI lower bound > -tolerance | Accept if >= 80% of paired cluster-bootstrap replicates exceed -tolerance; report `resolution`; flag `ablation_underpowered` when half-width > tolerance/2; importance screen to top 20 before elimination | Strict CI rule was a coin flip at low power; reporting power is more honest than silently failing removals |
| Baseline (§7) | univariate AUC of best signal | one-signal LightGBM OOF on the same population (top 3 by univariate AUC) | Univariate AUC on available rows flattered gappy sensors and failed the model gate unfairly |
| Economics (§8) | infra + OEM packages | + per-call OEM pricing driven by the highest polling cadence; explicit marginal vs attributed views; net-value-optimal alert rate | At list prices infra is pennies; polling cadence and build effort are the levers |
| Policy (§9) | 5 flags | 8 flags (+ `ablation_underpowered`, `suspicious_signals`, `cost_placeholders`) | Each corresponds to an honesty requirement discovered in testing |
| Data layer (§5) | SyntheticSource; SnowflakeSource stub | `FrameSource` base shared by `SyntheticSource` and a working `SnowflakeSource` (configurable tables, inferred coverage, deterministic VIN sampling, grid completion), tested with a production-shaped fake | The seam had to be real code, not a docstring |
| Layout (§2) | `api/` top-level | `motorq_de/api.py` inside the package; `agent/service.py` facade | `mde serve` works from any directory |
| Dependencies | pandas unpinned | `pandas>=2.2,<3` | Snowflake connector supports pandas 2.x |

## 18. Strengthening round (2026-09-19, later)

| Area | Change | Why |
|---|---|---|
| Metrics | Event-level metrics (events caught, median lead days, false-alert episodes per 100 vehicle-months) computed from vehicle-day scores keyed on (vehicle, event date); an alert-rate grid of 10 points | Vehicle-day recall is not what a fleet manager experiences; the old false-alert proxy over-counted several-fold |
| Economics | ROI uses event recall and the false-alert rate directly; `alert_burden` flag; `value_assumptions.yaml` owned by product | Human inputs belong in a reviewed file, not code |
| Ablation | Cost-aware order (importance per marginal dollar); daily-cadence ablation variant; a rejected signal is kept and elimination continues | Finds the cheapest non-inferior set; exposes the polling-cadence lever; cheap noise cannot hide behind a necessary expensive signal |
| Cross-OEM | Within-OEM benchmark and a `diagnosis` per OEM (`oem_lacks_signal` / `does_not_transfer` / `ok`) | Separates missing signal from model transfer failure |
| Policy | `policy.yaml` v1.1, thresholds loaded at import; `alert_burden` flag | Threshold changes are reviewed config, stamped in the brief |
| Ledger | `evidence.name`, `runs.kind` / `derived_from`, additive migration; `evidence_objects()` | What-if and replay reconstruct the runner's evidence map |
| Runner | `whatif()` (ECONOMICS -> POLICY -> REPORT on a parent's harness evidence, seconds); `replay()` with evidence-by-evidence diff; stages as methods | The roadmap review argues about assumptions, not experiments |
| Portfolio | `agent/portfolio.py`: latest study per capability, sorted; COGS report of signals and cadences no viable capability needs | The promised second half of the pitch |
| API / CLI | `/runs/{id}/whatif`, `/replay`, `/brief.md`, `/events` (SSE), `/portfolio`; `mde run replay|whatif`, `mde portfolio` | |
| Web | Live progress via SSE through a runtime proxy route; portfolio page; coverage heatmap; ablation table; operating-point curve; what-if panel; export; replay | |
| Integration | Slack-compatible webhook on run completion; Bedrock provider switch; SQL templates for `SIGNAL_CATALOG`, `SIGNALS_DAILY` (dynamic table), `VEHICLES`, `EVENTS`; `docs/DEPLOY.md` (Access/IAP, compose, owned inputs, scheduled portfolio job); privacy test | |
| Performance | Cluster bootstrap re-implemented as sort-once + multinomial vehicle weights (`AucSorter`), exact to 1e-9 vs explicit resampling; quality reads batched 16 signals per parquet read | 300 replicates on 150k rows: ~30 s -> 1.4 s; a 12-signal ablation on the 5k-vehicle dataset: ~45 min -> 3.5 min |
| Performance (folds) | Cross-validation folds run in parallel worker processes (`run_folds`), threads x workers bounded by `MDE_CPUS`; LightGBM deterministic mode verified bit-identical across thread counts | Continue-past-rejection ablation does ~5x more fits than stop-at-first-rejection; on 16 cores OOF predictions drop 11.2 s -> 4.8 s per 5-fold set with identical output |
| Canonical contract | Source tables were trusted | `data/contract.py` validates tables, grain, types, integrity and dates; every DEFINE stage fails closed on an error; `mde data check` | The integration surface with Motorq is three tables; the contract makes that explicit and testable |
| VSS mapping | Registry ids only | Each raw sensor signal carries its COVESA VSS path, verified against the spec sources | Vendor-neutral canonical naming; the leakage signals turn out to be VSS `Service.*` scheduling state |
| Quality checks | sparsity, staleness, drift | + frozen-sensor persistence test (bound plateaus excluded) and unit-range plausibility | Standard telematics QA; found negative kWh in the generator (fixed, 0.2.1) |
| Runtime fingerprint | none | Versions, platform, LightGBM parameter hash and code version recorded per run; replay reports `environment_identical` | LightGBM guarantees determinism within an environment, not across versions |
| Redundancy | none | Spearman groups among screened candidates, in the brief | Explains why ablation can drop signals without losing AUC |
| Cost gate | `max_run_cost_usd_month` accepted, unused | Gate when a ceiling is stated (policy 1.2) | A stated constraint must bind |
| Fit cache | Every study refit everything; 34 of 60 OOF fit sets in a study were duplicates | `harness/fitcache.py` keyed by row identity, ordered signals, model, folds, seed, numeric environment; replay bypasses | 166 s -> 101 s on the fixture, evidence identical |
| Learning curve | none | AUC on 25/50/75/100 % of vehicles; `data_still_improving` flag (policy 1.3) | Answers "was that much data needed" with evidence |
| `underpowered` semantics | Any step whose bootstrap half-width exceeded tolerance/2 | Only an **accepted** removal can be underpowered; a wide interval on a rejected removal (dropping the main sensor) is a clear decision. Rejections whose interval still reaches the non-inferior region are listed as `kept_conservatively` | Testing every candidate surfaced the flaw: removing `brake_pad_wear_pct` tripped the flag on a step that was never in doubt |

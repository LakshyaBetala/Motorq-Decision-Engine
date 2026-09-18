# Motorq Decision Engine

**A reproducible capability-feasibility and signal-economics harness for the Fuse roadmap.**

Given a candidate intelligence capability ("predict brake service 7 days out"), the engine computes — deterministically, with confidence intervals, and with every number traceable — whether the data supports it, on which OEMs, how robustly, what the required signal set costs to run, and what the ROI distribution looks like under explicit value assumptions. It produces the evidence brief a product decision needs in about an hour instead of three weeks. It does not make the decision.

---

## 1. Where Motorq is, and the question this answers

Motorq has solved access. The platform ingests roughly 250 billion data points a month through direct integrations across 25+ brands and 12+ OEM partners, normalizes them, and delivers them through the Fleet Portal, REST APIs, Kafka, and Snowflake (now on Snowpipe Streaming + Managed Iceberg, seconds of latency). In April 2026 Motorq launched **Fuse** — *Action Hub* ranks fleet issues by cost impact; *Assistant* answers natural-language questions and generates reports. The stated direction from leadership is to "build the AI layer on top: automated decisions, predictive workflows, systems our customers run every day."

That direction produces a recurring, concrete question. Every new Fuse action — a new detector, a new predictor, a new ranked recommendation — is a small build decision:

> Does the signal exist? On which OEMs and model years? At what frequency and quality? Is there real predictive lift over what we already show? What does it cost to ingest, store, and serve the signals it needs? Does the customer value clear that cost?

Today that is answered by a data scientist's notebook, a coverage spreadsheet, a PM's prioritization sheet, and a roadmap meeting. It takes weeks per candidate, the evidence is not reusable, and the cost side is usually hand-waved.

There is a second question underneath it that is about gross margin, not roadmap. OEM data is purchased per vehicle and often per data package; every signal pulled is then ingested, normalized, streamed, and stored. At Motorq's volume, **which signals at which frequency are actually earning their keep across the product portfolio** is a COGS question. Nobody answers it systematically because the tooling to do so doesn't exist.

The Decision Engine is that tooling.

---

## 2. What this is not

Stated up front so the scope is honest.

- **Not a decision-maker.** Build decisions depend on OEM contract terms, sales pipeline, competitive timing, and team capacity. None of that is in vehicle data. The engine produces the *evidence brief*; people decide.
- **Not a Fuse alternative or another fleet dashboard.** Fuse answers "what should the fleet do." This answers "should Motorq build the next Fuse capability, and what will it cost." It sits one layer upstream, in service of Fuse.
- **Not an architecture generator.** Motorq has an architecture. The engine scores a capability's *deployment fit* (batch on Snowflake vs. streaming on Kafka, by horizon and latency) with a rubric. It does not propose stacks.
- **Not a value oracle.** Dollar-per-event assumptions come from humans — ideally from the same models Fuse Action Hub already uses to rank by cost impact. The engine takes them as ranges, propagates the uncertainty, and shows which assumption the verdict hinges on.
- **Not an LLM product.** The LLM is an interface: free text → structured problem spec, and Q&A over evidence. The pipeline runs headless without it, and every reliability guarantee holds with the LLM turned off.

---

## 3. Core insight: signal value is decision-dependent — and at Motorq that's money

For a given prediction task, only a subset of available signals carries meaningful incremental value, and the subset differs per task. This is a hypothesis to test per capability, not a claim about Motorq's data in general.

```
Candidate signals for a 7-day brake-service predictor

  ~90 available    --quality/coverage filter-->  ~60 usable
                   --importance + CIs--------->  ~18 candidates
                   --ablation, paired boot---->    7 sufficient
```

A signal excluded from one model is not globally useless:

```
GPS / trip data
   brake-service prediction   -> low incremental value
   stolen-vehicle recovery    -> high
   utilization scoring        -> high
```

The output is never "stop pulling GPS." It is "this capability does not need GPS at trip-level frequency, so its run cost excludes it" — and, across the portfolio, "these signals are required by no shipped capability at their current frequency." The first informs a roadmap decision; the second informs a COGS conversation with the data-integrations team.

---

## 4. How it works

A deterministic pipeline with an optional LLM front end.

```
  capability request (free text or structured spec)
            |
            v
   +--------------------+
   |  ProblemSpec        |  target event . horizon . decision unit . delivery mode
   |  (validated schema) |  value assumptions as [low, base, high] . constraints
   +---------+----------+
             v
   +--------------------+   coverage by OEM x model year . freshness . missingness
   |  Data feasibility   |   frequency . drift . leakage traps
   +---------+----------+
             v
   +--------------------+   importance w/ bootstrap CIs . greedy ablation with
   |  Experiment harness |   paired-bootstrap dAUC . temporal split . leave-one-
   |                     |   OEM-out . baselines (majority, best single signal)
   +---------+----------+
             v
   +--------------------+   volume model from signal set x fleet x frequency
   |  Economics          |   price sheet (cited unit prices) -> run cost
   |                     |   value ranges -> Monte Carlo -> P(ROI>0), tornado
   +---------+----------+
             v
   +--------------------+   pure function of evidence: hard gates + uncertainty
   |  Decision policy    |   flags -> BUILD-READY / PILOT / NOT-FEASIBLE
   +---------+----------+
             v
   +--------------------+   every number cites an evidence_id
   |  Evidence brief     |   renderer refuses uncited claims
   +--------------------+
```

**The LLM's role, precisely:** parse a free-text request into a `ProblemSpec`; choose which optional experiments to add when predicates fire (e.g. cross-OEM variance above threshold → per-OEM ablation); write narrative that must cite evidence IDs or be dropped; answer follow-ups ("why was GPS excluded?") from the evidence store. It never emits a number that did not come from a tool. It cannot change the verdict — it can only request more evidence.

---

## 5. Decision policy

The verdict is computed, versioned, and unit-tested — not generated.

```
HARD GATES         any failure -> NOT-FEASIBLE
  data       required signals exist AND OEM coverage of the full set >= spec threshold
  model      lift over best baseline, CI lower bound > 0
  economics  P(ROI > 0) >= 0.5 under supplied value ranges
  delivery   at least one deployment pattern meets horizon/latency constraints

UNCERTAINTY FLAGS  any trip -> PILOT instead of BUILD-READY
  cross-OEM AUC variance above tau
  temporal degradation CI includes a material drop
  ROI 90% interval spans negative
  value assumptions not yet validated against pilot outcomes
  required signal set includes signals with < N months of history

BUILD-READY only if every gate passes with margin and no flag trips.
```

"BUILD-READY" means *the evidence supports building*. Whether to build is still a human call.

---

## 6. Reliability contract

These are tested invariants, not aspirations.

| Invariant | What it guarantees |
|---|---|
| **Determinism** | Same `dataset_hash + inputs_hash + seed` → byte-identical tool output. |
| **Provenance** | Every claim in the brief references an `evidence_id`; the renderer fails closed on uncited numbers. |
| **Statistical honesty** | No point estimates without intervals; ablation uses paired bootstrap; cross-OEM claims require leave-one-OEM-out; temporal claims require forward splits. |
| **Oracle validation** | On synthetic data with known structure, the harness must recover planted drivers and reject planted decoys. This is the harness's test suite. |
| **Policy purity** | The decision policy is a pure function with tests at every gate boundary. |
| **Replay** | Any past brief re-runs from its stored spec and dataset hash and can be diffed. |
| **Headless** | `--no-llm` runs the default plan end to end and produces the same evidence and verdict. |

---

## 7. Data: synthetic now, Snowflake next — and what each proves

The MVP runs on a synthetic world, and it is important to be exact about what that does and does not show.

**The synthetic world is not random columns.** Each vehicle carries hidden wear states (brakes, battery, engine, tires, fuel system) that evolve with usage, climate, and OEM-specific reliability. Observable signals are noisy, OEM-dependent *views* of those states — a coverage matrix decides which OEMs emit which signals, at what frequency, with what missingness, from which model year. Events are hazard-driven from the latent states: maintenance events from wear; theft from location and dwell patterns (so GPS genuinely earns its recovery value); utilization from trips. Ground truth is saved alongside the data and is the test oracle for the harness.

**What synthetic proves:** the harness recovers known structure, the policy behaves at its boundaries, the pipeline is deterministic and replayable, the brief is fully cited. **What it does not prove:** anything about Motorq's actual signals. "GPS has low value for brake prediction" on synthetic data is a property of the generator, not of the fleet.

**The path to real value is a single seam.** All tools consume a `DataSource` protocol:

```
list_signals()                     signal_metadata(id)
coverage_by_oem(signal)            quality(signal, window)
training_frame(spec) -> X, y, groups=oem, time
```

`SyntheticSource` implements it fully. `SnowflakeSource` is a typed stub with the SQL shape sketched against normalized per-VIN signal tables. Pointing the harness at Motorq's Iceberg tables is an adapter, not a rewrite. Until that adapter exists, this is a demonstration of a machine, not a finding about Motorq.

---

## 8. Architecture

Dependencies point strictly downward. The LLM lives only in `agent/`.

```
web/        Next.js - reasoning trail, evidence drill-down, brief
api/        FastAPI - POST /runs . GET /runs/{id} . POST /runs/{id}/ask
agent/      ProblemSpec parser . bounded state machine . LLM tool loop . Q&A
policy/     gates + flags -> verdict (pure)          report/  cited renderer
harness/    importance . ablation . temporal . cross-OEM . comparison
economics/  price sheet . volume model . value ranges . Monte Carlo . sensitivity
quality/    coverage . freshness . missingness . drift . leakage checks
ledger/     runs . steps . tool_calls . evidence . artifacts   (Postgres)
data/       DataSource protocol -> SyntheticSource | SnowflakeSource
world/      latent-degradation generator . signal registry . coverage matrix
```

**Stack:** Python 3.12, FastAPI, SQLAlchemy, PostgreSQL (SQLite fallback for local), scikit-learn + LightGBM, Anthropic SDK for the agent layer, Next.js + Tailwind for the trail UI, pytest for the reliability contract.

---

## 9. Worked example

**Request:** *"Should Fuse flag vehicles likely to need brake service in the next 7 days?"*

**Brief (abridged; every line cites evidence):**

| Section | Finding |
|---|---|
| Spec | target = brake_service_event, horizon = 7d, unit = vehicle-day, delivery = batch daily |
| Data | 7-signal sufficient set exists; full-set coverage 78% of fleet (OEM A/B/D 2022+; OEM C lacks brake-pad wear) `[ev_014, ev_019]` |
| Model | AUC 0.86 [0.84, 0.88] vs best single-signal baseline 0.71; lift CI excludes 0 `[ev_031]` |
| Robustness | Leave-one-OEM-out AUC range 0.79–0.88; OEM C held out drops to 0.79 — **flag** `[ev_037]`. Forward 3-month split: −0.01, not material `[ev_040]` |
| Economics | 7-signal run cost $X/mo vs 60-signal $Y/mo (−71%) `[ev_052]`. P(ROI>0) = 0.83; 90% interval spans negative at low preventable-fraction — **flag** `[ev_058]`. Verdict is most sensitive to $/avoided-event `[ev_059]` |
| Delivery | Batch daily on Snowflake fits horizon; streaming unnecessary `[ev_061]` |
| **Verdict** | **PILOT** — gates pass; two uncertainty flags (OEM C generalization, value-assumption sensitivity). Recommend a pilot on OEM A/B fleets to validate the $/event assumption before committing coverage work for OEM C. |

**Follow-up:** *"Why is GPS excluded?"* — the engine cites the ablation step where removing trip-level GPS moved AUC by +0.002 [−0.004, 0.008], notes GPS carries high importance in the recovery-prediction spec `[ev_recovery_022]`, and states that the exclusion applies to this capability's run cost only.

---

## 10. Build plan

Each step is useful on its own. Nothing depends on a later step to be real.

| Step | Deliverable | Useful alone because |
|---|---|---|
| 1 | `world/` + `data/` + `quality/` | Inspectable synthetic fleet with known truth; coverage and quality reports |
| 2 | `harness/` validated against the oracle | A DS can run importance/ablation/cross-OEM on any `DataSource` |
| 3 | `economics/` + deployment-fit rubric | CLI prints run cost and ROI distribution for any signal set |
| 4 | `policy/` + `ledger/` + `report/` | Headless end-to-end run produces a fully cited brief |
| 5 | `agent/` + `api/` | Free-text requests, follow-up Q&A, persisted runs |
| 6 | `web/` | Reasoning trail and evidence drill-down for PMs |
| 7 | `SnowflakeSource` | The step that turns a demonstration into a finding |

---

## 11. What would kill this

Being direct about failure modes, so they can be checked early.

- **Motorq's normalized tables don't expose per-signal cost attribution.** If ingest/storage cost can't be apportioned per signal × frequency, the COGS half of the pitch is a model, not a measurement. Mitigation: the price sheet is explicit and cited; the volume model is auditable; even a model beats a hand-wave.
- **Value assumptions are unavailable or contested.** If no one will commit to a $/event range, ROI is undefined. Mitigation: the brief still delivers feasibility, robustness, and cost; ROI is marked "requires value input."
- **Nobody runs it.** Internal tooling dies when it isn't in the workflow. Mitigation: the harness must produce a brief in the format the roadmap review already uses, and the CLI must be faster than the notebook it replaces.
- **The LLM layer eats the budget.** If most engineering time goes to the agent, the evidence quality suffers and the whole thing is a chatbot with charts. Mitigation: the LLM is the last step and is scoped to interface work only.

---

## 12. Summary

Motorq doesn't need an agent that decides what to build. It needs a machine that, for any candidate Fuse capability, produces in an hour the evidence a roadmap decision actually rests on — signal sufficiency, OEM coverage, robustness, run cost, ROI under stated assumptions — and does so reproducibly enough that the brief can be trusted, replayed, and argued with. The same machine, pointed at the portfolio, tells the data-integrations team which signals are earning their keep.

One deterministic pipeline. One cited brief. Human decision.

---

## Sources

- [Motorq launches Fuse (PR Newswire, Apr 2026)](https://www.prnewswire.com/news-releases/motorq-launches-fuse-to-turn-vehicle-insights-into-fleet-cost-savings-302741092.html) — Action Hub, Assistant, 250B data points/month, 25+ brands
- [Introducing Fuse (Motorq blog)](https://motorq.com/blog-news/motorq-fuse-ai-fleet-intelligence)
- [Motorq appoints Adi Bhashyam as President (PR Newswire, Mar 2026)](https://www.prnewswire.com/news-releases/motorq-appoints-adi-bhashyam-as-president-to-lead-the-ai-era-of-vehicle-intelligence-302727790.html) — "build the AI layer on top"
- [Motorq delivers connected vehicle data to Snowflake in seconds (Motorq blog)](https://motorq.com/blog-news/motorq-connected-vehicle-data-snowflake) — Snowpipe Streaming + Managed Iceberg, 60x latency improvement
- [OEM connectivity (motorq.com)](https://www.motorq.com/oem-connectivity) — supported brands, signal domains, coverage by model year
- [What is Motorq (Motorq blog)](https://motorq.com/blog-news/what-is-motorq) — ingest / normalize / deliver pipeline
- [Series B announcement (Motorq blog)](https://motorq.com/blog-news/announcing-motorqs-40-million-series-b-funding-round); [Tracxn profile](https://tracxn.com/d/companies/motor-q/__mBMdKFjIqOhOAidqahNykx9Dqy4nMY32YXWnaenJPso)

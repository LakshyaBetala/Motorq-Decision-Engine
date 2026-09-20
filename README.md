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
   |  ProblemSpec        |  target event . horizon . population . delivery mode
   |  (validated schema) |  value assumptions as [low, base, high] . constraints
   +---------+----------+
             v
   +--------------------+   measured event rate (Poisson CI) . coverage by OEM x
   |  Data feasibility   |   model year . freshness . drift . three-tier leakage
   |                     |   detector (dropped / confirm-availability)
   +---------+----------+
             v
   +--------------------+   signal-level permutation importance w/ CIs . cost-aware
   |  Experiment harness |   ablation (importance per dollar, paired cluster-bootstrap
   |                     |   non-inferiority) . daily-cadence ablation . forward
   |                     |   temporal split . leave-one-OEM-out with within-OEM
   |                     |   benchmark . fair one-signal baseline . EVENT-LEVEL
   |                     |   metrics: events caught, lead time, false alerts/100 veh-mo
   +---------+----------+
             v
   +--------------------+   sourced price sheet x volumes -> marginal & attributed
   |  Economics          |   run cost . PERT Monte Carlo ROI on event-level recall and
   |                     |   false-alert burden . net-value-optimal operating point .
   |                     |   tornado sensitivity
   +---------+----------+
             v
   +--------------------+   pure function of evidence: 4 hard gates (+ a cost gate when a
   |                     |   ceiling is requested) + 10 uncertainty
   |  Decision policy    |   flags (policy.yaml, versioned) -> BUILD_READY / PILOT /
   |                     |   NOT_FEASIBLE
   +---------+----------+
             v
   +--------------------+   every numeric line cites an evidence_id
   |  Evidence brief     |   renderer fails closed on uncited claims
   +--------------------+
```

**What humans supply and what the engine measures.** The target-event rate is *measured* from the data (with a confidence interval), not guessed. Humans supply only what data cannot know: the share of target events that would have caused the costly outcome (`value_bearing_fraction`), the share of that cost a correct early warning avoids, the dollar cost of one such outcome, and the fleet size — each as a `[low, base, high]` range. The verdict's sensitivity to each is reported.

**The LLM's role, precisely:** parse a free-text request into a `ProblemSpec` (never inventing business numbers; documented defaults are substituted and labelled as such); write a short narrative in which every sentence containing a number must cite an evidence id or is dropped; answer follow-ups ("why was GPS excluded?") using two tools that read the ledger. It never computes a number and cannot change the verdict. `mde run headless` runs the entire pipeline with no LLM at all.

---

## 5. Decision policy

The verdict is computed, versioned (`policy_version 1.0`), and unit-tested at every boundary — not generated.

```
HARD GATES         any failure -> NOT_FEASIBLE
  data       fleet share covered by the whole sufficient signal set >= min_oem_coverage (0.6)
  model      lower CI bound of (AUC - AUC of the best ONE-signal model on the same population) > 0
  economics  P(ROI > 0) >= 0.5 under the supplied value ranges
  delivery   at least one deployment pattern meets the horizon / latency constraints
  cost       (only when the request states max_run_cost_usd_month) marginal run cost <= ceiling

UNCERTAINTY FLAGS  any trip -> PILOT instead of BUILD_READY
  cross_oem_variance      leave-one-OEM-out AUC std > 0.03, or worst OEM > 0.05 below mean
  temporal_degradation    forward-split AUC below CV AUC by > 0.03 (upper bound)
  roi_spans_negative      5th-percentile ROI < 0
  value_unvalidated       value assumptions not yet validated against pilot outcomes
  short_history           a sufficient-set signal has < 6 months of history
  ablation_underpowered   an accepted removal rests on a bootstrap that could not resolve
                          tolerance/2 (0.0025 AUC)
  suspicious_signals      a sufficient-set signal is unusually strong - confirm it exists
                          before, not because of, the event
  cost_placeholders       the run cost rests on placeholder unit prices
  alert_burden            > 25 false-alert episodes per 100 vehicle-months at the chosen point
  data_still_improving    learning curve still rising (> 0.01 AUC from half to all vehicles):
                          the reported AUC is a lower bound

BUILD_READY only if every gate passes and no flag trips. Thresholds live in policy/policy.yaml
(version 1.3); the brief stamps the version.
```

"BUILD_READY" means *the evidence supports building*. Whether to build is still a human call.

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
| **Replay** | `mde run replay <run_id>` re-runs a study from its stored spec and diffs every evidence record; identical on the same dataset. |
| **Headless** | `mde run headless` runs the default plan end to end with no LLM and produces the same evidence and verdict. |

---

## 7. Data: the synthetic world and the Snowflake adapter

The engine ships with a synthetic world and a production adapter, and it is important to be exact about what each proves.

**The synthetic world is not random columns.** Each vehicle carries hidden wear states (brake wear, battery state-of-health, theft-risk context) that evolve with usage, climate, and OEM-specific reliability. Observable signals are noisy, OEM-dependent *views* of those states — a coverage matrix modeled on public sensor availability decides which OEMs emit which signals, at what cadence, with what missingness, from which model year. Events are hazard-driven from the latents; every rate parameter is sourced (FHWA mileage, brake-pad life, NICB theft, Geotab EV degradation, fleet-downtime studies) or explicitly marked as modeled in `world/params.yaml`. Ground truth — which signals drive which target, which are decoys (noisy functions of an observed driver), which are noise, which are leakage traps — is written next to the data and is the harness's test oracle.

**What synthetic proves:** the harness recovers planted structure and rejects decoys, leaks and noise; the policy behaves at its boundaries; the pipeline is byte-deterministic and replayable; the brief is fully cited. **What it does not prove:** anything about Motorq's actual signals.

**The adapter is real.** `SnowflakeSource` reads Motorq-shaped tables (`VEHICLES`, `SIGNALS_DAILY`, `EVENTS`, `SIGNAL_CATALOG`; names and columns are YAML configuration) and shares every line of labelling, coverage, quality and modelling code with the synthetic source through a common `FrameSource` base. It infers OEM coverage from data, samples VINs deterministically, and completes the vehicle-day grid. It is tested end-to-end with a fake query function that serves the synthetic dataset in production shape — the adapter must reproduce the synthetic source's labels and coverage exactly. See [docs/SNOWFLAKE.md](docs/SNOWFLAKE.md); the end-to-end pipeline, the canonical contract and the in-account deployment are in [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

```bash
mde run headless --example brake --snowflake snowflake.yaml --out brief.md
```

---

## 8. Architecture

Dependencies point strictly downward (enforced by import-linter). The LLM lives only in `agent/llm.py`.

```
web/                    Next.js - runs, verdict gates/flags, importance / ablation / cross-OEM /
                        tornado charts, cited brief with evidence drill-down, Q&A
motorq_de/api.py        FastAPI - /runs, /tickets, /runs/{id}/{brief,brief.md,evidence,events,ask,
                        whatif,replay}, /portfolio
motorq_de/agent/        runner (study, what-if, replay), portfolio (roadmap table + unused-signal
                        COGS report), service, llm (Anthropic, Bedrock or Gemini), webhook (Slack)
motorq_de/report/       typed brief; every numeric line cites; renderer fails closed
motorq_de/ledger/       runs . steps . evidence . messages   (Postgres or SQLite)
motorq_de/policy/       policy.yaml (versioned thresholds) . gates + flags -> verdict (pure)
motorq_de/economics/    price_sheet.yaml (sourced) . value_assumptions.yaml (owned by product) .
                        cost . value (PERT Monte Carlo on event-level metrics) . deployment
motorq_de/harness/      frames . stats (cluster bootstrap, event-level metrics) . models . experiments
motorq_de/quality/      coverage . quality . leakage . event_rate . usable_signals
motorq_de/data/         DataSource protocol . FrameSource . SyntheticSource . SnowflakeSource
motorq_de/world/        params.yaml (sourced) . coverage.yaml . registry . generator
```

**Stack:** Python 3.12, FastAPI, SQLAlchemy (PostgreSQL via `DATABASE_URL`, SQLite fallback), pandas, scikit-learn, LightGBM, Anthropic SDK (optional), Next.js 15 + Tailwind + Recharts, pytest.

---

## 9. Worked example — an actual run

`mde run headless --example brake` on the small synthetic fixture (600 vehicles, 200 days). Abridged; every line in the real brief carries its evidence id.

| Section | Finding |
|---|---|
| Event rate | Measured 1.05 brake services / vehicle-year [0.94, 1.16]; 2.0% of vehicle-days are positive at a 7-day horizon |
| Leakage | Two dealer-system fields caught and excluded: `service_appointment_scheduled` (flag lift 38x base rate) and `service_interval_remaining_days` (availability jump 4x, Spearman 1.0 with days-to-event). The real wear sensor (AUC 0.95 on the rows where it exists) is kept |
| Coverage | Fleet share with the full sufficient set 0.60 — exactly at the gate; OEMs C, E, G, H lack the wear sensor |
| Model | 13-signal sufficient set: AUC 0.877 [0.860, 0.894] vs full 90-signal set 0.872 [0.855, 0.888]; best one-signal model on the same population 0.838, so the lift CI lower bound is +0.023 (gate passes) |
| Robustness | Leave-one-OEM-out: mean 0.83, worst OEM H at 0.62 with 70% of features missing — **flag**. Forward split degradation +0.006 (upper bound 0.041) — **flag**, wide because the fixture is small |
| Cadence | The cost-aware sufficient set already needs no realtime polling — daily/weekly signals only |
| Economics | 7.2 GB/month vs 162 GB/month for the full set (−95% volume) but infra $2 vs $37 per month at list prices; the real levers are OEM polling cadence and build effort. Net-value-optimal operating point alerts the top 0.5% of vehicle-days: 18% of brake events caught with a median 6-day lead, 1.3 false-alert episodes per 100 vehicle-months. P(ROI > 0) = 0.92 under the stated value assumptions; the 5th-percentile ROI is still negative |
| **Verdict** | **PILOT**: all gates pass; flags trip on cross-OEM variance (the within-OEM benchmark says OEMs C/E/G/H *lack the signal*, the model transfers fine elsewhere), temporal degradation (wide on a small fixture), ROI tail, unvalidated value assumptions, underpowered ablation, and placeholder prices |

Change the value-bearing fraction in the what-if panel and the verdict recomputes in seconds, citing the same harness evidence. That is the tool doing its job: an honest, cited answer in five minutes, with the levers exposed, instead of a three-week study that arrives at "probably".

**What the first runs taught the design** (all fixed and tested):
- A noisier full-coverage signal can beat a precise sensor with OEM gaps — real, but the generator must not cheat by giving vendor signals the hidden state. Decoys are now noisy functions of the *observed* driver.
- "Best single signal" AUC on the rows where a signal exists is not a fair baseline for a model scored on everyone; the baseline is now a one-signal *model* on the same population.
- Asking humans for the event rate was one invented number too many; the harness measures it.
- Small data cannot resolve 0.005 AUC in ablation. The tool now reports its resolution and flags `ablation_underpowered` instead of pretending.
- Per-signal infra cost is pennies at list prices. Saying so plainly is more useful than a savings slide.
- Vehicle-day recall is not what a fleet manager experiences. Counting *events caught* and *false-alert episodes per 100 vehicle-months* changed the brake verdict from NOT_FEASIBLE to PILOT — the earlier proxy had over-counted false alerts several-fold.
- "Worst OEM at AUC 0.49" is ambiguous until you train on that OEM alone: within-OEM ≈ held-out means the OEM lacks the signal, not that the model fails to transfer.

---

## 10. Status and how to run it

| Step | Deliverable | Status |
|---|---|---|
| 1 | `world/` + `data/` + `quality/` — synthetic fleet with known truth, DataSource seam, leakage detector | done, tested |
| 2 | `harness/` — importance, ablation, temporal, cross-OEM, comparison; validated against the oracle | done, tested |
| 3 | `economics/` — sourced price sheet, cost, PERT Monte Carlo ROI, operating point, tornado, deployment fit | done, tested |
| 4 | `policy/` + `ledger/` + `report/` — verdict, provenance store, fail-closed cited brief; headless runner | done, tested |
| 5 | `agent/llm.py` + `api.py` — free text to spec, cited narrative, Q&A over evidence; HTTP API | done, tested with a stub client |
| 6 | `web/` — runs, gates/flags, charts, cited brief with evidence drill-down, Q&A | done |
| 7 | `SnowflakeSource` — production adapter | done, tested against a production-shaped fake; live connection needs credentials |
| 8 | Event-level metrics, cost-aware + cadence ablation, within-OEM benchmark, replay, what-if, portfolio + COGS report, policy/value config files, live progress, coverage heatmap, operating-point curve, Slack webhook, Bedrock switch, SQL templates, deployment guide | done, tested |
| 10 | Fit cache (no duplicate fits; bit-identical hits; replay bypasses it), learning curve with the `data_still_improving` flag (policy 1.3); dashboard redesigned as an evidence ledger: numbered stage rail, sticky verdict strip, evidence drawer, data-contract and quality table, redundancy groups, learning curve, runtime; Geist type, skeleton/empty/error states | done, tested |
| 9 | Canonical data contract (validated at every DEFINE, `mde data check`), COVESA VSS mapping of the signal registry, frozen-sensor and plausibility checks, runtime fingerprint with replay environment comparison, signal redundancy groups, cost-ceiling gate; [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) answers the methodology questions | done, tested |

```bash
uv sync --extra dev --extra api --extra agent        # python 3.12
uv run mde world generate --profile small             # ~7 s; `default` (5k vehicles, 18 months) ~2 min
uv run mde run headless --example brake --out brief.md   # ~3 min small (2 min with a warm fit cache), ~15 min on 5k vehicles
uv run mde data check                                 # canonical contract on the active source
uv run mde run list
uv run mde run evidence ev_<id>
uv run mde run replay <run_id>                        # re-run and diff every evidence record
uv run mde run whatif <run_id> --value-json '{...}'   # new human inputs, seconds
uv run mde portfolio                                  # roadmap table + signals no capability needs

# LLM interface (optional): ANTHROPIC_API_KEY, or MDE_LLM_PROVIDER=bedrock|gemini (see docs/DEPLOY.md)
uv run mde run ask "Should Fuse flag vehicles likely to need brake service in the next 7 days?"
uv run mde ask <run_id> "Why was GPS excluded?"

# API + dashboard
uv run mde serve                                      # http://127.0.0.1:8000
cd web && npm install && npm run dev                  # http://localhost:3000 (MDE_API_URL to point elsewhere)
docker compose up --build                             # Postgres + API + dashboard; see docs/DEPLOY.md

# tests (fast suite ~10 min; slow integration ~15 min; oracle on the default dataset ~30 min)
uv run pytest tests -q -m "not slow"
uv run pytest tests -q -m slow
MDE_ORACLE_DATASET=data/synthetic/<hash> uv run pytest tests/oracle -q
```

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

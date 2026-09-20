# Deploying inside Motorq

The engine is two containers (API + dashboard) and a Postgres ledger. Nothing else.

## 1. Bring-up

```bash
cp .env.example .env            # fill in what applies
docker compose up --build       # Postgres + API (:8000) + dashboard (:3000)
```

The API generates the default synthetic dataset on first start so the dashboard is usable
immediately. Point it at production data with `MDE_SNOWFLAKE_CONFIG=snowflake.yaml` plus the
`SNOWFLAKE_*` credentials (see [SNOWFLAKE.md](SNOWFLAKE.md) and the table templates in
`docs/sql/`).

On ECS/EKS: run the two images from the compose file with the same environment. The API needs
about 2 vCPU / 8 GB for 20k-vehicle studies; 8 vCPU roughly halves study time because
cross-validation folds run in parallel processes. Set `MDE_CPUS` to the task's CPU limit
(inside a container `os.cpu_count()` reports the host), or `MDE_FOLD_JOBS=1` to force
sequential folds. Parallelism never changes a number - only wall time (pinned by
`tests/harness/test_models.py`). Studies are serialised (one at a time per API process); run
more replicas behind a queue if concurrency matters.

## 1a. Keeping the data inside Motorq's boundary

The engine only ever needs read access to three tables, so the deployment that keeps vehicle
data inside is to run the API container **in Motorq's own Snowflake account on Snowpark
Container Services**: build the image, push it to the account's image repository, create a
compute pool (CPU is enough) and a service from `docker-compose.yml`'s API definition with the
same environment. The container reads over the internal connection; Snowflake's IAM, private
networking and secrets apply, and nothing egresses. The dashboard can run alongside it or on
ECS behind the identity layer below. An ECS/EKS deployment inside Motorq's VPC with a private
link to Snowflake is the equivalent alternative.

Whatever the placement: VINs are hashed before the engine sees them, the ledger holds
aggregates and statistics only, and with `MDE_LLM_PROVIDER=bedrock` the evidence digest the
model reads stays inside AWS as well.

## 2. Access control

Do not build auth into the app. Put the dashboard (and the API, if it must be reachable) behind
the identity layer Motorq already runs:

- **Cloudflare Access**: an Access application on the dashboard hostname with a policy of the
  Motorq Google Workspace group. Zero code.
- **Google IAP** (GCP) or an **ALB + OIDC** listener rule (AWS): same effect.

Keep the API on the private network; the dashboard proxies to it server-side (`MDE_API_URL`).

## 3. Inputs Motorq owns

| File | Owner | Cadence |
|---|---|---|
| `src/motorq_de/economics/price_sheet.yaml` | finance / platform | replace placeholders with contracted rates; bump `as_of` |
| `src/motorq_de/economics/value_assumptions.yaml` | product | ranges from the $-impact models Fuse Action Hub uses; review each roadmap cycle |
| `src/motorq_de/policy/policy.yaml` | data-science lead | bump `version` on any threshold change; briefs stamp it |
| `snowflake.yaml` | data platform | table names, VIN sample size, date window, event type map |

The `cost_placeholders` flag trips until no placeholder remains in the price sheet.

## 4. Where briefs go

- `MDE_WEBHOOK_URL` — a Slack-compatible incoming webhook. Every finished study posts the
  decision, gates, flags, headline numbers with their evidence ids, and a dashboard link
  (`MDE_DASHBOARD_URL`).
- `GET /runs/{id}/brief.md` — Markdown for Confluence / Notion.
- `mde portfolio --json` — machine-readable roadmap table and unused-signal report.

## 5. Scheduled portfolio / COGS report

Run monthly against the ledger to tell data integrations which signals and polling cadences no
viable capability needs:

```bash
scripts/portfolio_job.sh      # prints the table; posts a summary to MDE_WEBHOOK_URL when set
```

Schedule it with the platform's cron (Kubernetes CronJob, ECS Scheduled Task, or a GitHub
Actions `schedule:`). The job only reads the ledger and the signal catalog.

## 6. LLM provider

Headless mode needs no LLM. For free-text requests, narrative and Q&A:

- Anthropic API: `ANTHROPIC_API_KEY`
- Amazon Bedrock (AWS-native): `MDE_LLM_PROVIDER=bedrock`, `AWS_REGION`, AWS credentials on the
  task role; install with `uv sync --extra bedrock`. The model id is prefixed automatically.

- Google Gemini: `MDE_LLM_PROVIDER=gemini`, `GEMINI_API_KEY`; plain REST, no extra install.
  The free tier has a small per-model daily request cap; the client retries 429/5xx with backoff.

`MDE_LLM_MODEL` overrides the model (default `claude-opus-5`; `gemini-3.6-flash` for Gemini).
Whatever the provider, the model only parses requests, writes cited prose and answers
questions over stored evidence; it cannot set constraints the request did not state, and
any sentence with a number but no valid evidence citation is dropped.

## 7. Privacy

VINs never enter the ledger, briefs, webhooks or the LLM context — only per-OEM aggregates and
signal-level statistics. Asserted by
`tests/agent/test_runner.py::test_no_vehicle_ids_leak_into_evidence_or_brief`.

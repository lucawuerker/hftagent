# Product API Plan — FastAPI service over the QuantFundAgent pipeline

Status: **draft plan, 2026-07-28** (researched via codebase exploration + web research;
sources cited inline). Owner: Luca + co-founder. Product working name: **Lodestar**.

## 0. TL;DR

Build a Python **FastAPI** service (`service/` package, new) that wraps the existing
pipeline as **async jobs**:

- **Strategy research** (the hot path, 1–10 min): user brief → match against the
  curated **factor book** → Selector(bypass)/Architect/Statistician run → card +
  badge + equity curve, streamed live.
- **Factor research** (rare, minutes–hours): only when the factor book cannot serve
  the user's request.
- **Durable streaming**: jobs never write to the HTTP response. Every run appends
  events to a Postgres `job_events` log; the SSE endpoint **replays from
  `Last-Event-ID`, then tails** (LISTEN/NOTIFY). Reconnects, multi-device, and
  worker restarts are free.
- **Isolation**: every run is a **child process** with an explicit env dict and its
  own workspace `Scope` — mandated by the codebase (env-var seams + module-global
  caches + shared `factors/researcher/` package; see §3).
- **Orchestration**: **Hatchet** (Postgres-backed, one `hatchet-lite` container,
  retries + per-user concurrency caps + dashboard). Fallback: Procrastinate 3.x.
- **DB**: one shared **Postgres** (recommend Neon; migrate off PlanetScale MySQL),
  two schemas: `app.*` (TS backend owns) and `quant.*` (Python service owns).
  Heavy artifacts (joblib, full return series) → object storage (S3/R2) with
  URI + sha256 in the row.
- **MCP**: after the REST API stands, mount **FastMCP 3.x** at `/mcp` with 3
  curated tools (`submit_research_run`, `get_run_status`, `get_run_events`)
  calling the same service layer (~50 lines).

---

## 1. Target architecture

```
Browser (React/Vite)
   │  auth'd session; SSE via TS proxy (forward Last-Event-ID) or direct
   ▼
TS backend (owns users/auth/billing/chat)
   │  short-lived signed JWT: sub=user_id, aud=quant-service, exp≈60s
   ▼
Python FastAPI service  (/v1 REST + /mcp later)         ── uvicorn, 1–2 workers, I/O only
   │  enqueue                                            never runs pipeline code in-process
   ▼
Hatchet worker fleet (separate deployment)
   │  task = thin wrapper: spawn subprocess per run
   ▼
Pipeline child process (Scope-activated env, QF_USE_MCP=0)
   │  emits JSONL events → worker inserts into quant.job_events
   ▼
Postgres (app.* + quant.*)  +  Object storage (joblib, full series)
```

Key rules:

1. **The API tier only enqueues and serves SSE.** Anything >10 ms CPU goes to the
   queue. Deploys of the API never kill runs (workers are a separate deployment).
2. **The job never streams to the client directly.** It writes to the durable log;
   the HTTP layer replays + tails. This one rule makes resumability free.
3. **One child process per run, explicit env.** Same pattern as
   `comparison/rolling.py:143` and `comparison/downstream.py:39`.

---

## 2. Product flow: chat request → strategy

1. **User brief** arrives (TS backend → `POST /v1/jobs`, type `strategy_research`,
   `Idempotency-Key` header, payload = user's plain-English brief + constraints).
2. **Factor-book match** (fast, in the API tier or a quick job step): embed the
   brief and match against a **new embedding index over the factor book**
   (id/name/description/category/mechanism of each `FactorRecord`). The existing
   `knowledge/embed_store.py` machinery is reusable but is built over the *paper*
   corpus — build a small parallel index over the factor catalog.
   - **Coverage OK** → strategy research over the existing zoo.
   - **Gap detected** (no factor family matches the requested mechanism/style) →
     respond with "needs factor research" and (per product policy) enqueue a
     `factor_research` job; strategy research runs afterwards over the extended book.
3. **Strategy research job** (worker, subprocess):
   - Activate a per-run `Scope(config_name, prerun=f"user_{uid}_{job_id}")`,
     compose the runtime factor DB from the curated book.
   - **Bypass the Selector's own hypothesis invention**: `selector_graph.invoke({})`
     unconditionally invents its own hypothesis (`agents/selector/graph.py:93`), so
     for a user-driven product either (a) add a conditional skip in
     `formulate_hypothesis` when a hypothesis is supplied, or (b) build
     `{hypothesis, selected_factor_ids, factor_catalog}` from the user brief +
     factor match and call `architect_graph` directly (no repo change needed).
     Recommend (a) as a small upstream patch: user brief → hypothesis; the
     Selector still picks factors.
   - Run Architect → Statistician (`pipeline.run_strategy_pipeline` semantics),
     emitting semantic events per stage/trial.
   - Post-process with the **landing-examples machinery, which is already the
     product output spec**: `assemble_card_metrics` → `assign_badge` (deterministic,
     never the LLM's prose) → `verdict_note` → `recompute_equity_curves` →
     `svg_points`. Ship `card.json` + `equity_curve.json` shapes as the API result.
   - Publish: upload joblib + full series to object storage, insert
     `quant.strategies` + `quant.strategy_factors` rows, emit `done` event.
4. **Frontend** renders the progress feed (stage events), then the card
   (badge, PBO, DSR, OOS Sharpe, equity SVG polyline — all fields already produced
   by `showcase_pipeline/landing_examples/`).

### Runtime expectations (measured, from batch metas on disk)

| Job | Wall clock | LLM calls |
|---|---|---|
| Strategy attempt (25 tickers, gpt-4o-mini) | ~70 s – 4.6 min | ~5–15 |
| Strategy attempt (2 iters, 15 tickers) | ~23 s | ~7 |
| One-shot factor research | minutes | dozens |
| Evolution factor research | hours | hundreds |

Dominated by panel load + repeated backtests, not LLM latency. First panel load is
the big fixed cost → **pool warm workers per (config, scope)** rather than cold-spawning
per request once volume justifies it (v1: cold spawn is fine).

---

## 3. Why subprocess-per-run is non-negotiable (codebase findings)

- `Scope.activate()` mutates process-global `os.environ` (`workspace.py:188`);
  `FACTOR_DB_PATH` / `STRATEGY_RETURNS_DIR` / `MODEL_ARTIFACT_DIR` / `QF_SCOPE`
  are read live from env by services.
- **Unkeyed module-global caches**: `modeling/service.py:29` and
  `agents/architect/graph.py:659` hold `_PANEL_CACHE` (no key) and `_SIGNAL_CACHE`
  keyed only by `factor_id` — two concurrent runs with different universes corrupt
  each other.
- **Import-time constants**: `DATA_DIR`, per-role `*_LLM_MODEL` are read at module
  import; per-request env changes are no-ops after first import.
- `MCPBridge._instances` is one stdio subprocess per server for the whole process,
  env snapshotted at spawn; `llm._METER` is a single global usage meter (per-job
  cost attribution requires a process boundary — read `usage_summary()` at child exit).
- **Factor codegen writes into the shared `factors/researcher/` package** and
  live-reloads modules → concurrent *factor research* runs corrupt each other's
  module namespace: strictly **one factor-research job at a time per host**
  (Hatchet concurrency key), or containerise per run later.

Worker task shape: `subprocess.run(entrypoint, env={**base, **scope.export_env(),
"QF_CONFIG_FILE": ..., "QF_USE_MCP": "0", ...}, timeout=...)`, child emits JSONL
events on stdout/file, worker inserts them into `job_events`, kills the process
group on cancel/timeout, reads the LLM cost summary at exit for per-job billing.

---

## 4. API design (v1)

Service-to-service auth: static bearer (compare_digest) from day one; upgrade the
user identity to a short-lived signed JWT (`sub=user_id`, `aud=quant-service`)
before first real users — never accept a bare `user_id` header (confused deputy).
The Python service is never exposed to the public internet.

```
POST /v1/jobs                     submit {type: strategy_research|factor_research,
                                  user_id (from JWT), chat_session_id?, brief, constraints}
                                  Idempotency-Key header → unique (user_id, key)
                                  → 202 {job_id, status}
GET  /v1/jobs/{id}                status + summary
GET  /v1/jobs/{id}/events         SSE; replay from Last-Event-ID (or ?after=seq), then tail
POST /v1/jobs/{id}/cancel         kill process group, mark cancelled
GET  /v1/factors                  curated zoo catalog (id, name, category, horizon, IC stats)
GET  /v1/factors/match?q=...      embedding match of a brief against the zoo (coverage check)
GET  /v1/strategies?user_id=      list user's strategies (cards)
GET  /v1/strategies/{id}          card.json + equity_curve.json + provenance
GET  /v1/strategies/{id}/signals  signal history (later: live signals)
```

SSE specifics (production pitfalls): `id:` on every event; heartbeat comment every
15–30 s (ALB 60 s / Cloudflare ~100 s idle timeouts); `X-Accel-Buffering: no` +
`Cache-Control: no-cache`; no gzip middleware on the stream route; TS proxy must
stream and forward `Last-Event-ID` (or let the browser connect directly).

Event vocabulary (semantic, coalesced — never per-token):
`job_queued`, `stage_started` (selector/architect/statistician), `trial_completed`
(iteration metrics), `factor_matched`, `llm_note` (coalesced ~100–250 ms),
`stat_test_result`, `done` (terminal, carries card payload), `error` (terminal),
`cancelled` (terminal). `dump_candidate`'s JSON keys are the payload vocabulary;
`batch_meta["attempts"]` is the progress feed model.

Project layout (domain-package style, per zhanymkanov/fastapi-best-practices):
```
service/
  core/{config.py, db.py, security.py, logging.py}    # pydantic-settings, structlog
  jobs/{router.py, service.py, repository.py, schemas.py, models.py}
  streaming/{router.py, sse.py}
  factors/{router.py, matcher.py}
  strategies/{router.py, repository.py}
  mcp_app.py                                          # FastMCP mount (phase 4)
  worker/{tasks.py, runner.py}                        # Hatchet tasks + subprocess runner
  entrypoints/run_strategy_job.py                     # child-process entrypoint
```

---

## 5. Job orchestration

**Recommendation: Hatchet** (`hatchet-sdk` 1.33.x, MIT, self-host = one
`hatchet-lite` container on the same Postgres): async-native, typed Pydantic I/O,
retries, cron (later: signal generation), **per-user concurrency caps** (protects
the LLM bill), sticky workers, dashboard. 2026 notes: arq is maintenance-only;
Celery still has real asyncio friction; Temporal is overkill — the evolution loop
already checkpoints per generation (application-level durability), and a died
strategy run is simply retried.

Zero-new-container alternative: **Procrastinate 3.x** (Postgres-only queue,
SKIP LOCKED + LISTEN/NOTIFY). Choose it if adding the Hatchet container feels heavy.

Queue policies: per-user cap (e.g. 1–2 concurrent strategy runs), **global cap 1 on
`factor_research`** (shared researcher package, §3), job timeout + kill, retry
strategy runs at most once (idempotency key makes resubmission safe).

---

## 6. Database (one shared Postgres, two owners)

Move off PlanetScale MySQL → **Postgres** (Neon recommended: free tier, scale-to-
zero, branching; Supabase if we want its Realtime for later signal push; PlanetScale-
for-Postgres exists since 09/2025 but has no free tier). Rationale: real
unconditional FKs for user↔strategy↔subscription↔signal (PlanetScale MySQL FKs are
opt-in and vendor-discouraged), JSONB, SQLAlchemy 2.x + Alembic ecosystem. Current
footprint is one demo user table — migration will never be cheaper.

Ownership: `app.*` written by TS backend, `quant.*` written by Python; each side
reads the other read-only (Postgres roles enforce). Alembic migrates `quant.*` only.

**`app.*` (TS-owned):** `users`, `billing_*`, `chat_sessions`,
`chat_messages(…, research_job_id nullable FK)`.

**`quant.*` (Python-owned):** UUIDv7 PKs, `timestamptz` everywhere.

| Table | Key columns |
|---|---|
| `research_jobs` | id, user_id FK, chat_session_id?, type, status (queued/running/succeeded/failed/cancelled), request_payload JSONB, idempotency_key (UNIQUE (user_id, idempotency_key)), error JSONB, cost_usd, started_at, finished_at |
| `job_events` | id BIGSERIAL (SSE cursor), job_id FK, seq (UNIQUE (job_id, seq)), event_type, payload JSONB, created_at — append-only; replay+tail source |
| `factors` (curated zoo) | id, slug UNIQUE, name, description, code_ref (git path@commit or URI), code_sha256, prediction_horizon, inputs JSONB, provenance JSONB, status, version — new code = new version, never mutate |
| `strategies` | id, owner_user_id FK, origin_job_id FK, name, spec JSONB (StrategySpec), metrics JSONB, **promoted columns**: badge, oos_sharpe, pbo (for WHERE/ORDER BY), equity_curve JSONB (downsampled card series), model_artifact_uri + sha256, status, created_at |
| `strategy_factors` | (strategy_id, factor_id) PK, factor_version, weight — normalized m2m ("which strategies use factor F" matters when deprecating) |
| `subscriptions` | id, user_id FK, strategy_id FK, status, channel_prefs JSONB; UNIQUE (user_id, strategy_id) WHERE active — owner ≠ subscriber |
| `signals` | id, strategy_id FK, as_of, positions JSONB, signal_hash; UNIQUE (strategy_id, as_of) — idempotent generation |
| `signal_deliveries` | id, signal_id FK, subscription_id FK, channel, status, attempts, last_error, delivered_at; UNIQUE (signal_id, subscription_id, channel) — fan-out + audit trail |

Artifacts: joblib models + full-resolution return series + transcripts → object
storage (S3/R2), row stores `uri + sha256 + size` (the signal worker verifies the
hash of the model it loads — provenance is a product feature for "honest quant").

Live signals (later phase) = transactional outbox: scheduler computes positions per
active strategy, inserts `signals` row + one `signal_deliveries` row per
(active subscription × channel) **in one transaction**; a delivery worker claims
with `FOR UPDATE SKIP LOCKED`, at-least-once, idempotent per unique key. In-app
display needs no delivery machinery at all (read `signals` via subscriptions).
Push/websockets later are just additional channels on the same tables.

---

## 7. MCP exposure (phase 4, cheap by construction)

- **FastMCP 3.x** (PrefectHQ) mounted at `/mcp` on the same FastAPI app
  (Streamable HTTP transport). **Gotcha**: combine the MCP app's lifespan into
  FastAPI's or every request 500s.
- Hand-write 3 curated tools over the same service layer (don't auto-convert the
  whole OpenAPI): `submit_research_run(...) -> job_id` (idempotent),
  `get_run_status(job_id)`, `get_run_events(job_id, after_cursor)`.
- Long-running semantics: polling via the tools today; adopt **MCP Tasks
  (SEP-1686)** for call-now/fetch-later once client support lands.

---

## 8. Phases

**P1 — Walking skeleton (API + jobs + streaming), ~1 week of focused work**
- `service/` package: FastAPI + pydantic-settings + structlog; `/v1/jobs`,
  `/v1/jobs/{id}`, `/v1/jobs/{id}/events` (SSE replay+tail), static bearer auth,
  Idempotency-Key.
- Postgres (Neon) with `quant.research_jobs` + `quant.job_events`; Alembic baseline.
- Hatchet (or Procrastinate) worker; child-process entrypoint
  `entrypoints/run_strategy_job.py` = thin wrapper around
  `landing_examples.driver.run_batch(n_strategies=1)` semantics with per-run Scope,
  JSONL event emission, cost summary at exit.
- Small upstream patch: Selector honours a supplied hypothesis
  (`formulate_hypothesis` conditional skip).
- Deliverable: `curl` submits a brief → SSE shows stages → final event carries
  `card.json` + `equity_curve.json`.

**P2 — Product data model + factor matching**
- `quant.{factors,strategies,strategy_factors}` tables; publish step (object
  storage upload + rows); `GET /v1/strategies*`, `GET /v1/factors`.
- Factor-book embedding index + `GET /v1/factors/match` (coverage check);
  "gap → factor_research job" decision rule (global concurrency 1).
- Migrate TS demo user table PlanetScale → Neon `app.*`; JWT identity assertion
  TS→Python.
- Meter fix upstream: route Selector/Architect/Statistician/PM through
  `make_chat_llm` so per-job cost is real before billing on it.

**P3 — Frontend integration + hardening**
- TS backend proxy (or direct SSE) with Last-Event-ID forwarding; chat UI renders
  event feed + card; cancel button.
- Heartbeats, retention policy for `job_events` payloads, rate limits (slowapi
  backstop; real limits in TS backend), structured request logging, e2e tests.

**P4 — MCP + live signals**
- FastMCP mount + 3 tools.
- Signal scheduler (Hatchet cron) + outbox delivery worker + `subscriptions`/
  `signals`/`signal_deliveries`; in-app signal history endpoint.

---

## 9. Open decisions / risks

1. **Pipeline depth for the product**: full `run_strategy_pipeline`
   (`max_iterations=3–6`) vs a trimmed variant (fewer Architect iterations,
   smaller universe) for snappier UX. Measured spread is 23 s – 4.6 min/attempt;
   likely offer "quick" vs "thorough".
2. **Factor research trigger policy**: auto-enqueue on gap (hours!) vs "request
   queued, we'll notify you" UX. Recommend the latter + email/push on completion.
3. **Warm worker pools** per (config, scope) once cold-spawn panel loads hurt —
   v1 ships cold-spawn.
4. **Neon vs Supabase**: Neon now; revisit if/when Realtime push for signals is
   wanted.
5. **Selector/Architect/Statistician LLM calls are not metered today** (direct
   `ChatOpenAI`, not `make_chat_llm`) — must fix before per-user billing.
6. **Compliance**: keep the deterministic badge/`check_compliance`/`CAVEAT`
   separation from `landing_examples/verdict.py` — badges never come from LLM
   prose; every card carries the caveat.

## Appendix: key codebase seams

- `quant_fund_agent/pipeline.py` — `run_strategy_pipeline`, `run_research_session`,
  `persist_approved_strategy`, `run_pm_rebalance`.
- `quant_fund_agent/workspace.py` — `Scope`/`Book`, `export_env()`, `activate()`.
- `showcase_pipeline/landing_examples/{driver,metrics,verdict,export,transcript}.py`
  — the product output spec (card.json, equity_curve.json, badge, PBO, provenance).
- `quant_fund_agent/comparison/rolling.py:143` — the subprocess-per-run pattern.
- `quant_fund_agent/llm.py` — `UsageMeter`, `usage_summary()`, `QF_MAX_LLM_COST_USD`,
  `QF_LLM_TRANSCRIPT_PATH` (ready-made "show the reasoning" surface).
- `quant_fund_agent/mcp/` — service/client/server triads; `QF_USE_MCP=0` in-process
  mode is what the product child processes should use.

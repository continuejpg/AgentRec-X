# AgentRec-X Improvement Plan

> Status: implemented in the worktree; not committed or pushed in this step. This file describes
> the engineering changes motivated by public-project patterns. The existing deterministic
> recommendation/truth boundary is preserved.

## Public patterns used

| Pattern | Public reference | What we adopt | What we do not adopt |
| --- | --- | --- | --- |
| frozen experiment bundles, config/hash provenance | RecBole, MLflow | experiment manifest with cohort/pool/protocol hashes and driver version | a new framework dependency |
| paired uncertainty and sliced reporting | RecBole evaluation practices, paired bootstrap literature | paired bootstrap, head/body/tail slices | a single aggregate score |
| durable execution/checkpointing | LangGraph checkpointing, Temporal | explicit checkpoint state, idempotency keys, restart/resume tests | Temporal/Redis deployment in the local demo |
| operational tracing/metrics | OpenTelemetry, Phoenix | per-turn counters, latency quantiles, JSON metrics endpoint | a hosted tracing backend |
| bounded tool execution | OpenHands/SWE-agent | typed action/provenance already present; add operational accounting | autonomous code execution |

## Phase 1 — Evaluation harness

Deliverables:

- reusable experiment metadata object: protocol/cohort/candidate-pool versions, frozen test flag,
  code/driver version, data/model hashes;
- paired bootstrap utility (deterministic seed, paired cases);
- popularity slices: head/body/tail from a supplied item-popularity map;
- aggregation of per-case outcomes without inventing a single score;
- tests using synthetic deterministic cases.

Acceptance:

- the same inputs produce the same manifest and summary;
- the manifest refuses a non-frozen test or missing required hashes;
- paired bootstrap is case-paired and reproducible;
- slices use explicit popularity boundaries, not inferred labels;
- no change to the default serving path.

## Phase 2 — Agent operational metrics

Deliverables:

- per-turn metrics:
  - task success (`terminal.succeeded` / completion),
  - hard-constraint violation rate (from the response's verified projection/summary),
  - unverified recommendation rate,
  - clarification closure rate (answered pending -> terminal),
  - tool calls,
  - p50/p95 end-to-end latency;
- thread-safe collector;
- `GET /v1/demo/metrics` JSON snapshot;
- tests for counters, rate denominators, percentiles and empty state.

Acceptance:

- no metric is reported when its denominator is zero;
- constraint/unverified counts are not inferred from prose;
- endpoint is read-only and does not change session state;
- the existing trace remains the per-turn audit source.

## Phase 3 — Durable session/pending/task state

Deliverables:

- SQLite checkpoint store using the existing runtime database directory;
- sessions: profile, trusted history, turn sequence, decision mode, last presentation;
- pending clarifications: task/session key, kind, payload, creation time, round, budgets;
- task state: task preferences, rounds, second-round flag, clarified-session marker;
- runtime wiring to an opt-in durable database; in-memory remains the default for unit tests;
- restore/replay tests: process restart, concurrent sessions, duplicate/cross-session answers.

Acceptance:

- a session resumes after constructing a new runtime against the same checkpoint DB;
- an answered/consumed clarification cannot be replayed;
- two sessions never share task preferences or pending state;
- in-memory behavior remains available and unchanged when checkpointing is disabled;
- multi-worker deployment still requires a shared database and a per-session lease; the local
  demo does not claim multi-worker correctness in Phase 3.

## Sequence

1. Add the harness and tests; run targeted evaluation tests.
2. Add operational metrics and endpoint; run demo/API tests.
3. Add durable state behind an opt-in constructor/runtime argument; run session/agent tests.
4. Run the affected regression files, then a full suite if the changes are stable.

## Implemented interfaces

- Evaluation: `recommendation/evaluation/experiment.py` exports a frozen manifest,
  case-paired bootstrap, popularity slices, and a JSON bundle builder.
- Metrics: `GET /v1/demo/metrics` returns the process-local operational snapshot.
  The same collector is also used by the agent routes; it does not mutate a session.
- Durable state: `SQLiteCheckpointStore` (`recommendation/demo/checkpoint.py`) stores sessions,
  pending clarifications and task-scoped preference state. It is enabled automatically when the
  server builds its default SQLite memory database; `AGENTRECX_DURABLE_DB` can point to an
  explicit database path. `checkpoint_db=`/`checkpoint=` are explicit injection points for tests
  and future Redis/Postgres adapters.
- A durable checkpoint is local SQLite/WAL, not a distributed lease. Multiple workers sharing
  one file can read/write the rows, but per-session turn serialization is still in-process;
  production multi-worker correctness requires a shared lease or a durable workflow service.
- Token/cost telemetry is not fabricated: the current local Agent response does not expose
  provider usage, so the metrics snapshot marks it unavailable unless an adapter supplies it.

## Validation status

- `tests/test_evaluation_experiment.py`: passed.
- `tests/test_operational_metrics.py`: passed.
- `tests/test_durable_checkpoint.py`: passed (restart, pending preference, pending research).
- Directly affected regression set: 283 passed, 8 skipped.
- Full suite after the changes: **3202 passed, 40 skipped, 10 warnings, exit 0**
  (`timeout 1800`, 1379.42s).
- No training, no external provider call, no production deployment was performed.

## Non-goals

- no end-to-end LLM replacement of the recommender;
- no online RL or model retraining in these phases;
- no Redis/Postgres service dependency in this repository;
- no change to the frozen evaluation protocol or model checkpoints.

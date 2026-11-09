# Multi-turn shopping Agent benchmark

This directory is a benchmark harness around the existing AgentRec-X recommendation
components. It does not replace the serving path, retrain a model, add a recall source, or
use next-item evaluation labels as shopping success labels.

## Policy arms

| arm | policy | requires LLM |
|---|---|---|
| A | current-input LLM query rewrite, shared stateless candidate path | yes |
| B | full-dialogue-history LLM query rewrite, shared stateless candidate path | yes |
| C | rule Memory (M5) + R1 Retrieval Planner + M7 interaction | no |
| D | C + LLM structured preference proposal | yes |

A/B use the same SASRec, Two-Tower, BM25 catalogue source, candidate fusion, trusted
metadata and product-role checks. C/D use the existing stateful shopping service. BGE and
ItemCF are explicitly OFF for this benchmark.

If the LLM provider is missing, A/B/D are reported as `unavailable`; no fallback, no
simulated LLM output, and no fabricated trajectory. C is fully offline and runnable.

D first version records validated structured proposals but routes only the faithful
rewritten message through the C/M5 pipeline; it does not write a proposed fact into M5
without an explicit user confirmation path. The D comparison remains unverified until a
provider is configured.

## Audit reports

- [C_FAILURE_AUDIT.md](reports/C_FAILURE_AUDIT.md) — per-task C failure attribution.
- [C_FAILURE_AUDIT.csv](reports/C_FAILURE_AUDIT.csv) — machine-readable diagnosis.
- [BENCHMARK_VALIDITY.md](reports/BENCHMARK_VALIDITY.md) — fairness, isolation and D-chain
  validity audit.
- [P1A_M5_PENDING.md](reports/P1A_M5_PENDING.md) — M5 effective-projection and pending
  clarification repair, with before/after C results.
- [P1A_STATE_COMPARISON.csv](reports/P1A_STATE_COMPARISON.csv) — per-task before/after
  deterministic status and M5 evidence.
- [P1B0_STATE_TO_PLANNER.md](reports/P1B0_STATE_TO_PLANNER.md) — M5 effective-state to R1
  input-channel audit and Shadow/Applied comparison.
- [P1B0_SHADOW_APPLIED_COMPARISON.csv](reports/P1B0_SHADOW_APPLIED_COMPARISON.csv) — 20-task
  Shadow/Applied results.
- [P1B0_TURN_TRACES.jsonl](reports/P1B0_TURN_TRACES.jsonl) — structured per-turn traces
  from both state-mode arms.
- [P1B1_GROUNDED_ELIGIBILITY.md](reports/P1B1_GROUNDED_ELIGIBILITY.md) — P1B-1 role /
  eligibility audit and fixes.
- [P1B1_ROLE_METRICS.csv](reports/P1B1_ROLE_METRICS.csv) — independent role/identity
  metrics for the 20 development tasks.
- [P1B2_EFFECTIVE_INTENT_PLANNER.md](reports/P1B2_EFFECTIVE_INTENT_PLANNER.md) — P1B-2
  effective-intent, price-lifecycle and R1 target-change report.
- [P1B2_TURN_COMPARISON.csv](reports/P1B2_TURN_COMPARISON.csv) — per-turn Shadow/Applied
  comparison for the 20 development tasks.
- [P1B3_SOFT_MEMORY_RERANK.md](reports/P1B3_SOFT_MEMORY_RERANK.md) — P1B-3 M5 soft
  preference → M6.1 rerank report.
- [P1B3_SOFT_OFF_ON_COMPARISON.csv](reports/P1B3_SOFT_OFF_ON_COMPARISON.csv) — per-turn
  soft_off / soft_on ranking and retraction comparison.
- [P1C_EVALUATOR_VALIDITY.md](reports/P1C_EVALUATOR_VALIDITY.md) — evaluator V2,
  legacy/V2 comparison and failure-attribution audit.
- [P1C_LEGACY_V2_COMPARISON.csv](reports/P1C_LEGACY_V2_COMPARISON.csv) — per-task
  legacy metrics and independent V2 metrics/evidence.
- [P1C_POLICY_FAIRNESS.md](reports/P1C_POLICY_FAIRNESS.md) — A/B/C/D capability and
  availability fairness audit.
- [P1C_REPRODUCIBILITY.md](reports/P1C_REPRODUCIBILITY.md) — forward/repeat/reverse and
  state-isolation verification.
- [human_labels_v2_schema.md](human_labels_v2_schema.md) — independent human relevance
  annotation schema (provisional, not yet a completed label set).
- [human_labels_v2.template.csv](human_labels_v2.template.csv) — blank per-task template
  for the independent annotation pass (no fabricated human labels).

## Run

```bash
python -m benchmarks.agent --split dev --policies C --out runs/agent_benchmark/dev
python -m benchmarks.agent --split dev --policies A,B,C,D --out runs/agent_benchmark/dev
python -m benchmarks.agent --split frozen --policies C --out runs/agent_benchmark/frozen
```

The runner writes:

- `trajectories.jsonl` — per-turn observations, R1 actions, memory diagnostics, final
  products, latency and token usage;
- `evaluations.jsonl` — deterministic checks and human-label link only;
- `metrics.csv` — one row per task/policy;
- `summary.md` and `failures.md`;
- per-policy task directories for independent sessions and state/checkpoint directories.

## Task schema

Each task has:

- 3–5 user turns;
- trusted initial history;
- expected final product identities/roles;
- expected title evidence, forbidden roles and optional active preference attributes;
- optional expected R1 actions.

Expected answers are not passed to any policy adapter or Agent service; only
`task.history` and `task.turns[*].user_input` are used. The validator checks all expected
products against the real Sports & Outdoors metadata and rejects tasks whose user text
contains an expected ASIN.

## Development and frozen splits

- `tasks/dev.json`: 20 development tasks, used for harness debugging only.
- `tasks/frozen.json`: 8 frozen tasks, not used to tune task definitions or policy
  settings. See [FROZEN_PROTOCOL.md](FROZEN_PROTOCOL.md).
- `tasks/manifest.json`: SHA-256 of both task files; the runner verifies it before execution.

## Development status (first version)

Development split run with the offline policy C:

```text
20 runnable tasks
19 completed conversations
2 deterministic passes
```

This low task-completion rate is a real observation, not a reason to tune the task
expectations to the model. The rule Memory/Planner path completed most conversations but
often returned products outside the task's trusted role/identity assertion; one
conversation remained blocked on a clarification. The frozen split has **not** been run.

A, B and D were explicitly `unavailable` because no LLM provider was configured. The
harness did not simulate LLM rewrites or proposal output. Do not interpret the C-only
development result as evidence that Memory/Planner beats the strong B rewrite baseline;
the controlled comparison is pending a configured provider.

## Evaluation limits

The only automatic success criterion is deterministic product/state/turn completion.
Human relevance labels are a separate file and are never generated by the harness. A
passing deterministic run is not evidence that a user was satisfied.

# P11 Benchmark Schema

- Task file: `data/p11_benchmark/tasks.jsonl`
- Schema: `data/p11_benchmark/schema.json`
- Manifest: `data/p11_benchmark/manifest.json`
- Query pools: `data/p11_benchmark/query_pools.json`

## Data provenance

Real Amazon Reviews 2023 Sports & Outdoors user histories and product metadata; requests
are benchmark templates, **not real shopping-search logs**. Only `train_history` is
visible; no validation/test target is stored or used as the answer. Frozen Holdout is not
read.

## Evidence levels

| Level | Meaning |
|---|---|
| A | Explicit positive review evidence. **Not available locally; count = 0.** |
| B | Repeated matching prior behaviour/category. Behavioural affinity only, not a confirmed preference. |
| C | Current task-declared constraint/preference. Not historical personalization. |
| UNKNOWN | No sufficient evidence; never converted into a positive label. |

## Required task fields

`task_id`, `user_id`, `source_dataset`, `task_type`, `conversation_turns`,
`visible_user_history`, `history_evidence_refs`, `preference_evidence_level`,
`effective_state_per_turn`, `required_product_role`, `constraints`,
`verified_candidate_pool`, `expected_action_when_verifiable`, `adjudication_status`,
`split`, `provenance`.

Every task also records `query_source`, `user_history_source`, and
`candidate_pool_stats`.

## Candidate verification

The verified candidate pool is built from catalogue metadata predicates plus existing
Product Role and numeric price checks. BM25/BGE rankings are **not** used to define
positives. The pool is capped and sorted deterministically by ASIN; it is a
**bounded verified candidate pool**, not a full-catalogue oracle. Missing price or unknown
role remains UNKNOWN and is not filled or guessed.

## Task families

- `historical_personalization`: same query, Level B evidence vs no-evidence user, and
  different-history-profile pairs.
- `explicit_intent_with_history`: Level B history plus explicit current constraints.
- `multi_turn_state_change`: budget/goal transitions with expected per-turn state.
- `clarification_required`: compatibility/semantic attribute not verifiable from trusted
  metadata; expected ASK/UNKNOWN.
- `unsatisfiable_or_unknown`: no verified candidate under declared condition; expected
  UNKNOWN, never automatic success.

## Evaluation separation

- Retrieval: verified candidate reachability@10/@50 for BM25/BGE/Hybrid and History
  source; no Gold from retrieval ranks.
- Personalization: Level B history-affinity proxy only; not human-confirmed satisfaction.
- Agent: strict bounded-pool success, catalogue-validity success, InvalidFinalize,
  HardConstraintViolation, clarification behavior, and execution cost reported separately.

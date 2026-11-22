# P0 — CPU CurrentTaskConstraints execution fix

> **Status: implemented and verified at the P0 commit.**
> Scope: wire the already-persisted task-scoped constraints into the current Web CPU
> recommendation path and make a hard exclusion actually decide eligibility.  No new
> retrieval/ranking algorithm, no Qwen/GPU, no frontend/README changes.

## 1. Root cause

The MCP `submit_clarification_reply` tool already extracted and persisted
`CurrentTaskConstraints`, and `ElectronicsPlannerExecutor` already knew how to apply them.
The CPU `recommend_products` path lost them in two places:

1. `web_demo/reply_executor.py` passed only `task_requirements` to `recommend_products`;
   `reply["task_constraints"]` was ignored.
2. `recommendation/mcp/server.py::_recommend_products` had no `task_constraints` parameter
   and constructed `ElectronicsPlannerExecutor` without `task_constraints`.

A third, independent defect made the "prefer not to have" case unsafe: missing attribute
evidence was classified as `unknown` and remained `eligible`; the Gate could finalize an
UNKNOWN candidate.  The old direct-Gate test explicitly expected this behavior.

## 2. Fixed call chain

```text
submit_clarification_reply
  -> reply["task_constraints"] (merged, source-turn bound)
  -> _reply_task_constraints(reply, pending)       # MCP reply first, Cold Resume fallback
  -> recommend_products(task_constraints=...)
  -> _constraints_from_snapshot(...)              # same decoder as restore
  -> ElectronicsPlannerExecutor(task_constraints=...)
  -> effective_preferences(...)                    # task exclusion overrides M9 positive
  -> candidate_constraint_status(...)
  -> _score_and_inspect: exclude violations AND hard-unknown
  -> DecisionGate: defense-in-depth hard-unknown filter
  -> selected product or existing bounded refusal
```

The same constraints are included in the durable reply result and in `awaiting_clarification`
results, so ASK2's continuation and a later Cold Resume carry them.  They are **not** written
into M9.

## 3. Execution semantics

- `hard_exclusions`: a candidate is eligible only when trusted evidence proves the excluded
  attribute is absent.  `wired` can be proven absent by the existing `wireless` conflict pair;
  form factor / product type / brand use their known single-valued slots.
- A present excluded attribute is a violation.
- No trusted absence evidence is `hard_unknown`, not compliance.  `_score_and_inspect` removes
  hard-unknown rows before the Gate; `DecisionGate.check` also rejects an all-hard-unknown pool.
- `soft_preferences` remain ranking-only and do not bypass hard constraints.
- A current hard exclusion overrides a conflicting positive M9 preference in
  `effective_preferences`; the M9 history row is not deleted or superseded.
- If no candidate can verify the exclusion, the Gate returns
  `NO_VERIFIED_CANDIDATE_IN_SCOPE` (`current_task_constraint_unknown`); the BFF now treats
  `no_verified_candidate_in_scope` as a known durable refusal, not AMBIGUOUS.
- Budget Gate behavior is unchanged and still runs as an additional hard constraint.

## 4. Files changed

- `web_demo/reply_executor.py`: capture/pass constraints; include in durable result; known
  no-candidate refusal.
- `recommendation/mcp/server.py`: accept and decode `task_constraints`; pass to executor and
  ASK continuation; echo in response.
- `recommendation/preference/current_task_constraints.py`: `hard_unknown` / `hard_verified`
  evidence status; known-absence helper using existing conflict vocabulary.
- `experiments/p14c0/electronics_executor.py`: exclude hard-unknown rows; audit counters;
  final defense check.
- `recommendation/preference/decision_gate.py`: bounded refusal for all-hard-unknown pools.
- `web_demo/reply_assembly.py`: expose durable `task_constraints` in the reply response.

## 5. Tests

- `tests/test_task_constraints_execution.py`: 3 CPU executor tests:
  wired violation + unknown evidence excluded; M9 positive conflict overridden; all
  unknown/violating = safe refusal.
- `tests/test_web_task_constraints_e2e.py`: 2 real FastAPI + MCP stdio + BM25 E2E:
  - vague headphones → ASK → MCP restart → "Please avoid wired headphones." → wireless
    product finalized, wired absent, replay zero-MCP;
  - "推荐一台笔记本" → ASK1 → "游戏，avoid wired" → ASK2 → MCP restart → "8000 元";
    constraints present in the post-restart recommend call; no violating product.
- `tests/test_qwen_clarification_consistency.py`: updated shared semantics for
  hard-unknown vs verified absence; Gate all-unknown refusal.
- `tests/test_web_multiturn_ask.py`: BFF passes constraints into `recommend_products` and
  exposes them in the durable/response projection.

Run in this round:

```text
62 passed (constraints/memory/coordinator unit set)
2 passed (real task-constraint E2E)
18 passed (warm/cold real E2E)
3 passed (M4 multi-turn, requirement-update, demo-price real E2E)
83 passed (persistence/continuation/safety + reply executor/multi-turn)
95 passed (documentation checks)
```

## 6. Remaining limits

- Chinese negation/avoidance is still not extracted by the M9 rule extractor; this fix does
  not add a Chinese NLP system.
- Absence evidence is limited to the existing conflict pairs and single-valued slots; an
  arbitrary attribute with no catalogue evidence remains `hard_unknown` and therefore refuses,
  by design.
- The candidate pool is still bounded by top-20 retrieval; the constraint cannot retrieve a
  product outside it.
- No quality/human-label claim: this is an engineering safety fix.

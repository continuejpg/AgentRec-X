# M4 — CPU/BM25 Multi-Round ASK and Durable Recovery

> **Status: implemented and verified at the M4 commit (2026-11-22).**
> Scope is CPU/BM25 only.  No Qwen, no GPU, no new retrieval/recommendation algorithm and no
> Category-scoped BM25 integration.  Engineering correctness is verified; recommendation
> quality remains **NOT ESTABLISHED** (the M3 label limitation is unchanged).

## 1. What was broken

The single-round path is:

```text
ASK1 -> Reply1 (M9 write) -> get_active_preferences -> recommend_products
```

When `recommend_products` returned `awaiting_clarification` after Reply1, the coordinator had
no state transition for it:

- `web_demo.reply_result_contract.build_reply_payload` rejected the result (a reply required a
  finalized product);
- the assembler only accepted `finalized`;
- there was no Store operation that could consume the answered pending and create a new one in
  the same transaction;
- the executor therefore marked the request `AMBIGUOUS` and the HTTP layer returned
  **503 `clarification_reply_ambiguous`**.

That was a *known* next question, not an uncertain external outcome.  M4 adds the missing
state transition without weakening the genuine `AMBIGUOUS` path.

## 2. Implemented behavior

### 2.1 Real MCP next ASK

`recommendation/preference/clarification_policy.py` adds one small deterministic task-slot
policy (only for an explicitly extracted `laptop` category):

1. missing use case -> ask office/gaming;
2. missing budget -> ask the budget;
3. both present -> recommend normally.

`recommend_products` evaluates the merged task requirements.  On a **resumed reply**
(`task_requirements` supplied by the coordinator) it returns a real
`status="awaiting_clarification"` result containing:

- a newly issued `clarification_id`;
- the question and reason;
- the merged `task_requirements`;
- the exact task snapshot;
- a JSON-only continuation with session/task/expiry/TTL.

It does not return an error string, and it never fabricates a product.  The initial chat path
still calls `request_preference_clarification`; that tool now uses the same policy, so ASK1
and ASK2 are produced by the same business rule.

### 2.2 Atomic Web commit

`SQLiteWebStateStore.finalize_reply_with_clarification` commits, in one
`BEGIN IMMEDIATE` transaction:

- the Reply1 turn (identity issued at ASK1);
- old Pending `OPEN -> CONSUMED`;
- a new Pending and a new reply identity for ASK2, with `turn_id = Reply1 turn`;
- the stamped continuation (session/profile/task/pending/clarification/reset generation/TTL);
- Task `WAITING_FOR_USER` with merged requirements and `task_version + 1`;
- Request `COMPLETED` with `outcome="awaiting_clarification"` and the new ids.

A crash before commit leaves the old pending `OPEN`; a crash after commit leaves exactly one
`OPEN` ASK2.  There is no intermediate state with two open questions or a consumed reply.

### 2.3 Coordinator and API

The coordinator now validates the next ASK (new id, non-empty question, restorable
continuation, future expiry, different from the answered id) **before** checkpoint.  It then
checkpoints the complete result and finalizes it.  A checkpointed-but-not-finalized ASK2 is
finalized from SQLite with **zero MCP calls**.

HTTP success (200) for Reply1 carries:

```json
{
  "status": "awaiting_clarification",
  "recommendation": null,
  "clarification": {
    "clarification_id": "clar_...",
    "question": "...",
    "reason": "missing_laptop_budget",
    "expires_in_seconds": 1800.0,
    "memory_write_enabled": true
  },
  "message": "One more task detail is required ..."
}
```

The existing single-round final/recommendation/refusal response shape is unchanged.  The
additive `status="awaiting_clarification"` field is present only on the multi-round path;
`clarification` keeps its existing `null` value for a final reply.  The frontend already
consumes `response.clarification`; no rewrite was required.

### 2.4 Safety properties

- Reply1 is replayed from its committed result: it returns the **same ASK2** and performs zero
  MCP calls; it never creates ASK3.
- A concurrent duplicate reply has one claim winner; only that executor performs the M9 write
  and opens one ASK2.
- Reset cancels ASK2; the old pending/request cannot be resumed or replayed.
- `submit_clarification_reply` / `recommend_products` failure remains `AMBIGUOUS` (503, no
  automatic retry).  Only a complete, valid next ASK becomes durable.
- Each ASK gets a new in-process handle; after a child restart, the pending's continuation is
  validated and restored in the new child.  Old handles are never reused.
- Warm/Cold single-round resume, M9 idempotency, Budget Gate, replay and the request ledger are
  unchanged.

## 3. Verified scenario

`tests/test_web_multiturn_ask_e2e.py` uses the real FastAPI app, real MCP stdio child, real
CPU/BM25 C4 catalogue, isolated SQLite and `demo_historical_fx` pricing:

```text
"推荐一台笔记本"
  -> ASK1 "这台笔记本主要用来办公还是游戏？"
  -> real child restart
  -> "游戏"
  -> ASK2 "您的预算是多少？"       (HTTP 200 awaiting_clarification)
  -> real child restart
  -> "8000 元"
  -> finalized: MSI GF65, $899.00 -> ¥6,293.00 CNY (fixed demo 7.00 rate)
```

Verified in that run:

- ASK1 and ASK2 clarification ids, pending ids and reply-turn ids are all different;
- ASK2's `turn_id` is Reply1's turn id;
- ASK2's continuation is stamped with the new pending/session/task/generation;
- the task stays `WAITING_FOR_USER` with `category=laptop`, `use_cases=[gaming]`,
  `budget={max_amount:8000, currency:CNY}`;
- final Request/Pending/Task/Turn states agree and the final replay is zero-MCP;
- Reply1 replay returns ASK2 and creates no third pending.

### 3.1 M3-B compatibility check

The real 15-case Golden runner was re-run on M4 code as `m4-compat-01`:

| Metric | M4 run | M3 archive |
|---|---:|---:|
| Task outcome | **13 / 15** | 13 / 15 |
| Requirement fields | **45 / 45** | 45 / 45 |
| Budget violations | **0 / 12** | 0 / 12 |
| Raw-query P@5 | **1 / 15** | 1 / 15 |

No runner crash and no metric drift.  The two former 503 cases now return HTTP 200 with a real
next ASK; the single-round runner records them as `empty` (missing recommendation), so the two
task failures remain 2 but their failure type changes from `execution_error` to
`missing_recommendation`.  The M3 archive numbers are not rewritten.

## 4. Tests

- `tests/test_web_multiturn_ask.py`: 10 unit/regression tests (policy, atomic Store commit,
  replay, checkpoint finalize, reset, invalid-next-ASK rollback, AMBIGUOUS, concurrency).
- `tests/test_web_multiturn_ask_e2e.py`: 1 real two-restart E2E.
- Existing M1 Warm/Cold, requirement-update, demo-price, request-ledger and reply-executor
  suites are re-run for regression (results recorded in the M4 completion report).

## 5. Explicit limits

- Qwen persistent pending is still unsupported (409), unchanged.
- Multi-round is supported for `cpu_live`; there is no promise of an arbitrary number of
  rounds, and no new UI redesign.
- The laptop-slot policy is a deterministic rule extension, not a learned dialogue policy.
- Golden Dataset and Evaluation Runner are not redesigned; the M3 archived numbers remain
  historical.  Two M3-B cases that used to stop at the single-ASK 503 will now expose a real
  next ASK if re-run; this is the intended M4 behavior.
- Web SQLite and M9 SQLite remain separate transaction domains; cross-database exactly-once
  is still not promised.

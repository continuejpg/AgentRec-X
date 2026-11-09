# M5.2 — Interaction Feedback Loop & State Consistency

> **Status: IMPLEMENTED (engineering slice, opt-in).** M5.2 completes the multi-turn
> preference feedback loop on top of M5/M5.1: clarification answers become validated M5
> events, explicit corrections and M9 removals are mapped to M5 belief transitions, the
> trusted candidate/ranking snapshots are connected to `UnifiedDecisionState`, server-owned
> turn identity replaces the content-addressed fallback, and APPLY_CONSTRAINTS now exposes
> one consistent effective-constraint view. Default mode remains `OFF`; Serving V1 and the
> accepted M9 preference-memory schema are unchanged. No fine ranker is trained and no
> recommendation-quality claim is made.

---

## 1. Runtime event flow

The integrated entrypoint is still the multi-turn shopping route, because it is the only
local worktree entrypoint with server-owned `user_key`, `task_id` and turn identity:

```text
POST /v1/demo/sessions/{session}/agent/shopping          (DemoSessionManager.turn())
  -> DemoAgentService.run_shopping_turn(..., turn_id=allocation.turn_id,
                                              idempotency_key=payload.idempotency_key)
       logical turn identity = idempotency_key > server turn id > content fallback
       trusted parsing: constraints, M9 extraction preview, derive_intent
       M5 events:  TASK_CONTEXT_UPDATED
                   USER_PREFERENCE_STATED   (explicit statement / REPLACE / selected option)
                   PREFERENCE_RETRACTED     (accepted M9 REMOVE, resolved selector)
                   USER_CONSTRAINT_STATED   (trusted hard constraints)
       StateManager.apply_event / replay_state
       for_agent / for_ranker / for_constraints
       [APPLY] merged authorized constraints feed the existing AgentConstraintItem inputs
       existing loop -> response (unchanged schema)
       trusted response views -> CANDIDATE_POOL_REFERENCED / RANKING_SNAPSHOT_REFERENCED
                         (opaque references, counts, digests, schema version only)
       bounded diagnostics (process-local, last 16 tasks)
```

Suspended turns (conflict / undecidable) still record the validated facts that caused the
question; a turn that presents recommendations additionally records its trusted candidate
and ranking snapshots. The stateless `POST /v1/demo/agent/recommend` route still has no
stable task identity and therefore does not write M5 events in any mode.

---

## 2. Clarification handling

There are two clarification families in the accepted runtime, and M5.2 treats them
differently on purpose.

### 2.1 Candidate-aware preference question (structured option)

When the accepted preference-construction policy asks a question, the pending state stores
the evidence prompt and its executable options. The user's answer is resolved **only**
through the trusted `resolve_preference_option` path (option id first, exact visible label
otherwise; otherwise refused). M5.2 then records exactly one feedback event:

| Trusted option action | M5 event |
| --- | --- |
| `lower` on a numeric attribute | `USER_PREFERENCE_STATED` with `kind=feature`, `value=<attribute>`, `direction=minimize` |
| `higher` on a numeric attribute | `USER_PREFERENCE_STATED` with `kind=feature`, `value=<attribute>`, `direction=maximize` |
| `value` on color/material/brand | `USER_PREFERENCE_STATED` with the matching kind and the catalogue value |
| `none` / `skip` | **no preference event** (the answer creates no fact; only the turn context is recorded) |
| unresolved / free text not matching an option | refused by the existing resolver before M5.2 runs |

The event is TASK-scoped, actor `user`, with provenance `source_text = <trusted option
label>`. The existing response/rerank/UX is unchanged: M5.2 only records the accepted
feedback and updates the projections. No LLM interpretation is involved, and an ambiguous
or declining answer therefore cannot become an explicit preference.

### 2.2 Research / conflict / undecidable clarifications

Those answers already flow through the trusted resume path (`resolve_research_option`,
pending-state validation, explicit constraints). M5.2 records their validated facts (and
M9 extraction preview) via the same adapter used by normal turns - before the conflict
early-return branch, so a turn that suspends still leaves the budget/constraint facts in
the state. The answer is never re-interpreted by M5.2.

---

## 3. Preference correction and retraction rules

M5.2 maps the accepted M9 extraction to the existing M5 semantics; it does not invent
synchronisation events.

| M9 operation | M5 mapping |
| --- | --- |
| `ADD` explicit preference | `USER_PREFERENCE_STATED` (`scope=task`, actor `user`, provenance span) |
| `REPLACE` with `replaces` | `USER_PREFERENCE_STATED` with `replaces_value`: the corrected task belief is marked `SUPERSEDED` (or retracted if it was an inference) and the new explicit belief is active |
| `REPLACE` without `replaces` | `USER_PREFERENCE_STATED`; the updater applies its documented same-kind/opposite-polarity supersession rule |
| `REMOVE` | `PREFERENCE_RETRACTED`, after the selector is resolved against the current task state |

Retraction resolution is deliberately conservative:

* a kind selector retracts matching active task beliefs of that kind (optionally narrowed
  by value/polarity);
* a value-only selector is resolved **only when exactly one active task belief has that
  value**; zero matches (e.g. a removal of a long-term-only memory entry) or multiple
  distinct kinds are reported as a bounded legacy/M5 disagreement in diagnostics and no
  event is invented;
* retraction only ever touches TASK-scoped beliefs; long-term M5 beliefs, if any, and the
  production M9 memory database are never modified by a task-level retraction;
* supersession/retraction changes lifecycle/status and appends provenance; records are
  never deleted, so replay preserves the full audit trail and a retracted preference is
  absent from every projection used to build the effective state.

The M9 memory service remains the only long-term writer. M5.2 does not migrate the M9
database, does not write M5 reminders into it and does not use a legacy fallback to
resurrect a retracted task preference.

---

## 4. Turn identity and retry semantics

The M5.1 content-addressed turn key is replaced on the HTTP shopping route.

`resolve_turn_identity(task_id, idempotency_key, server_turn_id, fallback_content_key)`
uses this precedence and reports the source in diagnostics:

1. **client `idempotency_key`** (new optional field on `AgentShoppingRequest`) - scoped by
   task id, so a retried logical turn re-emits the same event ids and the append-only
   store's duplicate rule makes the retry a no-op (no duplicate beliefs, snapshots or
   events);
2. **server-owned turn id** (`<session_id>:<sequence>` from `DemoSessionManager.turn`) -
   every real turn gets a new identity, so two distinct turns with identical normalised
   message text never collapse; a transport retry without an idempotency key receives a
   new server turn id (documented fallback: no deduplication is claimed for that retry);
3. **content digest fallback** - only direct service callers/tests that supply neither
   identity; clearly labelled `content_fallback` in diagnostics and never used by the
   HTTP route.

Event ids are derived from `(user_key, task_id, logical turn identity, event type, event
content)`; they are never derived from message content alone.

Guarantees:

* **A.** Same logical turn retried with the same `idempotency_key` -> same event ids,
  event count and state digest; no duplicate rows.
* **B.** Two distinct turns with identical content -> different server turn ids ->
  different events.
* **C.** Two users remain isolated by `user_key`; event ids and the store queries are
  user-scoped.
* **D.** Two tasks of the same user remain isolated by `task_id`; task-local events are
  never replayed into another task.

`request_turn_key` remains as the documented content-addressed fallback for direct callers
only.

---

## 5. Candidate and ranking snapshot contract

After a trusted recommendation response presents candidates, M5.2 registers two existing
M5 reference events - no new event types are needed:

| Event | Payload |
| --- | --- |
| `CANDIDATE_POOL_REFERENCED` | `ref` = `cp_<sha256[:24]>` of the sorted grounded identity set; `source`; `candidate_count`; `digest` = full canonical SHA-256; `snapshot_schema_version = 1` |
| `RANKING_SNAPSHOT_REFERENCED` | `ref` = `rs_<sha256[:24]>` of the presented identity order; `source`; `method` (`presented_order` / `task_preference_rerank`); `candidate_count`; `digest`; `snapshot_schema_version = 1` |

Rules:

* only the trusted pipeline may produce them; the events are actor
  `trusted_component`, built from the response's own `grounded.candidates` and
  `recommendations` views after the loop finished;
* nothing beyond an opaque reference, a source label, a count, a canonical digest and the
  snapshot schema version crosses the boundary - **no product row, no per-item rank, no
  model score, no fused score** is stored in the belief state. The state model also types
  the reference as read-only;
* the events are validated against the same user/task/execution identity as the turn, and
  duplicate identities in a presented ranking are refused;
* snapshot identity is turn-scoped, so a retried turn re-uses the same snapshot ids;
* snapshot failure follows the mode policy: SHADOW records a bounded `snapshot_error` and
  leaves the answer unchanged; APPLY_CONSTRAINTS fails closed; OFF does nothing.

---

## 6. Effective-constraint view (APPLY_CONSTRAINTS consistency)

M5.1's gap - eligibility enforcing a merged constraint that `response.intent.hard` did not
show - is closed by one trusted effective view.

In APPLY_CONSTRAINTS the service builds:

```text
effective = merge_constraint_pairs(trusted_intent_hard, for_constraints(state))
```

and projects that exact ordered set into `response.intent.hard`. Each entry keeps its
provenance through the new `origin` field:

| origin | meaning |
| --- | --- |
| `user_stated` | stated in this turn (request constraint / this-turn extraction) |
| `validated_existing` | validated from an earlier turn and still active (M9 memory entry) |
| `decision_state` | authorized M5 constraint merged by this integration |

The eligibility evaluator consumes exactly the same merged pairs through the existing
`AgentConstraintItem` inputs. A remembered constraint is therefore never misrepresented as
a current-turn user statement, and eligibility/requirements/intent cannot disagree.
`UNKNOWN` remains a distinct three-state verdict; a missing metadata field is never turned
into a violation. In SHADOW and OFF the intent view is exactly the pre-M5.2 view.

---

## 7. Mode behaviour

* **OFF (default).** No state DB opened, no events, no snapshots, no projections, no
  response change, no extra model calls. Existing tests and Serving V1 behaviour are
  unchanged.
* **SHADOW.** Records validated feedback/correction/retraction/snapshot events, updates
  and replays state, produces the three projections and bounded non-sensitive
  diagnostics. The user-visible response stays identical to OFF for the same trusted
  inputs (volatile fields such as timing and run identities aside). A state failure is
  recorded in diagnostics and never changes the answer.
* **APPLY_CONSTRAINTS.** Educational/experimental: merges only authorized explicit M5
  constraints into the existing eligibility inputs, keeps inferred hypotheses out of
  filters, preserves the three-state verdicts and fail-closes on state integration
  errors. No ranking-score change is made.

No new feature gate is introduced in M5.2.

---

## 8. Trust boundaries

1. Only validated trusted facts become events: M9 extraction candidates/removals, trusted
   shopping-intent hard constraints, and trusted clarification-option resolutions. There
   is no adapter entry point for an inference, model text or arbitrary string.
2. Inferred hypotheses still cannot become hard constraints or explicit preferences.
3. Retraction selectors are resolved against the current task state; unmatched/ambiguous
   removals are reported, never guessed.
4. Task-level events never mutate long-term beliefs or the M9 database.
5. Snapshots carry only opaque references/counts/digests and are produced from trusted
   response views; no product identity or score enters the belief state.
6. The decision state never decides the recommendation answer; the accepted loop,
   verifier, grounding and renderer remain authoritative.
7. Default is OFF; a deployment that does not opt in has accepted behaviour.

---

## 9. Concurrency and safety

* One process owns one `ShoppingStateIntegration`/`StateManager`/SQLite event store; the
  manager serialises validate -> append -> replay and the store serialises appends
  (`BEGIN IMMEDIATE`). In-process concurrency is safe.
* **Cross-process writers of the same `AGENTRECX_STATE_DB` remain unsupported and
  untested** (pre-validation is per-replay). Deploy single-process or one DB per process.
* Event ordering, duplicate submission, cross-task/cross-user isolation and
  validation-before-append semantics are inherited unchanged from M5.
* Store failures behave per mode (SHADOW diagnostics-only, APPLY fail-closed).
* No distributed system, event compaction or snapshotting is added in this milestone.

---

## 10. Known limitations

* Only the main planning branch and the candidate-aware preference-answer branch are
  integrated; a research-clarification answer is recorded through the same main-path hook
  while its option resolution remains owned by the accepted resume path.
* Only explicit M9 statements/removals and trusted clarification options are mapped; no
  inferred preference is turned into feedback, and no LLM interpretation is used.
* M9 removals that only exist in long-term memory (not in the M5 task state) are reported
  as bounded disagreements, not mirrored as task events.
* Content digest fallback still exists for direct callers without server/idempotency
  identity; it is documented and never used by the HTTP shopping route.
* Diagnostics are process-local (last 16 tasks) and lost on restart; the event store
  persists.
* Snapshot refs are turn-scoped and reflect what the trusted response presented; they are
  not a general-purpose candidate archive.
* No ranking model/score change and no quality claim.

---

## 11. M6 ranking-integration readiness

M5.2 leaves M6 (fine-ranking integration) with what it needs and nothing more:

* a replayable, versioned `UnifiedDecisionState` with explicit correction/retraction
  semantics and full provenance;
* `for_ranker(state)` typed features with scope/lifecycle/origin/ordinal strength, and
  candidate/ranking snapshot references joined to the exact presented order;
* a stable logical-turn identity so feedback can be joined to presentations without
  content collisions;
* a trusted candidate/ranking snapshot digest as the join key for future offline training
  data, without storing product rows or scores in the state.

M6 must still add its own frozen evaluation protocol and a soft-signal rerank hook behind a
separate opt-in gate; no ranking-quality claim can be derived from M5.2.

---

## 12. Verification

Executed commands (Python 3.10 test venv; the repository `.venv` is a Linux venv):

```bash
python -m pytest -q tests/test_m5_2_state_adapter.py          # light, no torch
python -m pytest -q tests/test_m5_2_integration.py
python -m pytest -q tests/test_m5_2_acceptance.py
python -m experiments.m5_2_feedback_loop_smoke
```

### 12.1 Executed results (this worktree, 2026-10-30)

| Check | Result |
| --- | --- |
| `tests/test_m5_2_state_adapter.py` (no-torch interpreter) | **37 passed** |
| `tests/test_m5_2_contract.py` (Lead contract tests) | **4 passed** |
| `tests/test_m5_2_integration.py` (3.10 venv) | **21 passed** |
| `tests/test_m5_2_acceptance.py` (3.10 venv, black-box) | **21 passed** |
| Combined M5 + M5.1 + M5.2 gate (engine, adapter, integration, acceptance suites) | **294 passed** |
| Torch-free M5 engine/contract/adapters | **221 passed, 1 skipped** (accepted control-plane enum parity needs the model stack; it passes in the 3.10 venv) |
| Accepted regression batch: shopping route/web/intent, demo agent/sessions/multiturn/control-plane, durable checkpoints, operational metrics, M9 memory store/service/query adapter, preference construction, docs | **392 passed** |
| `experiments/m5_2_feedback_loop_smoke.py` | **SMOKE: PASS, 55/55 checks**, exit 0 |
| `experiments/m5_1_shopping_smoke.py` (M5.2-superset expectations) | **SMOKE: PASS, 27/27 checks** |
| `experiments/m5_unified_state_smoke.py` | **SMOKE: PASS, 37/37 checks** |

Six-turn smoke evidence: 20 events
`{task_context_updated: 5, user_preference_stated: 4, user_constraint_stated: 4,
preference_retracted: 1, candidate_pool_referenced: 3, ranking_snapshot_referenced: 3}`;
state digest `6580c1fc…` is identical from incremental application and replay after
reopening SQLite; the idempotency-keyed retry appends nothing (`turn_identity_source =
idempotency_key`); snapshot refs `cp_b7ac39e6…` / `rs_2c40ae98…` carry
`snapshot_schema_version=1`; APPLY's `intent.hard` carries `origin=decision_state` for the
M5-authorized constraint and agrees with `eligibility.requirements`. The smoke is
synthetic and makes no quality claim.

The M5.1 files that enumerated the exact event vocabulary were updated minimally to the
M5.2 superset (snapshot events and keyed retries); their parity, eligibility, UNKNOWN and
isolation assertions were not weakened.

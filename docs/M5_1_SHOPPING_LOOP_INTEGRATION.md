# M5.1 — Unified Decision State Integration with the Shopping Agent

> **Status: IMPLEMENTED (engineering slice, opt-in).** M5.1 connects the accepted
> multi-turn shopping Agent runtime with the M5-MVP unified decision state engine. The
> default mode is `OFF`: existing Serving V1 / shopping behaviour is byte-for-byte
> unchanged. The state engine is **not** reimplemented, no recommendation model is
> changed, no ranking model is trained, and no M9 memory database is migrated.

---

## 1. Actual integration entrypoint

The local worktree has several agent-ish surfaces. M5.1 deliberately chooses the
smallest one with sufficient identity:

| Surface | Code path | Stable task identity? | M5.1 |
| --- | --- | --- | --- |
| `POST /v1/recommend` | `RecommendationEngine` | no session at all | untouched |
| `POST /v1/demo/sessions/{id}/chat` | `AgentGraph` (DAG, chat contract) | session, but the chat route writes M9 memory through the graph | untouched |
| `POST /v1/demo/agent/recommend` | `DemoAgentService.run(payload)` | **no** (stateless, optional `user_key`, no task id) | untouched; events declined |
| `POST /v1/demo/sessions/{id}/agent/shopping` → `DemoAgentService.run_shopping_turn(...)` | control-plane shopping loop | **yes**: server-owned `user_key`, `task_id = demo session id`, server turn allocation | **integrated** |

`run_shopping_turn` receives:

* `session_id` = the M9 preference-memory namespace (`user_key`) of the demo session;
* `task_id` = the concrete demo session id, used as the stable `task_id`;
* `payload` = the validated `AgentRecommendRequest` built by the route from
  server-owned history;
* internally, a server-owned `turn_id`.

The state integration therefore happens **inside `run_shopping_turn` only**, after the
existing trusted parsing produced a `ShoppingIntent` and the accepted M9 extractor
produced a `PreferenceExtraction` preview, and before the existing loop runs. No new
route, no new agent framework and no new recommendation path is introduced.

Why not the stateless `run()`: it has no stable task identity (an optional `user_key` is
not a task id). Creating pseudo-tasks from request content would silently merge unrelated
shopping tasks; M5.1 declines persistence there instead. In `SHADOW`/`APPLY_CONSTRAINTS`
the stateless endpoint is therefore identical to `OFF` and writes no state events.

---

## 2. State event lifecycle

```text
POST /v1/demo/sessions/{session}/agent/shopping
  -> DemoSessionManager.turn()            server-owned turn allocation
  -> DemoAgentService.run_shopping_turn()
       existing trusted parsing:
         payload.constraints  -> explicit hard constraints
         M9 extractor         -> PreferenceExtraction (explicit preferences/removals)
         derive_intent(...)   -> ShoppingIntent (hard / soft / conflicts)
       M5.1 adapter (only when mode != OFF):
         recommendation/state/shopping_adapter.py
           request_turn_key(...)          content-addressed logical turn key
           build_turn_events(...)         TASK_CONTEXT_UPDATED
                                          USER_PREFERENCE_STATED   (explicit only)
                                          USER_CONSTRAINT_STATED   (trusted hard only)
         recommendation/demo/state_integration.py
           ShoppingStateIntegration.integrate_turn(...)
             StateManager.load_or_create / apply_event / replay_state
             for_agent / for_ranker / for_constraints
       APPLY_CONSTRAINTS only:
         authorized constraints merged into the existing AgentConstraintItem inputs
       existing loop (unchanged) -> same rendering / eligibility / response
```

Only three M5 event types are emitted by M5.1: `TASK_CONTEXT_UPDATED`,
`USER_PREFERENCE_STATED`, `USER_CONSTRAINT_STATED`. No hypothesis event, no knowledge
event, no observation event and no candidate/ranking reference is emitted: the shopping
runtime does not yet produce those in a form M5.1 can trust. Explicit corrections use the
accepted M9 extractor's `REPLACE` mode → M5 `replaces_value`; a retraction (M9 `REMOVE`)
is **not** converted in M5.1 (no retraction event is emitted), which is documented as a
limitation rather than silently dropped.

The integration point is the main planning branch of `run_shopping_turn`. Clarification
answer branches (`_answer_preference`) and conflict suspension branches are unchanged and
do not emit M5 events in M5.1. That keeps the patch narrow: a suspended turn keeps its
existing semantics, and the next planning turn is the one that records validated facts.

---

## 3. Identity and persistence semantics

| Field | Value | Owner |
| --- | --- | --- |
| `user_key` | demo session's M9 memory namespace | server (`DemoSessionManager`) |
| `session_id` | concrete demo session id (also the event envelope `session_id`) | server |
| `task_id` | concrete demo session id | server |
| `turn_key` | `request_turn_key(message, k, sources, constraints, clarification_id)` | M5.1 adapter, content-addressed |
| event id | deterministic from `(user_key, task_id, turn_key, event type, kind, value, polarity)` | adapter |

* `task_id` is never derived from `user_key` alone: several demo sessions may share a
  memory namespace, and each concrete session is its own task.
* All M5 events emitted by M5.1 are **TASK-scoped**. M5.1 never writes long-term state:
  the M9 preference-memory service remains the only long-term writer, so no production
  memory is migrated and no task preference silently becomes durable memory.
* `turn_key` is content-addressed so a retried logical turn re-emits the same event ids;
  the M5 store's duplicate rule makes the retry a no-op. Two genuinely different turns with
  identical normalised content collapse to one logical turn — an explicit limitation.
* State is derived: `replay_state(user_key, task_id)` reconstructs it from the store. The
  state database is a separate SQLite file (`AGENTRECX_STATE_DB`, default
  `artifacts/demo/decision_state.sqlite3`), never the M9 memory DB and never the trusted
  purchase-history data used by SASRec.
* Session reset deletes the process-local diagnostics entry but does **not** delete the
  append-only event history (no delete API exists by design); a new session has new
  server-owned identities, so old events remain isolated.

---

## 4. Modes

`AGENTRECX_STATE_MODE` ∈ `off` (default) / `shadow` / `apply_constraints`. Unknown values
fail closed at composition time (`503 demo_unavailable`); there is no silent fallback.

### OFF (default)

* Existing recommendation behaviour runs exactly as before.
* No M5 store is opened, no events are written, no projections are used.
* Response structure, ordering, scores, grounding, eligibility and memory behaviour are
  unchanged.

### SHADOW

* Validated explicit facts are converted to M5 events and appended; state is replayed and
  `for_agent` / `for_ranker` / `for_constraints` are produced.
* M5 **cannot** change candidate eligibility, ranking, user-facing recommendations or M9
  memory. The response is identical to OFF for the same request.
* A bounded, non-sensitive diagnostic record is kept per task (last 16 tasks): mode, event
  count, state digest, projection versions/counts, hard-constraint kinds, inferred count,
  suppressed count, and a kind-level comparison of the M5 authorized constraints against
  the existing trusted hard-constraint interpretation. No raw message, no source span, no
  user key is stored in diagnostics.
* A state failure (unavailable store, malformed event, rejected transition) is recorded as
  a bounded error code in the diagnostics; the request continues with OFF-identical
  behaviour. SHADOW is observational, so the failure is surfaced rather than silently
  ignored, but it never changes the answer.

### APPLY_CONSTRAINTS (experimental, opt-in)

* Uses `for_constraints(state)` — only user-authorized explicit hard constraints, with
  task-over-long-term precedence applied — and merges them into the **existing**
  `AgentConstraintItem` inputs (existing trusted order first, de-duplicated by
  kind + case-folded value).
* The accepted `CandidateEligibilityEvaluator` / `GroundedReasoner` path is reused
  unchanged. `MATCH` / `VIOLATION` / `UNKNOWN` semantics are preserved; `UNKNOWN` is never
  rewritten as a violation; inferred hypotheses never appear in `for_constraints` and
  therefore cannot hard-filter candidates; grounding and identity checks are untouched.
* A state-engine failure is **fail-closed**: the service raises a deterministic
  `DemoAgentError(code="state_integration_failed")` mapped to `502`, instead of silently
  falling back to legacy constraints.

---

## 5. Existing components reused (nothing replaced)

| Component | Reused as |
| --- | --- |
| `recommendation/state/` M5 engine | `StateEvent`, `SQLiteEventStore`, `StateManager`, `BeliefUpdater`, `StateProjector`, `UnifiedDecisionState` — unchanged |
| `recommendation.state.shopping_adapter` (new, light) | validated explicit facts → M5 events; content-addressed turn key; authorized constraint pairs |
| `DemoAgentService.run_shopping_turn` | existing entrypoint; only an additive hook before the loop |
| `ShoppingIntent` / `derive_intent` / M9 `RuleBasedPreferenceExtractor` | the **only** accepted interpretation path used to decide what is explicit/hard |
| `AgentConstraintItem` / `_task_state` / `CandidateEligibilityEvaluator` / `GroundedReasoner` | APPLY mode feeds the same trusted constraint inputs; no new eligibility code |
| `DemoSessionManager` | server-owned `user_key` / session id / turn allocation |
| `DemoRuntime` | process-scoped composition; new integration object is composed once and shared |
| `AgentRecommendResponse` | untouched wire schema; no M5 field is added to the response |
| M9 `PreferenceMemoryService` + SQLite schema | untouched; still the only long-term preference writer and never migrated |

---

## 6. Trust boundaries

1. **Only validated facts become events.** The adapter accepts `ExplicitPreference`
   (from the accepted M9 extractor) and `ExplicitConstraint` (from the existing trusted
   shopping intent's hard set). It has no API for an inference, model text or arbitrary
   string. `intent.soft` entries are never converted to constraints; the adapter has no
   function that could do so.
2. **Hard constraints stay user-authorized.** `USER_CONSTRAINT_STATED` is emitted only
   from the trusted hard set; APPLY consumes only `for_constraints`, which structurally
   contains only user-explicit constraints.
3. **No new long-term writes.** All events are TASK-scoped; M9 memory remains the only
   durable writer (no migration).
4. **Identity is server-owned.** The browser supplies neither history, `user_key` nor
   `task_id` on the shopping route; the event identity is built from server-owned values.
5. **Raw text is bounded/redacted** by the M5 payload models; diagnostics contain no raw
   text or user key.
6. **State never decides truth.** The state engine reports projections; the existing
   loop, verifier, grounding and renderer stay authoritative for the answer.
7. **Default is OFF.** A deployment that does not opt in has exactly the accepted
   behaviour; the state DB is not even opened.

---

## 7. Ranking projection behaviour

`for_ranker(state)` is called in SHADOW and APPLY modes and verified through diagnostics
(`ranker_signal_version`, feature/priority counts, soft flags). It is **not** injected
into any recommendation score in M5.1:

* no new fine ranker is trained or loaded;
* no uncalibrated soft-preference weight is added to SASRec/Two-Tower/BM25 or RRF scores;
* the accepted response ordering is unchanged in OFF and SHADOW modes;
* APPLY changes only which candidates are *eligible* (via the existing evaluator), never
  the scoring model or the tie-breaking rules.

`for_ranker` output is therefore used only for feature-contract verification, diagnostics,
and future ranking-integration tests. **No ranking-quality improvement is claimed.**

---

## 8. Concurrency boundary

* One process composes one `ShoppingStateIntegration`, hence one `StateManager` and one
  SQLite event store for the process.
* In-process, `StateManager.apply_event` serialises validate → append → replay under an
  RLock, and the store serialises appends with a lock plus `BEGIN IMMEDIATE`. Concurrent
  requests in one process are therefore safe.
* **Cross-process writers are not supported and not tested.** Two processes pointing at
  the same `AGENTRECX_STATE_DB` would each pre-validate against their own replay and could
  append an event that is invalid for the other's view; the append-only store would then
  fail later replays. M5.1 must be deployed single-process (the demo runtime already is),
  or with one database per process. This is a documented limitation, not a claim.
* M5.1 emits no long-term events, so the cross-task long-term event race M5-MVP warned
  about cannot arise from this integration.

---

## 9. Verification

Executed commands (Python 3.10 test venv with the accepted dependency closure; the
repository `.venv` is a Linux venv and is not used on Windows):

```bash
python -m pytest -q tests/test_m5_1_state_adapter.py          # light, no torch
python -m pytest -q tests/test_m5_1_shopping_integration.py
python -m pytest -q tests/test_m5_1_acceptance.py
python -m experiments.m5_1_shopping_smoke
```

### 9.1 Executed results (this worktree, 2026-10-30)

| Check | Result |
| --- | --- |
| `tests/test_m5_1_state_adapter.py` (no-torch interpreter) | **38 passed** |
| `tests/test_m5_1_shopping_integration.py` (3.10 test venv) | **16 passed** |
| `tests/test_m5_1_acceptance.py` (3.10 test venv) | **14 passed** (requirements A–E) |
| `experiments/m5_1_shopping_smoke.py` | **SMOKE: PASS, 27/27 checks**, exit 0 |
| M5 engine regression `tests/test_state_*.py` + `tests/test_m5_state_acceptance.py` | **142 passed** (the enum-parity test also runs now that the model stack is present) |
| Combined M5 + M5.1 gate (M5 engine + integration + acceptance suites) | **172 passed** |
| Existing regression batch: shopping route/web, agent API, demo sessions, multiturn, control plane, durable checkpoints, operational metrics, M9 memory store/service/query adapter, docs | **358 passed** |

Smoke evidence: SHADOW state digest
`461ccb5689d8c94ecbd59849d5e75467d9f579c7bd25555983cb20cd7f937292`; the same digest is
reproduced by replay after reopening the store. APPLY seeded-constraint effect:
recommendations `['cand-red', 'cand-blue', 'cand-black']` → `['cand-red', 'cand-black']`
through the existing eligibility evaluator, with a three-state assessment sum equal to the
number of assessments. The smoke is fully synthetic and asserts only engineering
behaviour; it makes no quality claim.

---

## 10. Known limitations

* Only the main planning branch of `run_shopping_turn` is integrated; clarification
  answers and conflict suspensions do not emit M5 events yet.
* Only three event types are emitted; no hypothesis, knowledge, observation, candidate or
  ranking-reference events in M5.1.
* M9 `REMOVE` (retraction) is not mapped to an M5 retraction; explicit corrections are
  mapped only through the extractor's `REPLACE` mode.
* `turn_key` is content-addressed: two different turns whose normalised requests are
  identical collapse. A server-owned stable turn id would be a future refinement.
  **Superseded by M5.2** on the HTTP shopping route: the route now supplies the
  server-owned turn id plus an optional client `idempotency_key`, and the content digest
  remains only as a direct-caller fallback (see
  [`M5_2_INTERACTION_FEEDBACK_LOOP.md`](M5_2_INTERACTION_FEEDBACK_LOOP.md) §4).
* **APPLY_CONSTRAINTS view consistency:** the response's `intent.hard` view is derived from
  the request body before the M5 merge. The merged M5-only constraints are enforced by
  eligibility (and shown in `eligibility.requirements`), but they may not appear in the
  response `intent.hard` view. This is an experimental-mode observation, documented rather
  than hidden; rebuilding the intent view after the merge is a later refinement.
* APPLY's fail-closed error surfaces as the existing generic `502 agent_failed`; the
  bounded state failure code is kept in the process-local diagnostics.
* SHADOW diagnostics are process-local (bounded, last 16 tasks) and not exposed on the
  HTTP wire schema; they are reachable through the composed service for tests/operations
  and are lost on restart (the event store itself persists).
* Cross-process store usage is unsupported (see §8).
* No ranking model integration, no ranking-quality claim, no policy/ASK/COMPARE/EXPLAIN
  changes, no M9 migration.
* The stateless agent route remains outside M5.1 because it has no stable task identity.

---

## 11. Next step toward preference-aware ranking

1. Add candidate/ranking-reference events when the trusted candidate plane freezes a
   window, so `for_ranker` can be joined to what was actually shown without storing
   product identity or scores in the decision state.
2. Feed `for_ranker` into a **typed soft-signal rerank hook** behind its own feature gate,
   with the accepted frozen evaluation protocol and a paired, pre-registered offline
   experiment; only then may a ranking model consume the projection.
3. Extend the adapter to clarification-answer turns and M9 retractions, and replace
   content-addressed turn keys with server-owned turn ids once the wire contract carries
   one.
4. Keep M5.1's OFF default and the M5 state engine as the single source of state truth.

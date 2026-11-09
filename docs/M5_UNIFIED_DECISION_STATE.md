# M5-MVP — Unified Decision State & Belief Management

> **Status: IMPLEMENTED (engineering slice).** This document describes the M5-MVP state
> infrastructure of the AgentRec-X 3.0 upgrade: a versioned unified decision state driven
> by an append-only event store and a deterministic belief updater. It is **not** a new
> recommendation model, a new Agent policy, or a memory-quality claim. Serving V1 and the
> accepted M9 preference memory are unchanged.

---

## 1. Scope and position in the architecture

M5-MVP implements only the state infrastructure below. The decision Agent and the fine
ranker are **consumers** of projections; they are not changed in this milestone.

```text
User interaction
    -> Event Store                  append-only, versioned, user/task scoped
    -> Belief Updater               deterministic event -> state reduction
    -> Unified Decision State       typed, immutable, versioned
    -> State Projector              for_agent / for_ranker / for_constraints
    -> Decision Agent / Recommendation Engine   (future consumer, unchanged here)
```

Deliberately out of scope: a new recommendation model, a new Agent policy, SASRec /
Two-Tower / BM25 training changes, production memory migration, a learned fine ranker,
and any LLM call. The event store is **not** the trusted chronological purchase history
SASRec consumes; that data stays application-owned and is never written here.

The code lives in `recommendation/state/`:

| File | Responsibility |
| --- | --- |
| `schemas.py` | shared enums, bounds, validation/redaction helpers, deterministic ids |
| `errors.py` | typed rejection errors |
| `events.py` | typed event payloads, event envelope, actor authority registry |
| `decision_state.py` | `UnifiedDecisionState` and its sub-models, deterministic ids |
| `event_store.py` | append-only SQLite event store with transaction/idempotency semantics |
| `belief_updater.py` | deterministic transition function and transition rules |
| `state_projector.py` | `for_agent` / `for_ranker` / `for_constraints` projections |
| `state_manager.py` | small orchestration API over store + reducer |
| `legacy_adapter.py` | read-only bridge from the accepted M9 memory snapshot |
| `README.md` | package-level contract summary |

### 1.1 Documentation versus code

`recommendation/control/task_state.py` is the accepted source of the scope/lifecycle
semantics, and this milestone reuses those semantics. It is **not imported** by
`recommendation/state/`, because importing `recommendation.control.task_state` executes
`recommendation/control/__init__.py`, which imports the recommendation tool chain and
therefore PyTorch. The state layer must be importable, replayable and smoke-testable with
no model stack. The two enums (`PreferenceScope`, `PreferenceLifecycle`) are therefore
re-declared value-for-value in `recommendation/state/schemas.py`, and
`tests/test_state_schemas.py` pins value/property parity against the accepted module
whenever it is importable. Preference kind/polarity/status, value bounds and the
credential check are imported directly from `recommendation.memory.schemas`; they are not
duplicated.

No other documentation/code conflict was found: the accepted code is treated as
implementation truth, and this document describes what was implemented.

---

## 2. State schema

`UnifiedDecisionState` (`recommendation/state/decision_state.py`) is a frozen Pydantic
model. Immutability is structural: a transition returns a new state; a consumer cannot
patch a belief in place.

| Field | Type | Meaning |
| --- | --- | --- |
| `state_version` | `int` | state contract version (`STATE_SCHEMA_VERSION = 1`) |
| `user_key` | `str` | owning user; every operation is scoped by it |
| `session_id` | `str \| None` | last session that wrote a task-scoped event |
| `task_id` | `str` | the task this state belongs to |
| `preference_beliefs` | `tuple[PreferenceBelief, ...]` | current beliefs, first-appearance order |
| `preference_priorities` | `tuple[FeaturePriority, ...]` | relative feature importance statements |
| `knowledge_beliefs` | `tuple[KnowledgeBelief, ...]` | attribute-understanding estimates |
| `task_context` | `TaskContext` | intent label + bounded/redacted request text |
| `hard_constraints` | `tuple[DecisionConstraint, ...]` | authorized feasibility constraints |
| `observations` | `tuple[DecisionObservation, ...]` | comparison observations (evidence only) |
| `candidate_pool_ref` | `CandidatePoolRef \| None` | read-only trusted candidate-pool reference |
| `ranking_snapshot_ref` | `RankingSnapshotRef \| None` | read-only trusted ranking-snapshot reference |
| `interaction` | `InteractionMetadata` | deterministic counters and last-applied event marker |

### 2.1 `PreferenceBelief`

| Field | Meaning |
| --- | --- |
| `belief_id` | deterministic id of the identity `(scope, kind, value, polarity)` |
| `kind` | accepted `PreferenceKind` (category/feature/brand/price_max/...) |
| `value` | cleaned value, M9 bounds (`MAX_VALUE_LENGTH`), credentials refused |
| `polarity` | accepted `PreferencePolarity` (`prefer` / `avoid`) |
| `direction` | optional `minimize` / `maximize` / `exact` |
| `lifecycle` | `INFERRED_HYPOTHESIS` / `EXPLICIT` / `CONFIRMED` / `RETRACTED` |
| `status` | accepted `PreferenceStatus`: `ACTIVE` / `SUPERSEDED` / `REMOVED` |
| `scope` | `SESSION` / `TASK` / `LONG_TERM` |
| `strength` | ordinal `WEAK` (rank 1) / `MODERATE` (rank 2) / `STRONG` (rank 3) — **not** a probability |
| `origin` | `USER_STATED` / `USER_CONFIRMED` / `SYSTEM_INFERRED` / `LEGACY_MEMORY` |
| `supporting_event_ids` | ordered, de-duplicated provenance |
| `last_event_id` | last event that changed this record (audit) |
| `legacy_memory_id` | set only by the read-only M9 adapter |

Rules:

* `INFERRED_HYPOTHESIS` is **soft evidence only**: it may affect a soft ranking signal or
  trigger a clarification; it may never become a hard constraint and never becomes a
  durable explicit preference silently.
* Only `EXPLICIT` / `CONFIRMED` may later be proposed for durable memory
  (`may_be_committed` reuses the accepted M9 rule).
* `strength` is discrete and ordinal. A system inference is `WEAK` by default and may be
  `MODERATE` only with a named trusted basis; `STRONG` is reserved for user-stated /
  user-confirmed beliefs. No confidence float exists anywhere in the state.
* Identity is `(scope, kind, normalised value, polarity)`. Updating a belief keeps its id
  and tuple position; retraction/supersession changes `lifecycle`/`status` and keeps the
  record for provenance. Nothing is deleted.

### 2.2 `KnowledgeBelief`

| Field | Meaning |
| --- | --- |
| `knowledge_id` | deterministic id from `(user_key, attribute)` |
| `attribute` | cleaned attribute label |
| `status` | `UNKNOWN` / `NEEDS_EXPLANATION` / `EXPLANATION_PROVIDED` / `UNDERSTANDING_DEMONSTRATED` |
| `supporting_event_ids` | provenance |
| `explanation_event_ids` | which "explanation shown" events touched this attribute |
| `last_event_id` | audit |

`EXPLANATION_PROVIDED` means an explanation was **displayed**. It does not mean the user
understood it. Only a `USER_UNDERSTANDING_DEMONSTRATED` event (which carries the user's own
evidence span) may produce `UNDERSTANDING_DEMONSTRATED`. A later explanation request or
display never silently downgrades a demonstrated understanding.

### 2.3 `DecisionConstraint`

A hard constraint is a feasibility statement, not a preference. It carries `kind`,
`value`, `polarity`, `scope`, `authority`, `status` and provenance. Only
`ConstraintAuthority.USER_EXPLICIT` is in `HARD_CONSTRAINT_AUTHORITIES`; only the
`USER_CONSTRAINT_STATED` event creates one. An inference has no field or event through
which it could create a constraint, which is what structurally prevents
"inferred preference becomes a filter".

A confirmed preference is a ranking signal, not a filter: it stays in
`preference_beliefs`, not in `hard_constraints`.

Task-scope and long-term constraints coexist without deletion. The projector applies a
deterministic precedence: an active task/session constraint suppresses the long-term
constraint of the same `kind` **for the current task only**, and reports the suppressed
ids; the long-term record remains in the state.

### 2.4 Observations and read-only references

* `DecisionObservation` records a comparison as evidence. It has no preference field;
  `implied_preferences` is always an empty tuple. A user's choice between products is
  **not** proof of a specific feature preference.
* `CandidatePoolRef` / `RankingSnapshotRef` are opaque references (`ref`, `source`,
  `candidate_count`, optional SHA-256 `digest`, `referenced_by_event_id`) with
  `read_only: Literal[True]`. There is no field for a product identity, a per-item rank,
  a raw score or a fused score; the trusted recommendation layer keeps those. Fabricated
  product data and model scores have nowhere to be stored.

### 2.5 Deterministic identity

Every id is derived from content, never from a counter or a clock:
`deterministic_id(prefix, *parts)` = prefix + first 24 hex chars of SHA-256 over the
`\x1f`-joined parts. Examples:

* `preference_belief_id(...)` — user + scope bucket + scope + kind + value + polarity,
  where the task id is part of the bucket except for `LONG_TERM` (so a long-term belief
  has one id across tasks);
* `decision_constraint_id(...)`, `feature_priority_id(...)`,
  `knowledge_belief_id(...)`, `comparison_observation_id(event_id)`.

The state's `state_digest()` is SHA-256 over canonical JSON of the whole state
(sorted keys, no insignificant whitespace). Replay determinism is asserted by comparing
digests.

---

## 3. Event schema

### 3.1 Envelope (`recommendation/state/events.py::StateEvent`)

| Field | Meaning |
| --- | --- |
| `event_id` | caller-supplied stable identity; duplicate id is idempotent **only** if semantics match |
| `user_key` | owner; isolation boundary |
| `session_id` | optional session scope metadata |
| `task_id` | task the event was emitted in |
| `event_type` | closed `EventType` enum |
| `schema_version` | `EVENT_SCHEMA_VERSION = 1`; unknown version refused |
| `payload` | one closed payload model selected by `event_type` (`extra="forbid"`) |
| `actor` | `user` / `agent` / `trusted_component` / `system` |
| `source` | producing component label (bounded) |
| `timestamp` | finite non-negative event time — **metadata only**, never ordering |
| `sequence` | store-assigned per-user monotonic order; `None` until appended |

`EVENT_PAYLOAD_MODELS` is the single `event_type -> payload model` registry.
`EVENT_ACTORS` is the declarative authority table. A payload/type mismatch, an actor not
allowed for the type, or an unknown schema version is a hard `ValidationError`.

### 3.2 Event types

Nine required types:

| Event | Actor | Meaning |
| --- | --- | --- |
| `USER_PREFERENCE_STATED` | user | explicit preference statement (scope defaults to TASK) |
| `PREFERENCE_HYPOTHESIS_PROPOSED` | agent/system/trusted | soft inference with named basis, TASK/SESSION scope, WEAK/MODERATE strength |
| `PREFERENCE_CONFIRMED` | user | promotes one inferred hypothesis, same scope |
| `PREFERENCE_RETRACTED` | user | withdraws a belief/kind; record kept |
| `USER_CONSTRAINT_STATED` | user | the only path to a hard constraint |
| `PRODUCT_COMPARISON_FEEDBACK` | user | observation only (comparison refs + selection) |
| `FEATURE_EXPLANATION_REQUESTED` | user | attribute needs explanation |
| `FEATURE_EXPLANATION_SHOWN` | agent/system | explanation displayed (not understanding) |
| `USER_UNDERSTANDING_DEMONSTRATED` | user | explicit evidence of understanding |

Additional types with a single clear purpose (not speculative):

| Event | Actor | Meaning |
| --- | --- | --- |
| `PREFERENCE_PRIORITY_STATED` | user | "comfort matters more than weight" — relative feature importance, not a new preference value |
| `TASK_CONTEXT_UPDATED` | agent/system | typed task context (intent label, bounded request text) |
| `CANDIDATE_POOL_REFERENCED` | trusted component | read-only candidate-pool reference |
| `RANKING_SNAPSHOT_REFERENCED` | trusted component | read-only ranking-snapshot reference |

Payload bounds: user-authored spans are whitespace-collapsed, truncated to the accepted
M9 bound (`MAX_SOURCE_TEXT_LENGTH = 500` for support spans, 1000 for request text) and
replaced by `[redacted]` when they look like a credential. Preference values and
attributes are bounded and refused (not truncated) when too long or secret-like.

---

## 4. Belief transition rules

The reducer is `BeliefUpdater.apply(state, event) -> state`. It is pure: a rejected event
raises and leaves the input state untouched. `apply_all` folds a store-ordered stream.

Common rules for every event:

1. `event.user_key` must equal `state.user_key`, otherwise `CrossUserStateError`.
2. An event for a different task is accepted **only** when its payload scope is
   `LONG_TERM` (that is how long-term state crosses tasks); otherwise
   `UnsupportedTransitionError`.
3. When `event.sequence` is present it must be strictly greater than the state's
   `interaction.last_event_sequence`; out-of-order/duplicate application is refused.
   Re-appending the exact last event id is refused as a duplicate.
4. `interaction.event_count += 1`; `user_turn_count += 1` when `actor == user`;
   `last_event_id`/`last_event_sequence` updated. `session_id` is updated only by an
   event from the state's own task.
5. Task-scoped events only ever touch task-scoped records. A `TASK`/`SESSION` event never
   modifies a `LONG_TERM` belief or constraint, and a task-scope retraction never removes
   long-term memory.

Per-event rules:

| Event | Transition |
| --- | --- |
| `USER_PREFERENCE_STATED` | Upsert belief as `EXPLICIT` / `ACTIVE` / `USER_STATED` / `STRONG`, same record if it exists, provenance appended. A conflicting **inferred** hypothesis of the same kind/value is retracted (`RETRACTED`/`REMOVED`), never silently kept. A conflicting explicit/confirmed belief of the same kind/value/opposite polarity is `SUPERSEDED` (record kept). `replaces_value` supersedes exactly that value with the same inferred-vs-explicit distinction. An existing active inferred belief of the same identity is upgraded to `EXPLICIT` because the user said it explicitly (not silently). |
| `PREFERENCE_HYPOTHESIS_PROPOSED` | Upsert belief as `INFERRED_HYPOTHESIS` / `ACTIVE` / `SYSTEM_INFERRED`, strength from payload. Refused if an explicit/confirmed belief of the same kind/value exists with the opposite polarity ("inference cannot contradict the user"). Same-polarity explicit stays explicit; only provenance is appended. A hypothesis also never overwrites an explicit/confirmed **record** of the same identity, even a historical one (retracted/superseded): the event is refused so user evidence is never erased. A competing inferred belief of the same kind/value/opposite polarity is retracted first. Long-term scope and `STRONG` are refused by the payload. |
| `PREFERENCE_CONFIRMED` | Allowed only from an active `INFERRED_HYPOTHESIS` with the same identity/scope. Sets `CONFIRMED` / `ACTIVE` / `USER_CONFIRMED` / `STRONG`, provenance appended. Confirming anything else raises `UnsupportedTransitionError`. |
| `PREFERENCE_RETRACTED` | Matches active beliefs by scope + kind (optionally value and polarity). Every match moves to `RETRACTED` / `REMOVED` with provenance; nothing is deleted. No match raises `UnsupportedTransitionError`. |
| `USER_CONSTRAINT_STATED` | Upsert `DecisionConstraint` with authority `USER_EXPLICIT`, status `ACTIVE`, scope from payload; provenance appended. Never touches beliefs; never deletes a long-term constraint. |
| `PREFERENCE_PRIORITY_STATED` | Upsert `FeaturePriority` for `(scope, kind, higher, lower)`; an existing reverse pair in the same scope is `SUPERSEDED` first, so last explicit statement wins deterministically. |
| `PRODUCT_COMPARISON_FEEDBACK` | Append a `DecisionObservation` (opaque refs, selection, bounded feedback). **No** belief, constraint or knowledge change; a choice alone cannot prove a feature preference. |
| `FEATURE_EXPLANATION_REQUESTED` | Knowledge status → `NEEDS_EXPLANATION` from `UNKNOWN`/`EXPLANATION_PROVIDED`; stays if already `NEEDS_EXPLANATION` or `UNDERSTANDING_DEMONSTRATED` (no silent downgrade). |
| `FEATURE_EXPLANATION_SHOWN` | `UNKNOWN`/`NEEDS_EXPLANATION` → `EXPLANATION_PROVIDED`; `EXPLANATION_PROVIDED`/`UNDERSTANDING_DEMONSTRATED` unchanged. **Never** `UNDERSTANDING_DEMONSTRATED`. |
| `USER_UNDERSTANDING_DEMONSTRATED` | Requires the user's evidence span; sets `UNDERSTANDING_DEMONSTRATED` from any status. |
| `TASK_CONTEXT_UPDATED` | Sets task context (intent label, bounded/redacted request text) and its provenance. |
| `CANDIDATE_POOL_REFERENCED` | Replaces the read-only candidate-pool reference (latest trusted reference wins). |
| `RANKING_SNAPSHOT_REFERENCED` | Replaces the read-only ranking-snapshot reference. |

Ordering/determinism guarantees:

* tuple fields preserve first-appearance order; updates replace in place;
* all ids are deterministic; all serialisation is canonical;
* the same ordered event stream always produces a byte-identical canonical state;
* no external model, LLM, wall clock or process-global state participates in reduction.

---

## 5. Event store

`SQLiteEventStore` uses the Python standard library only (`sqlite3`) — no framework is
added. Two tables: `meta` (`event_store_schema_version`) and `events`.

| Column | Notes |
| --- | --- |
| `event_id` | primary key (global identity) |
| `user_key` | isolation column; every read filters it |
| `session_id`, `task_id` | scope metadata |
| `event_type`, `schema_version` | closed vocabulary / contract version |
| `payload` | canonical JSON of the typed payload |
| `payload_hash` | SHA-256 of the canonical payload (duplicate/conflict check) |
| `actor`, `source` | authority audit |
| `timestamp` | event metadata, not ordering |
| `sequence` | per-user monotonic integer; `UNIQUE(user_key, sequence)`; **the ordering key** |
| `scope` | payload scope (`long_term` included, `NULL` otherwise) so long-term events can be replayed for every task |

API:

```text
append(event) -> StateEvent              # stored copy with sequence
get_events(task_id, user_key) -> tuple[StateEvent, ...]   # task events + user's long-term events, sequence order
replay(task_id, user_key) -> tuple[StateEvent, ...]       # the ordered stream a reducer folds
get_event(event_id) -> StateEvent | None
get_all_events(user_key) -> tuple[StateEvent, ...]
count(user_key) -> int
close()
```

Guarantees:

* **idempotent duplicate ids**: re-appending an event whose semantic fields match returns
  the stored event unchanged (no new sequence, no duplicate row);
* **conflicting duplicate ids rejected**: same id, different user/task/type/payload/actor/
  source raises `EventConflictError`;
* **deterministic order**: `sequence` is assigned inside one `BEGIN IMMEDIATE`
  transaction (`MAX(sequence)+1` for that user); reads order by it, never by timestamp;
* **cross-user isolation**: all queries filter `user_key`, ids are never reused across
  users, and a duplicate id submitted by a different user is a conflict;
* **transactional**: schema creation, duplicate check, sequence assignment and insert run
  in one transaction; failures roll back;
* **no LLM / no network**: reconstruction is a pure local fold;
* **bounded/redacted text** is enforced by the event payload models before the row exists;
* **separate from SASRec history**: there is no table, column or method for the trusted
  purchase history; the store is a conversational-decision store only.

---

## 6. Projections (`state_projector.py`)

Three read-only, frozen projections. All of them omit `user_key`, raw support spans and
anything that could be mistaken for a product fact.

### 6.1 `for_agent(state) -> AgentStateProjection`

Compact, structured decision-state summary for the Agent policy:

* task/session ids, state version, event count, user-turn count;
* task context (intent, bounded request text);
* preference summaries (id, kind, value, polarity, scope, lifecycle, strength, status,
  soft flag) — includes soft hypotheses so the Agent can choose a clarification;
* knowledge summaries and `clarification_attributes` (`NEEDS_EXPLANATION`);
* active feature priorities;
* authorized hard constraints (kind/value/polarity/scope);
* opaque candidate-pool / ranking-snapshot refs and counts;
* observation count and inferred/explicit preference counts.

### 6.2 `for_ranker(state) -> RankerStateProjection`

Typed preference features and soft-signal metadata for a future fine ranker. No model is
trained or changed in this milestone.

* one `RankerFeature` per active belief: `preference_id`, `kind`, `value`, `polarity`,
  optional `direction`, `scope`, `lifecycle`, `strength_rank` (`1..3`, ordinal),
  `is_soft` (true for hypotheses);
* `RankerPriority` for relative feature importance;
* opaque candidate-pool / ranking-snapshot refs;
* `ranker_signal_version = 1`.

Hard constraints are **not** included: the ranker may order candidates, it may not filter
them. Retracted/superseded beliefs are excluded. `strength_rank` is an ordering weight,
not a probability.

### 6.3 `for_constraints(state) -> ConstraintStateProjection`

Only authorized constraints:

* active `DecisionConstraint`s whose authority is `USER_EXPLICIT` (the only hard
  authority in M5-MVP), as `ConstraintView`s;
* deterministic precedence: an active task/session constraint suppresses the long-term
  constraint of the same `kind` for this projection; suppressed ids are reported in
  `suppressed_long_term_ids` (the underlying long-term record is unchanged);
* `excluded_hypothesis_count`: how many inferred hypotheses were deliberately **not**
  turned into filters;
* `constraints_version = 1`.

Unconfirmed inferred hypotheses never appear here, so they can never become hard filters.

---

## 7. Orchestration API (`state_manager.py`)

```python
manager = StateManager(SQLiteEventStore(database))

state   = manager.load_or_create(user_key, session_id, task_id)  # replay; empty state if no events
state   = manager.apply_event(event)                              # append (idempotent) then replay
state   = manager.get_state(user_key, task_id)                    # replay; empty state if none
state   = manager.replay_state(user_key, task_id)                 # explicit full reconstruction
```

Semantics:

* the state is **derived state**: it is always reconstructed from the stored event stream,
  so it can never diverge from the events;
* `apply_event` validates the transition **before** appending: it folds the event onto
  the current replay state as a pure dry-run, and only a legal event is appended. The
  append-only store has no delete operation, so persisting a rejected transition would
  corrupt every later replay — the dry-run is what makes "a rejected transition stores
  nothing" true. The store's transaction then owns sequence assignment, duplicate-id
  idempotency and conflict detection, and the manager replays once more to return the
  resulting state;
* appending the same event twice is idempotent (same stored event, same state digest);
  a re-submitted duplicate skips the dry-run and goes straight to the store's
  idempotency/conflict path;
* `load_or_create` sets the caller's current `session_id` on the returned state; replay
  uses the last task-scoped event's session;
* a rejected transition raises and stores nothing; the previously returned state remains
  valid;
* concurrency: the manager serialises `apply_event` with an in-process lock and the store
  uses one lock plus `BEGIN IMMEDIATE`, so appends are never interleaved and the store is
  safe to share across threads (`check_same_thread=False` under the lock), matching the
  accepted M9 store's approach. Per-task concurrent writers are out of scope for this
  MVP: a transaction writes in one conversational loop, so validate-then-append is not
  racing another writer of the same task.

The manager is intentionally small. It has no policy, no ranking, no memory write and no
model call.

---

## 8. Authority and trust boundaries

| Boundary | Rule |
| --- | --- |
| user text -> event | only through typed, validated payloads; raw text is bounded and credential-redacted |
| event -> state | only the deterministic reducer; no LLM, clock or randomness |
| inference vs truth | inference is always `INFERRED_HYPOTHESIS`, TASK/SESSION scope, WEAK/MODERATE strength; it has no path to a hard constraint or durable memory |
| explicit vs inference | explicit user statements and retractions win; a conflicting inference is retracted, and an inference may never contradict an explicit belief |
| task vs long-term | task-scope events never modify long-term records; precedence is applied only at projection time and is reported |
| knowledge | an explanation shown proves display, not understanding; only user demonstration moves to `UNDERSTANDING_DEMONSTRATED` |
| observations | a product choice is evidence, never a feature preference by itself |
| trusted references | candidate/ranking refs are opaque, read-only, and carry no product identity or score |
| store | user-scoped, append-only, transactional; conversational events are separate from SASRec's trusted purchase history |

---

## 9. Old-to-new architecture mapping

| Accepted component (current) | M5-MVP role | Migration path |
| --- | --- | --- |
| `recommendation/memory/` (`PreferenceMemoryService`, SQLite store, extraction) | unchanged; still the Serving V1 preference-memory writer | later: becomes a **projection** of validated long-term events; the store schema is not changed in this MVP |
| `PreferenceMemorySnapshot` (active explicit entries) | read-only input to `legacy_adapter.project_legacy_memory` | adapter maps active explicit entries to `LONG_TERM` / `EXPLICIT` beliefs with `legacy_memory_id`; no write to the old store, no production migration |
| `recommendation/control/task_state.py` (`TaskState`, `TaskConstraint`, `PreferenceScope`, `PreferenceLifecycle`) | semantics reused (lifecycle/scope) and superseded structurally by the unified state for new infrastructure | later: the control plane's task state can be derived from this state's task projection; accepted enums can re-export the state enums |
| `recommendation/control/memory_proposal.py` (`MemoryWriteProposal`, validator) | unchanged; still the only durable-write gate for M9 memory | later: approved proposals become `LONG_TERM` events instead of direct memory writes |
| `recommendation/demo/shopping_intent.py` task preferences | unchanged Serving V1 behaviour | later: task preferences read from `for_ranker` / `for_constraints` instead of separate ad-hoc state |
| `recommendation/control/constraint_eligibility.py` | unchanged | later: consumes `for_constraints` as its authorized constraint set |
| Decision Agent (`recommendation/demo/agent_service.py`) | unchanged in M5-MVP | integration plan below |
| Fine ranker (future) | not implemented in M5-MVP | consumes `for_ranker` projection |

No production memory records are migrated, no existing database schema changes, and
`recommendation/agent/graph.py` is untouched.

---

## 10. Integration plan for the Decision Agent and the Fine Ranker

### Decision Agent (next milestone)

1. Add a `StateManager` instance to the demo service composition (one SQLite database for
   decision events, e.g. from `AGENTRECX_STATE_DB`), separate from the M9 memory DB.
2. The trusted controller translates each validated turn into events:
   `TASK_CONTEXT_UPDATED`, `USER_PREFERENCE_STATED` / `USER_CONSTRAINT_STATED` from the
   accepted interpreter, `PREFERENCE_HYPOTHESIS_PROPOSED` from the candidate/comparison
   evidence layer (never from free model text), and knowledge events from the explanation
   path.
3. Pass `for_agent(state)` to the bounded policy as read-only context; the policy may
   choose a question or route but cannot write state. `for_constraints(state)` is the
   authoritative filter input; `for_ranker(state)` is the soft-signal input.
4. Long-term transitions become `MemoryWriteProposal` inputs (stage 4 above), so the old
   memory service remains the writer until it is itself made a projection.
5. No LLM may construct an event payload directly; the reducer's typed payload models and
   actor table remain the boundary.

### Fine Ranker (future milestone)

1. `for_ranker(state)` is the only preference input to the ranker; it contains typed,
   scoped, lineage-attributed soft signals and no hard constraints.
2. Train on logged decision events joined with presentation/feedback outcomes, using the
   same event ids as lineage keys. The ranker output stays a reordering of the trusted
   candidate pool; it never adds candidates, never filters, and never changes identity.
3. Quality claims require a frozen evaluation protocol; this document makes none.

---

## 11. Known limitations

* The event store is a single-writer local SQLite store. Sequence assignment is correct
  and transactional, but this is not a distributed log; a production deployment would
  need an external queued log.
* `apply_event` replays the whole task stream after each append. This is the simplest
  correct implementation and is fine for conversational task lengths; snapshotting is a
  later optimisation.
* Knowledge beliefs are currently reconstructed per task; cross-task knowledge
  persistence is not implemented (no long-term knowledge scope yet).
* The legacy adapter projects only active explicit M9 entries, and only as long-term
  beliefs; it does not project removals/supersession history and does not migrate
  anything.
* Constraint consistency is the caller's responsibility: two contradictory active
  constraints of the same kind are both retained (only projection precedence between
  task and long-term scopes is resolved). Cross-constraint satisfiability is a later
  milestone.
* No fine ranker is trained and no recommendation quality claim is made. Serving V1
  behaviour is unchanged.
* There is no event-snapshot cache, no event compaction and no encryption at rest.

---

## 12. Next milestone

1. Wire the trusted controller to emit M5 events for one conversational path.
2. Add a read-only adapter that turns `MemoryWriteProposal` approvals into `LONG_TERM`
   events, keeping the M9 writer until parity is proven.
3. Feed `for_constraints` into constraint eligibility and `for_ranker` into a typed
   soft-signal rerank path, with the accepted frozen evaluation protocol.
4. Only after that: event snapshots/compaction and a fine-ranker training protocol.

---

## 13. Verification

Run from the repository root (the accepted environment uses `.venv/bin/python`; on a
Windows host use the available Python 3.10+ interpreter):

```bash
.venv/bin/python -m pytest -q tests/test_state_schemas.py tests/test_state_event_store.py \
    tests/test_state_beliefs.py tests/test_state_projector.py tests/test_state_manager.py \
    tests/test_state_legacy_adapter.py tests/test_m5_state_acceptance.py
.venv/bin/python -m experiments.m5_unified_state_smoke
```

The tests use temporary SQLite databases and never touch a production memory database.
The smoke example needs no model checkpoint, no dataset and no network.

### 13.1 Executed verification (this worktree, 2026-10-30)

| Command | Result |
| --- | --- |
| `python -m pytest -q tests/test_state_schemas.py tests/test_state_event_store.py tests/test_state_manager.py tests/test_state_beliefs.py tests/test_state_projector.py tests/test_state_legacy_adapter.py tests/test_m5_state_acceptance.py` | **141 passed, 1 skipped** (the skip is the accepted-control-plane enum parity test, which needs the PyTorch-backed control plane) |
| `python -m experiments.m5_unified_state_smoke` | **SMOKE: PASS, 37/37 checks**, exit 0; incremental / replayed / JSON-round-trip / fresh-loader digests all equal |
| `python -m pytest -q tests/test_memory_service.py` | **59 passed** (accepted M9 regression, unchanged) |
| `python -m pytest -q tests/test_memory_query_adapter.py` (policy-helper tests deselected) | **15 passed**; the two deselected tests import the demo policy layer → `torch`, unavailable in this interpreter |
| `python -m compileall` on every new module/test/smoke | exit 0 |

The executed interpreter in that environment was Python 3.13 with Pydantic 2.10.3 and
pytest 8.3.4. The repository's Linux `.venv` is not runnable on this Windows host; the
M5 code deliberately has no model/API dependency, so this is sufficient for the slice.
`tests/test_memory_store.py` file-database cases and the API/Agent regression suites
could not run there (pytest's 0700 temporary directories are rejected by the host sandbox,
and `fastapi`/`torch` are absent); those are pre-existing environment limitations, not
M5 behaviour. The acceptance evidence above is what was actually executed.

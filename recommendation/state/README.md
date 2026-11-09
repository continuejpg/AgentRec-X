# `recommendation.state` — Unified Decision State & Belief Management (M5 / M5.1 / M5.2)

**Status: engineering slice.** The state engine itself is not on the serving path; the
default `serving_v1` path and the accepted M9 preference memory are unchanged. Opt-in
integrations (OFF default / SHADOW / APPLY_CONSTRAINTS) record validated shopping turns as
events in the demo runtime: M5.1 covers the main planning turn, M5.2 adds clarification
feedback, correction/retraction, stable turn identity and trusted candidate/ranking
snapshots — see
[`docs/M5_1_SHOPPING_LOOP_INTEGRATION.md`](../../docs/M5_1_SHOPPING_LOOP_INTEGRATION.md) and
[`docs/M5_2_INTERACTION_FEEDBACK_LOOP.md`](../../docs/M5_2_INTERACTION_FEEDBACK_LOOP.md).
This package trains no model, runs no Agent policy and calls no LLM.

## What this package is

`UnifiedDecisionState` is the central state abstraction of the AgentRec-X 3.0 direction.
Memory is not the centre of the agent: it is a projection of validated user information.
Every state change arrives as an event and is reduced deterministically:

```text
user interaction -> StateEvent -> append-only SQLite event store
                -> BeliefUpdater (pure reduction) -> UnifiedDecisionState
                -> StateProjector: for_agent / for_ranker / for_constraints
```

## Module map

| Module | Responsibility |
| --- | --- |
| `schemas.py` | shared enums, bounds, validation/redaction helpers, deterministic ids |
| `errors.py` | typed rejections (`StateEngineError` and subclasses) |
| `events.py` | typed event payloads, `StateEvent` envelope, actor-authority registry |
| `decision_state.py` | `UnifiedDecisionState` and sub-models, deterministic id helpers |
| `event_store.py` | append-only SQLite event store (idempotency, conflicts, ordering) |
| `belief_updater.py` | deterministic event-to-state transition function |
| `state_projector.py` | read-only `for_agent` / `for_ranker` / `for_constraints` projections |
| `state_manager.py` | small orchestration API (`load_or_create`, `apply_event`, `replay_state`) |
| `legacy_adapter.py` | read-only bridge from the accepted M9 `PreferenceMemorySnapshot` |
| `shopping_adapter.py` | trusted shopping-fact adapter: explicit statements, M9 REPLACE/REMOVE, trusted clarification options, turn-identity resolution and candidate/ranking snapshot references → M5 events (lightweight; no model/API imports) |

The full contract — state schema, event schema, transition rules, authority boundaries,
old-to-new mapping, integration plan and limitations — is
[`docs/M5_UNIFIED_DECISION_STATE.md`](../../docs/M5_UNIFIED_DECISION_STATE.md).

## Invariants

1. The state is **derived**: always reconstructed from the stored event stream, never
   edited in place. `UnifiedDecisionState` is frozen.
2. `INFERRED_HYPOTHESIS` is soft evidence only. It can affect a soft ranking signal or
   trigger a clarification; it can never create a hard constraint or become a durable
   explicit preference silently.
3. Only `USER_CONSTRAINT_STATED` creates a hard constraint. Confirmed preferences stay
   ranking signals, not filters.
4. `FEATURE_EXPLANATION_SHOWN` moves a knowledge belief to `EXPLANATION_PROVIDED`, never
   to `UNDERSTANDING_DEMONSTRATED`; only the user can demonstrate understanding.
5. A product comparison is an observation. It never implies a feature preference.
6. A task-scoped event never modifies long-term state. Projection-level precedence
   (task/session over long-term, same kind) is reported, and the long-term record stays.
7. Candidate/ranking references are opaque and read-only. No product identity and no
   model score is stored in the decision state.
8. The decision-event store is separate from the trusted chronological purchase history
   used to train SASRec; no API here can write a behavioural event.
9. Reduction is deterministic: no model, no network, no wall clock, no randomness.
   Replaying the same ordered stream produces the same canonical state digest.
10. User text is bounded and credential-redacted before it is persisted.

## Minimal usage

```python
from recommendation.state.event_store import SQLiteEventStore
from recommendation.state.events import new_event
from recommendation.state.state_manager import StateManager
from recommendation.state.state_projector import for_agent, for_ranker, for_constraints

with SQLiteEventStore("decision_state.sqlite3") as store:
    manager = StateManager(store)
    state = manager.load_or_create("user-1", "session-1", "task-1")
    event = new_event(
        event_type="user_preference_stated",
        event_id="evt-0001",
        user_key="user-1",
        task_id="task-1",
        session_id="session-1",
        actor="user",
        timestamp=1_700_000_000.0,
        kind="feature",
        value="lightweight",
        source_text="I want something lightweight.",
    )
    state = manager.apply_event(event)
    replay = manager.replay_state("user-1", "task-1")
    assert replay.state_digest() == state.state_digest()
```

## Tests and smoke

```bash
.venv/bin/python -m pytest -q tests/test_state_schemas.py tests/test_state_event_store.py \
    tests/test_state_beliefs.py tests/test_state_projector.py tests/test_state_manager.py \
    tests/test_state_legacy_adapter.py tests/test_m5_state_acceptance.py
.venv/bin/python -m experiments.m5_unified_state_smoke
```

Tests create temporary SQLite databases and require no model checkpoint, dataset, network
or LLM. The smoke example is the runnable end-to-end demonstration of the milestone.

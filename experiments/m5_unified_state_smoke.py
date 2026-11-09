"""M5-MVP unified decision state smoke: the 11-step hiking-backpack scenario.

This is a runnable, fully offline end-to-end demonstration of the milestone:

    user turn -> StateEvent -> append-only SQLite event store
              -> BeliefUpdater (deterministic reduction) -> UnifiedDecisionState
              -> for_agent / for_ranker / for_constraints

The script drives one conversational task for one user through :class:`StateManager`:

 1. a new task state and its typed task context (``intent=recommend``);
 2. a historical-browsing hypothesis that the user wants a lightweight pack;
 3. the system records it as an inferred *soft* belief (shown as INFERRED/soft);
 4. the user states an explicit budget constraint of $150 (``price_max``);
 5. trusted candidate-pool / ranking-snapshot references, an explanation display and a
    product-comparison choice -- these are references and an observation only;
 6. the user says comfort matters more than weight (relative feature priority);
 7. the explicit comfort preference is recorded; the three projections are printed;
 8. the user scopes a preference to this task (scope=task);
 9. no long-term (durable) belief was created by these task events;
10. the state survives canonical JSON serialization + reconstruction;
11. replaying the stored event stream reproduces the *same* effective state digest as
    the incrementally built state (printed side by side, then PASS).

No model checkpoint, dataset, network, LLM or provider key is used.  Everything is
deterministic: fixed event timestamps, explicit event ids, canonical JSON output, no
wall-clock value in the result.

Usage::

    python -m experiments.m5_unified_state_smoke
    python -m experiments.m5_unified_state_smoke --db /tmp/m5_state.sqlite3 --json /tmp/m5.json

By default a temporary SQLite database is created and removed afterwards; ``--db`` keeps
the database at the given path.  ``--json`` writes the complete machine-readable result.
Exit code is 0 only when every check passes.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
import time
import traceback
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from recommendation.state.decision_state import UnifiedDecisionState  # noqa: E402
from recommendation.state.event_store import SQLiteEventStore  # noqa: E402
from recommendation.state.events import (  # noqa: E402
    EVENT_PAYLOAD_MODELS,
    EventType,
    new_event,
)
from recommendation.state.schemas import (  # noqa: E402
    BeliefOrigin,
    ConstraintAuthority,
    KnowledgeStatus,
    PreferenceLifecycle,
    PreferenceScope,
    canonical_json,
)
from recommendation.state.state_manager import StateManager  # noqa: E402
from recommendation.state.state_projector import (  # noqa: E402
    for_agent,
    for_constraints,
    for_ranker,
)

#: Fixed identities and event time.  Replay determinism must not depend on a clock.
USER_KEY = "smoke-hiker"
SESSION_ID = "session-hiking-backpack-1"
TASK_ID = "task-hiking-backpack"
EVENT_TS = 1_700_000_000.0


class Checks:
    """Collects named PASS/FAIL checks and prints them deterministically."""

    def __init__(self) -> None:
        self.results: list[tuple[str, bool]] = []

    def check(self, name: str, condition: bool) -> bool:
        """Record and print one check; returns the truth value."""
        passed = bool(condition)
        self.results.append((name, passed))
        print(f"  [{'PASS' if passed else 'FAIL'}] {name}")
        return passed

    @property
    def passed(self) -> bool:
        """True only when at least one check ran and all of them passed."""
        return bool(self.results) and all(ok for _, ok in self.results)

    @property
    def passed_count(self) -> int:
        return sum(1 for _, ok in self.results if ok)

    def as_list(self) -> list[dict[str, Any]]:
        return [{"name": name, "passed": ok} for name, ok in self.results]


class EventFactory:
    """Builds validated events with fixed metadata and a deterministic timestamp."""

    def __init__(self) -> None:
        self.index = 0
        self.applied: list[tuple[str, str, str]] = []  # (event_id, event_type, actor)

    def make(
        self,
        event_type: str,
        event_id: str,
        actor: str,
        **payload_fields: Any,
    ):
        """Return one validated event for the smoke task/user/session.

        The payload model is built explicitly so a payload field named ``source`` (the
        candidate-pool / ranking-snapshot producing component) cannot collide with the
        event envelope's ``source`` keyword.
        """
        self.index += 1
        resolved_type = EventType(event_type)
        payload = EVENT_PAYLOAD_MODELS[resolved_type](**payload_fields)
        event = new_event(
            event_type=resolved_type,
            event_id=event_id,
            user_key=USER_KEY,
            task_id=TASK_ID,
            session_id=SESSION_ID,
            actor=actor,
            source="m5_unified_state_smoke",
            timestamp=EVENT_TS + self.index,
            payload=payload,
        )
        self.applied.append((event.event_id, event.event_type.value, event.actor.value))
        return event


def step(number: int, title: str) -> None:
    """Print one deterministic scenario step header."""
    print(f"\n--- Step {number}: {title} ---")


def show(label: str, model: Any) -> None:
    """Print a model as canonical (sorted, whitespace-free) JSON."""
    payload = model.model_dump(mode="json") if hasattr(model, "model_dump") else model
    print(f"  {label}: {canonical_json(payload)}")


def projection_payload(projection: Any) -> dict[str, Any]:
    """Return a JSON-serialisable projection payload."""
    return projection.model_dump(mode="json")


# --------------------------------------------------------------------------- #
# Steps 1-10 (the store stays open; step 11 reopens it)
# --------------------------------------------------------------------------- #


def run_steps_1_to_10(
    manager: StateManager,
    store: SQLiteEventStore,
    checks: Checks,
    factory: EventFactory,
    result: dict[str, Any],
) -> UnifiedDecisionState:
    """Run the conversational scenario; return the incrementally built final state."""
    # ---- Step 1: a new task state and its typed context ------------------- #
    step(1, "new task state + task context (intent=recommend)")
    state = manager.load_or_create(USER_KEY, SESSION_ID, TASK_ID)
    checks.check("new task state is empty", (
        state.event_count if hasattr(state, "event_count") else state.interaction.event_count
    ) == 0)
    checks.check("new task state has no beliefs or constraints", (
        state.preference_beliefs == () and state.hard_constraints == ()
    ))
    print(f"  new state: state_version={state.state_version} user={state.user_key!r} "
          f"task={state.task_id!r} session={state.session_id!r} events=0")

    state = manager.apply_event(
        factory.make(
            "task_context_updated",
            "evt-01-context",
            "agent",
            intent="recommend",
            request_text="I need a hiking backpack for a weekend trip, lightweight "
            "and under $150.",
        )
    )
    checks.check("task context intent is 'recommend'", state.task_context.intent == "recommend")
    checks.check("task context request text is bounded and present", (
        "hiking backpack" in state.task_context.request_text
        and state.task_context.updated_by_event_id == "evt-01-context"
    ))
    checks.check("context event was stored (event_count=1)", state.interaction.event_count == 1)
    print(f"  task_context: intent={state.task_context.intent!r} "
          f"request_text={state.task_context.request_text!r}")

    # ---- Step 2: browsing history proposes a soft hypothesis -------------- #
    step(2, "historical browsing proposes 'lightweight' (hypothesis event)")
    state = manager.apply_event(
        factory.make(
            "preference_hypothesis_proposed",
            "evt-02-hypothesis",
            "agent",
            kind="feature",
            value="lightweight",
            polarity="prefer",
            scope="task",
            strength="weak",
            basis="browsing_history",
            reason="recent browsing skewed toward lightweight day packs",
            evidence_refs=("browsing-history-2024-11",),
        )
    )
    hypothesis = state.inferred_hypotheses[0] if state.inferred_hypotheses else None
    checks.check("system hypothesis exists and is soft", hypothesis is not None and (
        hypothesis.lifecycle is PreferenceLifecycle.INFERRED_HYPOTHESIS
        and hypothesis.is_soft_signal is True
    ))

    # ---- Step 3: show the inferred/soft belief ---------------------------- #
    step(3, "system records an inferred SOFT belief (not truth)")
    belief = state.inferred_hypotheses[0]
    checks.check("hypothesis is INFERRED_HYPOTHESIS/ACTIVE/WEAK/SYSTEM_INFERRED", (
        belief.lifecycle is PreferenceLifecycle.INFERRED_HYPOTHESIS
        and belief.status.value == "active"
        and belief.strength.rank == 1
        and belief.origin is BeliefOrigin.SYSTEM_INFERRED
    ))
    checks.check("inferred belief is not committable durable memory",
                 belief.may_be_committed is False)
    ranker_after_hypothesis = for_ranker(state)
    soft_feature = [
        feature for feature in ranker_after_hypothesis.features
        if feature.preference_id == belief.belief_id
    ][0]
    checks.check("for_ranker flags the hypothesis is_soft=True", soft_feature.is_soft is True)
    show("inferred belief", belief)
    print(f"  for_ranker soft signal: kind={soft_feature.kind.value} "
          f"value={soft_feature.value!r} strength_rank={soft_feature.strength_rank} "
          f"is_soft={soft_feature.is_soft}")

    # ---- Step 4: explicit budget constraint -------------------------------- #
    step(4, "user states an explicit maximum budget of $150 (hard constraint)")
    state = manager.apply_event(
        factory.make(
            "user_constraint_stated",
            "evt-04-constraint",
            "user",
            kind="price_max",
            value="150",
            polarity="prefer",
            scope="task",
            source_text="My maximum budget is $150.",
        )
    )
    constraint = [c for c in state.hard_constraints if c.kind.value == "price_max"][0]
    checks.check("explicit budget is an authorized hard constraint", (
        constraint.authority is ConstraintAuthority.USER_EXPLICIT
        and constraint.is_authorized_hard is True
        and constraint.scope is PreferenceScope.TASK
    ))
    constraints_projection = for_constraints(state)
    checks.check("for_constraints projects price_max=150", (
        len(constraints_projection.constraints) == 1
        and constraints_projection.constraints[0].kind.value == "price_max"
        and constraints_projection.constraints[0].value == "150"
    ))
    show("hard constraint", constraint)

    # ---- Step 5: trusted references, explanation display, comparison ------ #
    step(5, "trusted candidate pool + ranking snapshot; explanation shown; "
            "comparison feedback (observation only)")
    state = manager.apply_event(
        factory.make(
            "candidate_pool_referenced",
            "evt-05-candidates",
            "trusted_component",
            ref="candidate-pool-hike-1",
            source="recommendation_tool",
            candidate_count=8,
        )
    )
    state = manager.apply_event(
        factory.make(
            "ranking_snapshot_referenced",
            "evt-06-ranking",
            "trusted_component",
            ref="ranking-snapshot-hike-1",
            source="serving_v1",
            method="rrf_rank_fusion",
            candidate_count=8,
        )
    )
    checks.check("candidate pool is an opaque read-only reference", (
        state.candidate_pool_ref is not None
        and state.candidate_pool_ref.ref == "candidate-pool-hike-1"
        and state.candidate_pool_ref.candidate_count == 8
        and state.candidate_pool_ref.read_only is True
    ))
    checks.check("ranking snapshot is an opaque read-only reference", (
        state.ranking_snapshot_ref is not None
        and state.ranking_snapshot_ref.ref == "ranking-snapshot-hike-1"
        and state.ranking_snapshot_ref.method == "rrf_rank_fusion"
        and state.ranking_snapshot_ref.read_only is True
    ))

    state = manager.apply_event(
        factory.make(
            "feature_explanation_shown",
            "evt-07-explanation",
            "agent",
            attribute="waterproofing",
            explanation_ref="explanation-waterproofing-1",
        )
    )
    knowledge = state.knowledge_beliefs[0]
    checks.check("explanation shown moves knowledge to EXPLANATION_PROVIDED only", (
        knowledge.status is KnowledgeStatus.EXPLANATION_PROVIDED
        and knowledge.is_understood is False
    ))
    print(f"  knowledge[{knowledge.attribute!r}] = {knowledge.status.value} "
          f"(shown != understood)")

    belief_ids_before = {b.belief_id for b in state.preference_beliefs}
    constraint_ids_before = {c.constraint_id for c in state.hard_constraints}
    state = manager.apply_event(
        factory.make(
            "product_comparison_feedback",
            "evt-08-comparison",
            "user",
            compared_refs=("candidate:A", "candidate:B"),
            selected_ref="candidate:B",
            feedback_text="Item B felt more comfortable.",
        )
    )
    observation = state.observations[0]
    checks.check("comparison is recorded as an observation", (
        state.observation_count if hasattr(state, "observation_count")
        else len(state.observations)
    ) == 1 and observation.selected_ref == "candidate:B")
    checks.check("observation implies no preference", observation.implied_preferences == ())
    checks.check("comparison created no belief/constraint", (
        {b.belief_id for b in state.preference_beliefs} == belief_ids_before
        and {c.constraint_id for c in state.hard_constraints} == constraint_ids_before
    ))
    print(f"  observation: compared={observation.compared_refs} "
          f"selected={observation.selected_ref!r} implied_preferences="
          f"{observation.implied_preferences!r}")

    # ---- Step 6: relative feature importance ------------------------------ #
    step(6, "user says comfort matters more than weight (relative priority)")
    state = manager.apply_event(
        factory.make(
            "preference_priority_stated",
            "evt-09-priority",
            "user",
            higher="comfort",
            lower="weight",
            kind="feature",
            scope="task",
            source_text="Comfort matters more to me than weight.",
        )
    )
    priorities = state.active_priorities
    checks.check("comfort>weight priority is active and scoped to the task", (
        len(priorities) == 1
        and priorities[0].higher == "comfort"
        and priorities[0].lower == "weight"
        and priorities[0].scope is PreferenceScope.TASK
    ))

    # ---- Step 7: explicit comfort preference; print the projections ------- #
    step(7, "user states comfort preference explicitly; show for_agent / "
            "for_ranker / for_constraints")
    state = manager.apply_event(
        factory.make(
            "user_preference_stated",
            "evt-10-comfort",
            "user",
            kind="feature",
            value="comfort",
            polarity="prefer",
            scope="task",
            source_text="Comfort is the most important thing for this trip.",
        )
    )
    comfort = [b for b in state.preference_beliefs if b.value == "comfort"][0]
    checks.check("explicit comfort belief is EXPLICIT/STRONG/USER_STATED", (
        comfort.lifecycle is PreferenceLifecycle.EXPLICIT
        and comfort.strength.rank == 3
        and comfort.origin is BeliefOrigin.USER_STATED
        and comfort.is_soft_signal is False
    ))
    ranker = for_ranker(state)
    comfort_feature = [
        feature for feature in ranker.features if feature.preference_id == comfort.belief_id
    ][0]
    checks.check("for_ranker has both the soft hypothesis and the explicit belief", (
        comfort_feature.is_soft is False
        and len([f for f in ranker.features if f.is_soft]) == 1
        and len(ranker.priorities) == 1
    ))
    agent = for_agent(state)
    constraints = for_constraints(state)
    checks.check("for_agent sees 1 inferred + 1 explicit preference and 1 observation", (
        agent.inferred_preference_count == 1
        and agent.explicit_preference_count == 1
        and agent.observation_count == 1
    ))
    checks.check("for_constraints excludes the inferred hypothesis", (
        len(constraints.constraints) == 1
        and constraints.excluded_hypothesis_count == 1
    ))
    checks.check("for_ranker contains no hard constraints (order, never filter)", (
        not hasattr(ranker, "constraints")
    ))
    print("  projection outputs (canonical JSON):")
    print(f"    for_agent        : {canonical_json(projection_payload(agent))}")
    print(f"    for_ranker       : {canonical_json(projection_payload(ranker))}")
    print(f"    for_constraints  : {canonical_json(projection_payload(constraints))}")

    # ---- Step 8: a statement scoped to this task only --------------------- #
    step(8, "user scopes a preference to this task only (scope=task)")
    state = manager.apply_event(
        factory.make(
            "user_preference_stated",
            "evt-11-task-scope",
            "user",
            kind="feature",
            value="waterproofing",
            polarity="prefer",
            scope="task",
            source_text="For this backpack task only, waterproofing matters.",
        )
    )
    waterproofing = [b for b in state.preference_beliefs if b.value == "waterproofing"][0]
    checks.check("task-specific statement is TASK-scoped, not durable", (
        waterproofing.scope is PreferenceScope.TASK
        and waterproofing.may_be_committed is False
    ))

    # ---- Step 9: nothing became long-term memory -------------------------- #
    step(9, "no long-term (durable) memory was created by the task events")
    checks.check("no LONG_TERM preference belief exists", all(
        belief.scope is not PreferenceScope.LONG_TERM
        for belief in state.preference_beliefs
    ))
    checks.check("no LONG_TERM constraint exists", all(
        constraint.scope is not PreferenceScope.LONG_TERM
        for constraint in state.hard_constraints
    ))
    checks.check("every belief is still scoped to the current task", (
        {belief.scope for belief in state.preference_beliefs} == {PreferenceScope.TASK}
    ))
    checks.check("the event stream contains no long-term event", all(
        event.scope is not PreferenceScope.LONG_TERM
        for event in store.get_all_events(USER_KEY)
    ))
    print("  durable-memory write path: none -- every event is task-scoped and no "
          "long-term belief/constraint exists in the state.")

    # ---- Step 10: canonical JSON round-trip ------------------------------- #
    step(10, "state serialization + reconstruction (canonical JSON round-trip)")
    canonical = state.to_canonical_json()
    restored = UnifiedDecisionState.model_validate_json(canonical)
    checks.check("JSON round-trip reproduces the identical state model", restored == state)
    checks.check("JSON round-trip reproduces the identical state digest",
                 restored.state_digest() == state.state_digest())
    result["state_json"] = json.loads(canonical)
    result["canonical_state"] = canonical

    # ---- Step 11 (part A): replay from the still-open store --------------- #
    step(11, "deterministic replay from the event store (digest equality)")
    replay = manager.replay_state(USER_KEY, TASK_ID)
    replayed_digest = replay.state_digest()
    incremental_digest = state.state_digest()
    restored_digest = restored.state_digest()
    checks.check("incremental state == full replay (model)", replay == state)
    checks.check("incremental state == full replay (state digest)",
                 replayed_digest == incremental_digest)
    checks.check("JSON round-trip digest == incremental digest",
                 restored_digest == incremental_digest)

    result["projections"] = {
        "for_agent": projection_payload(for_agent(state)),
        "for_ranker": projection_payload(for_ranker(state)),
        "for_constraints": projection_payload(for_constraints(state)),
    }
    result["digests"] = {
        "incremental": incremental_digest,
        "replayed": replayed_digest,
        "json_roundtrip": restored_digest,
    }
    print(f"  incremental state digest : {incremental_digest}")
    print(f"  replayed state digest    : {replayed_digest}")
    print(f"  JSON round-trip digest   : {restored_digest}")
    return state


# --------------------------------------------------------------------------- #
# Step 11 (part B): replay from a fresh loader over the persisted database
# --------------------------------------------------------------------------- #


def verify_fresh_loader(
    database: Path,
    checks: Checks,
    result: dict[str, Any],
) -> tuple[str, str]:
    """Reopen the persisted store and replay twice; return both digests."""
    store = SQLiteEventStore(database)
    try:
        manager = StateManager(store)
        first = manager.replay_state(USER_KEY, TASK_ID)
        second = manager.replay_state(USER_KEY, TASK_ID)
        events = store.get_events(TASK_ID, USER_KEY)
    finally:
        store.close()

    first_digest = first.state_digest()
    second_digest = second.state_digest()
    checks.check("fresh loader replays the same effective state digest",
                 first_digest == second_digest)
    checks.check("fresh loader replay equals the incrementally built digest",
                 first_digest == result["digests"]["incremental"])
    checks.check("the persisted stream is sequence-ordered 1..N", (
        [event.sequence for event in events] == list(range(1, len(events) + 1))
        and all(event.user_key == USER_KEY for event in events)
        and all(event.task_id == TASK_ID for event in events)
    ))
    checks.check("every stored event payload round-trips to the same payload hash",
                 all(
                     new_event(
                         event_type=event.event_type,
                         event_id=event.event_id,
                         user_key=event.user_key,
                         task_id=event.task_id,
                         session_id=event.session_id,
                         actor=event.actor,
                         timestamp=event.timestamp,
                         payload=event.payload,
                         source=event.source,
                     ).payload_hash()
                     == event.payload_hash()
                     for event in events
                 ))
    result["digests"]["fresh_loader_replay"] = first_digest
    return first_digest, second_digest


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #


def run(database: Path) -> tuple[bool, dict[str, Any]]:
    """Run the whole scenario against ``database``; return (passed, json payload)."""
    checks = Checks()
    factory = EventFactory()
    result: dict[str, Any] = {
        "milestone": "M5",
        "scenario": "hiking-backpack",
        "user_key": USER_KEY,
        "session_id": SESSION_ID,
        "task_id": TASK_ID,
        "event_time": EVENT_TS,
    }

    # Steps 1-10 plus the replay part of step 11, on one manager.
    store = SQLiteEventStore(database)
    try:
        manager = StateManager(store)
        final_state = run_steps_1_to_10(manager, store, checks, factory, result)
    finally:
        store.close()

    # Step 11 (part B): a completely fresh loader over the persisted database.
    fresh_digest, _ = verify_fresh_loader(database, checks, result)

    # Deterministic event log for the summary / JSON artifact.
    result["events"] = [
        {"event_id": event_id, "event_type": event_type, "actor": actor}
        for event_id, event_type, actor in factory.applied
    ]
    result["event_count"] = len(factory.applied)
    result["state_version"] = final_state.state_version
    result["checks"] = checks.as_list()
    result["passed"] = checks.passed
    result["digests"]["fresh_loader_replay"] = fresh_digest

    # ---- Final readable summary block ------------------------------------ #
    print("\n" + "=" * 78)
    print("M5-MVP unified decision state smoke -- hiking-backpack scenario")
    print("=" * 78)
    print(f"  user_key             : {USER_KEY}")
    print(f"  session_id           : {SESSION_ID}")
    print(f"  task_id              : {TASK_ID}")
    print(f"  event timestamp base : {EVENT_TS}")
    print(f"  events appended      : {len(factory.applied)}")
    print("  event stream         :")
    for event_id, event_type, actor in factory.applied:
        print(f"    - {event_id:<22} {event_type:<32} actor={actor}")
    print(f"  state digest (incremental) : {result['digests']['incremental']}")
    print(f"  state digest (replayed)    : {result['digests']['replayed']}")
    print(f"  state digest (json)        : {result['digests']['json_roundtrip']}")
    print(f"  state digest (fresh loader): {result['digests']['fresh_loader_replay']}")
    print("  final projections (canonical JSON):")
    for name, payload in result["projections"].items():
        print(f"    {name:<16}: {canonical_json(payload)}")
    print(f"  checks               : {checks.passed_count}/{len(checks.results)} passed")
    print(f"\nSMOKE: {'PASS' if checks.passed else 'FAIL'}")
    return checks.passed, result


def main(argv: list[str] | None = None) -> int:
    """Run the M5 smoke; returns 0 only when every gate passes."""
    parser = argparse.ArgumentParser(
        description="M5-MVP unified decision state smoke (hiking-backpack scenario)"
    )
    parser.add_argument(
        "--db",
        type=Path,
        default=None,
        help="SQLite event-store path (default: a temporary database that is removed)",
    )
    parser.add_argument(
        "--json",
        type=Path,
        default=None,
        help="write the complete machine-readable result to this path",
    )
    args = parser.parse_args(argv)

    print("=" * 78)
    print("M5-MVP unified decision state smoke")
    print("Offline and deterministic: no model checkpoint, dataset, network or LLM.")
    print("=" * 78)

    cleanup_root: Path | None = None
    if args.db is None:
        cleanup_root = Path(tempfile.gettempdir()) / (
            f"agentrecx_m5_smoke_{os.getpid()}_{time.time_ns()}"
        )
        cleanup_root.mkdir()
        database = cleanup_root / "decision_state.sqlite3"
        database_label = "temporary database (removed on exit)"
    else:
        database = args.db.expanduser().resolve()
        database.parent.mkdir(parents=True, exist_ok=True)
        database_label = str(database)

    print(f"database: {database_label}")
    try:
        try:
            passed, result = run(database)
        except Exception:  # pragma: no cover - defensive smoke boundary
            traceback.print_exc()
            print("\nSMOKE: FAIL (unexpected error)")
            return 1

        if args.json is not None:
            args.json.parent.mkdir(parents=True, exist_ok=True)
            args.json.write_text(
                json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
            print(f"\nwrote {args.json}")
        return 0 if passed else 1
    finally:
        if cleanup_root is not None:
            shutil.rmtree(cleanup_root, ignore_errors=True)


if __name__ == "__main__":  # pragma: no cover - manual invocation
    raise SystemExit(main())

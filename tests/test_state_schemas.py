"""Contract tests for the M5-MVP state/event schema layer.

These tests pin the *contract* rather than one code path: the closed event vocabulary,
the declarative actor-authority table, payload selection, redaction/bounds, the absence
of product/score fields from trusted references, deterministic ids and the canonical
state digest.  Reducer behaviour and projections are covered by their own test modules;
this file is the Lead-owned integration glue for the frozen schema.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from recommendation.state.decision_state import (  # noqa: E402
    CandidatePoolRef,
    PreferenceBelief,
    RankingSnapshotRef,
    UnifiedDecisionState,
    empty_state,
    preference_belief_id,
)
from recommendation.state.events import (  # noqa: E402
    EVENT_ACTORS,
    EVENT_PAYLOAD_MODELS,
    EventActor,
    EventType,
    StateEvent,
    StateEventPayload,
    UserPreferenceStated,
    new_event,
    payload_model_for,
)
from recommendation.state.schemas import (  # noqa: E402
    SECRET_REDACTION,
    BeliefOrigin,
    BeliefStrength,
    PreferenceKind,
    PreferenceLifecycle,
    PreferencePolarity,
    PreferenceScope,
    PreferenceStatus,
)
from tests.state_fixtures import (  # noqa: E402
    DEFAULT_TS,
    candidate_pool_event,
    comparison_event,
    context_event,
    hypothesis_event,
    preference_stated_event,
)


# --------------------------------------------------------------------------- #
# Event registry and payload selection
# --------------------------------------------------------------------------- #


def test_event_registry_has_exactly_one_payload_and_actor_rule_per_type() -> None:
    assert set(EVENT_PAYLOAD_MODELS) == set(EventType)
    assert set(EVENT_ACTORS) == set(EventType)
    for payload_model in EVENT_PAYLOAD_MODELS.values():
        assert issubclass(payload_model, StateEventPayload)
    for actors in EVENT_ACTORS.values():
        assert actors
        assert actors <= set(EventActor)


def test_payload_model_is_selected_by_event_type() -> None:
    event = hypothesis_event()
    assert payload_model_for("preference_hypothesis_proposed") is type(event.payload)
    with pytest.raises(ValueError):
        payload_model_for("not_an_event_type")


def test_mismatched_payload_fields_are_rejected_for_the_selected_type() -> None:
    with pytest.raises(ValidationError):
        StateEvent(
            event_id="evt-mismatch",
            user_key="u",
            task_id="t",
            event_type="user_preference_stated",
            payload={"kind": "feature", "value": "x", "source_text": "y", "score": 0.9},
            actor="user",
            timestamp=DEFAULT_TS,
        )


def test_payload_instance_of_another_event_type_is_rejected() -> None:
    foreign_payload = UserPreferenceStated(kind="feature", value="x", source_text="y")
    with pytest.raises(ValidationError):
        StateEvent(
            event_id="evt-foreign",
            user_key="u",
            task_id="t",
            event_type="preference_hypothesis_proposed",
            payload=foreign_payload,
            actor="agent",
            timestamp=DEFAULT_TS,
        )


# --------------------------------------------------------------------------- #
# Actor authority
# --------------------------------------------------------------------------- #


def test_actor_authority_table_is_enforced() -> None:
    # A user cannot propose a system hypothesis.
    with pytest.raises(ValueError):
        hypothesis_event(actor="user")
    # The agent cannot state a user preference or constraint.
    with pytest.raises(ValueError):
        preference_stated_event(actor="agent")
    # A user cannot reference a trusted candidate pool.
    with pytest.raises(ValueError):
        candidate_pool_event(actor="user")
    # A trusted component cannot confirm a user's hypothesis.
    with pytest.raises(ValueError):
        new_event(
            event_type="preference_confirmed",
            event_id="evt-actor",
            user_key="u",
            task_id="t",
            actor="trusted_component",
            timestamp=DEFAULT_TS,
            kind="feature",
            value="x",
            source_text="yes",
        )


def test_unknown_event_schema_version_is_rejected() -> None:
    payload = hypothesis_event().payload
    with pytest.raises(ValidationError):
        StateEvent(
            event_id="evt-version",
            user_key="u",
            task_id="t",
            event_type="preference_hypothesis_proposed",
            payload=payload,
            actor="agent",
            schema_version=2,
            timestamp=DEFAULT_TS,
        )


# --------------------------------------------------------------------------- #
# Closed payloads, read-only references, bounds and redaction
# --------------------------------------------------------------------------- #


def test_extra_payload_fields_are_rejected() -> None:
    with pytest.raises(ValidationError):
        StateEvent(
            event_id="evt-extra",
            user_key="u",
            task_id="t",
            event_type="candidate_pool_referenced",
            payload={
                "ref": "pool-1",
                "source": "recommendation_tool",
                "candidate_count": 3,
                "product_ids": ["B0001", "B0002"],
            },
            actor="trusted_component",
            timestamp=DEFAULT_TS,
        )


def test_trusted_references_carry_no_product_identity_or_score_fields() -> None:
    forbidden = {
        "product_id",
        "product_ids",
        "item_id",
        "item_ids",
        "parent_asin",
        "asin",
        "score",
        "scores",
        "raw_score",
        "rank",
        "fused_score",
    }
    for model in (CandidatePoolRef, RankingSnapshotRef):
        fields = set(model.model_fields)
        assert not (fields & forbidden), f"{model.__name__} exposes {fields & forbidden}"
        assert "read_only" in fields


def test_read_only_references_cannot_be_constructed_as_writable() -> None:
    with pytest.raises(ValidationError):
        CandidatePoolRef(ref="pool-1", source="tool", candidate_count=1, read_only=False)
    with pytest.raises(ValidationError):
        RankingSnapshotRef(
            ref="snap-1", source="engine", method="rrf_rank_fusion",
            candidate_count=1, read_only=False,
        )


def test_reference_digest_must_be_a_sha256_hex_string() -> None:
    candidate_pool_event(digest="a" * 64)
    with pytest.raises(ValueError):
        candidate_pool_event(digest="A" * 64)
    with pytest.raises(ValueError):
        candidate_pool_event(digest="too-short")


def test_raw_user_text_is_bounded_and_credentials_are_redacted() -> None:
    long_event = context_event(request_text="backpack " * 400)
    assert len(long_event.payload.request_text) <= 1000

    secret_event = preference_stated_event(
        source_text="my api key is sk-abcdefgh12345678 please remember it"
    )
    assert secret_event.payload.source_text == SECRET_REDACTION

    with pytest.raises(ValueError):
        preference_stated_event(value="sk-abcdefgh12345678")


def test_inference_cannot_be_long_term_or_strong() -> None:
    with pytest.raises(ValueError):
        hypothesis_event(scope="long_term")
    with pytest.raises(ValueError):
        hypothesis_event(strength="strong")
    moderate = hypothesis_event(strength="moderate")
    assert moderate.payload.strength is BeliefStrength.MODERATE


def test_comparison_selection_must_be_one_of_the_compared_refs() -> None:
    with pytest.raises(ValueError):
        comparison_event(compared_refs=("pool:a", "pool:b"), selected_ref="pool:z")


# --------------------------------------------------------------------------- #
# Identity and duplicate semantics
# --------------------------------------------------------------------------- #


def test_duplicate_identity_ignores_timestamp_but_not_payload() -> None:
    first = preference_stated_event(event_id="evt-dup", timestamp=DEFAULT_TS)
    retry = preference_stated_event(event_id="evt-dup", timestamp=DEFAULT_TS + 500)
    assert first.is_same_submission(retry)
    assert first.stored(7).semantic_fields() == retry.semantic_fields()

    different = preference_stated_event(event_id="evt-dup", value="comfort")
    assert not first.is_same_submission(different)


def test_belief_ids_are_content_deterministic_and_scope_aware() -> None:
    def belief_id(scope: PreferenceScope, task_id: str, value: str = "lightweight") -> str:
        return preference_belief_id(
            user_key="u",
            task_id=task_id,
            scope=scope,
            kind=PreferenceKind.FEATURE,
            value=value,
            polarity=PreferencePolarity.PREFER,
        )

    long_a = belief_id(PreferenceScope.LONG_TERM, "task-a")
    long_b = belief_id(PreferenceScope.LONG_TERM, "task-b")
    task_a = belief_id(PreferenceScope.TASK, "task-a")
    task_b = belief_id(PreferenceScope.TASK, "task-b")
    assert long_a == long_b  # long-term identity is user-level, not task-level
    assert task_a != task_b
    assert task_a != long_a
    # Value normalisation is case/whitespace insensitive.
    assert belief_id(PreferenceScope.TASK, "task-a", " Lightweight ") == task_a


# --------------------------------------------------------------------------- #
# Serialization and state digest
# --------------------------------------------------------------------------- #


def test_event_json_roundtrip_preserves_payload_type() -> None:
    event = hypothesis_event()
    restored = StateEvent.model_validate_json(event.model_dump_json())
    assert restored == event
    assert type(restored.payload) is type(event.payload)


def test_state_digest_is_stable_content_sensitive_and_serialization_safe() -> None:
    state = empty_state(user_key="u", task_id="t")
    digest = state.state_digest()
    restored = UnifiedDecisionState.model_validate_json(state.to_canonical_json())
    assert restored == state
    assert restored.state_digest() == digest

    belief = PreferenceBelief(
        belief_id=preference_belief_id(
            user_key="u",
            task_id="t",
            scope=PreferenceScope.TASK,
            kind=PreferenceKind.FEATURE,
            value="lightweight",
            polarity=PreferencePolarity.PREFER,
        ),
        kind=PreferenceKind.FEATURE,
        value="lightweight",
        lifecycle=PreferenceLifecycle.INFERRED_HYPOTHESIS,
        status=PreferenceStatus.ACTIVE,
        scope=PreferenceScope.TASK,
        strength=BeliefStrength.WEAK,
        origin=BeliefOrigin.SYSTEM_INFERRED,
        supporting_event_ids=("evt-1",),
    )
    changed = state.model_copy(update={"preference_beliefs": (belief,)})
    assert changed.state_digest() != digest


def test_state_and_event_versions_are_frozen() -> None:
    with pytest.raises(ValidationError):
        UnifiedDecisionState(user_key="u", task_id="t", state_version=99)
    event = hypothesis_event()
    with pytest.raises(ValidationError):
        StateEvent(
            event_id=event.event_id,
            user_key=event.user_key,
            task_id=event.task_id,
            event_type=event.event_type,
            payload=event.payload,
            actor=event.actor,
            schema_version=99,
            timestamp=event.timestamp,
        )


def test_blank_identities_are_refused() -> None:
    with pytest.raises(ValidationError):
        empty_state(user_key="   ", task_id="t")
    with pytest.raises(ValidationError):
        empty_state(user_key="u", task_id="")
    with pytest.raises(ValueError):
        preference_stated_event(user_key="   ")


def test_event_scope_property_matches_payload_scope() -> None:
    long_term = preference_stated_event(scope="long_term")
    task = preference_stated_event(scope="task")
    assert long_term.scope is PreferenceScope.LONG_TERM
    assert long_term.applies_long_term is True
    assert task.scope is PreferenceScope.TASK
    assert task.applies_long_term is False
    context = context_event()
    assert context.scope is None
    assert context.applies_long_term is False


def test_timestamp_must_be_finite_and_non_negative() -> None:
    for invalid in (-1.0, math.nan, math.inf):
        with pytest.raises(ValueError):
            preference_stated_event(timestamp=invalid)


def test_new_event_rejects_payload_and_keyword_fields_together() -> None:
    with pytest.raises(TypeError):
        new_event(
            event_type="user_preference_stated",
            event_id="evt-x",
            user_key="u",
            task_id="t",
            actor="user",
            timestamp=DEFAULT_TS,
            payload={"kind": "feature", "value": "a", "source_text": "b"},
            value="a",
        )


def test_snapshot_schema_version_roundtrips_into_the_state(tmp_path: Path) -> None:
    """M5.2 snapshot references carry a bounded trustworthy snapshot-schema version.

    The version is snapshot metadata (the trusted engine's contract), distinct from the
    event schema version; it must survive event -> state -> JSON serialization and the
    reference must stay read-only.
    """
    from recommendation.state.event_store import SQLiteEventStore
    from recommendation.state.state_manager import StateManager

    with SQLiteEventStore(tmp_path / "snapshot_version.sqlite3") as store:
        manager = StateManager(store)
        manager.apply_event(
            new_event(
                event_type="candidate_pool_referenced",
                event_id="evt-cp-v2",
                user_key="u",
                task_id="t",
                actor="trusted_component",
                timestamp=DEFAULT_TS,
                payload={
                    "ref": "cp_v2_reference",
                    "source": "trusted_pipeline",
                    "candidate_count": 3,
                    "digest": "b" * 64,
                    "snapshot_schema_version": 2,
                },
            )
        )
        state = manager.replay_state("u", "t")
        pool = state.candidate_pool_ref
        assert pool is not None
        assert pool.snapshot_schema_version == 2
        assert pool.read_only is True
        assert pool.ref == "cp_v2_reference"
        assert pool.digest == "b" * 64

        rating = manager.apply_event(
            new_event(
                event_type="ranking_snapshot_referenced",
                event_id="evt-rs-v1",
                user_key="u",
                task_id="t",
                actor="trusted_component",
                timestamp=DEFAULT_TS,
                payload={
                    "ref": "rs_v1_reference",
                    "source": "trusted_pipeline",
                    "method": "presented_order",
                    "candidate_count": 3,
                    "digest": "c" * 64,
                },
            )
        )
        assert rating.ranking_snapshot_ref is not None
        assert rating.ranking_snapshot_ref.snapshot_schema_version == 1
        restored = UnifiedDecisionState.model_validate_json(rating.to_canonical_json())
        assert restored.ranking_snapshot_ref is not None
        assert restored.ranking_snapshot_ref.snapshot_schema_version == 1

    for invalid in (0, 1001):
        with pytest.raises(ValidationError):
            CandidatePoolRef(
                ref="pool",
                source="tool",
                candidate_count=1,
                snapshot_schema_version=invalid,
            )


# --------------------------------------------------------------------------- #
# Reuse parity with the accepted control-plane enums
# --------------------------------------------------------------------------- #


def test_scope_and_lifecycle_match_the_accepted_control_plane() -> None:
    """The duplicated enums must stay value-for-value identical to the accepted ones.

    ``recommendation.control.task_state`` is only importable with the full model stack
    (it lives behind ``recommendation/control/__init__.py``); skip cleanly when the
    stack is unavailable, which is also why the duplication exists at all.
    """
    accepted = pytest.importorskip(
        "recommendation.control.task_state",
        reason="control-plane enums need the model stack",
    )
    for scope in PreferenceScope:
        legacy_scope = accepted.PreferenceScope(scope.value)
        assert legacy_scope.value == scope.value
        assert legacy_scope.is_ephemeral == scope.is_ephemeral
    for lifecycle in PreferenceLifecycle:
        legacy_lifecycle = accepted.PreferenceLifecycle(lifecycle.value)
        assert legacy_lifecycle.value == lifecycle.value
        assert legacy_lifecycle.may_be_committed == lifecycle.may_be_committed
        assert legacy_lifecycle.is_soft_evidence == lifecycle.is_soft_evidence

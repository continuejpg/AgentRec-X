"""Milestone 11 session-layer tests: profiles, sessions, turns, isolation, reset.

Fully offline: no checkpoint, no catalogue artifact, no network and no browser.  The
session layer is pure application logic, so it is tested directly rather than only
through HTTP (the HTTP contract is covered by ``tests/test_demo_api.py``).
"""

from __future__ import annotations

import json
import sys
import threading
import uuid
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tests.demo_fixture import PROFILE_A, PROFILE_B, demo_profiles  # noqa: E402
from recommendation.demo import (  # noqa: E402
    DEFAULT_MAX_SESSIONS,
    DemoProfile,
    DemoSessionManager,
    SessionCapacityExceeded,
    UnknownProfile,
    UnknownSession,
    build_demo_profiles,
    demo_profiles_from_artifact,
    looks_like_recommendation,
)
from recommendation.demo.decision import DIRECT_RESPONSE_TEXT, DemoDecisionModel  # noqa: E402


# --------------------------------------------------------------------------- #
# Demo profiles
# --------------------------------------------------------------------------- #


def synthetic_sequences(tmp_path: Path, records: list[tuple[int, list[str]]]) -> Path:
    """Write a sequences artifact with the accepted stored shape.

    The real artifact is ~348 MB, so the unit tests use this small stand-in and the
    formal smoke verifies the same rule against the real file.
    """
    payload = {
        "format": "agentrecx.sequences.v1",
        "category": "Sports_and_Outdoors",
        "sequences": [
            {"user_id": f"user-{index}", "user_int_id": index, "parent_asins": asins}
            for index, asins in records
        ],
    }
    path = tmp_path / "sequences.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_demo_profiles_come_from_the_accepted_sequences_artifact(tmp_path: Path) -> None:
    """The fixed profile set is derived deterministically, in stored record order."""
    artifact = synthetic_sequences(
        tmp_path,
        [
            (1, ["a1", "a2", "a3", "a4", "a5"]),
            (2, ["b1", "b2", "b3", "b4", "b5"]),
            (3, ["c1", "c2", "c3", "c4", "c5"]),
        ],
    )
    first = demo_profiles_from_artifact(count=3, sequences_file=artifact)
    second = demo_profiles_from_artifact(count=3, sequences_file=artifact)

    assert [p.profile_id for p in first] == [PROFILE_A, PROFILE_B, "demo-user-3"]
    assert [p.source_user_int_id for p in first] == [1, 2, 3]
    assert [p.trusted_user_history for p in first] == [p.trusted_user_history for p in second]
    assert build_demo_profiles(sequences_file=artifact) == build_demo_profiles(
        sequences_file=artifact
    )


def test_demo_profile_history_excludes_the_leave_one_out_test_target(tmp_path: Path) -> None:
    """The supplied history is the training prefix plus the validation target only."""
    artifact = synthetic_sequences(tmp_path, [(7, ["t1", "t2", "t3", "t4", "t5", "test-target"])])
    profile = demo_profiles_from_artifact(count=1, sequences_file=artifact)[0]

    assert profile.source_user_int_id == 7
    assert profile.trusted_user_history == ("t1", "t2", "t3", "t4", "t5")
    assert "test-target" not in profile.trusted_user_history
    assert profile.source_length == 6


def test_demo_profile_selection_skips_records_below_the_minimum_length(tmp_path: Path) -> None:
    artifact = synthetic_sequences(
        tmp_path,
        [
            (1, ["short1", "short2", "short3"]),
            (2, ["ok1", "ok2", "ok3", "ok4", "ok5"]),
        ],
    )
    profile = demo_profiles_from_artifact(count=1, sequences_file=artifact)[0]
    assert profile.source_user_int_id == 2
    # The rule is ``parent_asins[:-2] + [parent_asins[-2]]``: training prefix plus the
    # validation target, with the leave-one-out test target excluded.
    assert profile.trusted_user_history == ("ok1", "ok2", "ok3", "ok4")


def test_demo_profiles_are_deterministic_and_distinct() -> None:
    """Each profile gets a distinct id, display name and history; no real user is named."""
    profiles = demo_profiles()
    assert set(profiles) == {PROFILE_A, PROFILE_B}
    assert len({p.trusted_user_history for p in profiles.values()}) == 2
    assert len({p.display_name for p in profiles.values()}) == 2
    for profile in profiles.values():
        assert profile.history_length == 3
        assert profile.history_distinct == 3


def test_missing_sequences_artifact_is_reported_clearly(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        demo_profiles_from_artifact(sequences_file=tmp_path / "absent.json")


def test_too_few_eligible_records_is_reported_clearly(tmp_path: Path) -> None:
    artifact = tmp_path / "tiny.json"
    artifact.write_text(
        json.dumps({"sequences": [{"user_id": "u", "user_int_id": 1, "parent_asins": ["a", "b"]}]}),
        encoding="utf-8",
    )
    with pytest.raises(LookupError):
        demo_profiles_from_artifact(count=1, sequences_file=artifact)


# --------------------------------------------------------------------------- #
# Session ids, user keys and isolation
# --------------------------------------------------------------------------- #


def test_session_ids_are_opaque_uuid4_tokens() -> None:
    """A session id is a capability token, not a path, key or sequential number."""
    manager = DemoSessionManager(demo_profiles())
    session = manager.create(PROFILE_A)
    parsed = uuid.UUID(session.session_id)
    assert parsed.version == 4
    assert "/" not in session.session_id
    assert session.session_id != session.profile_id


def test_sessions_are_isolated_even_on_the_same_profile() -> None:
    """Distinct sessions never share a preference-memory namespace."""
    manager = DemoSessionManager(demo_profiles())
    first = manager.create(PROFILE_A)
    second = manager.create(PROFILE_A)

    assert first.session_id != second.session_id
    assert first.user_key != second.user_key
    assert first.user_key not in second.user_key
    assert first.profile_id == second.profile_id == PROFILE_A
    # Same profile, therefore the same trusted history -- but never the same memory.
    assert first.trusted_user_history == second.trusted_user_history


def test_user_key_never_encodes_the_profile_id() -> None:
    """Keying memory by profile would let two sessions contaminate each other."""
    manager = DemoSessionManager(demo_profiles())
    session = manager.create(PROFILE_A)
    assert PROFILE_A not in session.user_key
    assert session.session_id in session.user_key


def test_trusted_history_is_application_owned_and_copied() -> None:
    """The session holds its own copy; mutating the profile cannot change it."""
    profiles = demo_profiles()
    manager = DemoSessionManager(profiles)
    session = manager.create(PROFILE_A)
    assert session.trusted_user_history == profiles[PROFILE_A].trusted_user_history
    assert isinstance(session.trusted_user_history, tuple)


def test_unknown_profile_is_rejected() -> None:
    manager = DemoSessionManager(demo_profiles())
    with pytest.raises(UnknownProfile):
        manager.create("not-a-profile")


# --------------------------------------------------------------------------- #
# Turn allocation
# --------------------------------------------------------------------------- #


def test_turn_ids_are_server_owned_and_monotonic() -> None:
    manager = DemoSessionManager(demo_profiles())
    session = manager.create(PROFILE_A)
    seen: list[str] = []
    for _ in range(3):
        with manager.turn(session.session_id) as allocation:
            seen.append(allocation.turn_id)
            assert allocation.turn_number == len(seen)

    assert seen == [f"{session.session_id}:000001", f"{session.session_id}:000002",
                    f"{session.session_id}:000003"]
    assert len(set(seen)) == 3
    assert manager.get(session.session_id).turns_completed == 3


def test_turn_id_is_consumed_even_when_the_turn_fails() -> None:
    """A failed turn must never free its identifier for a retry to reuse."""
    manager = DemoSessionManager(demo_profiles())
    session = manager.create(PROFILE_A)

    with pytest.raises(RuntimeError):
        with manager.turn(session.session_id):
            raise RuntimeError("backend exploded")

    with manager.turn(session.session_id) as allocation:
        assert allocation.turn_number == 2


def test_concurrent_turns_on_one_session_get_distinct_ids() -> None:
    """The per-session lock serialises turns: no shared ids, no lost sequence."""
    manager = DemoSessionManager(demo_profiles())
    session = manager.create(PROFILE_A)
    observed: list[str] = []
    errors: list[BaseException] = []
    barrier = threading.Barrier(8)

    def run() -> None:
        try:
            barrier.wait(timeout=10)
            with manager.turn(session.session_id) as allocation:
                observed.append(allocation.turn_id)
        except BaseException as exc:  # noqa: BLE001 - collected and asserted
            errors.append(exc)

    threads = [threading.Thread(target=run) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=15)

    assert errors == []
    assert len(observed) == 8
    assert len(set(observed)) == 8, "two concurrent turns received the same turn id"
    assert manager.get(session.session_id).turns_completed == 8


def test_different_sessions_do_not_block_each_other() -> None:
    """There is no global lock: session A's turn lock must not gate session B."""
    manager = DemoSessionManager(demo_profiles())
    first = manager.create(PROFILE_A)
    second = manager.create(PROFILE_B)
    assert first.lock is not second.lock

    progress: list[str] = []
    release = threading.Event()

    def hold_first() -> None:
        with manager.turn(first.session_id):
            progress.append("first-held")
            release.wait(timeout=10)

    holder = threading.Thread(target=hold_first)
    holder.start()
    # Wait until the first session's lock is definitely held.
    for _ in range(200):
        if progress:
            break
        threading.Event().wait(0.01)

    # The second session must complete while the first one is still inside its turn.
    with manager.turn(second.session_id) as allocation:
        progress.append("second-ran")
        assert allocation.turn_number == 1

    release.set()
    holder.join(timeout=10)
    assert progress == ["first-held", "second-ran"]


# --------------------------------------------------------------------------- #
# Reset
# --------------------------------------------------------------------------- #


def test_delete_removes_only_the_target_session() -> None:
    manager = DemoSessionManager(demo_profiles())
    first = manager.create(PROFILE_A)
    second = manager.create(PROFILE_B)
    with manager.turn(second.session_id) as allocation:
        second_turn = allocation.turn_id

    removed = manager.delete(first.session_id)

    assert removed.session_id == first.session_id
    with pytest.raises(UnknownSession):
        manager.get(first.session_id)
    survivor = manager.get(second.session_id)
    assert survivor.session_id == second.session_id
    assert survivor.turns_completed == 1
    assert second_turn == f"{second.session_id}:000001"
    assert len(manager) == 1


def test_reset_retires_the_namespace_and_never_reissues_it() -> None:
    """A reset namespace can never be handed to a later session."""
    manager = DemoSessionManager(demo_profiles())
    first = manager.create(PROFILE_A)
    manager.delete(first.session_id)
    assert manager.retired_user_keys() == (first.user_key,)

    for _ in range(20):
        later = manager.create(PROFILE_A)
        assert later.user_key != first.user_key
    assert first.user_key in manager.retired_user_keys()


def test_unknown_session_reset_is_reported() -> None:
    manager = DemoSessionManager(demo_profiles())
    with pytest.raises(UnknownSession):
        manager.delete("00000000-0000-4000-8000-000000000000")


def test_deleting_a_session_while_a_turn_is_in_flight_is_safe() -> None:
    """A reset waits for the in-flight turn and no later turn can slip in."""
    manager = DemoSessionManager(demo_profiles())
    session = manager.create(PROFILE_A)
    inside = threading.Event()
    finished: list[str] = []

    def long_turn() -> None:
        with manager.turn(session.session_id):
            inside.set()
            threading.Event().wait(0.15)
        finished.append("turn-done")

    worker = threading.Thread(target=long_turn)
    worker.start()
    inside.wait(timeout=5)
    manager.delete(session.session_id)
    worker.join(timeout=10)

    assert finished == ["turn-done"], "the reset did not wait for the in-flight turn"
    with pytest.raises(UnknownSession):
        with manager.turn(session.session_id):
            pass  # pragma: no cover - the reset already removed the session


# --------------------------------------------------------------------------- #
# Capacity and expiry
# --------------------------------------------------------------------------- #


def test_capacity_is_bounded_and_reported() -> None:
    manager = DemoSessionManager(demo_profiles(), max_sessions=2)
    manager.create(PROFILE_A)
    manager.create(PROFILE_B)
    with pytest.raises(SessionCapacityExceeded):
        manager.create(PROFILE_A)
    assert len(manager) == 2
    assert manager.max_sessions == 2


def test_reset_frees_a_capacity_slot() -> None:
    manager = DemoSessionManager(demo_profiles(), max_sessions=1)
    first = manager.create(PROFILE_A)
    with pytest.raises(SessionCapacityExceeded):
        manager.create(PROFILE_B)
    manager.delete(first.session_id)
    assert manager.create(PROFILE_B).profile_id == PROFILE_B


def test_default_capacity_is_documented() -> None:
    manager = DemoSessionManager(demo_profiles())
    assert manager.max_sessions == DEFAULT_MAX_SESSIONS
    assert manager.ttl_seconds is None, "expiry is opt-in, not silently enabled"


def test_expiry_is_opt_in_and_uses_the_injected_clock() -> None:
    now = {"value": 1000.0}
    manager = DemoSessionManager(
        demo_profiles(), ttl_seconds=60.0, clock=lambda: now["value"]
    )
    session = manager.create(PROFILE_A)
    assert manager.get(session.session_id).session_id == session.session_id

    now["value"] += 61.0
    with pytest.raises(UnknownSession):
        manager.get(session.session_id)
    # An expired session is retired, so its namespace is not reused either.
    assert session.user_key in manager.retired_user_keys()


def test_creating_a_session_sweeps_expired_ones() -> None:
    now = {"value": 0.0}
    manager = DemoSessionManager(demo_profiles(), max_sessions=1, ttl_seconds=10.0,
                                 clock=lambda: now["value"])
    manager.create(PROFILE_A)
    now["value"] += 11.0
    assert manager.create(PROFILE_B).profile_id == PROFILE_B
    assert len(manager) == 1


def test_invalid_configuration_is_rejected() -> None:
    with pytest.raises(ValueError):
        DemoSessionManager(demo_profiles(), max_sessions=0)
    with pytest.raises(ValueError):
        DemoSessionManager(demo_profiles(), ttl_seconds=0.0)


# --------------------------------------------------------------------------- #
# Deterministic decision model
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "message",
    [
        "Recommend some products.",
        "recommend again",
        "Show me hiking gear.",
        "I'm looking for a water filter.",
        "any ideas?",
        "what else do you have",
    ],
)
def test_recommendation_trigger_phrases_route_to_recommend(message: str) -> None:
    assert looks_like_recommendation(message) is True


@pytest.mark.parametrize(
    "message",
    [
        "I don't want red.",
        "I prefer black.",
        "I don't care about color anymore.",
        "Actually, I prefer blue instead.",
        "hello",
        "thanks",
    ],
)
def test_non_recommendation_phrases_route_to_direct(message: str) -> None:
    assert looks_like_recommendation(message) is False


def test_decision_model_is_deterministic_and_offline() -> None:
    from recommendation.agent import build_decision_messages

    model = DemoDecisionModel(k=7)
    messages = build_decision_messages("Recommend some products.")
    first = model.decide(messages)
    second = model.decide(messages)

    assert first.action.value == "recommend"
    assert first.requested_k == 7
    assert first == second

    direct = DemoDecisionModel(k=3).decide(build_decision_messages("I prefer black."))
    assert direct.action.value == "direct_response"
    assert direct.direct_response == DIRECT_RESPONSE_TEXT


def test_decision_model_rejects_an_invalid_k() -> None:
    for value in (0, 101, -1):
        with pytest.raises(ValueError):
            DemoDecisionModel(k=value)


def test_decision_model_only_sees_the_messages_it_is_given() -> None:
    """It has no other input: the prompt carries the user message and nothing else."""
    from recommendation.agent import build_decision_messages

    model = DemoDecisionModel(k=5)
    model.decide(build_decision_messages("Recommend some products."))
    roles = [message.role for message in model.calls[-1]]
    assert roles == ["system", "user"]
    assert model.calls[-1][1].content == "Recommend some products."


def test_profile_history_is_never_used_to_choose_a_route() -> None:
    """No argument of the decision model can carry a trusted history."""
    import inspect

    signature = inspect.signature(DemoDecisionModel.decide)
    assert list(signature.parameters) == ["self", "messages"]
    assert isinstance(demo_profiles()[PROFILE_A], DemoProfile)

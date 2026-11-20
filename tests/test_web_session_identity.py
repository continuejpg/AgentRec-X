"""Regression tests for Web Demo session identity (M1-C S1 / D-1).

D-1: ``Session.reset_turns()`` used to rewind ``_turn_counter`` and ``_reply_counter``
to zero.  The identifiers it then issued repeated ids that had already been handed to
M9, and M9 keys its event idempotency on ``(user_key, turn_id, kind, value, polarity)``.
A user who reset the session and re-stated a preference they had withdrawn was
therefore silently ignored: the new event resolved to an existing (REMOVED) memory id
and was counted as ``skipped_duplicates`` instead of becoming ACTIVE.

These tests exercise the real ``Session``/``SessionStore`` and the real M9
``PreferenceMemoryService`` + ``InMemoryPreferenceStore`` + ``RuleBasedPreferenceExtractor``.
No idempotency or write logic is mocked.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from recommendation.memory.extraction import RuleBasedPreferenceExtractor  # noqa: E402
from recommendation.memory.service import PreferenceMemoryService  # noqa: E402
from recommendation.memory.store import InMemoryPreferenceStore  # noqa: E402
from web_demo.sessions import (  # noqa: E402
    PendingClarification,
    Session,
    SessionStore,
    Turn,
)

# Both sentences were verified against the real RuleBasedPreferenceExtractor:
#   "I prefer black."                   -> preferences [(color, black, prefer)]
#   "I don't care about color anymore." -> removals   [(kind=color, value=None)]
PREF = "I prefer black."
DROP = "I don't care about color anymore."


def _service() -> PreferenceMemoryService:
    """The exact construction ShoppingCompanionMemory performs (memory.py:545-549)."""
    return PreferenceMemoryService(InMemoryPreferenceStore(), RuleBasedPreferenceExtractor())


def _active(svc: PreferenceMemoryService, user_key: str) -> list[tuple[str, str, str]]:
    return [
        (entry.value, entry.polarity.value, entry.status.value)
        for entry in svc.get_active_preferences(user_key).entries
    ]


# --------------------------------------------------------------------------- #
# Identity uniqueness across reset
# --------------------------------------------------------------------------- #


def test_turn_id_does_not_repeat_across_reset() -> None:
    """A reset clears the transcript, not the turn sequence."""
    store = SessionStore()
    session = store.create("profile_0049")

    first = session.next_turn_id()
    second = session.next_turn_id()
    session.reset_turns()
    after_reset = session.next_turn_id()

    assert after_reset not in {first, second}
    assert after_reset == f"{session.session_id}-t3"


def test_reply_id_does_not_repeat_across_reset() -> None:
    """A reset must never reissue a reply id already recorded by M9 (D-1)."""
    store = SessionStore()
    session = store.create("profile_0057")

    first = session.next_reply_turn_id()
    second = session.next_reply_turn_id()
    session.reset_turns()
    after_reset = session.next_reply_turn_id()

    assert after_reset not in {first, second}
    assert after_reset == f"{session.session_id}-r3"


# --------------------------------------------------------------------------- #
# Reset keeps its original business semantics
# --------------------------------------------------------------------------- #


def test_reset_preserves_session_semantics() -> None:
    """Reset still clears the transcript and keeps identity, behaviour and M9."""
    store = SessionStore()
    session = store.create("profile_0057", behavior_enabled=False)
    session_id = session.session_id

    turn_id = session.next_turn_id()
    session.add_turn(Turn(turn_id=turn_id, user_text="Recommend a laptop under 5000 for me."))
    session.pending_clarification = PendingClarification(
        payload={"clarification_id": "clar-x", "question": "Office or gaming?"},
        query="Recommend a laptop under 5000 for me.",
        turn_id=turn_id,
        reply_turn_id=session.next_reply_turn_id(),
    )
    session.record_recommendation(
        {"recommendation_id": "rec-reset-sample", "query": "q", "items": []},
        turn_id=turn_id,
        ranking_source="unit-test",
    )
    session.last_query = "Recommend a laptop under 5000 for me."

    # M9 is a separate store: reset must not touch it.
    service = _service()
    service.process_turn(
        user_key=session_id, user_message=PREF, turn_id=turn_id, now=1000.0
    )
    history_before = len(service.get_memory_history(session_id).entries)

    session.reset_turns()

    assert session.session_id == session_id
    assert session.turns == []
    assert session.pending_clarification is None
    assert session.recommendations == {}
    assert session.latest_recommendation_id is None
    assert session.last_query == ""
    assert session.behavior_enabled is False

    # M9 unchanged: same number of events, preference still ACTIVE.
    assert len(service.get_memory_history(session_id).entries) == history_before
    assert ("black", "prefer", "active") in _active(service, session_id)

    # The public HTTP view contract is intact.
    view = session.view()
    assert view["session_id"] == session_id
    assert view["turns"] == []
    assert view["pending_clarification"] is None
    assert view["behavior_enabled"] is False


# --------------------------------------------------------------------------- #
# Real M9: withdraw, reset, re-state
# --------------------------------------------------------------------------- #


def test_withdrawn_preference_can_be_restated_after_reset() -> None:
    """The D-1 scenario: withdraw, reset, state it again -> it must become ACTIVE."""
    service = _service()
    store = SessionStore()
    session = store.create("profile_0057")
    user_key = session.session_id

    first_reply_id = session.next_reply_turn_id()
    added = service.process_turn(
        user_key=user_key, user_message=PREF, turn_id=first_reply_id, now=1000.0
    )
    assert len(added.update.added) == 1
    original_memory_id = added.update.added[0].memory_id

    second_reply_id = session.next_reply_turn_id()
    withdrawn = service.process_turn(
        user_key=user_key, user_message=DROP, turn_id=second_reply_id, now=1001.0
    )
    assert len(withdrawn.update.removed) == 1
    assert ("black", "prefer", "active") not in _active(service, user_key)

    session.reset_turns()

    third_reply_id = session.next_reply_turn_id()
    assert third_reply_id not in {first_reply_id, second_reply_id}

    restated = service.process_turn(
        user_key=user_key, user_message=PREF, turn_id=third_reply_id, now=1002.0
    )

    assert len(restated.update.added) == 1, "the re-statement must create a new memory event"
    assert restated.update.skipped_duplicates == 0
    assert ("black", "prefer", "active") in _active(service, user_key)
    assert restated.update.added[0].memory_id != original_memory_id


# --------------------------------------------------------------------------- #
# Genuine retries must stay idempotent
# --------------------------------------------------------------------------- #


def test_genuine_retry_stays_idempotent() -> None:
    """Preserving identities must not weaken idempotency for a real retry."""
    service = _service()
    store = SessionStore()
    session = store.create("profile_0057")
    user_key = session.session_id
    turn_id = session.next_turn_id()

    first = service.process_turn(
        user_key=user_key, user_message=PREF, turn_id=turn_id, now=2000.0
    )
    history_after_first = len(service.get_memory_history(user_key).entries)

    retry = service.process_turn(
        user_key=user_key, user_message=PREF, turn_id=turn_id, now=2001.0
    )
    history_after_retry = len(service.get_memory_history(user_key).entries)

    assert len(first.update.added) == 1
    assert retry.update.already_processed is True
    assert len(retry.update.added) == 0
    assert history_after_retry == history_after_first == 1
    assert ("black", "prefer", "active") in _active(service, user_key)


# --------------------------------------------------------------------------- #
# Counter restoration (in-memory object only)
# --------------------------------------------------------------------------- #


def test_session_can_be_constructed_with_existing_counters() -> None:
    """A future store can restore counters through the existing dataclass fields.

    This proves only the in-memory construction capability; no cross-process
    persistence is implemented in this milestone.
    """
    session = Session(
        session_id="web-restore01",
        profile_id="profile_0057",
        _turn_counter=5,
        _reply_counter=7,
    )

    assert session.next_turn_id() == "web-restore01-t6"
    assert session.next_reply_turn_id() == "web-restore01-r8"

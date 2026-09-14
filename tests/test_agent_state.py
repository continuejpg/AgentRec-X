"""Agent input/state tests (Milestone 7B).

The graph state is where the trusted-history ownership rule lives, so these tests
pin the invariants that make the Milestone 7B boundary real:

* the application supplies the history, and it is validated once at entry;
* nothing about a run can rewrite it - order, duplicates and the caller's own
  container are all preserved;
* a run without history fails loudly instead of continuing.

Fully offline: no torch, no checkpoint, no catalog, no network, no provider SDK.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tests.agent_fakes import HISTORY, HISTORY_WITH_DUPLICATE  # noqa: E402
from recommendation.agent import (  # noqa: E402
    AgentInput,
    MalformedDecision,
    history_digest,
    new_agent_state,
    read_trusted_history,
)


# --------------------------------------------------------------------------- #
# Input validation
# --------------------------------------------------------------------------- #


def test_valid_input_is_accepted_and_stripped() -> None:
    """Whitespace is normalised once, at the application boundary."""
    agent_input = AgentInput(user_message="  recommend a tent  ", trusted_user_history=HISTORY)
    assert agent_input.user_message == "recommend a tent"
    assert agent_input.trusted_user_history == HISTORY


def test_input_is_frozen() -> None:
    """A validated input cannot be mutated by a later stage."""
    agent_input = AgentInput(user_message="hi", trusted_user_history=HISTORY)
    with pytest.raises(ValidationError):
        agent_input.user_message = "changed"  # type: ignore[misc]


def test_input_rejects_extra_fields() -> None:
    """The input schema has no slot for extra, model-supplied data."""
    with pytest.raises(ValidationError):
        AgentInput(  # type: ignore[call-arg]
            user_message="hi",
            trusted_user_history=HISTORY,
            history=["B000000009"],
        )


@pytest.mark.parametrize("message", ["", "   ", "\n\t "])
def test_input_rejects_blank_user_message(message: str) -> None:
    """A blank message is not a user turn."""
    with pytest.raises(ValidationError):
        AgentInput(user_message=message, trusted_user_history=HISTORY)


@pytest.mark.parametrize("history", [(), [], ["   "], [""], ["ok", "  "]])
def test_input_rejects_empty_or_blank_history(history: object) -> None:
    """History must be non-empty and every entry a non-blank parent_asin."""
    with pytest.raises(ValidationError):
        AgentInput(user_message="hi", trusted_user_history=history)  # type: ignore[arg-type]


def test_input_rejects_a_bare_string_as_history() -> None:
    """A string is a sequence of characters, not a history; it must be rejected."""
    with pytest.raises(ValidationError):
        AgentInput(user_message="hi", trusted_user_history="B000000001")  # type: ignore[arg-type]


def test_input_preserves_order_and_duplicates() -> None:
    """Chronology is data: the boundary never sorts or deduplicates."""
    agent_input = AgentInput(
        user_message="hi", trusted_user_history=HISTORY_WITH_DUPLICATE
    )
    assert agent_input.trusted_user_history == HISTORY_WITH_DUPLICATE


def test_input_never_mutates_the_callers_container() -> None:
    """The caller's list is copied, not adopted."""
    source = ["B000000001", "B000000002"]
    snapshot = list(source)
    AgentInput(user_message="hi", trusted_user_history=source)
    assert source == snapshot


# --------------------------------------------------------------------------- #
# Initial state
# --------------------------------------------------------------------------- #


def test_new_state_copies_the_trusted_history() -> None:
    """Initial graph state carries exactly the supplied trusted history."""
    agent_input = AgentInput(user_message="hi", trusted_user_history=HISTORY)
    state = new_agent_state(agent_input)
    assert state["user_message"] == "hi"
    assert state["trusted_user_history"] == HISTORY
    assert isinstance(state["trusted_user_history"], tuple)


def test_new_state_contains_no_decision_or_result_yet() -> None:
    """A fresh run has taken no route and produced no output."""
    state = new_agent_state(AgentInput(user_message="hi", trusted_user_history=HISTORY))
    for key in ("decision", "tool_result", "final_response", "route"):
        assert key not in state


# --------------------------------------------------------------------------- #
# Reading trusted history
# --------------------------------------------------------------------------- #


def test_read_trusted_history_returns_an_immutable_tuple() -> None:
    """Consumers get a tuple, so no downstream node can mutate the history in place."""
    history = read_trusted_history({"trusted_user_history": list(HISTORY)})
    assert history == HISTORY
    assert isinstance(history, tuple)


@pytest.mark.parametrize("state", [{}, {"trusted_user_history": ()}, {"trusted_user_history": []}])
def test_read_trusted_history_fails_loudly_when_absent(state: dict) -> None:
    """No fallback: a run without trusted history is an error."""
    with pytest.raises(MalformedDecision):
        read_trusted_history(state)  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# History digest (diagnostics only)
# --------------------------------------------------------------------------- #


def test_history_digest_is_deterministic_and_short() -> None:
    """The digest is stable across calls and short enough for a log line."""
    first = history_digest(HISTORY)
    assert first == history_digest(tuple(HISTORY))
    assert len(first) == 16


def test_history_digest_is_order_sensitive_and_content_sensitive() -> None:
    """Reordering or editing history changes the digest, so leaks would show up."""
    assert history_digest(HISTORY) != history_digest(tuple(reversed(HISTORY)))
    assert history_digest(HISTORY) != history_digest(HISTORY + ("B000000004",))
    assert history_digest(()) != history_digest(HISTORY)


def test_history_digest_does_not_reveal_history_items() -> None:
    """The digest must not contain the raw identifiers it summarises."""
    digest = history_digest(HISTORY)
    assert all(item not in digest for item in HISTORY)

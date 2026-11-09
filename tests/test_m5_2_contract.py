"""M5.2 Lead-owned contract tests for the correction/retraction extensions.

These pin the small typed extensions M5.2 adds on top of the frozen M5 state engine:
the general REPLACE correction flag, retraction selector resolution and the trusted
candidate/ranking snapshot reference fields.  Behavioural integration coverage lives in
the M5.2 acceptance suites; this file protects the state contract itself.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from recommendation.state.decision_state import empty_state  # noqa: E402
from recommendation.state.belief_updater import BeliefUpdater  # noqa: E402
from recommendation.state.events import UserPreferenceStated, new_event  # noqa: E402
from recommendation.state.schemas import (  # noqa: E402
    PreferenceKind,
    PreferenceLifecycle,
    PreferencePolarity,
    PreferenceStatus,
)

DEFAULT_TS = 1_700_000_000.0
USER = "user-a"
TASK = "task-1"


def _event(event_id: str, value: str, *, replace_same_kind: bool = False, replaces: str | None = None):
    return new_event(
        event_type="user_preference_stated",
        event_id=event_id,
        user_key=USER,
        task_id=TASK,
        actor="user",
        timestamp=DEFAULT_TS,
        kind=PreferenceKind.COLOR,
        value=value,
        polarity=PreferencePolarity.PREFER,
        scope="task",
        source_text=f"the user said {value}",
        replace_same_kind=replace_same_kind,
        replaces_value=replaces,
    )


def _state_after(*events):
    updater = BeliefUpdater()
    state = empty_state(user_key=USER, task_id=TASK)
    for event in events:
        state = updater.apply(state, event)
    return state


def test_general_replace_supersedes_the_same_kind_same_polarity_value() -> None:
    """An M9 REPLACE without a named target corrects the prior same-kind value."""
    state = _state_after(
        _event("evt-1", "black"),
        _event("evt-2", "blue", replace_same_kind=True),
    )
    by_value = {belief.value: belief for belief in state.preference_beliefs}
    assert by_value["black"].status is PreferenceStatus.SUPERSEDED
    assert by_value["black"].lifecycle is PreferenceLifecycle.EXPLICIT
    assert by_value["blue"].status is PreferenceStatus.ACTIVE
    assert by_value["blue"].lifecycle is PreferenceLifecycle.EXPLICIT
    # Supersession is recorded, not deleted: provenance survives.
    assert "evt-2" in by_value["black"].supporting_event_ids


def test_plain_add_does_not_supersede_a_sibling_value() -> None:
    """A second independent statement remains an ADD, not a correction."""
    state = _state_after(_event("evt-1", "black"), _event("evt-2", "blue"))
    active = {belief.value for belief in state.active_preference_beliefs}
    assert active == {"black", "blue"}


def test_named_target_correction_is_unchanged() -> None:
    """Named `replaces_value` behaviour from M5.1 is preserved."""
    state = _state_after(
        _event("evt-1", "black"),
        _event("evt-2", "blue", replaces="black"),
    )
    by_value = {belief.value: belief for belief in state.preference_beliefs}
    assert by_value["black"].status is PreferenceStatus.SUPERSEDED
    assert by_value["blue"].status is PreferenceStatus.ACTIVE


def test_correction_selector_mutual_exclusion_is_a_hard_error() -> None:
    with pytest.raises(ValidationError):
        UserPreferenceStated(
            kind=PreferenceKind.COLOR,
            value="blue",
            source_text="actually blue instead",
            replaces_value="black",
            replace_same_kind=True,
        )

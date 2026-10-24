"""The empty-result text must match what the eligibility view actually proved.

Three states are kept apart (Phase 3.3):

* every candidate was **disproved** - "no candidate satisfies" is proved;
* at least one candidate is **undecided** - only "no candidate could be verified" is
  proved, and silence must not be reported as a violation;
* **nothing was assessed** - no candidate was available to check.

The older bug (Phase 3) was a different one: an empty constrained run was rendered with the
candidate-exhaustion sentence.  These tests keep that regression pinned too.
"""

from __future__ import annotations

from recommendation.agent.decision import AgentDecision
from recommendation.agent.pipeline import finalize_stage
from recommendation.agent.rendering import NO_CANDIDATES_TEXT, build_constraint_empty_response
from recommendation.control.constraint_eligibility import ConstraintRequirement
from recommendation.control.grounded_reasoning import ConstraintKind
from recommendation.tools.schemas import RecommendationToolResult


class _ConfirmedView:
    requirements = (ConstraintRequirement(kind=ConstraintKind.PRICE_MAX, expected="20"),)
    assessed_identities = ("B000000001", "B000000002", "B000000003")
    verified_eligible_count = 0
    ineligible_count = 3
    unresolved_count = 0


class _UnresolvedView(_ConfirmedView):
    ineligible_count = 2
    unresolved_count = 1


class _EmptyView(_ConfirmedView):
    assessed_identities = ()
    ineligible_count = 0
    unresolved_count = 0


class _Projection:
    is_empty = True

    def __init__(self, view):
        self.view = view


def _state(projection=None):
    state = {
        "decision": AgentDecision(action="recommend"),
        "trusted_user_history": ("B000000009",),
        "tool_result": RecommendationToolResult(
            recommendations=[],
            requested_k=5,
            returned_k=0,
            history_length=1,
            effective_history_length=1,
            history_truncated=False,
        ),
    }
    if projection is not None:
        state["constraint_projection"] = projection
    return state


def test_confirmed_no_match_states_that_no_candidate_satisfies() -> None:
    text = build_constraint_empty_response(_ConfirmedView())
    assert "No candidate satisfies the active constraints (price_max=20)." in text
    assert "Candidates assessed: 3" in text
    assert "verified eligible: 0" in text
    assert "ineligible: 3" in text
    assert "unresolved (catalogue cannot decide): 0" in text
    assert "not an exhausted interaction history" in text
    assert text != NO_CANDIDATES_TEXT


def test_unresolved_candidates_are_not_reported_as_confirmed_absence() -> None:
    text = build_constraint_empty_response(_UnresolvedView())
    assert "No candidate could be verified as satisfying the active constraints" in text
    assert "Candidates assessed: 3" in text
    assert "ineligible: 2" in text
    assert "unresolved (catalogue cannot decide): 1" in text
    assert "1 candidate(s) remain undecided" in text
    assert "missing facts are not treated as violations" in text
    assert "No candidate satisfies" not in text
    assert "not a confirmed absence of a matching product" in text
    assert text != NO_CANDIDATES_TEXT


def test_an_empty_candidate_list_says_nothing_was_available_to_check() -> None:
    text = build_constraint_empty_response(_EmptyView())
    assert "No candidate was available to evaluate against the active constraints" in text
    assert "Candidates assessed: 0" in text
    assert "no candidate was retrieved to verify" in text
    assert "No candidate satisfies" not in text
    assert text != NO_CANDIDATES_TEXT


def test_finalize_renders_the_confirmed_reason() -> None:
    out = finalize_stage(_state(_Projection(_ConfirmedView())))
    assert out["route"] == "recommend"
    assert "No candidate satisfies the active constraints" in out["final_response"]
    assert out["final_response"] != NO_CANDIDATES_TEXT


def test_finalize_keeps_the_unverified_semantics_when_candidates_are_undecided() -> None:
    out = finalize_stage(_state(_Projection(_UnresolvedView())))
    assert out["route"] == "recommend"
    assert "No candidate could be verified" in out["final_response"]
    assert "No candidate satisfies" not in out["final_response"]


def test_finalize_keeps_the_exhaustion_text_when_no_constraint_is_active() -> None:
    out = finalize_stage(_state())
    assert out["route"] == "recommend"
    assert out["final_response"] == NO_CANDIDATES_TEXT

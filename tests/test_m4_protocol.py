"""M4 preregistered regression tests: action-space projection, guard, and frozen rules.

Covers preregistration §7.3 (provider-visible schema), §8 C4-4 (the six deviation inputs),
§9 (termination rules), §11 (source-order invariance) and §5 (query-rule determinism).

These are the tests the preregistration promises. They are deliberately written against the
**behaviour a run would see**, not against internals, so a change in how the guard is wired cannot
silently weaken it.
"""

from __future__ import annotations

import itertools

import pytest
from pydantic import ValidationError

from recommendation.control.arguments import (
    CandidateSource,
    SearchCatalogArguments,
    SelectSourceArguments,
)
from recommendation.control.candidate_ledger import CandidateLedger
from recommendation.control.model_client import ModelRequest
from recommendation.control.schemas import ActionKind, ActionProposal

from experiments.m4_frozen_sources import (
    AccountedSources,
    DuplicateSourceQuery,
    FrozenHeadTool,
)
from experiments.m4_policies import (
    FIXED_SOURCE_ORDER,
    M4_ALLOWED_ACTIONS,
    FixedTraversalPolicy,
    GuardedAdaptivePolicy,
    ProtocolDeviation,
)
from experiments.m4_query_rules import (
    MAX_TERMS,
    deterministic_terms,
    newest_indexed,
)
from experiments.m4_schema_projection import (
    ADAPTIVE_PROVIDER_ACTIONS,
    SchemaProjectionClient,
    project_action_schema,
)

# --------------------------------------------------------------------------- #
# §7.3 provider-visible action schema
# --------------------------------------------------------------------------- #


def _full_schema() -> tuple[dict[str, object], ...]:
    """A schema shaped like the real one: every action the controller may offer."""
    return (
        {"action": "recommend_from_history", "arguments": [], "required_arguments": []},
        {"action": "search_catalog", "arguments": [{"name": "terms"}], "required_arguments": ["terms"]},
        {"action": "find_similar", "arguments": [{"name": "seed_parent_asin"}], "required_arguments": ["seed_parent_asin"]},
        {"action": "ask_clarification", "arguments": [{"name": "question"}], "required_arguments": ["question"]},
        {"action": "select_source", "arguments": [{"name": "source"}], "required_arguments": ["source"]},
        {"action": "finish", "arguments": [], "required_arguments": []},
    )


def test_the_projected_schema_is_exactly_select_source_and_finish() -> None:
    """PR §7.3: the provider must see only SELECT_SOURCE and FINISH."""
    projected = project_action_schema(_full_schema())
    assert tuple(e["action"] for e in projected) == ("select_source", "finish")
    assert set(ADAPTIVE_PROVIDER_ACTIONS) == {"select_source", "finish"}


@pytest.mark.parametrize(
    "forbidden",
    ["recommend_from_history", "search_catalog", "find_similar", "ask_clarification"],
)
def test_the_projected_schema_hides_every_other_action(forbidden: str) -> None:
    """PR §7.3: no other action may reach the provider."""
    projected = project_action_schema(_full_schema())
    assert forbidden not in {e["action"] for e in projected}


def test_the_projection_client_forwards_a_narrowed_request_to_the_provider() -> None:
    """PR §7.3: the *provider* receives the narrowed schema, not merely a local variable."""
    seen: list[tuple[str, ...]] = []

    class Recorder:
        def complete(self, request: ModelRequest) -> str:
            seen.append(tuple(str(e.get("action")) for e in request.action_schema))
            return "ok"

    client = SchemaProjectionClient(Recorder())
    request = ModelRequest(system_prompt="p", action_schema=_full_schema())
    client.complete(request)

    assert seen == [("select_source", "finish")], "provider saw the wrong action set"
    assert client.last_projected == ("select_source", "finish")
    assert "search_catalog" in client.last_original


def test_the_projection_client_preserves_every_other_request_field() -> None:
    """PR §7.3: projecting narrows disclosure; it must not rewrite the policy contract."""
    captured: list[ModelRequest] = []

    class Recorder:
        def complete(self, request: ModelRequest) -> str:
            captured.append(request)
            return "ok"

    client = SchemaProjectionClient(Recorder())
    original = ModelRequest(
        system_prompt="the contract",
        context_payload={"budget": {"remaining_tool_calls": 4}},
        action_schema=_full_schema(),
        correction="try again",
    )
    client.complete(original)
    forwarded = captured[0]
    assert forwarded.system_prompt == original.system_prompt
    assert forwarded.context_payload == original.context_payload
    assert forwarded.correction == original.correction


# --------------------------------------------------------------------------- #
# §8 C4-1 / C4-2 post-response guard
# --------------------------------------------------------------------------- #


class _ScriptedInner:
    """An inner policy that returns one fixed proposal then repeats it."""

    def __init__(self, proposal: ActionProposal) -> None:
        self._proposal = proposal
        self.calls = 0

    def choose(self, context: object) -> ActionProposal:
        self.calls += 1
        return self._proposal


class _Ctx:
    available_actions = (ActionKind.SELECT_SOURCE, ActionKind.FINISH)


def _guarded(proposal: ActionProposal, universe=FIXED_SOURCE_ORDER) -> GuardedAdaptivePolicy:
    return GuardedAdaptivePolicy(inner=_ScriptedInner(proposal), universe=universe)


def _select(source: CandidateSource, limit: int = 5) -> ActionProposal:
    return ActionProposal(
        action=ActionKind.SELECT_SOURCE,
        arguments=SelectSourceArguments(source=source, limit=limit),
        rationale="test",
    )


def test_c4_1_an_action_outside_the_allowed_pair_is_a_deviation() -> None:
    """PR §8 C4-1: RECOMMEND_FROM_HISTORY is parseable but must not execute."""
    policy = _guarded(
        ActionProposal(action=ActionKind.RECOMMEND_FROM_HISTORY, k=4, rationale="t")
    )
    with pytest.raises(ProtocolDeviation) as excinfo:
        policy.choose(_Ctx())
    assert excinfo.value.reason == "protocol_deviation:action_not_allowed"
    assert len(policy.deviations) == 1
    assert policy.terminated, "the run must terminate under the frozen deviation rule"


def test_c4_1_ask_clarification_is_refused_when_constructible() -> None:
    """PR §8 C4-1: ASK_CLARIFICATION is outside the frozen M4 action space."""
    from recommendation.control.arguments import AskClarificationArguments

    proposal = ActionProposal(
        action=ActionKind.ASK_CLARIFICATION,
        arguments=AskClarificationArguments(question="which brand?"),
        rationale="t",
    )
    policy = _guarded(proposal)
    with pytest.raises(ProtocolDeviation) as excinfo:
        policy.choose(_Ctx())
    assert excinfo.value.reason == "protocol_deviation:action_not_allowed"


def test_search_catalog_cannot_even_be_constructed_without_its_required_arguments() -> None:
    """Defence in depth (PR §7.3): the argument contract blocks it before the guard runs.

    The projected schema never shows ``terms``, so a provider cannot supply them; the repository's
    own proposal contract refuses the action outright. This test records that fact so the guard is
    never mistaken for the only protection.
    """
    with pytest.raises(ValidationError):
        ActionProposal(action=ActionKind.SEARCH_CATALOG, rationale="t")
    # And with arguments it *is* constructible -- which is why C4-1 must exist as well.
    constructible = ActionProposal(
        action=ActionKind.SEARCH_CATALOG,
        arguments=SearchCatalogArguments(terms=("camping",), limit=100),
        rationale="t",
    )
    policy = _guarded(constructible)
    with pytest.raises(ProtocolDeviation) as excinfo:
        policy.choose(_Ctx())
    assert excinfo.value.reason == "protocol_deviation:action_not_allowed"


def test_c4_2_a_source_outside_the_frozen_universe_is_a_deviation() -> None:
    """PR §8 C4-2(a): a source outside the four-source universe must be refused.

    ``TIGER`` is a real ``CandidateSource`` member but is not in M4's universe, so it is the
    natural out-of-universe probe.
    """
    policy = _guarded(_select(CandidateSource.TIGER))
    with pytest.raises(ProtocolDeviation) as excinfo:
        policy.choose(_Ctx())
    assert excinfo.value.reason == "protocol_deviation:source_outside_universe"
    assert policy.terminated


def test_c4_2_a_duplicate_source_is_a_deviation() -> None:
    """PR §8 C4-2(b) / §9: a second query of the same source must be refused."""
    policy = _guarded(_select(CandidateSource.HISTORY))
    first = policy.choose(_Ctx())
    assert first.action is ActionKind.SELECT_SOURCE
    with pytest.raises(ProtocolDeviation) as excinfo:
        policy.choose(_Ctx())
    assert excinfo.value.reason == "protocol_deviation:duplicate_source"
    assert policy.terminated


def test_a_guarded_select_is_pinned_to_the_frozen_depth() -> None:
    """PR §3: depth is an experimental constant, never a model-controlled quantity."""
    policy = _guarded(_select(CandidateSource.HISTORY, limit=1))
    proposal = policy.choose(_Ctx())
    assert proposal.arguments.limit == 100


def test_the_deviation_list_records_the_reason_for_the_sensitivity_analysis() -> None:
    """PR §10: deviations are listed per user, not silently dropped."""
    policy = _guarded(_select(CandidateSource.TIGER))
    with pytest.raises(ProtocolDeviation):
        policy.choose(_Ctx())
    assert [d.reason for d in policy.deviations] == [
        "protocol_deviation:source_outside_universe"
    ]
    assert policy.deviations[0].step_index == 1


# --------------------------------------------------------------------------- #
# §9 termination rules
# --------------------------------------------------------------------------- #


def test_zero_source_finish_is_refused() -> None:
    """PR §9: FINISH before any source was queried is prohibited."""
    policy = _guarded(ActionProposal(action=ActionKind.FINISH, rationale="t"))
    with pytest.raises(ProtocolDeviation) as excinfo:
        policy.choose(_Ctx())
    assert excinfo.value.reason == "zero_source_finish"
    assert policy.terminated


def test_finish_is_allowed_once_one_source_was_queried() -> None:
    """PR §9: single-source FINISH is legal and is the early-stopping outcome."""
    inner = _ScriptedInner(_select(CandidateSource.HISTORY))
    policy = GuardedAdaptivePolicy(inner=inner, universe=FIXED_SOURCE_ORDER)
    policy.choose(_Ctx())
    inner._proposal = ActionProposal(action=ActionKind.FINISH, rationale="done")
    proposal = policy.choose(_Ctx())
    assert proposal.action is ActionKind.FINISH


def test_the_allowed_action_set_is_the_frozen_pair() -> None:
    """PR §7.3 / §8 C4-1: guard and projection agree on the same pair."""
    assert M4_ALLOWED_ACTIONS == frozenset({ActionKind.SELECT_SOURCE, ActionKind.FINISH})
    assert {a.value for a in M4_ALLOWED_ACTIONS} == set(ADAPTIVE_PROVIDER_ACTIONS)


# --------------------------------------------------------------------------- #
# §9 one call per (user, source) -- the hard isolation guarantee
# --------------------------------------------------------------------------- #


def test_a_frozen_head_tool_refuses_a_second_call() -> None:
    """PR §9: the adapter refuses, so a duplicate can never reach the ledger."""
    tool = FrozenHeadTool(
        source=CandidateSource.HISTORY,
        score_kind="sasrec_logit",
        head=(("A", 1, 1.0),),
    )
    assert tool.propose(limit=100) == [("A", 1, 1.0)]
    with pytest.raises(DuplicateSourceQuery):
        tool.propose(limit=100)


def test_the_refusal_happens_before_anything_is_recorded() -> None:
    """PR §9: the exception is raised inside ``propose``, which runs before ledger recording.

    This is the property that makes "must not write CandidateLedger" true by construction rather
    than by promise.
    """
    tool = FrozenHeadTool(source=CandidateSource.HISTORY, score_kind="sasrec_logit", head=())
    tool.propose(limit=100)
    with pytest.raises(DuplicateSourceQuery):
        tool.propose(limit=100)


def test_each_user_gets_a_fresh_unconsumed_tool() -> None:
    """PR §9: the one-call rule is per user, enforced without cross-user state."""
    accounted = AccountedSources(
        heads={CandidateSource.HISTORY: {1: (("A", 1, 1.0),), 2: (("B", 1, 1.0),)}},
        score_kinds={CandidateSource.HISTORY: "sasrec_logit"},
    )
    first = accounted.for_user(1)[CandidateSource.HISTORY]
    second = accounted.for_user(2)[CandidateSource.HISTORY]
    assert first.propose(limit=100) == [("A", 1, 1.0)]
    assert second.propose(limit=100) == [("B", 1, 1.0)], "a fresh user must start unconsumed"


def test_a_user_absent_from_the_table_gets_an_empty_head() -> None:
    """A source with no materialised head for a user must not be satisfied from another user."""
    accounted = AccountedSources(
        heads={CandidateSource.HISTORY: {1: (("A", 1, 1.0),)}},
        score_kinds={CandidateSource.HISTORY: "sasrec_logit"},
    )
    tool = accounted.for_user(999)[CandidateSource.HISTORY]
    assert tool.propose(limit=100) == []
    assert tool.consumed, "the single permitted call was still consumed"


# --------------------------------------------------------------------------- #
# §11 source-order invariance
# --------------------------------------------------------------------------- #

_LISTS = {
    CandidateSource.HISTORY: ("A", "B", "C", "D", "X"),
    CandidateSource.CATALOG_SEARCH: ("C", "A", "E", "B"),
    CandidateSource.SIMILAR_ITEM: ("F", "A", "B"),
    CandidateSource.TWO_TOWER: ("B", "F", "G", "A"),
}


def _ledger_for(order: tuple[CandidateSource, ...]) -> CandidateLedger:
    ledger = CandidateLedger()
    known = frozenset({"A", "B", "C", "D", "E", "F", "G", "X"})
    for step, source in enumerate(order):
        rows = _LISTS[source]
        ledger.record_from_source(
            source=source,
            candidates=[(i, r + 1, 1.0 / (r + 1)) for r, i in enumerate(rows)],
            grounded=frozenset(set(rows) & known),
            step_index=step,
        )
    return ledger


@pytest.mark.parametrize(
    "subset",
    [
        FIXED_SOURCE_ORDER,
        (CandidateSource.HISTORY, CandidateSource.CATALOG_SEARCH),
        (CandidateSource.CATALOG_SEARCH, CandidateSource.TWO_TOWER),
        (CandidateSource.HISTORY, CandidateSource.SIMILAR_ITEM, CandidateSource.TWO_TOWER),
    ],
    ids=["all_four", "history+catalog", "catalog+two_tower", "history+similar+two_tower"],
)
def test_ranked_order_is_invariant_under_source_execution_order(subset) -> None:
    """PR §11: fusion depends on *which* sources were queried, not on *the order*.

    Without this, M4 would measure traversal order as well as subset selection.
    """
    outputs = {
        tuple(_ledger_for(perm).ranked_parent_asins())
        for perm in itertools.permutations(subset)
    }
    assert len(outputs) == 1, f"order changed the ranking: {outputs}"


def test_single_source_fusion_equals_that_source_order() -> None:
    """PR §4 property 1: with one source the fused order is exactly that source's order.

    This is what makes early stopping well defined: a one-source run returns that source's ranking.
    """
    for source in FIXED_SOURCE_ORDER:
        ledger = _ledger_for((source,))
        assert tuple(ledger.ranked_parent_asins()) == _LISTS[source]


# --------------------------------------------------------------------------- #
# §5 query-rule determinism
# --------------------------------------------------------------------------- #


def test_terms_are_deterministic_and_bounded() -> None:
    """PR §5: the rule is a pure function of h, with a hard term bound."""
    describe = {
        "A": "Camping Lantern Waterproof Bright",
        "B": "camping stove ultralight titanium",
    }
    first = deterministic_terms(("A", "B"), describe=describe.get)
    second = deterministic_terms(("A", "B"), describe=describe.get)
    assert first == second
    assert len(first) <= MAX_TERMS
    assert all(len(t) >= 3 for t in first)
    assert all(t == t.casefold() for t in first)


def test_the_newest_items_drive_the_query() -> None:
    """PR §5: the newest history is scanned first, so the query reflects current intent."""
    describe = {"old": "vintage compass", "new": "modern lantern"}
    terms = deterministic_terms(("old", "new"), describe=describe.get)
    assert terms[0] == "modern"


def test_an_unknown_identity_contributes_no_terms() -> None:
    """A description the catalogue does not have is skipped, never guessed at."""
    assert deterministic_terms(("missing",), describe=lambda _x: None) == ()


def test_no_description_provider_yields_no_terms() -> None:
    """Without a catalogue handle the rule declines rather than inventing a query."""
    assert deterministic_terms(("A",), describe=None) == ()


def test_newest_indexed_returns_the_most_recent_covered_item() -> None:
    """PR §5: the seed is the newest history item the index knows."""
    assert newest_indexed(("A", "B", "C"), is_indexed=lambda x: x in {"A", "C"}) == "C"


def test_newest_indexed_returns_none_when_nothing_is_covered() -> None:
    """``None`` means "skip this source", never "invent a seed"."""
    assert newest_indexed(("A", "B"), is_indexed=lambda _x: False) is None

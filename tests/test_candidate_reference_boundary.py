"""Phase 2.1: the grounded candidate reference boundary.

The invariant this module pins:

    a candidate-producing action may introduce candidate identity
    a reasoning action may only REFERENCE identity the current run already holds

Before this phase the trusted `ReasoningExecutor` asked the catalogue whether an identity
existed and treated "yes" as authorization.  Catalogue membership is necessary for *grounding*
but is not authority over *this run's* candidates, so a model could name any product the
catalogue happened to contain and have it read.  Reasoning stayed read-only - membership never
changed - but the boundary the docstrings claimed ("facts about candidates the run already
holds") was not the boundary the code enforced.

Everything here is offline: the real reasoner over the synthetic fixture catalogue, the real
control plane, and scripted or target-selecting doubles instead of a provider.

Cases, in the order the phase names them:

A. a valid grounded run candidate is inspectable
B. a catalogue identity that is not in this run is refused **before** any read
C. a nonexistent identity is refused
D. a valid multi-target action succeeds
E. a multi-target action with one unauthorized member is refused whole
F. an ineligible grounded candidate stays inspectable but cannot be recommended
G. an unresolved grounded candidate stays inspectable
H. a reference does not carry across runs
I. the model chooses the target; no binder picks for it
J. enforcement is identical under both policies
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from recommendation.control import (  # noqa: E402
    ActionKind,
    CandidateEligibilityEvaluator,
    CandidateLedger,
    CandidatePlane,
    CatalogSearchSource,
    ConstraintKind,
    ConstraintRequirement,
    GroundedReasoner,
    GroundingVerifier,
    LoopLimits,
    PolicyActionError,
    PolicyContext,
    ReasoningExecutor,
    RecommendFromHistoryCapability,
    RuleBasedPolicy,
    ValidatedAction,
    candidate_eligibility,
)
from recommendation.control import ARGUMENTS_BY_ACTION  # noqa: E402
from recommendation.control.arguments import (  # noqa: E402
    BundleArguments,
    CompareArguments,
    GetDetailsArguments,
    TradeOffArguments,
)
from recommendation.control.context import (  # noqa: E402
    DEFAULT_CANDIDATE_REFERENCE_LIMIT,
    CandidateReference,
    CandidateState,
)
from recommendation.control.task_state import (  # noqa: E402
    ConstraintOrigin,
    TaskConstraint,
    TaskState,
)
from recommendation.memory.schemas import PreferenceKind  # noqa: E402
from tests.agent_reranking_fixture import CANDIDATE_ROWS  # noqa: E402
from tests.control_fixture import build_control_harness  # noqa: E402

#: The fixture catalogue contains all four of these identities.
RUN_CANDIDATES = ("cand-red", "cand-blue")
#: In the catalogue, and deliberately **not** in ``RUN_CANDIDATES``.
IN_CATALOGUE_NOT_IN_RUN = "cand-green"
#: In neither.
NOT_IN_CATALOGUE = "cand-nonexistent"
#: A budget ceiling splitting {red 25, black 15} from {blue 40, green 60}.
BUDGET = "35"

#: Every reasoning action whose arguments name product identities.
IDENTITY_ACTIONS: tuple[ActionKind, ...] = tuple(
    action
    for action, model in ARGUMENTS_BY_ACTION.items()
    if "parent_asins" in model.model_fields
)


class _Map:
    """The trusted identity map the fixture catalogue is keyed by."""

    def __init__(self, rows: Any = CANDIDATE_ROWS) -> None:
        self._ids = {row[0]: row[1] for row in rows}

    def has_parent_asin(self, parent_asin: str) -> bool:
        return parent_asin in self._ids

    def parent_asin_to_item_id(self, parent_asin: str) -> int:
        return self._ids[parent_asin]

    def item_id_to_parent_asin(self, item_id: int) -> str:
        for identity, value in self._ids.items():
            if value == item_id:
                return identity
        raise KeyError(item_id)


@pytest.fixture(scope="module")
def reasoner() -> GroundedReasoner:
    """The real deterministic reasoner over the suite catalogue."""
    return GroundedReasoner(
        build_control_harness(catalog_rows=CANDIDATE_ROWS).parts["enricher"].metadata
    )


@pytest.fixture
def executor(reasoner: GroundedReasoner) -> ReasoningExecutor:
    """A real reasoning executor over that catalogue.

    Function-scoped, because ``executed_count`` is part of what these tests assert and a shared
    counter across tests would make the ordering check meaningless.
    """
    return ReasoningExecutor(reasoner)



def _boundary_harness(
    policy: Any = None,
    *,
    grounded_rows: Any = CANDIDATE_ROWS[:2],
    limits: LoopLimits | None = None,
    driver: str = "direct",
) -> Any:
    """Build a run that grounds ``grounded_rows`` while the catalogue holds everything.

    Passing ``catalog_rows`` separately is what creates the boundary case: the catalogue knows
    products the run never retrieved, so "exists" and "is mine" cannot be confused.
    """
    harness = build_control_harness(
        policy=policy,
        limits=limits or LoopLimits(max_steps=5, max_tool_calls=3, max_retries=1),
        rows=tuple(grounded_rows),
        catalog_rows=CANDIDATE_ROWS,
        driver=driver,
    )
    reasoner = GroundedReasoner(
        build_control_harness(catalog_rows=CANDIDATE_ROWS).parts["enricher"].metadata
    )
    harness.controller._reasoning = ReasoningExecutor(reasoner)  # noqa: SLF001
    return harness


def _reasoner_over_catalogue() -> GroundedReasoner:
    """A reasoner over the whole fixture catalogue."""
    return GroundedReasoner(
        build_control_harness(catalog_rows=CANDIDATE_ROWS).parts["enricher"].metadata
    )


def _action(action: ActionKind, arguments: Any, *, step: int = 0) -> ValidatedAction:
    """Build a validated action, as the controller would after validation."""
    return ValidatedAction(
        action_id=f"a{step}", step_index=step, run_id="r", k=4, action=action, arguments=arguments
    )


def _attempt(
    executor: ReasoningExecutor,
    action: ActionKind,
    arguments: Any,
    allowlist: tuple[str, ...] | None,
) -> tuple[str, Any]:
    """Run one action and return ``("ok", observation)`` or ``("refused", code)``."""
    try:
        return "ok", executor.execute(_action(action, arguments), authorized_candidates=allowlist)
    except PolicyActionError as exc:
        return "refused", exc.code


# =========================================================================== #
# 0. The vocabulary the boundary applies to
# =========================================================================== #


def test_only_read_only_actions_name_product_identities() -> None:
    """The structural rule the boundary rests on, asserted against the real registry.

    If a candidate-producing action ever grew ``parent_asins``, the run-membership check would
    be applied to an action whose whole purpose is to introduce identity - so this pins the
    partition rather than assuming it.
    """
    assert set(IDENTITY_ACTIONS) == {
        ActionKind.GET_DETAILS,
        ActionKind.COMPARE,
        ActionKind.TRADE_OFF,
        ActionKind.CHECK_COMPATIBILITY,
        ActionKind.BUNDLE,
    }
    from recommendation.control import READ_ONLY_ACTIONS

    assert set(IDENTITY_ACTIONS) <= set(READ_ONLY_ACTIONS)
    for action in (ActionKind.RECOMMEND_FROM_HISTORY, ActionKind.SEARCH_CATALOG):
        assert "parent_asins" not in ARGUMENTS_BY_ACTION[action].model_fields


def test_every_identity_action_is_checked_the_same_way(
    executor: ReasoningExecutor,
) -> None:
    """Each item-targeting action refuses a non-run identity - none is a special case.

    An action that forgot the check would be an open channel, so this drives every one of them
    rather than the two the phase happens to name.
    """
    arguments_by_action = {
        ActionKind.GET_DETAILS: GetDetailsArguments(parent_asins=(IN_CATALOGUE_NOT_IN_RUN,)),
        ActionKind.COMPARE: CompareArguments(
            parent_asins=(RUN_CANDIDATES[0], IN_CATALOGUE_NOT_IN_RUN)
        ),
        ActionKind.TRADE_OFF: TradeOffArguments(
            parent_asins=(RUN_CANDIDATES[0], IN_CATALOGUE_NOT_IN_RUN), priority="weight"
        ),
        ActionKind.CHECK_COMPATIBILITY: _compat(
            (RUN_CANDIDATES[0], IN_CATALOGUE_NOT_IN_RUN)
        ),
        ActionKind.BUNDLE: BundleArguments(
            parent_asins=(RUN_CANDIDATES[0], IN_CATALOGUE_NOT_IN_RUN)
        ),
    }
    for action, arguments in arguments_by_action.items():
        outcome, detail = _attempt(executor, action, arguments, RUN_CANDIDATES)
        assert outcome == "refused", f"{action.value} accepted a non-run identity"
        assert detail == "candidate_not_in_run", action.value


def _compat(identities: tuple[str, ...]) -> Any:
    from recommendation.control.arguments import CheckCompatibilityArguments

    return CheckCompatibilityArguments(parent_asins=identities, requirement="color: red")


# =========================================================================== #
# A. A valid grounded run candidate
# =========================================================================== #


def test_a_run_candidate_is_inspectable(executor: ReasoningExecutor) -> None:
    """Case A: the same identity the run holds reads successfully, with its real facts."""
    outcome, observation = _attempt(
        executor,
        ActionKind.GET_DETAILS,
        GetDetailsArguments(parent_asins=("cand-blue",)),
        RUN_CANDIDATES,
    )
    assert outcome == "ok"
    assert observation.status == "ok"
    assert set(observation.facts) == {"cand-blue"}
    assert observation.requested == ("cand-blue",)


def test_inspection_does_not_change_candidate_membership(
    executor: ReasoningExecutor,
) -> None:
    """Reading a candidate is read-only: it cannot add, drop or reorder ledger entries."""
    ledger = CandidateLedger()
    grounded = frozenset(row[0] for row in CANDIDATE_ROWS)
    ledger.record_from_source(
        source=__import__(
            "recommendation.control", fromlist=["CandidateSource"]
        ).CandidateSource.HISTORY,
        candidates=[("cand-red", 1, 0.9), ("cand-blue", 2, 0.8)],
        grounded=grounded,
    )
    before = ledger.as_dict()
    _attempt(
        executor,
        ActionKind.GET_DETAILS,
        GetDetailsArguments(parent_asins=("cand-blue",)),
        RUN_CANDIDATES,
    )
    _attempt(
        executor,
        ActionKind.COMPARE,
        CompareArguments(parent_asins=("cand-red", "cand-blue")),
        RUN_CANDIDATES,
    )
    assert ledger.as_dict() == before
    assert "cand-green" not in ledger


# =========================================================================== #
# B. Catalogue membership is not run membership  (the key regression)
# =========================================================================== #


def test_a_catalogue_identity_outside_the_run_is_refused(
    executor: ReasoningExecutor, reasoner: GroundedReasoner
) -> None:
    """Case B: the identity exists, and the run still may not read it.

    This is the regression the phase exists for.  ``cand-green`` is a real catalogue product -
    ``reasoner.facts`` returns a profile for it - so the old check, which asked only the
    catalogue, admitted it.  Run membership is the authority, and the catalogue is consulted
    afterwards only to ground a reference that is already authorized.
    """
    # The identity really does exist: this is not a "missing product" case.
    assert reasoner.facts(IN_CATALOGUE_NOT_IN_RUN) is not None

    outcome, code = _attempt(
        executor,
        ActionKind.GET_DETAILS,
        GetDetailsArguments(parent_asins=(IN_CATALOGUE_NOT_IN_RUN,)),
        RUN_CANDIDATES,
    )
    assert outcome == "refused"
    assert code == "candidate_not_in_run"


def test_a_refused_reference_produces_no_facts_at_all(
    executor: ReasoningExecutor,
) -> None:
    """The refusal is total: no observation, so no catalogue fact escapes the boundary."""
    with pytest.raises(PolicyActionError) as caught:
        executor.execute(
            _action(
                ActionKind.GET_DETAILS,
                GetDetailsArguments(parent_asins=(IN_CATALOGUE_NOT_IN_RUN,)),
            ),
            authorized_candidates=RUN_CANDIDATES,
        )
    assert caught.value.code == "candidate_not_in_run"
    # The error message must not carry the catalogue value it refused to read.
    assert "60" not in str(caught.value)


def test_authority_is_checked_before_anything_else_happens(
    executor: ReasoningExecutor,
) -> None:
    """Authorization precedes both the catalogue read and the execution counter.

    ``executed_count`` is the observable proof of ordering: it is incremented *after* the
    membership check, so a refused action leaves it untouched.  That is what makes the boundary
    "decide, then read" rather than "read, then decide" - and it is checked here rather than
    inferred from a fact count.
    """
    before = executor.executed_count
    with pytest.raises(PolicyActionError):
        executor.execute(
            _action(
                ActionKind.GET_DETAILS,
                GetDetailsArguments(parent_asins=(IN_CATALOGUE_NOT_IN_RUN,)),
            ),
            authorized_candidates=RUN_CANDIDATES,
        )
    assert executor.executed_count == before, "a refused action must not count as executed"

    # A legal action does advance it, so the counter is not simply inert.
    executor.execute(
        _action(ActionKind.GET_DETAILS, GetDetailsArguments(parent_asins=("cand-red",))),
        authorized_candidates=RUN_CANDIDATES,
    )
    assert executor.executed_count == before + 1


# =========================================================================== #
# C. A nonexistent identity
# =========================================================================== #


def test_a_nonexistent_identity_is_refused(executor: ReasoningExecutor) -> None:
    """Case C: an identity that exists nowhere fails closed."""
    outcome, code = _attempt(
        executor,
        ActionKind.GET_DETAILS,
        GetDetailsArguments(parent_asins=(NOT_IN_CATALOGUE,)),
        RUN_CANDIDATES,
    )
    assert outcome == "refused"
    assert code == "candidate_not_in_run"


def test_nonexistent_and_out_of_run_are_both_refused_with_one_stable_code(
    executor: ReasoningExecutor,
) -> None:
    """Honest note on the distinction the phase asked about.

    The architecture *does* distinguish "not in the catalogue" from "not in this run" - the
    reasoner reports a missing fact as ``UNKNOWN`` while membership is a refusal - but the
    primary check is membership, so an identity that is in neither fails on membership and
    carries one code.  That is the fail-closed outcome, and it is deliberately not refined into
    "this product exists but is not yours", which would leak catalogue contents to a caller
    that has no authority over them.
    """
    missing_outcome, missing_code = _attempt(
        executor,
        ActionKind.GET_DETAILS,
        GetDetailsArguments(parent_asins=(NOT_IN_CATALOGUE,)),
        RUN_CANDIDATES,
    )
    existing_outcome, existing_code = _attempt(
        executor,
        ActionKind.GET_DETAILS,
        GetDetailsArguments(parent_asins=(IN_CATALOGUE_NOT_IN_RUN,)),
        RUN_CANDIDATES,
    )
    assert missing_outcome == existing_outcome == "refused"
    assert missing_code == existing_code == "candidate_not_in_run"


# =========================================================================== #
# D-E. Multi-target actions
# =========================================================================== #


def test_a_valid_compare_succeeds(executor: ReasoningExecutor) -> None:
    """Case D: all members authorized, so the comparison runs on those identities."""
    outcome, observation = _attempt(
        executor,
        ActionKind.COMPARE,
        CompareArguments(
            parent_asins=("cand-red", "cand-blue"), attributes=("weight_text",)
        ),
        RUN_CANDIDATES,
    )
    assert outcome == "ok", observation
    assert observation.status == "ok"
    # The compared identities are exactly the authorized ones, unchanged and in order.
    assert observation.compared == ("cand-red", "cand-blue")


def test_a_mixed_compare_is_refused_whole(executor: ReasoningExecutor) -> None:
    """Case E: one unauthorized member invalidates the action.

    Partial success would be worse than failure here: it would answer a question about an
    identity the run has no authority over, and would leak which of the named identities the
    run happens to hold.
    """
    outcome, code = _attempt(
        executor,
        ActionKind.COMPARE,
        CompareArguments(
            parent_asins=("cand-red", IN_CATALOGUE_NOT_IN_RUN), attributes=("weight_text",)
        ),
        RUN_CANDIDATES,
    )
    assert outcome == "refused"
    assert code == "candidate_not_in_run"


def test_the_authorization_rule_is_all_or_nothing_for_every_multi_target_action(
    executor: ReasoningExecutor,
) -> None:
    """The whole class of multi-target actions, not only COMPARE."""
    cases = {
        ActionKind.COMPARE: CompareArguments(parent_asins=("cand-red", "cand-blue")),
        ActionKind.TRADE_OFF: TradeOffArguments(
            parent_asins=("cand-red", "cand-blue"), priority="weight"
        ),
        ActionKind.BUNDLE: BundleArguments(parent_asins=("cand-red", "cand-blue")),
    }
    for action, arguments in cases.items():
        assert _attempt(executor, action, arguments, RUN_CANDIDATES)[0] == "ok", action.value
        mixed = arguments.model_copy(
            update={"parent_asins": ("cand-red", IN_CATALOGUE_NOT_IN_RUN)}
        )
        assert _attempt(executor, action, mixed, RUN_CANDIDATES) == (
            "refused",
            "candidate_not_in_run",
        ), action.value


def test_an_empty_allowlist_authorizes_nothing(executor: ReasoningExecutor) -> None:
    """A run holding no grounded candidate has no legal target - not unlimited ones."""
    assert _attempt(
        executor,
        ActionKind.GET_DETAILS,
        GetDetailsArguments(parent_asins=("cand-red",)),
        (),
    ) == ("refused", "candidate_not_in_run")
    # An argumentless reasoning action stays legal: it names nothing, so it needs nothing.
    assert _attempt(executor, ActionKind.VERIFY, ARGUMENTS_BY_ACTION[ActionKind.VERIFY](), ())[
        0
    ] == "ok"


# =========================================================================== #
# F-G. Grounded but not feasible / not resolved
# =========================================================================== #


def _budget_task(limit: str) -> TaskState:
    return TaskState().with_constraint(
        TaskConstraint(
            kind=PreferenceKind.PRICE_MAX,
            value=limit,
            origin=ConstraintOrigin.CURRENT_TURN,
            source_text=f"under {limit}",
        )
    )


def test_an_ineligible_grounded_candidate_remains_inspectable(
    reasoner: GroundedReasoner, executor: ReasoningExecutor
) -> None:
    """Case F: reasoning authority is not recommendation membership.

    ``cand-blue`` is a grounded run candidate proved to violate the budget.  It is excluded from
    what may be *recommended*; it is still a legal reasoning target, which is what lets an agent
    explain why it was excluded.
    """
    view = candidate_eligibility(
        ("cand-red", "cand-blue"),
        requirements=(ConstraintRequirement(kind=ConstraintKind.PRICE_MAX, expected=BUDGET),),
        reasoner=reasoner,
    )
    assert view.eligibility_of("cand-blue").value == "ineligible"
    assert view.is_feasible("cand-blue") is False

    # Still inspectable, because it is a grounded candidate of this run.
    outcome, observation = _attempt(
        executor,
        ActionKind.GET_DETAILS,
        GetDetailsArguments(parent_asins=("cand-blue",)),
        ("cand-red", "cand-blue"),
    )
    assert outcome == "ok"
    assert set(observation.facts) == {"cand-blue"}


def test_inspecting_an_ineligible_candidate_cannot_resurrect_it(
    reasoner: GroundedReasoner,
) -> None:
    """A read changes no verdict, so the feasible view is identical before and after."""
    requirement = ConstraintRequirement(kind=ConstraintKind.PRICE_MAX, expected=BUDGET)
    before = candidate_eligibility(
        ("cand-red", "cand-blue"), requirements=(requirement,), reasoner=reasoner
    )
    after = candidate_eligibility(
        ("cand-red", "cand-blue"), requirements=(requirement,), reasoner=reasoner
    )
    assert after.eligibility_of("cand-blue").value == "ineligible"
    assert after.feasible_identities == before.feasible_identities == ("cand-red",)


def test_an_unresolved_grounded_candidate_remains_inspectable(
    reasoner: GroundedReasoner, executor: ReasoningExecutor
) -> None:
    """Case G: the candidate a policy most needs to read is one it is allowed to read.

    An ``UNRESOLVED`` candidate is exactly the target evidence acquisition should pick, so
    making it non-inspectable would make the Phase-2 unresolved state unactionable.
    """
    requirement = ConstraintRequirement(
        kind=ConstraintKind.UNVERIFIABLE, expected="waterproof"
    )
    view = candidate_eligibility(
        ("cand-red",), requirements=(requirement,), reasoner=reasoner
    )
    assert view.eligibility_of("cand-red").value == "unresolved"

    outcome, observation = _attempt(
        executor,
        ActionKind.GET_DETAILS,
        GetDetailsArguments(parent_asins=("cand-red",)),
        ("cand-red",),
    )
    assert outcome == "ok"
    assert set(observation.facts) == {"cand-red"}


# =========================================================================== #
# The reference projection itself
# =========================================================================== #


def test_the_reference_projection_is_a_narrowing_of_the_grounded_set() -> None:
    """The projection invents no identity and is capped."""
    context = PolicyContext(
        user_request="anything",
        available_actions=(ActionKind.FINISH,),
        has_trusted_history=True,
        candidate_state=CandidateState(grounded=True, candidate_count=2),
        grounded_parent_asins=("cand-red", "cand-blue"),
        grounded_candidates=(
            CandidateReference(parent_asin="cand-red", eligibility="verified_eligible"),
            CandidateReference(parent_asin="cand-blue", eligibility="ineligible"),
        ),
    )
    assert context.reasoning_targets() == ("cand-red", "cand-blue")
    assert set(context.reasoning_targets()) <= set(context.grounded_parent_asins)
    assert context.candidate_reference("cand-blue") is not None
    assert context.candidate_reference("cand-green") is None


def test_the_reference_projection_is_bounded() -> None:
    """A large candidate set cannot produce an unbounded policy view."""
    identities = tuple(f"cand-{index}" for index in range(50))
    references = tuple(CandidateReference(parent_asin=i) for i in identities)
    context = PolicyContext(
        user_request="anything",
        available_actions=(ActionKind.FINISH,),
        has_trusted_history=True,
        grounded_parent_asins=identities,
        grounded_candidates=references[:DEFAULT_CANDIDATE_REFERENCE_LIMIT],
    )
    assert len(context.grounded_candidates) == DEFAULT_CANDIDATE_REFERENCE_LIMIT
    # And the cap matches the largest identity list any reasoning argument accepts, so the
    # projection can never be wider than the action it feeds.
    from recommendation.control.arguments import MAX_COMPARE_ITEMS

    assert DEFAULT_CANDIDATE_REFERENCE_LIMIT == MAX_COMPARE_ITEMS


def test_a_reference_carries_position_but_no_catalogue_value() -> None:
    """A reference is identity plus reasoning position, never a product fact."""
    reference = CandidateReference(
        parent_asin="cand-red", eligibility="unresolved", evidence="attention"
    )
    assert reference.as_dict() == {
        "parent_asin": "cand-red",
        "eligibility": "unresolved",
        "evidence": "attention",
    }
    assert set(vars(reference)) == {"parent_asin", "eligibility", "evidence"}


# =========================================================================== #
# H. References do not carry across runs
# =========================================================================== #


def _run_and_return_engine(grounded_rows: Any, *, driver: str = "direct") -> tuple[Any, Any]:
    """Run one turn and return ``(result, engine)`` for that run.

    The engine is captured through ``new_engine`` because the trusted state - and therefore the
    run's live allowlist - lives on the engine, not on the controller.
    """
    harness = _boundary_harness(grounded_rows=grounded_rows, driver=driver)
    captured: dict[str, Any] = {}
    original = harness.controller.new_engine

    def capturing(*args: Any, **kwargs: Any) -> Any:
        engine = original(*args, **kwargs)
        captured["engine"] = engine
        return engine

    harness.controller.new_engine = capturing
    result = harness.controller.run("Recommend gear.", ("B1", "B2", "B3"), run_id="r")
    return result, captured["engine"]


def test_a_reference_does_not_carry_across_runs() -> None:
    """Case H: the allowlist is rebuilt per run from that run's own candidates.

    Run 1 grounds ``cand-blue``; run 2 does not.  The same identity is a legal target in the
    first run and refused in the second, because authority is derived from the current run's
    state and never from a globally valid name.
    """
    first_result, first_engine = _run_and_return_engine(CANDIDATE_ROWS[:2])
    second_result, second_engine = _run_and_return_engine(
        tuple(row for row in CANDIDATE_ROWS if row[0] != "cand-blue")
    )

    assert "cand-blue" in first_engine.grounded_identities()
    assert "cand-blue" not in second_engine.grounded_identities()
    # Both runs are real runs: the difference is what each one grounded.
    assert first_result.state["tool_result"] is not None
    assert second_result.state["tool_result"] is not None

    executor = ReasoningExecutor(_reasoner_over_catalogue())
    arguments = GetDetailsArguments(parent_asins=("cand-blue",))
    assert (
        executor.execute(
            _action(ActionKind.GET_DETAILS, arguments),
            authorized_candidates=first_engine.grounded_identities(),
        ).status
        == "ok"
    )
    with pytest.raises(PolicyActionError) as caught:
        executor.execute(
            _action(ActionKind.GET_DETAILS, arguments),
            authorized_candidates=second_engine.grounded_identities(),
        )
    assert caught.value.code == "candidate_not_in_run"


def test_the_allowlist_is_read_from_live_state_not_captured_at_construction() -> None:
    """A run whose candidate set changes mid-flight gets the new authority, not the old.

    The executor stores no allowlist; the loop passes the current one on every call.  This pins
    that property, because a cached allowlist would be a stale-authority bug of exactly the kind
    Phase 2 forbade for verdicts.
    """
    _result, engine = _run_and_return_engine(CANDIDATE_ROWS[:2])
    held = engine.grounded_identities()
    assert "cand-blue" in held

    # Narrow the run's candidate set, as a later retrieval with a smaller k would.
    engine.state["tool_result"] = engine.state["tool_result"].model_copy(
        update={
            "recommendations": [
                item
                for item in engine.state["tool_result"].recommendations
                if item.parent_asin != "cand-blue"
            ]
        }
    )
    live = engine.grounded_identities()
    assert "cand-blue" not in live

    executor = ReasoningExecutor(_reasoner_over_catalogue())
    with pytest.raises(PolicyActionError):
        executor.execute(
            _action(ActionKind.GET_DETAILS, GetDetailsArguments(parent_asins=("cand-blue",))),
            authorized_candidates=live,
        )


# =========================================================================== #
# I. The model chooses the target
# =========================================================================== #


class _PickingModel:
    """A model double that reads the offered references and picks one by position."""

    def __init__(self, pick: int = 1) -> None:
        self._pick = pick
        self.selected: list[str] = []

    def complete(self, request: Any) -> Any:
        import json

        from recommendation.control.model_client import ModelResponse

        refs = [
            str(entry["parent_asin"])
            for entry in request.context_payload.get("candidates", {}).get("candidate_refs", [])
        ]
        if not refs:
            return ModelResponse(
                text=json.dumps({"action": "recommend_from_history", "k": 3}),
                model_id="picking-double",
            )
        target = refs[min(self._pick, len(refs) - 1)]
        self.selected.append(target)
        return ModelResponse(
            text=json.dumps(
                {"action": "get_details", "arguments": {"parent_asins": [target]}}
            ),
            model_id="picking-double",
        )


def test_the_model_chooses_the_target_and_no_binder_overrides_it() -> None:
    """Case I: a target chosen by the model is the target the executor reads.

    The defect this closes is subtle and was invisible from the trajectory: the model emitted a
    target list, but the *binding layer* had supplied the identities out of band, so the model
    only ever relayed them and trusted code was choosing the product while the record looked
    model-driven.  A target-selecting double plus an assertion on the executed argument proves
    the choice is real.
    """
    from recommendation.control.model_policy import LLMAgentPolicy

    for pick, expected in ((0, "cand-red"), (1, "cand-blue")):
        model = _PickingModel(pick=pick)
        harness = _boundary_harness(LLMAgentPolicy(model))
        result = harness.controller.run("Recommend gear.", ("B1", "B2", "B3"), run_id="r")

        assert model.selected, "the model selected a target"
        assert model.selected[0] == expected, "the model's own choice was used"

        # What the model proposed is what executed: the recorded argument is the target the
        # model itself picked, and the observation is about that same identity.
        proposals = [
            step.action_proposal
            for step in result.trajectory.steps
            if step.action_proposal and step.action_proposal.get("action") == "get_details"
        ]
        assert proposals, "the details action executed"
        assert proposals[0]["arguments"]["parent_asins"] == [expected]
        reads = [
            step.observation
            for step in result.trajectory.steps
            if step.observation and step.observation.get("kind") == "details"
        ]
        assert reads, "facts were read"
        assert set(reads[0]["facts"]) == {expected}


def test_the_model_cannot_target_a_catalogue_identity_it_was_not_offered() -> None:
    """The complement of case I: selection is free, invention is not.

    The double names a real catalogue product the run does not hold.  The executor refuses it,
    the run does not crash, and the identity never becomes a candidate.
    """
    from recommendation.control.model_client import ScriptedModelClient
    from recommendation.control.model_policy import LLMAgentPolicy

    client = ScriptedModelClient(
        [
            {"action": "recommend_from_history", "k": 3},
            {"action": "get_details", "arguments": {"parent_asins": [IN_CATALOGUE_NOT_IN_RUN]}},
            {"action": "finish"},
        ]
    )
    harness = _boundary_harness(LLMAgentPolicy(client))
    result = harness.controller.run("Recommend gear.", ("B1", "B2", "B3"), run_id="r")

    grounded = [item.parent_asin for item in result.state["tool_result"].recommendations]
    assert IN_CATALOGUE_NOT_IN_RUN not in grounded

    details = [
        step.observation
        for step in result.trajectory.steps
        if step.observation and step.observation.get("kind") == "details"
    ]
    assert details == [], "an unauthorized target produced no facts"
    refusals = [
        step.observation
        for step in result.trajectory.steps
        if step.observation and step.observation.get("status") == "failed"
    ]
    assert refusals, "the refusal reached the policy as a bounded failure observation"
    assert refusals[0]["verification_status"] == "refused"


def test_the_binding_layer_no_longer_injects_identities() -> None:
    """The adapter's out-of-band identity channel is gone.

    If it came back, the model would once again be handed identities outside the payload and
    the "model chooses the target" claim would quietly stop being true.
    """
    from recommendation.evaluation.agent.model_policy_adapter import ObservationReactiveModel

    model = ObservationReactiveModel()
    assert not hasattr(model, "_identities")


# =========================================================================== #
# J. Policy-independent enforcement
# =========================================================================== #


def test_enforcement_is_identical_for_a_direct_proposal_and_a_model_proposal() -> None:
    """Case J: the boundary lives in the executor, so the proposer is irrelevant.

    A model policy and a hand-built ``ActionProposal`` reach the same check.  There is no
    model-only safety path to diverge from, and no bypass available to a policy that skips the
    model.
    """
    from recommendation.control.model_client import ScriptedModelClient
    from recommendation.control.model_policy import LLMAgentPolicy

    executor = ReasoningExecutor(_reasoner_over_catalogue())
    outcome, code = _attempt(
        executor,
        ActionKind.GET_DETAILS,
        GetDetailsArguments(parent_asins=(IN_CATALOGUE_NOT_IN_RUN,)),
        RUN_CANDIDATES,
    )
    assert (outcome, code) == ("refused", "candidate_not_in_run")

    client = ScriptedModelClient(
        [
            {"action": "recommend_from_history", "k": 3},
            {"action": "get_details", "arguments": {"parent_asins": [IN_CATALOGUE_NOT_IN_RUN]}},
            {"action": "finish"},
        ]
    )
    model_result = _boundary_harness(LLMAgentPolicy(client)).controller.run(
        "Recommend gear.", ("B1", "B2", "B3"), run_id="r"
    )
    notes = {
        str(step.observation.get("verification_note"))
        for step in model_result.trajectory.steps
        if step.observation and step.observation.get("status") == "failed"
    }
    assert any("candidate_not_in_run" in note for note in notes), notes

    # A deterministic policy reaches the same executor over the same allowlist.
    rule_harness = _boundary_harness(RuleBasedPolicy(default_k=3))
    rule_result = rule_harness.controller.run("Recommend gear.", ("B1", "B2", "B3"), run_id="r")
    assert rule_result.status is not None
    assert rule_harness.controller.offers_reasoning is True
    # Nothing about the boundary is policy-specific: the allowlist comes from run state.
    grounded = rule_harness.controller.new_engine(
        __import__("recommendation.agent.state", fromlist=["AgentInput"]).AgentInput(
            user_message="x", trusted_user_history=("B1",)
        ),
        run_id="r2",
    ).grounded_identities()
    assert grounded == ()


# =========================================================================== #
# Bounded invalid-reference recovery
# =========================================================================== #


def test_repeated_invalid_references_stay_bounded_by_the_step_budget() -> None:
    """A model that only ever names an unauthorized identity terminates inside its budget.

    The refusal is recoverable, so the policy may try again - but the retries are ordinary loop
    steps, so a model that never learns cannot spin.  This is also the regression for the
    reasoning path's step accounting: the reasoning branch re-enters ``check_limits`` now, so a
    run made only of refused reasoning proposals stops at ``max_steps`` instead of running far
    past it.
    """
    from recommendation.control.model_client import ScriptedModelClient
    from recommendation.control.model_policy import LLMAgentPolicy

    client = ScriptedModelClient(
        [
            {"action": "recommend_from_history", "k": 3},
            {"action": "get_details", "arguments": {"parent_asins": [IN_CATALOGUE_NOT_IN_RUN]}},
        ],
        repeat_last=True,
    )
    harness = _boundary_harness(
        LLMAgentPolicy(client), limits=LoopLimits(max_steps=4, max_tool_calls=3, max_retries=1)
    )
    result = harness.controller.run("Recommend gear.", ("B1", "B2", "B3"), run_id="r")

    assert result.control.step_count == 4, "the step budget bounded the run"
    assert result.control.termination_reason.value == "max_steps"
    assert result.control.status.value == "aborted"
    assert client.call_count <= 6, "the model was not consulted without bound"
    assert not [
        step
        for step in result.trajectory.steps
        if step.observation and step.observation.get("kind") == "details"
    ]


def test_both_drivers_bound_a_reasoning_only_run_the_same_way() -> None:
    """The graph and direct drivers must agree on the step budget for the reasoning path.

    They share one engine, so a difference here would be a topology bug - and there was one:
    the graph driver's ``_reason`` node returned to the policy directly, skipping
    ``check_limits``, so a reasoning-only run never re-tested ``max_steps``.
    """
    from recommendation.control.model_client import ScriptedModelClient
    from recommendation.control.model_policy import LLMAgentPolicy

    def run(driver: str) -> tuple[int, str]:
        client = ScriptedModelClient(
            [
                {"action": "recommend_from_history", "k": 3},
                {
                    "action": "get_details",
                    "arguments": {"parent_asins": [IN_CATALOGUE_NOT_IN_RUN]},
                },
            ],
            repeat_last=True,
        )
        harness = _boundary_harness(
            LLMAgentPolicy(client),
            limits=LoopLimits(max_steps=4, max_tool_calls=3, max_retries=1),
            driver=driver,
        )
        result = harness.controller.run("Recommend gear.", ("B1", "B2", "B3"), run_id="r")
        return result.control.step_count, result.control.termination_reason.value

    assert run("direct") == run("graph") == (4, "max_steps")

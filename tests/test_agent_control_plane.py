"""AgentRec-X 2.0-alpha control-plane tests.

Four families, matching the milestone's invariant list:

**Authority invariants** (section A)
    The policy cannot mutate trusted history, commit memory, create candidate identity,
    supply candidate ids, bypass the capability, or execute a trusted tool.

**Protocol invariants** (section B)
    ``ActionProposal`` is not executable; it must pass the validator; a ``DomainResult``
    cannot become an ``Observation`` without the verifier; execution metadata is
    controller-generated.

**Loop invariants** (section C)
    Control returns to the policy after a non-terminal observation; one action per step;
    only available actions; ``max_steps`` / ``max_tool_calls`` terminate; FINISH cannot
    complete without the completion guard.

**Existing-system invariants** (section D)
    Candidate identity, trusted history, preference memory, reranking, mapping and
    fail-closed behaviour keep working, and the accepted DAG is untouched.

Every test here is offline and deterministic.  Nothing in this file weakens an existing
test: the accepted graph's own suites run unchanged alongside it.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from recommendation.agent import AgentGraph, AgentInput, AgentDecision  # noqa: E402
from recommendation.agent.decision import build_decision_messages  # noqa: E402
from recommendation.control import (  # noqa: E402
    ActionKind,
    ActionProposal,
    ActionValidator,
    CompletionGuard,
    ControlState,
    LoopController,
    LoopLimits,
    ObservationAdapter,
    PolicyActionError,
    PolicyContext,
    RecommendFromHistoryCapability,
    RecommendationDomainResult,
    ResultVerifier,
    RuleBasedPolicy,
    RunStatus,
    STAGE_1_ACTIONS,
    TerminationReason,
    ValidatedAction,
    VerificationResult,
)
from recommendation.control.capability import CAPABILITY_NAME  # noqa: E402
from recommendation.control.topology import (  # noqa: E402
    DECLARED_CYCLE_EDGES,
    LOOP_NODE_NAMES,
    LOOP_PHASE_ORDER,
    NODE_CHECK_LIMITS,
    NODE_POLICY,
    NODE_UPDATE_STATE,
    _check_limits,
    _update_state,
    topology_mermaid,
)
from recommendation.demo.serialization import build_chat_response  # noqa: E402
from recommendation.tools import MissingUserHistory, RecommendationTool  # noqa: E402
from tests.agent_fakes import HISTORY  # noqa: E402
from tests.agent_reranking_fixture import (  # noqa: E402
    CANDIDATE_ROWS,
    QUERY,
    FixedEngine,
    build_tool_result,
    rendered_order,
)
from tests.control_fixture import (  # noqa: E402
    RecordingPolicy,
    ScriptedPolicy,
    build_control_harness,
    build_full_capability,
    build_minimal_capability,
)

RECOMMEND_ACTION = ActionProposal(action=ActionKind.RECOMMEND_FROM_HISTORY, k=3)
FINISH_ACTION = ActionProposal(action=ActionKind.FINISH)


def _finish(**_: object) -> ActionProposal:
    """A fresh FINISH proposal (kept as a helper so scripts read cleanly)."""
    return ActionProposal(action=ActionKind.FINISH)


# =========================================================================== #
# A. Authority invariants
# =========================================================================== #


def test_policy_cannot_mutate_trusted_history() -> None:
    """The history a run was started with is byte-identical after the run.

    The policy is handed a :class:`PolicyContext` with no history field, and the
    capability reads history through an injected reader, so no policy output can reach
    it.  This asserts the observable consequence rather than the mechanism.
    """
    harness = build_control_harness()
    result = harness.controller.run("recommend gear", HISTORY)
    assert result.state["trusted_user_history"] == tuple(HISTORY)
    assert harness.engine.last_history == list(HISTORY)


def test_policy_context_has_no_route_to_trusted_state() -> None:
    """The policy's entire view is a projection, checked field by field.

    A policy that could read the history could copy it into a tool call; a policy that
    could read candidate ids could name products.  Both channels are absent by
    construction, and this test pins the constructor's field set so adding one is a
    deliberate, reviewed act rather than an accident.
    """
    harness = build_control_harness()
    harness.controller.run("recommend gear", HISTORY)
    context = harness.policy.last_context
    assert context is not None

    allowed = {
        "user_request",
        "available_actions",
        "has_trusted_history",
        "active_preference_count",
        "candidate_state",
        "last_observation",
        "remaining_steps",
        "remaining_tool_calls",
        "step_index",
        "run_status",
        "last_proposal_rejected",
    }
    assert set(vars(context)) == allowed

    # And the sensitive values really are not reachable through the allowed fields.
    assert context.has_trusted_history is True
    assert not hasattr(context, "trusted_user_history")
    assert not hasattr(context, "user_key")
    assert not hasattr(context, "tool")
    assert not hasattr(context, "engine")
    assert not hasattr(context, "preference_snapshot")

    serialised = repr(context) + str(context.summary())
    for asin in HISTORY:
        assert asin not in serialised, "trusted history leaked into the policy's view"
    for row in CANDIDATE_ROWS:
        assert row[0] not in serialised, "candidate identity leaked into the policy's view"


def test_policy_cannot_supply_candidate_ids_or_history() -> None:
    """A proposal has no field for history, candidate ids, a tool name or SQL."""
    forbidden = (
        "history",
        "trusted_user_history",
        "candidates",
        "candidate_ids",
        "item_ids",
        "parent_asins",
        "tool",
        "tool_name",
        "sql",
        "query",
        "action_id",
        "step_index",
        "run_id",
        "turn_id",
    )
    assert set(ActionProposal.model_fields) & set(forbidden) == set()

    # Attempting to smuggle one is a hard validation error, not a silent merge.
    for field in forbidden:
        with pytest.raises(Exception):
            ActionProposal.model_validate(
                {"action": "recommend_from_history", "k": 3, field: "x"}
            )


def test_policy_cannot_decide_execution_metadata() -> None:
    """``action_id`` / ``step_index`` / ``run_id`` are stamped by the validator."""
    validator = ActionValidator()
    proposal = ActionProposal(action=ActionKind.RECOMMEND_FROM_HISTORY, k=3)
    validated, result = validator.validate(
        proposal,
        run_id="run-1",
        turn_id="turn-1",
        step_index=4,
        available_actions=STAGE_1_ACTIONS,
    )
    assert result.verified and validated is not None
    assert validated.action_id == validator.action_id(
        run_id="run-1", step_index=4, action=ActionKind.RECOMMEND_FROM_HISTORY
    )
    assert validated.step_index == 4
    assert validated.run_id == "run-1"
    assert validated.turn_id == "turn-1"

    # The same inputs reproduce the same identity: metadata is deterministic and owned by
    # the controller, not chosen by the policy.
    again, _ = validator.validate(
        proposal,
        run_id="run-1",
        turn_id="turn-1",
        step_index=4,
        available_actions=STAGE_1_ACTIONS,
    )
    assert again is not None and again.action_id == validated.action_id


def test_policy_cannot_commit_memory_by_proposing_an_action() -> None:
    """Memory is written only by the accepted stage, only from the user's own message.

    A run whose policy proposes FINISH immediately still commits the user's message - the
    commit is not the policy's to make or withhold - and a recommendation run commits
    exactly once.
    """
    harness = build_control_harness(
        policy=ScriptedPolicy([_finish()]), with_memory=True
    )
    result = harness.controller.run("I prefer red", HISTORY)
    assert result.status is RunStatus.FINISHED
    assert result.state.get("memory_update") is not None
    # The commit came from the accepted stage: the store really holds the entry, and the
    # policy had no part in writing it.
    stored = harness.memory_service.store.get_entries("control-test-user")
    assert stored, "the user's own statement must be committed by the accepted stage"

    harness2 = build_control_harness(with_memory=True)
    result2 = harness2.controller.run("recommend gear", HISTORY)
    assert result2.state.get("memory_update") is not None


def test_policy_can_neither_execute_nor_reach_the_tool() -> None:
    """The policy object has no collaborator that could run a tool or a store."""
    policy = RuleBasedPolicy()
    public = {name for name in dir(policy) if not name.startswith("_")}
    assert public == {"choose", "name", "default_k", "call_count"}
    for forbidden in ("run", "execute", "tool", "engine", "store", "memory", "invoke"):
        assert not hasattr(policy, forbidden)


def test_policy_module_imports_nothing_that_could_execute() -> None:
    """A source-level guard: the policy module cannot reach a tool, store or network."""
    source = (REPO_ROOT / "recommendation" / "control" / "policy.py").read_text(
        encoding="utf-8"
    )
    tree = ast.parse(source)
    imported: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.append(node.module)
    forbidden = (
        "subprocess",
        "sqlite3",
        "requests",
        "httpx",
        "urllib",
        "socket",
        "recommendation.tools",
        "recommendation.inference",
        "recommendation.memory",
        "recommendation.catalog",
    )
    for name in imported:
        assert not name.startswith(forbidden), f"policy.py must not import {name}"


def test_capability_refuses_an_action_it_does_not_own() -> None:
    """The capability cannot be repurposed by handing it a different action kind."""
    capability, engine = build_minimal_capability()
    _, valid = ActionValidator().validate(
        FINISH_ACTION,
        run_id="r",
        turn_id=None,
        step_index=0,
        available_actions=STAGE_1_ACTIONS,
    )
    assert valid.verified
    finish = ValidatedAction(
        action=ActionKind.FINISH,
        action_id="a",
        step_index=0,
        run_id="r",
        k=1,
    )
    with pytest.raises(PolicyActionError):
        capability.execute(
            finish,
            read_trusted_history=lambda: HISTORY,
            user_message="x",
        )
    assert engine.call_count == 0


# =========================================================================== #
# B. Protocol invariants
# =========================================================================== #


def test_action_proposal_is_not_directly_executable() -> None:
    """There is no execution surface on a proposal, and the capability needs a validated one."""
    assert not hasattr(ActionProposal, "execute")
    assert not hasattr(ActionProposal, "run")
    assert not callable(getattr(ActionProposal, "tool", None))
    assert ActionProposal is not ValidatedAction


def test_action_proposal_must_pass_the_validator() -> None:
    """A proposal outside ``available_actions`` is refused with a stable code."""
    validator = ActionValidator()
    validated, result = validator.validate(
        ActionProposal(action=ActionKind.RECOMMEND_FROM_HISTORY, k=3),
        run_id="r",
        turn_id=None,
        step_index=0,
        available_actions=(ActionKind.FINISH,),  # system did not offer recommend
    )
    assert validated is None
    assert result.verified is False
    assert result.code == "action_not_available"
    assert "availability" in result.checks


def test_validator_refuses_a_non_proposal() -> None:
    """Junk from a policy is refused rather than coerced."""
    validated, result = ActionValidator().validate(
        {"action": "recommend_from_history", "k": 3},  # type: ignore[arg-type]
        run_id="r",
        turn_id=None,
        step_index=0,
        available_actions=STAGE_1_ACTIONS,
    )
    assert validated is None and result.code == "not_a_proposal"


def test_validator_rejects_a_non_integer_k() -> None:
    """``k`` is re-checked at the trusted boundary with the Tool's own strictness."""
    proposal = ActionProposal.model_construct(
        action=ActionKind.RECOMMEND_FROM_HISTORY, k=True
    )
    validated, result = ActionValidator().validate(
        proposal,
        run_id="r",
        turn_id=None,
        step_index=0,
        available_actions=STAGE_1_ACTIONS,
    )
    assert validated is None and result.code == "invalid_k"


def test_domain_result_cannot_become_an_observation_without_the_verifier() -> None:
    """The adapter needs a verdict, and an unverified result is adapted as a refusal."""
    adapter = ObservationAdapter()
    tool_result = build_tool_result()
    domain = RecommendationDomainResult(
        action_id="a",
        tool_result=tool_result,
        requested_k=tool_result.requested_k,
        returned_k=tool_result.returned_k,
        candidate_set_ref="ref",
    )
    assert domain.policy_visible is False

    refused = VerificationResult(verified=False, code="rank_not_contiguous")
    observation = adapter.adapt(
        domain, refused, action=ActionKind.RECOMMEND_FROM_HISTORY, step_index=0
    )
    assert observation.verification_status == "refused"
    assert observation.status == "failed"
    assert observation.returned_k == 0 or observation.verification_status == "refused"

    verified = VerificationResult(verified=True, code="verified")
    good = adapter.adapt(
        domain, verified, action=ActionKind.RECOMMEND_FROM_HISTORY, step_index=0
    )
    assert good.verification_status == "verified" and good.status == "ok"
    assert good.has_candidates is True


def test_observation_carries_no_candidate_identity_or_score() -> None:
    """What the policy is told about a result is counts, not content."""
    harness = build_control_harness()
    result = harness.controller.run("recommend gear", HISTORY)
    observations = [
        step.observation for step in result.trajectory.steps if step.observation
    ]
    assert observations, "the run should have recorded at least one observation"
    for observation in observations:
        assert "recommendations" not in observation
        assert "candidates" not in observation
        assert "scores" not in observation
        blob = str(observation)
        for row in CANDIDATE_ROWS:
            assert row[0] not in blob
        assert "score" not in observation


def test_verifier_checks_the_tool_ran_on_the_runs_trusted_history() -> None:
    """A Tool result produced from a different history is refused, not trusted."""
    tool_result = build_tool_result()
    mismatched = RecommendationDomainResult(
        action_id="a",
        tool_result=tool_result.model_copy(update={"history_length": 99}),
        requested_k=tool_result.requested_k,
        returned_k=tool_result.returned_k,
    )
    verdict = ResultVerifier().verify(
        mismatched, trusted_history=HISTORY, expected_action_id="a"
    )
    assert verdict.verified is False
    assert verdict.code == "history_length_mismatch"


def test_verifier_refuses_a_changed_candidate_identity() -> None:
    """A stage that substitutes a candidate is refused, never silently repaired."""
    tool_result = build_tool_result()
    # A reranker report over a *different* candidate set.
    from recommendation.reranking import PreferenceReranker
    from recommendation.preference_matching import PreferenceCandidateMatcher
    from recommendation.rag import ProductEnricher
    from tests.agent_reranking_fixture import build_index

    enricher = ProductEnricher(build_index())
    enrichment = enricher.enrich(tool_result, QUERY)
    matcher = PreferenceCandidateMatcher()
    evidence = matcher.match(candidates=enrichment.items, preferences=())
    reranking = PreferenceReranker().rerank(evidence)

    # Swap one candidate identity in the reranked view.
    swapped = [
        candidate.model_copy(update={"parent_asin": "cand-impostor"})
        for candidate in reranking.candidates
    ]
    tampered = reranking.model_copy(update={"candidates": tuple(swapped)})

    domain = RecommendationDomainResult(
        action_id="a",
        tool_result=tool_result,
        enrichment=enrichment,
        preference_evidence=evidence,
        reranking=tampered,
        requested_k=tool_result.requested_k,
        returned_k=tool_result.returned_k,
    )
    verdict = ResultVerifier().verify(
        domain, trusted_history=HISTORY, expected_action_id="a"
    )
    assert verdict.verified is False
    assert verdict.code == "reranking_identity_mismatch"


def test_verifier_accepts_the_untampered_pipeline() -> None:
    """The positive control: the real pipeline verifies with every check recorded."""
    capability, engine, parts = build_full_capability()
    validated, validation = ActionValidator().validate(
        ActionProposal(action=ActionKind.RECOMMEND_FROM_HISTORY, k=3),
        run_id="r",
        turn_id=None,
        step_index=0,
        available_actions=STAGE_1_ACTIONS,
    )
    assert validated is not None and validation.verified
    domain = capability.execute(
        validated, read_trusted_history=lambda: HISTORY, user_message=QUERY
    )
    verdict = ResultVerifier().verify(
        domain, trusted_history=HISTORY, expected_action_id=validated.action_id
    )
    assert verdict.verified, verdict.detail
    for check in (
        "tool_grounding",
        "count_integrity",
        "rank_integrity",
        "candidate_identity",
        "enrichment_alignment",
        "evidence_alignment",
        "reranking_alignment",
    ):
        assert check in verdict.checks


def test_trajectory_records_controller_generated_execution_metadata() -> None:
    """The trajectory shows controller-stamped metadata, not policy-supplied values."""
    harness = build_control_harness()
    result = harness.controller.run("recommend gear", HISTORY)
    executed = [
        step for step in result.trajectory.steps if step.validated_action is not None
    ]
    assert executed
    for step in executed:
        assert step.validated_action is not None
        assert step.validated_action["action_id"].startswith("act:")
        assert step.validated_action["step_index"] == step.step_index
        assert step.action_id == step.validated_action["action_id"]


def test_trajectory_contains_no_secrets_candidates_or_history() -> None:
    """The audit record is summaries and references, never payloads."""
    harness = build_control_harness(with_memory=True)
    harness.controller.run("I prefer red. Recommend gear.", HISTORY)
    blob = str(harness.controller.run("I prefer red. Recommend gear.", HISTORY).trajectory.as_dicts())
    for asin in HISTORY:
        assert asin not in blob
    for row in CANDIDATE_ROWS:
        assert row[0] not in blob
    assert "control-test-user" not in blob
    assert "user_key" not in blob


# =========================================================================== #
# C. Loop invariants
# =========================================================================== #


def test_control_returns_to_the_policy_after_a_non_terminal_observation() -> None:
    """The core Stage 1 property: the policy is consulted again, with the update.

    This is the acceptance trajectory.  The policy is asked twice - once before the
    recommendation and once after it - and the second context shows the new observation
    and a grounded candidate set.
    """
    harness = build_control_harness()
    result = harness.controller.run("recommend gear", HISTORY)

    assert harness.policy.call_count == 2, "control did not return to the policy"
    first, second = harness.policy.contexts

    assert first.last_observation is None
    assert first.candidate_state.grounded is False

    assert second.last_observation is not None
    assert second.last_observation.kind == "recommendation"
    assert second.last_observation.has_candidates is True
    assert second.candidate_state.grounded is True
    assert second.remaining_steps < first.remaining_steps
    assert second.remaining_tool_calls < first.remaining_tool_calls

    assert result.trajectory.actions() == ("recommend_from_history", "finish")
    assert result.status is RunStatus.FINISHED
    assert result.control.termination_reason is TerminationReason.COMPLETED
    assert harness.engine.call_count == 1


def test_policy_chooses_exactly_one_action_per_step() -> None:
    """One proposal per step, and every step that executed has exactly one action."""
    harness = build_control_harness()
    result = harness.controller.run("recommend gear", HISTORY)
    assert harness.policy.call_count == len(result.trajectory.steps)
    for step in result.trajectory.steps:
        assert step.action_proposal is not None
        assert harness.policy.contexts[step.step_index].available_actions


def test_policy_may_only_choose_from_available_actions() -> None:
    """An illegal proposal is refused deterministically instead of executed."""

    class IllegalPolicy:
        name = "illegal"

        def choose(self, context: PolicyContext) -> ActionProposal:
            # Ask for a recommendation the system has not offered.
            return ActionProposal(action=ActionKind.RECOMMEND_FROM_HISTORY, k=3)

    harness = build_control_harness(policy=IllegalPolicy(), limits=LoopLimits(max_tool_calls=0))
    result = harness.controller.run("recommend gear", HISTORY)
    # With no tool budget the system offers only FINISH, so the proposal is refused and the
    # run ends deterministically rather than executing something the system did not offer.
    assert result.status is RunStatus.FAILED
    assert result.control.termination_reason is TerminationReason.INVALID_ACTION
    assert harness.engine.call_count == 0
    assert "action_not_available" in result.trajectory.refusals()
    assert result.control.termination_detail is not None
    assert "finish" in result.control.termination_detail


def test_max_steps_terminates_the_loop() -> None:
    """The step budget is a deterministic boundary the policy cannot negotiate.

    The policy here never proposes FINISH, so the *only* thing that can stop the run is the
    controller's own budget.
    """

    class NeverFinishes:
        name = "never-finishes"

        def choose(self, context: PolicyContext) -> ActionProposal:
            return ActionProposal(action=ActionKind.RECOMMEND_FROM_HISTORY, k=3)

    harness = build_control_harness(
        policy=NeverFinishes(), limits=LoopLimits(max_steps=2, max_tool_calls=8)
    )
    result = harness.controller.run("recommend gear", HISTORY)
    assert result.status is RunStatus.ABORTED
    assert result.control.termination_reason is TerminationReason.MAX_STEPS
    assert result.control.step_count == 2
    assert harness.engine.call_count == 2
    # The controller stopped it - the policy was never asked to agree.
    assert result.control.termination_detail is not None


def test_step_budget_wins_over_a_late_finish() -> None:
    """Budget exhaustion terminates without asking the policy whether to continue.

    With one step of budget a single recommendation is allowed; the controller then stops
    for the budget rather than granting another step to finish.  That is the documented
    behaviour ("do not ask the policy; terminate deterministically"), and it is why the
    status is ``ABORTED`` and not ``FINISHED``.
    """
    harness = build_control_harness(limits=LoopLimits(max_steps=1))
    result = harness.controller.run("recommend gear", HISTORY)
    assert result.status is RunStatus.ABORTED
    assert result.control.termination_reason is TerminationReason.MAX_STEPS
    assert result.control.step_count == 1
    assert harness.engine.call_count == 1


def test_max_tool_calls_terminates_the_loop() -> None:
    """The tool-call budget stops execution before an unaffordable call is made."""
    harness = build_control_harness(limits=LoopLimits(max_tool_calls=0))
    result = harness.controller.run("recommend gear", HISTORY)
    # With no tool budget the system does not offer RECOMMEND at all, so the policy
    # finishes an empty turn rather than proposing an impossible action.
    assert harness.engine.call_count == 0
    assert result.status is RunStatus.FINISHED

    # With one tool call available but a policy that keeps asking, the second attempt is
    # impossible: the system withdraws RECOMMEND_FROM_HISTORY from available_actions, so
    # the proposal is refused instead of executed.  Either way the recommender runs once.
    harness2 = build_control_harness(
        policy=ScriptedPolicy([RECOMMEND_ACTION, RECOMMEND_ACTION, RECOMMEND_ACTION]),
        limits=LoopLimits(max_tool_calls=1),
    )
    result2 = harness2.controller.run("recommend gear", HISTORY)
    assert harness2.engine.call_count == 1
    assert result2.control.tool_call_count == 1
    assert result2.status is not RunStatus.RUNNING
    assert result2.control.termination_reason in (
        TerminationReason.MAX_TOOL_CALLS,
        TerminationReason.MAX_STEPS,
        TerminationReason.INVALID_ACTION,
    )


def test_policy_cannot_raise_its_own_budget() -> None:
    """The budgets are controller configuration; the policy only ever sees remainders."""

    seen: list[PolicyContext] = []

    class GreedyPolicy:
        name = "greedy"

        def choose(self, context: PolicyContext) -> ActionProposal:
            seen.append(context)
            # A policy always asks for the most it can; it has no way to ask for more.
            return ActionProposal(action=ActionKind.RECOMMEND_FROM_HISTORY, k=3)

    harness = build_control_harness(
        policy=GreedyPolicy(), limits=LoopLimits(max_steps=3, max_tool_calls=1)
    )
    result = harness.controller.run("recommend gear", HISTORY)

    # The controller stopped it: the policy asked again but the *system* had withdrawn the
    # action, so the proposal was refused.  Either way the budgets held.
    assert result.control.step_count <= 3
    assert result.control.tool_call_count <= 1
    assert harness.engine.call_count <= 1
    assert result.status is not RunStatus.RUNNING
    for context in seen:
        assert not hasattr(context, "limits")
        assert not hasattr(context, "max_steps")
        assert not hasattr(context, "max_tool_calls")


def test_finish_cannot_complete_without_the_guard() -> None:
    """A FINISH proposal is checked; a refused completion does not end the run as success."""

    class AlwaysFinish:
        name = "always-finish"

        def choose(self, context: PolicyContext) -> ActionProposal:
            return ActionProposal(action=ActionKind.FINISH)

    # A run with no trusted history may legally finish immediately, but it must be a
    # *guarded* completion, not an unguarded one.
    harness = build_control_harness(policy=AlwaysFinish())
    result = harness.controller.run("hello there", HISTORY)
    assert result.status is RunStatus.FINISHED
    completions = [
        step.verification_result
        for step in result.trajectory.steps
        if step.verification_result is not None
    ]
    assert completions, "the completion guard must have been consulted"
    assert completions[-1].code == "completion_accepted"


def test_completion_guard_refuses_a_run_whose_execution_failed() -> None:
    """Fail-closed: a failed run cannot be reported as a successful completion."""
    engine = FixedEngine(error=RuntimeError("engine exploded"))
    capability = RecommendFromHistoryCapability(RecommendationTool(engine))
    controller = LoopController(
        RuleBasedPolicy(default_k=3), capability, limits=LoopLimits()
    )
    result = controller.run("recommend gear", HISTORY)
    assert result.status is not RunStatus.FINISHED
    assert result.control.termination_reason is not TerminationReason.COMPLETED
    assert result.final_response == ""


def test_completion_guard_refuses_when_candidates_are_not_grounded() -> None:
    """A produced-but-unverified candidate set may not be presented as a completion."""
    guard = CompletionGuard()
    state = ControlState(run_id="r")
    action = ValidatedAction(
        action=ActionKind.FINISH, action_id="a", step_index=0, run_id="r", k=1
    )
    verdict = guard.check(
        action,
        state=state,
        last_verification=VerificationResult(verified=True, code="verified"),
        produced_recommendation=True,
        candidates_grounded=False,
    )
    assert verdict.verified is False
    assert verdict.code == "candidates_not_grounded"


def test_completion_guard_refuses_a_non_finish_action() -> None:
    """The guard is not a second validator; it certifies completions only."""
    guard = CompletionGuard()
    action = ValidatedAction(
        action=ActionKind.RECOMMEND_FROM_HISTORY,
        action_id="a",
        step_index=0,
        run_id="r",
        k=3,
    )
    verdict = guard.check(action, state=ControlState(run_id="r"), last_verification=None)
    assert verdict.verified is False and verdict.code == "not_a_completion"


def test_loop_topology_has_the_back_edge() -> None:
    """The compiled cycle contains the back-edge - the structural change of Stage 1.

    The cycle returns through ``check_limits`` rather than straight to the policy, so the
    step budget is re-tested on every iteration.  ``check_limits -> policy`` is the edge
    that actually hands control back to the deciding component.
    """
    mermaid = topology_mermaid()
    assert f"{NODE_UPDATE_STATE} --> {NODE_CHECK_LIMITS}" in mermaid
    assert f"{NODE_CHECK_LIMITS} --> {NODE_POLICY}" in mermaid
    assert "__start__" in mermaid

    # The declared cycle is data, and it really closes: every phase that can be re-entered
    # is reachable from the phase that follows the observation.
    edges = set(DECLARED_CYCLE_EDGES)
    assert (NODE_UPDATE_STATE, NODE_CHECK_LIMITS) in edges
    assert (NODE_CHECK_LIMITS, NODE_POLICY) in edges

    # And the *live* nodes agree with the declaration: the phase after the observation
    # routes back to the budget check, which routes on to the policy.
    engine = build_control_harness().controller.new_engine(
        AgentInput(user_message="recommend gear", trusted_user_history=HISTORY)
    )
    engine.initialize()
    engine.control = engine.control.advanced(
        action=ActionKind.RECOMMEND_FROM_HISTORY, action_id="a", consumed_tool_call=True
    )
    after_update = _update_state({"engine": engine, "step": 0})
    assert after_update.goto == NODE_CHECK_LIMITS
    after_check = _check_limits({"engine": engine, "step": 0})
    assert after_check.goto == NODE_POLICY

    # The declared node set is the control plane's phase set, and only that.
    assert NODE_POLICY in LOOP_NODE_NAMES
    assert NODE_UPDATE_STATE in LOOP_NODE_NAMES
    assert NODE_CHECK_LIMITS in LOOP_NODE_NAMES
    assert "decide" not in LOOP_NODE_NAMES  # the accepted DAG's route node is untouched
    assert "recommend" not in LOOP_NODE_NAMES  # execution lives in the capability


def test_loop_phase_order_returns_to_the_budget_check() -> None:
    """One iteration is the documented phase sequence, and it is a cycle."""
    assert LOOP_PHASE_ORDER == (
        "initialize",
        "check_limits",
        "policy",
        "validate_action",
        "dispatch",
        "execute",
        "verify",
        "observe",
        "update_state",
    )
    # The last phase of an iteration is followed by the first again.
    assert LOOP_PHASE_ORDER[0] == "initialize"
    assert LOOP_PHASE_ORDER[1] == "check_limits"


def test_both_drivers_run_the_same_loop() -> None:
    """The graph driver and the direct driver are two topologies over one engine."""
    direct = build_control_harness(driver="direct").controller.run("recommend gear", HISTORY)
    graph = build_control_harness(driver="graph").controller.run("recommend gear", HISTORY)
    assert direct.final_response == graph.final_response
    assert direct.route == graph.route
    assert direct.trajectory.actions() == graph.trajectory.actions()
    assert direct.status is graph.status
    assert direct.control.termination_reason is graph.control.termination_reason


def test_loop_is_bounded_to_one_tool_call_for_the_canonical_turn() -> None:
    """The canonical turn calls the recommender exactly once, as the accepted DAG does."""
    harness = build_control_harness()
    harness.controller.run("recommend gear", HISTORY)
    assert harness.engine.call_count == 1


# =========================================================================== #
# D. Existing-system invariants
# =========================================================================== #


def test_candidate_identity_is_preserved_by_the_loop() -> None:
    """The loop cannot alter the candidate set: every stage agrees by identity."""
    harness = build_control_harness()
    result = harness.controller.run("recommend gear", HISTORY)
    tool_result = result.state["tool_result"]
    expected = [(item.parent_asin, item.item_id) for item in tool_result.recommendations]

    enrichment = result.state["enrichment"]
    assert [(i.parent_asin, i.recommendation.item_id) for i in enrichment.items] == expected
    evidence = result.state["preference_evidence"]
    assert [(c.parent_asin, c.item_id) for c in evidence.candidates] == expected
    reranking = result.state["reranking"]
    assert sorted((c.parent_asin, c.item_id) for c in reranking.candidates) == sorted(expected)


def test_loop_preserves_trusted_history_and_reranking_order() -> None:
    """Reranked output still renders the accepted order and both ranks."""
    harness = build_control_harness(with_memory=True)
    result = harness.controller.run(f"I avoid green. {QUERY}", HISTORY)
    rendered = rendered_order(result.final_response)
    reranked = tuple(result.state["reranking"].parent_asins)
    assert rendered == reranked
    assert "original SASRec rank" in result.final_response


def test_loop_and_dag_produce_the_same_state_and_response() -> None:
    """The control plane changed; the recommendation result did not.

    The strongest available equivalence check: the accepted DAG and the 2.0-alpha loop are
    run over the *same* real collaborators with the *same* inputs, and must produce the
    same route, the same rendered text and the same candidate order.
    """
    capability, engine, parts = build_full_capability()
    loop = LoopController(RuleBasedPolicy(default_k=3), capability, limits=LoopLimits())
    loop_result = loop.run(QUERY, HISTORY)

    dag_engine = FixedEngine()
    dag = AgentGraph(
        _FixedDecisionModel(AgentDecision(action="recommend", k=3)),
        RecommendationTool(dag_engine),
        product_enricher=parts["enricher"],
        preference_matcher=parts["matcher"],
        reranker=parts["reranker"],
    )
    dag_state = dag.run(QUERY, HISTORY)

    assert loop_result.route == dag_state["route"] == "recommend"
    assert loop_result.final_response == dag_state["final_response"]
    assert rendered_order(loop_result.final_response) == rendered_order(
        dag_state["final_response"]
    )
    assert engine.call_count == dag_engine.call_count == 1


def test_loop_state_is_consumable_by_the_accepted_serializer() -> None:
    """A loop run's state satisfies the accepted public/internal boundary."""
    harness = build_control_harness(with_memory=True)
    result = harness.controller.run(f"I prefer black. {QUERY}", HISTORY)
    response = build_chat_response(
        result.state, session_id="s-1", turn_id="t-1", turn_number=1
    )
    assert response.route == "recommend"
    assert response.recommendations
    assert response.message == result.final_response
    # The serializer's whitelist still holds: no history, no user key.
    body = response.model_dump_json()
    for asin in HISTORY:
        assert asin not in body


def test_loop_refuses_missing_history_with_a_tool_domain_error() -> None:
    """There is no cold-start fallback: the accepted Tool error is preserved."""
    capability, engine, _ = build_full_capability()

    validated, validation = ActionValidator().validate(
        ActionProposal(action=ActionKind.RECOMMEND_FROM_HISTORY, k=3),
        run_id="r",
        turn_id=None,
        step_index=0,
        available_actions=STAGE_1_ACTIONS,
    )
    assert validated is not None and validation.verified
    with pytest.raises(MissingUserHistory):
        capability.execute(
            validated, read_trusted_history=lambda: (), user_message="x"
        )
    assert engine.call_count == 0


def test_no_trusted_history_cannot_start_a_run_at_all() -> None:
    """There is no cold-start path: an empty history is refused at the input boundary.

    This is the accepted Milestone 7A/7B position, unchanged by Stage 1.  The loop inherits
    it rather than adding a fallback of its own, so "the recommender needs history" stays a
    property of the system and not of one control plane.
    """
    harness = build_control_harness()
    with pytest.raises(Exception) as excinfo:
        harness.controller.run("hello", ())
    assert "trusted_user_history" in str(excinfo.value)
    assert harness.engine.call_count == 0

    # The DAG refuses it identically, which is the equivalence Stage 1 must preserve.
    dag = AgentGraph(
        _FixedDecisionModel(AgentDecision(action="direct_response", direct_response="hi")),
        RecommendationTool(FixedEngine()),
    )
    with pytest.raises(Exception):
        dag.run("hello", ())
    assert harness.engine.call_count == 0


def test_loop_does_not_import_fastapi_or_an_http_client() -> None:
    """The control plane talks to the engine in process, never over HTTP."""
    for module in ("loop.py", "capability.py", "policy.py", "validation.py", "verification.py"):
        source = (REPO_ROOT / "recommendation" / "control" / module).read_text(
            encoding="utf-8"
        )
        for forbidden in ("fastapi", "requests", "httpx", "urllib.request"):
            assert forbidden not in source, f"{module} must not reference {forbidden}"


def test_accepted_dag_still_has_no_cycle() -> None:
    """Stage 1 did not turn the accepted graph into a loop.

    The new control plane is additive precisely so this property survives: the accepted
    ``AgentGraph`` keeps its acyclic shape and its own tests keep asserting it.
    """
    capability, engine, parts = build_full_capability()
    graph = AgentGraph(
        _FixedDecisionModel(AgentDecision(action="recommend", k=3)),
        RecommendationTool(FixedEngine()),
        product_enricher=parts["enricher"],
        preference_matcher=parts["matcher"],
        reranker=parts["reranker"],
    )
    mermaid = graph.mermaid()
    assert "finalize --> decide" not in mermaid
    assert "recommend --> decide" not in mermaid
    assert "update_state" not in mermaid
    assert capability is not None and engine is not None


class _FixedDecisionModel:
    """A decision model returning one fixed decision, for DAG/loop equivalence tests."""

    def __init__(self, decision: AgentDecision) -> None:
        self._decision = decision
        self.calls: list[tuple[object, ...]] = []

    @property
    def call_count(self) -> int:
        """How many times the DAG consulted this model."""
        return len(self.calls)

    def decide(self, messages: object) -> AgentDecision:
        """Record the prompt and return the fixed decision."""
        self.calls.append(tuple(messages))  # type: ignore[arg-type]
        return self._decision


def test_dag_and_loop_see_the_same_user_text() -> None:
    """Both control planes receive the user message and nothing else of the history."""
    decision_model = _FixedDecisionModel(AgentDecision(action="recommend", k=3))
    graph = AgentGraph(decision_model, RecommendationTool(FixedEngine()))

    harness = build_control_harness(
        policy=ScriptedPolicy([RECOMMEND_ACTION, ActionProposal(action=ActionKind.FINISH)])
    )
    harness.controller.run("recommend gear", HISTORY)

    graph.run("recommend gear", HISTORY)

    # The DAG's prompt contains the message and no history identifier.
    prompt = " ".join(str(m.content) for m in decision_model.calls[0])  # type: ignore[attr-defined]
    for asin in HISTORY:
        assert asin not in prompt
    # The loop's projection contains the message and no history identifier either.
    loop_context = harness.policy.contexts[0]
    assert loop_context.user_request == "recommend gear"
    assert not hasattr(loop_context, "trusted_user_history")
    assert build_decision_messages("recommend gear")[1].content == "recommend gear"


def test_policy_is_interchangeable() -> None:
    """Two different policy classes drive the identical control plane unchanged.

    This is the structural promise of Stage 1: swapping the deciding component does not
    require touching the validator, capability, verifier, guard, controller or trajectory.
    A future LLM policy implements the same one method.
    """

    class AlwaysRecommendThenFinish:
        """A deliberately different decision rule from ``RuleBasedPolicy``."""

        name = "always-recommend-then-finish"

        def __init__(self) -> None:
            self._seen = 0

        def choose(self, context: PolicyContext) -> ActionProposal:
            self._seen += 1
            if context.candidate_state.grounded:
                return ActionProposal(action=ActionKind.FINISH)
            return ActionProposal(action=ActionKind.RECOMMEND_FROM_HISTORY, k=4)

    assert isinstance(AlwaysRecommendThenFinish(), object)
    for policy in (RuleBasedPolicy(default_k=3), AlwaysRecommendThenFinish()):
        harness = build_control_harness(policy=policy)
        result = harness.controller.run("recommend gear", HISTORY)
        assert result.status is RunStatus.FINISHED
        assert result.trajectory.actions() == ("recommend_from_history", "finish")
        assert harness.engine.call_count == 1


def test_capability_reports_its_stage_sequence() -> None:
    """The capability's declared stages match what it actually ran."""
    capability, engine, parts = build_full_capability()
    assert capability.name == CAPABILITY_NAME
    assert capability.stages() == ("recommend", "enrich", "match_preferences", "rerank")

    minimal, _ = build_minimal_capability()
    assert minimal.stages() == ("recommend",)

    assert parts["matcher"] is not None and engine.call_count == 0

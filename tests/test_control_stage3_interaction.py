"""Stage 3 tests: task state, clarification and the memory proposal pipeline.

What this file is for
---------------------
Stage 3 introduces three things that are easy to get subtly wrong, and each has its own
family of tests here:

**A. Signal separation and scope.**  Behavioural history, long-term explicit memory and
current task intent are three different things.  The rule that matters is that a
``TASK``-scoped instruction takes effect **immediately** while a ``LONG_TERM`` one is
*proposed* for durable commit - and that a temporary override never deletes stored state.

**B. Clarification is a suspension.**  ``ASK_CLARIFICATION`` must not run a tool, must not
complete the run, must not fail it, and must be resumable.  ``WAITING_FOR_USER`` is a real
status, distinct from ``FINISHED`` and ``ABORTED``.

**C. A memory write is a proposal, not a commit.**  Nothing the interpreter produces may
reach persistent memory without passing the validator, and an inference may never become
durable truth.

Everything is offline and deterministic: no checkpoint, GPU, network or provider API.  Stage
1 and Stage 2 suites run unchanged alongside this file.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from recommendation.control import (  # noqa: E402
    ActionKind,
    ActionProposal,
    AskClarificationArguments,
    ClarificationObservation,
    LoopLimits,
    PolicyContext,
    RunStatus,
    TerminationReason,
)
from recommendation.control.arguments import ASK_QUESTION_MAX_LENGTH  # noqa: E402
from recommendation.control.memory_proposal import (  # noqa: E402
    MemoryProposalValidator,
    MemoryWriteOperation,
    MemoryWriteProposal,
    ProposalVerdict,
    approved_candidates,
    proposals_from_extraction,
)
from recommendation.control.task_state import (  # noqa: E402
    ConstraintOrigin,
    PreferenceLifecycle,
    PreferenceScope,
    TaskConstraint,
    TaskIntent,
    TaskState,
)
from recommendation.memory import RuleBasedPreferenceExtractor  # noqa: E402
from recommendation.memory.schemas import (  # noqa: E402
    PreferenceKind,
    PreferencePolarity,
)
from tests.agent_fakes import HISTORY  # noqa: E402
from tests.control_fixture import build_control_harness  # noqa: E402

# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #


def constraint(
    *,
    kind: PreferenceKind = PreferenceKind.COLOR,
    value: str = "red",
    polarity: PreferencePolarity = PreferencePolarity.AVOID,
    scope: PreferenceScope = PreferenceScope.TASK,
    lifecycle: PreferenceLifecycle = PreferenceLifecycle.EXPLICIT,
    origin: ConstraintOrigin = ConstraintOrigin.CURRENT_TURN,
    source_text: str = "",
) -> TaskConstraint:
    """Build a task constraint with sensible defaults."""
    return TaskConstraint(
        kind=kind,
        value=value,
        polarity=polarity,
        scope=scope,
        lifecycle=lifecycle,
        origin=origin,
        source_text=source_text or value,
    )


def proposal(
    *,
    operation: MemoryWriteOperation = MemoryWriteOperation.ADD,
    kind: PreferenceKind = PreferenceKind.COLOR,
    value: str = "red",
    polarity: PreferencePolarity = PreferencePolarity.AVOID,
    scope: PreferenceScope = PreferenceScope.LONG_TERM,
    lifecycle: PreferenceLifecycle = PreferenceLifecycle.EXPLICIT,
    source_span: str = "never red again",
) -> MemoryWriteProposal:
    """Build a memory write proposal with sensible defaults."""
    return MemoryWriteProposal(
        operation=operation,
        kind=kind,
        value=value,
        polarity=polarity,
        scope=scope,
        lifecycle=lifecycle,
        source_span=source_span,
        turn_id="t1",
        interpreter="test_interpreter",
    )


class ClarifyingPolicy:
    """Asks a question until the task holds a colour answer, then recommends.

    The *policy* decides when the missing information is material - not the framework.  It
    asks while it has no colour constraint for the task and recommends once it does, which is
    what makes the resumed turn proceed instead of asking again.
    """

    name = "clarifying"

    def __init__(self) -> None:
        self.contexts: list[PolicyContext] = []

    def choose(self, context: PolicyContext) -> ActionProposal:
        self.contexts.append(context)
        if context.candidate_state.grounded:
            return ActionProposal(action=ActionKind.FINISH)
        # A colour constraint in task state means the answer has arrived.
        if any("color" in text for text in context.task_constraints):
            return ActionProposal(action=ActionKind.RECOMMEND_FROM_HISTORY, k=3)
        if not context.action_available(ActionKind.ASK_CLARIFICATION):
            return ActionProposal(action=ActionKind.RECOMMEND_FROM_HISTORY, k=3)
        return ActionProposal(
            action=ActionKind.ASK_CLARIFICATION,
            arguments=AskClarificationArguments(
                question="Which colour would you like instead?",
                blocks="ranking",
                options=("blue", "black"),
            ),
        )


class AlwaysRecommend:
    """Recommends once and finishes; used where clarification must not happen."""

    name = "always-recommend"

    def choose(self, context: PolicyContext) -> ActionProposal:
        if context.candidate_state.grounded:
            return ActionProposal(action=ActionKind.FINISH)
        return ActionProposal(action=ActionKind.RECOMMEND_FROM_HISTORY, k=3)


# =========================================================================== #
# A. Signal separation and scope
# =========================================================================== #


def test_task_and_long_term_scope_are_distinguishable() -> None:
    """Scope is the semantic that keeps a temporary instruction from rewriting a durable one."""
    assert PreferenceScope.TASK.is_ephemeral is True
    assert PreferenceScope.SESSION.is_ephemeral is True
    assert PreferenceScope.LONG_TERM.is_ephemeral is False


def test_only_explicit_or_confirmed_signals_may_be_committed() -> None:
    """An inferred hypothesis is soft evidence and never durable truth."""
    assert PreferenceLifecycle.EXPLICIT.may_be_committed is True
    assert PreferenceLifecycle.CONFIRMED.may_be_committed is True
    assert PreferenceLifecycle.INFERRED_HYPOTHESIS.may_be_committed is False
    assert PreferenceLifecycle.INFERRED_HYPOTHESIS.is_soft_evidence is True
    assert PreferenceLifecycle.RETRACTED.may_be_committed is False


def test_task_constraints_define_the_feasible_set_and_inferences_do_not() -> None:
    """Only explicit task-scoped constraints are hard; an inference only orders."""
    state = TaskState(intent=TaskIntent.RECOMMEND).with_constraints(
        (
            constraint(value="red", polarity=PreferencePolarity.AVOID),
            constraint(
                kind=PreferenceKind.BRAND,
                value="Acme",
                polarity=PreferencePolarity.PREFER,
                lifecycle=PreferenceLifecycle.INFERRED_HYPOTHESIS,
                origin=ConstraintOrigin.INFERRED,
            ),
        )
    )
    hard = state.hard_constraints()
    soft = state.soft_constraints()
    assert [c.value for c in hard] == ["red"]
    assert [c.value for c in soft] == ["Acme"]
    assert state.avoid_values(PreferenceKind.COLOR) == ("red",)


def test_a_repeated_statement_collapses_instead_of_duplicating() -> None:
    """Task state is a set of current instructions, not a transcript."""
    state = TaskState()
    state = state.with_constraint(constraint(value="red"))
    state = state.with_constraint(constraint(value="red"))
    assert len(state.constraints) == 1
    state = state.with_constraint(constraint(value="blue"))
    assert len(state.constraints) == 2


def test_task_state_is_never_persisted_by_the_agent() -> None:
    """Task constraints live in the task projection; nothing here writes memory."""
    state = TaskState(request_text="not red").with_constraint(constraint(value="red"))
    # The only persistence surface on task state is none at all: there is no store, no
    # commit and no writer method.
    for forbidden in ("commit", "save", "persist", "write", "store"):
        assert not hasattr(state, forbidden)
    assert state.constraints[0].scope.is_ephemeral is True


def test_current_task_overrides_persistent_preference_without_deleting_it() -> None:
    """The same value stated for the task and for the long term coexist as separate records.

    A task-scoped override *suppresses* within the task by being the constraint the policy
    is shown; the long-term record is a different object with a different scope and is not
    mutated, so suppression is structurally not removal.
    """
    persistent = constraint(value="red", scope=PreferenceScope.LONG_TERM)
    override = constraint(
        value="blue",
        polarity=PreferencePolarity.PREFER,
        scope=PreferenceScope.TASK,
        source_text="blue this time",
    )
    state = TaskState().with_constraints((persistent, override))

    scopes = {(c.value, c.scope) for c in state.constraints}
    assert ("red", PreferenceScope.LONG_TERM) in scopes
    assert ("blue", PreferenceScope.TASK) in scopes
    # The long-term entry is still present and still long-term: nothing retired it.
    long_term = [c for c in state.constraints if c.scope is PreferenceScope.LONG_TERM]
    assert len(long_term) == 1 and long_term[0].value == "red"


def test_policy_context_projects_constraints_but_not_memory_internals() -> None:
    """The policy sees attributed constraint text, not memory entries."""
    state = TaskState(intent=TaskIntent.RECOMMEND).with_constraints(
        (constraint(value="red"), constraint(value="blue", polarity=PreferencePolarity.PREFER))
    )
    from recommendation.control.context import project_constraints, project_intent

    described, hard, inferred = project_constraints(state)
    assert described == ("color: avoids red", "color: prefers blue")
    assert hard == 2 and inferred == 0
    assert project_intent(state) == "recommend"
    blob = str(described)
    for forbidden in ("memory_id", "user_key", "superseded_by", "logical_seq"):
        assert forbidden not in blob


def test_policy_context_summary_records_counts_not_preference_values() -> None:
    """The trajectory summary must not become a copy of the user's preferences."""
    context = PolicyContext(
        user_request="show me some boots",
        available_actions=(ActionKind.FINISH,),
        has_trusted_history=True,
        # A value chosen so that no summary *key* contains it as a substring: the assertion
        # is about the user's preference value, not about incidental letter sequences.
        task_constraints=("color: avoids crimson",),
        hard_constraint_count=1,
        inferred_constraint_count=0,
    )
    summary = context.summary()
    assert summary["task_constraint_count"] == 1
    assert summary["hard_constraint_count"] == 1
    # The *count* is recorded; the constraint value and its description are not.
    assert "crimson" not in str(summary), "the summary must carry counts, not values"
    assert "avoids" not in str(summary)
    assert set(summary) >= {"task_constraint_count", "hard_constraint_count"}


# =========================================================================== #
# B. Clarification as a suspension
# =========================================================================== #


def test_clarification_suspends_without_running_a_tool() -> None:
    """ASK_CLARIFICATION is a suspension: no tool call, no completion, no failure."""
    policy = ClarifyingPolicy()
    harness = build_control_harness(policy=policy, limits=LoopLimits(max_steps=4))
    result = harness.controller.run("not red", HISTORY)

    assert result.status is RunStatus.WAITING_FOR_USER
    assert result.control.termination_reason is TerminationReason.AWAITING_USER
    assert harness.engine.call_count == 0, "a question must not consume a recommender call"
    assert result.control.tool_call_count == 0
    assert result.state["clarification_question"] == "Which colour would you like instead?"
    assert result.final_response == "Which colour would you like instead?"


def test_clarification_is_not_a_completion_and_not_a_failure() -> None:
    """The status is distinct from FINISHED and ABORTED, and the guard never ran."""
    harness = build_control_harness(policy=ClarifyingPolicy(), limits=LoopLimits(max_steps=4))
    result = harness.controller.run("not red", HISTORY)
    assert result.status is not RunStatus.FINISHED
    assert result.status is not RunStatus.ABORTED
    assert result.status is not RunStatus.FAILED
    assert result.succeeded is False
    # The completion guard was never consulted: the run did not propose FINISH.
    completion_codes = [
        step.verification_result.code
        for step in result.trajectory.steps
        if step.verification_result is not None
    ]
    assert "clarification_requested" in completion_codes
    assert "completion_accepted" not in completion_codes


def test_clarification_observation_records_the_question_and_what_it_blocks() -> None:
    """The trajectory can answer 'why did the run stop mid-way?'."""
    harness = build_control_harness(policy=ClarifyingPolicy(), limits=LoopLimits(max_steps=4))
    result = harness.controller.run("not red", HISTORY)
    observation = ClarificationObservation.model_validate(
        result.trajectory.steps[0].observation
    )
    assert observation.kind == "clarification"
    assert observation.status == "waiting_for_user"
    assert observation.blocks == "ranking"
    assert observation.options == ("blue", "black")


def test_a_suspended_run_resumes_and_continues_the_task() -> None:
    """The task is resumable: the answer reaches the policy and the run proceeds."""
    policy = ClarifyingPolicy()
    harness = build_control_harness(policy=policy, limits=LoopLimits(max_steps=5))

    first = harness.controller.run("not red", HISTORY)
    assert first.status is RunStatus.WAITING_FOR_USER

    suspended = TaskState(
        task_id="task-1",
        request_text="not red",
        pending_question=first.state["clarification_question"],
    )
    answered = suspended.resumed().with_constraint(
        constraint(
            value="blue",
            polarity=PreferencePolarity.PREFER,
            source_text="blue please",
        )
    )
    second = harness.controller.run("blue please", HISTORY, turn_id="t2", task_state=answered)

    assert second.status is RunStatus.FINISHED
    assert second.control.termination_reason is TerminationReason.COMPLETED
    assert harness.engine.call_count == 1, "the resumed turn runs the recommender once"
    # The policy saw the answer as a task constraint on the resumed turn.
    assert policy.contexts[-1].task_constraints == ("color: prefers blue",)
    assert policy.contexts[-1].awaiting_user is False


def test_resumed_task_state_carries_the_answer_forward() -> None:
    """Resuming clears the pending question but keeps the constraints already in force."""
    state = TaskState(request_text="not red").with_constraint(constraint(value="red"))
    suspended = state.suspended("Which colour instead?")
    assert suspended.is_waiting_for_user is True
    assert suspended.pending_question == "Which colour instead?"
    assert suspended.clarification_count == 1

    resumed = suspended.resumed()
    assert resumed.is_waiting_for_user is False
    # Suppression does not delete: the earlier constraint survives the suspension.
    assert [c.value for c in resumed.constraints] == ["red"]


def test_clarification_is_not_asked_twice_in_one_run() -> None:
    """Once a run has asked, the action is withdrawn so the loop cannot interrogate."""
    harness = build_control_harness(policy=ClarifyingPolicy(), limits=LoopLimits(max_steps=6))
    result = harness.controller.run("not red", HISTORY)
    # The run suspended at the first opportunity rather than asking repeatedly.
    assert result.control.step_count == 1
    assert result.trajectory.actions() == ("ask_clarification",)


def test_clarification_arguments_are_bounded() -> None:
    """A question cannot become an essay, and options are bounded."""
    with pytest.raises(Exception):
        AskClarificationArguments(question="")
    with pytest.raises(Exception):
        AskClarificationArguments(question="x" * (ASK_QUESTION_MAX_LENGTH + 1))
    with pytest.raises(Exception):
        AskClarificationArguments(question="ok", options=tuple(f"o{i}" for i in range(20)))


def test_clarification_does_not_touch_the_candidate_set() -> None:
    """Asking a question cannot add, drop or alter a candidate."""
    harness = build_control_harness(policy=ClarifyingPolicy(), limits=LoopLimits(max_steps=4))
    result = harness.controller.run("not red", HISTORY)
    assert "tool_result" not in result.state
    assert "reranking" not in result.state


# =========================================================================== #
# C. Memory write proposals
# =========================================================================== #


def test_an_explicit_long_term_proposal_is_approved() -> None:
    """What the user actually said, for the long term, may be committed."""
    validator = MemoryProposalValidator()
    verdict = validator.validate(proposal(), turn_text="please never red again")
    assert verdict.verdict is ProposalVerdict.APPROVED
    assert verdict.may_commit is True


def test_a_task_scoped_proposal_is_soft_only_and_never_persisted() -> None:
    """A task-scoped instruction applies now and is never written to memory."""
    validator = MemoryProposalValidator()
    verdict = validator.validate(
        proposal(scope=PreferenceScope.TASK), turn_text="please never red again"
    )
    assert verdict.verdict is ProposalVerdict.SOFT_ONLY
    assert verdict.may_commit is False
    assert verdict.code == "ephemeral_scope_not_persisted"


def test_an_inferred_hypothesis_can_never_be_committed() -> None:
    """The central rule: an inference is evidence, not truth."""
    validator = MemoryProposalValidator()
    verdict = validator.validate(
        proposal(lifecycle=PreferenceLifecycle.INFERRED_HYPOTHESIS),
        turn_text="please never red again",
    )
    assert verdict.verdict is ProposalVerdict.SOFT_ONLY
    assert verdict.may_commit is False
    assert verdict.may_use_as_evidence is True
    assert verdict.code == "inferred_signal_is_soft_evidence"
    assert approved_candidates((verdict,)) == ()


def test_an_ungrounded_span_is_rejected() -> None:
    """A durable commit must be supported by the user's own words."""
    validator = MemoryProposalValidator()
    missing = validator.validate(
        proposal(source_span=""), turn_text="please never red again"
    )
    assert missing.verdict is ProposalVerdict.REJECTED
    assert missing.code == "ungrounded_span"

    invented = validator.validate(
        proposal(source_span="I demand purple sandals"),
        turn_text="please never red again",
    )
    assert invented.verdict is ProposalVerdict.REJECTED
    assert invented.code == "span_not_in_turn"


def test_a_negative_preference_is_not_a_memory_deletion() -> None:
    """"I don't like red" adds an avoidance; it does not retract anything."""
    validator = MemoryProposalValidator()

    avoidance = proposal(
        operation=MemoryWriteOperation.ADD,
        value="red",
        polarity=PreferencePolarity.AVOID,
    )
    assert validator.validate(avoidance, turn_text="never red again").verdict is (
        ProposalVerdict.APPROVED
    )

    # The same avoidance expressed *as* a removal is refused, because it would delete
    # rather than record.
    collapsed = proposal(
        operation=MemoryWriteOperation.REMOVE,
        value="",
        polarity=PreferencePolarity.AVOID,
    )
    verdict = validator.validate(collapsed, turn_text="never red again")
    assert verdict.verdict is ProposalVerdict.REJECTED
    assert verdict.code == "avoidance_expressed_as_removal"


def test_a_genuine_retraction_is_approved() -> None:
    """An explicit forget-this statement is a removal, and is legal."""
    validator = MemoryProposalValidator()
    verdict = validator.validate(
        proposal(
            operation=MemoryWriteOperation.REMOVE,
            value="",
            polarity=PreferencePolarity.PREFER,
            source_span="forget my colour preferences",
        ),
        turn_text="forget my colour preferences please",
    )
    assert verdict.verdict is ProposalVerdict.APPROVED


def test_a_removal_must_not_carry_a_value() -> None:
    """A retraction directive stores no value; carrying one is refused."""
    validator = MemoryProposalValidator()
    verdict = validator.validate(
        proposal(operation=MemoryWriteOperation.REMOVE, value="red"),
        turn_text="never red again",
    )
    assert verdict.verdict is ProposalVerdict.REJECTED
    assert verdict.code == "removal_carries_value"


def test_an_addition_must_carry_a_value() -> None:
    """An addition with nothing to store is refused."""
    validator = MemoryProposalValidator()
    verdict = validator.validate(
        proposal(operation=MemoryWriteOperation.ADD, value="  "),
        turn_text="never red again",
    )
    assert verdict.verdict is ProposalVerdict.REJECTED
    assert verdict.code == "missing_value"


def test_duplicate_proposals_within_a_turn_collapse() -> None:
    """One turn stating the same fact twice is one proposal."""
    validator = MemoryProposalValidator()
    validations = validator.validate_all(
        (proposal(), proposal()), turn_text="never red again"
    )
    assert len(validations) == 1


def test_the_accepted_extractor_feeds_the_proposal_pipeline_unchanged() -> None:
    """The migration path: the accepted rule-based extractor becomes proposals."""
    extraction = RuleBasedPreferenceExtractor().extract("I don't like red")
    proposals = proposals_from_extraction(
        extraction, turn_id="t1", interpreter="rule_based_preference_extractor"
    )
    validations = MemoryProposalValidator().validate_all(
        proposals, turn_text="I don't like red"
    )
    assert proposals, "the accepted extractor should find the avoidance"
    assert all(v.verdict is ProposalVerdict.APPROVED for v in validations)

    candidates = approved_candidates(validations)
    assert [(c.kind.value, c.polarity.value, c.value) for c in candidates] == [
        ("color", "avoid", "red")
    ]


def test_only_approved_proposals_become_committable_candidates() -> None:
    """Soft-only and rejected proposals produce no candidate at all."""
    validator = MemoryProposalValidator()
    mixed = (
        proposal(),  # approved
        proposal(
            value="blue",
            polarity=PreferencePolarity.PREFER,
            source_span="never red again",  # still grounded in the turn text
        ),
        proposal(
            kind=PreferenceKind.BRAND,
            value="Acme",
            lifecycle=PreferenceLifecycle.INFERRED_HYPOTHESIS,
        ),
        proposal(kind=PreferenceKind.FEATURE, value="waterproof", source_span="not in turn"),
    )
    validations = validator.validate_all(mixed, turn_text="never red again")
    approved = approved_candidates(validations)
    values = {c.value for c in approved}
    assert values == {"red", "blue"}
    assert "Acme" not in values, "an inference must not become a durable preference"
    assert "waterproof" not in values, "an ungrounded span must not become a preference"


def test_proposals_carry_no_store_handle_or_user_key() -> None:
    """A proposal cannot reach the store, and does not know whose memory it is."""
    fields = set(MemoryWriteProposal.model_fields)
    for forbidden in ("user_key", "store", "database", "connection", "memory_id"):
        assert forbidden not in fields
    assert "scope" in fields and "lifecycle" in fields and "source_span" in fields


def test_validator_is_deterministic() -> None:
    """The same proposal and turn always produce the same verdict."""
    validator = MemoryProposalValidator()
    first = validator.validate(proposal(), turn_text="never red again")
    second = validator.validate(proposal(), turn_text="never red again")
    assert first.model_dump() == second.model_dump()


# =========================================================================== #
# D. Preserved behaviour
# =========================================================================== #


def test_stage_1_policy_menu_still_offers_the_fast_path() -> None:
    """Clarification is offered alongside the Stage 1 actions, not instead of them."""
    harness = build_control_harness(policy=AlwaysRecommend(), limits=LoopLimits(max_steps=5))
    result = harness.controller.run("recommend gear", HISTORY)
    assert result.status is RunStatus.FINISHED
    assert harness.engine.call_count == 1
    assert result.trajectory.actions() == ("recommend_from_history", "finish")


def test_a_policy_that_never_clarifies_is_unaffected() -> None:
    """The clarification node exists but is never entered without a proposal."""
    harness = build_control_harness(policy=AlwaysRecommend(), limits=LoopLimits(max_steps=5))
    result = harness.controller.run("recommend gear", HISTORY)
    steps = [step.action_proposal.get("action") for step in result.trajectory.steps]
    assert "ask_clarification" not in steps


def test_both_drivers_agree_on_suspension() -> None:
    """The graph driver and the direct driver suspend identically."""
    outcomes = []
    for driver in ("graph", "direct"):
        harness = build_control_harness(
            policy=ClarifyingPolicy(), limits=LoopLimits(max_steps=4), driver=driver
        )
        result = harness.controller.run("not red", HISTORY)
        outcomes.append(
            (
                result.status.value,
                result.control.termination_reason.value,
                result.final_response,
                harness.engine.call_count,
            )
        )
    assert outcomes[0] == outcomes[1]

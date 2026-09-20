"""The bounded agent loop's engine and its two drivers (AgentRec-X 2.0-alpha).

> AgentPolicy controls **choices**.  LoopController controls the **loop**.

The loop is one canonical cycle::

    policy -> validate_action -> dispatch -> execute -> verify -> observe
           -> update_state -> check_limits -> policy

It can be driven two ways, and both run the *same* engine:

* :class:`LoopController` - the *driver*.  With the default ``driver="graph"`` it compiles
  the cycle as a LangGraph state graph with a real back-edge
  (``update_state -> policy``); with ``driver="direct"`` it runs the identical steps in
  Python.  The direct driver exists so the control semantics can be tested at step
  granularity without a framework in the way.
* :class:`_LoopEngine` - the *step semantics*.  Exactly one implementation of "what
  happens on this step", called by both drivers.  There is no second copy of the control
  logic anywhere in this package.

What the controller owns: run lifecycle, step/tool-call/retry counting,
``available_actions`` computation, validation/execution/verification/observation dispatch,
budget and progress checks, deterministic termination, completion checks, and trajectory
recording.

What it never owns: recommendation scoring, semantic recommendation reasoning, SASRec
inference, catalogue truth, preference interpretation, memory commit, candidate identity or
rendering.  Those stay behind the accepted components, reached only through
:class:`~recommendation.control.capability.RecommendFromHistoryCapability` and the shared
accepted stages in :mod:`recommendation.agent.pipeline`.
"""

from __future__ import annotations

import uuid
from typing import Any

from recommendation.agent.decision import AgentDecision
from recommendation.agent.pipeline import (
    ROUTE_DIRECT,
    finalize_stage,
    persist_memory_stage,
)
from recommendation.agent.rendering import NO_CANDIDATES_TEXT
from recommendation.agent.state import AgentGraphState, AgentInput, TrustedHistory
from recommendation.tools.schemas import RecommendationToolResult, ToolRecommendation

from .capability import RecommendFromHistoryCapability
from .completion import CompletionGuard
from .context import CandidateState, PolicyContext, project_constraints, project_intent
from langgraph.errors import GraphRecursionError

from .schemas import (
    CANDIDATE_ACTIONS,
    READ_ONLY_ACTIONS,
    ActionKind,
    ActionProposal,
    AgentPolicy,
    ClarificationObservation,
    ControlState,
    LoopLimits,
    PolicyActionError,
    RecommendationDomainResult,
    RecommendationObservation,
    RunStatus,
    StateChange,
    TerminationReason,
    ValidatedAction,
    VerificationResult,
)
from .task_state import TaskState
from .trajectory import TrajectoryRecorder
from .validation import ActionValidator
from .verification import ObservationAdapter, ResultVerifier

__all__ = [
    "LOOP_CONTROLLER_VERSION",
    "LoopController",
    "LoopResult",
    "StepOutcome",
    "build_run_id",
]

#: Version of the loop contract.
LOOP_CONTROLLER_VERSION = 1


def build_run_id() -> str:
    """Return a controller-generated run identity.

    Run identities are controller-owned: a policy has no field for one, so it can neither
    choose nor forge it.  ``uuid4`` is used only for identity, never for a control
    decision - the loop's decisions are fully deterministic.
    """
    return uuid.uuid4().hex


class StepOutcome:
    """The result of one engine step: what happened, and whether to keep going.

    ``continue_loop`` is the controller's own decision.  A step that leaves it true
    returns control to the policy; a step that sets it false carries a terminal status and
    reason, so the loop can only end in a state the controller chose.
    """

    __slots__ = ("continue_loop", "detail", "initialized", "reason", "status")

    def __init__(
        self,
        *,
        continue_loop: bool = True,
        status: RunStatus | None = None,
        reason: TerminationReason | None = None,
        detail: str | None = None,
        initialized: bool = False,
    ) -> None:
        self.continue_loop = continue_loop
        self.status = status
        self.reason = reason
        self.detail = detail
        self.initialized = initialized

    @classmethod
    def proceed(cls) -> StepOutcome:
        """Return the "control returns to the policy" outcome."""
        return cls(continue_loop=True)

    @classmethod
    def initialized_outcome(cls) -> StepOutcome:
        """Return the initialize outcome."""
        return cls(continue_loop=True, initialized=True)

    @classmethod
    def stop(
        cls,
        status: RunStatus,
        reason: TerminationReason,
        detail: str | None = None,
    ) -> StepOutcome:
        """Return a deterministic termination outcome."""
        return cls(continue_loop=False, status=status, reason=reason, detail=detail)


class LoopResult:
    """The outcome of one loop run: trusted state, control state and trajectory."""

    def __init__(
        self,
        *,
        state: AgentGraphState,
        control: ControlState,
        trajectory: TrajectoryRecorder,
    ) -> None:
        self.state = state
        self.control = control
        self.trajectory = trajectory

    @property
    def final_response(self) -> str:
        """The rendered response text (empty when the run never finalized)."""
        return str(self.state.get("final_response", ""))

    @property
    def route(self) -> str:
        """The reported route, compatible with the accepted DAG's state shape."""
        return str(self.state.get("route", ROUTE_DIRECT))

    @property
    def status(self) -> RunStatus:
        """Terminal status of the run."""
        return self.control.status

    @property
    def steps(self) -> int:
        """How many steps the controller executed."""
        return self.control.step_count

    @property
    def succeeded(self) -> bool:
        """True only for a run that finished through an accepted completion."""
        return (
            self.control.status is RunStatus.FINISHED
            and self.control.termination_reason is TerminationReason.COMPLETED
        )


# --------------------------------------------------------------------------- #
# The engine: exactly one implementation of the step semantics
# --------------------------------------------------------------------------- #


class _LoopEngine:
    """Controller-owned mutable state plus one method per loop phase.

    This class is the *only* place the step semantics live.  Both drivers call it, so a
    behaviour difference between the graph driver and the direct driver is a topology bug,
    never a semantics bug.
    """

    def __init__(
        self,
        *,
        policy: AgentPolicy,
        capability: RecommendFromHistoryCapability,
        validator: ActionValidator,
        verifier: ResultVerifier,
        completion_guard: CompletionGuard,
        observation_adapter: ObservationAdapter,
        memory_service: Any,
        user_key: str | None,
        limits: LoopLimits,
        user_message: str,
        trusted_history: TrustedHistory,
        turn_id: str | None,
        run_id: str,
        task_state: TaskState | None = None,
        reasoning_executor: Any = None,
        candidate_plane: Any = None,
    ) -> None:
        self.policy = policy
        self.capability = capability
        self.validator = validator
        self.verifier = verifier
        self.completion_guard = completion_guard
        self.observation_adapter = observation_adapter
        self.memory_service = memory_service
        self.user_key = user_key
        self.limits = limits
        self.turn_id = turn_id
        self.run_id = run_id

        self.state: AgentGraphState = AgentGraphState(
            user_message=user_message,
            trusted_user_history=tuple(trusted_history),
        )
        if turn_id:
            self.state["turn_id"] = turn_id

        self.control = ControlState(run_id=run_id, turn_id=turn_id, limits=limits)
        self.trajectory = TrajectoryRecorder(run_id=run_id, turn_id=turn_id)

        # Per-step scratch, deliberately NOT part of the graph state: a policy has no field
        # for any of it.
        self.proposal: ActionProposal | None = None
        #: The accepted decision object the policy chose, when the policy is decision-model
        #: backed.  Recorded so finalization can branch on **what the policy actually
        #: decided** - a direct turn must render the decision's own reply text, exactly as
        #: the accepted DAG does - instead of inferring a route from what happened to run.
        self.decision: AgentDecision | None = None
        self.validation: VerificationResult | None = None
        self.validated: ValidatedAction | None = None
        self.domain_result: RecommendationDomainResult | None = None
        self.verification: VerificationResult | None = None
        self.observation: RecommendationObservation | None = None
        self.completion: VerificationResult | None = None
        self.last_action_id = f"init:{run_id}"
        self.last_proposal_rejected = False
        #: Diagnostics reported by the policy for the current step (empty for a deterministic
        #: policy).  Safe, prompt-free and payload-free by contract.
        self.policy_metadata: dict[str, Any] = {}

        #: Stage 4 read-only reasoning executor, or ``None`` when the deployment configures
        #: no catalogue reasoning.  Validation already refused a reasoning action that was
        #: not offered; the executor is what makes the offered ones executable.
        self.reasoning = reasoning_executor
        #: Stage 2 multi-source candidate plane, or ``None`` for the Stage 1 single-source
        #: fast path.  When configured, the additional trusted sources become actions the
        #: controller offers the policy.
        self.candidate_plane = candidate_plane

        self.last_observation: Any = None
        self.last_verification: VerificationResult | None = None
        self.produced_recommendation = False
        self.candidates_grounded = False
        self.initialized = False

        #: Stage 3 task state: what is true of **this task**.  Mutable during the
        #: interaction and never persisted.  Seeded from the application-supplied task
        #: context so a resumed run continues the same task.
        self.task_state: TaskState = task_state or TaskState()
        #: The clarification observation produced by an ``ASK_CLARIFICATION`` action, kept so
        #: finalization can render the question.
        self.clarification: ClarificationObservation | None = None

        #: Step index whose validation phase has already run.  ``-1`` means none.
        #:
        #: The graph driver needs this because LangGraph resolves a node's outgoing branch
        #: from the snapshot taken *before* that node's body executes.  A phase that has
        #: already decided the current step must therefore report completion, while a phase
        #: that finds itself entered without work must report that it made no progress -
        #: otherwise the back-edge can revisit a step forever without the controller's step
        #: budget incrementing, which is a spin rather than a bounded loop.
        self.validated_step = -1

    # -- initialize -------------------------------------------------------- #

    def initialize(self) -> StepOutcome:
        """Load the turn's read-only preference snapshot, once, before any decision.

        Turn ordering is the accepted Milestone 9 ordering: the snapshot is read before
        the decision and never refreshed, so a turn cannot observe its own write and this
        turn's candidates cannot have been influenced by this turn's statement.
        """
        if self.initialized:  # pragma: no cover - defensive
            return StepOutcome.proceed()
        if self.memory_service is not None and self.user_key is not None:
            self.state["preference_snapshot"] = (
                self.memory_service.get_active_preferences(self.user_key)
            )
        self.initialized = True
        return StepOutcome.initialized_outcome()

    # -- check_limits ------------------------------------------------------ #

    def check_limits(self) -> StepOutcome:
        """Enforce the step budget **before** asking the policy anything.

        Exhaustion terminates deterministically.  The controller does not ask the policy
        whether it would like to continue, because the budget is not the policy's to
        negotiate.
        """
        if self.control.step_count >= self.control.limits.max_steps:
            self.trajectory.record(
                step_index=self.control.step_count,
                action_id=self.last_action_id,
                policy_context_summary={},
                note=f"step budget exhausted (max_steps={self.control.limits.max_steps})",
                policy_metadata=self.policy_metadata,
            )
            return self._terminate(
                RunStatus.ABORTED,
                TerminationReason.MAX_STEPS,
                f"max_steps={self.control.limits.max_steps} reached",
            )
        return StepOutcome.proceed()

    # -- policy ------------------------------------------------------------ #

    def build_context(self) -> PolicyContext:
        """Build the controlled projection the policy is allowed to see."""
        snapshot = self.state.get("preference_snapshot")
        active_count = 0
        if snapshot is not None:
            active_count = len(getattr(snapshot, "active_entries", ()) or ())

        constraints, hard_count, inferred_count = project_constraints(self.task_state)

        return PolicyContext(
            user_request=str(self.state.get("user_message", "")),
            available_actions=self.available_actions(),
            has_trusted_history=bool(self.state.get("trusted_user_history")),
            active_preference_count=active_count,
            candidate_state=self.candidate_state(),
            last_observation=self.last_observation,
            remaining_steps=self.control.steps_remaining,
            remaining_tool_calls=self.control.tool_calls_remaining,
            step_index=self.control.step_count,
            run_status=self.control.status,
            last_proposal_rejected=self.last_proposal_rejected,
            task_intent=project_intent(self.task_state),
            grounded_parent_asins=self.grounded_identities(),
            task_constraints=constraints,
            hard_constraint_count=hard_count,
            inferred_constraint_count=inferred_count,
            awaiting_user=self.task_state.is_waiting_for_user,
        )

    def available_actions(self) -> tuple[ActionKind, ...]:
        """Compute the action space the **system** currently offers.

        Stage 1 keeps both actions available at every step and lets the policy decide,
        because the policy's context tells it whether a grounded candidate set already
        exists.  The controller's authority is the *budget*, not the menu: it withdraws
        ``RECOMMEND_FROM_HISTORY`` only when executing it is impossible, and it always
        offers ``FINISH`` so every run can end.
        """
        actions: list[ActionKind] = []
        if self.control.tool_calls_remaining > 0:
            actions.append(ActionKind.RECOMMEND_FROM_HISTORY)
            # Stage 2: the extra candidate sources, offered only when the deployment
            # configures a plane that can execute them.  A source this process does not run
            # is absent from the menu, never offered and then substituted.
            if self.candidate_plane is not None:
                actions.extend(self._plane_candidate_actions())
        # Clarification is offered only while the task has not already asked something.
        # A suspended task is resumed by the *next* turn, so asking twice in one run would
        # make the loop an interrogation rather than a control loop.
        if not self.task_state.is_waiting_for_user and self.clarification is None:
            actions.append(ActionKind.ASK_CLARIFICATION)
        # Stage 4: read-only reasoning is offered only when an executor is configured, and
        # only while there is a candidate to reason about.  Offering a facts question with
        # nothing to ask about would invite a policy to burn the step budget.
        if self.reasoning is not None and self.candidates_grounded:
            actions.extend(self.reasoning.available_actions())
        actions.append(ActionKind.FINISH)
        return tuple(actions)

    def _plane_candidate_actions(self) -> tuple[ActionKind, ...]:
        """The candidate actions the configured plane can actually execute.

        Derived from the plane's registered sources rather than from the action enum, so the
        menu never advertises a source the deployment does not run.
        """
        from .arguments import CandidateSource

        offered: list[ActionKind] = []
        if self.candidate_plane.has_source(CandidateSource.CATALOG_SEARCH):
            offered.append(ActionKind.SEARCH_CATALOG)
        if self.candidate_plane.has_source(CandidateSource.SIMILAR_ITEM):
            offered.append(ActionKind.FIND_SIMILAR)
            offered.append(ActionKind.SELECT_SOURCE)
        return tuple(offered)

    def grounded_identities(self) -> tuple[str, ...]:
        """Return the identities of the run's grounded candidates, in order.

        Read from the **verified** candidate set the run holds, never from a request and never
        from the ledger's ungrounded audit entries.  A policy uses these to *refer* to products
        in read-only reasoning actions; it cannot add one.
        """
        tool_result = self.state.get("tool_result")
        if tool_result is None:
            return ()
        return tuple(
            item.parent_asin for item in getattr(tool_result, "recommendations", ()) or ()
        )

    def candidate_state(self) -> CandidateState:
        """Project the trusted candidate state into the policy-visible summary.

        Counts and an opaque reference only: no ``parent_asin``, no ``item_id``, no score.
        """
        tool_result = self.state.get("tool_result")
        if tool_result is None:
            return CandidateState()
        count = len(getattr(tool_result, "recommendations", ()) or ())
        return CandidateState(
            grounded=bool(self.candidates_grounded),
            candidate_count=count,
            candidate_set_ref=f"verified:{count}",
            verification_status="verified" if self.candidates_grounded else "unverified",
        )

    def choose(self) -> StepOutcome:
        """Ask the policy for exactly one action proposal.

        A policy that raises :class:`PolicyActionError` is reporting that it has no legal
        action; that is a deterministic abort.  Any other exception is a policy failure,
        which is a failed run - never a reason to keep looping.

        The per-step scratch is *not* cleared here.  Clearing it here meant the freshly
        chosen proposal could be discarded by a later phase that had not run yet, and the
        phases downstream - which receive the branch LangGraph resolved from a pre-node
        snapshot - could then find themselves with nothing to act on.  The scratch belongs
        to the phase that produces it, and :meth:`validate_action` owns it.
        """
        step_index = self.control.step_count
        context = self.build_context()
        try:
            self.proposal = self.policy.choose(context)
            self.decision = getattr(self.policy, "last_decision", None)
            # A policy may report safe diagnostics about how it decided.  Recorded on the step
            # that follows, so a refusal caused by a bad proposal still shows why.
            self.policy_metadata = self._policy_metadata()
        except PolicyActionError as exc:
            self.policy_metadata = self._policy_metadata()
            self.trajectory.record(
                step_index=step_index,
                action_id=self.last_action_id,
                policy_context_summary=context.summary(),
                note=f"policy raised PolicyActionError: {exc}",
                policy_metadata=self.policy_metadata,
            )
            return self._terminate(
                RunStatus.ABORTED, TerminationReason.NO_AVAILABLE_ACTION, str(exc)
            )
        except Exception as exc:  # noqa: BLE001 - a policy failure is normalised
            self.policy_metadata = self._policy_metadata()
            self.trajectory.record(
                step_index=step_index,
                action_id=self.last_action_id,
                policy_context_summary=context.summary(),
                note=f"policy failed: {type(exc).__name__}",
                policy_metadata=self.policy_metadata,
            )
            return self._terminate(
                RunStatus.FAILED,
                TerminationReason.EXECUTION_FAILED,
                f"policy failed: {type(exc).__name__}",
            )
        return StepOutcome.proceed()

    def _policy_metadata(self) -> dict[str, Any]:
        """The policy's self-reported diagnostics, or its declared identity as a fallback.

        A model-driven policy reports its own metadata (name, attempts, model calls).  Every other
        policy is anonymous unless the loop names it, and an unnamed policy makes a run's
        provenance unverifiable: an ablation arm could silently fall back to a different policy
        while still being filed under the intended variant.  So the declared ``name`` is recorded
        when no richer metadata exists - identity only, never a decision input.
        """
        reported = dict(getattr(self.policy, "last_metadata", {}) or {})
        if reported:
            return reported
        declared = getattr(self.policy, "name", None) or type(self.policy).__name__
        return {"policy": str(declared)}

    # -- validate_action --------------------------------------------------- #

    def validate_action(self) -> StepOutcome:
        """Turn the untrusted proposal into a controller-stamped action, or refuse.

        This phase owns the per-step action scratch: it clears the previous step's result
        itself and then decides the new one.  Nothing downstream depends on a *previous*
        phase having cleared it, which is what makes each phase safe to enter on its own.

        A run the **policy phase already stopped** is left untouched.  A policy that raised
        ``PolicyActionError`` terminates the run with ``NO_AVAILABLE_ACTION``, but a LangGraph
        conditional edge is resolved from the pre-node snapshot, so this node can still be
        reached on that terminal run.  Clearing the scratch here overwrote the controller's own
        verdict and reported ``INVALID_ACTION`` - a misleading reason, since the policy's
        failure was a legitimate abort rather than a malformed proposal.  Nothing has been
        proposed in that case, so there is nothing to validate.
        """
        if self.control.is_terminal:
            return StepOutcome.proceed()
        self.validation = None
        self.validated = None
        self.completion = None
        self.validated_step = -1
        if self.proposal is None:  # noqa: SIM108 - explicit: nothing was proposed
            self.trajectory.record(
                step_index=self.control.step_count,
                action_id=self.last_action_id,
                policy_context_summary=self.build_context().summary(),
                note="no proposal was produced for this step",
                policy_metadata=self.policy_metadata,
            )
            return self._terminate(
                RunStatus.FAILED,
                TerminationReason.INVALID_ACTION,
                "the step reached validation with no action proposal",
            )
        step_index = self.control.step_count
        context = self.build_context()
        validated, validation = self.validator.validate(
            self.proposal,
            run_id=self.run_id,
            turn_id=self.turn_id,
            step_index=step_index,
            available_actions=context.available_actions,
        )
        self.validation = validation
        self.validated = validated

        if validated is None:
            # A policy proposing something the system did not offer is a protocol
            # violation; retrying would reproduce it and burn the loop budget.
            self.trajectory.record(
                step_index=step_index,
                action_id=self.last_action_id,
                policy_context_summary=context.summary(),
                action_proposal=self.proposal,
                validation_result=validation,
                note="proposal refused; terminating deterministically",
                policy_metadata=self.policy_metadata,
            )
            return self._terminate(
                RunStatus.FAILED,
                TerminationReason.INVALID_ACTION,
                f"{validation.code}: {validation.detail}",
            )
        self.last_action_id = validated.action_id
        self.validated_step = step_index
        return StepOutcome.proceed()

    # -- dispatch ---------------------------------------------------------- #

    def _is_plane_step(self) -> bool:
        """True when the current action was executed by the multi-source candidate plane."""
        return self.validated is not None and self.validated.action in (
            ActionKind.SEARCH_CATALOG,
            ActionKind.FIND_SIMILAR,
            ActionKind.SELECT_SOURCE,
        )

    def _self_verified_step(self) -> bool:
        """True when this step's phase already produced its own verification and observation.

        Three phases do: **reasoning** (grounded facts, no raw payload), **candidate-plane
        execution** (the plane grounds and records, so there is nothing left to re-verify) and
        **clarification** (nothing executes).  The generic verify/observe phases must skip
        them, or they would re-verify a result that does not have the shape
        :class:`~recommendation.control.verification.ResultVerifier` expects - which is exactly
        how the candidate plane first broke: the plane's result carries no history-Tool payload,
        the verifier correctly refused it, and a successful retrieval was reported as a failure.
        """
        if self.validated is None:
            return False
        if self.validated.action in READ_ONLY_ACTIONS:
            return True
        return self.validated.action in (
            ActionKind.SEARCH_CATALOG,
            ActionKind.FIND_SIMILAR,
            ActionKind.SELECT_SOURCE,
        )

    def has_current_action(self) -> bool:
        """True when a validated action exists **for the step in progress**.

        A LangGraph conditional edge is evaluated for every declared branch, so an
        action-dependent node can be entered on a path where validation has not produced an
        action for this step.  Every such phase checks this first.

        The step comparison matters: it is not enough for *some* action to exist, it must
        belong to the step being processed.  Without the comparison a phase could act on
        the previous step's action, which is exactly the class of stale-state bug the
        single-owner-per-phase rule exists to prevent.
        """
        return (
            self.validated is not None
            and self.validated_step == self.control.step_count
        )

    def dispatch_target(self) -> str:
        """Return which branch the validated action takes: ``execute`` or ``complete``.

        Returns ``"execute"`` when no action was validated.  A LangGraph conditional edge is
        evaluated for its node even on runs where that node is never entered, so this must
        answer without asserting; when validation refused the proposal the run is already
        terminal and the graph routes to ``END`` before either branch runs.
        """
        if self.validated is None:
            return "execute"
        if self.validated.action is ActionKind.FINISH:
            return "complete"
        if self.validated.action is ActionKind.ASK_CLARIFICATION:
            return "clarify"
        if self.validated.action in READ_ONLY_ACTIONS:
            return "reason"
        return "execute"

    # -- execute ----------------------------------------------------------- #

    def check_tool_budget(self) -> StepOutcome:
        """Enforce the tool-call budget before executing anything.

        Terminating here rather than executing and then discovering the budget is gone
        keeps the recorded reason truthful: the run stopped for the budget, not for a
        result it did not actually produce.
        """
        if not self.has_current_action():
            return StepOutcome.proceed()
        if self.control.tool_calls_remaining > 0:
            return StepOutcome.proceed()
        self.trajectory.record(
            step_index=self.control.step_count,
            action_id=self.validated.action_id,
            policy_context_summary=self.build_context().summary(),
            action_proposal=self.proposal,
            validation_result=self.validation,
            validated_action=self.validated,
            note=(
                "tool-call budget exhausted; terminating rather than executing "
                f"(max_tool_calls={self.control.limits.max_tool_calls})"
            ),
            policy_metadata=self.policy_metadata,
        )
        return self._terminate(
            RunStatus.ABORTED,
            TerminationReason.MAX_TOOL_CALLS,
            f"max_tool_calls={self.control.limits.max_tool_calls} reached",
        )

    def execute(self) -> StepOutcome:
        """Run the accepted recommendation pipeline behind the capability boundary.

        The capability receives trusted history through a *reader*, never through the
        action, so a policy has no channel through which to supply or forge history.  A
        capability failure is normalised into a failed observation - the loop does not
        crash, and the completion guard will refuse to certify the run.
        """
        if not self.has_current_action():
            return StepOutcome.proceed()
        # Stage 2: an action that consults a *non-history* candidate source goes through the
        # plane, which grounds every identity before the ledger records it.  The history
        # action keeps using the accepted capability so the Stage 1 path is untouched.
        if self.candidate_plane is not None and self.validated.action in (
            ActionKind.SEARCH_CATALOG,
            ActionKind.FIND_SIMILAR,
            ActionKind.SELECT_SOURCE,
        ):
            return self._execute_via_plane()
        try:
            self.domain_result = self.capability.execute(
                self.validated,
                read_trusted_history=lambda: tuple(
                    self.state.get("trusted_user_history", ())
                ),
                user_message=str(self.state.get("user_message", "")),
                preference_snapshot=self.state.get("preference_snapshot"),
                tool_call_count=self.control.tool_call_count,
            )
        except Exception as exc:  # noqa: BLE001 - normalised into an observation
            self.domain_result = None
            self.verification = VerificationResult(
                verified=False,
                code="execution_failed",
                detail=f"capability raised {type(exc).__name__}",
                checks=("execution",),
            )
            self.observation = self.observation_adapter.failed_observation(
                action_id=self.validated.action_id,
                step_index=self.control.step_count,
                action=self.validated.action,
                code=_error_code(exc),
                requested_k=self.validated.k,
            )
        return StepOutcome.proceed()

    def _execute_via_plane(self) -> StepOutcome:
        """Execute one non-history candidate action through the multi-source plane.

        The plane grounds identities and records provenance; this phase then builds the
        minimised observation the policy sees.  Tool-call accounting is unchanged: this is
        still a candidate-producing action and still consumes one call.
        """
        assert self.validated is not None
        try:
            result = self.candidate_plane.execute(
                self.validated,
                read_trusted_history=lambda: tuple(
                    self.state.get("trusted_user_history", ())
                ),
                step_index=self.control.step_count,
            )
        except Exception as exc:  # noqa: BLE001 - normalised into an observation
            self.domain_result = None
            self.verification = VerificationResult(
                verified=False,
                code="candidate_source_failed",
                detail=f"the candidate plane raised {type(exc).__name__}",
                checks=("execution",),
            )
            self.observation = self.observation_adapter.failed_observation(
                action_id=self.validated.action_id,
                step_index=self.control.step_count,
                action=self.validated.action,
                code=_error_code(exc),
                requested_k=self.validated.k,
            )
            return StepOutcome.proceed()

        # Adopt the plane's candidate set into the run's trusted state so the accepted
        # renderer can present it.  The plane returned grounded identities only, so this is
        # the same "verified candidates only" rule the history path follows.
        self.domain_result = result
        self.verification = VerificationResult(
            verified=True,
            code="candidates_grounded",
            detail=None,
            checks=("candidate_identity", "provenance_recorded"),
        )
        source = getattr(result, "source", "") or self.validated.action.value
        self.observation = self.candidate_plane.observe(
            action=self.validated,
            source=_source_for(self.validated.action, result),
            requested_k=result.requested_k,
            returned_k=result.returned_k,
            ungrounded_count=int(getattr(result, "ungrounded_count", 0) or 0),
            step_index=self.control.step_count,
            candidate_set_ref=result.candidate_set_ref,
        )
        self._adopt_plane_candidates(result)
        return StepOutcome.proceed()

    def _adopt_plane_candidates(self, result: Any) -> None:
        """Record a plane's grounded candidate set in the run's trusted state.

        The plane returns candidate identities; the run's state needs a rendering shape.  Only
        grounded identities are adopted, and the provenance stays in the ledger, so the
        renderer can present them while the audit trail remains in one place.
        """
        identities = tuple(getattr(result, "grounded_parent_asins", ()) or ())
        if not identities:
            return
        item_ids = dict(getattr(result, "item_ids", {}) or {})
        recommendations = tuple(
            ToolRecommendation(
                rank=position,
                parent_asin=parent_asin,
                item_id=item_ids.get(parent_asin, position),
                # No score: a lexical or similarity source's score is not a SASRec score, and
                # presenting one as though it were would misrepresent the source.
                score=0.0,
            )
            for position, parent_asin in enumerate(identities, start=1)
        )
        # Record which trusted source produced this candidate set, so the renderer can name it
        # instead of implying the sequential recommender.
        self.state["candidate_source"] = str(getattr(result, "source", "") or "")
        self.state["tool_result"] = RecommendationToolResult(
            recommendations=list(recommendations),
            requested_k=result.requested_k,
            returned_k=len(recommendations),
            history_length=len(tuple(self.state.get("trusted_user_history", ()))),
            effective_history_length=0,
            history_truncated=False,
            eligible_candidates=0,
        )

    # -- verify ------------------------------------------------------------ #

    def verify(self) -> StepOutcome:
        """Verify a raw domain result.  A result that already failed stays failed.

        A reasoning step set its own verdict in :meth:`reason` - there is no raw domain
        payload to verify a second time - so this phase leaves it alone.
        """
        if not self.has_current_action():
            return StepOutcome.proceed()
        if self._self_verified_step():
            return StepOutcome.proceed()
        if self.domain_result is None:
            return StepOutcome.proceed()
        self.verification = self.verifier.verify(
            self.domain_result,
            trusted_history=tuple(self.state.get("trusted_user_history", ())),
            expected_action_id=self.validated.action_id,
        )
        return StepOutcome.proceed()

    # -- observe ----------------------------------------------------------- #

    def observe(self) -> StepOutcome:
        """Adapt the verified (or refused) result into the policy-visible observation.

        A reasoning step already produced its observation in :meth:`reason`; adapting it again
        would either duplicate it or overwrite grounded facts with a refusal.
        """
        if not self.has_current_action():
            return StepOutcome.proceed()
        if self._self_verified_step():
            return StepOutcome.proceed()
        if self.domain_result is None or self.verification is None:
            return StepOutcome.proceed()
        self.observation = self.observation_adapter.adapt(
            self.domain_result,
            self.verification,
            action=self.validated.action,
            step_index=self.control.step_count,
        )
        return StepOutcome.proceed()

    # -- update_state ------------------------------------------------------ #

    def update_state(self) -> StepOutcome:
        """Adopt a verified candidate set, record the step, then return to the policy."""
        if not self.has_current_action():
            return StepOutcome.proceed()
        if self.verification is None:
            return self._terminate(
                RunStatus.FAILED,
                TerminationReason.EXECUTION_FAILED,
                "the step produced no verification verdict",
            )

        self.last_verification = self.verification
        self.last_observation = self.observation

        is_reasoning = (
            self.validated is not None and self.validated.action in READ_ONLY_ACTIONS
        )
        if self.verification.verified and self.domain_result is not None and not is_reasoning:
            self.produced_recommendation = True
            self.candidates_grounded = self.domain_result.returned_k > 0
            # A plane step adopted its candidates at execution time (the plane returns
            # identities, not a history-Tool payload), so adopting again here would overwrite
            # them with ``None`` fields.
            if not self._is_plane_step():
                self._adopt_candidates(self.domain_result)
        elif is_reasoning:
            # Reasoning adds no candidates and removes none: the candidate state is left
            # exactly as it was, which is what keeps "reasoning cannot widen the candidate
            # set" true through the update phase as well as the execution phase.
            pass
        else:
            self.produced_recommendation = False
            self.candidates_grounded = False

        self.trajectory.record(
            step_index=self.validated.step_index,
            action_id=self.validated.action_id,
            policy_context_summary=self.build_context().summary(),
            action_proposal=self.proposal,
            validation_result=self.validation,
            validated_action=self.validated,
            tool_result_ref=(
                None
                if self.domain_result is None
                else (self.domain_result.candidate_set_ref or None)
            ),
            verification_result=self.verification,
            observation=self.observation,
            state_delta=StateChange(
                step_index=self.validated.step_index,
                action=self.validated.action,
                produced_candidates=self.produced_recommendation,
                candidate_count=(
                    self.domain_result.returned_k if self.domain_result is not None else 0
                ),
                run_status=RunStatus.RUNNING,
            ),
            note=(
                None
                if self.verification.verified
                else f"result refused: {self.verification.code}"
            ),
            policy_metadata=self.policy_metadata,
        )
        # Only a *candidate-producing* action consumes a tool call.  Reasoning reads facts
        # about candidates the run already holds, and clarification runs nothing at all; if
        # either were charged to the tool budget, a facts question would silently reduce the
        # run's ability to retrieve, which is what the budget exists to bound.
        self.control = self.control.advanced(
            action=self.validated.action,
            action_id=self.validated.action_id,
            consumed_tool_call=self.validated.action in CANDIDATE_ACTIONS,
        )
        self.last_proposal_rejected = False
        return StepOutcome.proceed()

    # -- reasoning branch -------------------------------------------------- #

    def reason(self) -> StepOutcome:
        """Execute one read-only reasoning action through the trusted executor.

        Reasoning actions are read-only by contract: they read facts about candidates the run
        already holds and can never add one.  This phase therefore runs **no** candidate
        source, touches no ledger and consumes no tool-call budget - a question about facts is
        not a retrieval.

        A reasoning failure is normalised into a failed observation rather than crashing the
        loop, exactly as a capability failure is.
        """
        if not self.has_current_action():  # pragma: no cover - see dispatch_target
            return StepOutcome.proceed()
        if self.reasoning is None:
            # No catalogue reasoning is configured.  The action was offered only if an
            # executor exists, so reaching here means the deployment changed mid-run; refuse
            # rather than silently succeed.
            self.verification = VerificationResult(
                verified=False,
                code="reasoning_unavailable",
                detail="no reasoning executor is configured for this deployment",
                checks=("reasoning_available",),
            )
            self.observation = self.observation_adapter.failed_observation(
                action_id=self.validated.action_id,
                step_index=self.control.step_count,
                action=self.validated.action,
                code="reasoning_unavailable",
            )
            return StepOutcome.proceed()
        try:
            # The reasoning result *is* the observation: unlike a candidate action there is no
            # raw domain payload to minimise, because the reasoner already returns only
            # whitelisted, grounded facts.  Nothing is adopted into the candidate state.
            self.observation = self.reasoning.execute(self.validated)
            self.verification = VerificationResult(
                verified=True,
                code="reasoning_completed",
                detail=None,
                checks=("grounded_facts_only", "read_only"),
            )
        except Exception as exc:  # noqa: BLE001 - normalised into an observation
            self.verification = VerificationResult(
                verified=False,
                code="reasoning_failed",
                detail=f"reasoning raised {type(exc).__name__}",
                checks=("execution",),
            )
            self.observation = self.observation_adapter.failed_observation(
                action_id=self.validated.action_id,
                step_index=self.control.step_count,
                action=self.validated.action,
                code=_error_code(exc),
            )
        return StepOutcome.proceed()

    # -- clarification branch ---------------------------------------------- #

    def finish_clarification(self) -> None:
        """Render a suspended run's question through the accepted finalizer.

        Called by the driver after ``ask_clarification`` has recorded the suspension, so the
        response text and route are produced by the same code path every other run uses.
        """
        self._finalize_response()

    def ask_clarification(self) -> StepOutcome:
        """Suspend the run to ask the user a question.

        Clarification is a **suspension**, not a completion and not a failure.  The run
        records ``WAITING_FOR_USER`` with :attr:`TerminationReason.AWAITING_USER`, the task
        state remembers the question, and the finalizer renders it.  The task is resumable:
        the next turn continues the same task with the user's answer.

        No tool runs, the candidate set is untouched, and nothing is written to persistent
        memory.  A question is the *cheapest* possible action in terms of irreversible
        effects, which is why it is safe to let a policy propose one.
        """
        if not self.has_current_action():  # pragma: no cover - see dispatch_target
            return StepOutcome.proceed()
        arguments = self.validated.arguments
        question = str(getattr(arguments, "question", "") or "").strip()
        if not question:
            return self._terminate(
                RunStatus.FAILED,
                TerminationReason.INVALID_ACTION,
                "ASK_CLARIFICATION carried no question",
            )

        blocks = str(getattr(arguments, "blocks", "candidate_set") or "candidate_set")
        options = tuple(getattr(arguments, "options", ()) or ())
        self.clarification = ClarificationObservation(
            action_id=self.validated.action_id,
            step_index=self.control.step_count,
            action=self.validated.action,
            question=question,
            blocks=blocks,
            options=options,
            verification_note=(
                f"clarification #{self.task_state.clarification_count + 1} for this task"
            ),
        )
        # The task state records the suspension so a resumed run knows what it asked.
        self.task_state = self.task_state.suspended(question)
        self.state["clarification_question"] = question

        self.trajectory.record(
            step_index=self.validated.step_index,
            action_id=self.validated.action_id,
            policy_context_summary=self.build_context().summary(),
            action_proposal=self.proposal,
            validation_result=self.validation,
            validated_action=self.validated,
            verification_result=VerificationResult(
                verified=True,
                code="clarification_requested",
                detail=None,
                checks=("question_present", "not_a_completion"),
            ),
            observation=self.clarification,
            state_delta=StateChange(
                step_index=self.validated.step_index,
                action=ActionKind.ASK_CLARIFICATION,
                produced_candidates=False,
                candidate_count=0,
                run_status=RunStatus.WAITING_FOR_USER,
            ),
            note="run suspended awaiting a user answer",
            policy_metadata=self.policy_metadata,
        )
        self.control = self.control.advanced(
            action=ActionKind.ASK_CLARIFICATION,
            action_id=self.validated.action_id,
            consumed_tool_call=False,
        )
        return self._terminate(
            RunStatus.WAITING_FOR_USER,
            TerminationReason.AWAITING_USER,
            f"awaiting the user's answer to: {blocks}",
        )

    # -- completion branch ------------------------------------------------- #

    def run_completion_guard(self) -> StepOutcome:
        """Check whether the validated FINISH proposal is a legal end to the run."""
        if not self.has_current_action():
            return StepOutcome.proceed()
        self.completion = self.completion_guard.check(
            self.validated,
            state=self.control,
            last_verification=self.last_verification,
            last_observation=self.last_observation,
            produced_recommendation=self.produced_recommendation,
            candidates_grounded=self.candidates_grounded,
            execution_failed=(
                self.last_verification is not None and not self.last_verification.verified
            ),
        )
        return StepOutcome.proceed()

    def completion_target(self) -> str:
        """Return ``finalize``, ``refuse`` or ``abort`` for the guard's verdict.

        The guard always runs before this is consulted, so ``completion`` is set.  The
        defensive branch exists for the same reason as in :meth:`dispatch_target`: a
        conditional edge may be evaluated on a run that never reaches the node.
        """
        if self.completion is None:  # pragma: no cover - edge evaluation on other runs
            return "abort"
        if self.completion.verified:
            return "finalize"
        if self.completion_guard.can_retry(self.completion, state=self.control):
            return "refuse"
        return "abort"

    def refuse_completion(self) -> StepOutcome:
        """Record a retryable refusal and return control to the policy."""
        if not self.has_current_action() or self.completion is None:
            return StepOutcome.proceed()
        self.control = self.control.model_copy(
            update={"retry_count": self.control.retry_count + 1}
        )
        self.trajectory.record(
            step_index=self.validated.step_index,
            action_id=self.validated.action_id,
            policy_context_summary=self.build_context().summary(),
            action_proposal=self.proposal,
            validation_result=self.validation,
            validated_action=self.validated,
            verification_result=self.completion,
            note="completion refused; returning control to the policy",
            policy_metadata=self.policy_metadata,
        )
        self.control = self.control.advanced(
            action=ActionKind.FINISH,
            action_id=self.validated.action_id,
            consumed_tool_call=False,
        )
        self.last_observation = None
        self.last_proposal_rejected = True
        return StepOutcome.proceed()

    def finalize(self) -> StepOutcome:
        """Render with the accepted renderer, then let memory commit; the run ends.

        Rendering is ``recommendation.agent.pipeline.finalize_stage`` - the *same*
        function the accepted DAG uses.  The memory commit is
        ``recommendation.agent.pipeline.persist_memory_stage`` - again the same function,
        after the response exists, writing only the user's own message.
        """
        if not self.has_current_action() or self.completion is None:
            return StepOutcome.proceed()
        self._finalize_response()
        self._commit_memory()
        self.trajectory.record(
            step_index=self.validated.step_index,
            action_id=self.validated.action_id,
            policy_context_summary=self.build_context().summary(),
            action_proposal=self.proposal,
            validation_result=self.validation,
            validated_action=self.validated,
            verification_result=self.completion,
            state_delta=StateChange(
                step_index=self.validated.step_index,
                action=ActionKind.FINISH,
                produced_candidates=self.produced_recommendation,
                # Only a candidate-producing observation reports a candidate count.  A
                # reasoning or clarification observation has no such field, and reading one
                # unconditionally is how a facts question crashed finalization.
                candidate_count=int(getattr(self.observation, "returned_k", 0) or 0),
                memory_write_attempted=self.memory_service is not None,
            ),
            note="completion accepted; run finished",
            policy_metadata=self.policy_metadata,
        )
        self.control = self.control.advanced(
            action=ActionKind.FINISH,
            action_id=self.validated.action_id,
            consumed_tool_call=False,
        )
        # The engine records its own terminal state, exactly as ``abort_completion`` and
        # every other stopping step does, so both drivers read the same ``control`` object
        # after the run and neither has to be handed a stop outcome it might ignore.
        self.control = self.control.terminated(
            RunStatus.FINISHED, TerminationReason.COMPLETED
        )
        return StepOutcome.proceed()

    def abort_completion(self) -> StepOutcome:
        """Terminate for a refusal no further step can fix."""
        if self.completion is None:
            return StepOutcome.proceed()
        self.trajectory.record(
            step_index=self.control.step_count,
            action_id=self.last_action_id,
            policy_context_summary=self.build_context().summary(),
            action_proposal=self.proposal,
            validation_result=self.validation,
            validated_action=self.validated,
            verification_result=self.completion,
            note="completion refused and not retryable; terminating",
            policy_metadata=self.policy_metadata,
        )
        return self._terminate(
            self.completion_guard.terminal_status(self.completion),
            TerminationReason.COMPLETION_REFUSED,
            f"{self.completion.code}: {self.completion.detail}",
        )

    # -- results ----------------------------------------------------------- #

    def _terminate(
        self,
        status: RunStatus,
        reason: TerminationReason,
        detail: str | None = None,
    ) -> StepOutcome:
        """Record a terminal state and return the stopping outcome.

        Recording here - rather than only *returning* an outcome for a driver to interpret -
        is what makes ``control`` the single authoritative record of why a run stopped.
        Every stopping step goes through this method, so the graph topology, the direct
        driver and the completion guard all read the same status.  An earlier version only
        returned the outcome from ``check_limits``, which the direct driver consumed and the
        topology could not see at all - so ``max_steps`` was silently not enforced on the
        graph driver.
        """
        self.control = self.control.terminated(status, reason, detail)
        return StepOutcome.stop(status, reason, detail)

    def result(self, outcome: StepOutcome) -> LoopResult:
        """Build the terminal result from a stopping outcome."""
        status = outcome.status or RunStatus.FAILED
        reason = outcome.reason or TerminationReason.EXECUTION_FAILED
        return LoopResult(
            state=self.state,
            control=self.control.terminated(status, reason, outcome.detail),
            trajectory=self.trajectory,
        )

    def terminal_result(self) -> LoopResult:
        """Build the terminal result from an already-terminal control state."""
        return LoopResult(
            state=self.state,
            control=self.control,
            trajectory=self.trajectory,
        )

    # -- internals --------------------------------------------------------- #

    def _adopt_candidates(self, result: RecommendationDomainResult) -> None:
        """Record a **verified** candidate set in the trusted state.

        Only verified artifacts are adopted, so a refused stage cannot leave
        half-populated state that a later renderer might present as a real answer.
        """
        if result.tool_result is not None:
            self.state["tool_result"] = result.tool_result
        if result.enrichment is not None:
            self.state["enrichment"] = result.enrichment
        if result.preference_evidence is not None:
            self.state["preference_evidence"] = result.preference_evidence
        if result.reranking is not None:
            self.state["reranking"] = result.reranking

    def _finalize_response(self) -> None:
        """Render with the accepted renderer, driven by what the policy decided.

        The accepted ``finalize_stage`` branches on the run's ``AgentDecision`` because that
        is how the DAG records the route.  The loop therefore publishes the decision it
        actually acted on, which keeps the two control planes rendering through one code
        path:

        * the policy proposed RECOMMEND and a verified candidate set exists - render the
          candidates, in the accepted order;
        * the policy proposed FINISH after a recommendation - the run still produced
          candidates, so render them;
        * the policy proposed FINISH without one - the run is a **direct turn**, and its
          text is the decision model's own reply, exactly as the DAG would render it.

        A synthesised ``no candidates`` message is used only for the case the DAG has no
        equivalent of: an action that produced no candidates *and* carried no reply text
        (for example a test policy that only proposes FINISH).
        """
        if self.clarification is not None and not self.produced_recommendation:
            # A suspended run is not a direct turn: the response *is* the question.  It is
            # rendered through the accepted finalizer as a direct response so the two control
            # planes keep one presentation path, and the question text comes from the
            # validated action rather than from a model string.
            self.state["decision"] = AgentDecision(
                action="direct_response", direct_response=self.clarification.question
            )
            self.state.update(finalize_stage(self.state))
            return
        if self.produced_recommendation and self.state.get("tool_result") is not None:
            self.state["decision"] = AgentDecision(action="recommend")
        elif self.decision is not None and self.decision.direct_response is not None:
            self.state["decision"] = self.decision
        else:
            self.state["decision"] = AgentDecision(
                action="direct_response", direct_response=NO_CANDIDATES_TEXT
            )
        self.state.update(finalize_stage(self.state))

    def _commit_memory(self) -> None:
        """Commit this turn's user-authored preferences through the accepted stage."""
        self.state.update(
            persist_memory_stage(
                self.state,
                memory_service=self.memory_service,
                user_key=self.user_key,
            )
        )


# --------------------------------------------------------------------------- #
# The driver
# --------------------------------------------------------------------------- #


class LoopController:
    """Drive the bounded loop for one turn.

    Parameters
    ----------
    policy:
        Any object with ``choose(context) -> ActionProposal``.  Injected, so a later stage
        can supply an LLM policy without changing anything else in this module.
    capability:
        The recommendation capability the loop may invoke.  The loop never reaches the
        Tool, engine, enricher, matcher or reranker directly.
    driver:
        ``"graph"`` (default) compiles the cycle as a LangGraph state graph with the real
        back-edge ``update_state -> policy``.  ``"direct"`` runs the same engine steps in
        Python.  Both produce identical results; the direct driver exists so control
        semantics can be tested at step granularity.
    memory_service, user_key:
        The accepted Milestone 9 read/persist seam.
    limits:
        The deterministic termination budgets.
    """

    def __init__(
        self,
        policy: AgentPolicy,
        capability: RecommendFromHistoryCapability,
        *,
        validator: ActionValidator | None = None,
        verifier: ResultVerifier | None = None,
        completion_guard: CompletionGuard | None = None,
        observation_adapter: ObservationAdapter | None = None,
        memory_service: Any = None,
        user_key: str | None = None,
        limits: LoopLimits | None = None,
        driver: str = "graph",
        reasoning_executor: Any = None,
        candidate_plane: Any = None,
    ) -> None:
        if not callable(getattr(policy, "choose", None)):
            raise PolicyActionError("policy must provide a callable choose(context) method")
        if not isinstance(capability, RecommendFromHistoryCapability):
            raise PolicyActionError(
                f"capability must be a RecommendFromHistoryCapability, got "
                f"{type(capability).__name__}"
            )
        if driver not in ("graph", "direct"):
            raise PolicyActionError(f"driver must be 'graph' or 'direct', got {driver!r}")
        if memory_service is not None:
            for method in ("get_active_preferences", "process_turn"):
                if not callable(getattr(memory_service, method, None)):
                    raise PolicyActionError(
                        f"memory_service must provide a callable {method}() method"
                    )
            if not isinstance(user_key, str) or not user_key.strip():
                raise PolicyActionError(
                    "a non-empty user_key is required when memory_service is configured"
                )

        self._policy = policy
        self._capability = capability
        self._validator = validator or ActionValidator()
        self._verifier = verifier or ResultVerifier()
        self._completion_guard = completion_guard or CompletionGuard()
        self._observation_adapter = observation_adapter or ObservationAdapter()
        self._memory_service = memory_service
        self._user_key = user_key.strip() if isinstance(user_key, str) else None
        self._limits = limits or LoopLimits()
        self._driver = driver
        self._reasoning = reasoning_executor
        self._candidate_plane = candidate_plane

    # -- metadata ---------------------------------------------------------- #

    @property
    def policy(self) -> AgentPolicy:
        """The injected next-action source."""
        return self._policy

    @property
    def capability(self) -> RecommendFromHistoryCapability:
        """The injected recommendation capability."""
        return self._capability

    @property
    def limits(self) -> LoopLimits:
        """The budgets the controller enforces."""
        return self._limits

    @property
    def driver(self) -> str:
        """Which driver runs the loop: ``"graph"`` or ``"direct"``."""
        return self._driver

    @property
    def reasoning_executor(self) -> Any:
        """The Stage 4 read-only reasoning executor, or ``None`` when not configured."""
        return self._reasoning

    @property
    def candidate_plane(self) -> Any:
        """The Stage 2 multi-source candidate plane, or ``None`` for the single-source path."""
        return self._candidate_plane

    @property
    def offers_multiple_sources(self) -> bool:
        """True when this deployment can execute more than the history source."""
        return self._candidate_plane is not None

    @property
    def offers_reasoning(self) -> bool:
        """True when this deployment can execute read-only reasoning actions."""
        return self._reasoning is not None

    @property
    def uses_memory(self) -> bool:
        """True when the loop loads and commits conversational preference memory."""
        return self._memory_service is not None

    # -- entry points ------------------------------------------------------ #

    def invoke(
        self,
        agent_input: AgentInput,
        *,
        run_id: str | None = None,
        task_state: TaskState | None = None,
    ) -> LoopResult:
        """Run the loop for one validated application input."""
        if not isinstance(agent_input, AgentInput):
            raise PolicyActionError(
                f"invoke expects an AgentInput, got {type(agent_input).__name__}"
            )
        return self._run(
            user_message=agent_input.user_message,
            trusted_history=tuple(agent_input.trusted_user_history),
            turn_id=agent_input.turn_id,
            run_id=run_id or build_run_id(),
            task_state=task_state,
        )

    def run(
        self,
        user_message: str,
        trusted_user_history: TrustedHistory,
        *,
        turn_id: str | None = None,
        run_id: str | None = None,
        task_state: TaskState | None = None,
    ) -> LoopResult:
        """Convenience wrapper around :meth:`invoke` with validated inputs.

        ``task_state`` carries the current task's state, including any question a prior turn
        left pending.  A suspended task is resumed by handing its state back in, which is how
        a multi-turn interaction continues without the loop holding anything across runs.
        """
        return self.invoke(
            AgentInput(
                user_message=user_message,
                trusted_user_history=trusted_user_history,
                turn_id=turn_id,
            ),
            run_id=run_id,
            task_state=task_state,
        )

    # -- drivers ----------------------------------------------------------- #

    def new_engine(
        self,
        agent_input: AgentInput,
        *,
        run_id: str | None = None,
        task_state: TaskState | None = None,
        candidate_plane: Any = None,
    ) -> _LoopEngine:
        """Build an engine for one run.

        Exposed so the loop topology can be driven node-by-node (and asserted structurally)
        without the controller hiding it, and so both drivers demonstrably share one engine.
        """
        if not isinstance(agent_input, AgentInput):
            raise PolicyActionError(
                f"new_engine expects an AgentInput, got {type(agent_input).__name__}"
            )
        return _LoopEngine(
            policy=self._policy,
            capability=self._capability,
            validator=self._validator,
            verifier=self._verifier,
            completion_guard=self._completion_guard,
            observation_adapter=self._observation_adapter,
            memory_service=self._memory_service,
            user_key=self._user_key,
            limits=self._limits,
            user_message=agent_input.user_message,
            trusted_history=tuple(agent_input.trusted_user_history),
            turn_id=agent_input.turn_id,
            run_id=run_id or build_run_id(),
            task_state=task_state,
            reasoning_executor=self._reasoning,
            candidate_plane=(
                candidate_plane if candidate_plane is not None else self._candidate_plane
            ),
        )

    def _run(
        self,
        *,
        user_message: str,
        trusted_history: TrustedHistory,
        turn_id: str | None,
        run_id: str,
        task_state: TaskState | None = None,
    ) -> LoopResult:
        """Run one turn with the configured driver."""
        engine = self.new_engine(
            AgentInput(
                user_message=user_message,
                trusted_user_history=trusted_history,
                turn_id=turn_id,
            ),
            run_id=run_id,
            task_state=task_state,
        )
        if self._driver == "graph":
            return self._run_graph_driver(engine)
        return self._run_direct_driver(engine)

    def _run_direct_driver(self, engine: _LoopEngine) -> LoopResult:
        """Run the engine's steps in Python, in the canonical cycle order."""
        engine.initialize()
        while True:
            outcome = engine.check_limits()
            if not outcome.continue_loop:
                return engine.result(outcome)

            outcome = engine.choose()
            if not outcome.continue_loop:
                return engine.result(outcome)

            outcome = engine.validate_action()
            if not outcome.continue_loop:
                return engine.result(outcome)

            target = engine.dispatch_target()
            if target == "reason":
                engine.reason()
                engine.verify()
                engine.observe()
                outcome = engine.update_state()
                if not outcome.continue_loop:  # pragma: no cover - defensive
                    return engine.result(outcome)
                continue
            if target == "clarify":
                outcome = engine.ask_clarification()
                engine.finish_clarification()
                return engine.result(outcome)
            if target == "complete":
                engine.run_completion_guard()
                verdict = engine.completion_target()
                if verdict == "finalize":
                    engine.finalize()
                    return engine.terminal_result()
                if verdict == "refuse":
                    engine.refuse_completion()
                    continue
                return engine.result(engine.abort_completion())

            outcome = engine.check_tool_budget()
            if not outcome.continue_loop:
                return engine.result(outcome)

            engine.execute()
            engine.verify()
            engine.observe()
            outcome = engine.update_state()
            if not outcome.continue_loop:  # pragma: no cover - defensive
                return engine.result(outcome)
            # Control returns to the policy: the loop continues with the updated context.

    def _run_graph_driver(self, engine: _LoopEngine) -> LoopResult:
        """Run the compiled LangGraph cycle, whose back-edge is ``update_state -> policy``."""
        from .topology import build_loop_graph

        graph = build_loop_graph()
        try:
            graph.invoke(
                {"engine": engine, "step": 0},
                config={"recursion_limit": recursion_limit(self._limits)},
            )
        except GraphRecursionError:
            # The framework backstop fired.  The controller still owns the verdict: report the
            # budget termination it would have reached, rather than letting a topology fault
            # surface as an exception.  A test asserts a runaway loop ends this way.
            engine.check_limits()
            if not engine.control.is_terminal:
                engine._terminate(  # noqa: SLF001 - the engine owns its control state
                    RunStatus.ABORTED,
                    TerminationReason.MAX_STEPS,
                    "the loop graph reached its recursion limit",
                )

        if engine.control.status is RunStatus.RUNNING:  # pragma: no cover - safety net
            return engine.result(
                StepOutcome.stop(
                    RunStatus.FAILED,
                    TerminationReason.EXECUTION_FAILED,
                    "the loop topology stopped without recording a terminal state",
                )
            )
        return engine.terminal_result()


def recursion_limit(limits: LoopLimits) -> int:
    """Derive LangGraph's recursion limit from the controller's own budget.

    The controller's ``max_steps`` remains the authoritative boundary; this is a second,
    framework-level backstop so a topology bug cannot spin forever.

    The multiplier is deliberately generous.  One loop step visits up to eight nodes
    (``check_limits`` -> ``policy`` -> ``validate_action`` -> ``dispatch`` -> ``execute``
    -> ``verify`` -> ``observe`` -> ``update_state``), and a refused completion can add a
    ``complete``/``refuse`` pair on top.  A tighter derivation made LangGraph's limit fire
    *before* the controller's own boundary on small budgets, which turned a controller
    decision into a framework crash; the limit is now sized so the controller always
    reaches its own verdict first.
    """
    per_step = 12
    retry_headroom = 2 * (limits.max_retries + 1)
    # A generous multiplier on purpose.  LangGraph's limit is a *backstop*; the controller's
    # own budgets are the authoritative boundary, and a framework limit that fires first turns
    # a deterministic controller decision into a framework crash.  The graph driver also
    # survives a recursion stop by reporting MAX_STEPS, so reaching it is never fatal - but it
    # should not be reached in normal operation.
    return per_step * (limits.max_steps + 1) + retry_headroom + 32


def _error_code(exc: BaseException) -> str:
    """Return a stable, payload-free code for an execution failure.

    Tool domain errors carry their own stable code; everything else is reported by type
    name only, so no exception message (which may name internals) reaches the trajectory.
    """
    code = getattr(exc, "code", None)
    if isinstance(code, str) and code:
        return code
    return type(exc).__name__


def _source_for(action: ActionKind, result: Any) -> Any:
    """Return the candidate source a plane result came from."""
    from .arguments import CandidateSource

    mapping = {
        ActionKind.SEARCH_CATALOG: CandidateSource.CATALOG_SEARCH,
        ActionKind.FIND_SIMILAR: CandidateSource.SIMILAR_ITEM,
        ActionKind.SELECT_SOURCE: CandidateSource.CATALOG_SEARCH,
    }
    declared = getattr(result, "source", None)
    if isinstance(declared, CandidateSource):
        return declared
    return mapping.get(action, CandidateSource.CATALOG_SEARCH)

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

from .capability import RecommendFromHistoryCapability
from .completion import CompletionGuard
from .context import CandidateState, PolicyContext
from .schemas import (
    ActionKind,
    ActionProposal,
    AgentPolicy,
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

        self.last_observation: Any = None
        self.last_verification: VerificationResult | None = None
        self.produced_recommendation = False
        self.candidates_grounded = False
        self.initialized = False

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
        actions.append(ActionKind.FINISH)
        return tuple(actions)

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
        except PolicyActionError as exc:
            self.trajectory.record(
                step_index=step_index,
                action_id=self.last_action_id,
                policy_context_summary=context.summary(),
                note=f"policy raised PolicyActionError: {exc}",
            )
            return self._terminate(
                RunStatus.ABORTED, TerminationReason.NO_AVAILABLE_ACTION, str(exc)
            )
        except Exception as exc:  # noqa: BLE001 - a policy failure is normalised
            self.trajectory.record(
                step_index=step_index,
                action_id=self.last_action_id,
                policy_context_summary=context.summary(),
                note=f"policy failed: {type(exc).__name__}",
            )
            return self._terminate(
                RunStatus.FAILED,
                TerminationReason.EXECUTION_FAILED,
                f"policy failed: {type(exc).__name__}",
            )
        return StepOutcome.proceed()

    # -- validate_action --------------------------------------------------- #

    def validate_action(self) -> StepOutcome:
        """Turn the untrusted proposal into a controller-stamped action, or refuse.

        This phase owns the per-step action scratch: it clears the previous step's result
        itself and then decides the new one.  Nothing downstream depends on a *previous*
        phase having cleared it, which is what makes each phase safe to enter on its own.
        """
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

    # -- verify ------------------------------------------------------------ #

    def verify(self) -> StepOutcome:
        """Verify a raw domain result.  A result that already failed stays failed."""
        if not self.has_current_action():
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
        """Adapt the verified (or refused) result into the policy-visible observation."""
        if not self.has_current_action():
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

        if self.verification.verified and self.domain_result is not None:
            self.produced_recommendation = True
            self.candidates_grounded = self.domain_result.returned_k > 0
            self._adopt_candidates(self.domain_result)
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
        )
        self.control = self.control.advanced(
            action=self.validated.action,
            action_id=self.validated.action_id,
            consumed_tool_call=True,
        )
        self.last_proposal_rejected = False
        return StepOutcome.proceed()

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
                candidate_count=(
                    self.observation.returned_k if self.observation is not None else 0
                ),
                memory_write_attempted=self.memory_service is not None,
            ),
            note="completion accepted; run finished",
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
    def uses_memory(self) -> bool:
        """True when the loop loads and commits conversational preference memory."""
        return self._memory_service is not None

    # -- entry points ------------------------------------------------------ #

    def invoke(self, agent_input: AgentInput, *, run_id: str | None = None) -> LoopResult:
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
        )

    def run(
        self,
        user_message: str,
        trusted_user_history: TrustedHistory,
        *,
        turn_id: str | None = None,
        run_id: str | None = None,
    ) -> LoopResult:
        """Convenience wrapper around :meth:`invoke` with validated inputs."""
        return self.invoke(
            AgentInput(
                user_message=user_message,
                trusted_user_history=trusted_user_history,
                turn_id=turn_id,
            ),
            run_id=run_id,
        )

    # -- drivers ----------------------------------------------------------- #

    def new_engine(
        self,
        agent_input: AgentInput,
        *,
        run_id: str | None = None,
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
        )

    def _run(
        self,
        *,
        user_message: str,
        trusted_history: TrustedHistory,
        turn_id: str | None,
        run_id: str,
    ) -> LoopResult:
        """Run one turn with the configured driver."""
        engine = self.new_engine(
            AgentInput(
                user_message=user_message,
                trusted_user_history=trusted_history,
                turn_id=turn_id,
            ),
            run_id=run_id,
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

            if engine.dispatch_target() == "complete":
                engine.run_completion_guard()
                target = engine.completion_target()
                if target == "finalize":
                    engine.finalize()
                    return engine.terminal_result()
                if target == "refuse":
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
        graph.invoke(
            {"engine": engine, "step": 0},
            config={"recursion_limit": recursion_limit(self._limits)},
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
    per_step = 8
    retry_headroom = 2 * (limits.max_retries + 1)
    return per_step * (limits.max_steps + 1) + retry_headroom + 16


def _error_code(exc: BaseException) -> str:
    """Return a stable, payload-free code for an execution failure.

    Tool domain errors carry their own stable code; everything else is reported by type
    name only, so no exception message (which may name internals) reaches the trajectory.
    """
    code = getattr(exc, "code", None)
    if isinstance(code, str) and code:
        return code
    return type(exc).__name__

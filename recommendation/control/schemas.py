"""Control-plane contracts for AgentRec-X 2.0-alpha (bounded agent loop).

This module defines **who may decide the next step**, and nothing about how a
recommendation is produced.  Stage 1 keeps the recommendation pipeline frozen and moves
only the control authority, so the types here are deliberately narrow.

The authority chain
-------------------
::

    AgentPolicy            proposes            ActionProposal      (untrusted)
        |
        v
    ActionValidator        checks + stamps      ValidatedAction     (trusted metadata)
        |
        v
    Capability             executes             DomainResult        (raw tool output)
        |
        v
    ResultVerifier         checks               VerificationResult
        |
        v
    ObservationAdapter     minimises            Observation         (safe for Policy)
        |
        v
    LoopController         owns the loop        ControlState

Four separations do the work, and each is a type boundary rather than a convention:

``ActionProposal != ValidatedAction``
    The policy proposes; the controller validates and stamps.  ``ActionProposal`` has no
    field for an ``action_id``, a ``step_index``, a ``turn_id``, a ``run_id``, a tool
    name, SQL, candidate ids or execution metadata, so a policy *cannot* express them.
    ``ValidatedAction`` carries the controller-generated identity and the validated
    arguments.

``PolicyContext != AgentGraphState``
    The policy sees a controlled projection (:class:`PolicyContext`), never the trusted
    runtime state.  There is no field for trusted interaction history, a memory store, a
    metadata index, a database handle, a user key, an engine or a credential.

``DomainResult != Observation``
    A capability's raw output cannot reach the policy.  Only a
    :class:`~recommendation.control.verification.VerificationResult` that passed can be
    adapted into an :class:`Observation`, and the observation carries counts, status and
    a *reference* to the candidate set - never the candidate list itself.

``ActionProposal.FINISH != finished``
    ``FINISH`` is a proposal like any other.  It passes the validator and then the
    :class:`~recommendation.control.completion.CompletionGuard`; a policy can never mark
    a run complete by itself.

Reserved for later stages (declared, not implemented): ``WAITING_FOR_USER`` and every
action other than ``RECOMMEND_FROM_HISTORY`` / ``FINISH``.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Literal, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field

from recommendation.tools.schemas import DEFAULT_K, MAX_K, MIN_K

__all__ = [
    "CONTROL_PLANE_VERSION",
    "OBSERVATION_VERSION",
    "ActionKind",
    "ActionProposal",
    "AgentPolicy",
    "ControlState",
    "DomainResult",
    "LoopLimits",
    "Observation",
    "PolicyActionError",
    "RecommendationDomainResult",
    "RecommendationObservation",
    "RunStatus",
    "STAGE_1_ACTIONS",
    "StateChange",
    "TerminationReason",
    "TrajectoryStep",
    "ValidatedAction",
    "VerificationResult",
]

#: Version of the control-plane contract.  Bumped when the action space, the validation
#: rules or the observation shape change.
CONTROL_PLANE_VERSION = 1

#: Version of the observation contract.
OBSERVATION_VERSION = 1


class PolicyActionError(Exception):
    """A policy or a control-plane component violated the control protocol.

    Raised instead of defaulting to an action, exactly as
    :class:`~recommendation.agent.decision.MalformedDecision` is raised instead of
    defaulting to a route.
    """


# --------------------------------------------------------------------------- #
# Action space
# --------------------------------------------------------------------------- #


class ActionKind(str, Enum):
    """The Stage 1 semantic action space.

    Two actions only.  ``SEARCH_CATALOG``, ``ASK_USER``, ``COMPARE``, ``GET_DETAILS``,
    ``BUNDLE``, ``CHECKOUT`` and ``MEMORY_WRITE`` are explicitly **out of scope for
    Stage 1** and are therefore not members of this enum - adding them later is a
    deliberate control-plane version bump, not a policy decision.
    """

    RECOMMEND_FROM_HISTORY = "recommend_from_history"
    FINISH = "finish"


#: The Stage 1 action space as an ordered tuple (declaration order is the documented
#: preference order used by the stage-1 policies when both actions are available).
STAGE_1_ACTIONS: tuple[ActionKind, ...] = (
    ActionKind.RECOMMEND_FROM_HISTORY,
    ActionKind.FINISH,
)


class ActionProposal(BaseModel):
    """An **untrusted** proposal produced by an :class:`AgentPolicy`.

    What this type deliberately does not have:

    * ``action_id`` / ``step_index`` / ``turn_id`` / ``run_id`` - controller-owned
      execution metadata;
    * ``tool`` / ``tool_name`` / ``sql`` / ``candidates`` / ``item_ids`` /
      ``parent_asins`` - there is no channel through which a policy could name a tool,
      supply a query, or hand over candidate identities;
    * ``history`` / ``trusted_user_history`` - a proposal cannot carry or forge
      behavioural history;
    * ``finished`` / ``status`` / ``termination_reason`` - a proposal can never mark a
      run complete.

    A proposal is only ever *proposed*: :class:`~recommendation.control.validation.ActionValidator`
    turns it into a :class:`ValidatedAction` or refuses it.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    action: ActionKind
    version: int = CONTROL_PLANE_VERSION

    #: Requested number of candidates.  Only meaningful for
    #: ``RECOMMEND_FROM_HISTORY``; the Tool's own range is reused rather than restated,
    #: and the validator re-checks it at the trusted boundary.
    k: int | None = Field(default=None, ge=MIN_K, le=MAX_K)

    #: Short, non-authoritative explanation of *why* the policy chose this action.  It is
    #: recorded in the trajectory and is never used as a control signal, a tool argument
    #: or a user-facing claim.  Claims about products are forbidden here by contract: it
    #: is a routing rationale, not evidence.
    rationale: str | None = Field(default=None, max_length=280)

    def model_post_init(self, __context: Any) -> None:
        """Enforce the cross-field rules the per-field schema cannot express."""
        if self.action is ActionKind.RECOMMEND_FROM_HISTORY:
            if self.k is None:
                raise ValueError("k is required when action is 'recommend_from_history'")
        elif self.k is not None:
            raise ValueError("k must be omitted when action is 'finish'")

    @property
    def requested_k(self) -> int:
        """The candidate count this proposal asks for; only valid on the recommend action."""
        if self.action is not ActionKind.RECOMMEND_FROM_HISTORY:
            raise PolicyActionError("requested_k is only defined for 'recommend_from_history'")
        return DEFAULT_K if self.k is None else self.k


class ValidatedAction(BaseModel):
    """A proposal the controller has accepted, stamped with controller-owned metadata.

    The controller adds:

    * ``action_id`` - a deterministic identity for this action within the run;
    * ``step_index`` - the loop step the action belongs to;
    * ``run_id`` / ``turn_id`` - the application-owned run and turn identities.

    Note what a policy still cannot do even after validation: ``ValidatedAction`` has no
    history field, no candidate field and no tool field.  Validation authorises an
    *action kind*, never a payload.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    action: ActionKind
    version: int = CONTROL_PLANE_VERSION

    action_id: str
    step_index: int = Field(..., ge=0)
    run_id: str
    turn_id: str | None = None

    k: int = Field(..., ge=MIN_K, le=MAX_K)

    #: The proposal's rationale, carried through for trajectory purposes only.
    rationale: str | None = None

    @property
    def is_terminal_proposal(self) -> bool:
        """True when this action asks to end the run (still subject to CompletionGuard)."""
        return self.action is ActionKind.FINISH


# --------------------------------------------------------------------------- #
# Loop budgets and run status
# --------------------------------------------------------------------------- #


class LoopLimits(BaseModel):
    """Deterministic execution budgets.

    These are the termination boundary.  They are controller configuration, not policy
    input: no policy can read, raise or override them, and exhaustion terminates the run
    deterministically without asking the policy whether it would like to continue.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    #: ``ge=0`` is deliberate for the two execution budgets: zero is the only way to
    #: express "this run may not execute at all", and it is exactly the case the
    #: controller's budget guards exist for.  A budget that cannot be zero cannot be
    #: tested.
    max_steps: int = Field(default=6, ge=0, le=64)
    max_tool_calls: int = Field(default=4, ge=0, le=64)
    max_retries: int = Field(default=1, ge=0, le=8)


class RunStatus(str, Enum):
    """Lifecycle status of one loop run."""

    RUNNING = "running"
    FINISHED = "finished"
    ABORTED = "aborted"
    FAILED = "failed"
    #: Reserved for a future clarification stage.  Stage 1 never sets it.
    WAITING_FOR_USER = "waiting_for_user"


class TerminationReason(str, Enum):
    """Why a run stopped.  Exactly one is recorded for every terminal run."""

    #: A FINISH proposal passed the CompletionGuard.
    COMPLETED = "completed"
    #: The step budget ran out.
    MAX_STEPS = "max_steps"
    #: The tool-call budget ran out.
    MAX_TOOL_CALLS = "max_tool_calls"
    #: The policy proposed an action outside ``available_actions``, or an action whose
    #: arguments failed validation.
    INVALID_ACTION = "invalid_action"
    #: The CompletionGuard refused to end the run and no progress remained possible.
    COMPLETION_REFUSED = "completion_refused"
    #: The run had no legal action available at all.
    NO_AVAILABLE_ACTION = "no_available_action"
    #: A capability or verifier raised a terminal failure.
    EXECUTION_FAILED = "execution_failed"


class ControlState(BaseModel):
    """The controller-owned, frozen snapshot of one loop run.

    Stage 1 keeps this deliberately small: budgets, counters, the last action kind and
    the terminal status.  It carries no candidate, no history and no score.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    run_id: str
    turn_id: str | None = None

    status: RunStatus = RunStatus.RUNNING
    termination_reason: TerminationReason | None = None
    termination_detail: str | None = Field(default=None, max_length=280)

    step_count: int = Field(default=0, ge=0)
    tool_call_count: int = Field(default=0, ge=0)
    retry_count: int = Field(default=0, ge=0)

    limits: LoopLimits = Field(default_factory=LoopLimits)

    last_action_kind: ActionKind | None = None
    last_action_id: str | None = None

    @property
    def is_terminal(self) -> bool:
        """True once the run has stopped for any reason."""
        return self.status is not RunStatus.RUNNING

    @property
    def steps_remaining(self) -> int:
        """Steps left before the controller terminates deterministically."""
        return max(0, self.limits.max_steps - self.step_count)

    @property
    def tool_calls_remaining(self) -> int:
        """Tool calls left before the controller terminates deterministically."""
        return max(0, self.limits.max_tool_calls - self.tool_call_count)

    def advanced(
        self,
        *,
        action: ActionKind,
        action_id: str,
        consumed_tool_call: bool,
    ) -> ControlState:
        """Return the state after one completed step (immutable update)."""
        return self.model_copy(
            update={
                "step_count": self.step_count + 1,
                "tool_call_count": self.tool_call_count + int(consumed_tool_call),
                "last_action_kind": action,
                "last_action_id": action_id,
            }
        )

    def terminated(
        self,
        status: RunStatus,
        reason: TerminationReason,
        detail: str | None = None,
    ) -> ControlState:
        """Return the terminal state (immutable update)."""
        return self.model_copy(
            update={
                "status": status,
                "termination_reason": reason,
                "termination_detail": detail,
            }
        )


# --------------------------------------------------------------------------- #
# Observations
# --------------------------------------------------------------------------- #


class Observation(BaseModel):
    """Base class for everything a policy is allowed to see about an executed action.

    An observation is a **minimised** projection: status, counts, a reference and a
    verification verdict.  It is not the domain result and never contains one.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    version: int = OBSERVATION_VERSION
    action_id: str
    step_index: int = Field(..., ge=0)
    action: ActionKind
    kind: str = "observation"
    verification_status: Literal["verified", "refused"] = "verified"


class RecommendationObservation(Observation):
    """Stage 1's only concrete observation.

    It answers exactly the questions a *control* decision needs - did the action work,
    how many candidates came back, is the result grounded - and answers nothing else.
    Raw scores, product metadata, internal item ids, trusted history, memory contents and
    the candidate list itself are all absent by construction: the policy gets a
    ``candidate_set_ref`` (an opaque identity for the verified set), not candidates.
    """

    kind: str = "recommendation"
    source: str = "recommend_from_history"
    status: Literal["ok", "empty", "failed"]
    requested_k: int = Field(..., ge=0)
    returned_k: int = Field(..., ge=0)
    has_candidates: bool
    candidate_set_ref: str
    #: Short, non-authoritative summary of what verification checked.  It never carries
    #: a product fact or a score.
    verification_note: str | None = Field(default=None, max_length=280)


# --------------------------------------------------------------------------- #
# Domain results
# --------------------------------------------------------------------------- #


class DomainResult(BaseModel):
    """Base class for a capability's raw, unverified output.

    A ``DomainResult`` is **not** an observation.  It may contain the full internal
    payload - the Tool result, the enrichment, the evidence report, the reranking - and
    it is never handed to a policy.  It must pass a
    :class:`~recommendation.control.verification.ResultVerifier` before an observation is
    built from it.
    """

    model_config = ConfigDict(extra="forbid", frozen=True, arbitrary_types_allowed=True)

    action_id: str
    status: Literal["ok", "empty", "failed"] = "ok"


class RecommendationDomainResult(DomainResult):
    """Raw output of :class:`~recommendation.control.capability.RecommendFromHistoryCapability`.

    The internal artifacts are carried so the *trusted* side of the system (verifier,
    renderer, serializer) can use them.  ``policy_visible`` is ``False`` and stays
    ``False``: this object never crosses the observation boundary.
    """

    source: str = "recommend_from_history"

    #: The accepted Tool result - the authoritative candidate identity, order and score.
    tool_result: Any = None
    #: Milestone 8 candidate-scoped evidence, or ``None`` when no enricher is configured.
    enrichment: Any = None
    #: Milestone 10A evidence report, or ``None`` when no matcher is configured.
    preference_evidence: Any = None
    #: Milestone 10B reranking report, or ``None`` when no reranker is configured.
    reranking: Any = None

    requested_k: int = Field(default=0, ge=0)
    returned_k: int = Field(default=0, ge=0)
    eligible_candidates: int = Field(default=0, ge=0)

    #: Opaque identity of the verified candidate set handed to the policy instead of the
    #: candidates themselves.
    candidate_set_ref: str = ""

    #: Always ``False``: a domain result is never policy-visible.  Present as an explicit,
    #: assertable statement of the boundary rather than an implicit convention.
    policy_visible: Literal[False] = False


# --------------------------------------------------------------------------- #
# Verification
# --------------------------------------------------------------------------- #


class VerificationResult(BaseModel):
    """The verdict of a :class:`~recommendation.control.verification.ResultVerifier`.

    A refusal is a first-class outcome, not an exception: the run fails closed, and the
    refusal becomes an observation the policy may react to.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    verified: bool
    code: str
    detail: str | None = Field(default=None, max_length=280)
    checks: tuple[str, ...] = ()


# --------------------------------------------------------------------------- #
# Trajectory
# --------------------------------------------------------------------------- #


class StateChange(BaseModel):
    """The compact record of what one step changed in the trusted state.

    Only whitelisted, non-sensitive facts are recorded.  Trusted history, memory
    contents, candidate ids and raw scores are never part of a state delta.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    step_index: int = Field(..., ge=0)
    action: ActionKind
    produced_candidates: bool = False
    candidate_count: int = Field(default=0, ge=0)
    memory_write_attempted: bool = False
    run_status: RunStatus = RunStatus.RUNNING


class TrajectoryStep(BaseModel):
    """Everything needed to answer "why did the next step happen?".

    Recorded for every loop iteration from the first version, because a control plane
    nobody can audit is not an improvement.  It deliberately records *summaries*: no
    secrets, no credentials, no trusted history, no full prompts, no candidate list and
    no raw domain payload.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    run_id: str
    turn_id: str | None = None
    step_index: int = Field(..., ge=0)
    action_id: str

    #: What the policy was told (a summary of :class:`PolicyContext`, not the state).
    policy_context_summary: dict[str, Any] = Field(default_factory=dict)
    #: What the policy proposed, or ``None`` when the policy failed to propose.
    action_proposal: dict[str, Any] | None = None
    validation_result: VerificationResult | None = None
    validated_action: dict[str, Any] | None = None

    #: Opaque reference to the raw domain payload; the payload itself is not recorded.
    tool_result_ref: str | None = None
    verification_result: VerificationResult | None = None
    observation: dict[str, Any] | None = None

    state_delta: StateChange | None = None

    #: Set when the step ended without a validated action (invalid proposal, exhausted
    #: budget, refusal).  Kept short and payload-free.
    note: str | None = Field(default=None, max_length=280)


# --------------------------------------------------------------------------- #
# Policy seam
# --------------------------------------------------------------------------- #


@runtime_checkable
class AgentPolicy(Protocol):
    """The single method the loop requires from a next-action source.

    ::

        choose(context: PolicyContext) -> ActionProposal

    This is the dependency-injection seam that keeps 2.0-alpha offline: Stage 1 ships a
    deterministic :class:`~recommendation.control.policy.RuleBasedPolicy`, and a later
    stage can supply an LLM-backed policy **without changing the controller, the
    validator, the verifier, the completion guard or the capability**.

    What a policy may do: read the :class:`PolicyContext` it is given and return exactly
    one :class:`ActionProposal` drawn from ``context.available_actions``.

    What a policy may not do (enforced structurally, not by convention): execute a tool,
    write trusted state, increment counters, touch trusted history, supply candidate ids,
    commit memory, bypass the validator, bypass the capability, render the final answer,
    judge its own output valid, or declare the run finished.
    """

    def choose(self, context: Any) -> ActionProposal:
        """Return exactly one proposal for the given control context."""
        ...

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

``WAITING_FOR_USER`` is a real status from Stage 3 onwards: a run that proposed
``ASK_CLARIFICATION`` suspends rather than completing or failing, and is resumable.

Every action in :class:`ActionKind` is declared; whether a capability is *configured* to
execute it is a separate, deployment-level fact computed by the controller and offered to
the policy through ``available_actions``.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Literal, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field

from recommendation.tools.schemas import DEFAULT_K, MAX_K, MIN_K

from .arguments import (
    AcquireEvidenceArguments,
    FindSimilarArguments,
    AskClarificationArguments,
    BundleArguments,
    CheckCompatibilityArguments,
    CompareArguments,
    EmptyArguments,
    GetDetailsArguments,
    SearchCatalogArguments,
    SelectSourceArguments,
    TradeOffArguments,
)


__all__ = [
    "ARGUMENTS_BY_ACTION",
    "CANDIDATE_ACTIONS",
    "EVIDENCE_ACTIONS",
    "TOOL_CALL_ACTIONS",
    "CONTROL_PLANE_VERSION",
    "NON_EXECUTING_ACTIONS",
    "READ_ONLY_ACTIONS",
    "OBSERVATION_VERSION",
    "ActionKind",
    "ActionProposal",
    "AgentPolicy",
    "BundleObservation",
    "CandidateSetObservation",
    "ClarificationObservation",
    "ComparisonObservation",
    "ControlState",
    "CompatibilityObservation",
    "DetailObservation",
    "DomainResult",
    "EvidenceObservation",
    "FailureObservation",
    "LoopLimits",
    "Observation",
    "PolicyActionError",
    "RecommendationDomainResult",
    "RecommendationObservation",
    "RunStatus",
    "STAGE_1_ACTIONS",
    "StateChange",
    "TerminationReason",
    "TradeOffObservation",
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

    ``code`` is an optional **stable** machine-readable class, so a failure that a policy can
    act on reaches it as a code rather than as a message.  It exists because a trust-boundary
    refusal and an execution failure call for different recoveries, and a policy that only saw
    "something failed" could not tell them apart.  The message may name internals; the code
    must not.
    """

    def __init__(self, message: str, *, code: str | None = None) -> None:
        super().__init__(message)
        #: Stable failure class, or ``None`` for a generic protocol violation.
        self.code = code


# --------------------------------------------------------------------------- #
# Action space
# --------------------------------------------------------------------------- #


class ActionKind(str, Enum):
    """The semantic action space.

    Grouped by what an action is allowed to change.  Nothing in this enum executes a
    commercial transaction: there is no cart, checkout, payment or order action, and no
    later stage adds one.  The project's endpoint is a recommendation decision-support
    agent.

    **Stage 1 - control-plane proof.**

    * ``RECOMMEND_FROM_HISTORY`` - candidates from the trusted history recommender;
    * ``FINISH`` - propose ending the run (the CompletionGuard still authorises it).

    **Stage 2 - multi-source candidate plane.**

    * ``SELECT_SOURCE`` - consult one explicitly named trusted candidate source;
    * ``SEARCH_CATALOG`` - full-catalogue lexical search (needs an explicit query);
    * ``FIND_SIMILAR`` - similar-item retrieval from an item-item relationship index.

    **Stage 3 - interaction / personalization.**

    * ``ASK_CLARIFICATION`` - suspend the run and ask a question that would materially
      change the outcome;
    * ``PROPOSE_MEMORY_WRITE`` - propose a persistent-memory change for validation.

    **Stage 4 - recommendation reasoning (all read-only over grounded facts).**

    * ``GET_DETAILS`` - expand evidence for candidates the run already holds;
    * ``COMPARE`` - structured comparison over grounded attributes;
    * ``TRADE_OFF`` - comparison under a stated priority;
    * ``CHECK_COMPATIBILITY`` - deterministic compatibility verdict against a requirement;
    * ``BUNDLE`` - compositional reasoning over a set of grounded candidates;
    * ``VERIFY`` - re-check the current candidate set against the active constraints.

    ``CHECKOUT`` is deliberately absent and will not be added: recommendation decision
    support stops here.
    """

    # Stage 1
    RECOMMEND_FROM_HISTORY = "recommend_from_history"
    FINISH = "finish"
    # Stage 2
    SELECT_SOURCE = "select_source"
    SEARCH_CATALOG = "search_catalog"
    FIND_SIMILAR = "find_similar"
    # Stage 3
    ASK_CLARIFICATION = "ask_clarification"
    PROPOSE_MEMORY_WRITE = "propose_memory_write"
    # Stage 4
    GET_DETAILS = "get_details"
    COMPARE = "compare"
    TRADE_OFF = "trade_off"
    CHECK_COMPATIBILITY = "check_compatibility"
    BUNDLE = "bundle"
    VERIFY = "verify"
    # Phase 4: reach a trusted evidence source for facts the base catalogue projection does not
    # carry.  Not a reasoning action: it does not read facts the run already has, it *acquires*
    # new ones, so it consumes a tool call like the candidate sources do.
    ACQUIRE_EVIDENCE = "acquire_evidence"


#: The accepted Stage 1 action space.  Kept as a named constant because the Stage 1 policy
#: and its tests depend on the *exact* Stage 1 menu, and because a Stage 2 deployment that
#: configures no new source must behave identically to Stage 1.
STAGE_1_ACTIONS: tuple[ActionKind, ...] = (
    ActionKind.RECOMMEND_FROM_HISTORY,
    ActionKind.FINISH,
)

#: Actions that may produce candidate identities.  Every one of them is executed by a
#: trusted tool; a policy selects among them but never produces a candidate itself.
CANDIDATE_ACTIONS: tuple[ActionKind, ...] = (
    ActionKind.RECOMMEND_FROM_HISTORY,
    ActionKind.SELECT_SOURCE,
    ActionKind.SEARCH_CATALOG,
    ActionKind.FIND_SIMILAR,
)

#: Actions that only read or reason over candidates the run already holds.  They can add
#: evidence but never a candidate, which is what keeps "reasoning cannot widen the
#: candidate set" true by construction.
READ_ONLY_ACTIONS: tuple[ActionKind, ...] = (
    ActionKind.GET_DETAILS,
    ActionKind.COMPARE,
    ActionKind.TRADE_OFF,
    ActionKind.CHECK_COMPATIBILITY,
    ActionKind.BUNDLE,
    ActionKind.VERIFY,
)

#: Phase 4: actions that acquire **new trusted facts** about candidates the run already holds.
#:
#: Deliberately its own class.  They are not candidate-producing - the candidate set is untouched -
#: and they are not read-only reasoning either, because reasoning re-reads facts the run already
#: has while these reach a genuinely separate trusted source for facts it does not.  That
#: difference is why they consume a tool call and reasoning actions do not.
EVIDENCE_ACTIONS: tuple[ActionKind, ...] = (
    ActionKind.ACQUIRE_EVIDENCE,
)

#: Every action that reaches a tool and therefore consumes the tool-call budget.
TOOL_CALL_ACTIONS: tuple[ActionKind, ...] = (
    *CANDIDATE_ACTIONS,
    *EVIDENCE_ACTIONS,
)

#: Actions that do not execute anything and do not touch the candidate set.
NON_EXECUTING_ACTIONS: tuple[ActionKind, ...] = (
    ActionKind.FINISH,
    ActionKind.ASK_CLARIFICATION,
    ActionKind.PROPOSE_MEMORY_WRITE,
)

#: The argument contract for **every** action, in one place.
#:
#: The mapping is total on purpose: a new action cannot be added without declaring its
#: arguments here, and :meth:`ActionProposal.model_post_init` refuses a proposal whose
#: arguments do not have the declared type.  That is what stops one action from carrying
#: another action's arguments.
ARGUMENTS_BY_ACTION: dict[ActionKind, type[BaseModel]] = {
    ActionKind.RECOMMEND_FROM_HISTORY: EmptyArguments,
    ActionKind.FINISH: EmptyArguments,
    ActionKind.SELECT_SOURCE: SelectSourceArguments,
    ActionKind.SEARCH_CATALOG: SearchCatalogArguments,
    ActionKind.FIND_SIMILAR: FindSimilarArguments,
    ActionKind.ASK_CLARIFICATION: AskClarificationArguments,
    ActionKind.PROPOSE_MEMORY_WRITE: EmptyArguments,
    ActionKind.GET_DETAILS: GetDetailsArguments,
    ActionKind.COMPARE: CompareArguments,
    ActionKind.TRADE_OFF: TradeOffArguments,
    ActionKind.CHECK_COMPATIBILITY: CheckCompatibilityArguments,
    ActionKind.BUNDLE: BundleArguments,
    ActionKind.VERIFY: EmptyArguments,
    ActionKind.ACQUIRE_EVIDENCE: AcquireEvidenceArguments,
}


#: Actions that accept no domain arguments at all.
_ARGUMENTLESS_ACTIONS: frozenset[ActionKind] = frozenset(
    {
        ActionKind.FINISH,
        ActionKind.RECOMMEND_FROM_HISTORY,
        ActionKind.VERIFY,
    }
)


class ActionProposal(BaseModel):
    """An **untrusted** proposal produced by an :class:`AgentPolicy`.

    What this type deliberately does not have:

    * ``action_id`` / ``step_index`` / ``turn_id`` / ``run_id`` - controller-owned
      execution metadata;
    * ``tool`` / ``tool_name`` / ``sql`` - there is no channel through which a policy could
      name a tool or supply a query in a language the trusted side executes;
    * ``history`` / ``trusted_user_history`` - a proposal cannot carry or forge
      behavioural history;
    * ``finished`` / ``status`` / ``termination_reason`` - a proposal can never mark a
      run complete.

    ``arguments`` carries **typed, per-action** arguments (see
    :mod:`recommendation.control.arguments`).  Two of the reasoning actions accept
    ``parent_asins`` because comparing products requires naming them; those actions are
    read-only over candidates the run already holds and can never add one.  No action
    accepts a way to *invent* a candidate identity.

    A proposal is only ever *proposed*:
    :class:`~recommendation.control.validation.ActionValidator` turns it into a
    :class:`ValidatedAction` or refuses it.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    action: ActionKind
    version: int = CONTROL_PLANE_VERSION

    #: Requested number of candidates, for the actions whose result *is* a candidate list.
    #: The Tool's own range is reused rather than restated, and the validator re-checks it
    #: at the trusted boundary.
    k: int | None = Field(default=None, ge=MIN_K, le=MAX_K)

    #: Action-specific arguments.  ``None`` for an action that takes none.
    arguments: Any = None

    #: Short, non-authoritative explanation of *why* the policy chose this action.  It is
    #: recorded in the trajectory and is never used as a control signal, a tool argument
    #: or a user-facing claim.  Claims about products are forbidden here by contract: it
    #: is a routing rationale, not evidence.
    rationale: str | None = Field(default=None, max_length=280)

    def model_post_init(self, __context: Any) -> None:
        """Enforce the cross-field rules the per-field schema cannot express.

        Every argument model is frozen with ``extra="forbid"``, so this hook is what stops
        one action from carrying another action's arguments: the shape is checked against
        the action kind, not against a union of plausible fields.
        """
        if self.action is ActionKind.RECOMMEND_FROM_HISTORY:
            if self.k is None:
                raise ValueError("k is required when action is 'recommend_from_history'")
        elif self.k is not None:
            raise ValueError(f"k must be omitted when action is '{self.action.value}'")

        expected = ARGUMENTS_BY_ACTION.get(self.action)
        if expected is None:
            raise ValueError(f"no argument contract is declared for '{self.action.value}'")
        if self.action in _ARGUMENTLESS_ACTIONS:
            if self.arguments is not None:
                raise ValueError(
                    f"action '{self.action.value}' takes no arguments"
                )
            return
        if self.arguments is None:
            raise ValueError(
                f"action '{self.action.value}' requires {expected.__name__} arguments"
            )
        if isinstance(self.arguments, expected):
            return
        if isinstance(self.arguments, BaseModel):
            raise ValueError(
                f"action '{self.action.value}' requires {expected.__name__}, got "
                f"{type(self.arguments).__name__}"
            )
        # Validate a mapping into the declared model, so a caller may build a proposal
        # from plain data without importing the argument type.
        object.__setattr__(self, "arguments", expected.model_validate(self.arguments))

    @property
    def requested_k(self) -> int:
        """The candidate count this proposal asks for; only valid on the recommend action."""
        if self.action is not ActionKind.RECOMMEND_FROM_HISTORY:
            raise PolicyActionError("requested_k is only defined for 'recommend_from_history'")
        return DEFAULT_K if self.k is None else self.k

    @property
    def is_candidate_action(self) -> bool:
        """True when executing this action may produce candidate identities."""
        return self.action in CANDIDATE_ACTIONS

    @property
    def is_read_only(self) -> bool:
        """True when this action may only read candidates the run already holds."""
        return self.action in READ_ONLY_ACTIONS

    @property
    def is_non_executing(self) -> bool:
        """True when this action neither executes a tool nor touches the candidate set."""
        return self.action in NON_EXECUTING_ACTIONS


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

    #: The **validated** action arguments, carried through from the proposal.
    #:
    #: The validator guarantees the type matches ``ARGUMENTS_BY_ACTION[action]`` before this
    #: is populated, so an executor can rely on the shape.  Shape is not authorisation: a
    #: capability must still check that the values are permitted - is this source
    #: registered, is this identity grounded - because validation authorises an action
    #: kind, never a payload.
    arguments: Any = None

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
    #: Stage 3: the run suspended itself to ask the user a question.  This is neither a
    #: success nor a failure - the task is resumable, and the status is
    #: :attr:`RunStatus.WAITING_FOR_USER` rather than ``FINISHED`` or ``ABORTED``.
    AWAITING_USER = "awaiting_user"


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


class CandidateSetObservation(Observation):
    """Stage 2: what a **candidate-producing** action returned, minimised.

    A policy that may choose between candidate sources has to be able to see whether the
    source it chose actually helped - otherwise "Observation can change the next action"
    is not achievable, and adaptive retrieval degenerates into fixed fan-out.  What it is
    told is therefore exactly the *control-relevant* summary:

    * how many candidates this source contributed;
    * whether that was zero (the signal to try a different source);
    * how many of the run's candidates are now supported by more than one source, which is
      the ledger's agreement signal;
    * whether any identity the source returned could **not** be grounded - a finding about
      the source, surfaced rather than hidden;
    * an opaque reference to the verified set.

    What it is **not** told: any ``parent_asin``, any item id, any raw score, any product
    title or attribute, and any per-candidate ranking.  The counts are the whole channel.
    ``source`` names the trusted source, which is a fact about the system rather than about
    a product.
    """

    kind: str = "candidate_set"
    source: str
    status: Literal["ok", "empty", "failed"]
    requested_k: int = Field(..., ge=0)
    returned_k: int = Field(..., ge=0)
    has_candidates: bool
    candidate_set_ref: str
    #: How many candidate sources have contributed to this run so far.
    sources_used: int = Field(default=0, ge=0)
    #: How many of the run's grounded candidates are supported by more than one source.
    multi_source_count: int = Field(default=0, ge=0)
    #: How many identities the source returned that the trusted mapping does not know.
    #: Non-zero means the source drifted or hallucinated; it is never a candidate.
    ungrounded_count: int = Field(default=0, ge=0)
    verification_note: str | None = Field(default=None, max_length=280)


class ClarificationObservation(Observation):
    """Stage 3: the run suspended itself to ask a question.

    An ``ASK_CLARIFICATION`` action is not a failure and not a completion: it is a
    *suspension*.  This observation records that the run is now waiting for a user answer,
    what the question was, and which decision the answer is expected to change, so a
    resumed run can pick up with the same state and the trajectory can show why the run
    stopped mid-way.
    """

    kind: str = "clarification"
    status: Literal["waiting_for_user"] = "waiting_for_user"
    #: The question actually put to the user.  It is the policy's own text, echoed back
    #: for the trajectory and for the response layer; it never becomes a product claim.
    question: str = Field(..., max_length=400)
    #: The decision the policy claims the answer would change.
    blocks: str = Field(default="candidate_set", max_length=40)
    options: tuple[str, ...] = ()
    verification_note: str | None = Field(default=None, max_length=280)


class FailureObservation(Observation):
    """A structured failure a policy can act on, with its recovery class.

    The distinction that matters is :attr:`recoverable`.  A recoverable failure is a
    legitimate reason to try a different action; an unrecoverable one is a trust or
    integrity failure that must abort rather than be worked around.  The policy chooses
    only among the actions the deterministic recovery rules permit
    (:attr:`permitted_actions`), so recovery is bounded and auditable rather than a free
    choice.
    """

    kind: str = "failure"
    status: Literal["failed"] = "failed"
    #: Stable machine-readable failure class, e.g. ``no_candidates``,
    #: ``invalid_product_id``, ``missing_required_fact``.
    code: str
    recoverable: bool = True
    #: The recovery actions the controller permits for this failure class.  A policy may
    #: choose among these and nothing else.
    permitted_actions: tuple[str, ...] = ()
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

    # -- Stage 2 candidate-plane payload ------------------------------------ #
    #: The **grounded** identities this result contributes, in the source's own order.  Only
    #: identities confirmed by the grounding verifier appear here; the ledger's ungrounded
    #: audit entries never do.
    grounded_parent_asins: tuple[str, ...] = ()
    #: Trusted identity -> model item id, from the item mapping.  Used to give an adopted
    #: candidate set a rendering shape; it is never derived from a source's own claim.
    item_ids: dict[str, int] = Field(default_factory=dict)
    #: Which trusted source produced this result, when it was not the history recommender.
    source: str | None = None
    #: How many identities the source returned that the trusted side did not know.  A finding
    #: about the source, surfaced rather than hidden.
    ungrounded_count: int = Field(default=0, ge=0)
    #: The provenance summary for the trajectory: one record per grounded identity.
    provenance: tuple[dict[str, Any], ...] = ()


class DetailObservation(Observation):
    """Stage 4: grounded facts for candidates the run already holds.

    This is the one observation family that **does** carry product facts, and the reason is
    that a reasoning action's whole purpose is to obtain them.  The boundary it respects is
    different from the candidate boundary rather than looser than it:

    * every value here was copied out of the normalized catalogue by the trusted reasoner;
    * an attribute the catalogue does not contain appears as ``UNKNOWN``, never as an
      approximation;
    * nothing here can add, drop or reorder a candidate - the action is read-only, and the
      observation carries no candidate-generation channel.

    ``facts`` is keyed by ``parent_asin`` and holds only the whitelisted display attributes
    the reasoner exposes.
    """

    kind: str = "details"
    status: Literal["ok", "empty", "failed"] = "ok"
    facts: dict[str, dict[str, Any]] = Field(default_factory=dict)
    requested: tuple[str, ...] = ()
    unknown: tuple[str, ...] = ()
    verification_note: str | None = Field(default=None, max_length=280)


class ComparisonObservation(Observation):
    """Stage 4: a side-by-side comparison over grounded attributes.

    ``rows`` carries one entry per compared attribute with each product's value, using
    ``"UNKNOWN"`` where the catalogue is silent.  A row whose values are all unknown is still
    present: the absence of a shared fact is itself a finding.
    """

    kind: str = "comparison"
    status: Literal["ok", "empty", "failed"] = "ok"
    compared: tuple[str, ...] = ()
    rows: tuple[dict[str, Any], ...] = ()
    unsupported_attributes: tuple[str, ...] = ()
    verification_note: str | None = Field(default=None, max_length=280)


class TradeOffObservation(Observation):
    """Stage 4: an ordering by a stated priority, with every absence reported.

    ``order`` contains only the products that actually carry the fact.  ``unknown`` names the
    rest, so a policy can see that the ordering is partial rather than assuming it is total.
    ``supported`` is false when the catalogue cannot ground the priority at all, in which case
    ``order`` is empty and nothing was ordered.
    """

    kind: str = "trade_off"
    status: Literal["ok", "empty", "failed"] = "ok"
    priority: str = ""
    attribute: str | None = None
    supported: bool = False
    order: tuple[str, ...] = ()
    unknown: tuple[str, ...] = ()
    reason: str | None = Field(default=None, max_length=200)
    verification_note: str | None = Field(default=None, max_length=280)


class CompatibilityObservation(Observation):
    """Stage 4: a three-state compatibility verdict.

    ``unknown`` is a first-class outcome.  This catalogue carries no interface or fitment
    semantics, so a requirement naming one is unresolved rather than compatible - the
    observation never upgrades "we could not check" into "it fits".

    The single :attr:`verdict` is the **aggregate** over :attr:`members`; on its own it
    cannot say *which* candidate failed, which is why a policy could previously check a
    constraint and still be unable to tell a mixed pass/violation set from a uniform one.
    :attr:`assessments` therefore carries the per-candidate verdicts the aggregate was
    derived from, so the detail is not discarded at return.  Each entry is a
    :class:`~recommendation.control.grounded_reasoning.ConstraintReport` dumped to a mapping:
    identity, constraint kind, expected value, verdict and the observed value.
    """

    kind: str = "compatibility"
    status: Literal["ok", "empty", "failed"] = "ok"
    verdict: Literal["compatible", "incompatible", "unknown"] = "unknown"
    requirement_attribute: str = ""
    requirement_value: str = ""
    members: tuple[str, ...] = ()
    checks_performed: tuple[str, ...] = ()
    #: Per-candidate verdicts behind :attr:`verdict`, in ``members`` order.  Empty when the
    #: requirement could not be mapped onto a deterministic catalogue check at all, which is
    #: itself the honest answer rather than a fabricated per-candidate verdict.
    assessments: tuple[dict[str, Any], ...] = ()
    verification_note: str | None = Field(default=None, max_length=280)

    @property
    def violated_members(self) -> tuple[str, ...]:
        """The members proved to fail the requirement.  Never includes an unknown."""
        return tuple(
            str(entry.get("parent_asin", ""))
            for entry in self.assessments
            if entry.get("verdict") == "violated"
        )

    @property
    def unresolved_members(self) -> tuple[str, ...]:
        """The members the trusted catalogue could not decide for."""
        return tuple(
            str(entry.get("parent_asin", ""))
            for entry in self.assessments
            if entry.get("verdict") == "unknown"
        )

    @property
    def satisfied_members(self) -> tuple[str, ...]:
        """The members proved to satisfy the requirement."""
        return tuple(
            str(entry.get("parent_asin", ""))
            for entry in self.assessments
            if entry.get("verdict") == "satisfied"
        )


class EvidenceObservation(Observation):
    """Phase 4: what one evidence acquisition produced, minimised.

    Reports *whether* new trusted facts arrived and *which attributes* they covered - never the
    values, and never a verdict.  The values belong to the constraint report, where they are
    attached to the ``SATISFIED`` / ``VIOLATED`` / ``UNKNOWN`` they produced, so a reader sees a
    fact next to the decision it justified rather than loose in a trajectory.

    ``status`` uses the acquisition vocabulary from the evidence contract
    (``found`` / ``not_found`` / ``unsupported``), which is deliberately *not* the constraint
    verdict vocabulary: "we looked and there was nothing" must not be mistakable for "the
    constraint failed".
    """

    kind: str = "evidence"
    status: Literal["found", "not_found", "unsupported"] = "not_found"
    #: The candidate the evidence is about.
    parent_asin: str = ""
    #: The attribute the policy asked for.  Echoed so a trajectory shows the request even when the
    #: source had nothing.
    requested_attribute: str = ""
    #: Which trusted source answered, for provenance.
    source: str = ""
    #: Whether the source stated anything at all.
    found: bool = False
    #: How many facts the document carried.
    attribute_count: int = Field(default=0, ge=0)
    #: The attributes the document covered, in order.
    acquired_attributes: tuple[str, ...] = ()
    #: The attributes that were **new** to the run's evidence store.  Empty means the acquisition
    #: was a no-op, which is how the policy learns that asking again would change nothing.
    newly_acquired_attributes: tuple[str, ...] = ()
    #: The source's own provenance note for this record, when it carries one.
    provenance_note: str | None = Field(default=None, max_length=200)
    #: A payload-light view of the whole evidence store: counts, attribute names and provenance
    #: identities, with conflicts reported.  No fact values.
    evidence_state: dict[str, Any] = Field(default_factory=dict)

    @property
    def made_progress(self) -> bool:
        """True when this acquisition added a fact the run did not already hold."""
        return bool(self.newly_acquired_attributes)


class BundleObservation(Observation):
    """Stage 4: what a set of grounded products supports as a bundle.

    Aggregates are reported with their own status: ``grounded`` only when every member carries
    the fact, otherwise ``unknown`` with the blocking members named.  A partial sum is never
    reported as the total, and no cart, order or transaction concept exists here.
    """

    kind: str = "bundle"
    status: Literal["ok", "empty", "failed"] = "ok"
    members: tuple[str, ...] = ()
    member_count: int = Field(default=0, ge=0)
    total_price: dict[str, Any] = Field(default_factory=dict)
    total_weight_kg: dict[str, Any] = Field(default_factory=dict)
    shared_categories: tuple[str, ...] = ()
    verification_note: str | None = Field(default=None, max_length=280)


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

    #: Safe diagnostics about *who decided* this step, when the policy reports any.
    #:
    #: A model-driven policy records which policy was used, which action it proposed, whether
    #: parsing succeeded, how many model calls it took, and whether an answer needed
    #: correction - enough to answer "was this a model decision, and did it go well?" without
    #: putting a prompt, a model's chain-of-thought, or any product fact into the trajectory.
    #: A deterministic policy reports nothing here, which is itself the signal.
    policy_metadata: dict[str, Any] = Field(default_factory=dict)


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

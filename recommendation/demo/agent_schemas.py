"""Wire contract for the single-turn AgentRec-X recommendation endpoint.

Scope
-----
This module adds one endpoint that exposes the **existing** AgentRec-X runtime: the
bounded agent loop (:mod:`recommendation.control`) driven over the accepted
recommendation pipeline.  It adds no agent architecture, no planner, no new tool and no
new ranking rule.  It exists so an interviewer can send one HTTP request and inspect what
the measured system actually did.

Why these schemas are explicit rather than a dump of internal state
------------------------------------------------------------------
:class:`~recommendation.agent.state.AgentGraphState` is a mutable working area that carries
trusted behavioural history and raw domain payloads.  Serialising it would leak both, so
every response field here is an explicit whitelist, ``extra="forbid"``, and the trajectory
is a **projection** (see :mod:`recommendation.demo.agent_trajectory`) rather than a
pass-through of the controller's record.

Reused internal models
----------------------
The provenance block is the control plane's own
:class:`~recommendation.control.CandidateProvenance` model rather than a re-modelled twin,
because a second definition of provenance would be a second source of truth; it already
labels every score with its ``score_kind`` so two sources' scores cannot be compared.

The eligibility block is the control plane's own ``as_dict`` output, kept as dictionaries
on purpose: ``ConstraintRequirement``, ``CandidateConstraintAssessment`` and
``FeasibilityProjection`` are dataclasses with their own deliberate serialisation (verdicts
stay three-state ``match``/``violation``/``unknown``, observed values included).  Pydantic
cannot generate a core schema for them, and re-modelling them here would fork the meaning of
"eligible" - so the contract carries their JSON form unchanged.

Two separated kinds of quality
------------------------------
The response reports the ``route``, the ``terminal`` block, the ``trajectory`` and the
``timing`` of one *agent run*, and it reports ``recommendations`` produced by the
**existing** measured ranking pipeline.  Nothing in this contract scores recommendation
quality; the measured Recall/NDCG/HR comparison lives in the Phase-5 benchmark
(``docs/PHASE5_HANDOFF.md``) and is deliberately not re-computed per request.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from recommendation.api.schemas import MAX_K, MIN_K
from recommendation.control.candidate_ledger import (
    CandidateProvenance,
    CandidateVerificationStatus,
)
from recommendation.preference_matching.schemas import PreferenceEvidence

__all__ = [
    "AGENT_API_VERSION",
    "AGENT_OUTCOMES",
    "MAX_CONSTRAINTS",
    "MAX_HISTORY_ITEMS",
    "AgentCandidateView",
    "AgentConstraintItem",
    "AgentEligibilityView",
    "AgentExecutionView",
    "AgentObservationView",
    "AgentOutcome",
    "AgentRecommendationView",
    "AgentRecommendRequest",
    "AgentRecommendResponse",
    "AgentShoppingRequest",
    "AgentPreferenceEvidenceView",
    "AgentPreferenceOptionView",
    "AgentPreferenceOutcomeView",
    "AgentReplanView",
    "AgentResearchActionView",
    "AgentResearchClarificationView",
    "AgentResearchOptionView",
    "AgentResearchView",
    "ReplanDecisionView",
    "ReplanObservationView",
    "AgentTerminalView",
    "AgentTimingView",
    "GroundedCandidateView",
    "TrajectoryStepView",
    "TrajectorySummaryView",
]

#: Version of this wire contract.  Separate from ``DEMO_API_VERSION`` because the session
#: contract and the single-turn agent contract evolve independently.
AGENT_API_VERSION = "1.0.0"

#: Upper bound on a supplied history.  Bounded for the same reason the chat message is: one
#: request must not carry an unbounded amount of input.  This is a *request* bound, not a
#: protocol bound - the accepted split and the model window are unaffected by it.
MAX_HISTORY_ITEMS = 500

#: Upper bound on stated hard constraints.  Bounded so one request cannot carry an unbounded
#: constraint set; the accepted evaluator does one fact lookup per candidate per constraint.
MAX_CONSTRAINTS = 8

#: A non-blank, bounded user message.  Same shape as the chat contract's message.
Message = Annotated[str, Field(min_length=1, max_length=2000, strict=True)]

#: A non-blank opaque identifier (a ``parent_asin``, a source name, a run id).
OpaqueId = Annotated[str, Field(min_length=1, max_length=200)]

#: Strict, accepted recommendation count bound (reused from the Milestone 6 contract).
StrictK = Annotated[int, Field(ge=MIN_K, le=MAX_K, strict=True)]


# --------------------------------------------------------------------------- #
# request
# --------------------------------------------------------------------------- #


class AgentConstraintItem(BaseModel):
    """One hard constraint the caller states for this run.

    ``kind`` is the accepted preference ontology
    (:class:`~recommendation.memory.PreferenceKind`) - the same vocabulary preference memory
    already uses - and ``value`` is the caller's own stated value.  Nothing here interprets
    the value: the control plane narrows the kind onto a checkable catalogue dimension and
    returns ``unknown`` when the trusted catalogue cannot decide it, which keeps a
    stated-but-untestable constraint active instead of silently satisfied.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: str = Field(
        ...,
        description="Constraint dimension, e.g. price_max, brand, category, color, material.",
    )
    value: str = Field(..., min_length=1, max_length=120, strict=True)

    @field_validator("kind")
    @classmethod
    def _strip_kind(cls, value: str) -> str:
        stripped = value.strip().lower()
        if not stripped:
            raise ValueError("kind must not be blank")
        return stripped

    @field_validator("value")
    @classmethod
    def _strip_value(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("value must not be blank")
        return stripped


class AgentRecommendRequest(BaseModel):
    """Body of ``POST /v1/demo/agent/recommend``.

    ``history`` is the caller's own trusted history.  It is supplied explicitly because this
    endpoint is **stateless**: it creates no session and touches no stored preference memory
    namespace unless the caller also names one.  That keeps the endpoint honest - there is no
    hidden per-caller state - and it keeps the request the only channel through which
    history can arrive, exactly as the accepted ``POST /v1/recommend`` does for the model
    path.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    message: Message = Field(
        ...,
        description="Natural-language shopping message. Untrusted text.",
        examples=["Recommend some hiking gear."],
    )
    history: Annotated[list[OpaqueId], Field(min_length=1, max_length=MAX_HISTORY_ITEMS)] = Field(
        ...,
        description=(
            "Chronological parent_asin history the run may recommend from. Duplicates are "
            "real repeated interactions and are preserved."
        ),
        examples=[["B000000001", "B000000002"]],
    )
    k: StrictK = Field(
        default=5,
        description=f"Number of recommendations to request, {MIN_K}..{MAX_K} (strict integer).",
    )
    user_key: OpaqueId | None = Field(
        default=None,
        description=(
            "Optional server-side preference-memory namespace. Omit it and the run reads and "
            "writes no preference memory at all, which is the default and the reproducible mode."
        ),
    )
    clarification_id: OpaqueId | None = Field(
        default=None,
        description=(
            "Optional id of a pending clarification this turn answers. Only meaningful on the "
            "multi-turn shopping turn; a normal request leaves it null."
        ),
    )
    clarification_option: OpaqueId | None = Field(
        default=None,
        description=(
            "Optional id of a structured option on the pending clarification. Used only with "
            "clarification_id; the option tells the resumed run which read-only research step "
            "to take (compare another attribute, show known facts, or stop research)."
        ),
    )
    sources: tuple[str, ...] = Field(
        default=("history",),
        description=(
            "Which trusted candidate sources this run consults, in order. 'history' is the "
            "accepted sequential recommender; 'catalog_search' is full-catalogue lexical "
            "retrieval over normalized catalogue text. Every named source is consulted and the "
            "control plane's rank fusion orders the union - the request chooses the sources "
            "because an adaptive selector was measured to underperform fusing them "
            "(docs/PHASE5_HANDOFF.md)."
        ),
        examples=[["history", "catalog_search"]],
    )
    constraints: Annotated[tuple[AgentConstraintItem, ...], Field(max_length=MAX_CONSTRAINTS)] = (
        Field(
            default=(),
            description=(
                "Optional hard constraints. An empty tuple means no hard constraint is active, "
                "which the response distinguishes from 'constraints active but not evaluable'."
            ),
            examples=[[{"kind": "price_max", "value": "120"}]],
        )
    )
    preference_discovery: bool = Field(
        default=False,
        description=(
            "Opt in to candidate-aware preference construction.  When true, a shopping turn "
            "may suspend once on a trusted attribute difference between its own candidates.  "
            "Default false keeps the existing shopping/recommendation behaviour unchanged."
        ),
    )

    @field_validator("message")
    @classmethod
    def _strip_message(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("message must not be blank")
        return stripped

    @field_validator("history")
    @classmethod
    def _strip_history(cls, value: list[str]) -> list[str]:
        stripped = [item.strip() for item in value]
        if not stripped or any(not item for item in stripped):
            raise ValueError("history must contain no blank identifier")
        return stripped

    @field_validator("user_key")
    @classmethod
    def _strip_user_key(cls, value: str | None) -> str | None:
        if value is None:
            return None
        stripped = value.strip()
        if not stripped:
            raise ValueError("user_key must not be blank when supplied")
        return stripped


class AgentShoppingRequest(BaseModel):
    """Body of ``POST /v1/demo/sessions/{session_id}/agent/shopping``.

    The session owns the trusted interaction history and the preference-memory namespace,
    so this body deliberately carries **no** history and no ``user_key``: a caller cannot
    point a shopping turn at another namespace.  ``message``, ``k``, ``sources`` and
    ``constraints`` are validated exactly like the single-turn agent request; the only
    additional field is the optional ``clarification_id`` that answers a pending question.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    message: Message = Field(
        ...,
        description="Natural-language shopping message for this turn. Untrusted text.",
        examples=["I need a yoga mat, budget $30"],
    )
    k: StrictK = Field(
        default=5,
        description=f"Number of recommendations to request, {MIN_K}..{MAX_K} (strict integer).",
    )
    sources: tuple[str, ...] = Field(
        default=("history", "catalog_search"),
        description=(
            "Which trusted candidate sources this turn consults, in order. Defaults to the "
            "fusion the deployment measured; every named source is consulted and fused."
        ),
        examples=[["history", "catalog_search"]],
    )
    constraints: Annotated[tuple[AgentConstraintItem, ...], Field(max_length=MAX_CONSTRAINTS)] = (
        Field(
            default=(),
            description=(
                "Optional hard constraints for this turn. They are also projected into the "
                "shopping intent and checked by the deterministic eligibility layer."
            ),
        )
    )
    clarification_id: OpaqueId | None = Field(
        default=None,
        description=(
            "The id of a pending clarification this turn answers. Omit it for a new shopping "
            "turn; an unknown, expired or cross-session id is rejected."
        ),
    )
    clarification_option: OpaqueId | None = Field(
        default=None,
        description=(
            "Optional structured option id from the pending clarification. Only valid together "
            "with clarification_id; unknown option ids are rejected without consuming the "
            "pending clarification."
        ),
    )
    preference_discovery: bool = Field(
        default=False,
        description=(
            "Opt in to candidate-aware preference construction for this shopping session.  "
            "The prompt is presented as evidence, never as a final recommendation list."
        ),
    )


# --------------------------------------------------------------------------- #
# response blocks
# --------------------------------------------------------------------------- #


class AgentRecommendationView(BaseModel):
    """One recommendation, in the exact order the backend produced it."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    rank: int = Field(..., ge=1, description="1-based position in the ranked candidate set.")
    item_id: int = Field(..., ge=1, description="Internal model item id; PAD (0) is never returned.")
    parent_asin: OpaqueId = Field(..., description="External Amazon parent_asin identity.")
    score: float = Field(
        ...,
        description=(
            "The score this identity's ranking source reported for it. NOT a probability, "
            "confidence or purchase likelihood; only meaningful for ordering, and only "
            "comparable with another score that carries the same ``score_kind``."
        ),
    )
    score_kind: str | None = Field(
        default=None,
        description=(
            "What ``score`` measures (for example ``sasrec_logit`` or ``bm25``), or null when the "
            "identity's source reported no score. Present precisely so two sources' scores are "
            "never mistaken for one scale."
        ),
    )
    title: str | None = Field(
        default=None, description="Normalised catalogue title, or null when absent."
    )
    reason: str | None = Field(
        default=None,
        description=(
            "Deterministic, evidence-bound reason for this product, derived from the current "
            "shopping intent and this product's own catalogue record. Null when no shopping "
            "intent was available. Every factual clause is traceable to the record."
        ),
    )
    sources: tuple[str, ...] = Field(
        default=(),
        description=(
            "Every trusted candidate source that proposed this identity, in first-seen order. "
            "This is provenance, not a score."
        ),
    )
    # -- memory-aware ranking (E-02A): optional, and absent unless it actually ran ---------- #
    reranked_rank: int | None = Field(
        default=None,
        ge=1,
        description=(
            "1-based position under the accepted M10B preference key, or null when no memory-aware "
            "ranking ran. ``rank`` keeps the run's baseline (fused) position, so the two orders "
            "stay distinguishable; the list itself is emitted in ``reranked_rank`` order."
        ),
    )
    match_count: int = Field(default=0, ge=0)
    violation_count: int = Field(default=0, ge=0)
    unknown_count: int = Field(default=0, ge=0)
    evidence: tuple[PreferenceEvidence, ...] = Field(
        default=(),
        description=(
            "Real M10A evidence for this identity: one record per ACTIVE preference, each with its "
            "three-state verdict, the catalogue field it read and its reason code. Empty when no "
            "memory-aware ranking ran, and an undecidable preference is reported as 'unknown' - "
            "never as a match or a violation."
        ),
    )
    rerank_reason: str | None = Field(
        default=None,
        description="M10B's own machine-readable reason for this candidate's placement, if any.",
    )


class AgentObservationView(BaseModel):
    """What one finished run's own record proves about tools, candidates and budget (E-03A).

    Every field names a record the controller wrote: the terminal block for the verdict and the
    budgets, the trajectory for what each source did, the grounding block for what was verified,
    and the request/final list for ``k`` and the presented count.  Nothing here is inferred from
    prose, from a product title or from a model's own account, and no field scores relevance: the
    observation counts products, it does not judge them.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    run_ref: str = Field(..., description="The controller's run identity; run-local and opaque.")
    requested_k: int = Field(..., ge=0, description="The k this run was asked for.")
    plan: tuple[str, ...] = Field(default=(), description="The sources the authorised plan named.")
    executed_sources: tuple[str, ...] = Field(
        default=(), description="The trusted candidate sources this run actually consulted."
    )
    unexecuted_sources: tuple[str, ...] = Field(
        default=(),
        description="Sources this deployment offers that the run did not consult - the only ones a "
        "later retrieval could legally use.",
    )
    source_calls: dict[str, int] = Field(
        default_factory=dict,
        description="How often each source was consulted.  An action that cannot name its source "
        "is reported under its own action name rather than guessed at.",
    )
    source_outcomes: dict[str, str] = Field(
        default_factory=dict,
        description="Each consulted source's outcome: 'candidates', 'empty' (it ran and returned "
        "nothing) or 'failed' (a validation/verification failure or a failure observation).",
    )
    failed_sources: tuple[str, ...] = ()
    empty_sources: tuple[str, ...] = ()
    grounded_count: int = Field(
        default=0, ge=0,
        description="Products the run verified.  The ledger is identity-keyed, so this is already "
        "de-duplicated across sources.",
    )
    ungrounded_count: int = Field(
        default=0, ge=0,
        description="Identities a source returned that the trusted mapping does not know.  They are "
        "never candidates.",
    )
    multi_source_count: int = Field(
        default=0, ge=0, description="Grounded products more than one source proposed."
    )
    presented_count: int = Field(
        default=0, ge=0, description="Products the final answer actually presents."
    )
    candidates_sufficient: bool = Field(
        default=False,
        description="True when the answer presents at least the requested k.  A count, never a "
        "claim about relevance.",
    )
    termination_status: str = ""
    termination_reason: str | None = None
    steps_used: int = Field(default=0, ge=0)
    steps_limit: int = Field(default=0, ge=0)
    tool_calls_used: int = Field(default=0, ge=0)
    tool_calls_limit: int = Field(default=0, ge=0)
    retries_used: int = Field(default=0, ge=0)
    remaining_steps: int = Field(default=0, ge=0)
    remaining_tool_calls: int = Field(default=0, ge=0)
    has_further_budget: bool = False
    eligibility_evaluated: bool = Field(
        default=False,
        description="True when a hard constraint was active and verdicts were computed.  False "
        "means 'not evaluated', never 'everything passed'.",
    )
    eligible_candidate_count: int = Field(default=0, ge=0)


class ReplanDecisionView(BaseModel):
    """The gate's decision for one finished run (E-03A): whether to retrieve again, and why.

    This round only decides.  A ``continue_recommended`` answer is a bounded suggestion - the run
    has budget and a legal source it has not consulted - and it never widens a budget, invents a
    source, or overrides a refusal.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    continue_recommended: bool
    reason_code: str = Field(..., description="One code from the gate's closed vocabulary.")
    eligible_sources: tuple[str, ...] = ()
    remaining_tool_budget: int = Field(default=0, ge=0)
    remaining_step_budget: int = Field(default=0, ge=0)
    decision_source: str = Field(
        default="deterministic_rules",
        description="Who decided.  Only the deterministic rules exist in this milestone.",
    )
    explanation: str | None = Field(
        default=None, max_length=280, description="A factual sentence naming the reason only."
    )


class ReplanObservationView(BaseModel):
    """A compact before/after snapshot of one run's candidate state (E-03B).

    The *initial* snapshot is taken when the replan decision is made (inside the run, before any
    appended source), the *final* one after the run finished - so "what the first sources found" and
    "what the request ended with" can never be read as the same number.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    candidates: int = Field(default=0, ge=0, description="Verified candidates at that moment.")
    sources_consulted: tuple[str, ...] = ()
    source_outcomes: dict[str, str] = Field(
        default_factory=dict,
        description="Each consulted source's outcome: 'candidates', 'empty' or 'failed'.",
    )
    remaining_tool_calls: int = Field(default=0, ge=0)
    remaining_steps: int = Field(default=0, ge=0)


class AgentReplanView(BaseModel):
    """What bounded replanning actually did in one run (E-03B).

    A decision *and* its execution, distinguished by :attr:`status` from the closed vocabulary:
    the run never needed it, the deterministic gate blocked it, the model declined, the model's
    answer was invalid or its call failed, or a source was appended and produced candidates, none,
    or a failure.  ``model_called`` is what makes "asked and declined" different from "never asked".
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    status: str = Field(..., description="One code from the replanning status vocabulary.")
    reason_code: str = ""
    model_called: bool = False
    chosen_source: str | None = None
    eligible_sources: tuple[str, ...] = ()
    decision_step: int | None = Field(
        default=None, description="The step index at which the one decision was taken."
    )
    initial: ReplanObservationView = Field(default_factory=ReplanObservationView)
    final: ReplanObservationView = Field(default_factory=ReplanObservationView)


class AgentCandidateView(BaseModel):
    """One candidate identity the run grounded, with its full provenance."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    parent_asin: OpaqueId
    grounded: bool = Field(
        ..., description="True when the trusted identity map confirmed this product exists."
    )
    verification_status: CandidateVerificationStatus
    provenance: tuple[CandidateProvenance, ...] = Field(
        default=(),
        description=(
            "One record per source sighting: source, that source's own rank, its own raw score "
            "and the score's kind. Scores from different sources are never comparable."
        ),
    )


class GroundedCandidateView(BaseModel):
    """The run's grounded candidate universe, as counts plus identities."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    grounded_count: int = Field(..., ge=0)
    ungrounded_count: int = Field(
        ..., ge=0, description="Identities a source returned that the mapping does not know."
    )
    multi_source_count: int = Field(
        ..., ge=0, description="Identities more than one independent source proposed."
    )
    sources_present: tuple[str, ...] = Field(
        default=(), description="Distinct trusted sources that contributed a grounded identity."
    )
    candidates: tuple[AgentCandidateView, ...] = Field(default=())


class AgentEligibilityView(BaseModel):
    """Task-scoped constraint eligibility, with its three-state verdicts intact.

    ``evaluated`` is the honest distinction between "no hard constraint was active" and
    "every candidate was proved eligible".  When it is false the other blocks are empty
    because nothing was evaluated - not because everything passed.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    evaluated: bool
    requirements: tuple[dict[str, object], ...] = Field(
        default=(),
        description=(
            "The active hard constraints, each as {kind, expected}. Empty when nothing was "
            "constrained."
        ),
    )
    verified_eligible_count: int = Field(default=0, ge=0)
    ineligible_count: int = Field(default=0, ge=0)
    unresolved_count: int = Field(default=0, ge=0)
    assessments: tuple[dict[str, object], ...] = Field(
        default=(),
        description=(
            "Per-candidate verdicts as {parent_asin, eligibility, reports}, where each report "
            "carries its own three-state verdict and observed value so an exclusion is "
            "explainable rather than inferred from a boolean."
        ),
    )
    projection: dict[str, object] | None = Field(
        default=None,
        description=(
            "The narrowed set a compliant recommendation may be built from, as "
            "{presented, excluded, ...}, or null when no hard constraint was active."
        ),
    )


class AgentTerminalView(BaseModel):
    """Why the run stopped, in the controller's own vocabulary."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    run_id: OpaqueId
    status: str = Field(..., description="Terminal run status recorded by the controller.")
    termination_reason: str | None = Field(
        default=None, description="Controller-owned reason, or null while/if never terminated."
    )
    termination_detail: str | None = Field(default=None, max_length=280)
    succeeded: bool = Field(
        ...,
        description=(
            "True only for a run that finished through an accepted completion. A refusal, an "
            "exhausted budget and a backend failure all report false."
        ),
    )
    steps: int = Field(..., ge=0)
    tool_calls: int = Field(..., ge=0)
    retries: int = Field(..., ge=0)
    limits: dict[str, int] = Field(
        default_factory=dict, description="The budgets this run was allowed."
    )


class TrajectoryStepView(BaseModel):
    """One step of the run, projected for debugging and demo inspection.

    Exactly the six fields the packaging phase asked for, plus the proposal's own action
    and the refusal codes.  A projection rather than a dump: no prompt, no product facts,
    no trusted history, no reasoning transcript.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    step: int = Field(..., ge=0)
    action: str | None = Field(
        default=None, description="The action the policy proposed, or null if it proposed none."
    )
    validated: bool = Field(
        ..., description="True when the action passed the validator and was executed."
    )
    observation_type: str | None = Field(
        default=None,
        description=(
            "The type name of the observation the step produced, or null when the step produced "
            "none. It is the observation's class, never its payload."
        ),
    )
    candidate_count: int = Field(
        default=0, ge=0, description="Candidates this step observed, 0 when it observed none."
    )
    eligibility_state: str | None = Field(
        default=None,
        description=(
            "The eligibility identifier the step's context exposed, or null when no hard "
            "constraint was active."
        ),
    )
    refusal_codes: tuple[str, ...] = Field(
        default=(), description="Validation/verification refusal codes recorded for this step."
    )
    note: str | None = Field(default=None, max_length=280)


class TrajectorySummaryView(BaseModel):
    """A compact, ordered view of one run's control decisions."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    step_count: int = Field(..., ge=0)
    actions: tuple[str, ...] = Field(
        default=(), description="The proposed action kinds, in execution order."
    )
    refusals: tuple[str, ...] = Field(
        default=(), description="Every refusal code the run recorded, in order."
    )
    steps: tuple[TrajectoryStepView, ...] = Field(default=())


class AgentTimingView(BaseModel):
    """Wall-clock cost of this request, split into the parts a reviewer cares about."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    total_ms: float = Field(..., ge=0.0)
    agent_ms: float = Field(
        ..., ge=0.0, description="Wall time inside the agent runtime (loop plus tools)."
    )
    serialize_ms: float = Field(..., ge=0.0, description="Wall time building this response.")


#: What one agent-backed turn turned out to be.  A closed set, because "no products in this
#: response" has three different causes and a caller must be able to tell them apart instead of
#: reading every one of them as a successful direct answer:
#:
#: ``direct``
#:     the decision policy chose a direct turn; no controller ran at all;
#: ``recommended``
#:     a controller ran and the candidates it produced are being presented;
#: ``no_candidates``
#:     a controller ran, finished, and had nothing presentable;
#: ``refused``
#:     a controller ran and its completion guard refused to finish the run
#:     (``TerminationReason.COMPLETION_REFUSED``: nothing grounded, or the last execution failed);
#: ``failed``
#:     a controller ran and did not complete and was not refused - a capability failure, an
#:     invalid action or an exhausted budget.  The terminal block carries the controller's own
#:     precise reason, so this label never has to stand in for a taxonomy.
AGENT_OUTCOMES: tuple[str, ...] = (
    "direct",
    "recommended",
    "no_candidates",
    "refused",
    "failed",
)

AgentOutcome = Literal["direct", "recommended", "no_candidates", "refused", "failed"]


class AgentExecutionView(BaseModel):
    """What one chat turn's agent execution actually did, restricted to safe, already-public facts.

    This is the chat-side home of the execution summary
    :func:`~recommendation.demo.agent_chat_adapter.agent_execution_digest` produces for callers
    that log or assert.  It exists because ``route`` alone cannot describe an agent turn: a
    recommendation the controller could not complete and a turn the policy decided needed no
    products both present an empty list, and only the run's own record tells them apart.

    Every field is copied from the executed run - the planned names are the validated plan's own,
    ``terminal``/``trajectory``/``grounded``/``timing`` are the very views
    ``POST /v1/demo/agent/recommend`` already publishes - so this block can never disagree with
    the run, and it widens no boundary: no prompt, no model prose, no stack trace, no raw memory,
    no internal object.  ``action_counts`` is derived here (and only here) from the trajectory's
    *validated* steps, which is what makes "which sources ran, and how many times" answerable
    without re-deriving it from a log.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    outcome: AgentOutcome = Field(
        ..., description="Which of the closed outcomes above this turn actually was."
    )
    planned_sources: tuple[str, ...] = Field(
        default=(),
        description=(
            "The candidate sources the validated decision plan named, in the plan's order. The "
            "names the model chose - not a default list, and not a claim that they produced "
            "anything (see grounded.sources_present for what did)."
        ),
    )
    controller_executed: bool = Field(
        ...,
        description=(
            "True only when the bounded recommendation controller actually ran. False for a "
            "direct turn, which is the one case where no tool was executed at all."
        ),
    )
    action_counts: dict[str, int] = Field(
        default_factory=dict,
        description=(
            "How many times each action kind was **executed** (validated steps only), so a source "
            "the controller legally consulted twice is visible as two. A proposal the validator "
            "refused is not counted as an execution."
        ),
    )
    terminal: AgentTerminalView | None = Field(
        default=None, description="The controller's own terminal record; null when none ran."
    )
    trajectory: TrajectorySummaryView | None = Field(
        default=None,
        description=(
            "The run's projected control record: proposed action kinds, their counts, the "
            "observation types and the refusal codes. Never a reasoning transcript."
        ),
    )
    grounded: GroundedCandidateView | None = Field(
        default=None,
        description=(
            "The run's grounded candidate universe with per-source provenance (source, that "
            "source's own rank, its own raw score and score kind). Null when no controller ran."
        ),
    )
    timing: AgentTimingView | None = Field(
        default=None, description="The run's wall-clock split; null when no controller ran."
    )


class AgentIntentConstraintView(BaseModel):
    """One active entry of the current shopping intent."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: str
    value: str
    polarity: str
    strength: Literal["hard", "soft"]
    source_turn_id: str = ""


class AgentIntentView(BaseModel):
    """The deterministic current intent a shopping turn was executed against."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    hard: tuple[AgentIntentConstraintView, ...] = ()
    soft: tuple[AgentIntentConstraintView, ...] = ()
    conflicts: tuple[str, ...] = ()
    turn_ids: tuple[str, ...] = ()
    active_preference_count: int = Field(default=0, ge=0)


class AgentResearchOptionView(BaseModel):
    """One executable choice attached to a research clarification.

    The option is derived from a real Observation: ``compare_attributes`` only appears for
    attributes the comparison actually marked comparable; ``known_facts`` only appears when
    at least one catalogue value was verified; ``stop_research`` is always safe.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    option_id: str
    label: str
    action: Literal["compare_attributes", "show_known_facts", "stop_research"]
    attributes: tuple[str, ...] = ()
    products: tuple[str, ...] = ()


class AgentResearchClarificationView(BaseModel):
    """The structured research context behind a pending question."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    reason: str
    missing_attributes: tuple[str, ...] = ()
    products: tuple[str, ...] = ()
    known_facts: dict[str, dict[str, Any]] = Field(default_factory=dict)
    inspected_products: tuple[str, ...] = ()
    compared: bool = False
    options: tuple[AgentResearchOptionView, ...] = ()


class AgentPreferenceOptionView(BaseModel):
    """One executable option in a candidate-aware preference question."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    option_id: str
    label: str
    action: Literal["lower", "higher", "value", "none", "skip"]
    attribute: str
    value: str | float | None = None


class AgentPreferenceEvidenceView(BaseModel):
    """The trusted candidate comparison behind a preference question.

    This is deliberately separate from ``recommendations``: a pending preference question
    presents evidence, not a final list.  Values are copied from the trusted catalogue
    comparison; the field never asserts a missing or title-inferred value.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    attribute: str
    label: str
    reason: str
    products: tuple[str, ...]
    values: dict[str, Any] = Field(default_factory=dict)
    comparison_kind: Literal["numeric", "categorical"]
    missing: tuple[str, ...] = ()
    options: tuple[AgentPreferenceOptionView, ...] = ()
    interaction: Literal["ASK_DIRECT", "SHOW_CONTRAST", "EXPLAIN"] = "SHOW_CONTRAST"
    status: Literal["known", "confirmed", "unconfirmed", "revoked"] = "unconfirmed"
    impact: float = 0.0
    baseline_top_k: tuple[str, ...] = ()
    counterfactual: dict[str, tuple[str, ...]] = Field(default_factory=dict)
    explanation: str | None = None


class AgentPreferenceOutcomeView(BaseModel):
    """Causal check between the selected preference and the final top-K."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    selected_option_id: str
    attribute: str
    before_top_k: tuple[str, ...] = ()
    expected_top_k: tuple[str, ...] = ()
    actual_top_k: tuple[str, ...] = ()
    changed: bool = False
    verified: bool = False


class AgentClarificationView(BaseModel):
    """A pending question the caller can answer on the next shopping turn."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    clarification_id: str
    question: str
    kind: str = "unverified"
    blocks: str = "candidate_set"
    options: tuple[str, ...] = ()
    reason: str | None = None
    missing_attributes: tuple[str, ...] = ()
    products: tuple[str, ...] = ()
    known_facts: dict[str, dict[str, Any]] = Field(default_factory=dict)
    research_options: tuple[AgentResearchOptionView, ...] = ()
    preference: AgentPreferenceEvidenceView | None = None


class AgentResearchActionView(BaseModel):
    """One Phase 6B product-research action, with its real input and result.

    Facts/rows are copied from the trusted observations.  The view is deliberately separate
    from ``message``: research can be visible to the caller without changing the final
    recommendation text or its product order.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    tool: Literal["inspect_product", "compare_products"]
    status: str
    products: tuple[str, ...] = ()
    attributes: tuple[str, ...] = ()
    missing_attributes: tuple[str, ...] = ()
    unsupported_attributes: tuple[str, ...] = ()
    code: str | None = None
    facts: dict[str, dict[str, Any]] = Field(default_factory=dict)
    rows: tuple[dict[str, Any], ...] = ()


class AgentResearchView(BaseModel):
    """What product research actually did in one run, and why it is visible."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    executed: bool = False
    actions: tuple[AgentResearchActionView, ...] = ()
    note: str = ""


class AgentRecommendResponse(BaseModel):
    """Body of a successful ``POST /v1/demo/agent/recommend``."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    api_version: str = AGENT_API_VERSION
    route: Literal["direct", "recommend"] = Field(
        ...,
        description=(
            "Which route the decision step chose. 'direct' means no recommendation work ran, so "
            "recommendations is empty - that is the honest outcome, not a failure."
        ),
    )
    message: str = Field(..., description="The agent's rendered, grounded response text.")
    control_plane: str = Field(
        ...,
        description=(
            "Which control plane served this request. Recorded because the two are not "
            "interchangeable claims."
        ),
    )
    recommendations: tuple[AgentRecommendationView, ...] = Field(default=())
    grounded: GroundedCandidateView
    eligibility: AgentEligibilityView
    terminal: AgentTerminalView
    trajectory: TrajectorySummaryView
    observation: AgentObservationView | None = Field(
        default=None,
        description=(
            "What this run's own execution record proves about tools, candidates and budget, with "
            "the deterministic replan decision (E-03A).  Present on every executed run."
        ),
    )
    research: AgentResearchView | None = Field(
        default=None,
        description=(
            "Phase 6B product-research actions actually executed, with their facts, missing "
            "attributes and failures.  Absent when the request did not justify research."
        ),
    )
    preference: AgentPreferenceEvidenceView | None = Field(
        default=None,
        description=(
            "Candidate-aware preference evidence when the shopping agent asks one bounded "
            "question about a real catalogue difference.  Evidence, not a final list."
        ),
    )
    preference_outcome: AgentPreferenceOutcomeView | None = Field(
        default=None,
        description=(
            "The counterfactual top-K check for a preference answer, when this turn applied "
            "one.  Present only after an actual user confirmation."
        ),
    )
    replan: ReplanDecisionView | None = Field(
        default=None,
        description=(
            "Whether another retrieval is worth attempting, from the run's own record alone.  The "
            "terminal gate's decision: it never executes anything (E-03A)."
        ),
    )
    replan_execution: AgentReplanView | None = Field(
        default=None,
        description=(
            "What bounded replanning did during the run, present only when the capability is "
            "enabled (E-03B): the initial observation, the decision, and the appended source's "
            "real outcome."
        ),
    )
    clarification: AgentClarificationView | None = Field(
        default=None,
        description=(
            "Present when this run suspended to ask the user a question. The question is "
            "also the message; answering it on the next shopping turn resumes the task."
        ),
    )
    intent: AgentIntentView | None = Field(
        default=None,
        description="The current shopping intent this turn was executed against, when available.",
    )
    timing: AgentTimingView = Field(
        default_factory=lambda: AgentTimingView(total_ms=0.0, agent_ms=0.0, serialize_ms=0.0)
    )

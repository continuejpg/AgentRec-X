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

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from recommendation.api.schemas import MAX_K, MIN_K
from recommendation.control.candidate_ledger import (
    CandidateProvenance,
    CandidateVerificationStatus,
)

__all__ = [
    "AGENT_API_VERSION",
    "MAX_CONSTRAINTS",
    "MAX_HISTORY_ITEMS",
    "AgentConstraintItem",
    "AgentCandidateView",
    "AgentEligibilityView",
    "AgentRecommendationView",
    "AgentRecommendRequest",
    "AgentRecommendResponse",
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
    sources: tuple[str, ...] = Field(
        default=(),
        description=(
            "Every trusted candidate source that proposed this identity, in first-seen order. "
            "This is provenance, not a score."
        ),
    )


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
    timing: AgentTimingView = Field(
        default_factory=lambda: AgentTimingView(total_ms=0.0, agent_ms=0.0, serialize_ms=0.0)
    )

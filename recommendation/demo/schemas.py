"""External HTTP and session schemas for the Milestone 11 multi-turn web demo.

This module defines the **public wire contract** only.  It is deliberately separate
from :class:`~recommendation.agent.state.AgentGraphState`: the browser never sees
graph state, and no graph state is ever serialized blindly.  Every response model is
an explicit whitelist of fields a demo client may read.

Design rules enforced here
--------------------------
* ``extra="forbid"`` everywhere, so a browser cannot inject internal fields.  There is
  no request field for ``trusted_user_history``, ``parent_asin`` history,
  ``preference_snapshot``, a reranking report or a memory ``user_key`` -- all of those
  are server-owned.
* request bodies are additionally ``frozen=True``; they are immutable values, not
  mutable session state.
* ``k`` is **strict**, so ``"5"`` or ``5.0`` are rejected rather than coerced, and it
  reuses the accepted Milestone 6/7A bounds.
* evidence ``status`` is passed through as the accepted Milestone 10A three-state
  value (``match`` / ``violation`` / ``unknown``).  It is never collapsed to a boolean
  and ``unknown`` is never rewritten as a negative fact.
* the raw SASRec ``sasrec_score`` is documented as a ranking score, never as a
  probability, confidence, rating or preference score.

Nothing here computes, orders, filters or scores anything.
"""

from __future__ import annotations

from typing import Any, Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

# The agent run's execution view lives with the agent response it belongs to; this module
# re-exports it for its long-standing importers (``serialization``, ``demo/__init__``) instead of
# redefining it, so there is exactly one definition of the execution record.
from .agent_schemas import AgentExecutionView

from recommendation.api.schemas import MAX_K, MIN_K

__all__ = [
    "DECISION_MODES",
    "DECISION_MODE_DETERMINISTIC",
    "DECISION_MODE_LLM",
    "DEFAULT_DECISION_MODE",
    "DEMO_API_VERSION",
    "MAX_MESSAGE_LENGTH",
    "ActivePreferenceView",
    "AuditView",
    "ChatRequest",
    "ChatResponse",
    "CreateSessionRequest",
    "DecisionMode",
    "AgentExecutionView",
    "DecisionModeView",
    "DecisionModesResponse",
    "DemoHealthResponse",
    "DemoProfileView",
    "EvidenceView",
    "MemoryUpdateView",
    "PreferenceMutationView",
    "ProductMetadataView",
    "ProfileListResponse",
    "RecommendationCard",
    "ResetResponse",
    "SessionResponse",
    "SessionStateResponse",
    "TracePreferenceAction",
]

#: Version of the demo wire contract, published in session responses.
DEMO_API_VERSION = "1.0.0"

#: The decision modes this wire contract accepts.  ``deterministic`` is the default and needs no
#: credential, so an existing client that never sends the field keeps exactly its old behaviour.
DECISION_MODE_DETERMINISTIC = "deterministic"
DECISION_MODE_LLM = "llm"
#: The agent mode: the request is planned by the model, validated, then executed by the controller
#: (the E-01A path).  Additive: a client that never sends it keeps its previous behaviour.
DECISION_MODE_LLM_AGENT = "llm_agent"
DECISION_MODES: tuple[str, ...] = (
    DECISION_MODE_DETERMINISTIC,
    DECISION_MODE_LLM,
    DECISION_MODE_LLM_AGENT,
)
DEFAULT_DECISION_MODE = DECISION_MODE_DETERMINISTIC

#: The wire type of a decision mode.  A closed set, so an unknown mode is a validation error
#: rather than an unrecognised string a session would have to interpret.
DecisionMode = Literal["deterministic", "llm", "llm_agent"]

#: Upper bound on a chat message.  A demo turn is a shopping sentence, not a document;
#: bounding it keeps one request from carrying an unbounded amount of prompt text.
MAX_MESSAGE_LENGTH = 2000

#: A non-blank, bounded user message.
Message = Annotated[str, Field(min_length=1, max_length=MAX_MESSAGE_LENGTH, strict=True)]

#: A non-blank opaque identifier.
OpaqueId = Annotated[str, Field(min_length=1, max_length=200, strict=True)]

#: Strict, accepted recommendation count bound (reused from the Milestone 6 contract).
StrictK = Annotated[int, Field(ge=MIN_K, le=MAX_K, strict=True)]


# --------------------------------------------------------------------------- #
# requests
# --------------------------------------------------------------------------- #


class CreateSessionRequest(BaseModel):
    """Body of ``POST /v1/demo/sessions``."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    profile_id: OpaqueId = Field(
        default="demo-user-1",
        description="Identifier of a server-owned demo profile. Never a trusted history.",
        examples=["demo-user-1"],
    )
    decision_mode: DecisionMode = Field(
        default=DEFAULT_DECISION_MODE,
        description=(
            "Which decision policy serves this session's turns. 'deterministic' (the default) "
            "runs offline; 'llm' needs a configured provider and is rejected when it is not "
            "configured, rather than falling back."
        ),
    )

    @field_validator("profile_id")
    @classmethod
    def _strip_profile(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("profile_id must not be blank")
        return stripped


class ChatRequest(BaseModel):
    """Body of ``POST /v1/demo/sessions/{session_id}/chat``.

    The message is untrusted display text and untrusted extraction input.  It can never
    carry interaction history: the only fields that exist are the message, ``k`` and the
    optional decision mode.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    message: Message = Field(
        ...,
        description="Natural-language shopping message. Untrusted text.",
        examples=["Recommend some hiking gear."],
    )
    k: StrictK = Field(
        default=5,
        description=f"Number of recommendations to request, {MIN_K}..{MAX_K} (strict integer).",
    )
    decision_mode: DecisionMode | None = Field(
        default=None,
        description=(
            "Optional switch of this session's decision mode for this turn and every later "
            "turn. Omitted means 'use the mode the session already has', so a client that "
            "never sends it stays deterministic."
        ),
    )

    @field_validator("message")
    @classmethod
    def _strip_message(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("message must not be blank")
        return stripped


# --------------------------------------------------------------------------- #
# public views
# --------------------------------------------------------------------------- #


class DemoProfileView(BaseModel):
    """Public view of a server-owned demo profile.

    Deliberately excludes the raw trusted history: the browser never needs it, and
    exposing it would leak the behavioural record the trust boundary protects.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    profile_id: str
    display_name: str
    history_length: int = Field(..., ge=0, description="How many history items back this profile.")
    history_distinct: int = Field(..., ge=0, description="Distinct items in that history.")


class SessionResponse(BaseModel):
    """Body of ``POST /v1/demo/sessions``."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    api_version: str = DEMO_API_VERSION
    session_id: str = Field(..., description="Opaque demo-session capability token (UUID4).")
    profile: DemoProfileView
    turn: int = Field(..., ge=0, description="Turns completed in this session.")
    decision_mode: DecisionMode = Field(
        default=DEFAULT_DECISION_MODE,
        description="The decision policy this session runs its turns with.",
    )


class PreferenceMutationView(BaseModel):
    """One preference the current turn added, replaced or removed."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: str
    polarity: str
    value: str


class MemoryUpdateView(BaseModel):
    """What this turn's explicit statement changed in preference memory.

    A direct projection of the accepted Milestone 9 ``MemoryUpdateSummary``: an
    explicit REPLACE appears as one ``added`` entry plus the ``superseded`` entry it
    corrected, and a retraction appears as ``removed``.  The memory ``user_key`` and the
    raw extraction payload are deliberately **not** part of this view.

    A non-empty ``added`` means the preference is stored for **future** turns: the
    recommendation returned by the *same* turn was ranked against the snapshot loaded
    before it (see :attr:`AuditView.ranked_with_preferences`).
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    changed: bool = False
    added: tuple[PreferenceMutationView, ...] = ()
    superseded: tuple[PreferenceMutationView, ...] = ()
    removed: tuple[PreferenceMutationView, ...] = ()
    skipped_duplicates: int = Field(default=0, ge=0)
    already_processed: bool = False
    removal_directives: int = Field(default=0, ge=0)


class ActivePreferenceView(BaseModel):
    """One ACTIVE explicit preference.

    Only ACTIVE entries are ever exposed here.  Superseded and removed entries exist in
    the audit trail but are not part of the user-visible preference panel.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: str = Field(..., description="Accepted M9 preference kind, e.g. 'color'.")
    polarity: str = Field(..., description="'prefer' or 'avoid'.")
    value: str


class EvidenceView(BaseModel):
    """One Milestone 10A evidence record, rendered without reinterpretation."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: str
    polarity: str
    value: str
    status: Literal["match", "violation", "unknown"] = Field(
        ...,
        description=(
            "'unknown' means the available metadata cannot decide the preference. It is "
            "NOT a negative fact and must not be displayed as 'does not match'."
        ),
    )
    metadata_field: str | None = None
    metadata_value: str | None = None
    detail: str | None = None


class ProductMetadataView(BaseModel):
    """Grounded catalogue facts for one candidate.

    Every field is either what the accepted Milestone 8 metadata artifact supplied or
    ``None``.  A missing field is rendered by the client as an explicit
    "unavailable", never as invented text.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    title: str | None = None
    store: str | None = None
    main_category: str | None = None
    price_text: str | None = None
    categories: tuple[str, ...] = ()
    details: tuple[tuple[str, str], ...] = Field(
        default=(), description="Source attribute bag, key/value pairs, order preserved."
    )


class KnowledgeEvidenceView(BaseModel):
    """One catalogue fact cited by a card's grounded reason (V2 E-02B).

    ``evidence_id`` is ``"<parent_asin>:<field path>"`` and identifies a unit of *this product's own*
    retrieved knowledge; ``field_path`` and ``value`` are what the catalogue actually holds and what
    the validated claim repeated.  The user's own feedback is never part of this block.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    evidence_id: str
    parent_asin: str
    item_id: int
    field_path: str
    value: str
    source: str = "catalog_metadata"
    retrieval_score: float = 0.0
    truncated: bool = Field(
        default=False, description="True when the cited value was shortened for display."
    )


class RagAuditView(BaseModel):
    """What the product-knowledge explanation step did for this turn (V2 E-02B)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    enabled: bool = False
    model_called: bool = False
    products_examined: int = Field(default=0, ge=0)
    products_with_evidence: int = Field(default=0, ge=0)
    evidence_units: int = Field(default=0, ge=0)
    explanations_accepted: int = Field(default=0, ge=0)
    explanations_rejected: int = Field(default=0, ge=0)
    explanations_abstained: int = Field(default=0, ge=0)
    failure: str | None = None
    reason_source: str | None = Field(
        default="template_from_validated_claims",
        description=(
            "Where an accepted reason's sentence came from.  Never free model prose: this "
            "deployment renders it from the claims it validated against the product's own "
            "catalogue evidence."
        ),
    )


class ReplanAuditView(BaseModel):
    """The replan gate's decision for this turn (V2 E-03A), and what bounded replanning did (E-03B).

    A decision, never an execution, in the E-03A fields; the E-03B fields report what actually
    happened inside the run, with ``status`` from a closed vocabulary so "never asked", "asked and
    declined", "invalid answer" and "a source really ran" can never be confused.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    continue_recommended: bool = False
    reason_code: str = ""
    eligible_sources: tuple[str, ...] = ()
    remaining_tool_budget: int = Field(default=0, ge=0)
    remaining_step_budget: int = Field(default=0, ge=0)
    decision_source: str = "deterministic_rules"
    status: str | None = Field(
        default=None,
        description=(
            "The replanning outcome when the capability ran: not_triggered / gate_blocked / "
            "model_declined / invalid_output / model_failed / appended_candidates / "
            "appended_empty / appended_failed.  Null when the capability is disabled."
        ),
    )
    model_called: bool = False
    chosen_source: str | None = None
    initial_candidate_count: int = Field(default=0, ge=0)
    final_candidate_count: int = Field(default=0, ge=0)


class ProductFeedbackView(BaseModel):
    """What one product-level feedback statement did (V2 E-03C).

    Six stages, kept apart on purpose: the user's wording (``action``/``span``), the reference the
    system parsed (``reference``), the **verified** identity (``parent_asin``/``item_id``/
    ``final_position``), whether a record was written (``written``/``memory_id``/``entry_status``),
    what is ACTIVE now (``active_product_feedback``), and the effect the next turn will therefore
    apply (``effect``).  ``status`` is the honest verdict, including refusals: an ambiguous reference
    or a product this session never presented writes nothing.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    status: str = Field(
        ..., description="none | resolved | ambiguous | unknown_product | no_presentation"
    )
    action: str = Field(
        default="none", description="like | dislike | exclude | correct | retract | none"
    )
    reference: str = ""
    span: str = ""
    detail: str = ""
    parent_asin: str | None = None
    item_id: int | None = None
    final_position: int | None = Field(
        default=None,
        description="The position the user actually saw, after fusion and ranking - never a raw rank.",
    )
    written: bool = False
    entry_status: str | None = Field(
        default=None,
        description=(
            "The lifecycle state the stored entry ended in (active / superseded / removed).  The "
            "store's own entry identifier is deliberately not published: 'a record exists and here "
            "is its state' is what the contract needs, and a store handle is not."
        ),
    )
    effect: str | None = Field(
        default=None,
        description=(
            "promote | demote | replace | retract.  This deployment orders candidates by M10A "
            "evidence, so an explicit exclusion request is reported as ``demote``: a disliked "
            "product sinks and is absent whenever the pool is larger than k, but it is not deleted "
            "from the candidate set."
        ),
    )
    active_product_feedback: tuple[dict[str, str], ...] = ()
    active_before: tuple[dict[str, str], ...] = ()


class RecommendationCard(BaseModel):
    """One recommendation, with both ranks and its own evidence attached by identity."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    reranked_rank: int | None = Field(
        default=None,
        ge=1,
        description=(
            "Position under the configured explicit-preference policy, or None when no "
            "reranking ran. When present, cards are ordered by this value."
        ),
    )
    original_rank: int = Field(..., ge=1, description="SASRec rank, unchanged from the Tool.")
    parent_asin: str
    item_id: int
    sasrec_score: float = Field(
        ...,
        description=(
            "Raw SASRec ranking score. NOT a probability, confidence value, rating or "
            "preference score; it is only meaningful for ordering candidates."
        ),
    )
    score_kind: str | None = Field(
        default=None,
        description=(
            "'rrf_rank_fusion' when Serving V1 fused several sources for this card. Distinguishes a "
            "fused ranking score from the single-model score the accepted path reported."
        ),
    )
    fusion_sources: tuple[str, ...] = Field(
        default=(), description="Which sources ranked this candidate (Serving V1 only)."
    )
    fusion_source_ranks: dict[str, int] = Field(
        default_factory=dict, description="Each source's own 1-based rank for this candidate."
    )
    fused_rank: int | None = Field(
        default=None, description="The candidate's position in the fused order, before preference reranking."
    )
    metadata_status: Literal["found", "missing"] = Field(
        ..., description="'missing' means the catalogue has no record for this candidate."
    )
    metadata: ProductMetadataView | None = None

    match_count: int = Field(default=0, ge=0)
    violation_count: int = Field(default=0, ge=0)
    unknown_count: int = Field(default=0, ge=0)
    evidence: tuple[EvidenceView, ...] = ()
    fallback_reason: str | None = Field(
        default=None, description="Why no evidence fragment was retrieved, when applicable."
    )
    movement_summary: str | None = Field(
        default=None,
        description=(
            "Factual rank-movement sentence, or None when the candidate did not move. "
            "Never a quality or relevance claim."
        ),
    )
    # -- grounded explanation (V2 E-02B): about the product, never about the user ----------------- #
    reason: str | None = Field(
        default=None,
        description=(
            "A sentence rendered from validated catalogue claims for this product, or null when "
            "nothing was validated.  Never free model prose."
        ),
    )
    reason_status: str = Field(
        default="not_attempted",
        description=(
            "grounded | insufficient_evidence | rejected | not_attempted.  A card with no verified "
            "reason says 'not_attempted' rather than leaving the field empty, so 'we did not try' "
            "and 'we tried and found nothing' stay distinguishable."
        ),
    )
    knowledge_evidence: tuple[KnowledgeEvidenceView, ...] = Field(
        default=(),
        description="The catalogue facts the reason cites, one per validated claim.",
    )
    # -- the user's own attitude (V2 E-03C): not a fact about the product ------------------------- #
    feedback: str | None = Field(
        default=None,
        description=(
            "The feedback that was ACTIVE when this answer was ranked: 'liked' or 'disliked'.  It "
            "is the user's attitude, never a catalogue fact, and it never appears in ``reason`` or "
            "``knowledge_evidence``."
        ),
    )


class AuditView(BaseModel):
    """Compact, factual developer/audit block.

    Contains no chain-of-thought, no prompts, no trusted history and no store contents.

    ``ranked_with_preferences`` is the deliberate, explicit answer to "what did this
    turn's ranking actually see?": it is the Milestone 9 snapshot loaded at the **start**
    of the turn, which is why a preference written by this same turn does not appear in
    it.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    reranking_applied: bool
    candidate_count: int = Field(..., ge=0)
    moved_count: int = Field(..., ge=0)
    original_order: tuple[str, ...] = ()
    reranked_order: tuple[str, ...] = ()
    ranked_with_preference_count: int = Field(default=0, ge=0)
    ranked_with_preferences: tuple[ActivePreferenceView, ...] = ()
    rag: RagAuditView | None = Field(
        default=None,
        description="The product-knowledge explanation step's own record (V2 E-02B), or null.",
    )
    replan: ReplanAuditView | None = Field(
        default=None,
        description=(
            "The observation-and-replan gate's decision for this turn, or null when no controller "
            "ran (a direct turn).  A decision only: E-03A does not execute a second plan."
        ),
    )
    feedback: ProductFeedbackView | None = Field(
        default=None,
        description=(
            "Product-level feedback for this turn (E-03C): the resolved reference, the verified "
            "identity, the stored record and the effect the next turn applies."
        ),
    )


class TraceMemoryChanges(BaseModel):
    """Persisted preference mutations for one turn, taken from the accepted write result."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    changed: bool = False
    added: tuple[str, ...] = ()
    removed: tuple[str, ...] = ()
    superseded: tuple[str, ...] = ()


class TraceEvidenceSummary(BaseModel):
    """Preference evidence already computed by M10A, counted per candidate."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    match_candidates: int = Field(default=0, ge=0)
    violation_candidates: int = Field(default=0, ge=0)
    unknown_candidates: int = Field(default=0, ge=0)


class TraceRankChange(BaseModel):
    """One candidate's movement between the SASRec order and the reranked order."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    parent_asin: str
    title: str | None = Field(
        default=None,
        description="Card title when the existing card already carried one. Never looked up here.",
    )
    original_rank: int = Field(..., ge=1, description="SASRec rank, unchanged from the Tool.")
    final_rank: int = Field(..., ge=1, description="Presentation order produced by reranking.")
    moved: bool = False


class TraceGroundingView(BaseModel):
    """Placeholder for catalogue-grounding counts.

    Deliberately unpopulated on the browser chat path: the AgentGraph capability route has no
    candidate plane and no ledger, so there is no authoritative grounded/verified count to
    read. It exists so a future path with such a source can fill it without a schema change.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    verified: int = Field(default=0, ge=0)
    total: int = Field(default=0, ge=0)


class TracePreferenceAction(BaseModel):
    """One preference action a decision policy proposed for this turn.

    Only a *validated* action ever appears here: when the LLM decision mode is used, an action
    reaches this view only after the policy's plan schema accepted it.  ``applied`` is the
    separate, stronger fact, taken from the accepted memory write: ``True`` only when the
    trusted store actually changed an ACTIVE entry for this value this turn.  A validated
    action that the store declined (a duplicate, or a removal that matched nothing) is
    therefore visible as ``applied=False`` rather than being reported as a change.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    action: Literal["add", "remove"]
    value: str
    kind: str | None = Field(
        default=None,
        description="Accepted preference kind for an add; None for a removal, which is value-scoped.",
    )
    applied: bool = Field(
        default=False,
        description="True only when the persisted memory write changed an active entry for this value.",
    )


class DecisionModeView(BaseModel):
    """One decision mode a browser may select, with its real availability.

    ``available`` is read from execution state (is a provider configured for this deployment,
    and can this control plane serve the mode), never hardcoded.  ``detail`` carries the
    human-readable reason a mode is unavailable, authored by the API layer.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    mode: DecisionMode
    label: str
    available: bool
    reason: str | None = Field(
        default=None, description="Stable machine-readable reason, e.g. 'missing_api_key'."
    )
    detail: str | None = Field(default=None, description="Human-readable explanation for a UI.")
    provider: str | None = None
    model: str | None = None


class DecisionModesResponse(BaseModel):
    """Body of ``GET /v1/demo/decision-modes``: what this deployment can actually serve."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    api_version: str = DEMO_API_VERSION
    default: DecisionMode = DEFAULT_DECISION_MODE
    modes: tuple[DecisionModeView, ...] = ()


class RecommendationTrace(BaseModel):
    """A consolidated VIEW of the decisions one turn already made.

    Every value here is copied from authoritative state that the response already carries --
    the recommendation cards, the audit block, the persisted memory write result and the
    accepted M10A evidence. Nothing is recomputed: candidates are not re-counted by a second
    rule, evidence is not re-matched, preferences are not re-extracted and ranking is not
    re-derived. The trace adds no product identity that is not already in
    ``ChatResponse.recommendations``.

    ``source`` and ``grounding`` are optional and stay None on the browser chat path, where no
    authoritative value exists; a path that has one may populate them without a schema change.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    route: Literal["direct", "recommend"]
    candidate_count: int | None = Field(
        default=None, ge=0, description="Omitted when the turn produced no candidates."
    )
    decision_mode: str = Field(
        default=DEFAULT_DECISION_MODE,
        description=(
            "Which decision policy served this turn: 'deterministic' or 'llm'. Read from how the "
            "turn actually executed, never inferred from the request."
        ),
    )
    provider: str | None = Field(
        default=None,
        description=(
            "Provider that answered this turn's decision call, as declared by the client that "
            "made it. None in deterministic mode, and None in LLM mode when the client declares "
            "no identity - an unknown provider is never guessed."
        ),
    )
    model: str | None = Field(
        default=None,
        description="Model identifier declared by the client that answered. Never a credential.",
    )
    proposed_route: str | None = Field(
        default=None,
        description=(
            "Route the decision policy proposed this turn, when a policy proposed one. The "
            "executed route is the separate 'route' field, so the two can be compared."
        ),
    )
    preference_actions: tuple[TracePreferenceAction, ...] = Field(
        default=(),
        description=(
            "Validated preference actions this turn's policy proposed, with the persisted "
            "'applied' outcome. Never a rejected or unvalidated proposal."
        ),
    )
    active_preferences: tuple[ActivePreferenceView, ...] = ()
    memory_changes: TraceMemoryChanges = Field(default_factory=TraceMemoryChanges)
    evidence: TraceEvidenceSummary | None = Field(
        default=None,
        description=(
            "Present only when the turn actually ranked against active preferences, because "
            "the evidence was computed against the start-of-turn snapshot. Zeroes on a turn "
            "with no preferences would look like a measurement that did not happen."
        ),
    )
    ranking_changes: tuple[TraceRankChange, ...] = ()
    source: str | None = Field(default=None, description="Unpopulated on the browser chat path.")
    grounding: TraceGroundingView | None = Field(
        default=None, description="Unpopulated on the browser chat path."
    )


class ChatResponse(BaseModel):
    """Body of a successful ``POST /v1/demo/sessions/{session_id}/chat``."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    api_version: str = DEMO_API_VERSION
    session_id: str
    turn_id: str = Field(..., description="Server-owned turn identifier: '<session_id>:<seq>'.")
    turn: int = Field(..., ge=1, description="1-based turn number within this session.")
    route: Literal["direct", "recommend"]
    message: str = Field(..., description="The agent's rendered, grounded response text.")
    active_preferences: tuple[ActivePreferenceView, ...] = Field(
        default=(),
        description=(
            "ACTIVE preference memory as it stands AFTER this turn's write. This is the "
            "panel state, not what ranked this turn -- see audit.ranked_with_preferences."
        ),
    )
    memory_update: MemoryUpdateView = Field(default_factory=MemoryUpdateView)
    recommendations: tuple[RecommendationCard, ...] = Field(
        default=(),
        description=(
            "Structured recommendation cards in the exact order the backend produced. "
            "The client renders this sequence and must not re-sort it."
        ),
    )
    agent: AgentExecutionView | None = Field(
        default=None,
        description=(
            "The agent run's own execution view, present when a controller executed this turn (the "
            "llm_agent mode).  Null for the deterministic, llm and direct paths, which run no "
            "controller - stated rather than defaulted, because 'no controller ran' and 'a "
            "controller ran and found nothing' are different facts."
        ),
    )
    audit: AuditView
    trace: RecommendationTrace | None = Field(
        default=None,
        description=(
            "Optional consolidated view of this turn's deterministic decisions. Additive and "
            "optional: a client that ignores it behaves exactly as before."
        ),
    )


class SessionStateResponse(BaseModel):
    """Body of ``GET /v1/demo/sessions/{session_id}``."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    api_version: str = DEMO_API_VERSION
    session_id: str
    profile: DemoProfileView
    turn: int = Field(..., ge=0)
    decision_mode: DecisionMode = Field(
        default=DEFAULT_DECISION_MODE,
        description="The decision policy this session will use for its next turn.",
    )
    active_preferences: tuple[ActivePreferenceView, ...] = ()
    active_preference_count: int = Field(default=0, ge=0)


class DemoHealthResponse(BaseModel):
    """Body of ``GET /v1/demo/health`` (a demo-specific readiness view).

    This is deliberately a **separate** endpoint from the accepted Milestone 6
    ``/health``: that contract's schema is frozen and must not change.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    api_version: str = DEMO_API_VERSION
    status: str = Field(..., description="'ok' only when the full demo chain is ready.")
    model_loaded: bool
    metadata_loaded: bool
    demo_ready: bool
    profiles: int = Field(default=0, ge=0)
    active_sessions: int = Field(default=0, ge=0)
    max_sessions: int = Field(default=0, ge=0)
    detail: str | None = None


class ProfileListResponse(BaseModel):
    """Body of ``GET /v1/demo/profiles``: the server-owned demo profiles on offer."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    api_version: str = DEMO_API_VERSION
    profiles: tuple[DemoProfileView, ...] = ()


class ResetResponse(BaseModel):
    """Body of a successful ``DELETE /v1/demo/sessions/{session_id}``.

    States the reset semantics explicitly rather than leaving them to be inferred, since
    "reset" could otherwise mean "clear the transcript", "clear preferences", or both.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    api_version: str = DEMO_API_VERSION
    session_id: str
    reset: bool = True
    detail: str = Field(
        ...,
        description=(
            "The session id was removed from the live registry and its preference-memory "
            "namespace was retired; it is unreachable through the API. A new session must "
            "be created explicitly."
        ),
    )


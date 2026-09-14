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

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from recommendation.api.schemas import MAX_K, MIN_K

__all__ = [
    "DEMO_API_VERSION",
    "MAX_MESSAGE_LENGTH",
    "ActivePreferenceView",
    "AuditView",
    "ChatRequest",
    "ChatResponse",
    "CreateSessionRequest",
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
]

#: Version of the demo wire contract, published in session responses.
DEMO_API_VERSION = "1.0.0"

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
    carry interaction history: the only fields that exist are the message and ``k``.
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
    audit: AuditView


class SessionStateResponse(BaseModel):
    """Body of ``GET /v1/demo/sessions/{session_id}``."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    api_version: str = DEMO_API_VERSION
    session_id: str
    profile: DemoProfileView
    turn: int = Field(..., ge=0)
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

"""Graph state for the minimal LangGraph agent (Milestone 7B).

Two things live here and they are deliberately different:

``AgentInput``
    What the **application** supplies to start a run.  The natural-language user
    message is untrusted text; ``trusted_user_history`` is trusted application
    state (chronological Amazon ``parent_asin`` values sourced from a session,
    profile or batch job - never from model output).

``AgentGraphState``
    What LangGraph threads through the nodes during a run.

Neither is a model-facing type.  The only thing the decision model ever sees is
the tuple produced by
:func:`~recommendation.agent.decision.build_decision_messages`, which takes the
user message and nothing else.
Why history cannot be rewritten
-------------------------------
``trusted_user_history`` has **no reducer**: LangGraph replaces a channel value
only with what a node returns, and the decision node returns only ``decision``.
The Tool node reads the history but does not write it.  Together with
``AgentDecision``'s ``extra="forbid"`` schema, that means no decision output can
add to, drop from, reorder or otherwise alter the history a run was started with.

The state also never carries internal item ids, encoded model histories, SASRec
tensors, mapping internals or checkpoint internals.  Candidate identity appears
only as external ``parent_asin`` values inside the Tool's typed result.

Original order versus reranked order
------------------------------------
Milestone 10D adds two *derived* channels and rewrites nothing:

``tool_result``
    The accepted Tool result, in SASRec order.  Its ``rank`` values are the
    authoritative ``original_rank``; it is never mutated to look reranked.
``enrichment``
    The same candidates in the same order, plus candidate-scoped metadata.
``preference_evidence``
    M10A evidence over those candidates, still in the upstream order.
``reranking``
    The M10B result: the same candidate identities in policy order, each carrying
    both ``original_rank`` and ``reranked_rank``.

A consumer can therefore always recover the original order from the upstream
channels even when the final response is presented in reranked order.
"""

from __future__ import annotations

import hashlib
from typing import Annotated, Any, TypedDict

from pydantic import BaseModel, ConfigDict, Field, field_validator

from recommendation.tools.schemas import RecommendationContext, RecommendationToolResult

from .decision import AgentDecision, MalformedDecision

__all__ = [
    "AgentGraphState",
    "AgentInput",
    "TrustedHistory",
    "history_digest",
    "new_agent_state",
    "read_trusted_history",
]

#: Immutable, chronological, already-validated ``parent_asin`` history.
TrustedHistory = tuple[str, ...]

#: Length of the short SHA-256 prefix used for logging/diagnostics.  The digest is
#: one-way, so logs can show *which* history produced a run without printing it.
_DIGEST_CHARS = 16


def _normalise_user_message(value: object) -> str:
    """Strip a user message and reject a blank one.

    The raised ``ValueError`` deliberately does not echo the rejected value: this
    keeps validation messages payload-free and consistent with the decision schema.
    """
    if not isinstance(value, str):
        raise ValueError("user_message must be a string")
    stripped = value.strip()
    if not stripped:
        raise ValueError("user_message must not be blank")
    return stripped


def _normalise_trusted_history(value: object) -> TrustedHistory:
    """Validate a trusted history sequence through the Tool's own context schema.

    Reusing :class:`~recommendation.tools.schemas.RecommendationContext` rather than
    restating its rules (non-empty, non-blank entries, whitespace stripped, order
    preserved) means the agent boundary and the Tool boundary cannot drift apart.
    A bare string is rejected, because it is a sequence of characters, not a history.
    """
    if isinstance(value, (str, bytes)) or not isinstance(value, (list, tuple)):
        raise ValueError("trusted_user_history must be a sequence of parent_asin strings")
    if not value:
        raise ValueError("trusted_user_history must not be empty")
    validated = RecommendationContext(user_history=value)
    return tuple(validated.user_history)


class AgentInput(BaseModel):
    """Validated entry point for one agent run.

    ``user_message`` is the only natural-language input.  ``trusted_user_history``
    is required and must be non-empty: the accepted recommender has no cold-start
    fallback, so a run without history cannot produce recommendations.

    Validation reuses the Tool's context schema and never echoes the rejected
    value, so an invalid input cannot leak history into an error message.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    user_message: Annotated[str, Field(min_length=1)] = Field(
        ...,
        description="Natural-language message from the user. Untrusted text.",
    )
    trusted_user_history: TrustedHistory = Field(
        ...,
        description=(
            "Trusted chronological parent_asin history from application state. "
            "Never supplied by a model."
        ),
    )
    turn_id: str | None = Field(
        default=None,
        description=(
            "Application-supplied identifier of this user turn. Used only to make "
            "preference-memory writes idempotent; never produced by a model."
        ),
    )

    @field_validator("user_message")
    @classmethod
    def _validate_user_message(cls, value: object) -> str:
        return _normalise_user_message(value)

    @field_validator("trusted_user_history")
    @classmethod
    def _validate_trusted_history(cls, value: object) -> TrustedHistory:
        return _normalise_trusted_history(value)


class AgentGraphState(TypedDict, total=False):
    """LangGraph state channels for one agent run.

    ``total=False`` because the input supplies only the first two channels and the
    nodes fill in the rest as the run progresses.
    """

    # -- input, written once by the application ---------------------------- #
    user_message: str
    trusted_user_history: TrustedHistory
    #: Optional application-supplied identifier of this user turn, used for
    #: idempotent preference-memory writes.  It is not model output.
    turn_id: str

    # -- written by the decision node ------------------------------------- #
    decision: AgentDecision

    # -- written by the tool node ----------------------------------------- #
    tool_result: RecommendationToolResult

    # -- written by the enrichment node (Milestone 8, optional) ------------ #
    #: Candidate-scoped product evidence.  Typed as ``Any`` so this module does not
    #: depend on the RAG package; when present it is an
    #: ``recommendation.rag.schemas.EnrichmentResult``.
    enrichment: Any

    # -- written by the memory nodes (Milestone 9, optional) --------------- #
    #: Immutable snapshot of the user's ACTIVE preferences, read before the decision.
    #: Typed ``Any`` so this module does not depend on the memory package; when present
    #: it is a ``recommendation.memory.schemas.PreferenceMemorySnapshot``.  Trusted
    #: interaction history is deliberately *not* part of this or any memory type.
    preference_snapshot: Any
    #: Result of persisting this turn's explicit preferences (audit/diagnostics).
    memory_update: Any

    # -- written by the preference nodes (Milestone 10D, optional) --------- #
    #: M10A evidence for the *already-enriched* candidates.  Typed ``Any`` so this
    #: module does not depend on the preference-matching package; when present it is a
    #: ``recommendation.preference_matching.schemas.PreferenceEvidenceReport``.
    #:
    #: This channel is evidence, never an order: the candidates inside it appear in the
    #: exact upstream SASRec order, each keeping its ``original_rank``, ``item_id``,
    #: ``parent_asin`` and raw ``sasrec_score`` unchanged.
    preference_evidence: Any
    #: M10B reranking result.  Typed ``Any`` so this module does not depend on the
    #: reranking package; when present it is a
    #: ``recommendation.reranking.schemas.RerankingReport``.
    #:
    #: This is the *derived* view: it carries the reranked order.  It is additive
    #: state -- ``tool_result``, ``enrichment`` and ``preference_evidence`` keep the
    #: authoritative upstream order and are never rewritten to look reranked.
    reranking: Any

    # -- written by the finalizing node ----------------------------------- #
    final_response: str

    # -- run diagnostics --------------------------------------------------- #
    route: str
    error: str


def read_trusted_history(state: AgentGraphState) -> TrustedHistory:
    """Return the run's trusted history as an immutable tuple.

    Raises :class:`MalformedDecision` when the history is missing or empty, rather
    than letting the run continue toward a fabricated answer.
    """
    history = state.get("trusted_user_history")
    if not history:
        raise MalformedDecision(
            "the agent run has no trusted user history; supply it from application state"
        )
    return tuple(history)


def history_digest(history: TrustedHistory) -> str:
    """Return a short, one-way digest of a history for diagnostics.

    Lets a smoke test or log line prove two runs saw the *same* history without
    printing the user's interactions.  It is not a security primitive and is not
    used for any decision.
    """
    joined = "\x1f".join(history).encode("utf-8")
    return hashlib.sha256(joined).hexdigest()[:_DIGEST_CHARS]


def new_agent_state(agent_input: AgentInput) -> AgentGraphState:
    """Build the initial graph state from validated application input.

    Note what is *not* here: there is no preference-memory field seeded from model
    output, and no interaction-history field that a preference could feed.  Trusted
    history arrives from the application and stays exactly as supplied.
    """
    state = AgentGraphState(
        user_message=agent_input.user_message,
        trusted_user_history=tuple(agent_input.trusted_user_history),
    )
    if agent_input.turn_id:
        state["turn_id"] = agent_input.turn_id
    return state

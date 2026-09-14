"""Explicit adapter from an :class:`AgentGraphState` to the public demo response.

This module is the whole public/internal boundary.  It is a **whitelist**, not a
projection of graph state: every field of :class:`ChatResponse` is built by hand from a
named source, and there is deliberately no ``dict(graph_state)`` (or
``model_dump()`` of graph state) anywhere in it.

What may cross the boundary
---------------------------
* the agent's rendered, grounded response text;
* candidate identity and both ranks, plus the accepted raw SASRec score;
* catalogue facts copied verbatim from the candidate's **own** Milestone 8 metadata;
* Milestone 10A evidence records, with their three-state status unaltered;
* the Milestone 9 ACTIVE preference set *after* the write, and the accepted write
  summary;
* the Milestone 9 snapshot that actually ranked this turn (audit block).

What may not
------------
``trusted_user_history`` (or any part of it), the memory ``user_key``, memory store
paths, checkpoint paths, internal ``source_user_int_id``, raw extraction payloads,
internal ``memory_id`` values, exception text and stack traces.

Ordering
--------
Cards are emitted in the order the backend produced, and the ordering source is chosen
explicitly:

1. ``reranking.candidates`` when a Milestone 10B report exists -- that **is** the
   reranked order, and the frontend renders it as given;
2. otherwise ``enrichment.items`` (Milestone 8 order);
3. otherwise ``tool_result.recommendations`` (Milestone 7A order).

Card metadata and evidence are attached by ``(parent_asin, item_id)`` identity, never by
list position, so a reordered candidate can never inherit its neighbour's facts.
"""

from __future__ import annotations

from typing import Any, Mapping

from .schemas import (
    ActivePreferenceView,
    AuditView,
    ChatResponse,
    EvidenceView,
    MemoryUpdateView,
    PreferenceMutationView,
    ProductMetadataView,
    RecommendationCard,
)

__all__ = [
    "active_preference_views",
    "build_chat_response",
    "build_cards",
    "movement_summary",
]


def active_preference_views(snapshot: Any) -> tuple[ActivePreferenceView, ...]:
    """Public views of the ACTIVE entries of a Milestone 9 snapshot.

    Superseded and removed entries are excluded here by the accepted snapshot property,
    so they can never reach the user-visible preference panel.
    """
    if snapshot is None:
        return ()
    return tuple(
        ActivePreferenceView(
            kind=entry.kind.value if hasattr(entry.kind, "value") else str(entry.kind),
            polarity=(
                entry.polarity.value if hasattr(entry.polarity, "value") else str(entry.polarity)
            ),
            value=str(entry.value),
        )
        for entry in snapshot.active_entries
    )


def _mutation_views(entries: Any) -> tuple[PreferenceMutationView, ...]:
    """Public views of the entries an M9 write summary added, superseded or removed."""
    if not entries:
        return ()
    return tuple(
        PreferenceMutationView(
            kind=entry.kind.value if hasattr(entry.kind, "value") else str(entry.kind),
            polarity=(
                entry.polarity.value if hasattr(entry.polarity, "value") else str(entry.polarity)
            ),
            value=str(entry.value),
        )
        for entry in entries
    )


def memory_update_view(state: Mapping[str, Any]) -> MemoryUpdateView:
    """Build the public write summary, or an empty one when memory is not configured."""
    result = state.get("memory_update")
    summary = getattr(result, "update", None)
    if summary is None:
        return MemoryUpdateView()
    return MemoryUpdateView(
        changed=bool(summary.changed),
        added=_mutation_views(summary.added),
        superseded=_mutation_views(summary.superseded),
        removed=_mutation_views(summary.removed),
        skipped_duplicates=int(summary.skipped_duplicates),
        already_processed=bool(summary.already_processed),
        removal_directives=int(summary.removal_directives),
    )


def movement_summary(original_rank: int, reranked_rank: int | None) -> str | None:
    """Return a factual movement sentence, or ``None`` when the candidate did not move.

    Wording is deliberately limited to facts the Milestone 10B report actually produced.
    It never uses the Milestone 10B reason label (Milestone 10C showed its tail wording
    can be imprecise), never mentions ``item_id`` (unreachable for valid input with
    unique original ranks) and never claims the move makes a product better or more
    relevant.
    """
    if reranked_rank is None or reranked_rank == original_rank:
        return None
    return (
        f"Moved from rank {original_rank} to rank {reranked_rank} under the explicit-"
        "preference policy."
    )


def _metadata_view(metadata: Any) -> ProductMetadataView | None:
    """Copy the whitelisted catalogue fields of an M8 metadata record verbatim."""
    if metadata is None:
        return None
    return ProductMetadataView(
        title=metadata.title,
        store=metadata.store,
        main_category=metadata.main_category,
        price_text=metadata.price_text,
        categories=tuple(metadata.categories),
        details=tuple(metadata.details),
    )


def _evidence_views(evidence: Any) -> tuple[EvidenceView, ...]:
    """Copy M10A evidence records, preserving status exactly and in order."""
    return tuple(
        EvidenceView(
            kind=(
                record.preference_kind.value
                if hasattr(record.preference_kind, "value")
                else str(record.preference_kind)
            ),
            polarity=(
                record.preference_polarity.value
                if hasattr(record.preference_polarity, "value")
                else str(record.preference_polarity)
            ),
            value=str(record.preference_value),
            status=record.status.value,
            metadata_field=record.metadata_field,
            metadata_value=record.metadata_value,
            detail=record.detail,
        )
        for record in evidence
    )


def build_cards(state: Mapping[str, Any]) -> tuple[RecommendationCard, ...]:
    """Build the ordered, identity-aligned recommendation cards for a run.

    Returns an empty tuple when the run produced no candidates -- candidate exhaustion is
    a legal outcome, and no fallback product is ever invented.
    """
    reranking = state.get("reranking")
    enrichment = state.get("enrichment")
    tool_result = state.get("tool_result")

    enriched_by_identity: dict[tuple[str, int], Any] = {}
    if enrichment is not None:
        enriched_by_identity = {
            (item.parent_asin, item.recommendation.item_id): item for item in enrichment.items
        }

    evidence_by_identity: dict[tuple[str, int], Any] = {}
    report = state.get("preference_evidence")
    if report is not None:
        evidence_by_identity = {
            (candidate.parent_asin, candidate.item_id): candidate for candidate in report.candidates
        }

    cards: list[RecommendationCard] = []

    if reranking is not None:
        for candidate in reranking.candidates:
            identity = (candidate.parent_asin, candidate.item_id)
            item = enriched_by_identity.get(identity)
            cards.append(
                _card(
                    parent_asin=candidate.parent_asin,
                    item_id=candidate.item_id,
                    original_rank=candidate.original_rank,
                    reranked_rank=candidate.reranked_rank,
                    sasrec_score=candidate.sasrec_score,
                    match_count=candidate.match_count,
                    violation_count=candidate.violation_count,
                    unknown_count=candidate.unknown_count,
                    evidence=evidence_by_identity.get(identity),
                    enriched_item=item,
                )
            )
        return tuple(cards)

    if enrichment is not None:
        for item in enrichment.items:
            identity = (item.parent_asin, item.recommendation.item_id)
            evidence = evidence_by_identity.get(identity)
            record = getattr(evidence, "evidence", ()) if evidence is not None else ()
            cards.append(
                _card(
                    parent_asin=item.parent_asin,
                    item_id=item.recommendation.item_id,
                    original_rank=item.rank,
                    reranked_rank=None,
                    sasrec_score=item.score,
                    match_count=sum(1 for r in record if str(r.status.value) == "match"),
                    violation_count=sum(1 for r in record if str(r.status.value) == "violation"),
                    unknown_count=sum(1 for r in record if str(r.status.value) == "unknown"),
                    evidence=evidence,
                    enriched_item=item,
                )
            )
        return tuple(cards)

    if tool_result is not None:
        for recommendation in tool_result.recommendations:
            cards.append(
                _card(
                    parent_asin=recommendation.parent_asin,
                    item_id=recommendation.item_id,
                    original_rank=recommendation.rank,
                    reranked_rank=None,
                    sasrec_score=recommendation.score,
                    match_count=0,
                    violation_count=0,
                    unknown_count=0,
                    evidence=None,
                    enriched_item=None,
                )
            )
    return tuple(cards)


def _card(
    *,
    parent_asin: str,
    item_id: int,
    original_rank: int,
    reranked_rank: int | None,
    sasrec_score: float,
    match_count: int,
    violation_count: int,
    unknown_count: int,
    evidence: Any,
    enriched_item: Any,
) -> RecommendationCard:
    """Assemble one card, taking metadata and evidence from this candidate's own records."""
    metadata_status = "missing"
    metadata_view: ProductMetadataView | None = None
    fallback_reason: str | None = None
    if enriched_item is not None:
        metadata_status = enriched_item.metadata_status
        metadata_view = _metadata_view(enriched_item.metadata)
        fallback_reason = enriched_item.fallback_reason

    return RecommendationCard(
        reranked_rank=reranked_rank,
        original_rank=original_rank,
        parent_asin=parent_asin,
        item_id=item_id,
        sasrec_score=sasrec_score,
        metadata_status=metadata_status,
        metadata=metadata_view,
        match_count=match_count,
        violation_count=violation_count,
        unknown_count=unknown_count,
        evidence=_evidence_views(getattr(evidence, "evidence", ())),
        fallback_reason=fallback_reason,
        movement_summary=movement_summary(original_rank, reranked_rank),
    )


def build_audit(state: Mapping[str, Any], cards: tuple[RecommendationCard, ...]) -> AuditView:
    """Build the compact audit block from the graph's own derived state."""
    reranking = state.get("reranking")
    enrichment = state.get("enrichment")
    tool_result = state.get("tool_result")

    if reranking is not None:
        original_order = tuple(
            candidate.parent_asin
            for candidate in sorted(reranking.candidates, key=lambda item: item.original_rank)
        )
        reranked_order = tuple(reranking.parent_asins)
        moved_count = int(reranking.moved_count)
    elif enrichment is not None:
        original_order = tuple(enrichment.parent_asins)
        reranked_order = original_order
        moved_count = 0
    elif tool_result is not None:
        original_order = tuple(item.parent_asin for item in tool_result.recommendations)
        reranked_order = original_order
        moved_count = 0
    else:
        original_order = ()
        reranked_order = ()
        moved_count = 0

    ranked_with = active_preference_views(state.get("preference_snapshot"))
    return AuditView(
        reranking_applied=reranking is not None,
        candidate_count=len(cards),
        moved_count=moved_count,
        original_order=original_order,
        reranked_order=reranked_order,
        ranked_with_preference_count=len(ranked_with),
        ranked_with_preferences=ranked_with,
    )


def build_chat_response(
    state: Mapping[str, Any],
    *,
    session_id: str,
    turn_id: str,
    turn_number: int,
) -> ChatResponse:
    """Build the public chat response for one completed turn.

    ``session_id``/``turn_id``/``turn_number`` come from the server-owned session layer,
    never from graph state, so a caller cannot influence them.
    """
    cards = build_cards(state)
    return ChatResponse(
        session_id=session_id,
        turn_id=turn_id,
        turn=turn_number,
        route=str(state.get("route", "direct")),
        message=str(state.get("final_response", "")),
        active_preferences=active_preference_views(
            getattr(state.get("memory_update"), "active", None)
        ),
        memory_update=memory_update_view(state),
        recommendations=cards,
        audit=build_audit(state, cards),
    )

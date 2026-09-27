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

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from .schemas import (
    DEFAULT_DECISION_MODE,
    RecommendationTrace,
    TraceEvidenceSummary,
    TraceMemoryChanges,
    TracePreferenceAction,
    TraceRankChange,
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
    "TurnDecision",
    "active_preference_views",
    "build_chat_response",
    "build_cards",
    "build_trace",
    "build_trace_with_decision",
    "movement_summary",
]


@dataclass(frozen=True)
class TurnDecision:
    """How one turn's decisions were actually made, as read from execution state.

    ``decision_mode`` is a fact about the run the caller performed - the deterministic path
    reports the deterministic policy because that is what it ran, and the policy-backed path
    reports the provider that answered - never an echo of what a request asked for.  The
    optional fields stay ``None`` when no policy proposed anything, so the trace shows
    "deterministic" without inventing a provider, a route or an action.
    """

    decision_mode: str = DEFAULT_DECISION_MODE
    provider: str | None = None
    model: str | None = None
    proposed_route: str | None = None
    preference_actions: tuple[TracePreferenceAction, ...] = ()


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


def preference_only_acknowledgement(
    memory_update: Any,
    active_preferences: Sequence[Any],
) -> str | None:
    """A deterministic acknowledgement for a preference-only turn, or ``None``.

    Every word is derived from the PERSISTED write result (``MemoryUpdateView``), never from
    the user's text. ``None`` is returned whenever nothing was actually applied, so an
    unmatched removal, a duplicate or a no-op turn keeps the honest generic response instead
    of claiming a change that did not happen.
    """
    if memory_update is None or not getattr(memory_update, "changed", False):
        return None
    added = [item.value for item in getattr(memory_update, "added", ())]
    removed = [item.value for item in getattr(memory_update, "removed", ())]
    replaced = [item.value for item in getattr(memory_update, "superseded", ())]
    if not added and not removed and not replaced:
        return None

    if removed and added:
        opening = f"Got it. I replaced your {_and_list(removed)} preference with {_and_list(added)}."
    elif removed:
        opening = f"Got it. I removed your {_and_list(removed)} preference."
    elif replaced and added:
        opening = f"Got it. I updated your {_and_list(replaced)} preference to {_and_list(added)}."
    else:
        opening = f"Got it. I saved your {_and_list(added)} preference for future turns."

    active = [item.value for item in active_preferences]
    if active:
        return f"{opening} Active preference: {_and_list(active)}."
    return f"{opening} You have no active preferences."


def _and_list(values: Sequence[str]) -> str:
    """Join 1-2 values naturally, or three-or-more with an Oxford-free comma list."""
    items = [str(value) for value in values]
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    if len(items) == 2:
        return f"{items[0]} and {items[1]}"
    return ", ".join(items[:-1]) + " and " + items[-1]


def build_trace(
    *,
    route: str,
    audit: AuditView,
    cards: Sequence[RecommendationCard],
    active_preferences: tuple[ActivePreferenceView, ...],
    memory_update: MemoryUpdateView,
) -> RecommendationTrace:
    """Consolidate one turn's already-decided values into the public trace view.

    Built from the serialized values that the response already carries, so the trace cannot
    disagree with them: candidate counts come from the audit block, ranks and titles from the
    cards, mutations from the persisted update, and evidence is only counted -- never
    re-matched. Optional ``source``/``grounding`` stay None because the browser path has no
    authoritative value for them.
    """
    evidence: TraceEvidenceSummary | None = None
    if audit.ranked_with_preference_count > 0 and cards:
        matched = sum(1 for card in cards if any(e.status == "match" for e in card.evidence))
        violated = sum(1 for card in cards if any(e.status == "violation" for e in card.evidence))
        unknown = sum(1 for card in cards if any(e.status == "unknown" for e in card.evidence))
        evidence = TraceEvidenceSummary(
            match_candidates=matched,
            violation_candidates=violated,
            unknown_candidates=unknown,
        )

    ranking_changes = tuple(
        TraceRankChange(
            parent_asin=card.parent_asin,
            title=card.metadata.title if card.metadata is not None else None,
            original_rank=card.original_rank,
            final_rank=(
                card.reranked_rank if card.reranked_rank is not None else card.original_rank
            ),
            moved=card.reranked_rank is not None and card.reranked_rank != card.original_rank,
        )
        for card in cards
    )

    return RecommendationTrace(
        route=route,
        candidate_count=audit.candidate_count if cards else None,
        active_preferences=active_preferences,
        memory_changes=TraceMemoryChanges(
            changed=memory_update.changed,
            added=tuple(item.value for item in memory_update.added),
            removed=tuple(item.value for item in memory_update.removed),
            superseded=tuple(item.value for item in memory_update.superseded),
        ),
        evidence=evidence,
        ranking_changes=ranking_changes,
    )


def build_trace_with_decision(
    trace: RecommendationTrace,
    *,
    decision: TurnDecision,
    memory_update: MemoryUpdateView,
) -> RecommendationTrace:
    """Add this turn's decision facts to an already-built trace.

    Kept separate from :func:`build_trace` so that builder stays exactly what it was - a view
    over the values the response already carries - and so the deterministic path can pass its
    default :class:`TurnDecision` and get a trace that says "deterministic" with no provider,
    no proposed route and no actions.

    Like every other builder in this module this is an explicit whitelist: each field is named
    from a typed source, and nothing is dumped or copied blindly.

    ``applied`` on each action is computed here from the **persisted** memory write, not from
    the plan: an action the model proposed is reported as applied only when the accepted store
    actually changed an ACTIVE entry for that value this turn.  Where a value is absent from the
    write summary the flag stays ``False``, so a duplicate add or an unmatched removal is visible
    as a proposal that did not change anything.
    """
    return RecommendationTrace(
        route=trace.route,
        candidate_count=trace.candidate_count,
        decision_mode=decision.decision_mode,
        provider=decision.provider,
        model=decision.model,
        proposed_route=decision.proposed_route,
        preference_actions=tuple(
            TracePreferenceAction(
                action=action.action,
                value=action.value,
                kind=action.kind,
                applied=_action_applied(action, memory_update),
            )
            for action in decision.preference_actions
        ),
        active_preferences=trace.active_preferences,
        memory_changes=trace.memory_changes,
        evidence=trace.evidence,
        ranking_changes=trace.ranking_changes,
        source=trace.source,
        grounding=trace.grounding,
    )


def _action_applied(action: TracePreferenceAction, memory_update: MemoryUpdateView) -> bool:
    """True when the persisted write stopped or started an ACTIVE entry for this value.

    ============  ==================================================================
    action        applied when the write summary reports the value as
    ============  ==================================================================
    ``add``       added, or superseded (the accepted REPLACE shape: new added + old
                  superseded), so a replacement counts as having applied
    ``remove``    removed, or superseded (the value stopped being the active entry)
    ============  ==================================================================
    """
    if action.action == "add":
        entries = list(memory_update.added) + list(memory_update.superseded)
    else:
        entries = list(memory_update.removed) + list(memory_update.superseded)
    target = str(action.value).casefold()
    return any(str(entry.value).casefold() == target for entry in entries)


def build_chat_response(
    state: Mapping[str, Any],
    *,
    session_id: str,
    turn_id: str,
    turn_number: int,
    decision: TurnDecision | None = None,
) -> ChatResponse:
    """Build the public chat response for one completed turn.

    ``session_id``/``turn_id``/``turn_number`` come from the server-owned session layer,
    never from graph state, so a caller cannot influence them.  ``decision`` describes how the
    turn's decisions were reached; the default is the deterministic policy, which is what every
    caller that does not pass one actually ran.
    """
    cards = build_cards(state)
    audit = build_audit(state, cards)
    route = str(state.get("route", "direct"))
    preferences = active_preference_views(getattr(state.get("memory_update"), "active", None))
    update = memory_update_view(state)
    message = str(state.get("final_response", ""))
    if route == "direct" and not cards:
        # A preference-only turn otherwise answers with the generic capability text. Replace it
        # only when a mutation was actually persisted.
        acknowledgement = preference_only_acknowledgement(update, preferences)
        if acknowledgement is not None:
            message = acknowledgement
    trace = build_trace(
        route=route,
        audit=audit,
        cards=cards,
        active_preferences=preferences,
        memory_update=update,
    )
    return ChatResponse(
        session_id=session_id,
        turn_id=turn_id,
        turn=turn_number,
        route=route,
        message=message,
        active_preferences=preferences,
        memory_update=update,
        recommendations=cards,
        audit=audit,
        trace=build_trace_with_decision(
            trace,
            decision=decision if decision is not None else TurnDecision(),
            memory_update=update,
        ),
    )

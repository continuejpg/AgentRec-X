"""Pure presentation helpers shared by the agent DAG and the 2.0-alpha loop.

This module was extracted from :mod:`recommendation.agent.graph` **without changing a
single character of the rendered output**.  The extraction exists because AgentRec-X
2.0-alpha runs the accepted recommendation pipeline from two control planes:

* :class:`~recommendation.agent.graph.AgentGraph` - the accepted Milestone 7B-10D DAG;
* :class:`~recommendation.control.loop.LoopController` - the 2.0-alpha bounded agent
  loop.

Both must render a run with the *same* code.  Duplicating the presentation layer would
let the two control planes drift apart, which is exactly what the milestone forbids:
Stage 1 changes **who decides the next step**, never how a recommendation is produced or
presented.

Everything here is a pure function of already-computed artifacts.  Nothing in this module
scores, ranks, filters, retrieves, matches, reranks or mutates a candidate: it turns
structured state into text and keeps the honest disclaimers (raw model scores are ranking
scores, metadata facts are quoted, policy adherence is not relevance) attached.

Trust boundary: no function here receives trusted interaction history, internal item ids,
mapping internals, checkpoint internals or a memory store.
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "NO_CANDIDATES_TEXT",
    "SCORE_DISCLAIMER",
    "FIELD_LABELS",
    "RERANK_HEADER",
    "RERANK_FOOTER",
    "METADATA_PROVENANCE_ROOT",
    "active_preference_lines",
    "alignment_pairs",
    "build_grounded_response",
    "build_recommendation_response",
    "build_reranked_response",
    "evidence_line",
    "field_label",
    "memory_context",
    "movement_line",
    "preference_block",
    "preference_block_for_reranking",
    "shorten",
]

#: Text used when the recommender legitimately has no eligible candidate left.
#: Candidate exhaustion is a normal outcome, never turned into an error or padded
#: with fabricated items.
NO_CANDIDATES_TEXT = (
    "No unseen product is left to recommend from this interaction history, so "
    "there are no candidates to show."
)

#: Footer that stops raw model scores from being read as product evidence.  The
#: recommender's scores order candidates; they say nothing about a product.
SCORE_DISCLAIMER = (
    "Note: these are raw sequential-model ranking scores used only to order the "
    "candidates. They are not probabilities or confidence values, and they are not "
    "evidence about a product's attributes, quality or availability."
)

#: Provenance root of the catalogue metadata used by the Milestone 8 enricher.
METADATA_PROVENANCE_ROOT = "amazon_reviews_2023:meta_categories"

#: Field labels used when rendering grounded metadata facts.
FIELD_LABELS: dict[str, str] = {
    "title": "Title",
    "store": "Store",
    "main_category": "Main category",
    "categories": "Category",
    "features": "Feature",
    "description": "Description",
    "details": "Detail",
}

#: Header for the final response when M10D reranking is active.  It names the policy
#: and nothing else: no "best", no "relevant", no quality word.
RERANK_HEADER = (
    "candidate(s) from the sequential recommender, ordered by the configured "
    "explicit-preference policy (fewer supported violations first, then more supported "
    "preference matches, then the original ranking order), with catalogue facts where "
    "available:"
)

#: Footer clauses for the reranked response.  Each is a statement about *policy
#: adherence*, never about product quality or relevance.
RERANK_FOOTER = (
    "Original SASRec ranks are shown so the upstream order stays auditable; the "
    "recommendation itself always comes from the sequential recommender.",
    "A preference match or violation describes what this candidate's catalogue "
    "metadata says about your stored explicit preferences. It is not a product-quality "
    "judgement, and policy adherence is not a relevance or satisfaction measure.",
)


def active_preference_lines(snapshot: Any) -> list[str]:
    """Render a preference snapshot as short, attributed constraint lines."""
    if snapshot is None:
        return []
    lines: list[str] = []
    for entry in snapshot.active_entries:
        marker = "prefers" if entry.polarity.value == "prefer" else "avoids"
        lines.append(f"{entry.kind.value}: {marker} {entry.value}")
    return lines


def memory_context(query: str, snapshot: Any) -> str:
    """Augment a retrieval query with active explicit preferences.

    Format is explicit and documented rather than implicit::

        <user query>
        preferences:
        - <kind>: <prefer|avoid> <value>
        - ...

    This only changes *which evidence fragments* are selected from the metadata of
    the already-fixed candidate set.  It cannot add, drop or reorder a candidate, and
    it never touches trusted interaction history.
    """
    lines = active_preference_lines(snapshot)
    if not lines:
        return query
    return "\n".join([query, "preferences:", *[f"- {line}" for line in lines]])


def preference_block(preferences: Any) -> list[str]:
    """Render the user's stored preferences as an honest, clearly-labelled block.

    The block states what the user said.  It deliberately contains **no match score,
    no "perfectly matches" claim and no ranking hint** -- Milestone 9 has no
    preference-to-product scoring, and presenting one would be fabrication.
    """
    lines = active_preference_lines(preferences)
    if not lines:
        return []
    return [
        "Your stated preferences (stored from your own messages; not used to rank "
        "these candidates):",
        *[f"- {line}" for line in lines],
        "",
    ]


def preference_block_for_reranking(preferences: Any) -> list[str]:
    """Render the stored preferences that the reranking policy consumed.

    Unlike :func:`preference_block` this says the preferences **were** used, because
    with M10D active they are exactly what the matching node evaluated.  It still makes
    no claim that a candidate satisfies them; that claim appears per candidate, and only
    where M10A produced it.
    """
    lines = active_preference_lines(preferences)
    if not lines:
        return []
    return [
        "Your stated preferences (stored from your own messages; evaluated as explicit "
        "preference evidence by the reranking policy):",
        *[f"- {line}" for line in lines],
        "",
    ]


def build_recommendation_response(result: Any, preferences: Any = None) -> str:
    """Render a Tool result as candidate lines plus an honest score disclaimer."""
    if not result.recommendations:
        return NO_CANDIDATES_TEXT

    lines = [
        f"Top {result.returned_k} candidate(s) from the sequential recommender:",
        "",
        *preference_block(preferences),
    ]
    lines.extend(
        f"{item.rank}. {item.parent_asin} (score {item.score:+.4f})"
        for item in result.recommendations
    )
    if result.returned_k < result.requested_k:
        lines.extend(
            [
                "",
                f"Only {result.returned_k} of {result.requested_k} requested candidates "
                "were available after excluding items already in the history.",
            ]
        )
    lines.extend(["", SCORE_DISCLAIMER])
    return "\n".join(lines)


def shorten(text: str, limit: int = 240) -> str:
    """Trim a rendered fact for display, marking the elision explicitly.

    Only presentation is shortened; the structured evidence keeps the full verbatim
    text, so nothing is lost from the grounding record.
    """
    if len(text) <= limit:
        return text
    return text[: limit - 1].rstrip() + "…"


def field_label(evidence: Any) -> str:
    """Return the display label for one evidence fragment."""
    label = FIELD_LABELS.get(evidence.field, evidence.field)
    if evidence.detail_key:
        label = evidence.detail_key
    return label


def build_grounded_response(enrichment: Any, preferences: Any = None) -> str:
    """Render enriched candidates with attributed catalogue facts.

    Milestone 8 is the first point where product attribute claims may appear, and
    only facts actually present in normalized metadata are printed.  A candidate the
    source does not cover is shown with its identity and an explicit
    "metadata unavailable" note rather than an invented description.  The raw SASRec
    score is labelled as a ranking score, never as a probability or a rating.
    """
    if not enrichment.items:
        return NO_CANDIDATES_TEXT

    lines = [
        f"Top {enrichment.returned_k} candidate(s) from the sequential recommender, "
        "with catalogue facts where available:",
        "",
        *preference_block(preferences),
    ]

    for item in enrichment.items:
        lines.extend(
            ["", f"{item.rank}. {item.parent_asin} (ranking score {item.score:+.4f})"]
        )
        if item.metadata_status == "missing":
            lines.append("   metadata unavailable for this item")
            continue
        if not item.evidence:
            reason = item.fallback_reason or "no_evidence"
            lines.append(f"   no matching catalogue detail retrieved ({reason})")
            continue
        for evidence in item.evidence:
            lines.append(f"   {field_label(evidence)}: {shorten(evidence.text)}")

    lines.extend(
        [
            "",
            "Facts above are quoted from Amazon Reviews 2023 product metadata for the "
            "listed item and are not present for every item.",
            SCORE_DISCLAIMER,
            "Evidence is selected only from these candidates; the ranking itself comes "
            "from the sequential recommender.",
        ]
    )
    return "\n".join(lines)


def alignment_pairs(reranking: Any, enrichment: Any) -> list[tuple[Any, Any]]:
    """Pair each reranked candidate with the enriched candidate of the same identity.

    Alignment is by ``(parent_asin, item_id)``, never by position, so a reranked order
    of ``C, A, B`` prints ``C``'s metadata next to ``C`` rather than whatever metadata
    happened to sit at that index.  Any mismatch is a hard error: the graph must not
    print one product's facts beside another product's identity, and it must not paper
    over a reranker that changed the candidate set.

    The raised type is :class:`~recommendation.agent.graph.AgentGraphError`; it is
    imported lazily so this presentation module does not depend on the graph module at
    import time.
    """
    from .graph import AgentGraphError

    if len(reranking.candidates) != len(enrichment.items):
        raise AgentGraphError(
            "reranking changed the candidate count "
            f"({len(reranking.candidates)} != {len(enrichment.items)})"
        )

    by_identity = {
        (item.parent_asin, item.recommendation.item_id): item for item in enrichment.items
    }
    if len(by_identity) != len(enrichment.items):  # pragma: no cover - defensive
        raise AgentGraphError("the enriched candidate set contains a duplicate identity")

    pairs: list[tuple[Any, Any]] = []
    for candidate in reranking.candidates:
        item = by_identity.get((candidate.parent_asin, candidate.item_id))
        if item is None:
            raise AgentGraphError(
                "reranking produced a candidate that is not in the enriched candidate set"
            )
        pairs.append((candidate, item))
    return pairs


def movement_line(candidate: Any) -> str | None:
    """Describe a rank movement using only directly supported facts.

    Deliberately does **not** use the M10B ``rerank_reason`` label: Milestone 10C
    established that its tail ``DETERMINISTIC_TIE_BREAK`` wording can be imprecise (the
    candidate actually lost on the original-rank key), so it is not used as a
    user-facing explanation.  Only ``original_rank``/``reranked_rank`` -- facts the
    reranker really produced -- are stated.
    """
    if not candidate.moved:
        return None
    return (
        f"   moved from rank {candidate.original_rank} to rank {candidate.reranked_rank} "
        "under the configured explicit-preference policy"
    )


def evidence_line(candidate: Any) -> str:
    """The per-candidate M10A evidence counts, stated as counts only."""
    return (
        f"   preference evidence: {candidate.match_count} supported match(es), "
        f"{candidate.violation_count} supported violation(s), "
        f"{candidate.unknown_count} unknown"
    )


def build_reranked_response(
    reranking: Any, enrichment: Any, preferences: Any = None
) -> str:
    """Render the recommendation in **reranked** order, grounded and auditable.

    The candidate sequence follows ``reranking.candidates``; the catalogue facts come
    from each candidate's *own* enriched record, matched by identity.  No metadata is
    looked up again and no ranked value is recomputed -- this function only decides
    presentation.
    """
    if not reranking.candidates:
        return NO_CANDIDATES_TEXT

    pairs = alignment_pairs(reranking, enrichment)

    lines = [
        f"Top {len(pairs)} {RERANK_HEADER}",
        "",
        *preference_block_for_reranking(preferences),
    ]

    for candidate, item in pairs:
        lines.extend(
            [
                "",
                f"{candidate.reranked_rank}. {candidate.parent_asin} "
                f"(original SASRec rank {candidate.original_rank}, "
                f"ranking score {candidate.sasrec_score:+.4f})",
                evidence_line(candidate),
            ]
        )
        movement = movement_line(candidate)
        if movement is not None:
            lines.append(movement)

        if item.metadata_status == "missing":
            lines.append("   metadata unavailable for this item")
            continue
        if not item.evidence:
            reason = item.fallback_reason or "no_evidence"
            lines.append(f"   no matching catalogue detail retrieved ({reason})")
            continue
        for evidence in item.evidence:
            lines.append(f"   {field_label(evidence)}: {shorten(evidence.text)}")

    lines.extend(
        [
            "",
            "Facts above are quoted from Amazon Reviews 2023 product metadata for the "
            "listed item and are not present for every item.",
            *RERANK_FOOTER,
            SCORE_DISCLAIMER,
        ]
    )
    return "\n".join(lines)

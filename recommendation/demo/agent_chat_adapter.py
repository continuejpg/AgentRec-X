"""Agent -> Chat response adapter (V2 E-01A-3A, extended by E-01A-3B).

Converts one **already executed** :class:`AgentRecommendResponse` into the public
:class:`ChatResponse` without inventing anything.  The rules, in order of importance:

* product identities, their order and their provenance come from the controller's own response -
  this module never adds, drops, reorders or re-scores a candidate;
* a failed or empty run stays failed or empty: ``route`` and the recommendation list are copied
  from the response, so a refused completion can never look like a successful recommendation.
  What this module adds is the *distinction*: :class:`AgentExecutionView` records whether the plan
  asked for a direct turn, produced candidates, found none, or was refused, and the answer text
  falls back to an authored sentence for the outcomes whose own text is empty - never to silence,
  which would read as a successful direct turn;
* nothing is fabricated to satisfy a legacy field.  ``RecommendationCard.sasrec_score`` is the
  only score slot the legacy card has, so the controller's real score is placed there **and**
  ``score_kind`` is always written (defaulting to :data:`AGENT_SCORE_KIND`), which is what tells a
  consumer that the number is not necessarily a SASRec logit;
* catalogue evidence is reported as found/missing exactly as the response does (a missing title is
  ``"missing"``, never an invented one).  ``metadata`` is filled with a **title-only**
  :class:`ProductMetadataView` when the response carries a title - every other attribute keeps its
  ``None`` default, which a client renders as unavailable - because completing the rest would be
  fabrication;
* the run's execution record (terminal state, executed actions with their counts, grounded
  provenance, timing) travels in :class:`AgentExecutionView`, the additive optional field the chat
  contract reserves for it.  It is *not* squeezed into :class:`RecommendationTrace`, whose fields
  mean specific other things, and :func:`agent_execution_digest` still returns the same facts as
  plain data for a caller that logs or asserts on them.

``build_chat_response(state)`` is untouched: this is a separate, direct adapter, because forging a
graph-shaped ``state`` just to reuse that builder would mean inventing the very fields the rules
above forbid.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from recommendation.demo.agent_schemas import AgentExecutionView, AgentRecommendResponse
from recommendation.demo.schemas import (
    DECISION_MODE_LLM_AGENT,
    ActivePreferenceView,
    AuditView,
    ChatResponse,
    KnowledgeEvidenceView,
    MemoryUpdateView,
    ProductFeedbackView,
    ProductMetadataView,
    RagAuditView,
    ReplanAuditView,
    RecommendationCard,
    RecommendationTrace,
    TraceMemoryChanges,
)
from recommendation.demo.serialization import evidence_views, movement_summary

#: The label written when the agent response does not name its own score kind: the number is the
#: controller's score, which may come from any source or from fusion - not a SASRec logit.
AGENT_SCORE_KIND = "agent_controller_score"

#: The terminal reasons that mean the completion guard refused to end the run.  Named here
#: rather than imported from the control plane: this module projects a *wire* fact, and the
#: values are part of the controller's published vocabulary
#: (``TerminationReason.COMPLETION_REFUSED`` / ``NO_VERIFIED_CANDIDATE``).
COMPLETION_REFUSED = "completion_refused"
NO_VERIFIED_CANDIDATE = "no_verified_candidate"

#: What the turn answers when the controller's own text is empty and the run produced nothing.
#: Each states only what the run's own record proves - no apology, no cause the record does not
#: carry, and never a product.  The controller's text is used whenever it rendered one.
NO_CANDIDATES_MESSAGE = (
    "The recommendation run found nothing it could verify for that request, so there are no "
    "products to show."
)
REFUSED_MESSAGE = (
    "The recommendation run could not be completed: its completion guard refused to finish it, "
    "so no products are being recommended. The run's own reason is in this response's agent "
    "block."
)
FAILED_MESSAGE = (
    "The recommendation run did not complete, so no products are being recommended. The run's "
    "own status and reason are in this response's agent block."
)

#: The authored text per non-successful outcome.
OUTCOME_MESSAGES: dict[str, str] = {
    "no_candidates": NO_CANDIDATES_MESSAGE,
    "refused": REFUSED_MESSAGE,
    "failed": FAILED_MESSAGE,
}


#: What a product's line says when the explanation step produced no validated reason for it.
#: Stated instead of a summary, so an unexplained product is never given an invented one.
NO_VERIFIED_REASON = "no verified catalogue reason for this request"


def recommendation_narration(cards: Sequence[RecommendationCard]) -> str:
    """The recommendation prose, rendered from the authoritative cards and nothing else.

    This is the one place an ``llm_agent`` recommendation turn's text is produced, and it reads
    only the response's own final card list: the products that survived the accepted fusion,
    the memory-aware ranking and the presentation limit, in the order the response presents them.
    It therefore cannot name a product the run retrieved but did not show, cannot repeat a rank the
    ranking changed, and cannot state a product fact that the explanation step did not validate -
    a card without a validated ``reason`` gets :data:`NO_VERIFIED_REASON`, never a written summary.

    The controller's own narration is deliberately **not** reused here.  It is rendered inside the
    run, before the multi-source accumulation, the preference ranking and the ``k`` limit are
    applied, so it describes the run's intermediate candidate state; reusing it is what produced a
    prose list that disagreed with the cards.  Non-recommendation turns (a direct answer, a run with
    no candidates, a refused or failed run) keep that accepted narration: this function never runs
    for them, because there is no card to render.

    The per-line tail quotes only card fields - the ranking score and its kind, the sources that
    proposed the product, and the accepted M10A counts.  Those are the run's own published facts,
    not claims about the product.
    """
    lines = [
        f"Top {len(cards)} recommendation{'s' if len(cards) != 1 else ''} from the sources this "
        "turn consulted, in the order this response presents them:",
        "",
    ]
    for position, card in enumerate(cards, start=1):
        reason = str(card.reason or "").strip() or NO_VERIFIED_REASON
        sources = ", ".join(str(source) for source in card.fusion_sources) or "none recorded"
        score_kind = card.score_kind or "unspecified"
        lines.append(f"{position}. {card.parent_asin} (item {card.item_id}) - {reason}")
        lines.append(
            f"   ranking score {card.sasrec_score} ({score_kind}); sources: {sources}; "
            f"preference evidence: {card.match_count} match(es), {card.violation_count} "
            f"violation(s), {card.unknown_count} unknown"
        )
    if any(str(card.reason or "").strip() for card in cards):
        lines.extend(
            ["", "Each reason above was validated against that product's own catalogue evidence."]
        )
    return "\n".join(lines)


def agent_outcome(response: AgentRecommendResponse, *, proposed_route: str | None) -> str:
    """Which of the closed outcomes this executed run was, from the run's own record.

    The decision, in order, and every branch reads execution state rather than the request:

    1. a run that produced candidates presents them: ``recommended`` - checked first, because a
       caller that does not know the plan cannot make presented products look like a direct turn;
    2. a plan that did not ask for a recommendation never ran a controller: ``direct``;
    3. a run that stopped without candidates and whose terminal says the completion guard refused
       it is ``refused`` - the controller's own ``termination_reason``, never an inference from an
       empty list;
    4. a run that ended without a terminal success and was not refused is ``failed`` (a capability
       failure, an invalid action or an exhausted budget - the terminal block carries which);
    5. anything else finished with nothing to present: ``no_candidates``.
    """
    if response.recommendations:
        return "recommended"
    if proposed_route != "recommend":
        return "direct"
    terminal = response.terminal
    if not terminal.succeeded:
        if str(terminal.termination_reason or "") in (
            COMPLETION_REFUSED,
            NO_VERIFIED_CANDIDATE,
        ):
            return "refused"
        return "failed"
    return "no_candidates"


def agent_execution_view(
    response: AgentRecommendResponse,
    *,
    planned_sources: Sequence[str] = (),
    proposed_route: str | None = None,
) -> AgentExecutionView:
    """The executed run as the chat contract carries it: the plan, what ran, and how it ended.

    ``action_counts`` counts **executed** steps only (``TrajectoryStepView.validated``), so a
    proposal the validator refused is not reported as a run, while a source the controller legally
    consulted more than once is reported with its real count.  ``finish`` is counted too: it is an
    action kind that really executed.
    """
    trajectory = response.trajectory
    counts: dict[str, int] = {}
    for step in trajectory.steps or ():
        if step.validated and step.action:
            counts[str(step.action)] = counts.get(str(step.action), 0) + 1
    return AgentExecutionView(
        outcome=agent_outcome(response, proposed_route=proposed_route),
        planned_sources=tuple(str(name) for name in planned_sources),
        controller_executed=True,
        action_counts=counts,
        terminal=response.terminal,
        trajectory=trajectory,
        grounded=response.grounded,
        timing=response.timing,
    )


def direct_agent_execution(planned_sources: Sequence[str] = ()) -> AgentExecutionView:
    """The execution record of a direct turn: the plan is known, and nothing else happened.

    ``controller_executed=False`` and every controller-owned block ``None`` are the honest
    description of a turn that ran no recommendation controller - and they are exactly what
    distinguishes it from a recommendation the controller could not complete.
    """
    return AgentExecutionView(
        outcome="direct",
        planned_sources=tuple(str(name) for name in planned_sources),
        controller_executed=False,
        action_counts={},
        terminal=None,
        trajectory=None,
        grounded=None,
        timing=None,
    )


def _explanation_fields(
    candidate: Any, explanations: Mapping[str, Any] | None
) -> dict[str, Any]:
    """The card's explanation fields, from a validated outcome - never from model text directly."""
    if explanations is None:
        return {}
    # The outcome is an ``ExplanationOutcome`` (or anything with the same read-only shape): the
    # adapter asks it about one identity rather than re-deriving anything itself.
    lookup = getattr(explanations, "for_identity", None)
    outcome = (
        lookup(str(candidate.parent_asin)) if callable(lookup)
        else explanations.get(str(candidate.parent_asin))
    )
    if outcome is None:
        return {}
    evidence = tuple(
        KnowledgeEvidenceView(
            evidence_id=str(unit.evidence_id),
            parent_asin=str(unit.parent_asin),
            item_id=int(unit.item_id),
            field_path=str(unit.field_path),
            value=str(unit.value),
            source=str(unit.source),
            retrieval_score=float(unit.retrieval_score),
            truncated=bool(unit.truncated),
        )
        for unit in getattr(outcome, "units", ()) or ()
    )
    return {
        "reason": getattr(outcome, "reason", None),
        "reason_status": str(getattr(outcome, "status", "not_attempted")),
        "knowledge_evidence": evidence,
    }


def _replan_audit(decision: Any, execution: Any = None) -> ReplanAuditView | None:
    """The gate's decision as the chat contract's compact block (E-03A).

    ``None`` when the run produced no decision (a direct turn runs no controller), so the block is
    absent rather than defaulted - "no controller ran" and "the gate said no" are different facts.
    """
    if decision is None:
        return None
    audit = ReplanAuditView(
        continue_recommended=bool(getattr(decision, "continue_recommended", False)),
        reason_code=str(getattr(decision, "reason_code", "") or ""),
        eligible_sources=tuple(str(name) for name in getattr(decision, "eligible_sources", ()) or ()),
        remaining_tool_budget=int(getattr(decision, "remaining_tool_budget", 0) or 0),
        remaining_step_budget=int(getattr(decision, "remaining_step_budget", 0) or 0),
        decision_source=str(
            getattr(decision, "decision_source", "deterministic_rules") or "deterministic_rules"
        ),
    )
    if execution is None:
        return audit
    # E-03B: what bounded replanning really did inside the run, alongside the terminal gate's
    # own (post-run) decision.  The two are different facts and are reported as such.
    initial = getattr(execution, "initial", None)
    final = getattr(execution, "final", None)
    return audit.model_copy(
        update={
            "status": str(getattr(execution, "status", "") or "") or None,
            "model_called": bool(getattr(execution, "model_called", False)),
            "chosen_source": getattr(execution, "chosen_source", None),
            "initial_candidate_count": int(getattr(initial, "candidates", 0) or 0),
            "final_candidate_count": int(getattr(final, "candidates", 0) or 0),
        }
    )


def _card_for(
    candidate: Any,
    explanations: Mapping[str, Any] | None = None,
    feedback_marks: Mapping[str, str] | None = None,
) -> RecommendationCard:
    """One card from one controller candidate.  Only fields the response actually carries.

    The card's ``metadata`` is filled with a **title-only** catalogue view when the response
    carries a title, and left unset when it does not.  That is the honest use of the field: every
    other attribute of :class:`ProductMetadataView` keeps its ``None`` default, which the client
    renders as "unavailable", so a title can be shown without inventing a store, a category, a
    price or a rating.  ``metadata_status`` then reports whether the trusted catalogue knew the
    identity at all.
    """
    title = getattr(candidate, "title", None)
    score_kind = getattr(candidate, "score_kind", None) or AGENT_SCORE_KIND
    # The memory-aware ranking, when it ran.  ``original_rank`` stays the run's baseline (fused)
    # position, ``reranked_rank`` is the accepted M10B position, and the evidence is M10A's own -
    # none of it rewrites the score, the score kind or the provenance copied below.
    reranked_rank = getattr(candidate, "reranked_rank", None)
    original_rank = int(candidate.rank)
    return RecommendationCard(
        original_rank=original_rank,
        parent_asin=str(candidate.parent_asin),
        item_id=int(candidate.item_id),
        # The controller's real score; ``score_kind`` below is what identifies it honestly.
        sasrec_score=float(candidate.score),
        score_kind=str(score_kind),
        fusion_sources=tuple(getattr(candidate, "sources", ()) or ()),
        reranked_rank=None if reranked_rank is None else int(reranked_rank),
        match_count=int(getattr(candidate, "match_count", 0) or 0),
        violation_count=int(getattr(candidate, "violation_count", 0) or 0),
        unknown_count=int(getattr(candidate, "unknown_count", 0) or 0),
        evidence=evidence_views(getattr(candidate, "evidence", ()) or ()),
        movement_summary=movement_summary(
            original_rank, None if reranked_rank is None else int(reranked_rank)
        ),
        # Title only: the response carries no other catalogue fact, and completing the rest would
        # be fabrication.  A candidate with no title keeps ``metadata`` unset.
        metadata=ProductMetadataView(title=str(title)) if title else None,
        metadata_status="found" if title else "missing",
        # Grounded explanation (V2 E-02B), when the capability ran for this turn.  The reason and
        # its citations are separate from the preference evidence above, by construction.
        **_explanation_fields(candidate, explanations),
        # The user's own attitude (V2 E-03C), when this turn was ranked under one.  It is not a
        # catalogue fact, so it never enters ``reason`` or ``knowledge_evidence``.
        feedback=None if feedback_marks is None else feedback_marks.get(str(candidate.parent_asin)),
    )


def _audit_for(
    cards: Sequence[RecommendationCard],
    *,
    ranked_with: Sequence[ActivePreferenceView] = (),
    rag: RagAuditView | None = None,
    replan: Any | None = None,
    execution: Any | None = None,
    feedback: Mapping[str, Any] | None = None,
) -> AuditView:
    """The audit block, derived from the cards the response actually carries.

    Its semantics are the accepted ones (see ``serialization.build_audit``): ``original_order`` is
    read from each card's unchanged baseline rank, ``reranked_order`` from the memory-aware position,
    and the two are equal - with ``reranking_applied=False`` - when no card carries one.  A card's
    ``reranked_rank`` is only ever set by the memory-aware ranking step, so this block cannot claim
    a reranking that did not run, and ``ranked_with_preferences`` names the ACTIVE preferences the
    *ranking* was performed under (which, because a turn reads memory before it writes it, is the
    pre-write set - a turn never ranks with its own new statement).
    """
    applied = any(card.reranked_rank is not None for card in cards)
    original_order = tuple(
        card.parent_asin for card in sorted(cards, key=lambda card: card.original_rank)
    )
    ranked_with_views = tuple(ranked_with) if applied else ()
    return AuditView(
        reranking_applied=applied,
        candidate_count=len(cards),
        moved_count=sum(
            1
            for card in cards
            if card.reranked_rank is not None and card.reranked_rank != card.original_rank
        ),
        original_order=original_order,
        reranked_order=(
            tuple(card.parent_asin for card in sorted(cards, key=lambda card: card.reranked_rank or 0))
            if applied
            else original_order
        ),
        ranked_with_preference_count=len(ranked_with_views),
        ranked_with_preferences=ranked_with_views,
        rag=rag,
        replan=_replan_audit(replan, execution),
        feedback=None if feedback is None else ProductFeedbackView(**feedback),
    )


def agent_execution_digest(response: AgentRecommendResponse) -> dict[str, Any]:
    """The run's real execution evidence, as plain data.

    Everything here is copied from the response: a caller may store, log or project it, but the
    adapter will not pretend the chat schema has a field for it.
    """
    terminal = response.terminal
    trajectory = response.trajectory
    grounded = response.grounded
    return {
        "control_plane": response.control_plane,
        "api_version": response.api_version,
        "terminal": {
            "run_id": terminal.run_id,
            "status": terminal.status,
            "succeeded": bool(terminal.succeeded),
            "termination_reason": terminal.termination_reason,
            "termination_detail": terminal.termination_detail,
            "steps": int(terminal.steps),
            "tool_calls": int(terminal.tool_calls),
            "retries": int(terminal.retries),
            "limits": dict(terminal.limits or {}),
        },
        "trajectory": {
            "step_count": int(trajectory.step_count),
            "actions": list(trajectory.actions or ()),
            "refusals": list(trajectory.refusals or ()),
        },
        "grounding": {
            "grounded_count": int(grounded.grounded_count),
            "ungrounded_count": int(grounded.ungrounded_count),
            "multi_source_count": int(grounded.multi_source_count),
            "sources_present": list(grounded.sources_present or ()),
        },
        "titles": {str(candidate.parent_asin): candidate.title
                   for candidate in (response.recommendations or ())
                   if getattr(candidate, "title", None)},
        "timing_ms": {
            "total_ms": float(response.timing.total_ms),
            "agent_ms": float(response.timing.agent_ms),
            "serialize_ms": float(response.timing.serialize_ms),
        },
    }


def build_agent_chat_response(
    response: AgentRecommendResponse,
    *,
    session_id: str,
    turn_id: str,
    turn: int,
    message: str | None = None,
    active_preferences: Sequence[ActivePreferenceView] = (),
    memory_update: MemoryUpdateView | None = None,
    ranked_with: Sequence[ActivePreferenceView] = (),
    explanations: Mapping[str, Any] | None = None,
    rag: RagAuditView | None = None,
    provider: str | None = None,
    model: str | None = None,
    proposed_route: str | None = None,
    planned_sources: Sequence[str] = (),
    feedback: Mapping[str, Any] | None = None,
    feedback_marks: Mapping[str, str] | None = None,
) -> ChatResponse:
    """Convert one executed agent run into the chat contract.

    ``session_id``/``turn_id``/``turn`` belong to the server-owned session layer and are therefore
    explicit parameters: the agent response cannot know them, and guessing them would be a lie.

    ``message`` is the turn's prose, and which prose it is depends on what the turn is:

    * a turn that **presents products** narrates them from its own final cards, in their order
      (:func:`recommendation_narration`).  The run's intermediate narration is not reused, because
      it is rendered before the multi-source accumulation, the preference ranking and the ``k``
      limit, so it can name products the response does not show and ranks the ranking changed;
    * every other turn - a direct answer, a run that produced no candidates, a refused run, a failed
      run - keeps the run's own rendered text, falling back to the authored text of its outcome only
      when that text is empty (a refused run renders nothing, and answering with an empty string
      would read as a successful direct turn with nothing to say);
    * an explicit ``message`` argument still overrides both, for a caller that has its own text.

    ``provider``/``model``/``proposed_route``/``planned_sources`` are the decision facts of the
    validated tool plan that authorised this run.  They are passed in rather than invented here:
    the agent response describes what *executed* and cannot know which model planned it, and the
    trace reports the proposed route next to the executed one precisely so the two can be compared
    - including the case where a run produced no candidate set and the honest executed route is
    ``direct``.
    """
    cards = tuple(
        _card_for(candidate, explanations, feedback_marks)
        for candidate in (response.recommendations or ())
    )
    update = memory_update or MemoryUpdateView()
    preferences = tuple(active_preferences)
    agent = agent_execution_view(
        response, planned_sources=planned_sources, proposed_route=proposed_route
    )
    audit = _audit_for(
        cards,
        ranked_with=ranked_with,
        rag=rag,
        replan=response.replan,
        execution=getattr(response, "replan_execution", None),
        feedback=feedback,
    )
    if message is not None:
        # An explicit caller override still wins; the runtime never passes one today.
        text = str(message)
    elif cards:
        # A recommendation that presents products narrates exactly those products, in their final
        # order.  See ``recommendation_narration`` for why the run's own text is not reused.
        text = recommendation_narration(cards)
    else:
        # A direct answer, a run with no candidates, a refused or a failed run: the accepted
        # narration (or the authored text for its outcome) is the honest answer and is unchanged.
        text = str(response.message or "")
        if not text.strip():
            text = OUTCOME_MESSAGES.get(agent.outcome, "")
    return ChatResponse(
        session_id=session_id,
        turn_id=turn_id,
        turn=turn,
        route=response.route,
        message=text,
        active_preferences=preferences,
        memory_update=update,
        recommendations=cards,
        audit=audit,
        trace=_trace_for(
            response,
            cards=cards,
            active_preferences=preferences,
            memory_update=update,
            provider=provider,
            model=model,
            proposed_route=proposed_route,
        ),
        agent=agent,
    )


def _trace_for(
    response: AgentRecommendResponse,
    *,
    cards: Sequence[RecommendationCard],
    active_preferences: Sequence[ActivePreferenceView],
    memory_update: MemoryUpdateView,
    provider: str | None,
    model: str | None,
    proposed_route: str | None,
) -> RecommendationTrace:
    """The decision facts of this run, in the chat contract's existing trace view.

    Only facts the plan and the response carry are placed here, and each one is named:

    * ``decision_mode`` is this mode, because that is the policy that served the turn;
    * ``proposed_route`` is the route the validated plan proposed, next to the response's executed
      ``route``, so a plan that asked for products and a run that produced none stay comparable;
    * ``candidate_count`` is the number of cards the response actually carries, so an empty or
      refused run reports no candidates rather than zero-of-something;
    * ``memory_changes`` is derived from the same ``memory_update`` the response carries, so the
      trace cannot disagree with it.  The tool-selection path runs memory-free, which is why that
      view is empty there - see :mod:`recommendation.demo.llm_agent`;
    * ``preference_actions`` stays empty: a tool plan proposes tools, never preference mutations;
    * ``evidence`` stays ``None``: it is meaningful only for a turn that ranked against the
      start-of-turn preference snapshot, and this path loads none.

    The run's *execution* record deliberately does **not** live here.  Terminal state, executed
    actions and grounding counts are not decision facts, and the trace's fields mean specific
    other things, so they travel in :class:`~recommendation.demo.agent_schemas.AgentExecutionView`
    instead - one home per fact, and no field asked to mean two things.
    """
    return RecommendationTrace(
        route=response.route,
        candidate_count=len(cards) or None,
        decision_mode=DECISION_MODE_LLM_AGENT,
        provider=provider,
        model=model,
        proposed_route=proposed_route,
        preference_actions=(),
        active_preferences=tuple(active_preferences),
        memory_changes=TraceMemoryChanges(
            changed=memory_update.changed,
            added=tuple(item.value for item in memory_update.added),
            removed=tuple(item.value for item in memory_update.removed),
            superseded=tuple(item.value for item in memory_update.superseded),
        ),
        evidence=None,
        ranking_changes=(),
        source=None,
        grounding=None,
    )


__all__ = [
    "AGENT_SCORE_KIND",
    "NO_VERIFIED_REASON",
    "OUTCOME_MESSAGES",
    "agent_execution_digest",
    "agent_execution_view",
    "agent_outcome",
    "build_agent_chat_response",
    "direct_agent_execution",
    "recommendation_narration",
]

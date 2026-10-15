"""Projection of one agent run into the JSON shapes the agent endpoint returns.

Why a projection and not a serializer
-------------------------------------
The controller already records an inspectable trajectory
(:class:`~recommendation.control.trajectory.TrajectoryRecorder`), and that record is the
single source of truth for "why did this step happen".  This module does **not** add a
second tracing system: it reads those existing records plus the run's existing
``candidate_eligibility`` / ``constraint_projection`` state and the existing
:class:`~recommendation.control.CandidateLedger`, and reshapes them into the six fields the
packaging phase asked for.

What is deliberately dropped
----------------------------
Every input here is already payload-light, and the projection narrows it further: the
policy's own arguments, the observation payload, the memory update and the rendered
recommendation text are all left out.  A demo client needs to see *that* a step happened,
*what kind* of observation it produced and *how many* candidates it saw - not the raw
domain payload the loop keeps for its own rendering.

Reused, not re-derived
----------------------
* counts and identities come from ``CandidateLedger`` (grounded/ungrounded/multi-source);
* eligibility comes from the controller's stored ``candidate_eligibility`` view and its
  ``constraint_projection``, both produced by ``refresh_feasibility`` - this module never
  re-evaluates a constraint, because a second evaluation would be a second verdict;
* the per-source recommendation provenance is read from the same ledger entries.
"""

from __future__ import annotations

from typing import Any, Mapping

from .agent_schemas import (
    AgentCandidateView,
    AgentEligibilityView,
    AgentRecommendationView,
    AgentTerminalView,
    GroundedCandidateView,
    TrajectoryStepView,
    TrajectorySummaryView,
)

__all__ = [
    "build_eligibility_view",
    "build_grounded_view",
    "build_recommendations_view",
    "build_terminal_view",
    "build_trajectory_summary",
]


def build_trajectory_summary(result: Any) -> TrajectorySummaryView:
    """Project a :class:`~recommendation.control.LoopResult` into the compact view.

    ``result`` is the loop result object itself, so this reads the controller's own
    ``trajectory`` records rather than re-deriving steps from state.
    """
    steps: list[TrajectoryStepView] = []
    for record in result.trajectory.steps:
        proposal = record.action_proposal or {}
        context = record.policy_context_summary or {}
        observation = record.observation or {}
        refusal_codes: list[str] = []
        for check in (record.validation_result, record.verification_result):
            if check is not None and not check.verified:
                refusal_codes.append(str(check.code))
        steps.append(
            TrajectoryStepView(
                step=int(record.step_index),
                action=str(proposal["action"]) if proposal.get("action") else None,
                validated=record.validated_action is not None,
                observation_type=_observation_type(observation),
                candidate_count=_observation_candidate_count(observation, context),
                eligibility_state=_eligibility_state(context),
                refusal_codes=tuple(refusal_codes),
                note=record.note,
            )
        )
    return TrajectorySummaryView(
        step_count=result.steps,
        actions=result.trajectory.actions(),
        refusals=result.trajectory.refusals(),
        steps=tuple(steps),
    )


def build_terminal_view(result: Any) -> AgentTerminalView:
    """Project the controller's terminal control state."""
    control = result.control
    limits = control.limits
    return AgentTerminalView(
        run_id=control.run_id,
        status=control.status.value,
        termination_reason=(
            None if control.termination_reason is None else control.termination_reason.value
        ),
        termination_detail=control.termination_detail,
        succeeded=result.succeeded,
        steps=int(control.step_count),
        tool_calls=int(control.tool_call_count),
        retries=int(control.retry_count),
        limits={
            "max_steps": int(limits.max_steps),
            "max_tool_calls": int(limits.max_tool_calls),
            "max_retries": int(limits.max_retries),
        },
    )


def build_grounded_view(result: Any, ledger: Any | None) -> GroundedCandidateView:
    """Project the run's grounded candidate universe, when a candidate plane was wired.

    With no ledger (the history-only capability) the universe is the accepted Tool's own
    recommendation list, so the counts come from the trusted tool result instead.  That is
    the honest description of what the run grounded: there was no multi-source plane to
    record provenance, and reporting zeros would wrongly suggest retrieval found nothing.
    """
    if ledger is None or not ledger.entries:
        # A capability-only run (or a plane run that grounded nothing) has no ledger record, so
        # the accepted Tool's own verified candidate list *is* the grounded universe.  Reporting
        # zeros here would say "retrieval found nothing" when it plainly returned candidates.
        tool_result = result.state.get("tool_result")
        recommendations = tuple(getattr(tool_result, "recommendations", ()) or ())
        # No plane ran, so the accepted history recommender produced these candidates: it is
        # the only source a capability-only run can have.  A plane run records its own source
        # name in state, so this branch never mislabels one.
        declared = str(result.state.get("candidate_source") or "history")
        return GroundedCandidateView(
            grounded_count=len(recommendations),
            ungrounded_count=0,
            multi_source_count=0,
            sources_present=(declared,) if recommendations else (),
            candidates=tuple(
                AgentCandidateView(
                    parent_asin=str(item.parent_asin),
                    grounded=True,
                    verification_status="grounded",
                    provenance=(),
                )
                for item in recommendations
            ),
        )
    grounded = ledger.grounded_entries
    # A run can hold candidates the plane never recorded - the accepted capability's sequential
    # source is not a plane source - and they are presentable.  They are reported here so the
    # grounded universe the response describes is the universe it may actually show; a plane-only
    # run adds nothing and reports exactly what it reported before.
    recorded = {str(entry.parent_asin): entry for entry in ledger.entries}
    presented = _run_candidates(result.state)
    views: list[AgentCandidateView] = []
    seen: set[str] = set()
    for item in presented:
        parent_asin = str(item.parent_asin)
        seen.add(parent_asin)
        entry = recorded.get(parent_asin)
        if entry is None:
            # No plane sighting exists for this candidate, so it carries no fabricated
            # provenance: an empty tuple is the honest record.
            views.append(AgentCandidateView(
                parent_asin=parent_asin,
                grounded=True,
                verification_status="grounded",
                provenance=(),
            ))
            continue
        views.append(AgentCandidateView(
            parent_asin=parent_asin,
            grounded=entry.verification_status.value == "grounded",
            verification_status=entry.verification_status.value,
            provenance=tuple(entry.provenance),
        ))
    # Everything else the ledger holds - including ungrounded identities, which are retained for
    # audit and are never presented - keeps its ledger order after the run's presentable order.
    for entry in ledger.entries:
        if str(entry.parent_asin) in seen:
            continue
        views.append(AgentCandidateView(
            parent_asin=str(entry.parent_asin),
            grounded=entry.verification_status.value == "grounded",
            verification_status=entry.verification_status.value,
            provenance=tuple(entry.provenance),
        ))
    # The run's own execution order first (it is the order the trajectory shows), then any source
    # the ledger knows and the retained rankings do not - which cannot happen today, but reporting
    # a source the run really used must never depend on that.
    sources = list(_run_sources(result.state))
    for source in ledger.sources_present():
        if source.value not in sources:
            sources.append(source.value)
    adopted_count = sum(1 for item in presented if str(item.parent_asin) not in recorded)
    return GroundedCandidateView(
        grounded_count=len(grounded) + adopted_count,
        ungrounded_count=len(ledger.ungrounded_entries),
        multi_source_count=len(ledger.multi_source_parent_asins()),
        sources_present=tuple(sources),
        candidates=tuple(views),
    )


def _run_candidates(state: Mapping[str, Any]) -> tuple[Any, ...]:
    """The candidates the run holds, from the accumulated artifact when there is one."""
    artifact = state.get("presentable_tool_result")
    if artifact is None:
        artifact = state.get("tool_result")
    return tuple(getattr(artifact, "recommendations", ()) or ())


def _run_sources(state: Mapping[str, Any]) -> tuple[str, ...]:
    """The trusted sources this run executed, in first-seen order.

    Read from the loop's retained per-source rankings; a state built without a controller (a
    fixture, a replay) reports nothing rather than guessing a source name.
    """
    rankings = state.get("candidate_rankings")
    if not isinstance(rankings, dict):
        return ()
    return tuple(str(name) for name in rankings)


def build_eligibility_view(result: Any) -> AgentEligibilityView:
    """Project the controller's stored constraint verdicts, without re-evaluating them.

    ``evaluated`` is false when the controller stored no eligibility view, which happens
    when no hard constraint was active or no evaluator was configured.  Both cases mean
    "nothing was evaluated", and neither is "everything passed".
    """
    view = result.state.get("candidate_eligibility")
    projection = result.state.get("constraint_projection")
    if view is None:
        return AgentEligibilityView(evaluated=False)
    return AgentEligibilityView(
        evaluated=True,
        requirements=tuple(view.as_dict()["requirements"]),
        verified_eligible_count=int(view.verified_eligible_count),
        ineligible_count=int(view.ineligible_count),
        unresolved_count=int(view.unresolved_count),
        assessments=tuple(view.as_dict()["assessments"]),
        projection=None if projection is None else projection.as_dict(),
    )


def presentable_candidates(state: Mapping[str, Any]) -> tuple[Any, ...]:
    """The candidates the run may present, in the backend's order - the one presentation rule.

    The set is the controller's accumulated candidate artifact: every source the run legally
    executed, fused by the accepted reciprocal-rank rule (``state["presentable_tool_result"]``,
    written by the loop's adoption step).  When a run has only ever adopted one source - the whole
    history-only path - that artifact *is* the adopted Tool result, byte for byte, so nothing about
    a single-source run changes.  ``state["tool_result"]`` remains the fallback for a state built
    without a controller (a projection, a fixture, a replay).

    Used both by :func:`build_recommendations_view` and by the memory-aware ranking adapter
    (E-02A), so the set that gets ranked is byte-for-byte the set that gets published: a hard
    constraint narrows it identically in both places, and nothing can be ranked that the response
    would then refuse to show.
    """
    artifact = state.get("presentable_tool_result")
    if artifact is None:
        artifact = state.get("tool_result")
    if artifact is None:
        return ()
    permitted = _presentable_identities(state)
    items = tuple(getattr(artifact, "recommendations", ()) or ())
    if permitted is None:
        return items
    return tuple(item for item in items if str(item.parent_asin) in permitted)


def build_recommendations_view(
    state: Mapping[str, Any],
    ledger: Any | None,
    *,
    titles: Mapping[str, str] | None = None,
    ranking: Any | None = None,
    limit: int | None = None,
) -> tuple[AgentRecommendationView, ...]:
    """Project the ranked recommendation list, attaching each item's source provenance.

    **Only candidates the run is permitted to present appear here.**  When a hard constraint is
    active the controller stores a feasibility projection naming what may be shown, and this
    function honours it: a candidate that was retrieved, grounded and then proved ineligible is
    reported in the ``grounded`` block (with its provenance and the verdict against it) but
    never in the answer.  Presenting it here while the projection excluded it would be exactly
    the failure the constraint kernel exists to prevent - a violator shown as a recommendation.

    Without ``ranking`` the order is the backend's order (the run's fused order) and is never
    re-sorted.  With a :class:`~recommendation.demo.memory_ranking.MemoryRanking` the *list*
    follows the accepted M10B order while ``rank`` keeps each candidate's baseline (fused)
    position, so the memory-aware order and the fusion order are both readable in the same
    response and neither rewrites the other.  Scores, score kinds and provenance are copied from
    the ledger in both cases.

    ``limit`` is applied **after** the ordering, so the request's ``k`` bounds what is shown while
    the order it is taken from is the run's own fused (or memory-aware) ranking of the whole
    candidate set - never an arbitrary subset of one source.

    Provenance is looked up from the ledger by identity; an item the ledger does not know (the
    capability-only path) reports no sources rather than a guessed one.
    """
    items = presentable_candidates(state)
    if not items:
        return ()
    resolution = _ledger_sources(ledger)
    ranked = None if ranking is None else ranking.by_identity()
    if ranked is not None:
        # The ranking was computed over exactly this set, so re-read it in its own order rather
        # than trusting the caller's list to agree with it.
        items = tuple(sorted(items, key=lambda item: ranked[str(item.parent_asin)].reranked_rank))
    if limit is not None:
        items = items[: max(int(limit), 0)]
    views: list[AgentRecommendationView] = []
    for item in items:
        parent_asin = str(item.parent_asin)
        score, score_kind = _score_for(parent_asin, item, ledger)
        entry = None if ranked is None else ranked.get(parent_asin)
        views.append(
            AgentRecommendationView(
                # Ranked over the presented set, so the numbering a client renders has no gap.
                rank=len(views) + 1 if entry is None else entry.rank,
                item_id=int(item.item_id),
                parent_asin=parent_asin,
                score=score,
                score_kind=score_kind,
                title=None if titles is None else titles.get(parent_asin),
                sources=resolution.get(parent_asin, ()),
                reranked_rank=None if entry is None else entry.reranked_rank,
                match_count=0 if entry is None else entry.match_count,
                violation_count=0 if entry is None else entry.violation_count,
                unknown_count=0 if entry is None else entry.unknown_count,
                evidence=() if entry is None else entry.evidence,
                rerank_reason=None if entry is None else entry.reason,
            )
        )
    return tuple(views)


def _presentable_identities(state: Mapping[str, Any]) -> frozenset[str] | None:
    """Identities the run may present, or ``None`` when nothing narrowed the set.

    ``None`` means no hard constraint was active, so the whole verified candidate set is
    presentable - which is the pre-constraint behaviour, unchanged.
    """
    projection = state.get("constraint_projection")
    if projection is None:
        return None
    return frozenset(str(identity) for identity in projection.presented_identities)


# --------------------------------------------------------------------------- #
# internals
# --------------------------------------------------------------------------- #


def _observation_type(observation: Mapping[str, Any]) -> str | None:
    """The observation's declared kind, or ``None`` when the step produced none."""
    if not observation:
        return None
    kind = observation.get("kind")
    return None if kind is None else str(kind)


def _observation_candidate_count(
    observation: Mapping[str, Any], context: Mapping[str, Any]
) -> int:
    """How many candidates this step observed.

    A candidate-producing observation reports ``returned_k``; a step that only read context
    reports the candidate count the policy context carried.  Reading this unconditionally
    from either place is exactly the bug the controller documents in ``finalize`` - a
    reasoning observation has no ``returned_k`` at all.
    """
    returned = observation.get("returned_k")
    if isinstance(returned, int):
        return returned
    count = context.get("candidate_count")
    return int(count) if isinstance(count, int) else 0


def _eligibility_state(context: Mapping[str, Any]) -> str | None:
    """A compact eligibility identifier for one step, or ``None`` when nothing was constrained.

    Deliberately built from the counts the context already exposes: it is a label for the
    trajectory, not a new verdict, and the authoritative per-candidate verdicts live in the
    response's ``eligibility`` block.
    """
    if not context.get("active_constraint_count"):
        return None
    return (
        f"verified:{int(context.get('verified_eligible_count', 0))}"
        f"/ineligible:{int(context.get('ineligible_count', 0))}"
        f"/unresolved:{int(context.get('unresolved_count', 0))}"
    )


def _score_for(parent_asin: str, item: Any, ledger: Any | None) -> tuple[float, str | None]:
    """The identity's own source score and what that score measures.

    The accepted recommendation path reports the SASRec logit on the tool result.  A
    multi-source plane instead records each source's own score, with its ``score_kind``, on
    the ledger provenance - and it deliberately zeroes the rendering score, because a BM25
    value presented as a SASRec logit would be a fabricated number.  So the ledger is
    consulted first when it has a real score, and the tool result is the fallback.
    """
    if ledger is not None:
        entry = ledger.get(parent_asin)
        if entry is not None:
            for record in entry.provenance:
                if record.source_score is not None:
                    return float(record.source_score), record.score_kind
    return float(getattr(item, "score", 0.0) or 0.0), "sasrec_logit"


def _ledger_sources(ledger: Any | None) -> dict[str, tuple[str, ...]]:
    """Map each known identity to the distinct sources that proposed it, in ledger order."""
    if ledger is None:
        return {}
    resolution: dict[str, tuple[str, ...]] = {}
    for entry in ledger.entries:
        seen: list[str] = []
        for record in entry.provenance:
            value = record.source.value
            if value not in seen:
                seen.append(value)
        if seen:
            resolution[str(entry.parent_asin)] = tuple(seen)
    return resolution

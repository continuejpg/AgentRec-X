"""``ResultVerifier`` and ``ObservationAdapter`` - the raw/observation boundary.

Two jobs live here and they are different jobs:

``ResultVerifier``
    Decides whether a capability's raw :class:`DomainResult` is trustworthy enough to be
    shown, using only **structural and identity** facts - never a score, never a
    preference, never a model opinion.  It verifies that the recommendation that came back
    is the recommendation that was asked for, over the candidate set the accepted pipeline
    actually produced, with the candidate identity the accepted Tool actually assigned.

``ObservationAdapter``
    Turns a *verified* result into the minimised, policy-visible
    :class:`RecommendationObservation`.  The policy sees status, counts, an opaque set
    reference and a verification verdict - never the candidate list, never a score, never
    a product fact.

Why this is not ceremony: the point of AgentRec-X's trust model is that
``DomainResult != Observation``.  If a tool payload could flow straight into the policy,
the policy would inherit a channel it is not supposed to have (candidate ids, scores,
metadata) and the "policy cannot name products" property would be a comment rather than a
fact.  Here the boundary is a function that cannot be bypassed, because the adapter
accepts only a verified result.

What verification checks (Stage 1)
----------------------------------
* **Action/result agreement** - the result belongs to the action that produced it.
* **Tool grounding** - the accepted Tool ran on the run's *actual* trusted history: the
  result's ``history_length`` must equal the supplied history length.  This is the
  strongest available proof that no forged or truncated history was scored, and it is a
  count only, so no history content is copied anywhere.
* **Count integrity** - ``returned_k`` equals the number of recommendations, ``returned_k``
  never exceeds ``requested_k``, counts are non-negative and internally consistent.
* **Rank integrity** - ranks are contiguous ``1..n`` and unique (the accepted ranking
  contract).
* **Candidate identity** - every candidate carries a non-empty external ``parent_asin``
  and an internal ``item_id >= 1`` (item id 0 is PAD and is never a candidate).
* **Stage alignment** - when enrichment / evidence / reranking are present, they describe
  the *same* candidate set by ``(parent_asin, item_id)`` identity and the same count.  A
  stage that invents, drops or substitutes a candidate is a refusal, never something the
  verifier silently repairs.

A refusal is a first-class outcome, not an exception, so the loop can record it and fail
closed deterministically.
"""

from __future__ import annotations

from typing import Any

from recommendation.agent.state import TrustedHistory

from .schemas import (
    ActionKind,
    DomainResult,
    RecommendationDomainResult,
    RecommendationObservation,
    VerificationResult,
)

__all__ = ["ObservationAdapter", "ResultVerifier"]


class ResultVerifier:
    """Verify a domain result against the run's trusted inputs.

    Deterministic and side-effect free.  Never mutates a result, never fabricates a
    missing stage, never downgrades a refusal to a warning.
    """

    def verify(
        self,
        result: DomainResult,
        *,
        trusted_history: TrustedHistory,
        expected_action_id: str,
    ) -> VerificationResult:
        """Return the verification verdict for ``result``."""
        if not isinstance(result, RecommendationDomainResult):
            return VerificationResult(
                verified=False,
                code="unsupported_result",
                detail=f"no verifier for {type(result).__name__}",
                checks=("result_type",),
            )

        checks: list[str] = ["result_type"]

        # -- 1. the result belongs to the action that produced it ------------- #
        if result.action_id != expected_action_id:
            return VerificationResult(
                verified=False,
                code="action_mismatch",
                detail="the result does not belong to the action being verified",
                checks=(*checks, "action_identity"),
            )
        checks.append("action_identity")

        tool_result = result.tool_result
        if tool_result is None:
            return VerificationResult(
                verified=False,
                code="missing_tool_result",
                detail="the capability returned no Tool result to verify",
                checks=checks,
            )
        checks.append("tool_result_present")

        # -- 2. the Tool ran on the run's actual trusted history -------------- #
        if tool_result.history_length != len(trusted_history):
            return VerificationResult(
                verified=False,
                code="history_length_mismatch",
                detail=(
                    "the Tool result was produced from a different history length than "
                    "the run's trusted history"
                ),
                checks=(*checks, "tool_grounding"),
            )
        checks.append("tool_grounding")

        recommendations = tool_result.recommendations

        # -- 3. count integrity ----------------------------------------------- #
        if result.returned_k != len(recommendations):
            return VerificationResult(
                verified=False,
                code="returned_k_mismatch",
                detail="the declared returned_k disagrees with the candidate list",
                checks=(*checks, "count_integrity"),
            )
        if len(recommendations) > tool_result.requested_k:
            return VerificationResult(
                verified=False,
                code="returned_exceeds_requested",
                detail="more candidates were returned than were requested",
                checks=(*checks, "count_integrity"),
            )
        if tool_result.requested_k != result.requested_k:
            return VerificationResult(
                verified=False,
                code="requested_k_mismatch",
                detail="the result and its Tool payload disagree on the requested count",
                checks=(*checks, "count_integrity"),
            )
        checks.append("count_integrity")

        # -- 4. rank integrity ------------------------------------------------- #
        expected_ranks = list(range(1, len(recommendations) + 1))
        actual_ranks = [item.rank for item in recommendations]
        if actual_ranks != expected_ranks:
            return VerificationResult(
                verified=False,
                code="rank_not_contiguous",
                detail="candidate ranks are not the contiguous sequence 1..n",
                checks=(*checks, "rank_integrity"),
            )
        checks.append("rank_integrity")

        # -- 5. candidate identity --------------------------------------------- #
        for item in recommendations:
            if not isinstance(item.parent_asin, str) or not item.parent_asin.strip():
                return VerificationResult(
                    verified=False,
                    code="empty_candidate_identity",
                    detail="a candidate carries no usable external identity",
                    checks=(*checks, "candidate_identity"),
                )
            if item.item_id < 1:
                return VerificationResult(
                    verified=False,
                    code="pad_item_in_candidates",
                    detail="a candidate carries the reserved padding item id",
                    checks=(*checks, "candidate_identity"),
                )
        checks.append("candidate_identity")

        tool_identities = [(item.parent_asin, item.item_id) for item in recommendations]

        # -- 6. stage alignment ------------------------------------------------ #
        if result.enrichment is not None:
            enriched = result.enrichment.items
            if len(enriched) != len(tool_identities):
                return VerificationResult(
                    verified=False,
                    code="enrichment_count_mismatch",
                    detail="enrichment changed the number of candidates",
                    checks=(*checks, "enrichment_alignment"),
                )
            # ``EnrichedRecommendation`` exposes identity through properties and nests the
            # Tool recommendation, so read both explicitly rather than by attribute path.
            enriched_identities = [
                (entry.parent_asin, entry.recommendation.item_id) for entry in enriched
            ]
            if enriched_identities != tool_identities:
                return VerificationResult(
                    verified=False,
                    code="enrichment_identity_mismatch",
                    detail=(
                        "enrichment changed candidate identity or order; the accepted "
                        "enricher must be order- and identity-preserving"
                    ),
                    checks=(*checks, "enrichment_alignment"),
                )
            checks.append("enrichment_alignment")

        if result.preference_evidence is not None:
            evidence_candidates = result.preference_evidence.candidates
            if len(evidence_candidates) != len(tool_identities):
                return VerificationResult(
                    verified=False,
                    code="evidence_count_mismatch",
                    detail="preference evidence changed the number of candidates",
                    checks=(*checks, "evidence_alignment"),
                )
            evidence_identities = [
                (entry.parent_asin, entry.item_id) for entry in evidence_candidates
            ]
            if evidence_identities != tool_identities:
                return VerificationResult(
                    verified=False,
                    code="evidence_identity_mismatch",
                    detail=(
                        "preference evidence changed candidate identity or order; the "
                        "accepted matcher must be order- and identity-preserving"
                    ),
                    checks=(*checks, "evidence_alignment"),
                )
            checks.append("evidence_alignment")

        if result.reranking is not None:
            reranked = result.reranking.candidates
            if len(reranked) != len(tool_identities):
                return VerificationResult(
                    verified=False,
                    code="reranking_count_mismatch",
                    detail="reranking changed the number of candidates",
                    checks=(*checks, "reranking_alignment"),
                )
            reranked_identities = [(entry.parent_asin, entry.item_id) for entry in reranked]
            if sorted(reranked_identities) != sorted(tool_identities):
                return VerificationResult(
                    verified=False,
                    code="reranking_identity_mismatch",
                    detail=(
                        "reranking changed the candidate set; the accepted policy may "
                        "reorder candidates but never add, drop or substitute one"
                    ),
                    checks=(*checks, "reranking_alignment"),
                )
            if sorted(entry.original_rank for entry in reranked) != expected_ranks:
                return VerificationResult(
                    verified=False,
                    code="reranking_original_rank_mismatch",
                    detail="reranked candidates do not carry the upstream original ranks",
                    checks=(*checks, "reranking_alignment"),
                )
            if sorted(entry.reranked_rank for entry in reranked) != expected_ranks:
                return VerificationResult(
                    verified=False,
                    code="reranking_rank_not_contiguous",
                    detail="reranked ranks are not the contiguous sequence 1..n",
                    checks=(*checks, "reranking_alignment"),
                )
            checks.append("reranking_alignment")

        return VerificationResult(
            verified=True,
            code="verified",
            detail=None,
            checks=tuple(checks),
        )


class ObservationAdapter:
    """Convert a **verified** result into the minimised policy-visible observation.

    The adapter accepts a result together with its verification verdict and refuses to
    build an observation from an unverified one: an unverified payload either never
    reaches the policy or reaches it as a refusal, never as content.
    """

    def adapt(
        self,
        result: RecommendationDomainResult,
        verification: VerificationResult,
        *,
        action: ActionKind,
        step_index: int,
    ) -> RecommendationObservation:
        """Build the observation for one executed recommendation action.

        Only counts, booleans, the opaque ``candidate_set_ref`` and a short verification
        note cross into the policy's view.  No candidate identity, no score, no metadata.
        """
        tool_result = result.tool_result
        returned_k = result.returned_k
        has_candidates = bool(tool_result is not None and returned_k > 0)

        if verification.verified:
            status = "ok" if has_candidates else "empty"
            note = (
                f"verified: {len(verification.checks)} structural/identity checks passed"
            )
        else:
            status = "failed"
            note = f"refused: {verification.code}"

        return RecommendationObservation(
            action_id=result.action_id,
            step_index=step_index,
            action=action,
            verification_status="verified" if verification.verified else "refused",
            source=result.source,
            status=status,
            requested_k=result.requested_k,
            returned_k=returned_k,
            has_candidates=has_candidates,
            candidate_set_ref=result.candidate_set_ref,
            verification_note=note,
        )

    def failed_observation(
        self,
        *,
        action_id: str,
        step_index: int,
        action: ActionKind,
        code: str,
        requested_k: int = 0,
    ) -> RecommendationObservation:
        """Build the observation for an action whose execution raised.

        The failure is reported as a status and a code.  The exception text is *not*
        copied in: it may name internals, and a policy does not need it to decide what to
        do next.
        """
        return RecommendationObservation(
            action_id=action_id,
            step_index=step_index,
            action=action,
            verification_status="refused",
            status="failed",
            requested_k=requested_k,
            returned_k=0,
            has_candidates=False,
            candidate_set_ref="",
            verification_note=f"execution failed: {code}",
        )

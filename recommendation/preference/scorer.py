"""P14-B0 Preference-Aware Scorer V0 (deterministic, explainable, no search).

Contract
--------
``score(preferences, current_query, candidates, hard_requirements, namespace)``

Inputs:
* ``preferences``: the *active* preferences recovered through the legal history
  reader (already canonicalised by the caller; this module never reads memory or
  evaluator labels);
* ``current_query``: the shopping request (recorded, not used for new retrieval);
* ``candidates``: a FIXED candidate set produced by the accepted search layer,
  each carrying ``product_ref``, ``query_rank``, ``query_score`` and the product
  attribute evidence available from the catalogue;
* ``hard_requirements``: stated hard constraints, already extracted from the
  agent-visible query (not from labels);
* ``namespace``: the expected catalogue namespace (cross-namespace candidates are
  rejected, never silently accepted).

Output: one record per candidate with ranked product references, query relevance,
positive/negative preference contributions, matched/conflicting/unknown preferences,
hard-constraint status, evidence refs and a stable rank.

Scoring (weights are fixed before this run; not tuned on the Pilot Dev set):
* query relevance is rank-based within the fixed pool: (n - rank) / (n - 1);
* each matched positive preference adds ``positive_weight`` (capped at 3 terms);
* each negative preference whose attribute the product explicitly has subtracts
  ``negative_weight``;
* a missing attribute is UNKNOWN: it never earns a positive contribution and is
  recorded explicitly;
* hard constraints are tiered: verified (tier 0) > unknown (tier 1) > violated
  (tier 2).  A violated candidate ranks below every non-violated candidate no matter
  how relevant it is, and is marked ``presentable=False``; unknown-tier candidates
  are ranked below verified ones but above violators.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

__all__ = ["PreferenceAwareScorer", "ScorerConfig", "SCORER_VERSION"]

SCORER_VERSION = "preference_scorer_v0.1"


@dataclass(frozen=True)
class ScorerConfig:
    """Fixed weights for the pilot baseline (chosen before evaluation; not tuned)."""

    query_weight: float = 0.50
    positive_weight: float = 0.20
    negative_weight: float = 0.35
    max_positive_terms: int = 3
    hard_unknown_penalty: float = 0.10

    def __post_init__(self) -> None:
        for value in (self.query_weight, self.positive_weight, self.negative_weight):
            if value < 0:
                raise ValueError("weights must be non-negative")
        if self.max_positive_terms < 1:
            raise ValueError("max_positive_terms must be >= 1")


def _norm(text: Any) -> str:
    return " ".join(str(text or "").lower().split())


def _ref(item: Mapping[str, Any]) -> dict[str, str]:
    ref = item.get("product_ref") or {}
    namespace = str(ref.get("namespace") or "")
    item_id = str(ref.get("item_id") or "")
    if not namespace or not item_id:
        raise ValueError("candidate.product_ref needs namespace and item_id")
    return {"namespace": namespace, "item_id": item_id}


def _hard_status(requirement: Mapping[str, Any], candidate: Mapping[str, Any]) -> str:
    """Return verified / unknown / violated for one candidate against one requirement."""
    kind = str(requirement.get("kind") or "")
    attributes = candidate.get("attributes") or {}
    if kind == "unverifiable":
        return "unknown"
    value = _norm(requirement.get("value"))
    if kind == "form_factor":
        observed = _norm(candidate.get("form_factor"))
        if not observed:
            return "unknown"
        return "verified" if observed == value else "violated"
    return "verified" if value in {_norm(name) for name in attributes} else "unknown"


class PreferenceAwareScorer:
    """Deterministic reranker over a fixed candidate set."""

    def __init__(self, config: ScorerConfig | None = None) -> None:
        self._config = config or ScorerConfig()

    @property
    def config(self) -> ScorerConfig:
        return self._config

    @property
    def version(self) -> str:
        return SCORER_VERSION

    def score(
        self,
        *,
        preferences: Sequence[Mapping[str, Any]],
        current_query: str,
        candidates: Sequence[Mapping[str, Any]],
        hard_requirements: Sequence[Mapping[str, Any]] = (),
        namespace: str,
    ) -> list[dict[str, Any]]:
        expected_namespace = str(namespace or "").strip()
        if not expected_namespace:
            raise ValueError("namespace is required")
        if not candidates:
            return []
        refs = [_ref(candidate) for candidate in candidates]
        for ref in refs:
            if ref["namespace"] != expected_namespace:
                raise ValueError(
                    f"candidate namespace {ref['namespace']!r} != expected {expected_namespace!r}"
                )

        # stable dedupe by (namespace, item_id), preserving the first (best) rank
        seen: set[tuple[str, str]] = set()
        unique: list[Mapping[str, Any]] = []
        for candidate in candidates:
            ref = _ref(candidate)
            key = (ref["namespace"], ref["item_id"])
            if key in seen:
                continue
            seen.add(key)
            unique.append(candidate)
        n = len(unique)
        if n == 0:
            return []

        active_prefs = []
        for preference in preferences:
            value = _norm(preference.get("value"))
            polarity = str(preference.get("polarity") or "positive").lower()
            if not value:
                continue
            active_prefs.append(
                {
                    "memory_id": preference.get("memory_id"),
                    "kind": preference.get("kind"),
                    "value": value,
                    "polarity": "negative" if polarity in {"avoid", "negative", "dislike"} else "positive",
                    "source_turn_id": preference.get("source_turn_id"),
                }
            )

        scored: list[dict[str, Any]] = []
        for candidate in unique:
            ref = _ref(candidate)
            rank = int(candidate.get("query_rank") or n)
            rank = max(1, min(rank, n))
            q_norm = (n - rank) / (n - 1) if n > 1 else 1.0
            attributes = candidate.get("attributes") or {}
            present = {_norm(name) for name in attributes}

            positive_terms = [p for p in active_prefs if p["polarity"] == "positive"]
            negative_terms = [p for p in active_prefs if p["polarity"] == "negative"]
            matched = [p for p in positive_terms if p["value"] in present]
            conflicting = [p for p in negative_terms if p["value"] in present]
            unknown = [
                p for p in positive_terms + negative_terms
                if p["value"] not in present
            ]
            positive_score = self._config.positive_weight * min(
                len(matched), self._config.max_positive_terms
            )
            conflict_score = self._config.negative_weight * len(conflicting)

            hard_states = {
                str(req.get("key") or req.get("value") or index): _hard_status(req, candidate)
                for index, req in enumerate(hard_requirements)
            }
            if hard_states:
                if any(state == "violated" for state in hard_states.values()):
                    hard_tier = 2
                elif any(state == "unknown" for state in hard_states.values()):
                    hard_tier = 1
                else:
                    hard_tier = 0
            else:
                hard_tier = 0
            hard_penalty = self._config.hard_unknown_penalty if hard_tier == 1 else 0.0
            raw = (
                self._config.query_weight * q_norm
                + positive_score
                - conflict_score
                - hard_penalty
            )
            if hard_tier == 2:
                final_score = -1.0 - rank / max(n, 1)  # below every non-violator
            else:
                final_score = round(raw, 6)
            evidence_refs: list[dict[str, Any]] = []
            for preference in sorted(matched + conflicting, key=lambda item: str(item["value"])):
                name = preference["value"]
                fact = attributes.get(name)
                evidence_refs.append(
                    {
                        "attribute": name,
                        "memory_id": preference.get("memory_id"),
                        "source_turn_id": preference.get("source_turn_id"),
                        "catalogue_evidence": fact.get("evidence") if isinstance(fact, Mapping) else None,
                    }
                )
            scored.append(
                {
                    "product_ref": ref,
                    "final_score": final_score,
                    "query_relevance_score": round(q_norm, 6),
                    "query_rank": rank,
                    "query_raw_score": candidate.get("query_score"),
                    "preference_match_score": round(positive_score, 6),
                    "preference_conflict_score": round(conflict_score, 6),
                    "hard_requirement_penalty": hard_penalty,
                    "hard_status": hard_states,
                    "hard_tier": hard_tier,
                    "matched_preferences": [dict(p) for p in matched],
                    "conflicting_preferences": [dict(p) for p in conflicting],
                    "unknown_preferences": [dict(p) for p in unknown],
                    "presentable": bool(hard_tier == 0 and not conflicting),
                    "evidence_refs": evidence_refs,
                    "scoring_version": SCORER_VERSION,
                }
            )

        scored.sort(
            key=lambda row: (
                row["hard_tier"],
                -row["final_score"],
                row["query_rank"],
                row["product_ref"]["namespace"],
                row["product_ref"]["item_id"],
            )
        )
        for index, row in enumerate(scored, start=1):
            row["rank"] = index
        return scored

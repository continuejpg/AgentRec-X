"""Frozen paired evaluation for the M6.1 ranking arms (offline lab only).

This module implements the preregistered protocol in
``docs/M6_1_RANKING_PROTOCOL.md``: three deterministic arms (A0/A1/A2) are ranked over
**one** shared eligibility projection of one frozen candidate snapshot, and the run is
reported as contract metrics plus -- only when documented preference-conditioned labels
exist -- paired quality metrics with a fixed-seed bootstrap confidence interval.

What this module is not
-----------------------
* It is not a serving path.  It is never imported by the demo/serving runtime.
* It is **not** a learned ranker and produces no ranking-model training signal.
* It does **not** turn ranking change into quality evidence.  ``change_rate_vs_a0`` and
  ``top_k_overlap_vs_a0`` are behavioural/contract diagnostics; the only quality fields
  are the labeled metrics and their paired deltas.

Label boundary
--------------
``FrozenRankingTask`` bundles the labels, but the feature builder **cannot** receive
them: the evaluator calls
``builder.build(snapshot=task.snapshot, ranker_projection=task.ranker_projection)`` and
reads ``task.labels`` itself only to compute quality metrics.  That is a structural
no-leakage boundary, not a convention.

Quality definition
------------------
When labels exist they are binary relevance (``relevance > 0``).  The accepted metric
helpers (:func:`recommendation.evaluation.metrics.hit_at_k` /
:func:`~recommendation.evaluation.metrics.ndcg_at_k`) are single-positive, so each labeled
task is reduced to **one** positive target: the labeled candidate with the highest
relevance (ties broken by the candidate's trusted original rank, then item id).  MRR,
Recall@K and NDCG@K are then exactly the accepted single-positive definitions, and a
target excluded by eligibility counts as "not in the answer" (rank ``inf`` -> 0 metrics).
Paired deltas are per-task differences of those metric vectors, with the paired bootstrap
from :func:`recommendation.evaluation.experiment.paired_bootstrap_delta` at the
preregistered seed/draws.
"""

from __future__ import annotations

import hashlib
import statistics
from typing import Any, Literal, Sequence

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from recommendation.evaluation.experiment import (
    ExperimentCaseResult,
    paired_bootstrap_delta,
)
from recommendation.evaluation.metrics import hit_at_k, ndcg_at_k
from recommendation.state.schemas import canonical_json

from .errors import RankingProtocolError
from .policies import (
    PREREGISTERED_ARMS,
    rank_all_arms,
    shared_eligibility_projection,
)
from .schemas import (
    ArmContractMetrics,
    ArmEvaluation,
    ArmRanking,
    CandidateFeatureSet,
    CandidateRankingFeatures,
    EvaluationReport,
    FrozenRankingTask,
    PairedDelta,
    QualityMetrics,
    RankingArm,
)

__all__ = [
    "DEFAULT_PRIMARY_QUESTION",
    "DEFAULT_PROMOTION_CRITERIA",
    "DEFAULT_PROTOCOL",
    "EvaluationProtocol",
    "FrozenPairedEvaluator",
    "default_feature_builder",
]

#: The main research question, frozen from the preregistration document (section 1).
DEFAULT_PRIMARY_QUESTION = (
    "Can a UnifiedDecisionState for_ranker projection be joined to a frozen trusted "
    "candidate set, turned into typed grounded features, and consumed by a deterministic "
    "belief-aware ranking policy that preserves candidate identity, candidate-set "
    "membership and hard-constraint eligibility, preserves the accepted M10B explicit "
    "preference priority, uses inferred hypotheses only as an ordinal tie-break among "
    "candidates equal under explicit priority, and produces identical results under "
    "replay -- without changing Serving V1 and without claiming recommendation-quality "
    "improvement from ranking change alone?"
)

#: Promotion criteria frozen from the preregistration document (section 8).
DEFAULT_PROMOTION_CRITERIA: tuple[str, ...] = (
    "every M6.1 contract invariant passes with zero violations",
    "explicit-priority preservation and candidate-set equality tests pass",
    "the feature schema is stable, with documented catalogue coverage gaps",
    "preference-conditioned relevance labels exist under a frozen protocol, and a paired "
    "evaluation shows a non-degenerate signal on the preregistered metrics",
    "no Serving V1 change, no new production flag and no frozen-protocol modification",
)

#: Contract-level diagnostics that are never quality evidence.
_CONTRACT_ONLY_NOTE = (
    "change rates and top-K overlaps are contract/behavioural diagnostics only; they are "
    "never effectiveness evidence and are never converted into a quality claim"
)


def _protocol_digest(payload: dict[str, Any]) -> str:
    """SHA-256 of the canonical JSON encoding of a protocol body."""
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


class EvaluationProtocol(BaseModel):
    """The frozen preregistered M6.1 evaluation protocol.

    The defaults are exactly ``docs/M6_1_RANKING_PROTOCOL.md``; any change produces a new
    digest, and every report records the digest it was produced under.  Changing the
    protocol (rather than passing an amended object) is what makes a result readable
    against the version it used.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    protocol_name: str = Field(
        default="m6.1.paired-ranking.v1", min_length=1, max_length=120
    )
    primary_question: str = Field(
        default=DEFAULT_PRIMARY_QUESTION, min_length=1, max_length=800
    )
    dataset_source: str = Field(
        default="synthetic_trusted_fixture", min_length=1, max_length=120
    )
    arms: tuple[RankingArm, ...] = PREREGISTERED_ARMS
    k_values: tuple[int, ...] = (1, 3, 5)
    bootstrap_seed: int = 20260201
    bootstrap_draws: int = Field(default=2000, ge=1)
    missing_evidence_policy: Literal["unknown_neutral"] = "unknown_neutral"
    quality_status_default: Literal["NOT_EXECUTED_NO_LABELS"] = (
        "NOT_EXECUTED_NO_LABELS"
    )
    promotion_criteria: tuple[str, ...] = DEFAULT_PROMOTION_CRITERIA
    label_protocol: str | None = Field(default=None, max_length=200)

    @field_validator("k_values")
    @classmethod
    def _validate_k_values(cls, value: tuple[int, ...]) -> tuple[int, ...]:
        if not value:
            raise ValueError("at least one K value is required")
        cleaned = []
        for k in value:
            if isinstance(k, bool) or not isinstance(k, int) or k < 1:
                raise ValueError(f"K must be a positive integer, got {k!r}")
            cleaned.append(k)
        if list(cleaned) != sorted(set(cleaned)):
            raise ValueError("K values must be unique and ascending")
        return tuple(cleaned)

    @field_validator("promotion_criteria")
    @classmethod
    def _validate_criteria(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if not value:
            raise ValueError("promotion criteria must not be empty")
        for entry in value:
            if not str(entry).strip():
                raise ValueError("a promotion criterion must not be blank")
        return value

    @model_validator(mode="after")
    def _validate_arms(self) -> "EvaluationProtocol":
        if self.arms != PREREGISTERED_ARMS:
            raise ValueError(
                "the preregistered arms are exactly A0/A1/A2 in that order; an amended "
                "protocol must be introduced as a new protocol version"
            )
        return self

    @property
    def digest(self) -> str:
        """SHA-256 digest of the canonical protocol body (recorded on every report)."""
        return _protocol_digest(self.model_dump(mode="json"))


#: The frozen default protocol.  All M6.1 runs use this object.
DEFAULT_PROTOCOL = EvaluationProtocol()


def default_feature_builder() -> Any:
    """Build the accepted M6.1 feature builder (imported lazily: offline lab only)."""
    from .features import BeliefAwareFeatureBuilder

    return BeliefAwareFeatureBuilder()


def _require_types(tasks: Sequence[FrozenRankingTask]) -> tuple[FrozenRankingTask, ...]:
    """Validate and return the task sequence (non-empty, unique, frozen snapshots)."""
    if isinstance(tasks, (str, bytes)) or not isinstance(tasks, Sequence):
        raise RankingProtocolError("tasks must be a sequence of FrozenRankingTask")
    collected = tuple(tasks)
    if not collected:
        raise RankingProtocolError("the paired evaluation needs at least one task")
    seen: set[str] = set()
    for task in collected:
        if not isinstance(task, FrozenRankingTask):
            raise RankingProtocolError(
                f"tasks must be FrozenRankingTask instances, got {type(task).__name__}"
            )
        if task.task_id in seen:
            raise RankingProtocolError(f"duplicate task id in the evaluation: {task.task_id!r}")
        seen.add(task.task_id)
        if task.snapshot.candidate_count < 1:
            raise RankingProtocolError(
                f"task {task.task_id!r} has an empty candidate snapshot; there is nothing "
                "to rank"
            )
    return collected


def _validate_feature_set(
    feature_set: Any, task: FrozenRankingTask
) -> CandidateFeatureSet:
    """Validate that the builder returned features for exactly this frozen task."""
    if not isinstance(feature_set, CandidateFeatureSet):
        raise RankingProtocolError(
            "the feature builder must return a CandidateFeatureSet"
        )
    snapshot = task.snapshot
    identity_checks = (
        (feature_set.task_id, task.task_id, "task_id"),
        (feature_set.user_key, task.user_key, "user_key"),
        (feature_set.turn_key, task.turn_key, "turn_key"),
        (feature_set.snapshot_id, snapshot.snapshot_id, "snapshot_id"),
        (
            feature_set.snapshot_candidate_count,
            snapshot.candidate_count,
            "snapshot_candidate_count",
        ),
    )
    for produced, expected, name in identity_checks:
        if produced != expected:
            raise RankingProtocolError(
                f"feature set {name} does not match the frozen task"
            )
    snapshot_by_id = {candidate.candidate_id: candidate for candidate in snapshot.candidates}
    if set(feature_set.candidate_ids) != set(snapshot_by_id):
        raise RankingProtocolError(
            "feature set candidate identities do not match the frozen snapshot"
        )
    for feature in feature_set.features:
        record = snapshot_by_id[feature.candidate_id]
        if (
            feature.item_id != record.item_id
            or feature.original_rank != record.original_rank
        ):
            raise RankingProtocolError(
                f"feature identity for {feature.candidate_id!r} does not match the snapshot"
            )
        if feature.eligibility is not record.eligibility:
            raise RankingProtocolError(
                f"feature eligibility for {feature.candidate_id!r} does not match the "
                "snapshot assessment"
            )
    return feature_set


def _explicit_key_sequence(
    ranking: ArmRanking, features_by_id: dict[str, CandidateRankingFeatures]
) -> list[tuple[int, int]]:
    """The A1 explicit-preference key of every placed candidate, in arm order."""
    return [features_by_id[item.candidate_id].explicit_key for item in ranking.ranked]


def _explicit_priority_preserved(
    ranking: ArmRanking, features_by_id: dict[str, CandidateRankingFeatures]
) -> bool:
    """True when the arm's order never places worse explicit evidence ahead of better.

    This is the A1/A2 explicit-priority invariant: the sequence of
    ``(explicit_violations, -explicit_matches)`` keys must be non-decreasing.
    """
    keys = _explicit_key_sequence(ranking, features_by_id)
    return all(left <= right for left, right in zip(keys, keys[1:]))


def _a2_changes_only_explicit_ties(
    a1: ArmRanking,
    a2: ArmRanking,
    features_by_id: dict[str, CandidateRankingFeatures],
) -> bool:
    """True when A2 reorders, relative to A1, only candidates tied on explicit priority."""
    a1_positions = {candidate_id: index for index, candidate_id in enumerate(a1.candidate_ids)}
    a2_positions = {candidate_id: index for index, candidate_id in enumerate(a2.candidate_ids)}
    identifiers = list(a1.candidate_ids)
    for left_index in range(len(identifiers)):
        for right_index in range(left_index + 1, len(identifiers)):
            left = identifiers[left_index]
            right = identifiers[right_index]
            a1_says_left_first = a1_positions[left] < a1_positions[right]
            a2_says_left_first = a2_positions[left] < a2_positions[right]
            if a1_says_left_first != a2_says_left_first:
                if (
                    features_by_id[left].explicit_key
                    != features_by_id[right].explicit_key
                ):
                    return False
    return True


def _change_rate(
    reference: ArmRanking, candidate: ArmRanking
) -> tuple[int, int]:
    """Return ``(moved, total)`` candidate position changes between two arm orders."""
    if not reference.candidate_ids or not candidate.candidate_ids:
        return 0, 0
    reference_positions = {
        candidate_id: index for index, candidate_id in enumerate(reference.candidate_ids)
    }
    total = len(candidate.candidate_ids)
    moved = sum(
        1
        for index, candidate_id in enumerate(candidate.candidate_ids)
        if reference_positions.get(candidate_id) != index
    )
    return moved, total


def _top_k_overlap(
    reference: ArmRanking, candidate: ArmRanking, k: int
) -> tuple[int, int]:
    """Return ``(shared, denominator)`` for the top-K prefix overlap."""
    pool = len(candidate.candidate_ids)
    if pool == 0:
        return 0, 0
    prefix = min(k, pool)
    shared = len(
        set(candidate.candidate_ids[:prefix]) & set(reference.candidate_ids[:prefix])
    )
    return shared, prefix


def _inference_counts(
    features: Sequence[CandidateRankingFeatures],
) -> tuple[int, int]:
    """Return ``(covered, unknown)`` inferred-evidence coverage for the pool.

    * covered: the candidate has at least one decided inferred signal (a positive/negative
      match or a non-zero ordinal peer comparison);
    * unknown: the candidate only has undecided (UNKNOWN) inferred evidence.
    """
    covered = 0
    unknown = 0
    for feature in features:
        decided = (
            feature.inferred_matches + feature.inferred_violations > 0
            or feature.inferred_ordinal_support > 0
            or feature.inferred_ordinal_penalty > 0
        )
        if decided:
            covered += 1
        elif feature.inferred_unknown > 0:
            unknown += 1
    return covered, unknown


def _positions(ranking: ArmRanking) -> dict[str, int]:
    """Candidate id -> 1-based arm position."""
    return {candidate_id: index for index, candidate_id in enumerate(ranking.candidate_ids, 1)}


def _positive_target(
    task: FrozenRankingTask,
) -> tuple[str, int] | None:
    """The single positive target of a labeled task (or ``None`` when no label is positive).

    Highest relevance wins; ties are broken by the frozen original rank then the item id,
    so the reduction is deterministic and cannot depend on label tuple order.
    """
    positive = [label for label in task.labels if label.relevance > 0.0]
    if not positive:
        return None
    records = {candidate.candidate_id: candidate for candidate in task.snapshot.candidates}
    chosen = min(
        positive,
        key=lambda label: (
            -float(label.relevance),
            records[label.candidate_id].original_rank,
            records[label.candidate_id].item_id,
        ),
    )
    return chosen.candidate_id, records[chosen.candidate_id].item_id


def _task_quality_vector(
    ranking: ArmRanking, target_id: str, k_values: Sequence[int]
) -> dict[str, float]:
    """Per-task quality metrics under the single-positive label reduction."""
    positions = _positions(ranking)
    vector: dict[str, float] = {}
    rank = positions.get(target_id)
    if rank is None:
        # The labeled target is not in this arm's presentable pool (e.g. eligibility
        # excluded it): it was not delivered, which is a miss, not an error.
        vector["mrr"] = 0.0
        for k in k_values:
            vector[f"recall@{k}"] = 0.0
            vector[f"ndcg@{k}"] = 0.0
        return vector
    vector["mrr"] = 1.0 / float(rank)
    for k in k_values:
        vector[f"recall@{k}"] = hit_at_k(rank, k)
        vector[f"ndcg@{k}"] = ndcg_at_k(rank, k)
    return vector


def _labels_are_evaluable(tasks: Sequence[FrozenRankingTask]) -> bool:
    """True when every task carries labels and one common, non-blank label protocol."""
    protocols: set[str] = set()
    for task in tasks:
        if not task.labels:
            return False
        protocol = str(task.label_protocol or "").strip()
        if not protocol:
            return False
        protocols.add(protocol)
    return len(protocols) == 1


class FrozenPairedEvaluator:
    """Run the preregistered paired evaluation over frozen ranking tasks.

    Parameters
    ----------
    feature_builder:
        Any object with ``build(*, snapshot, ranker_projection) -> CandidateFeatureSet``.
        ``None`` (default) builds the accepted
        :class:`~recommendation.ranking.features.BeliefAwareFeatureBuilder` lazily.  The
        builder never receives the task labels.
    protocol:
        The frozen :class:`EvaluationProtocol`; the default is
        :data:`DEFAULT_PROTOCOL`.  Its digest is recorded on the report.
    """

    def __init__(
        self,
        *,
        feature_builder: Any = None,
        protocol: EvaluationProtocol = DEFAULT_PROTOCOL,
    ) -> None:
        if not isinstance(protocol, EvaluationProtocol):
            raise RankingProtocolError("protocol must be an EvaluationProtocol")
        self._feature_builder = feature_builder
        self._protocol = protocol

    @property
    def protocol(self) -> EvaluationProtocol:
        """The frozen protocol this evaluator runs."""
        return self._protocol

    @property
    def feature_builder(self) -> Any:
        """The injected builder, or ``None`` when the accepted default is used lazily."""
        return self._feature_builder

    # -- evaluation -------------------------------------------------------- #

    def evaluate(self, tasks: Sequence[FrozenRankingTask]) -> EvaluationReport:
        """Evaluate all preregistered arms over ``tasks`` and return the frozen report."""
        validated = _require_types(tasks)
        builder = self._resolve_builder()
        protocol = self._protocol
        self._validate_protocol_labels(validated)

        per_task: list[dict[str, Any]] = []
        coverage_totals = {
            "found": 0,
            "missing": 0,
            "readable": {},
            "supported": set(),
            "unsupported": set(),
        }
        for task in validated:
            feature_set = _validate_feature_set(
                builder.build(
                    snapshot=task.snapshot,
                    ranker_projection=task.ranker_projection,
                ),
                task,
            )
            features_by_id = {
                feature.candidate_id: feature for feature in feature_set.features
            }
            rankings = rank_all_arms(feature_set, task.snapshot)
            # Determinism: rank the same features a second time and compare the orders.
            repeat = rank_all_arms(feature_set, task.snapshot)
            projection = shared_eligibility_projection(feature_set)
            eligible = set(projection.eligible_ids)
            snapshot_ids = set(task.snapshot.candidate_ids)
            deterministic = all(
                left.candidate_ids == right.candidate_ids
                for left, right in zip(rankings, repeat)
            )
            if not deterministic:  # pragma: no cover - deterministic by construction
                raise RankingProtocolError(
                    f"arm ordering is not deterministic for task {task.task_id!r}"
                )
            for ranking in rankings:
                # A0 is the trusted baseline order and is deliberately not required to
                # follow explicit preference evidence; A1 and A2 are.
                if ranking.arm is RankingArm.A0_ORIGINAL:
                    continue
                if not _explicit_priority_preserved(ranking, features_by_id):
                    raise RankingProtocolError(
                        f"arm {ranking.arm.value} placed worse explicit evidence ahead of "
                        f"better in task {task.task_id!r}"
                    )
            a1, a2 = rankings[1], rankings[2]
            if not _a2_changes_only_explicit_ties(a1, a2, features_by_id):
                raise RankingProtocolError(
                    "A2 reordered candidates that are not tied on explicit preference "
                    f"priority in task {task.task_id!r}"
                )
            # Cross-arm candidate-set equality: an arm may reorder, never add or drop one.
            if len({frozenset(ranking.candidate_ids) for ranking in rankings}) != 1:
                raise RankingProtocolError(
                    "the arms disagree on the candidate set for task "
                    f"{task.task_id!r}; every arm must rank the same shared pool"
                )
            coverage = feature_set.coverage
            coverage_totals["found"] += coverage.metadata_found_count
            coverage_totals["missing"] += coverage.metadata_missing_count
            for field, count in coverage.readable_fields.items():
                coverage_totals["readable"][field] = (
                    coverage_totals["readable"].get(field, 0) + int(count)
                )
            coverage_totals["supported"].update(coverage.supported_inferred_features)
            coverage_totals["unsupported"].update(coverage.unsupported_inferred_features)
            per_task.append(
                {
                    "task": task,
                    "feature_set": feature_set,
                    "features": tuple(feature_set.features),
                    "rankings": {ranking.arm: ranking for ranking in rankings},
                    "eligible_ids": projection.eligible_ids,
                    "projection": projection,
                    "eligible": eligible,
                    "snapshot_ids": snapshot_ids,
                    "deterministic": deterministic,
                }
            )

        arms = tuple(
            self._evaluate_arm(rec["arm"], per_task) for rec in self._arm_evaluations()
        )
        quality_status, paired_deltas, notes = self._evaluate_quality(per_task)
        report = EvaluationReport(
            protocol_name=protocol.protocol_name,
            protocol_digest=protocol.digest,
            dataset_source=protocol.dataset_source,
            task_count=len(per_task),
            candidate_count_total=sum(
                rec["task"].snapshot.candidate_count for rec in per_task
            ),
            arms=arms,
            paired_deltas=paired_deltas,
            quality_status=quality_status,
            notes=tuple(notes)
            + self._coverage_notes(coverage_totals, per_task)
            + (_CONTRACT_ONLY_NOTE,),
        )
        return report.with_digest()

    # -- internals --------------------------------------------------------- #

    def _resolve_builder(self) -> Any:
        """Return the feature builder, refusing one without the frozen build seam."""
        builder = self._feature_builder
        if builder is None:
            builder = default_feature_builder()
        build = getattr(builder, "build", None)
        if not callable(build):
            raise RankingProtocolError(
                "the feature builder must provide a build(snapshot=..., "
                "ranker_projection=...) method"
            )
        return builder

    def _validate_protocol_labels(
        self, tasks: Sequence[FrozenRankingTask]
    ) -> None:
        """If the protocol pins a label protocol, every labeled task must match it."""
        pinned = self._protocol.label_protocol
        if pinned is None:
            return
        for task in tasks:
            if task.labels and str(task.label_protocol or "").strip() != pinned:
                raise RankingProtocolError(
                    f"task {task.task_id!r} labels do not follow the protocol's label "
                    f"protocol {pinned!r}"
                )

    def _arm_evaluations(self) -> list[dict[str, Any]]:
        """The preregistered arm records, in order."""
        return [{"arm": arm} for arm in PREREGISTERED_ARMS]

    def _evaluate_arm(
        self, arm: RankingArm, per_task: Sequence[dict[str, Any]]
    ) -> ArmEvaluation:
        """Aggregate one arm's contract metrics across the validated tasks."""
        totals = {
            "candidate_count": 0,
            "identity": True,
            "equal_snapshot": True,
            "equal_expected_pool": True,
            "deterministic": True,
            "eligibility_violations": 0,
            "moved_vs_a0": 0,
            "total_vs_a0": 0,
            "overlap_a0": {k: 0 for k in self._protocol.k_values},
            "denominator_a0": {k: 0 for k in self._protocol.k_values},
            "moved_vs_a1": 0,
            "total_vs_a1": 0,
            "overlap_a1": {k: 0 for k in self._protocol.k_values},
            "denominator_a1": {k: 0 for k in self._protocol.k_values},
            "explicit_violations": 0,
            "explicit_matches": 0,
            "inference_covered": 0,
            "inference_unknown": 0,
        }
        for record in per_task:
            ranking = record["rankings"][arm]
            a0 = record["rankings"][RankingArm.A0_ORIGINAL]
            a1 = record["rankings"][RankingArm.A1_EXPLICIT_RERANK]
            snapshot_ids = record["snapshot_ids"]
            eligible = record["eligible"]
            identities = ranking.candidate_ids
            totals["candidate_count"] += len(identities)
            totals["deterministic"] = totals["deterministic"] and bool(
                record["deterministic"]
            )
            if (
                len(set(identities)) != len(identities)
                or not set(identities) <= snapshot_ids
                or set(identities) != eligible
            ):
                totals["identity"] = False
            totals["equal_snapshot"] = totals["equal_snapshot"] and (
                set(identities) == snapshot_ids
            )
            # The expected pool is the snapshot when nothing was assessed, otherwise the
            # shared projection's verified-eligible set (the only rankable candidates).
            totals["equal_expected_pool"] = totals["equal_expected_pool"] and (
                set(identities) == set(record["eligible_ids"])
            )
            totals["eligibility_violations"] += len(set(identities) ^ eligible)
            moved, total = _change_rate(a0, ranking)
            totals["moved_vs_a0"] += moved
            totals["total_vs_a0"] += total
            for k in self._protocol.k_values:
                shared, denominator = _top_k_overlap(a0, ranking, k)
                totals["overlap_a0"][k] += shared
                totals["denominator_a0"][k] += denominator
                shared, denominator = _top_k_overlap(a1, ranking, k)
                totals["overlap_a1"][k] += shared
                totals["denominator_a1"][k] += denominator
            moved, total = _change_rate(a1, ranking)
            totals["moved_vs_a1"] += moved
            totals["total_vs_a1"] += total
            totals["explicit_violations"] += sum(
                item.explicit_violations for item in ranking.ranked
            )
            totals["explicit_matches"] += sum(item.explicit_matches for item in ranking.ranked)
            covered, unknown = _inference_counts(record["features"])
            totals["inference_covered"] += covered
            totals["inference_unknown"] += unknown

        def rate(moved: int, total: int) -> float:
            return float(moved) / float(total) if total else 0.0

        def overlaps(
            counts: dict[int, int], denominators: dict[int, int]
        ) -> dict[str, float]:
            return {
                str(k): (
                    float(counts[k]) / float(denominators[k])
                    if denominators[k]
                    else 0.0
                )
                for k in self._protocol.k_values
            }

        contract = ArmContractMetrics(
            candidate_count=totals["candidate_count"],
            candidate_identity_preserved=totals["identity"],
            candidate_set_equal_to_snapshot=totals["equal_snapshot"],
            candidate_set_equal_to_expected_pool=totals["equal_expected_pool"],
            ordering_deterministic=totals["deterministic"],
            eligibility_invariant_violations=totals["eligibility_violations"],
            change_rate_vs_a0=rate(totals["moved_vs_a0"], totals["total_vs_a0"]),
            top_k_overlap_vs_a0=overlaps(
                totals["overlap_a0"], totals["denominator_a0"]
            ),
            change_rate_vs_a1=rate(totals["moved_vs_a1"], totals["total_vs_a1"]),
            top_k_overlap_vs_a1=overlaps(
                totals["overlap_a1"], totals["denominator_a1"]
            ),
            explicit_violations_total=totals["explicit_violations"],
            explicit_matches_total=totals["explicit_matches"],
            inference_covered_candidates=totals["inference_covered"],
            inference_unknown_candidates=totals["inference_unknown"],
        )
        quality, _ = self._arm_quality(arm, per_task)
        return ArmEvaluation(arm=arm, contract=contract, quality=quality)

    def _quality_context(
        self, per_task: Sequence[dict[str, Any]]
    ) -> tuple[bool, list[dict[str, Any]], str | None]:
        """Decide whether quality metrics are evaluable and collect labeled tasks."""
        tasks = [record["task"] for record in per_task]
        evaluable = _labels_are_evaluable(tasks)
        labeled: list[dict[str, Any]] = []
        label_protocol: str | None = None
        if evaluable:
            label_protocol = str(tasks[0].label_protocol or "").strip()
            for record in per_task:
                target = _positive_target(record["task"])
                labeled.append({**record, "target": target})
        return evaluable, labeled, label_protocol

    def _arm_quality(
        self, arm: RankingArm, per_task: Sequence[dict[str, Any]]
    ) -> tuple[QualityMetrics, dict[str, dict[str, float]]]:
        """Quality metrics for one arm, or the explicit no-labels status."""
        evaluable, labeled, label_protocol = self._quality_context(per_task)
        if not evaluable:
            return (
                QualityMetrics(
                    status="NOT_EXECUTED_NO_LABELS",
                    note=(
                        "no documented preference-conditioned labels were supplied for "
                        "every task; historical purchase relevance is not "
                        "preference-conditioned, so no quality metrics are reported"
                    ),
                ),
                {},
            )
        k_values = self._protocol.k_values
        vectors: dict[str, dict[str, float]] = {}
        for record in labeled:
            target = record["target"]
            if target is None:
                zero = {"mrr": 0.0}
                for k in k_values:
                    zero[f"recall@{k}"] = 0.0
                    zero[f"ndcg@{k}"] = 0.0
                vectors[record["task"].task_id] = zero
                continue
            vectors[record["task"].task_id] = _task_quality_vector(
                record["rankings"][arm], target[0], k_values
            )
        metric_names = ["mrr"] + [
            f"{name}@{k}" for k in k_values for name in ("recall", "ndcg")
        ]
        means = {
            name: statistics.fmean(vector[name] for vector in vectors.values())
            for name in metric_names
        }
        quality = QualityMetrics(
            status="MEASURED",
            label_protocol=label_protocol,
            task_count_with_labels=len(vectors),
            mrr=means["mrr"],
            recall_at_k={
                str(k): means[f"recall@{k}"] for k in k_values
            },
            ndcg_at_k={str(k): means[f"ndcg@{k}"] for k in k_values},
            note=(
                "binary relevance (relevance > 0) reduced to one positive target per "
                "task: highest relevance, ties by frozen original rank then item id; "
                "MRR/Recall@K/NDCG@K use the accepted single-positive metric helpers"
            ),
        )
        return quality, vectors

    def _evaluate_quality(
        self, per_task: Sequence[dict[str, Any]]
    ) -> tuple[str, tuple[PairedDelta, ...], list[str]]:
        """Compute quality status, paired deltas and the label-related report notes."""
        evaluable, labeled, label_protocol = self._quality_context(per_task)
        if not evaluable:
            return (
                "NOT_EXECUTED_NO_LABELS",
                (),
                [
                    "quality_status=NOT_EXECUTED_NO_LABELS: no documented "
                    "preference-conditioned relevance labels exist for every task; "
                    "purchase history is not preference-conditioned evidence, and no "
                    "quality metric or quality delta is reported.",
                ],
            )
        vectors_by_arm: dict[RankingArm, dict[str, dict[str, float]]] = {}
        for arm in PREREGISTERED_ARMS:
            _, vectors = self._arm_quality(arm, per_task)
            vectors_by_arm[arm] = vectors
        metric_names = ["mrr"] + [
            f"{name}@{k}"
            for k in self._protocol.k_values
            for name in ("recall", "ndcg")
        ]
        target_item_ids = {
            record["task"].task_id: (
                record["target"][1]
                if record["target"] is not None
                else record["task"].snapshot.candidates[0].item_id
            )
            for record in labeled
        }

        def case_results(
            arm: RankingArm,
        ) -> list[ExperimentCaseResult]:
            return [
                ExperimentCaseResult(
                    case_id=record["task"].task_id,
                    target_item_id=target_item_ids[record["task"].task_id],
                    metrics=dict(vectors_by_arm[arm][record["task"].task_id]),
                )
                for record in labeled
            ]

        deltas: list[PairedDelta] = []
        comparisons = (
            (RankingArm.A2_BELIEF_TIEBREAK, RankingArm.A0_ORIGINAL),
            (RankingArm.A2_BELIEF_TIEBREAK, RankingArm.A1_EXPLICIT_RERANK),
        )
        for candidate_arm, baseline_arm in comparisons:
            baseline = case_results(baseline_arm)
            candidate = case_results(candidate_arm)
            for name in metric_names:
                comparison = f"{candidate_arm.value} - {baseline_arm.value}"
                if len(baseline) < 2:
                    differences = [
                        candidate_case.metrics[name] - baseline_case.metrics[name]
                        for candidate_case, baseline_case in zip(candidate, baseline)
                    ]
                    deltas.append(
                        PairedDelta(
                            metric=name,
                            comparison=comparison,
                            mean_delta=(
                                statistics.fmean(differences) if differences else 0.0
                            ),
                            ci_low=None,
                            ci_high=None,
                            sample_size=len(baseline),
                            status="NOT_COMPUTABLE",
                            note=(
                                "fewer than two labeled tasks; a paired bootstrap CI is "
                                "not reported"
                            ),
                        )
                    )
                    continue
                delta = paired_bootstrap_delta(
                    baseline,
                    candidate,
                    metric=name,
                    draws=self._protocol.bootstrap_draws,
                    seed=self._protocol.bootstrap_seed,
                )
                deltas.append(
                    PairedDelta(
                        metric=name,
                        comparison=comparison,
                        mean_delta=delta.mean,
                        ci_low=delta.lower,
                        ci_high=delta.upper,
                        sample_size=delta.n,
                        status="MEASURED",
                        note=(
                            f"paired bootstrap {delta.confidence:.0%} CI, "
                            f"draws={delta.draws}, seed={delta.seed}"
                        ),
                    )
                )
        return (
            "MEASURED",
            tuple(deltas),
            [
                f"quality_status=MEASURED with documented label protocol "
                f"{label_protocol!r} on {len(labeled)} labeled task(s); paired deltas use "
                f"fixed seed {self._protocol.bootstrap_seed} and "
                f"{self._protocol.bootstrap_draws} draws."
            ],
        )

    def _coverage_notes(
        self,
        coverage_totals: dict[str, Any],
        per_task: Sequence[dict[str, Any]],
    ) -> tuple[str, ...]:
        """Feature-coverage and eligibility-exclusion report notes (never quality)."""
        excluded_total = sum(
            record["projection"].excluded_count for record in per_task
        )
        eligible_total = sum(record["projection"].eligible_count for record in per_task)
        assessed_tasks = sum(
            1 for record in per_task if record["projection"].assessed
        )
        readable = dict(sorted(coverage_totals["readable"].items()))
        unsupported = sorted(coverage_totals["unsupported"])
        supported = sorted(coverage_totals["supported"])
        notes = [
            (
                f"eligibility projection applied once before all arms: "
                f"{eligible_total} candidate(s) ranked, {excluded_total} excluded by "
                f"hard-constraint eligibility across {len(per_task)} task(s) "
                f"({assessed_tasks} task(s) carried an assessment). All arms rank the "
                "identical pool (cross-arm candidate-set equality is asserted); the "
                "expected pool is the snapshot when nothing was assessed, otherwise the "
                "VERIFIED_ELIGIBLE set. candidate_set_equal_to_snapshot is False only "
                "because the trusted eligibility decision legitimately removes "
                "INELIGIBLE/UNRESOLVED candidates, which are never rankable."
            ),
            (
                "feature coverage: "
                f"metadata_found={coverage_totals['found']}, "
                f"metadata_missing={coverage_totals['missing']}, "
                f"readable_fields={readable}, "
                f"supported_inferred_features={supported}, "
                f"unsupported_inferred_features={unsupported}"
            ),
        ]
        return tuple(notes)

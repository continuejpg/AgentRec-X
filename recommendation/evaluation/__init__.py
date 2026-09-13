"""Unified, model-independent evaluation protocol (Milestone 2A).

This package is the single source of truth for how AgentRec-X recommendation
models are compared.  It contains **no recommender**: ItemCF, SASRec and
SID-OneRec are implemented in later milestones and plug into the scorer boundary
defined here.

Layout
------
``split.py``
    Deterministic temporal leave-two-out splitting, evaluation-cohort selection,
    catalog definition, and loading of preprocessing artifacts.
``metrics.py``
    The ranking kernel (canonical tie rule), target-rank computation, HR@K /
    Recall@K / NDCG@K, and deterministic aggregation.
``evaluator.py``
    The model-independent driver: a scorer supplies one score per catalog item and
    the evaluator owns PAD exclusion, seen-item masking, target retention, ranking
    and metric computation.

Quick start
-----------
::

    from recommendation.evaluation import (
        FullRankingEvaluator, build_cohort_from_artifacts, cohort_summary,
    )

    cases, split = build_cohort_from_artifacts(
        "data/processed/Sports_and_Outdoors_sample_sequences.json",
        "data/processed/Sports_and_Outdoors_sample_mappings.json",
    )
    print(cohort_summary(cases, split))

    evaluator = FullRankingEvaluator(num_items=split.catalog_size, k_values=(5, 10, 20))

    def score_fn(history, target_item_id):      # later: ItemCF / SASRec
        ...                                     # -> one score per catalog item

    outcome = evaluator.evaluate(cases, score_fn, mode="test")
    print(outcome.report.format())

Protocol summary
----------------
For a chronological sequence ``[i1, ..., i(n-1), in]`` with ``n >= 3``::

    train_history = [i1, ..., i(n-2)]
    validation    : history = train_history,            target = i(n-1)
    test          : history = train_history + [i(n-1)], target = in

Candidates are the full catalog ``1..num_items`` minus seen items
(``set(history) - {target}``), so the target always stays eligible - including when
it also appears earlier in the history.  PAD (0) is never a candidate.  Ranking is
by higher score first, ties broken by lower item id.
"""

from __future__ import annotations

from .evaluator import (
    CohortOutcome,
    FullRankingEvaluator,
    Scorer,
    build_cohort,
    build_cohort_from_artifacts,
    cohort_summary,
)
from .metrics import (
    DEFAULT_K_VALUES,
    METRIC_NAMES,
    CaseResult,
    EvaluationError,
    EvaluationReport,
    aggregate,
    compare_hr_recall,
    evaluate_case,
    hit_at_k,
    ndcg_at_k,
    rank_of_target,
    rank_of_target_via_sort,
    sorted_ranking,
    valid_candidates,
    validate_k_values,
    validate_scores_finite,
)
from .split import (
    MIN_SEQUENCE_LENGTH,
    PROTOCOL_NAME,
    PROTOCOL_VERSION,
    EvaluationCase,
    SplitError,
    SplitReport,
    build_catalog,
    load_cohort_from_artifacts,
    load_sequences_artifact,
    sequences_from_artifact,
    split_cohort,
    split_sequence,
    train_history_statistics,
    user_int_ids_from_artifact,
    validate_item_ids,
)

__all__ = [
    # protocol
    "MIN_SEQUENCE_LENGTH",
    "PROTOCOL_NAME",
    "PROTOCOL_VERSION",
    "DEFAULT_K_VALUES",
    "METRIC_NAMES",
    # split
    "EvaluationCase",
    "SplitError",
    "SplitReport",
    "build_catalog",
    "load_cohort_from_artifacts",
    "load_sequences_artifact",
    "sequences_from_artifact",
    "split_cohort",
    "split_sequence",
    "train_history_statistics",
    "user_int_ids_from_artifact",
    "validate_item_ids",
    # metrics
    "CaseResult",
    "EvaluationError",
    "EvaluationReport",
    "aggregate",
    "compare_hr_recall",
    "evaluate_case",
    "hit_at_k",
    "ndcg_at_k",
    "rank_of_target",
    "rank_of_target_via_sort",
    "sorted_ranking",
    "valid_candidates",
    "validate_k_values",
    "validate_scores_finite",
    # evaluator
    "CohortOutcome",
    "FullRankingEvaluator",
    "Scorer",
    "build_cohort",
    "build_cohort_from_artifacts",
    "cohort_summary",
]

__version__ = "0.1.0"

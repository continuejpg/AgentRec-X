"""Model-independent evaluation driver.

The evaluator is the only place that turns "a recommender produced a score for
every catalog item" into metrics.  Models never mask, never rank and never compute
metrics themselves, so ItemCF, SASRec and SID-OneRec cannot drift into
incompatible evaluation semantics (AGENTS.md section 6).

Boundary contract
-----------------
A **scorer** is a callable that receives one already-selected evaluation case::

    score_fn(history: tuple[int, ...], target_item_id: int) -> Sequence[float]

and returns one score per catalog item, indexed by item id::

    scores[item_id]  for item_id in 1..num_items

``scores[0]`` is the PAD slot.  It is accepted (and ignored) only so that an
indexing off-by-one mistake surfaces as an explicit length error instead of
silently shifting every rank by one.

The evaluator owns:

* PAD exclusion
* seen-item masking (via ``history``; the evaluator, not the model, decides what
  is masked)
* target retention
* ranking semantics and tie handling
* metric computation and aggregation

Predictions are therefore *never* a ranked id list.  Ranking a model's own
candidate shortlist would reintroduce per-model candidate protocols, which is
exactly what this milestone exists to prevent.

Because history/target selection happens in the evaluator before the scorer is
called, the same scoring function is evaluated in validation mode and in test mode
without the model needing to know which is which.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from recommendation import config

from .metrics import (
    DEFAULT_K_VALUES,
    METRIC_NAMES,
    CaseResult,
    EvaluationError,
    EvaluationReport,
    aggregate,
    compare_hr_recall,
    evaluate_case,
    rank_of_target,
    validate_history_items,
    validate_k_values,
    validate_scores_finite,
)
from .split import (
    PROTOCOL_NAME,
    EvaluationCase,
    SplitReport,
    load_cohort_from_artifacts,
    split_cohort,
    train_history_statistics,
)

#: A scorer maps one selected case to one score per catalog item (PAD included).
Scorer = Callable[[tuple[int, ...], int], Sequence[float]]


@dataclass
class CohortOutcome:
    """Result of evaluating one cohort in one mode."""

    mode: str
    report: EvaluationReport
    results: list[CaseResult]

    @property
    def hr_recall_agree(self) -> dict[int, bool]:
        """Per K, whether HR@K and Recall@K coincide (they must in this protocol)."""
        return compare_hr_recall(self.report)


class FullRankingEvaluator:
    """Full-ranking evaluator over the catalog ``1..num_items``.

    Parameters
    ----------
    num_items:
        Catalogue size from the preprocessing artifact.
    k_values:
        Cutoffs; canonical default ``(5, 10, 20)``.
    """

    def __init__(
        self,
        num_items: int,
        k_values: Sequence[int] = DEFAULT_K_VALUES,
    ) -> None:
        if isinstance(num_items, bool) or not isinstance(num_items, int) or num_items < 1:
            raise EvaluationError(f"num_items must be a positive int, got {num_items!r}")
        self.num_items = num_items
        self.k_values = validate_k_values(k_values)

    # -- score contract ---------------------------------------------------- #

    def expected_score_length(self) -> int:
        """Number of scores a conforming scorer must provide (PAD slot included)."""
        return self.num_items + 1

    def validate_scores(self, scores: Sequence[float], *, context: str = "scores") -> None:
        """Raise :class:`EvaluationError` if the score vector is malformed.

        A vector must be a sized sequence of ``num_items + 1`` **finite** numbers.
        Finiteness is part of the score contract because a ``NaN`` score silently
        corrupts ranking (every comparison against NaN is false, so a NaN target is
        ranked first); see :func:`~recommendation.evaluation.metrics.validate_scores_finite`.
        """
        if scores is None:
            raise EvaluationError(f"{context} is None; expected one score per catalog item")
        try:
            length = len(scores)
        except TypeError as exc:
            raise EvaluationError(
                f"{context} must be a sized sequence, got {type(scores).__name__}"
            ) from exc
        if length != self.expected_score_length():
            raise EvaluationError(
                f"{context} has length {length}, expected num_items + 1 = "
                f"{self.expected_score_length()} (index 0 is the PAD slot)"
            )
        validate_scores_finite(scores, context=context)

    # -- single case ------------------------------------------------------- #

    def evaluate_case(
        self,
        scores: Sequence[float],
        target_item_id: int,
        *,
        history: Sequence[int] = (),
        user_id: str = "",
        context: str = "scores",
    ) -> CaseResult:
        """Evaluate one case; the evaluator owns all masking semantics."""
        self.validate_scores(scores, context=context)
        return evaluate_case(
            scores,
            target_item_id,
            history=history,
            k_values=self.k_values,
            num_items=self.num_items,
            user_id=user_id,
        )

    def rank(
        self,
        scores: Sequence[float],
        target_item_id: int,
        *,
        history: Sequence[int] = (),
    ) -> int:
        """Return the 1-based target rank under the canonical tie rule."""
        self.validate_scores(scores)
        return rank_of_target(
            scores,
            target_item_id,
            history=history,
            num_items=self.num_items,
        )

    def candidate_count(
        self,
        history: Sequence[int] = (),
        target_item_id: int | None = None,
    ) -> int:
        """Number of rankable candidates once seen items are masked out."""
        seen = set(validate_history_items(history, self.num_items))
        if target_item_id is not None:
            seen.discard(target_item_id)
        return self.num_items - len(seen)

    # -- batched path (Milestone 5) ---------------------------------------- #

    def evaluate_cases_batched(
        self,
        cases: Sequence[EvaluationCase],
        score_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None = None,
        *,
        mode: str = "test",
        batch_size: int = 512,
        device: torch.device | str = "cpu",
        cohort: str | None = None,
        score_matrix_fn: Callable[[Sequence[tuple[tuple[int, ...], int]]], Any] | None = None,
    ):
        """Evaluate ``cases`` through the vectorized full-ranking path.

        Exactly the same ranking semantics as :meth:`evaluate`, but target ranks are
        computed for a whole batch at once.  Either ``score_fn`` (called per batch with
        the encoded histories and targets, returning a ``[batch, num_items + 1]``
        score tensor) or ``score_matrix_fn`` (called with the ``(history, target)``
        pairs of a batch) must be supplied.

        Returns a :class:`~recommendation.evaluation.batched.BatchedEvaluationResult`.
        """
        from .batched import evaluate_batched

        if score_fn is None and score_matrix_fn is None:
            raise EvaluationError("evaluate_cases_batched needs score_fn or score_matrix_fn")
        if batch_size < 1:
            raise EvaluationError(f"batch_size must be >= 1, got {batch_size}")
        if not cases:
            raise EvaluationError(
                "cannot evaluate an empty cohort; check the split policy and the "
                "minimum sequence length (no user was evaluation-eligible)"
            )

        pairs = self.case_inputs(cases, mode=mode)
        chunks = [pairs[i : i + batch_size] for i in range(0, len(pairs), batch_size)]

        def batches():
            for chunk in chunks:
                histories = [history for history, _ in chunk]
                targets = [target for _, target in chunk]
                if score_matrix_fn is not None:
                    scores = score_matrix_fn(list(chunk))
                else:
                    assert score_fn is not None
                    scores = score_fn(
                        _stack_histories(histories), torch.tensor(targets, dtype=torch.long)
                    )
                yield histories, targets, scores

        return evaluate_batched(
            num_items=self.num_items,
            score_batches=batches(),
            k_values=self.k_values,
            cohort=cohort or mode,
            protocol=PROTOCOL_NAME,
            device=device,
        )

    # -- cohorts ----------------------------------------------------------- #

    def case_inputs(
        self,
        cases: Sequence[EvaluationCase],
        mode: str = "test",
    ) -> list[tuple[tuple[int, ...], int]]:
        """Return the ``(history, target)`` pairs a scorer will be called with."""
        if mode not in ("test", "validation"):
            raise EvaluationError(f"mode must be 'test' or 'validation', got {mode!r}")
        if mode == "test":
            return [(case.test_history, case.test_target) for case in cases]
        return [(case.validation_history, case.validation_target) for case in cases]

    def evaluate(
        self,
        cases: Sequence[EvaluationCase],
        score_fn: Scorer,
        *,
        mode: str = "test",
        cohort: str | None = None,
    ) -> CohortOutcome:
        """Evaluate ``score_fn`` over ``cases`` in ``mode`` and aggregate.

        Cases are scored one at a time, so a model never has to materialise a score
        matrix for the whole cohort.  Aggregation happens once at the end, which
        keeps the result independent of scoring order.
        """
        inputs = self.case_inputs(cases, mode=mode)
        if not inputs:
            raise EvaluationError(
                "cannot evaluate an empty cohort; check the split policy and the "
                "minimum sequence length (no user was evaluation-eligible)"
            )

        results: list[CaseResult] = []
        for case, (history, target) in zip(cases, inputs):
            scores = score_fn(history, target)
            results.append(
                self.evaluate_case(
                    scores,
                    target,
                    history=history,
                    user_id=case.user_id,
                    context=f"scores for user {case.user_id!r}",
                )
            )

        report = aggregate(
            results,
            k_values=self.k_values,
            catalog_size=self.num_items,
            cohort=cohort or mode,
            protocol=PROTOCOL_NAME,
        )
        return CohortOutcome(mode=mode, report=report, results=results)


# --------------------------------------------------------------------------- #
# Cohort helpers
# --------------------------------------------------------------------------- #


def build_cohort(
    sequences: dict[str, Sequence[int]],
    num_items: int,
    user_int_ids: dict[str, int] | None = None,
) -> tuple[list[EvaluationCase], SplitReport]:
    """Split raw ``user_id -> item ids`` sequences into evaluation cases."""
    return split_cohort(sequences, num_items, user_int_ids=user_int_ids)


def build_cohort_from_artifacts(
    sequences_path: str,
    mappings_path: str | None = None,
) -> tuple[list[EvaluationCase], SplitReport]:
    """Split a preprocessing run into evaluation cases."""
    return load_cohort_from_artifacts(sequences_path, mappings_path)


def cohort_summary(cases: Sequence[EvaluationCase], report: SplitReport) -> dict[str, Any]:
    """Return a JSON-serialisable summary of a cohort (counts and invariants only)."""
    targets = [case.validation_target for case in cases] + [case.test_target for case in cases]
    summary = report.as_dict()
    summary["train_history"] = train_history_statistics(cases)
    summary["targets"] = {
        "count": len(targets),
        "min": min(targets) if targets else None,
        "max": max(targets) if targets else None,
        "pad_present": config.PAD_ID in set(targets),
        "all_within_catalog": all(
            config.FIRST_REAL_ID <= t <= report.catalog_size for t in targets
        ),
    }
    summary["history"] = {
        "pad_present": any(
            config.PAD_ID in case.train_history or config.PAD_ID in case.test_history
            for case in cases
        ),
        "test_history_extends_train_by_one": all(
            len(case.test_history) == len(case.train_history) + 1 for case in cases
        ),
        "test_history_prefix_matches_train": all(
            case.test_history[: len(case.train_history)] == case.train_history for case in cases
        ),
        "test_history_last_is_validation_target": all(
            case.test_history[-1] == case.validation_target for case in cases
        ),
        "validation_history_is_train_history": all(
            case.validation_history == case.train_history for case in cases
        ),
    }
    return summary


__all__ = [
    "DEFAULT_K_VALUES",
    "METRIC_NAMES",
    "CohortOutcome",
    "EvaluationCase",
    "EvaluationError",
    "EvaluationReport",
    "FullRankingEvaluator",
    "Scorer",
    "SplitReport",
    "build_cohort",
    "build_cohort_from_artifacts",
    "cohort_summary",
    "compare_hr_recall",
]

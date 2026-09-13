"""Tests for the ranking kernel, metrics and evaluator (Milestone 2A).

Everything here is hand-computable: catalogs are tiny, score vectors are literal
lists, and every expected rank/metric is written out in the test rather than
derived from the implementation under test.  The explicit-sort oracle
(:func:`rank_of_target_via_sort`) is used as an independent reference for the
optimised target-rank implementation.
"""

from __future__ import annotations

import math
import random
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from recommendation import config  # noqa: E402
from recommendation.evaluation.evaluator import (  # noqa: E402
    FullRankingEvaluator,
    build_cohort,
    cohort_summary,
)
from recommendation.evaluation.metrics import (  # noqa: E402
    DEFAULT_K_VALUES,
    EvaluationError,
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
)
from recommendation.evaluation.split import EvaluationCase  # noqa: E402

PAD = config.PAD_ID


def scores_for(entries: dict[int, float], num_items: int) -> list[float]:
    """Build a score vector indexed by item id; unlisted items score 0.0.

    The PAD slot is a finite sentinel: scores must be finite everywhere, including
    index 0 (see :func:`test_non_finite_scores_are_rejected`).
    """
    vector = [-1e9] * (num_items + 1)  # index 0 = PAD, never read, must stay finite
    for item_id, score in entries.items():
        vector[item_id] = score
    return vector


# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #


def test_canonical_default_k_values() -> None:
    """The frozen default cutoffs are (5, 10, 20)."""
    assert DEFAULT_K_VALUES == (5, 10, 20)


def test_validate_k_values_sorts_dedupes_and_rejects_bad_input() -> None:
    """K values are normalised deterministically and validated."""
    assert validate_k_values((20, 5, 10, 5)) == (5, 10, 20)
    for bad in ((0,), (-1,), (), (1.5,), (True,), ("5",)):
        try:
            validate_k_values(bad)  # type: ignore[arg-type]
        except EvaluationError:
            continue
        raise AssertionError(f"expected EvaluationError for k_values={bad!r}")


# --------------------------------------------------------------------------- #
# Target rank: hand-computed cases
# --------------------------------------------------------------------------- #


def test_target_rank_one_when_target_scores_highest() -> None:
    """Rank 1 when nothing else outranks the target."""
    scores = scores_for({1: 0.1, 2: 0.2, 3: 9.0}, num_items=3)
    assert rank_of_target(scores, 3, num_items=3) == 1


def test_target_rank_equals_number_of_better_candidates_plus_one() -> None:
    """Rank counts strictly-better candidates and adds one."""
    scores = scores_for({1: 5.0, 2: 4.0, 3: 9.0, 4: 1.0, 5: 6.0}, num_items=5)
    # better than target(3)=9.0 -> none
    assert rank_of_target(scores, 3, num_items=5) == 1
    # better than target(2)=4.0 -> 1(5.0), 3(9.0), 5(6.0) = 3 -> rank 4
    assert rank_of_target(scores, 2, num_items=5) == 4


def test_target_rank_last_when_target_scores_lowest() -> None:
    """The worst item ranks last."""
    scores = scores_for({1: 5.0, 2: 4.0, 3: 9.0, 4: 1.0, 5: 6.0}, num_items=5)
    assert rank_of_target(scores, 4, num_items=5) == 5


def test_pad_slot_is_never_ranked_even_when_it_scores_highest() -> None:
    """PAD has the top score in this vector yet must not affect the rank."""
    scores = [1e9, 0.5, 0.4, 0.3]  # PAD scores highest
    assert rank_of_target(scores, 3, num_items=3) == 3
    assert rank_of_target(scores, 1, num_items=3) == 1
    assert PAD not in sorted_ranking(scores, num_items=3)
    assert PAD not in valid_candidates(3)


# --------------------------------------------------------------------------- #
# Deterministic tie handling
# --------------------------------------------------------------------------- #


def test_equal_score_lower_item_id_ranks_first_target_loses_tie() -> None:
    """On a tie the lower id wins, so the higher-id target is pushed down."""
    scores = scores_for({2: 0.5, 5: 0.5}, num_items=5)
    # item2 and item5 tie; item2 (lower id) ranks first, target 5 ranks second
    assert rank_of_target(scores, 5, num_items=5) == 2
    assert rank_of_target(scores, 2, num_items=5) == 1


def test_equal_score_lower_item_id_ranks_first_target_wins_tie() -> None:
    """A lower-id target outranks equal-scoring higher-id candidates."""
    scores = scores_for({2: 0.5, 4: 0.5, 5: 0.5}, num_items=5)
    assert rank_of_target(scores, 2, num_items=5) == 1
    assert rank_of_target(scores, 4, num_items=5) == 2


def test_many_way_tie_is_still_deterministic() -> None:
    """A 4-way tie resolves strictly by ascending item id."""
    scores = scores_for({1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0}, num_items=4)
    assert sorted_ranking(scores, num_items=4) == [1, 2, 3, 4]
    assert [rank_of_target(scores, t, num_items=4) for t in (1, 2, 3, 4)] == [1, 2, 3, 4]


def test_tie_handling_is_independent_of_dict_insertion_order() -> None:
    """Insertion order of the score mapping must not change any rank."""
    entries = {4: 0.7, 1: 0.7, 3: 0.7, 2: 0.7}
    forward = scores_for(entries, num_items=4)
    reversed_vector = scores_for(dict(reversed(list(entries.items()))), num_items=4)
    assert forward == reversed_vector
    assert [rank_of_target(forward, t, num_items=4) for t in (1, 2, 3, 4)] == [1, 2, 3, 4]


# --------------------------------------------------------------------------- #
# Seen-item masking and target retention
# --------------------------------------------------------------------------- #


def test_seen_items_are_excluded_from_candidates() -> None:
    """History items are masked out of the candidate pool."""
    scores = scores_for({1: 9.0, 2: 9.0, 3: 1.0}, num_items=3)
    # without masking, target 3 would be rank 3
    assert rank_of_target(scores, 3, num_items=3) == 3
    # masking items 1 and 2 leaves only the target -> rank 1
    assert rank_of_target(scores, 3, history=[1, 2], num_items=3) == 1
    assert valid_candidates(3, history=[1, 2], target_item_id=3) == [3]


def test_target_retained_even_when_it_appears_earlier_in_history() -> None:
    """The target must stay eligible even if the user consumed it before.

    This is the explicit regression test for ``excluded_seen = set(history) -
    {target}``: if the target were masked like any other history item it would be
    unrankable and the metric would silently collapse.
    """
    scores = scores_for({1: 0.1, 2: 9.0, 3: 0.2}, num_items=3)
    # target 2 is repeated: it is in the history *and* is the target
    rank = rank_of_target(scores, 2, history=[2, 3], num_items=3)
    assert rank == 1, "repeated target must remain eligible and rank normally"

    # and the candidate list still contains the target
    candidates = valid_candidates(3, history=[2, 3], target_item_id=2)
    assert 2 in candidates
    assert 3 not in candidates  # other history items stay masked
    assert 1 in candidates


def test_repeated_target_rank_is_not_artificially_improved() -> None:
    """Retention must not skip real competitors, only the seen ones."""
    scores = scores_for({1: 9.0, 2: 5.0, 3: 0.1}, num_items=3)
    # history contains the target 2 and the low item 3; item 1 has no history
    rank = rank_of_target(scores, 2, history=[2, 3], num_items=3)
    assert rank == 2, "item 1 legitimately outranks the target and must be counted"


def test_duplicate_history_items_are_equivalent_to_distinct_set() -> None:
    """Masking is set-based, so repeated history entries change nothing."""
    scores = scores_for({1: 9.0, 2: 0.5, 3: 0.6}, num_items=3)
    # item 1 (the only competitor) is masked by the history, so the target ranks 1
    once = rank_of_target(scores, 3, history=[1], num_items=3)
    many = rank_of_target(scores, 3, history=[1, 1, 1], num_items=3)
    assert once == many == 1


def test_masking_changes_the_rank_and_is_owned_by_the_evaluator() -> None:
    """Without the history the competitor counts; with it masked, the target wins."""
    scores = scores_for({1: 9.0, 2: 0.5, 3: 0.6}, num_items=3)
    assert rank_of_target(scores, 3, num_items=3) == 2           # item 1 outranks
    assert rank_of_target(scores, 3, history=[1], num_items=3) == 1  # item 1 masked


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #


def test_hit_at_k_boundary() -> None:
    """HR@K is 1.0 exactly when rank <= K."""
    assert hit_at_k(1, 10) == 1.0
    assert hit_at_k(10, 10) == 1.0
    assert hit_at_k(11, 10) == 0.0


def test_ndcg_at_rank_one_is_exactly_one() -> None:
    """NDCG@K at rank 1 is 1/log2(2) == 1.0 (guards the rank+1 indexing)."""
    for k in (1, 5, 10):
        assert ndcg_at_k(1, k) == 1.0


def test_ndcg_hand_computed_values() -> None:
    """NDCG@K equals 1/log2(rank+1) when in range, else 0.0."""
    assert ndcg_at_k(2, 5) == 1.0 / math.log2(3)
    assert ndcg_at_k(3, 5) == 1.0 / math.log2(4)
    assert ndcg_at_k(3, 5) == 0.5
    assert ndcg_at_k(4, 5) == 1.0 / math.log2(5)
    assert ndcg_at_k(6, 5) == 0.0


def test_ndcg_has_no_off_by_one_error_at_cutoff() -> None:
    """Rank == K scores; rank == K+1 does not (the classic off-by-one)."""
    assert ndcg_at_k(5, 5) == 1.0 / math.log2(6)
    assert ndcg_at_k(6, 5) == 0.0
    assert hit_at_k(5, 5) == 1.0
    assert hit_at_k(6, 5) == 0.0


def test_metric_helpers_reject_bad_ranks() -> None:
    """Ranks are 1-based integers."""
    for bad in (0, -1, 1.5, "1", True):
        for fn in (hit_at_k, ndcg_at_k):
            try:
                fn(bad, 5)  # type: ignore[arg-type]
            except EvaluationError:
                continue
            raise AssertionError(f"expected EvaluationError for rank={bad!r}")


def test_evaluate_case_target_rank_one_all_cutoffs() -> None:
    """A rank-1 target hits every cutoff with perfect NDCG."""
    scores = scores_for({1: 0.1, 2: 0.2, 3: 9.0}, num_items=3)
    result = evaluate_case(scores, 3, num_items=3, k_values=(5, 10, 20))

    assert result.rank == 1
    assert result.metrics["HR@5"] == 1.0
    assert result.metrics["Recall@5"] == 1.0
    assert result.metrics["NDCG@5"] == 1.0
    assert result.metrics["NDCG@20"] == 1.0
    assert result.num_candidates == 3


def test_evaluate_case_target_rank_equals_k_and_k_plus_one() -> None:
    """Rank == K hits; rank == K+1 misses, for identical score layouts."""
    # target 5 at rank 3: two better items
    scores = scores_for({1: 9.0, 2: 8.0, 5: 1.0}, num_items=5)

    at_k = evaluate_case(scores, 5, num_items=5, k_values=(3,))
    assert at_k.rank == 3
    assert at_k.metrics["HR@3"] == 1.0
    assert at_k.metrics["NDCG@3"] == 1.0 / math.log2(4) == 0.5

    below_k = evaluate_case(scores, 5, num_items=5, k_values=(2,))
    assert below_k.rank == 3
    assert below_k.metrics["HR@2"] == 0.0
    assert below_k.metrics["Recall@2"] == 0.0
    assert below_k.metrics["NDCG@2"] == 0.0


def test_evaluate_case_reports_masked_count() -> None:
    """num_masked / num_candidates describe the candidate pool."""
    scores = scores_for({1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0, 5: 1.0}, num_items=5)
    result = evaluate_case(scores, 5, history=[1, 2], num_items=5)
    assert result.num_candidates == 3          # 3, 4, 5
    assert result.num_masked == 2


def test_hr_and_recall_are_identical_per_case() -> None:
    """Single-positive protocol: HR@K == Recall@K for every case."""
    scores = scores_for({1: 9.0, 2: 8.0, 3: 1.0}, num_items=3)
    result = evaluate_case(scores, 3, num_items=3, k_values=(1, 2, 5))
    for k in (1, 2, 5):
        assert result.metrics[f"HR@{k}"] == result.metrics[f"Recall@{k}"]


# --------------------------------------------------------------------------- #
# Optimised rank vs explicit-sort oracle
# --------------------------------------------------------------------------- #


def test_optimised_rank_equals_sort_oracle_on_hand_case() -> None:
    """Both implementations agree on a tie-heavy hand-computed vector."""
    scores = scores_for({1: 0.5, 2: 0.9, 3: 0.5, 4: 0.9, 5: 0.5}, num_items=5)
    assert sorted_ranking(scores, num_items=5) == [2, 4, 1, 3, 5]
    for target in (1, 2, 3, 4, 5):
        for history in ([], [1], [1, 2], [5, 4, 3]):
            if target in history:
                continue
            assert rank_of_target(scores, target, history=history, num_items=5) == \
                rank_of_target_via_sort(scores, target, history=history, num_items=5)


def test_optimised_rank_equals_sort_oracle_randomised() -> None:
    """Randomised (seeded) equivalence check over many tie-heavy vectors.

    Scores are drawn from a small integer set so ties are frequent; the seed is
    fixed so the check is deterministic.
    """
    rng = random.Random(20260913)
    comparisons = 0
    for _ in range(300):
        num_items = rng.randint(3, 25)
        score_pool = [0.0, 1.0, 2.0, 3.0]  # heavy ties
        scores = [-1e9] + [rng.choice(score_pool) for _ in range(num_items)]
        target = rng.randint(1, num_items)
        history = [rng.randint(1, num_items) for _ in range(rng.randint(0, num_items))]
        history = [h for h in history if h != target]

        optimised = rank_of_target(scores, target, history=history, num_items=num_items)
        oracle = rank_of_target_via_sort(scores, target, history=history, num_items=num_items)

        assert optimised == oracle, (
            f"rank mismatch: optimised={optimised} oracle={oracle} "
            f"num_items={num_items} target={target} history={history}"
        )
        assert 1 <= optimised <= num_items
        comparisons += 1
    assert comparisons == 300


def test_sort_oracle_candidate_list_excludes_seen_but_keeps_target() -> None:
    """The oracle's candidate list obeys the same masking rule."""
    scores = scores_for({1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0}, num_items=4)
    ranking = sorted_ranking(scores, history=[1, 4], target_item_id=4, num_items=4)
    assert 4 in ranking          # target retained
    assert 1 not in ranking      # other seen item masked
    assert sorted(ranking) == [2, 3, 4]


# --------------------------------------------------------------------------- #
# Invalid input handling
# --------------------------------------------------------------------------- #


def test_score_vector_length_is_validated() -> None:
    """A wrong-length score vector fails loudly instead of shifting ranks."""
    evaluator = FullRankingEvaluator(num_items=5)
    assert evaluator.expected_score_length() == 6

    for bad in ([0.0] * 5, [0.0] * 7, []):
        try:
            evaluator.rank(bad, 3)
        except EvaluationError as exc:
            assert "length" in str(exc)
            continue
        raise AssertionError(f"expected EvaluationError for len={len(bad)}")

    try:
        evaluator.rank(None, 3)  # type: ignore[arg-type]
    except EvaluationError:
        pass
    else:
        raise AssertionError("expected EvaluationError for None scores")


def test_invalid_target_ids_are_rejected() -> None:
    """PAD, out-of-range, negative and non-integer targets are all rejected."""
    scores = scores_for({}, num_items=5)
    for bad in (0, -1, 6, 100, 2.0, "2", True, None):
        try:
            rank_of_target(scores, bad, num_items=5)  # type: ignore[arg-type]
        except EvaluationError:
            continue
        raise AssertionError(f"expected EvaluationError for target={bad!r}")


def test_invalid_history_items_are_rejected() -> None:
    """History containing PAD or out-of-catalog ids is an explicit error."""
    scores = scores_for({}, num_items=5)
    for bad_history in ([0], [6], [-1], [1, 0], ["1"], [1, 2.5]):
        try:
            rank_of_target(scores, 3, history=bad_history, num_items=5)  # type: ignore[arg-type]
        except EvaluationError:
            continue
        raise AssertionError(f"expected EvaluationError for history={bad_history!r}")


def test_invalid_num_items_is_rejected() -> None:
    """A non-positive catalogue size is rejected at construction time."""
    for bad in (0, -1, True, 2.5):
        try:
            FullRankingEvaluator(num_items=bad)  # type: ignore[arg-type]
        except EvaluationError:
            continue
        raise AssertionError(f"expected EvaluationError for num_items={bad!r}")


# --------------------------------------------------------------------------- #
# Aggregation
# --------------------------------------------------------------------------- #


def make_case(user_id: str, user_int_id: int, history: tuple[int, ...], target: int) -> EvaluationCase:
    """Build an EvaluationCase directly for evaluator tests."""
    return EvaluationCase(
        user_id=user_id,
        user_int_id=user_int_id,
        train_history=history,
        validation_target=target,
        test_target=target,
        sequence_length=len(history) + 2,
    )


def test_aggregate_means_over_cases() -> None:
    """Aggregated metrics are the arithmetic mean over cases."""
    scores = scores_for({1: 9.0, 2: 0.5, 3: 0.4}, num_items=3)
    # target 1 -> rank 1 (hit); target 3 -> rank 3 (miss at K=1)
    hit = evaluate_case(scores, 1, num_items=3, k_values=(1,))
    miss = evaluate_case(scores, 3, num_items=3, k_values=(1,))

    report = aggregate([hit, miss], k_values=(1,), catalog_size=3)
    assert report.num_cases == 2
    assert report.metrics["HR"][1] == 0.5
    assert report.metrics["Recall"][1] == 0.5
    assert report.metrics["NDCG"][1] == (1.0 + 0.0) / 2
    assert report.mean_target_rank == 2.0


def test_aggregate_is_order_independent() -> None:
    """Aggregation does not depend on the order cases are supplied in."""
    scores = scores_for({1: 9.0, 2: 5.0, 3: 1.0, 4: 0.1}, num_items=4)
    results = [
        evaluate_case(scores, 1, num_items=4, k_values=(2,)),
        evaluate_case(scores, 2, num_items=4, k_values=(2,)),
        evaluate_case(scores, 3, num_items=4, k_values=(2,)),
    ]
    forward = aggregate(results, k_values=(2,), catalog_size=4)
    backward = aggregate(list(reversed(results)), k_values=(2,), catalog_size=4)
    assert forward.metrics == backward.metrics
    assert forward.rank_histogram == backward.rank_histogram


def test_aggregate_empty_cohort_raises() -> None:
    """An empty cohort is an explicit error, never NaN metrics."""
    try:
        aggregate([], k_values=(5,), catalog_size=10)
    except EvaluationError as exc:
        assert "empty" in str(exc)
        return
    raise AssertionError("expected EvaluationError for an empty cohort")


def test_report_hr_recall_equivalence_is_reported() -> None:
    """The report exposes the documented HR/Recall equivalence per K."""
    scores = scores_for({1: 1.0, 2: 0.5}, num_items=2)
    results = [evaluate_case(scores, 1, num_items=2, k_values=(5, 20))]
    report = aggregate(results, k_values=(5, 20), catalog_size=2)
    assert compare_hr_recall(report) == {5: True, 20: True}
    assert report.as_dict()["metrics"]["HR"]["@5"] == 1.0
    assert "cohort=" in report.format()


def test_report_is_serialisable() -> None:
    """The report dictionary is JSON-friendly."""
    import json

    scores = scores_for({1: 1.0, 2: 0.5}, num_items=2)
    results = [evaluate_case(scores, 1, num_items=2, k_values=(5,))]
    report = aggregate(results, k_values=(5,), catalog_size=2)
    payload = json.loads(json.dumps(report.as_dict()))
    assert payload["num_cases"] == 1 and payload["catalog_size"] == 2


# --------------------------------------------------------------------------- #
# Evaluator driver
# --------------------------------------------------------------------------- #


def perfect_scorer(history: tuple[int, ...], target: int):
    """A scorer that always ranks the target first (oracle model)."""
    vector = [0.0] * 11
    vector[target] = 1.0
    return vector


def worst_scorer(history: tuple[int, ...], target: int):
    """A scorer that always ranks the target last."""
    vector = [0.0] * 11
    vector[target] = -1.0
    return vector


def test_evaluator_perfect_scorer_hits_everything() -> None:
    """A perfect scorer yields HR=Recall=NDCG=1.0 at every cutoff."""
    cases = [make_case("u1", 1, (1, 2), 3), make_case("u2", 2, (4, 5), 6)]
    evaluator = FullRankingEvaluator(num_items=10, k_values=(5, 10, 20))
    outcome = evaluator.evaluate(cases, perfect_scorer, mode="test")

    assert outcome.report.num_cases == 2
    for k in (5, 10, 20):
        assert outcome.report.metrics["HR"][k] == 1.0
        assert outcome.report.metrics["Recall"][k] == 1.0
        assert outcome.report.metrics["NDCG"][k] == 1.0
    assert outcome.hr_recall_agree == {5: True, 10: True, 20: True}


def test_evaluator_worst_scorer_misses_everything() -> None:
    """A worst-case scorer yields 0.0 metrics, never a negative or NaN."""
    cases = [make_case("u1", 1, (1, 2), 3)]
    evaluator = FullRankingEvaluator(num_items=10, k_values=(5,))
    outcome = evaluator.evaluate(cases, worst_scorer, mode="test")

    assert outcome.report.metrics["HR"][5] == 0.0
    assert outcome.report.metrics["NDCG"][5] == 0.0
    # History (1, 2) is masked, so the pool is {3..10} = 8 candidates and the
    # target scores below all of them -> rank 8, i.e. last *among candidates*.
    assert outcome.report.mean_target_rank == 8.0
    assert outcome.report.mean_num_candidates == 8.0


def test_evaluator_validation_and_test_modes_use_different_histories() -> None:
    """Validation uses train history; test additionally sees the validation item."""
    case = EvaluationCase(
        user_id="u1",
        user_int_id=1,
        train_history=(1, 2),
        validation_target=3,
        test_target=4,
        sequence_length=4,
    )
    seen: list[tuple[tuple[int, ...], int]] = []

    def recording_scorer(history, target):
        seen.append((history, target))
        return [0.0] * 11

    evaluator = FullRankingEvaluator(num_items=10, k_values=(5,))
    evaluator.evaluate([case], recording_scorer, mode="validation")
    evaluator.evaluate([case], recording_scorer, mode="test")

    assert seen[0] == ((1, 2), 3)         # validation: train history, val target
    assert seen[1] == ((1, 2, 3), 4)      # test: train + validation, test target


def test_evaluator_rejects_unknown_mode() -> None:
    """An unknown mode is an explicit error."""
    evaluator = FullRankingEvaluator(num_items=5)
    try:
        evaluator.evaluate([make_case("u", 1, (1,), 2)], perfect_scorer, mode="banana")
    except EvaluationError:
        return
    raise AssertionError("expected EvaluationError for an unknown mode")


def test_evaluator_rejects_empty_cohort() -> None:
    """Evaluating zero cases is an explicit error."""
    evaluator = FullRankingEvaluator(num_items=5)
    try:
        evaluator.evaluate([], perfect_scorer, mode="test")
    except EvaluationError as exc:
        assert "empty" in str(exc)
        return
    raise AssertionError("expected EvaluationError for an empty cohort")


def test_evaluator_propagates_invalid_scores_with_context() -> None:
    """A model returning the wrong score length is reported per user."""
    cases = [make_case("u_bad", 1, (1,), 2)]

    def bad_scorer(history, target):
        return [0.0] * 5  # too short for num_items=10

    evaluator = FullRankingEvaluator(num_items=10)
    try:
        evaluator.evaluate(cases, bad_scorer, mode="test")
    except EvaluationError as exc:
        assert "u_bad" in str(exc) and "length" in str(exc)
        return
    raise AssertionError("expected EvaluationError for a short score vector")


def test_evaluator_candidate_count_helper() -> None:
    """The candidate-count helper mirrors the masking semantics."""
    evaluator = FullRankingEvaluator(num_items=5)
    assert evaluator.candidate_count() == 5
    assert evaluator.candidate_count((1, 2)) == 3
    # A repeated target is retained: only item 1 stays masked, so 5 - 1 = 4.
    assert evaluator.candidate_count((1, 2), target_item_id=2) == 4
    assert evaluator.candidate_count((2,), target_item_id=2) == 5


def test_cohort_summary_reports_invariants_only() -> None:
    """The cohort summary exposes counts/invariants and no quality metrics."""
    cases, report = build_cohort({"u1": [1, 2, 3], "u2": [1, 2]}, num_items=5, user_int_ids={"u1": 1, "u2": 2})
    summary = cohort_summary(cases, report)

    assert summary["num_users_total"] == 2
    assert summary["num_users_eligible"] == 1
    assert summary["num_users_excluded"] == 1
    assert summary["targets"]["pad_present"] is False
    assert summary["targets"]["all_within_catalog"] is True
    assert summary["history"]["pad_present"] is False
    assert summary["history"]["test_history_extends_train_by_one"] is True
    assert summary["history"]["test_history_last_is_validation_target"] is True
    assert summary["history"]["validation_history_is_train_history"] is True
    # no ranking metrics are produced without a scorer
    assert "metrics" not in summary


# --------------------------------------------------------------------------- #
# Non-finite score contract (evaluation-layer hardening)
# --------------------------------------------------------------------------- #


def test_non_finite_scores_are_rejected() -> None:
    """NaN and +/-inf anywhere in the vector are explicit errors.

    Regression test for a real defect: before this guard, a ``NaN`` target score
    was silently ranked **first** (every IEEE-754 comparison against NaN is false,
    so ``rank`` never incremented), which reported a hit for an unscoreable case.
    """
    num_items = 5
    for bad_value in (float("nan"), float("inf"), float("-inf")):
        base = scores_for({1: 0.5, 2: 0.4, 3: 0.3}, num_items=num_items)

        # as the target score
        target_bad = list(base)
        target_bad[3] = bad_value
        # as a competitor score
        rival_bad = list(base)
        rival_bad[1] = bad_value
        # inside the PAD slot
        pad_bad = list(base)
        pad_bad[PAD] = bad_value

        for label, vector in (("target", target_bad), ("rival", rival_bad), ("pad", pad_bad)):
            for call in (
                lambda v=vector: rank_of_target(v, 3, num_items=num_items),
                lambda v=vector: rank_of_target_via_sort(v, 3, num_items=num_items),
                lambda v=vector: sorted_ranking(v, num_items=num_items),
                lambda v=vector: evaluate_case(v, 3, num_items=num_items, k_values=(5,)),
            ):
                try:
                    call()
                except EvaluationError as exc:
                    assert "finite" in str(exc)
                    continue
                raise AssertionError(
                    f"{bad_value!r} accepted in {label} position; ranking must reject it"
                )


def test_nan_target_would_otherwise_rank_first() -> None:
    """Documents the exact defect the guard prevents.

    The assertion is about the *comparison semantics* the guard protects against,
    not about a supported code path: NaN comparisons are all false, so a naive
    count-the-better-items rank yields 1 for an unscoreable target.
    """
    scores = [0.0, 0.9, 0.8, float("nan"), 0.7, 0.6]
    naive_rank = 1
    for item_id in range(1, 6):
        if item_id == 3:
            continue
        if scores[item_id] > scores[3]:          # always False for NaN
            naive_rank += 1
        elif scores[item_id] == scores[3]:       # always False for NaN
            naive_rank += 1
    assert naive_rank == 1, "naive NaN ranking claims the best possible rank"

    # ...and the hardened API refuses to produce it.
    try:
        rank_of_target(scores, 3, num_items=5)
    except EvaluationError:
        return
    raise AssertionError("expected EvaluationError for a NaN target score")


def test_finite_extreme_scores_are_still_accepted() -> None:
    """Large finite values remain valid; only non-finite values are rejected."""
    scores = scores_for({1: 1e308, 2: -1e308, 3: 0.0}, num_items=3)
    assert rank_of_target(scores, 3, num_items=3) == 2
    assert rank_of_target(scores, 1, num_items=3) == 1


def test_non_numeric_score_entries_are_rejected() -> None:
    """A string or None inside the vector is reported with its index."""
    for bad in ("0.5", None):
        vector = scores_for({1: 0.5, 2: 0.4, 3: 0.3}, num_items=3)
        vector[2] = bad  # type: ignore[assignment]
        try:
            rank_of_target(vector, 3, num_items=3)  # type: ignore[arg-type]
        except EvaluationError as exc:
            assert "scores[2]" in str(exc)
            continue
        raise AssertionError(f"expected EvaluationError for a {bad!r} score entry")

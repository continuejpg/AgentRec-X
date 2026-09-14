"""Deterministic serving-time ranking tests (Milestone 6).

The ranking layer is production semantics, deliberately different from the evaluator:
there is no target, so candidates are the catalog minus the seen items, PAD is never
eligible, and ties break by lower item id.

Every case is checked against an explicit full-sort oracle so the vectorised path can
never silently drift from the documented rule.
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402

from recommendation.inference.ranking import (  # noqa: E402
    PAD_ID,
    RankingError,
    eligible_candidate_count,
    rank_top_k,
    reference_rank_top_k,
    validate_k,
    validate_score_vector,
)

NUM_ITEMS = 8


def scores_for(entries: dict[int, float], num_items: int = NUM_ITEMS) -> list[float]:
    """Build a score vector; unlisted items score 0.0, PAD is a low sentinel."""
    vector = [-1e9] * (num_items + 1)
    for item_id, value in entries.items():
        vector[item_id] = value
    return vector


def ids(result) -> list[int]:
    """Extract item ids from a ranking result."""
    return [item.item_id for item in result]


def assert_matches_oracle(scores, *, seen=(), k=5, num_items=NUM_ITEMS) -> list:
    """Assert the vectorised ranking equals the explicit-sort oracle."""
    produced = rank_top_k(scores, num_items=num_items, seen_item_ids=seen, k=k)
    oracle = reference_rank_top_k(scores, num_items=num_items, seen_item_ids=seen, k=k)
    assert [(r.rank, r.item_id, r.score) for r in produced] == [
        (r.rank, r.item_id, r.score) for r in oracle
    ], f"vectorised={produced} oracle={oracle}"
    return produced


# --------------------------------------------------------------------------- #
# Basic ordering and rank numbering
# --------------------------------------------------------------------------- #


def test_descending_score_order() -> None:
    """Higher scores come first and ranks are 1-based and contiguous."""
    scores = scores_for({1: 0.1, 2: 0.9, 3: 0.5, 4: 0.7})
    result = assert_matches_oracle(scores, k=4)
    assert ids(result) == [2, 4, 3, 1]
    assert [item.rank for item in result] == [1, 2, 3, 4]


def test_k_limits_result_length() -> None:
    """At most k items are returned."""
    scores = scores_for({i: float(i) for i in range(1, NUM_ITEMS + 1)})
    assert len(rank_top_k(scores, num_items=NUM_ITEMS, k=3)) == 3
    assert ids(rank_top_k(scores, num_items=NUM_ITEMS, k=3)) == [8, 7, 6]


# --------------------------------------------------------------------------- #
# Tie rule
# --------------------------------------------------------------------------- #


def test_all_scores_equal_ranks_by_item_id() -> None:
    """With every score equal, ranking is exactly ascending item id."""
    scores = scores_for({i: 1.0 for i in range(1, NUM_ITEMS + 1)})
    result = assert_matches_oracle(scores, k=NUM_ITEMS)
    assert ids(result) == list(range(1, NUM_ITEMS + 1))


def test_lower_item_id_wins_exact_tie() -> None:
    """An exact score tie is broken by the smaller item id."""
    scores = scores_for({3: 0.5, 5: 0.5, 7: 0.5})
    result = assert_matches_oracle(scores, k=3)
    assert ids(result) == [3, 5, 7]


def test_ties_at_top_k_boundary_are_deterministic() -> None:
    """Which tied item falls inside vs outside the cut is decided by item id.

    Items 2,4,6,8 all tie; with k=2 the two lowest ids must be selected, and the
    result must be identical on every call.
    """
    scores = scores_for({2: 1.0, 4: 1.0, 6: 1.0, 8: 1.0})
    first = assert_matches_oracle(scores, k=2)
    assert ids(first) == [2, 4]
    for _ in range(5):
        assert ids(rank_top_k(scores, num_items=NUM_ITEMS, k=2)) == [2, 4]


def test_duplicate_scores_and_unique_winner() -> None:
    """A unique top score wins regardless of tie structure below it."""
    scores = scores_for({1: 0.9, 2: 0.9, 3: 0.9, 5: 5.0})
    result = assert_matches_oracle(scores, k=4)
    assert ids(result) == [5, 1, 2, 3]


# --------------------------------------------------------------------------- #
# PAD and seen-item masking
# --------------------------------------------------------------------------- #


def test_pad_is_never_recommended_even_when_highest() -> None:
    """PAD (index 0) is excluded positionally, whatever its score."""
    scores = scores_for({2: 0.1})
    scores[PAD_ID] = 1e9
    result = assert_matches_oracle(scores, k=NUM_ITEMS)
    assert PAD_ID not in ids(result)
    assert ids(result)[0] == 2


def test_seen_items_are_excluded_even_when_highest() -> None:
    """Seen items with the top scores are masked out; unseen items rank below them.

    Scores are given explicitly for every catalog item so the expected order is
    unambiguous (unlisted items would otherwise take the low sentinel and still be
    legitimate candidates).
    """
    scores = [0.0] + [0.0] * NUM_ITEMS
    scores[1], scores[2], scores[3], scores[4] = 9.0, 8.0, 1.0, 0.5
    result = assert_matches_oracle(scores, seen=(1, 2), k=2)
    assert ids(result) == [3, 4]  # the two highest-scoring *unseen* items
    assert 1 not in ids(result) and 2 not in ids(result)


def test_seen_mask_uses_the_full_history_not_the_truncated_window() -> None:
    """Seen masking is driven by the caller's history, so long histories mask fully."""
    long_history = list(range(1, NUM_ITEMS))  # items 1..7 seen
    scores = scores_for({i: float(10 - i) for i in range(1, NUM_ITEMS + 1)})
    result = assert_matches_oracle(scores, seen=long_history, k=NUM_ITEMS)
    assert ids(result) == [NUM_ITEMS]


def test_out_of_catalog_seen_ids_are_tolerated() -> None:
    """Ids the catalog does not contain cannot be recommended, so they are ignored.

    Only the in-catalog seen id (1) removes a candidate; 999 and PAD (0) are simply
    not candidates in the first place.
    """
    scores = [0.0] + [0.0] * NUM_ITEMS
    scores[1], scores[2] = 1.0, 0.5
    result = rank_top_k(scores, num_items=NUM_ITEMS, seen_item_ids=[1, 999, 0], k=3)
    assert ids(result) == [2, 3, 4]
    assert 1 not in ids(result)
    assert eligible_candidate_count(NUM_ITEMS, [1, 999, 0]) == NUM_ITEMS - 1


def test_no_duplicate_recommendations() -> None:
    """Every returned item appears exactly once."""
    scores = scores_for({i: 1.0 for i in range(1, NUM_ITEMS + 1)})
    result = rank_top_k(scores, num_items=NUM_ITEMS, k=NUM_ITEMS)
    assert len(ids(result)) == len(set(ids(result)))


# --------------------------------------------------------------------------- #
# Candidate exhaustion
# --------------------------------------------------------------------------- #


def test_fewer_than_k_candidates_returns_available() -> None:
    """When fewer than k unseen items remain, return what exists."""
    scores = scores_for({i: float(i) for i in range(1, NUM_ITEMS + 1)})
    result = assert_matches_oracle(scores, seen=(3, 4, 5, 6, 7, 8), k=5)
    assert ids(result) == [2, 1]
    assert len(result) == 2


def test_zero_candidates_returns_empty_list() -> None:
    """A fully-seen catalog yields an empty recommendation list, not an error."""
    scores = scores_for({i: 1.0 for i in range(1, NUM_ITEMS + 1)})
    result = assert_matches_oracle(scores, seen=range(1, NUM_ITEMS + 1), k=10)
    assert result == []


def test_eligible_candidate_count_helper() -> None:
    """Eligible candidates are the catalog minus distinct valid seen items."""
    assert eligible_candidate_count(NUM_ITEMS) == NUM_ITEMS
    assert eligible_candidate_count(NUM_ITEMS, [1, 2]) == NUM_ITEMS - 2
    assert eligible_candidate_count(NUM_ITEMS, [1, 1, 1]) == NUM_ITEMS - 1
    assert eligible_candidate_count(NUM_ITEMS, [999]) == NUM_ITEMS


# --------------------------------------------------------------------------- #
# Non-finite scores fail fast
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_non_finite_scores_are_rejected(bad: float) -> None:
    """NaN / +Inf / -Inf anywhere in the vector fail fast rather than ranking."""
    scores = scores_for({1: 1.0, 2: 0.5})
    scores[3] = bad
    with pytest.raises(RankingError, match="non-finite"):
        rank_top_k(scores, num_items=NUM_ITEMS, k=3)
    with pytest.raises(RankingError, match="non-finite"):
        validate_score_vector(scores, NUM_ITEMS)


def test_non_finite_pad_score_is_also_rejected() -> None:
    """The PAD slot is validated too, so a NaN-padded vector cannot slip through."""
    scores = scores_for({1: 1.0})
    scores[PAD_ID] = float("nan")
    with pytest.raises(RankingError):
        validate_score_vector(scores, NUM_ITEMS)


def test_numpy_input_is_accepted() -> None:
    """An ndarray works and produces the same result as a list."""
    values = scores_for({2: 0.4, 5: 0.9})
    assert ids(rank_top_k(np.asarray(values), num_items=NUM_ITEMS, k=2)) == ids(
        rank_top_k(values, num_items=NUM_ITEMS, k=2)
    )


# --------------------------------------------------------------------------- #
# Vector / k validation
# --------------------------------------------------------------------------- #


def test_score_vector_length_is_validated() -> None:
    """A wrong-length vector is rejected with the expected length in the message."""
    for bad in ([0.0] * NUM_ITEMS, [0.0] * (NUM_ITEMS + 2), []):
        with pytest.raises(RankingError, match="length"):
            rank_top_k(bad, num_items=NUM_ITEMS, k=1)


def test_multidimensional_scores_are_rejected() -> None:
    """Only a flat catalog vector is a valid score vector."""
    with pytest.raises(RankingError, match="1-D"):
        rank_top_k(np.zeros((2, NUM_ITEMS + 1)), num_items=NUM_ITEMS, k=1)


def test_invalid_num_items_is_rejected() -> None:
    """A non-positive catalogue size is rejected."""
    for bad in (0, -1, True, 2.5):
        with pytest.raises(RankingError):
            rank_top_k([0.0] * 4, num_items=bad, k=1)  # type: ignore[arg-type]


def test_k_validation() -> None:
    """k must be an integer in 1..100."""
    assert validate_k(1) == 1
    assert validate_k(100) == 100
    for bad in (0, -1, 101, 1.5, "10", True):
        with pytest.raises(RankingError):
            validate_k(bad)  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# Randomised equivalence with the explicit-sort oracle
# --------------------------------------------------------------------------- #


def test_randomised_equivalence_with_oracle() -> None:
    """Seeded randomised check across tie-heavy and unique-score regimes."""
    rng = random.Random(2026)
    for trial in range(200):
        num_items = rng.randint(1, 20)
        pool = rng.choice(([0.0, 1.0], [0.0, 0.5, 1.0, 2.0], [0.0, 1e-9, 1e9]))
        scores = [-1e9] + [rng.choice(pool) for _ in range(num_items)]
        scores[0] = rng.choice(pool)
        seen = [rng.randint(0, num_items + 2) for _ in range(rng.randint(0, num_items))]
        k = rng.randint(1, 100)
        produced = rank_top_k(scores, num_items=num_items, seen_item_ids=seen, k=k)
        oracle = reference_rank_top_k(scores, num_items=num_items, seen_item_ids=seen, k=k)
        assert [(r.rank, r.item_id, r.score) for r in produced] == [
            (r.rank, r.item_id, r.score) for r in oracle
        ], f"trial {trial}: num_items={num_items} seen={seen} k={k}"


def test_ranking_is_repeatable() -> None:
    """Identical inputs always produce identical output."""
    rng = random.Random(7)
    scores = [-1e9] + [rng.choice([0.0, 1.0, 2.0]) for _ in range(30)]
    seen = [1, 5, 9]
    baseline = [(r.rank, r.item_id, r.score) for r in rank_top_k(scores, num_items=30, seen_item_ids=seen, k=10)]
    for _ in range(10):
        assert [
            (r.rank, r.item_id, r.score) for r in rank_top_k(scores, num_items=30, seen_item_ids=seen, k=10)
        ] == baseline

"""Phase 2B.1: tests for the exploratory metrics used by the ablation driver.

The functions are imported from the driver itself, so the test exercises the same code the
experiment ran rather than a copy of it.
"""

from __future__ import annotations

import pytest

from experiments.p2b_agent_source_limit_ablation import (
    ARMS,
    hr_at_k,
    ndcg_at_k,
    paired_ci,
)


def test_hit_inside_k_scores_one() -> None:
    assert hr_at_k(["a", "b", "c"], "b", 3) == 1.0


def test_hit_at_exactly_k_scores_one() -> None:
    assert hr_at_k(["a", "b", "c"], "c", 3) == 1.0


def test_hit_outside_k_scores_zero() -> None:
    assert hr_at_k(["a", "b", "c", "d"], "d", 3) == 0.0


def test_miss_scores_zero() -> None:
    assert hr_at_k(["a", "b"], "z", 5) == 0.0


def test_empty_list_scores_zero_and_is_not_an_error() -> None:
    assert hr_at_k([], "a", 5) == 0.0
    assert ndcg_at_k([], "a", 5) == 0.0


def test_short_list_scores_zero_without_padding() -> None:
    assert hr_at_k(["a"], "z", 5) == 0.0


def test_ndcg_at_rank_one_is_one() -> None:
    assert ndcg_at_k(["a", "b"], "a", 5) == pytest.approx(1.0)


def test_ndcg_decays_with_rank() -> None:
    assert ndcg_at_k(["a", "b", "c"], "c", 5) == pytest.approx(0.5)


def test_ndcg_outside_k_scores_zero() -> None:
    assert ndcg_at_k(["a", "b", "c", "d"], "d", 3) == 0.0


def test_duplicate_identities_do_not_inflate_ndcg() -> None:
    # The first occurrence is the hit; a later duplicate cannot add credit.
    assert ndcg_at_k(["a", "a", "b"], "b", 3) == pytest.approx(0.5)


def test_a_missing_target_scores_zero() -> None:
    # A missing label cannot be a hit: the driver treats the case as a miss, never as a drop.
    assert hr_at_k(["a", "b"], None, 2) == 0.0
    assert hr_at_k(["a", "b"], "nope", 2) == 0.0


def test_paired_ci_is_deterministic_and_brackets_the_mean() -> None:
    diffs = [0.0] * 8 + [1.0] * 4
    first = paired_ci(diffs, draws=200)
    second = paired_ci(diffs, draws=200)
    assert first == second
    assert first[0] <= sum(diffs) / len(diffs) <= first[1]


def test_paired_ci_of_identical_arms_is_zero_width() -> None:
    assert paired_ci([0.0] * 5, draws=100) == [0.0, 0.0]


def test_the_arm_ladder_is_fixed() -> None:
    assert ARMS == (5, 20, 50, 100)

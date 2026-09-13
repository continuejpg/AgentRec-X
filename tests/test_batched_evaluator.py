"""Batched vs canonical full-ranking evaluator equivalence (Milestone 5).

The canonical per-case evaluator is the semantic oracle.  These tests prove the
batched path reproduces it exactly - same target ranks, same HR/Recall/NDCG - across
every adversarial case the milestone lists, including ties, PAD/seen high scores,
repeated targets and non-finite scores.  No formal training may begin until this gate
passes.
"""

from __future__ import annotations

import math
import random
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

torch = pytest.importorskip("torch", reason="batched evaluator tests require PyTorch")

from recommendation.evaluation import (  # noqa: E402
    EvaluationCase,
    EvaluationError,
    FullRankingEvaluator,
    aggregate,
    evaluate_case,
    rank_of_target,
)
from recommendation.evaluation.batched import (  # noqa: E402
    batched_target_ranks,
    build_candidate_mask,
    canonical_ranks_for_reference,
    ensure_finite_scores_tensor,
    evaluate_batched,
)

NUM_ITEMS = 12
K_VALUES = (5, 10, 20)


def scores_from(rows: list[list[float]]) -> torch.Tensor:
    """Build a [batch, num_items+1] tensor, validating the width."""
    tensor = torch.tensor(rows, dtype=torch.float32)
    assert tensor.shape[1] == NUM_ITEMS + 1
    return tensor


def vector(**entries: float) -> list[float]:
    """Build one score row from ``i1=...`` style keyword arguments."""
    row = [-1e9] * (NUM_ITEMS + 1)
    for key, value in entries.items():
        row[int(key[1:])] = value
    return row


def both_ranks(rows, histories, targets):
    """Return ``(canonical_ranks, batched_ranks)`` for the same inputs."""
    scores = scores_from(rows)
    canonical = canonical_ranks_for_reference(rows, histories, targets, num_items=NUM_ITEMS)
    mask = build_candidate_mask(len(targets), NUM_ITEMS, histories, targets)
    batched = batched_target_ranks(scores, mask, targets)
    return canonical, batched


def assert_equivalent(rows, histories, targets) -> list[int]:
    """Assert canonical == batched ranks and return them.

    Rows are recycled if fewer are supplied than there are targets, which keeps the
    shared-score fixtures concise.
    """
    if len(rows) < len(targets):
        rows = [rows[i % len(rows)] for i in range(len(targets))]
    canonical, batched = both_ranks(rows, histories, targets)
    assert batched == canonical, f"rank mismatch: canonical={canonical} batched={batched}"
    return canonical


# --------------------------------------------------------------------------- #
# Masking semantics
# --------------------------------------------------------------------------- #


def test_pad_is_never_a_candidate() -> None:
    """PAD scores highest of all yet must not change any rank."""
    histories: list[list[int]] = [[], []]
    targets = [3, 7]
    rows = [vector(i3=0.5), vector(i7=0.5)]
    for row in rows:  # PAD at index 0 scores highest
        row[0] = 1e9
    ranks = assert_equivalent(rows, histories, targets)
    assert ranks == [1, 1]

    mask = build_candidate_mask(2, NUM_ITEMS, histories, targets)
    assert not bool(mask[:, 0].any()), "PAD must never be masked in as a candidate"


def test_pad_high_score_does_not_inflate_rank() -> None:
    """A huge PAD score must leave the target rank unchanged."""
    baseline = assert_equivalent(
        [vector(i1=0.9, i3=0.5)], [[]], [3]
    )
    with_pad = [vector(i1=0.9, i3=0.5)]
    with_pad[0][0] = 1e12
    assert assert_equivalent(with_pad, [[]], [3]) == baseline == [2]


def test_seen_items_are_excluded() -> None:
    """History items are masked, so a high-scoring seen item does not beat the target."""
    rows = [vector(i1=9.0, i2=8.0, i3=0.5)]
    unseen_rank = assert_equivalent(rows, [[]], [3])
    seen_rank = assert_equivalent(rows, [[1, 2]], [3])
    assert unseen_rank == [3]
    assert seen_rank == [1], "masked seen items must not count as competitors"


def test_repeated_target_is_retained() -> None:
    """A target that also appears earlier in the history stays eligible."""
    rows = [vector(i1=9.0, i2=0.5, i3=0.1)]
    # target 2 repeats in the history; without retention it would be unrankable
    ranks = assert_equivalent(rows, [[1, 2]], [2])
    # item 1 (score 9.0) is masked because it is in the history; every other item
    # scores below the target, so the retained target ranks first.
    assert ranks == [1], "the masked seen item must not count; the target is retained"

    mask = build_candidate_mask(1, NUM_ITEMS, [[1, 2]], [2])
    assert bool(mask[0, 2]), "the repeated target must remain a candidate"
    assert not bool(mask[0, 1]), "other history items stay masked"


def test_target_previously_seen_many_times() -> None:
    """Heavy repetition of the target in the history does not change its rank."""
    rows = [vector(i4=0.7, i5=0.9, i2=0.3)]
    once = assert_equivalent(rows, [[2]], [2])
    many = assert_equivalent(rows, [[2, 2, 2, 2]], [2])
    assert once == many


def test_target_retention_with_seen_high_scorer() -> None:
    """Target retained while a higher-scoring *seen* item is masked."""
    rows = [vector(i1=9.0, i2=9.0, i5=0.1)]
    ranks = assert_equivalent(rows, [[1, 5]], [5])
    # item 1 (9.0) is masked; item 2 (9.0) is an unseen legal candidate and outranks
    # the target, so the retained target lands at rank 2.
    assert ranks == [2], "masked seen item excluded, unseen high scorer still counts"


# --------------------------------------------------------------------------- #
# Tie rule
# --------------------------------------------------------------------------- #


def test_deterministic_tie_rule_lower_id_first() -> None:
    """Equal scores resolve by lower item id, for both target positions."""
    rows = [vector(i2=0.5, i5=0.5), vector(i2=0.5, i5=0.5)]
    ranks = assert_equivalent(rows, [[], []], [5, 2])
    assert ranks == [2, 1]


def test_all_equal_scores_equivalence() -> None:
    """With every score equal, rank is exactly the item id."""
    rows = [[0.0] + [1.0] * NUM_ITEMS for _ in range(NUM_ITEMS)]
    targets = list(range(1, NUM_ITEMS + 1))
    ranks = assert_equivalent(rows, [[] for _ in targets], targets)
    assert ranks == targets, "all-equal scores must rank by ascending item id"

    # ...and masking still shifts ranks predictably
    masked = assert_equivalent(rows, [[1] for _ in targets], targets)
    assert masked[0] == 1  # target 1 retained
    assert masked[1] == 1  # item 1 masked away -> target 2 now first


def test_intentional_ties_with_mixtures() -> None:
    """Mixed tie groups plus a unique item resolve deterministically."""
    rows = [vector(i1=0.5, i2=0.5, i3=0.5, i4=0.9, i5=0.1)]
    assert assert_equivalent(rows, [[]], [3]) == [4]
    assert assert_equivalent(rows, [[]], [4]) == [1]
    assert assert_equivalent(rows, [[]], [5]) == [5]


def test_target_as_smallest_and_largest_item_id() -> None:
    """Boundary item ids behave identically in both evaluators."""
    rows = [vector(i1=0.5, i2=0.5, i12=0.5)]
    smallest = assert_equivalent(rows, [[]], [1])
    largest = assert_equivalent(rows, [[]], [12])
    assert smallest == [1], "smallest id wins the tie"
    assert largest == [3], "largest id loses to both equal-scoring lower ids"


# --------------------------------------------------------------------------- #
# Target rank boundaries
# --------------------------------------------------------------------------- #


def test_target_rank_one_k_and_k_plus_one() -> None:
    """Ranks 1, K and K+1 all agree, and drive HR/NDCG boundaries."""
    # target 5 at rank 1
    r1_rows = [vector(i5=9.0)]
    # target 5 at rank 5 (four better items)
    r5_rows = [vector(i1=9.0, i2=8.0, i3=7.0, i4=6.0, i5=1.0)]
    # target 5 at rank 6 (five better items)
    r6_rows = [vector(i1=9.0, i2=8.0, i3=7.0, i4=6.0, i6=5.0, i5=1.0)]

    assert assert_equivalent(r1_rows, [[]], [5]) == [1]
    assert assert_equivalent(r5_rows, [[]], [5]) == [5]
    assert assert_equivalent(r6_rows, [[]], [5]) == [6]

    for rows, expected_hit in ((r5_rows, 1.0), (r6_rows, 0.0)):
        scores = scores_from(rows)[0]
        result = evaluate_case(scores, 5, num_items=NUM_ITEMS, k_values=(5,))
        assert result.metrics["HR@5"] == expected_hit
        assert result.metrics["Recall@5"] == expected_hit


# --------------------------------------------------------------------------- #
# Histories
# --------------------------------------------------------------------------- #


def test_short_and_long_histories_equivalence() -> None:
    """Empty, short and long/truncated histories all agree."""
    rows = [vector(i1=0.9, i2=0.8, i3=0.7, i7=0.5)]
    histories = [[], [1], [1, 2], [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]]
    expected = assert_equivalent(rows, histories, [7, 7, 7, 7])
    # masking more items can only improve (lower) the target's rank
    assert expected[0] >= expected[1] >= expected[2] >= expected[3]
    assert expected[3] == 1, "with everything else masked the target ranks first"


def test_duplicate_targets_across_rows() -> None:
    """Two rows sharing a target: it is a candidate in one and seen in the other."""
    rows = [vector(i1=0.5, i2=0.9, i3=0.4), vector(i1=0.9, i2=0.5, i3=0.4)]
    targets = [3, 3]
    histories: list[list[int]] = [[], [3]]  # row 1 sees 3 in its own history
    assert_equivalent(rows, histories, targets)


# --------------------------------------------------------------------------- #
# Randomised equivalence
# --------------------------------------------------------------------------- #


def test_random_scores_equivalence_seeded() -> None:
    """Seeded randomised equivalence over many batches and score regimes."""
    rng = random.Random(2026)
    for trial in range(25):
        batch = rng.randint(1, 24)
        pool = rng.choice(([0.0, 1.0], [0.0, 0.5, 1.0, 2.0], [0.0, 1e-8, 1e8]))
        rows = [
            [rng.choice(pool) for _ in range(NUM_ITEMS + 1)] for _ in range(batch)
        ]
        for row in rows:
            row[0] = rng.choice(pool)  # PAD can be arbitrary
        histories = [
            [rng.randint(1, NUM_ITEMS) for _ in range(rng.randint(0, 6))] for _ in range(batch)
        ]
        targets = [rng.randint(1, NUM_ITEMS) for _ in range(batch)]
        canonical, batched = both_ranks(rows, histories, targets)
        assert batched == canonical, f"trial {trial}: {canonical} != {batched}"


def test_random_scores_equivalence_with_extreme_magnitudes() -> None:
    """Large finite magnitudes and exact ties mixed together still agree."""
    rng = random.Random(7)
    rows = [[0.0] + [rng.choice([-1e12, 0.0, 1e12, 1e12]) for _ in range(NUM_ITEMS)] for _ in range(10)]
    histories = [[1, 2] for _ in range(10)]
    targets = [rng.randint(3, NUM_ITEMS) for _ in range(10)]
    assert_equivalent(rows, histories, targets)


# --------------------------------------------------------------------------- #
# Non-finite handling (fail fast, unchanged policy)
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_non_finite_batched_scores_fail_fast(bad: float) -> None:
    """A non-finite score anywhere in a batch is rejected, never clamped."""
    rows = [vector(i3=0.5)]
    rows[0][5] = bad
    scores = torch.tensor(rows, dtype=torch.float32)
    with pytest.raises(EvaluationError):
        ensure_finite_scores_tensor(scores)

    with pytest.raises(EvaluationError):
        evaluate_batched(
            num_items=NUM_ITEMS,
            score_batches=[([[]], [3], scores)],
            k_values=(5,),
        )


def test_non_finite_pad_score_also_fails_fast() -> None:
    """The whole vector is validated, including the PAD slot."""
    rows = [vector(i3=0.5)]
    rows[0][0] = float("nan")
    with pytest.raises(EvaluationError):
        ensure_finite_scores_tensor(torch.tensor(rows, dtype=torch.float32))


def test_canonical_and_batched_reject_non_finite_consistently() -> None:
    """Both evaluators refuse NaN, so neither can silently disagree."""
    rows = [vector(i3=0.5)]
    rows[0][4] = float("nan")
    with pytest.raises(EvaluationError):
        rank_of_target(scores_from(rows)[0], 3, num_items=NUM_ITEMS)
    with pytest.raises(EvaluationError):
        ensure_finite_scores_tensor(scores_from(rows))


# --------------------------------------------------------------------------- #
# Aggregation / metric equivalence
# --------------------------------------------------------------------------- #


def test_metric_aggregation_equivalence_batched_vs_canonical() -> None:
    """HR/Recall/NDCG at K=5,10,20 are identical for both evaluators."""
    rng = random.Random(99)
    rows = [[0.0] + [rng.random() for _ in range(NUM_ITEMS)] for _ in range(30)]
    histories = [[rng.randint(1, NUM_ITEMS) for _ in range(rng.randint(0, 4))] for _ in range(30)]
    targets = [rng.randint(1, NUM_ITEMS) for _ in range(30)]

    canonical_results = [
        evaluate_case(scores_from(rows)[row], targets[row], history=histories[row],
                      num_items=NUM_ITEMS, k_values=K_VALUES)
        for row in range(len(targets))
    ]
    canonical_report = aggregate(canonical_results, k_values=K_VALUES, catalog_size=NUM_ITEMS)

    batched = evaluate_batched(
        num_items=NUM_ITEMS,
        score_batches=[(histories, targets, scores_from(rows))],
        k_values=K_VALUES,
    )

    for k in K_VALUES:
        for name in ("HR", "Recall", "NDCG"):
            assert batched.report.metrics[name][k] == pytest.approx(
                canonical_report.metrics[name][k], rel=0, abs=0
            ), f"{name}@{k} differs"
    assert batched.report.mean_target_rank == canonical_report.mean_target_rank
    assert batched.report.rank_histogram == canonical_report.rank_histogram


def test_hr_equals_recall_under_single_positive() -> None:
    """The single-positive identity holds in the batched path at every K."""
    rng = random.Random(5)
    rows = [[0.0] + [rng.random() for _ in range(NUM_ITEMS)] for _ in range(20)]
    targets = [rng.randint(1, NUM_ITEMS) for _ in range(20)]
    batched = evaluate_batched(
        num_items=NUM_ITEMS,
        score_batches=[([[] for _ in targets], targets, scores_from(rows))],
        k_values=K_VALUES,
    )
    assert all(batched.hr_recall_agree.values())
    for k in K_VALUES:
        assert batched.report.metrics["HR"][k] == batched.report.metrics["Recall"][k]


def test_empty_cohort_raises() -> None:
    """An empty cohort is an explicit error, not NaN metrics."""
    with pytest.raises(EvaluationError):
        evaluate_batched(num_items=NUM_ITEMS, score_batches=[], k_values=(5,))


def test_score_matrix_shape_is_validated() -> None:
    """A wrongly shaped batch is rejected."""
    with pytest.raises(EvaluationError):
        evaluate_batched(
            num_items=NUM_ITEMS,
            score_batches=[([[]], [3], torch.zeros((2, NUM_ITEMS + 1)))],
            k_values=(5,),
        )


def test_invalid_target_in_batch_is_rejected() -> None:
    """PAD / out-of-range targets fail loudly."""
    rows = [[0.0] * (NUM_ITEMS + 1)]
    for bad in (0, NUM_ITEMS + 1, -1):
        with pytest.raises(EvaluationError):
            build_candidate_mask(1, NUM_ITEMS, [[]], [bad])


# --------------------------------------------------------------------------- #
# Multi-batch consistency and evaluator integration
# --------------------------------------------------------------------------- #


def test_batching_partition_does_not_change_results() -> None:
    """Splitting the same cases into different batch sizes is result-invariant."""
    rng = random.Random(1234)
    total = 23
    rows = [[0.0] + [rng.random() for _ in range(NUM_ITEMS)] for _ in range(total)]
    histories = [[rng.randint(1, NUM_ITEMS) for _ in range(rng.randint(0, 3))] for _ in range(total)]
    targets = [rng.randint(1, NUM_ITEMS) for _ in range(total)]

    def run(batch_size: int):
        batches = []
        for start in range(0, total, batch_size):
            end = min(start + batch_size, total)
            batches.append((histories[start:end], targets[start:end], scores_from(rows[start:end])))
        return evaluate_batched(num_items=NUM_ITEMS, score_batches=batches, k_values=K_VALUES)

    reference = run(total)
    for size in (1, 2, 5, 7, 23):
        other = run(size)
        assert other.target_ranks == reference.target_ranks
        assert other.report.metrics == reference.report.metrics
        assert other.num_batches == math.ceil(total / size)


def test_evaluator_batched_method_matches_canonical_evaluate() -> None:
    """``FullRankingEvaluator.evaluate_cases_batched`` agrees with ``evaluate``."""
    rng = random.Random(4242)
    cases = []
    for index in range(16):
        history = tuple(rng.randint(1, NUM_ITEMS) for _ in range(rng.randint(1, 3)))
        target = rng.randint(1, NUM_ITEMS)
        cases.append(
            EvaluationCase(
                user_id=f"u{index}",
                user_int_id=index,
                train_history=history,
                validation_target=target,
                test_target=target,
                sequence_length=len(history) + 2,
            )
        )

    score_vectors = {
        (case.test_history, case.test_target): [0.0] + [rng.random() for _ in range(NUM_ITEMS)]
        for case in cases
    }

    def score_matrix_fn(pairs):
        return torch.tensor([score_vectors[pair] for pair in pairs], dtype=torch.float32)

    evaluator = FullRankingEvaluator(num_items=NUM_ITEMS, k_values=K_VALUES)
    canonical = evaluator.evaluate(cases, lambda history, target: score_vectors[(history, target)], mode="test")
    batched = evaluator.evaluate_cases_batched(
        cases, score_matrix_fn=score_matrix_fn, mode="test", batch_size=5
    )

    for k in K_VALUES:
        for name in ("HR", "Recall", "NDCG"):
            assert batched.report.metrics[name][k] == pytest.approx(
                canonical.report.metrics[name][k], rel=0, abs=0
            )
    assert batched.report.num_cases == canonical.report.num_cases
    assert batched.report.rank_histogram == canonical.report.rank_histogram


def test_float32_tensor_precision_note_and_exact_equivalence() -> None:
    """Both evaluators must see bit-identical values for equivalence to be meaningful.

    Ranking uses exact equality for the tie rule, so a float32 tensor compared
    against float64 Python floats can legitimately differ by one rank when rounding
    creates or destroys an exact tie.  ``canonical_ranks_for_reference`` therefore
    accepts a tensor and compares the same Python floats the batched path uses.
    """
    rng = random.Random(31)
    rows = [[0.0] + [rng.random() for _ in range(NUM_ITEMS)] for _ in range(40)]
    histories = [[1, 2] for _ in rows]
    targets = [rng.randint(3, NUM_ITEMS) for _ in rows]
    tensor = torch.tensor(rows, dtype=torch.float32)

    # the helper converts the tensor, so the oracle sees float32-rounded values
    oracle_ranks = canonical_ranks_for_reference(tensor, histories, targets, num_items=NUM_ITEMS)
    mask = build_candidate_mask(len(targets), NUM_ITEMS, histories, targets)
    batched = batched_target_ranks(tensor, mask, targets)
    assert batched == oracle_ranks

    # and at float64 there is no rounding at all
    double = torch.tensor(rows, dtype=torch.float64)
    assert batched_target_ranks(
        double, build_candidate_mask(len(targets), NUM_ITEMS, histories, targets), targets
    ) == canonical_ranks_for_reference(rows, histories, targets, num_items=NUM_ITEMS)

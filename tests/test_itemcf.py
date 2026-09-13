"""Tests for the ItemCF baseline (Milestone 2B).

Fixtures are tiny and hand-computable.  The canonical algorithmic oracle from the
milestone specification is::

    u1: [1,2,3]
    u2: [1,2]
    u3: [1,3]

    freq(1)=3  freq(2)=2  freq(3)=2
    cooc(1,2)=2  cooc(1,3)=2  cooc(2,3)=1
    sim(1,2)=2/sqrt(6)  sim(1,3)=2/sqrt(6)  sim(2,3)=1/2

Nothing here depends on the Amazon artifact.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from recommendation import config  # noqa: E402
from recommendation.baselines.itemcf import (  # noqa: E402
    DEFAULT_SCORE,
    ItemCF,
    ItemCFError,
    fit_from_cohort,
    make_scorer,
    similarity_pair_count,
    unique_items,
    validate_history,
)
from recommendation.evaluation import (  # noqa: E402
    DEFAULT_K_VALUES,
    EvaluationCase,
    FullRankingEvaluator,
    build_cohort,
)

TOL = 1e-12
ORACLE_TRAIN = [[1, 2, 3], [1, 2], [1, 3]]


def oracle_model() -> ItemCF:
    """Fit the specification's oracle fixture on a 3-item catalog."""
    model = ItemCF(num_items=3)
    model.fit(ORACLE_TRAIN)
    return model


def make_case(user_id: str, user_int_id: int, history: tuple[int, ...], target: int) -> EvaluationCase:
    """Build an EvaluationCase directly (test helper)."""
    return EvaluationCase(
        user_id=user_id,
        user_int_id=user_int_id,
        train_history=history,
        validation_target=target,
        test_target=target,
        sequence_length=len(history) + 2,
    )


# --------------------------------------------------------------------------- #
# 1-2. Item frequencies and co-occurrence
# --------------------------------------------------------------------------- #


def test_item_frequencies_match_oracle() -> None:
    """freq(i) counts training users whose unique history contains i."""
    model = oracle_model()
    assert model.item_freq == {1: 3, 2: 2, 3: 2}


def test_cooccurrence_counts_match_oracle() -> None:
    """cooc(i,j) counts training users whose unique history contains both."""
    model = oracle_model()
    assert model.cooccurrence_between(1, 2) == 2
    assert model.cooccurrence_between(1, 3) == 2
    assert model.cooccurrence_between(2, 3) == 1


# --------------------------------------------------------------------------- #
# 3-6. Similarity
# --------------------------------------------------------------------------- #


def test_cosine_similarity_matches_hand_calculation() -> None:
    """sim(i,j) = cooc / sqrt(freq_i * freq_j), exactly."""
    model = oracle_model()
    assert abs(model.similarity_between(1, 2) - 2 / math.sqrt(6)) < TOL
    assert abs(model.similarity_between(1, 3) - 2 / math.sqrt(6)) < TOL
    assert abs(model.similarity_between(2, 3) - 1 / 2) < TOL


def test_similarity_is_symmetric() -> None:
    """sim(i,j) == sim(j,i) for every learned pair."""
    model = oracle_model()
    for (i, j) in ((1, 2), (1, 3), (2, 3)):
        assert model.similarity_between(i, j) == model.similarity_between(j, i)


def test_similarity_symmetry_on_a_larger_fixture() -> None:
    """Symmetry holds on a fixture with uneven item frequencies."""
    model = ItemCF(num_items=6)
    model.fit([[1, 2], [1, 2, 3], [1, 4], [2, 4], [1, 2, 3, 4], [5, 6], [5]])
    for i in range(1, 7):
        for j in range(1, 7):
            assert model.similarity_between(i, j) == model.similarity_between(j, i)
            assert model.cooccurrence_between(i, j) == model.cooccurrence_between(j, i)


def test_self_similarity_contributes_zero() -> None:
    """sim(i,i) == 0 and no item lists itself as a neighbour."""
    model = oracle_model()
    for i in range(1, 4):
        assert model.similarity_between(i, i) == 0.0
        assert i not in model.neighbors(i)


def test_no_cooccurrence_pair_gives_zero() -> None:
    """Items that never co-occur have zero similarity, in both accessors."""
    model = ItemCF(num_items=4)
    model.fit([[1, 2], [3, 4]])
    assert model.cooccurrence_between(1, 3) == 0
    assert model.similarity_between(1, 3) == 0.0
    assert model.similarity_between(1, 4) == 0.0
    assert model.similarity_between(2, 3) == 0.0
    # the observed pairs are still there
    assert model.cooccurrence_between(1, 2) == 1


def test_never_seen_item_has_no_similarity() -> None:
    """An item absent from training contributes nothing."""
    model = oracle_model()
    assert model.neighbors(99) == {}
    assert model.similarity_between(99, 1) == 0.0


# --------------------------------------------------------------------------- #
# 7-8. Repeated interactions count once
# --------------------------------------------------------------------------- #


def test_repeated_training_item_does_not_increase_support() -> None:
    """A user consuming one item three times still contributes 1 to freq."""
    repeated = ItemCF(num_items=3)
    repeated.fit([[1, 1, 1, 2], [1, 2]])
    once = ItemCF(num_items=3)
    once.fit([[1, 2], [1, 2]])

    assert repeated.item_freq == once.item_freq
    assert repeated.item_freq == {1: 2, 2: 2}


def test_repeated_training_item_does_not_increase_cooccurrence() -> None:
    """Repetition within a user cannot inflate co-occurrence either."""
    repeated = ItemCF(num_items=3)
    repeated.fit([[1, 1, 1, 2, 2]])
    once = ItemCF(num_items=3)
    once.fit([[1, 2]])

    assert repeated.cooccurrence_between(1, 2) == 1
    assert repeated.cooccurrence_between(1, 2) == once.cooccurrence_between(1, 2)
    assert repeated.similarity_between(1, 2) == once.similarity_between(1, 2)


def test_duplicate_item_across_users_still_counts_per_user() -> None:
    """Distinct users each contribute, so support is per-user not per-row."""
    model = ItemCF(num_items=2)
    model.fit([[1, 1], [1, 1], [1, 2]])
    assert model.item_freq == {1: 3, 2: 1}
    assert model.cooccurrence_between(1, 2) == 1


# --------------------------------------------------------------------------- #
# 9-13. Scoring
# --------------------------------------------------------------------------- #


def test_exact_score_for_one_history_item() -> None:
    """score(j) for a single history item is exactly sim(history_item, j)."""
    model = oracle_model()
    scores = model.score([2])
    assert len(scores) == 4
    assert abs(scores[1] - 2 / math.sqrt(6)) < TOL
    assert abs(scores[3] - 0.5) < TOL
    assert scores[2] == 0.0  # self-similarity contributes nothing


def test_exact_summed_score_for_multiple_history_items() -> None:
    """Scores sum contributions from every distinct history item."""
    model = oracle_model()
    scores = model.score([1, 2])
    # item 3 receives sim(1,3) + sim(2,3); item 1 receives sim(2,1); item 2 receives sim(1,2)
    assert abs(scores[3] - (2 / math.sqrt(6) + 0.5)) < TOL
    assert abs(scores[1] - 2 / math.sqrt(6)) < TOL
    assert abs(scores[2] - 2 / math.sqrt(6)) < TOL


def test_repeated_inference_history_item_does_not_double_count() -> None:
    """Repeating a history item must not multiply its contribution."""
    model = oracle_model()
    assert model.score([1, 1, 1, 2]) == model.score([1, 2])
    assert model.score([2, 2]) == model.score([2])


def test_unseen_training_item_produces_zero_similarity() -> None:
    """A catalog item absent from training yields an all-zero row."""
    model = ItemCF(num_items=4)
    model.fit([[1, 2], [1, 2]])
    assert model.neighbors(3) == {}
    assert model.neighbors(4) == {}
    assert model.similarity_between(3, 4) == 0.0
    assert model.score([3]) == [0.0] * 5


def test_catalog_item_unseen_during_fit_receives_valid_zero_score() -> None:
    """Items never seen in training still appear in the vector with score 0.0."""
    model = ItemCF(num_items=5)
    model.fit([[1, 2], [1, 2]])
    scores = model.score([1])
    assert len(scores) == 6
    assert scores[4] == 0.0 and scores[5] == 0.0
    assert all(value == value for value in scores)  # no NaN


def test_score_vector_length_is_num_items_plus_one() -> None:
    """The vector covers the catalog plus the PAD slot."""
    model = ItemCF(num_items=7)
    model.fit([[1, 2]])
    assert len(model.score([1])) == 8
    assert len(model.score([])) == 8


def test_empty_history_scores_all_zero() -> None:
    """An empty history has nothing to aggregate, so every score is 0.0."""
    model = oracle_model()
    scores = model.score([])
    assert scores == [DEFAULT_SCORE] * 4


def test_all_scores_are_finite() -> None:
    """ItemCF never emits NaN or infinity, even for degenerate input."""
    model = ItemCF(num_items=5)
    model.fit([[1, 2], [3, 4], [1, 2, 3, 4], [5]])
    for history in ([], [1], [1, 2, 3, 4, 5], [5], [2, 2, 2]):
        for value in model.score(history):
            assert math.isfinite(value)


# --------------------------------------------------------------------------- #
# 14-16. PAD and invalid ids
# --------------------------------------------------------------------------- #


def test_pad_has_no_itemcf_similarity_contribution() -> None:
    """PAD (0) is not a catalog item: it is never a neighbour and always scores 0."""
    model = oracle_model()
    scores = model.score([1, 2, 3])
    assert scores[config.PAD_ID] == DEFAULT_SCORE

    for i in range(1, 4):
        assert i not in model.neighbors(config.PAD_ID)
        assert model.similarity_between(config.PAD_ID, i) == 0.0
        assert model.cooccurrence_between(config.PAD_ID, i) == 0
    assert config.PAD_ID not in model.item_freq
    assert config.PAD_ID not in model.similarity


def test_pad_in_history_is_rejected() -> None:
    """A history containing PAD is an explicit error, not silently ignored."""
    model = oracle_model()
    for bad in ([0], [1, 0], [0, 1, 2]):
        with pytest.raises(ItemCFError) as excinfo:
            model.score(bad)
        assert "PAD" in str(excinfo.value)


def test_invalid_item_ids_are_rejected() -> None:
    """Out-of-range, negative, non-integer and boolean ids are rejected."""
    model = oracle_model()
    for bad in ([4], [99], [-1], [1, 4], [1.5], ["1"], [True], [None]):
        with pytest.raises(ItemCFError):
            model.score(bad)  # type: ignore[arg-type]


def test_validate_history_helper_contract() -> None:
    """The validation helper accepts a valid history and rejects the rest."""
    assert validate_history([1, 2, 3], 3) == (1, 2, 3)
    assert validate_history([], 3) == ()
    with pytest.raises(ItemCFError):
        validate_history([0], 3)
    with pytest.raises(ItemCFError):
        validate_history([4], 3)


def test_num_items_and_min_cooccurrence_are_validated() -> None:
    """Constructor arguments are validated explicitly."""
    for bad_num_items in (0, -1, True, 2.5, "5"):
        with pytest.raises(ItemCFError):
            ItemCF(bad_num_items)  # type: ignore[arg-type]
    for bad_min in (0, -1, 1.5, "1", True):
        with pytest.raises(ItemCFError):
            ItemCF(5, min_cooccurrence=bad_min)  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# 17. Fit validation / lifecycle
# --------------------------------------------------------------------------- #


def test_fit_with_invalid_histories_is_rejected() -> None:
    """Invalid ids in the fit data are an explicit error naming the user."""
    model = ItemCF(num_items=3)
    with pytest.raises(ItemCFError) as excinfo:
        model.fit([[1, 2], [0, 3]])
    assert "#1" in str(excinfo.value)

    with pytest.raises(ItemCFError):
        model.fit([[1, 2], [1, 9]])


def test_fit_ignores_empty_histories_but_counts_non_empty_users() -> None:
    """Empty train histories contribute no support and are not counted as users."""
    model = ItemCF(num_items=3)
    stats = model.fit([[], [1, 2], []])
    assert stats.num_fit_users == 1
    assert model.item_freq == {1: 1, 2: 1}


def test_scoring_before_fit_raises() -> None:
    """Using an unfitted model is an explicit error."""
    model = ItemCF(num_items=3)
    assert model.is_fitted is False
    with pytest.raises(ItemCFError):
        model.score([1])
    with pytest.raises(ItemCFError):
        model.stats
    with pytest.raises(ItemCFError):
        model.to_dict()


def test_fit_stats_are_reported() -> None:
    """Fit statistics describe what the fit actually used."""
    model = oracle_model()
    stats = model.stats
    assert stats.model == "itemcf_cosine"
    assert stats.num_items == 3
    assert stats.num_fit_users == 3
    assert stats.num_train_interactions == 7  # 3 + 2 + 2 rows across the three users
    assert stats.num_unique_train_items == 3
    assert stats.num_similarity_pairs == 3  # the three undirected pairs
    assert stats.item_training_coverage == 1.0
    assert stats.fit_seconds >= 0.0
    payload = stats.as_dict()
    assert payload["similarity"] == "cosine"
    assert payload["rating_weighting"] is False
    assert payload["self_similarity"] == 0.0


def test_unique_items_helper_is_sorted_and_deduplicated() -> None:
    """The distinct-item helper is deterministic ascending order."""
    assert unique_items([3, 1, 3, 2, 1]) == (1, 2, 3)
    assert unique_items([]) == ()


# --------------------------------------------------------------------------- #
# 18. Determinism
# --------------------------------------------------------------------------- #


def test_fit_is_deterministic_under_input_order_variation() -> None:
    """Shuffling the user order (and item order within users) changes nothing."""
    histories = [[1, 2, 3], [1, 2], [1, 3], [4, 5], [1, 2, 4, 5], [3, 5]]
    forward = ItemCF(num_items=5)
    forward.fit(histories)
    backward = ItemCF(num_items=5)
    backward.fit([list(reversed(h)) for h in reversed(histories)])

    assert forward.item_freq == backward.item_freq
    assert forward.cooccurrence == backward.cooccurrence
    # compare through the symmetric accessor so storage order is irrelevant
    for i in range(1, 6):
        for j in range(1, 6):
            assert forward.similarity_between(i, j) == backward.similarity_between(i, j)
    for history in ([1], [1, 2], [4, 5], [1, 2, 3, 4, 5]):
        assert forward.score(history) == backward.score(history)


def test_similarity_storage_order_is_canonical() -> None:
    """Sparse neighbour lists are emitted in ascending id order."""
    model = ItemCF(num_items=6)
    model.fit([[1, 2, 3, 4, 5, 6], [1, 2, 3, 4, 5, 6]])
    for item_id, neighbours in model.similarity.items():
        assert list(neighbours) == sorted(neighbours)
    for item_id, counts in model.cooccurrence.items():
        assert list(counts) == sorted(counts)


def test_serialisation_is_deterministic_and_sorted() -> None:
    """to_dict() emits ascending ids, so the payload is byte-stable."""
    model = ItemCF(num_items=5)
    model.fit([[1, 2, 3], [3, 4, 5], [1, 3]])
    payload = model.to_dict()
    assert [int(k) for k in payload["item_freq"]] == sorted(int(k) for k in payload["item_freq"])
    for key, neighbours in payload["similarity"].items():
        assert [int(x) for x in neighbours] == sorted(int(x) for x in neighbours)

    # ``fit_seconds`` is wall-clock timing and intentionally variable, exactly like
    # the execution-time fields Milestone 2A excluded from its reproducibility
    # comparison; everything else must match byte-for-byte.
    other = ItemCF(num_items=5)
    other.fit([[5, 4, 3], [3, 2, 1], [3, 1]])
    other_payload = other.to_dict()
    payload["stats"].pop("fit_seconds")
    other_payload["stats"].pop("fit_seconds")
    assert other_payload == payload


def test_similarity_pair_count_helper() -> None:
    """The directed-pair count matches the sparse neighbour total.

    Each undirected pair is stored under both endpoints (so scoring is independent
    of which endpoint is iterated first), hence 3 pairs -> 6 directed entries.
    """
    model = oracle_model()
    assert similarity_pair_count(model) == sum(len(n) for n in model.similarity.values())
    assert similarity_pair_count(model) == 6
    assert model.stats.num_similarity_pairs == 3          # undirected pairs
    assert model.cooccurrence[2][1] == model.cooccurrence[1][2] == 2


# --------------------------------------------------------------------------- #
# 19-21. Leakage: fit data must exclude targets and ineligible users
# --------------------------------------------------------------------------- #


def test_validation_and_test_targets_do_not_leak_into_fit_stats() -> None:
    """Fitting from train histories ignores both held-out targets.

    The cohort is built so the two targets (90, 91) and an item that *only* appears
    as a target (92) must never appear in frequencies or similarities.
    """
    sequences = {
        "u1": [1, 2, 90, 91],
        "u2": [1, 2, 91, 92],
        "u3": [1, 3, 92, 90],
    }
    cases, report = build_cohort(sequences, num_items=92)
    model, stats = fit_from_cohort(cases, report.catalog_size)

    train_items = {1, 2, 3}
    assert set(model.item_freq) == train_items
    for target in (90, 91, 92):
        assert target not in model.item_freq
        assert target not in model.similarity
        for i in train_items:
            assert model.similarity_between(target, i) == 0.0

    # and the fit input really was only the train histories
    assert stats.num_train_interactions == sum(len(c.train_history) for c in cases)
    assert all(len(c.train_history) == 2 for c in cases)


def test_target_only_items_absent_from_fit_but_scoreable() -> None:
    """A target unseen during fit still yields a valid (zero) score."""
    sequences = {"u1": [1, 2, 90, 91]}
    cases, report = build_cohort(sequences, num_items=91)
    model, _ = fit_from_cohort(cases, report.catalog_size)

    scores = model.score(cases[0].test_history)
    assert len(scores) == report.catalog_size + 1
    assert scores[91] == 0.0
    assert math.isfinite(scores[91])


def test_ineligible_users_do_not_enter_fit_data() -> None:
    """Users excluded by the Milestone 2A cohort (len < 3) contribute nothing."""
    sequences = {
        "eligible": [1, 2, 3],
        "too_short_a": [7, 8],
        "too_short_b": [9],
    }
    cases, report = build_cohort(sequences, num_items=10)
    model, stats = fit_from_cohort(cases, report.catalog_size)

    assert report.num_users_excluded == 2
    assert stats.num_fit_users == 1
    assert set(model.item_freq) == {1}
    # item 7 never appears, so it must not be in the fit statistics at all
    assert 7 not in model.item_freq and 8 not in model.item_freq and 9 not in model.item_freq


def test_fit_from_cohort_uses_only_train_history_not_targets() -> None:
    """Directly verify the fit input: train history only, per the contract."""
    cases = [
        EvaluationCase(
            user_id="u1",
            user_int_id=1,
            train_history=(1, 2),
            validation_target=80,
            test_target=81,
            sequence_length=4,
        ),
        EvaluationCase(
            user_id="u2",
            user_int_id=2,
            train_history=(1, 3),
            validation_target=82,
            test_target=83,
            sequence_length=4,
        ),
    ]
    model, stats = fit_from_cohort(cases, num_items=100)

    assert model.item_freq == {1: 2, 2: 1, 3: 1}
    assert stats.num_train_interactions == 4
    for held_out in (80, 81, 82, 83):
        assert held_out not in model.item_freq


# --------------------------------------------------------------------------- #
# 22-23. Masking ownership stays with the evaluator
# --------------------------------------------------------------------------- #


def test_scorer_does_not_mask_seen_items() -> None:
    """ItemCF returns scores for seen items too; masking is the evaluator's job."""
    model = oracle_model()
    scores = model.score([1, 2])
    # item 1 and 2 are in the history yet still carry scores: masking them out is
    # the evaluator's job, not the scorer's.  (Scores come from the *other* history
    # item's similarity row, because self-similarity is zero.)
    assert scores[1] > 0.0 and scores[2] > 0.0
    # PAD is present in the vector as well (unmasked)
    assert scores[config.PAD_ID] == DEFAULT_SCORE


def test_repeated_target_can_still_receive_a_raw_score() -> None:
    """A target that also appears in the history is scored, not suppressed."""
    model = ItemCF(num_items=3)
    model.fit([[1, 2], [2, 3], [1, 3]])
    history = [1, 2]
    target = 2  # repeated target: it is in the history *and* the prediction target
    scores = model.score(history)
    assert scores[target] > 0.0, "scorer must not suppress a repeated target"

    evaluator = FullRankingEvaluator(num_items=3, k_values=(5,))
    rank = evaluator.rank(scores, target, history=history)
    # Item 3 is similar to both history items, so it outranks them; items 1 and 2 tie
    # and the frozen tie rule puts the lower id first.  The point of the test is that
    # the repeated target (2) is still *ranked* rather than suppressed.
    assert scores[3] > scores[2] > 0.0
    assert rank == 2
    assert evaluator.rank(scores, 1, history=history) == 2


def test_scorer_output_feeds_the_evaluator_unchanged() -> None:
    """The scorer's vector satisfies the evaluator's score contract exactly."""
    model = oracle_model()
    evaluator = FullRankingEvaluator(num_items=3, k_values=DEFAULT_K_VALUES)
    scores = model.score([1, 2])
    evaluator.validate_scores(scores)  # no exception -> contract satisfied
    assert len(scores) == evaluator.expected_score_length() == 4


# --------------------------------------------------------------------------- #
# 24. Integration with the unified evaluator
# --------------------------------------------------------------------------- #


def test_itemcf_plugs_into_the_unified_evaluator() -> None:
    """ItemCF is evaluated exclusively through the Milestone 2A evaluator."""
    sequences = {
        "u1": [1, 2, 3],
        "u2": [1, 2, 3],
        "u3": [1, 4, 5],
        "u4": [1, 2, 4],
        "u5": [2, 3, 4],
    }
    cases, report = build_cohort(sequences, num_items=5)
    assert len(cases) == 5

    model, stats = fit_from_cohort(cases, report.catalog_size)
    evaluator = FullRankingEvaluator(num_items=report.catalog_size, k_values=(5, 10, 20))
    score_fn = make_scorer(model)

    for mode in ("validation", "test"):
        outcome = evaluator.evaluate(cases, score_fn, mode=mode)
        assert outcome.report.num_cases == len(cases)
        for k in (5, 10, 20):
            for name in ("HR", "Recall", "NDCG"):
                value = outcome.report.metrics[name][k]
                assert math.isfinite(value)
                assert 0.0 <= value <= 1.0
        # HR and Recall must coincide under the single-positive protocol
        assert all(outcome.hr_recall_agree.values())

    # The fit really used only the eligible users' train histories.
    assert stats.num_fit_users == 5
    assert stats.num_train_interactions == sum(len(c.train_history) for c in cases)


def test_itemcf_scorer_ignores_the_target_argument() -> None:
    """The scorer scores the whole catalog regardless of which target is passed.

    This is what proves the model is not doing its own candidate selection: the same
    history yields the same vector for every target the evaluator may ask about.
    """
    model = oracle_model()
    score_fn = make_scorer(model)
    base = score_fn((1, 2), 1)
    for target in (1, 2, 3):
        assert score_fn((1, 2), target) == base
    evaluator = FullRankingEvaluator(num_items=3)
    assert evaluator.rank(base, 3, history=(1, 2)) == 1


def test_deterministic_end_to_end_report() -> None:
    """Identical training data produces identical reports across model instances."""
    sequences = {"u1": [1, 2, 3], "u2": [1, 2, 4], "u3": [2, 3, 4], "u4": [1, 3, 4]}
    cases, report = build_cohort(sequences, num_items=4)

    outcomes = []
    for _ in range(2):
        model, _ = fit_from_cohort(cases, report.catalog_size)
        evaluator = FullRankingEvaluator(num_items=report.catalog_size, k_values=(5, 10, 20))
        outcomes.append(evaluator.evaluate(cases, make_scorer(model), mode="test").report.as_dict())

    assert outcomes[0] == outcomes[1]


def test_min_cooccurrence_drops_pairs_from_both_maps() -> None:
    """A raised co-occurrence floor removes the pair from similarity and co-occurrence."""
    model = ItemCF(num_items=4, min_cooccurrence=2)
    model.fit([[1, 2], [1, 2], [1, 3]])
    assert model.cooccurrence_between(1, 2) == 2
    assert model.cooccurrence_between(1, 3) == 0       # dropped from both views
    assert model.similarity_between(1, 3) == 0.0
    assert model.similarity_between(1, 2) > 0.0
    assert model.stats.num_similarity_pairs == 1

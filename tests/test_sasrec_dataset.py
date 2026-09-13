"""Tests for the SASRec training dataset and inference encoder (Milestone 3, Part A).

All fixtures are tiny and hand-computable.  The critical property under test is that
nothing downstream of ``train_history`` — validation target, test target — ever
reaches the training dataset or the negative sampler.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from recommendation.datasets.sasrec import (  # noqa: E402
    MIN_TRAIN_HISTORY_FOR_TRANSITION,
    PAD_ID,
    SASRecDataError,
    SASRecDatasetConfig,
    build_arrays,
    build_dataset,
    build_sample,
    cohort_structure_digest,
    encode_batch,
    encode_inference_history,
    sample_negative,
    validation_history,
)
from recommendation.evaluation import EvaluationCase, build_cohort  # noqa: E402

CFG = SASRecDatasetConfig(max_seq_len=5, seed=7)


def case(user_int_id: int, history: tuple[int, ...], val: int, test: int) -> EvaluationCase:
    """Build an EvaluationCase with explicit held-out targets."""
    return EvaluationCase(
        user_id=f"u{user_int_id}",
        user_int_id=user_int_id,
        train_history=history,
        validation_target=val,
        test_target=test,
        sequence_length=len(history) + 2,
    )


# --------------------------------------------------------------------------- #
# 1-7. Sequence construction
# --------------------------------------------------------------------------- #


def test_length_two_history_gives_exactly_one_transition() -> None:
    """A 2-item train history has exactly one next-item transition."""
    inputs, positives, num_valid = build_arrays([7, 9], max_seq_len=5)
    assert inputs == (PAD_ID, PAD_ID, PAD_ID, PAD_ID, 7)
    assert positives == (PAD_ID, PAD_ID, PAD_ID, PAD_ID, 9)
    assert num_valid == 1


def test_longer_history_shift_alignment_is_exact() -> None:
    """positive_ids[t] is the item that followed input_ids[t]."""
    history = [1, 2, 3, 4]
    inputs, positives, num_valid = build_arrays(history, max_seq_len=5)

    assert inputs == (0, 0, 1, 2, 3)
    assert positives == (0, 0, 2, 3, 4)
    assert num_valid == 3

    non_pad = [(i, p) for i, p in zip(inputs, positives) if p != PAD_ID]
    assert non_pad == [(1, 2), (2, 3), (3, 4)]
    # every aligned input/positive pair is adjacent in the source history
    positions = {item: idx for idx, item in enumerate(history)}
    for source, target in non_pad:
        assert positions[target] == positions[source] + 1


def test_left_padding_with_pad_zero() -> None:
    """Padding is PAD 0 and sits on the left.

    A history of m items yields m-1 transitions, so the arrays hold m-1 real slots
    (inputs are the first m-1 items, positives the last m-1) padded on the left.
    """
    history = [4, 5, 6]
    inputs, positives, num_valid = build_arrays(history, max_seq_len=6)
    assert inputs == (0, 0, 0, 0, 4, 5)
    assert positives == (0, 0, 0, 0, 5, 6)
    assert num_valid == 2
    # padding only ever on the left; no PAD among the real positions
    first_real = next(idx for idx, v in enumerate(inputs) if v != PAD_ID)
    assert all(v == PAD_ID for v in inputs[:first_real])
    assert all(v != PAD_ID for v in inputs[first_real:])
    assert all(v != PAD_ID for v in positives[first_real:])


def test_recent_history_truncation() -> None:
    """Truncation keeps the most recent transitions."""
    inputs, positives, num_valid = build_arrays([1, 2, 3, 4, 5, 6], max_seq_len=3)
    assert num_valid == 3
    # the last three transitions of [1..6] are (3->4), (4->5), (5->6)
    assert inputs == (3, 4, 5)
    assert positives == (4, 5, 6)


def test_input_positive_alignment_after_truncation() -> None:
    """Alignment survives truncation, with padding still only on the left."""
    history = list(range(1, 13))
    inputs, positives, num_valid = build_arrays(history, max_seq_len=7)

    assert num_valid == 7
    assert inputs == (5, 6, 7, 8, 9, 10, 11)
    assert positives == (6, 7, 8, 9, 10, 11, 12)
    for source, target in zip(inputs, positives):
        assert target == source + 1

    # ...and with padding present
    inputs2, positives2, num_valid2 = build_arrays([1, 2, 3, 4], max_seq_len=6)
    assert inputs2 == (0, 0, 0, 1, 2, 3)
    assert positives2 == (0, 0, 0, 2, 3, 4)
    assert num_valid2 == 3


def test_pad_appears_only_in_padded_positions() -> None:
    """PAD never appears among real inputs and only marks padding in positives."""
    inputs, positives, num_valid = build_arrays([2, 4, 6], max_seq_len=5)
    real_inputs = [v for v in inputs if v != PAD_ID]
    assert real_inputs == [2, 4]          # m-1 = 2 real input slots
    assert real_inputs.count(PAD_ID) == 0
    # PAD positives are exactly the padded prefix
    assert [v == PAD_ID for v in positives] == [True, True, True, False, False]
    assert sum(1 for v in positives if v != PAD_ID) == num_valid == 2


def test_chronological_order_is_preserved() -> None:
    """Order is never permuted: arrays follow the history's own order."""
    history = [9, 3, 7, 1]
    inputs, positives, _ = build_arrays(history, max_seq_len=5)
    assert [v for v in inputs if v != PAD_ID] == history[:-1]
    assert [v for v in positives if v != PAD_ID] == history[1:]


def test_build_arrays_rejects_history_shorter_than_two() -> None:
    """A history with no transition cannot yield arrays."""
    for history in ([], [5]):
        with pytest.raises(SASRecDataError):
            build_arrays(history, max_seq_len=5)


def test_build_arrays_rejects_bad_max_seq_len() -> None:
    """max_seq_len must be a positive int."""
    for bad in (0, -1, 1.5, True):
        with pytest.raises(SASRecDataError):
            build_arrays([1, 2], max_seq_len=bad)  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# 8. Source immutability
# --------------------------------------------------------------------------- #


def test_source_history_is_not_mutated() -> None:
    """Building arrays and samples leaves the source sequence untouched."""
    history = [5, 4, 3, 2, 1]
    snapshot = list(history)

    build_arrays(history, max_seq_len=3)
    build_sample(1, history, num_items=10, config_=CFG)

    assert history == snapshot
    assert isinstance(build_arrays(history, 3)[0], tuple)


def test_build_dataset_does_not_mutate_cases() -> None:
    """The cohort's cases and their histories are never modified."""
    cases = [case(1, (1, 2, 3), 90, 91), case(2, (4, 5), 92, 93)]
    snapshot = [(c.train_history, c.validation_target, c.test_target) for c in cases]

    build_dataset(cases, num_items=100, config_=CFG)

    assert [(c.train_history, c.validation_target, c.test_target) for c in cases] == snapshot


# --------------------------------------------------------------------------- #
# 9-10. Trainable-user policy
# --------------------------------------------------------------------------- #


def test_one_train_item_gives_zero_transitions() -> None:
    """len(train_history) == 1 produces no gradient-producing sample."""
    cases = [case(1, (5,), 90, 91)]
    dataset = build_dataset(cases, num_items=100, config_=CFG)

    assert len(dataset) == 0
    assert dataset.stats.trainable_users == 0
    assert dataset.stats.users_with_zero_transitions == 1
    assert dataset.stats.users_with_one_train_item == 1
    assert dataset.stats.raw_next_item_transitions == 0
    assert dataset.total_valid_positions == 0


def test_one_train_item_user_still_in_evaluation_cohort() -> None:
    """Our exclusion is dataset-level only; the evaluation cohort is untouched."""
    sequences = {"a": [1, 90, 91], "b": [1, 2, 92, 93]}  # 'a' has train_history len 1
    cases, report = build_cohort(sequences, num_items=100)

    assert report.num_users_eligible == 2          # both remain eligible
    assert {c.user_id for c in cases} == {"a", "b"}

    dataset = build_dataset(cases, num_items=100, config_=CFG)
    assert dataset.stats.evaluation_users == 2     # cohort size unchanged
    assert dataset.stats.trainable_users == 1      # only 'b' is trainable
    assert dataset.stats.users_with_zero_transitions == 1


def test_trainable_user_policy_boundary() -> None:
    """Length 2 is the minimum trainable length."""
    assert MIN_TRAIN_HISTORY_FOR_TRANSITION == 2
    cases = [case(1, (1,), 90, 91), case(2, (1, 2), 92, 93), case(3, (1, 2, 3), 94, 95)]
    dataset = build_dataset(cases, num_items=100, config_=CFG)
    assert [s.user_int_id for s in dataset] == [2, 3]


# --------------------------------------------------------------------------- #
# 11-17. Negative sampling
# --------------------------------------------------------------------------- #


def test_negatives_are_in_catalog() -> None:
    """Every sampled negative is a real catalog item."""
    sample = build_sample(3, [1, 2, 3, 4], num_items=50, config_=CFG)
    for positive, negative in zip(sample.positive_ids, sample.negative_ids):
        if positive == PAD_ID:
            continue
        assert 1 <= negative <= 50


def test_negatives_are_never_pad() -> None:
    """PAD is never sampled as a negative."""
    sample = build_sample(3, [1, 2, 3], num_items=20, config_=CFG)
    for positive, negative in zip(sample.positive_ids, sample.negative_ids):
        if positive != PAD_ID:
            assert negative != PAD_ID


def test_negatives_not_in_train_history_exclusion_set() -> None:
    """A negative never comes from the user's train history."""
    history = [4, 7, 11, 13, 17]
    sample = build_sample(5, history, num_items=40, config_=CFG)
    exclusion = set(history)
    for positive, negative in zip(sample.positive_ids, sample.negative_ids):
        if positive != PAD_ID:
            assert negative not in exclusion


def test_negative_sampling_is_deterministic() -> None:
    """Identical seed and sample identity give identical negatives."""
    first = build_sample(9, [1, 2, 3, 4, 5], num_items=30, config_=CFG)
    second = build_sample(9, [1, 2, 3, 4, 5], num_items=30, config_=CFG)
    assert first.negative_ids == second.negative_ids

    left = sample_negative(1, 0, frozenset({2}), 30, CFG)
    right = sample_negative(1, 0, frozenset({2}), 30, CFG)
    assert left == right


def test_negative_sampling_varies_by_epoch_and_position() -> None:
    """Different epochs or positions resample, while structure stays identical."""
    base = SASRecDatasetConfig(max_seq_len=5, seed=7, epoch=0)
    bumped = SASRecDatasetConfig(max_seq_len=5, seed=7, epoch=1)

    sample_a = build_sample(4, [1, 2, 3, 4, 5], num_items=200, config_=base)
    sample_b = build_sample(4, [1, 2, 3, 4, 5], num_items=200, config_=bumped)

    assert sample_a.input_ids == sample_b.input_ids
    assert sample_a.positive_ids == sample_b.positive_ids
    assert sample_a.num_valid_positions == sample_b.num_valid_positions
    assert sample_a.negative_ids != sample_b.negative_ids  # resampled per epoch

    # different positions within one sample differ (4 valid positions, 200 items)
    negatives = [n for p, n in zip(sample_a.positive_ids, sample_a.negative_ids) if p != PAD_ID]
    assert len(set(negatives)) > 1


def test_negative_sampler_ignores_validation_target() -> None:
    """The validation target does not influence negative sampling."""
    history = (1, 2, 3)
    kwargs = dict(user_int_id=1, position=0, exclusion=frozenset(history), num_items=40)

    first = sample_negative(**kwargs, config_=SASRecDatasetConfig(seed=1))
    second = sample_negative(**kwargs, config_=SASRecDatasetConfig(seed=1))
    assert first == second

    # build two full datasets whose *only* difference is the validation target
    left = build_dataset([case(1, history, 90, 91)], num_items=40,
                         config_=SASRecDatasetConfig(max_seq_len=5, seed=1))
    right = build_dataset([case(1, history, 77, 91)], num_items=40,
                          config_=SASRecDatasetConfig(max_seq_len=5, seed=1))
    assert left.samples[0].negative_ids == right.samples[0].negative_ids


def test_negative_sampler_ignores_test_target() -> None:
    """The test target does not influence negative sampling."""
    history = (1, 2, 3)
    left = build_dataset([case(1, history, 90, 91)], num_items=40,
                         config_=SASRecDatasetConfig(max_seq_len=5, seed=1))
    right = build_dataset([case(1, history, 90, 55)], num_items=40,
                          config_=SASRecDatasetConfig(max_seq_len=5, seed=1))
    assert left.samples[0].negative_ids == right.samples[0].negative_ids

    # and a target-only item can legitimately BE a negative, because at training
    # time it is an ordinary unseen item (excluding it would leak the future)
    targets = {90, 91}
    for negative in left.samples[0].negative_ids:
        if negative != PAD_ID:
            assert negative in range(1, 41)  # no assertion about targets


def test_dataset_digest_unchanged_when_only_targets_change() -> None:
    """Changing targets cannot change the dataset at all."""
    history = (2, 4, 6, 8, 10)
    left = build_dataset([case(1, history, 90, 91)], num_items=60, config_=CFG)
    right = build_dataset([case(1, history, 11, 12)], num_items=60, config_=CFG)

    assert left.digest() == right.digest()
    assert left.samples[0].as_dict() == right.samples[0].as_dict()


def test_cohort_structure_digest_ignores_targets() -> None:
    """The structural digest hashes train histories only."""
    left = [case(1, (1, 2, 3), 90, 91), case(2, (4, 5), 92, 93)]
    right = [case(1, (1, 2, 3), 11, 12), case(2, (4, 5), 13, 14)]
    assert cohort_structure_digest(left) == cohort_structure_digest(right)

    different = [case(1, (1, 2, 9), 90, 91), case(2, (4, 5), 92, 93)]
    assert cohort_structure_digest(left) != cohort_structure_digest(different)


def test_no_valid_negative_fails_explicitly() -> None:
    """When the whole catalog is the exclusion set, sampling fails loudly."""
    with pytest.raises(SASRecDataError) as excinfo:
        sample_negative(1, 0, frozenset({1, 2, 3}), 3, CFG)
    assert "no valid negative" in str(excinfo.value)


def test_negative_sampler_terminates_when_pool_is_tiny() -> None:
    """A single-item pool still resolves through the deterministic fallback."""
    for position in range(20):
        assert sample_negative(1, position, frozenset(range(1, 40)), 40, CFG) == 40


def test_negative_sampler_rejects_out_of_catalog_exclusion() -> None:
    """An exclusion set containing invalid ids is rejected."""
    with pytest.raises(SASRecDataError):
        sample_negative(1, 0, frozenset({0}), 10, CFG)
    with pytest.raises(SASRecDataError):
        sample_negative(1, 0, frozenset({99}), 10, CFG)


# --------------------------------------------------------------------------- #
# 18-22. Inference encoding
# --------------------------------------------------------------------------- #


def test_validation_history_encoding() -> None:
    """Validation history is the train history, left-padded."""
    case_ = case(1, (3, 4), 90, 91)
    history = validation_history(case_)
    assert history == (3, 4)

    encoded = encode_inference_history(history, max_seq_len=5, num_items=100)
    assert encoded == (0, 0, 0, 3, 4)
    assert 90 not in encoded and 91 not in encoded  # no target leakage


def test_test_history_encoding() -> None:
    """Test history is train history + validation target (never the test target)."""
    case_ = case(1, (3, 4), 90, 91)
    history = history_for_test(case_)
    assert history == (3, 4, 90)

    encoded = encode_inference_history(history, max_seq_len=5, num_items=100)
    assert encoded == (0, 0, 3, 4, 90)
    assert 91 not in encoded  # the test target is never appended


def test_inference_keeps_most_recent_items() -> None:
    """Only the newest max_seq_len items are kept, in chronological order."""
    encoded = encode_inference_history([1, 2, 3, 4, 5, 6, 7], max_seq_len=4, num_items=10)
    assert encoded == (4, 5, 6, 7)


def test_inference_preserves_chronological_order() -> None:
    """Encoding never permutes the history."""
    history = [9, 2, 8, 1, 7]
    encoded = encode_inference_history(history, max_seq_len=8, num_items=10)
    assert [v for v in encoded if v != PAD_ID] == history


def test_empty_inference_history_is_rejected() -> None:
    """An empty history cannot be encoded."""
    with pytest.raises(SASRecDataError):
        encode_inference_history([], max_seq_len=5, num_items=10)


def test_invalid_ids_are_rejected_in_inference() -> None:
    """PAD, out-of-range, negative and non-integer ids are rejected."""
    for bad in ([0], [1, 0], [11], [-1], [1.5], ["1"], [True]):
        with pytest.raises(SASRecDataError):
            encode_inference_history(bad, max_seq_len=5, num_items=10)  # type: ignore[arg-type]


def test_inference_encoding_boundaries() -> None:
    """A single-item history is allowed; max_seq_len must be positive."""
    assert encode_inference_history([5], max_seq_len=3, num_items=10) == (0, 0, 5)
    with pytest.raises(SASRecDataError):
        encode_inference_history([5], max_seq_len=0, num_items=10)


def test_encode_batch_preserves_order() -> None:
    """Batch encoding maps histories in input order."""
    batch = encode_batch([[1], [2, 3]], max_seq_len=3, num_items=10)
    assert batch == [(0, 0, 1), (0, 2, 3)]


# --------------------------------------------------------------------------- #
# 23. No target leakage into training samples
# --------------------------------------------------------------------------- #


def test_no_target_leakage_into_training_samples() -> None:
    """Target-only items never enter inputs or positives, and targets are invisible."""
    cases = [
        case(1, (1, 2, 3), 900, 901),
        case(2, (4, 5, 6, 7), 902, 903),
    ]
    dataset = build_dataset(cases, num_items=1000, config_=CFG)

    target_only = {900, 901, 902, 903}
    for sample in dataset:
        assert not (set(sample.input_ids) & target_only)
        assert not (set(sample.positive_ids) & target_only)

    # validating the fit-input contract: raw interactions equal the train histories
    assert dataset.stats.raw_train_interactions == 3 + 4
    assert dataset.stats.raw_next_item_transitions == (3 - 1) + (4 - 1)
    assert sum(1 for _ in dataset) == 2


def test_dataset_reads_only_train_history() -> None:
    """A case whose *other* fields are nonsense still produces the same dataset.

    ``train_history`` is the only field the builder may consult, so mutating the
    targets must not affect the output at all (compared above); here we prove the
    reverse direction: identical train histories with wildly different targets give
    an identical structural digest and identical samples.
    """
    base = [case(1, (1, 2), 5, 6), case(2, (1, 2), 7, 8)]
    dataset = build_dataset(base, num_items=50, config_=CFG)
    assert dataset.stats.raw_train_interactions == 4

    other = [case(1, (1, 2), 30, 31), case(2, (1, 2), 32, 33)]
    assert build_dataset(other, num_items=50, config_=CFG).digest() == dataset.digest()


def test_dataset_statistics_report_all_required_fields() -> None:
    """The statistics block carries every quantity the milestone asks for."""
    cases = [case(1, (1,), 90, 91), case(2, (1, 2), 92, 93), case(3, (1, 2, 3, 4), 94, 95)]
    dataset = build_dataset(cases, num_items=100, config_=SASRecDatasetConfig(max_seq_len=2, seed=3))
    stats = dataset.stats.as_dict()

    assert stats["evaluation_users"] == 3
    assert stats["trainable_users"] == 2
    assert stats["users_with_zero_transitions"] == 1
    assert stats["users_with_one_train_item"] == 1
    assert stats["raw_train_interactions"] == 1 + 2 + 4
    assert stats["raw_next_item_transitions"] == 0 + 1 + 3
    assert stats["max_seq_len"] == 2
    assert stats["effective_transitions"] == 1 + 2   # user 3 truncated to 2
    assert stats["users_truncated"] == 1
    assert 0.0 <= stats["transition_retention"] <= 1.0
    assert 0.0 <= stats["interaction_retention"] <= 1.0


def test_dataset_order_is_deterministic() -> None:
    """Samples follow ascending user_int_id regardless of input order."""
    cases = [case(3, (1, 2), 90, 91), case(1, (1, 2), 90, 91), case(2, (1, 2), 90, 91)]
    forward = build_dataset(cases, num_items=50, config_=CFG)
    backward = build_dataset(list(reversed(cases)), num_items=50, config_=CFG)

    assert [s.user_int_id for s in forward] == [1, 2, 3]
    assert forward.digest() == backward.digest()


def test_valid_mask_matches_positions() -> None:
    """The sample's valid mask and count agree with the arrays."""
    sample = build_sample(1, [1, 2, 3], num_items=20, config_=CFG)
    assert sample.valid_mask == tuple(p != PAD_ID for p in sample.positive_ids)
    assert sum(sample.valid_mask) == sample.num_valid_positions == 2


def test_dataset_config_is_validated() -> None:
    """Bad configuration values are rejected at construction."""
    for kwargs in ({"max_seq_len": 0}, {"negative_retry_limit": 0}, {"seed": 1.5}, {"epoch": "0"}):
        with pytest.raises(SASRecDataError):
            SASRecDatasetConfig(**kwargs)  # type: ignore[arg-type]


def test_dataset_rejects_invalid_ids_in_cohort() -> None:
    """A cohort carrying PAD or out-of-catalog ids is rejected."""
    with pytest.raises(SASRecDataError):
        build_dataset([case(1, (1, 0), 90, 91)], num_items=50, config_=CFG)
    with pytest.raises(SASRecDataError):
        build_dataset([case(1, (1, 99), 90, 91)], num_items=50, config_=CFG)


def history_for_test(case: EvaluationCase) -> tuple[int, ...]:
    """Test-mode inference history: train history plus the validation interaction.

    Kept as a local helper (not imported from the package) so the test asserts the
    *contract* independently: the test target must never be appended, and the
    validation interaction legitimately may be.
    """
    return tuple(case.train_history) + (case.validation_target,)


# --------------------------------------------------------------------------- #
# Audit: negative sampling must be blind to future targets
# --------------------------------------------------------------------------- #


def test_changing_every_target_leaves_the_whole_dataset_identical() -> None:
    """Changed-target audit: future targets must not affect ANY training content.

    Two case sets share user ids and train histories but have completely different
    validation/test targets.  The strongest available assertion is a full dataset
    digest match, which also proves negative sampling is unchanged - not merely that
    inputs and positives match.
    """
    histories = {1: (1, 2, 3, 4), 2: (2, 3, 4, 5), 3: (3, 4, 5, 6), 4: (1, 3, 5, 7)}
    config = SASRecDatasetConfig(max_seq_len=5, seed=17, epoch=2)
    num_items = 30

    left = [case(uid, hist, 91, 92) for uid, hist in histories.items()]
    right = [case(uid, hist, 77, 78) for uid, hist in histories.items()]

    dataset_a = build_dataset(left, num_items=num_items, config_=config)
    dataset_b = build_dataset(right, num_items=num_items, config_=config)

    # whole-dataset digest equality is the strongest form of the guarantee
    assert dataset_a.digest() == dataset_b.digest()
    assert dataset_a.stats.as_dict() == dataset_b.stats.as_dict()
    assert dataset_a.user_int_ids == dataset_b.user_int_ids

    for sample_a, sample_b in zip(dataset_a.samples, dataset_b.samples):
        assert sample_a.user_int_id == sample_b.user_int_id
        assert sample_a.input_ids == sample_b.input_ids
        assert sample_a.positive_ids == sample_b.positive_ids
        assert sample_a.negative_ids == sample_b.negative_ids
        assert sample_a.num_valid_positions == sample_b.num_valid_positions


def test_future_target_may_legally_be_sampled_as_a_negative() -> None:
    """A target item absent from train_history is a legal training negative.

    Excluding it merely because the sampler "knows" it becomes a future target would
    itself be temporal leakage, and the sampler has no access to that information
    anyway: it only receives the train-history exclusion set.

    A bounded deterministic search finds a (seed, epoch) under which the sampler
    genuinely selects the target-only item; the rest of the dataset is then asserted
    to ignore the targets entirely.
    """
    num_items = 12
    history = (1, 2, 3)
    future_target = 11                       # never in the history
    assert future_target not in history

    cases = [case(1, history, 90, future_target)]
    exclusion = frozenset(history)

    found: tuple[int, int, int] | None = None
    for seed in range(0, 200):
        for epoch in (0, 1):
            candidate = sample_negative(
                1, 0, exclusion, num_items, SASRecDatasetConfig(seed=seed, epoch=epoch)
            )
            if candidate == future_target:
                found = (seed, epoch, candidate)
                break
        if found:
            break

    assert found is not None, "expected some deterministic seed to select the target item"
    seed, epoch, candidate = found
    assert candidate == future_target
    # the sampled negative is a legal item: in catalog, non-PAD, outside the history
    assert 1 <= candidate <= num_items
    assert candidate != PAD_ID
    assert candidate not in exclusion

    # the presence of that same item as a *target* changes nothing anywhere
    dataset = build_dataset(
        cases, num_items=num_items, config_=SASRecDatasetConfig(max_seq_len=5, seed=seed, epoch=epoch)
    )
    sample = dataset.samples[0]
    assert future_target in sample.negative_ids          # genuinely sampled
    assert future_target not in sample.input_ids
    assert future_target not in sample.positive_ids

    other_targets = build_dataset(
        [case(1, history, 91, 92)],
        num_items=num_items,
        config_=SASRecDatasetConfig(max_seq_len=5, seed=seed, epoch=epoch),
    )
    assert other_targets.digest() == dataset.digest()


def test_train_history_items_are_never_sampled_as_negatives() -> None:
    """The real exclusion rule stays strict, including repeated interactions.

    Relaxing the future-target rule must not weaken this: every history item is
    forbidden as a negative, and repetition in the history does not change that.
    """
    histories = [
        (4, 5, 6, 7),                 # plain history
        (3, 3, 3, 8, 8, 9),           # repeated interactions
        (2, 2, 10, 11, 11, 12),       # repeats at both ends
        (1, 13, 13, 14, 14, 14),      # heavy repetition
    ]
    num_items = 25

    for index, history in enumerate(histories, start=1):
        dataset = build_dataset(
            [case(index, history, 90, 91)],
            num_items=num_items,
            config_=SASRecDatasetConfig(max_seq_len=8, seed=5, epoch=0),
        )
        sample = dataset.samples[0]
        forbidden = set(history)
        for positive, negative in zip(sample.positive_ids, sample.negative_ids):
            if positive == PAD_ID:
                assert negative == PAD_ID
                continue
            assert negative not in forbidden, (
                f"history item {negative} was sampled as a negative for history {history}"
            )

        # ...and across many epochs/seeds, still never a history item
        for seed in (0, 1, 2, 13):
            for epoch in (0, 1, 5):
                config = SASRecDatasetConfig(max_seq_len=8, seed=seed, epoch=epoch)
                rebuilt = build_dataset([case(index, history, 90, 91)], num_items, config)
                for positive, negative in zip(
                    rebuilt.samples[0].positive_ids, rebuilt.samples[0].negative_ids
                ):
                    if positive != PAD_ID:
                        assert negative not in forbidden

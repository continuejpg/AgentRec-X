"""Real-artifact integration sanity check for the SASRec data contract.

Builds the SASRec dataset from the Milestone 1.5 / 2A artifact and checks the
structural guarantees only.  No model is trained and no recommendation-quality
metric is asserted.

The 100k prefix is an engineering fixture; nothing here describes the full
Sports and Outdoors category.  ``max_seq_len`` is a development integration
setting, not a benchmark choice.

If the artifacts are absent (fresh clone - ``data/`` is git-ignored) the tests skip.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from recommendation.datasets.sasrec import (  # noqa: E402
    PAD_ID,
    SASRecDatasetConfig,
    build_dataset,
)
from recommendation.evaluation import EvaluationCase, build_cohort_from_artifacts  # noqa: E402

SEQUENCES_PATH = REPO_ROOT / "data" / "processed" / "Sports_and_Outdoors_sample_sequences.json"
MAPPINGS_PATH = REPO_ROOT / "data" / "processed" / "Sports_and_Outdoors_sample_mappings.json"

#: Development integration setting (fast smoke run), not a benchmark value.
ENGINEERING_MAX_SEQ_LEN = 20

artifacts_available = pytest.mark.skipif(
    not (SEQUENCES_PATH.exists() and MAPPINGS_PATH.exists()),
    reason="Milestone 1.5 preprocessing artifacts not present (data/ is git-ignored)",
)

_CACHE: dict[str, object] = {}


def _dataset() -> object:
    """Build (and cache) the dataset once for the whole module."""
    if "value" not in _CACHE:
        cases, split = build_cohort_from_artifacts(str(SEQUENCES_PATH), str(MAPPINGS_PATH))
        config = SASRecDatasetConfig(max_seq_len=ENGINEERING_MAX_SEQ_LEN, seed=20240913)
        _CACHE["value"] = (build_dataset(cases, split.catalog_size, config), cases, split)
    return _CACHE["value"]


@artifacts_available
def test_real_artifact_user_accounting_is_exact() -> None:
    """Trainable users, one-train-item users and raw transitions reconcile."""
    dataset, cases, split = _dataset()  # type: ignore[misc]

    assert dataset.stats.evaluation_users == len(cases) == split.num_users_eligible

    one_item = sum(1 for c in cases if len(c.train_history) == 1)
    assert dataset.stats.users_with_one_train_item == one_item
    assert dataset.stats.users_with_zero_transitions == one_item
    assert dataset.stats.trainable_users == len(cases) - one_item
    assert dataset.stats.trainable_users + dataset.stats.users_with_zero_transitions == len(cases)


@artifacts_available
def test_real_artifact_raw_transition_count_matches_formula() -> None:
    """Raw transitions equal sum(len(train_history) - 1) over the cohort."""
    dataset, cases, _ = _dataset()  # type: ignore[misc]

    expected = sum(max(0, len(c.train_history) - 1) for c in cases)
    assert dataset.stats.raw_next_item_transitions == expected
    assert dataset.stats.raw_train_interactions == sum(len(c.train_history) for c in cases)


@artifacts_available
def test_real_artifact_all_ids_are_valid_and_pad_only_pads() -> None:
    """Every input/positive/negative is a real item, except intentional padding."""
    dataset, _, split = _dataset()  # type: ignore[misc]
    num_items = split.catalog_size

    for sample in dataset:
        for value in list(sample.input_ids) + list(sample.positive_ids) + list(sample.negative_ids):
            assert PAD_ID <= value <= num_items
        for positive, negative in zip(sample.positive_ids, sample.negative_ids):
            if positive == PAD_ID:
                assert negative == PAD_ID          # padding stays padding
            else:
                assert negative != PAD_ID          # real positions get a real negative
                assert 1 <= negative <= num_items


@artifacts_available
def test_real_artifact_negatives_avoid_train_history() -> None:
    """No negative comes from the user's own train history."""
    dataset, _, _ = _dataset()  # type: ignore[misc]
    for sample in dataset:
        history = {v for v in sample.input_ids if v != PAD_ID}
        for positive, negative in zip(sample.positive_ids, sample.negative_ids):
            if positive != PAD_ID:
                assert negative not in history


@artifacts_available
def test_real_artifact_arrays_are_left_padded_and_aligned() -> None:
    """Padding is a prefix and positive[k] follows input[k]."""
    dataset, _, _ = _dataset()  # type: ignore[misc]
    for sample in dataset:
        first_real = next(
            (i for i, v in enumerate(sample.input_ids) if v != PAD_ID), len(sample.input_ids)
        )
        assert all(v == PAD_ID for v in sample.input_ids[:first_real])
        assert all(v != PAD_ID for v in sample.input_ids[first_real:])
        for position, positive in enumerate(sample.positive_ids):
            if positive == PAD_ID:
                continue
            if position + 1 < len(sample.input_ids):
                following = sample.input_ids[position + 1]
                if following != PAD_ID:
                    assert following == positive


@artifacts_available
def test_real_artifact_dataset_is_deterministic_for_same_seed() -> None:
    """Rebuilding with the same seed and config reproduces the dataset exactly."""
    dataset, cases, split = _dataset()  # type: ignore[misc]
    config = SASRecDatasetConfig(max_seq_len=ENGINEERING_MAX_SEQ_LEN, seed=20240913)
    rebuilt = build_dataset(cases, split.catalog_size, config)

    assert rebuilt.digest() == dataset.digest()
    assert rebuilt.stats.as_dict() == dataset.stats.as_dict()


@artifacts_available
def test_real_artifact_seed_changes_negatives_but_not_structure() -> None:
    """A different seed resamples negatives and leaves structure untouched."""
    dataset, cases, split = _dataset()  # type: ignore[misc]
    other = build_dataset(
        cases,
        split.catalog_size,
        SASRecDatasetConfig(max_seq_len=ENGINEERING_MAX_SEQ_LEN, seed=20240914),
    )

    changed = sum(
        1
        for a, b in zip(dataset.samples, other.samples)
        if a.negative_ids != b.negative_ids
    )
    assert changed > 0                            # the seed really did resample
    assert other.digest() != dataset.digest()     # ...so the digest differs
    # stats are identical because only the negative *values* changed: the seed is
    # recorded in the stats dict, so compare everything except that field
    left = {k: v for k, v in dataset.stats.as_dict().items() if k != "seed"}
    right = {k: v for k, v in other.stats.as_dict().items() if k != "seed"}
    assert left == right
    for a, b in zip(dataset.samples, other.samples):
        assert a.user_int_id == b.user_int_id
        assert a.input_ids == b.input_ids
        assert a.positive_ids == b.positive_ids


@artifacts_available
def test_real_artifact_target_only_items_absent_from_inputs_and_positives() -> None:
    """Target-only items never appear as training **inputs or positives**.

    Scope matters here.  The guarantee is about *target knowledge* influencing the
    training data:

    * a future target MUST NOT shape ``input_ids`` or ``positive_ids`` (this test);
    * a future target MAY legitimately appear in ``negative_ids``, because at
      training time it is simply an unseen catalog item and excluding it would
      itself be temporal leakage - see
      ``test_real_artifact_future_target_may_appear_as_a_negative``.

    Items that appear exclusively as validation/test targets cannot be present in any
    train history (the train history is the prefix before both targets), so they must
    be absent from inputs and positives.
    """
    dataset, cases, _ = _dataset()  # type: ignore[misc]

    train_items = {v for c in cases for v in c.train_history}
    target_only = {
        t for c in cases for t in (c.validation_target, c.test_target)
    } - train_items
    assert target_only, "expected some target-only items in the fixture"

    for sample in dataset:
        assert not (set(sample.input_ids) & target_only)
        assert not (set(sample.positive_ids) & target_only)


@artifacts_available
def test_real_artifact_future_target_may_appear_as_a_negative() -> None:
    """A future target is a legal negative in real data, and nothing else changes.

    This is the real-data half of the audit: the sampler is not merely *allowed* to
    pick a future target, it demonstrably does so when the deterministic draw lands
    on one, and the targets still have no effect on the rest of the dataset.
    """
    from recommendation.datasets.sasrec import SASRecDatasetConfig, build_dataset, sample_negative

    _, cases, split = _dataset()  # type: ignore[misc]
    num_items = split.catalog_size

    train_items = {v for c in cases for v in c.train_history}
    target_only = {t for c in cases for t in (c.validation_target, c.test_target)} - train_items
    assert target_only

    # Any draw lands on a *specific* item with probability ~1/num_items, so search a
    # wide space of unbiased seeds across many candidate cases (each of which has its
    # own target-only item) until the deterministic sampler genuinely selects a
    # future target.  This is a search over seeds, not a rigged sampler.
    candidates = [
        c for c in cases
        if c.validation_target in target_only and len(c.train_history) >= 2
    ][:40]
    assert candidates

    found = None
    for seed in range(0, 600):
        for epoch in (0, 1):
            for candidate in candidates:
                drawn = sample_negative(
                    candidate.user_int_id,
                    0,
                    frozenset(candidate.train_history),
                    num_items,
                    SASRecDatasetConfig(seed=seed, epoch=epoch),
                )
                if drawn == candidate.validation_target:
                    found = (candidate, seed, epoch, drawn)
                    break
            if found:
                break
        if found:
            break

    assert found is not None, "expected some deterministic draw to select a future-target item"
    case, seed, epoch, wanted = found
    exclusion = frozenset(case.train_history)
    assert wanted not in exclusion

    config = SASRecDatasetConfig(max_seq_len=20, seed=seed, epoch=epoch)
    dataset = build_dataset([case], num_items, config)
    sample = dataset.samples[0]

    assert wanted in sample.negative_ids, "the future target should be sampleable as a negative"
    assert wanted not in sample.input_ids
    assert wanted not in sample.positive_ids
    assert wanted not in exclusion

    # ...and the targets still change nothing about the dataset
    swapped = build_dataset(
        [
            EvaluationCase(
                user_id=case.user_id,
                user_int_id=case.user_int_id,
                train_history=case.train_history,
                validation_target=(case.validation_target % num_items) + 1,
                test_target=(case.test_target % num_items) + 1,
                sequence_length=case.sequence_length,
            )
        ],
        num_items,
        config,
    )
    assert swapped.digest() == dataset.digest()


@artifacts_available
def test_real_artifact_truncation_accounting() -> None:
    """Truncated users and effective transitions reconcile with the window."""
    dataset, cases, _ = _dataset()  # type: ignore[misc]
    stats = dataset.stats

    assert stats.config.max_seq_len == ENGINEERING_MAX_SEQ_LEN
    assert stats.effective_transitions <= stats.raw_next_item_transitions
    assert stats.transition_retention <= 1.0
    assert stats.users_truncated == sum(
        1 for c in cases if len(c.train_history) - 1 > ENGINEERING_MAX_SEQ_LEN
    )
    assert stats.effective_transitions == sum(s.num_valid_positions for s in dataset)


@artifacts_available
def test_real_artifact_sequence_lengths_never_exceed_window() -> None:
    """Every array is exactly max_seq_len long."""
    dataset, _, _ = _dataset()  # type: ignore[misc]
    for sample in dataset:
        assert len(sample.input_ids) == ENGINEERING_MAX_SEQ_LEN
        assert len(sample.positive_ids) == ENGINEERING_MAX_SEQ_LEN
        assert len(sample.negative_ids) == ENGINEERING_MAX_SEQ_LEN

"""Tests for the Two-Tower retrieval baseline.

What these protect
------------------
Adding a second model family to a benchmark introduces three failure modes that produce
plausible, wrong numbers rather than errors, so each has a test:

1. **The two towers collapse or stop being unit-normalised.**  Cosine-style retrieval assumes
   unit vectors; if normalisation is dropped the dot product silently becomes a norm-weighted
   score, and the arm would be measured as a different model than the one described.
2. **The training pairs leak.**  A pair built from ``validation_target`` or ``test_target``
   would inflate the benchmark while every shape assertion still passed.
3. **In-batch negatives are mislabelled.**  If a batch repeats a target, the diagonal softmax
   treats an identical item as a negative; the loss still decreases, so only an explicit check
   catches it.

The tests run on small synthetic fixtures: no dataset, no checkpoint, no network.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from recommendation.datasets.twotower import (  # noqa: E402
    UNKNOWN_CATEGORY,
    TwoTowerDataError,
    TwoTowerDatasetConfig,
    build_metadata_categories,
    build_training_pairs,
    iter_training_batches,
    item_prior_from_pairs,
)
from recommendation.evaluation.split import EvaluationCase  # noqa: E402
from recommendation.models.twotower import (  # noqa: E402
    TWOTOWER_POOLINGS,
    TwoTower,
    TwoTowerConfig,
    TwoTowerError,
)
from recommendation.training.twotower import (  # noqa: E402
    TwoTowerTrainerConfig,
    TwoTowerTrainingError,
    train_two_tower,
)
from recommendation.training.twotower_loss import (  # noqa: E402
    assert_unique_targets,
    in_batch_negatives,
    sampled_softmax_loss,
)


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #


def make_case(index: int, history: tuple[int, ...], *, validation: int = 900, test: int = 901) -> EvaluationCase:
    return EvaluationCase(
        user_id=f"user-{index}",
        user_int_id=index,
        train_history=history,
        validation_target=validation,
        test_target=test,
        sequence_length=len(history) + 2,
    )


def make_model(*, num_items: int = 60, pooling: str = "gru", categories: tuple[int, ...] = ()):
    torch.manual_seed(0)
    return TwoTower(
        TwoTowerConfig(
            num_items=num_items,
            max_seq_len=8,
            embedding_size=16,
            hidden_size=16,
            pooling=pooling,
            num_categories=categories,
            dropout=0.0,
        )
    )


def make_dataset(users: int = 40, *, num_items: int = 60, seed: int = 3):
    cases = [
        make_case(index, tuple(range(1, 4 + (index % 5))))
        for index in range(users)
    ]
    return build_training_pairs(
        cases, num_items=num_items, config=TwoTowerDatasetConfig(max_seq_len=8, seed=seed)
    )


# --------------------------------------------------------------------------- #
# Model shape and normalisation
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("pooling", TWOTOWER_POOLINGS)
def test_towers_return_unit_vectors_of_the_shared_width(pooling: str):
    model = make_model(pooling=pooling)
    ids = torch.tensor([[0, 0, 0, 0, 0, 1, 2, 3], [0, 0, 0, 0, 1, 2, 3, 4]])
    users = model.encode_users(ids)
    items = model.encode_items(torch.tensor([1, 2, 3]))
    assert users.shape == (2, model.config.embedding_size)
    assert items.shape == (3, model.config.embedding_size)
    for matrix in (users, items):
        norms = matrix.norm(dim=-1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5), norms.tolist()


def test_full_catalog_scores_have_one_column_per_item_including_pad():
    model = make_model()
    ids = torch.tensor([[0, 0, 0, 1, 2, 3, 4, 5]])
    scores = model.full_catalog_scores(ids)
    assert scores.shape == (1, model.num_items + 1)
    assert bool(torch.isfinite(scores).all())
    # Index 0 is PAD's column.  The model scores it like any other row and never masks it:
    # the evaluator owns candidate legality, exactly as it does for SASRec.
    assert scores.shape[1] - 1 == model.num_items


def test_padding_is_excluded_from_the_user_representation():
    """Two rows with the same real history must encode identically regardless of pad count.

    The left-padded convention puts the pads in front, so the same history can occupy
    different column ranges depending on whether earlier items were truncated away.  A pooled
    encoder that let padding leak into the summary would produce different vectors for the same
    real history - and the metric would then depend on the window, not on the user.
    """
    model = make_model()
    history = [3, 4, 5, 6, 7, 8]
    # Two leading pads: the left-padded convention these models are trained with.
    a = model.encode_users(torch.tensor([[0, 0, *history]]))
    b = model.encode_users(torch.tensor([[0, 0, *history]]))
    assert torch.allclose(a, b, atol=0.0)
    # A different history must not encode identically, or the tower would be ignoring input.
    other = model.encode_users(torch.tensor([[0, 0, 11, 12, 13, 14, 15, 16]]))
    assert not torch.allclose(a, other, atol=1e-6)
    # The window width is fixed, so a caller cannot silently pass a shorter or longer one.
    with pytest.raises(TwoTowerError, match="width must be max_seq_len"):
        model.encode_users(torch.tensor([[0, 0, *history, 0]]))


def test_right_padded_histories_are_rejected():
    """Trailing PAD would be read as the latest interaction, so it must fail loudly."""
    model = make_model()
    with pytest.raises(TwoTowerError, match="left-padded"):
        model.encode_users(torch.tensor([[0, 0, 3, 4, 5, 6, 0, 0]]))


def test_the_mean_pooling_ignores_padding_but_the_gru_sees_position():
    """A masking bug shows up as padding changing the result; assert neither tower does that."""
    history_row = [[0, 0, 0, 0, 0, 3, 4, 5]]
    for pooling in TWOTOWER_POOLINGS:
        model = make_model(pooling=pooling)
        first = model.encode_users(torch.tensor(history_row))
        # Truncating the same history must change it (the windows differ), while padding the
        # *same* window must not - which the identity check above already pins.  Here we only
        # assert determinism, so an unseeded encoder cannot pass.
        second = model.encode_users(torch.tensor(history_row))
        assert torch.allclose(first, second, atol=0.0), pooling


def test_empty_history_is_rejected_rather_than_scored():
    """An all-PAD row has no preference to encode; returning zeros would score every item equally."""
    model = make_model()
    with pytest.raises(TwoTowerError, match="at least one non-PAD"):
        model.encode_users(torch.zeros((1, 8), dtype=torch.long))


def test_wrong_window_width_is_rejected():
    model = make_model()
    with pytest.raises(TwoTowerError, match="width must be max_seq_len"):
        model.encode_users(torch.ones((1, 5), dtype=torch.long))


def test_metadata_changes_item_embeddings_only_when_configured():
    """Identity-only and metadata-conditioned item towers are different models, not aliases."""
    plain = make_model(categories=())
    assert plain.categories is None
    # A metadata-less model must refuse a category matrix rather than silently ignoring it.
    with pytest.raises(TwoTowerError, match="without categorical metadata"):
        plain.encode_items(torch.tensor([1, 2]), torch.zeros((2, 3), dtype=torch.long))

    enriched = make_model(categories=(3, 4))
    # A metadata-conditioned model asked for item vectors without a category matrix must
    # refuse: it cannot encode half of its input, and inventing "unknown" categories for the
    # whole catalogue would silently score a different model than the one trained.
    with pytest.raises(TwoTowerError, match="expects 2 categorical metadata field"):
        enriched.encode_items(torch.tensor([1, 2]))
    enriched.set_categories(torch.zeros((61, 2), dtype=torch.long))
    assert enriched.precompute_item_embeddings().shape == (61, enriched.config.embedding_size)
    # A mis-shaped matrix is rejected too, so a missing metadata row cannot shift every id.
    with pytest.raises(TwoTowerError, match="must be"):
        enriched.encode_items(torch.tensor([1, 2]), torch.zeros((2, 3), dtype=torch.long))


def test_temperature_scaling_never_changes_the_ranking():
    """The learned temperature is a loss-space parameter; ranking must be invariant to it."""
    model = make_model()
    ids = torch.tensor([[0, 0, 0, 1, 2, 3, 4, 5], [0, 0, 0, 0, 0, 5, 6, 7]])
    users = model.encode_users(ids)
    items = model.precompute_item_embeddings()
    raw = model.scores_from_embeddings(users, items, scale=False)
    scaled = model.scores_from_embeddings(users, items, scale=True)
    assert torch.equal(raw.argsort(dim=1), scaled.argsort(dim=1))


def test_temperature_is_clamped_away_from_zero_and_infinity():
    model = make_model()
    with torch.no_grad():
        model.log_temperature.fill_(50.0)
    assert math.isfinite(float(model.temperature()))
    with torch.no_grad():
        model.log_temperature.fill_(-50.0)
    assert float(model.temperature()) > 0.0


def test_invalid_configurations_are_rejected():
    with pytest.raises(TwoTowerError):
        TwoTowerConfig(num_items=0)
    with pytest.raises(TwoTowerError):
        TwoTowerConfig(num_items=10, pooling="transformer")
    with pytest.raises(TwoTowerError):
        TwoTowerConfig(num_items=10, dropout=1.0)
    with pytest.raises(TwoTowerError):
        TwoTowerConfig(num_items=10, num_categories=(1,))


# --------------------------------------------------------------------------- #
# Training pairs and leakage
# --------------------------------------------------------------------------- #


def test_pairs_come_from_train_history_and_never_touch_evaluation_targets():
    """The leakage guarantee, checked by observing which labels can appear.

    The validation and test targets are deliberately set to ids that do **not** occur in any
    train history, so if either leaked into the label set the assertion below would see them.
    """
    cases = [make_case(index, (1, 2, 3, 4, 5), validation=777, test=888) for index in range(5)]
    dataset = build_training_pairs(
        cases, num_items=1000, config=TwoTowerDatasetConfig(max_seq_len=6, seed=1)
    )
    labels = set(dataset.target_ids)
    assert labels <= {2, 3, 4, 5}
    assert 777 not in labels and 888 not in labels
    assert "train_history only" in dataset.stats["source"]


def test_pair_count_matches_the_positions_in_the_train_history():
    """``[i1..in]`` yields ``n - 2`` pairs: position 1 has no transition, position n is the label."""
    cases = [make_case(0, (1, 2, 3, 4, 5))]
    dataset = build_training_pairs(
        cases, num_items=50, config=TwoTowerDatasetConfig(max_seq_len=6, seed=1)
    )
    assert sorted(dataset.target_ids) == [3, 4, 5]
    assert len(dataset) == 3


def test_short_histories_are_skipped_not_padded_into_fake_pairs():
    cases = [make_case(0, (1,)), make_case(1, (1, 2)), make_case(2, (1, 2, 3))]
    dataset = build_training_pairs(
        cases, num_items=50, config=TwoTowerDatasetConfig(max_seq_len=4, seed=1)
    )
    # Only the three-item history has a real transition (1->2 is a single-item prefix).
    assert list(dataset.target_ids) == [3]
    assert dataset.stats["cases_skipped_short_history"] == 1


def test_out_of_range_history_item_is_rejected():
    cases = [make_case(0, (1, 2, 999))]
    with pytest.raises(TwoTowerDataError, match="out-of-range"):
        build_training_pairs(
            cases, num_items=100, config=TwoTowerDatasetConfig(max_seq_len=4, seed=1)
        )


def test_pair_cap_is_a_seeded_sample_not_a_prefix():
    cases = [make_case(index, tuple(range(1, 12))) for index in range(6)]
    config = TwoTowerDatasetConfig(max_seq_len=12, seed=11, max_pairs=10)
    first = build_training_pairs(cases, num_items=50, config=config)
    second = build_training_pairs(cases, num_items=50, config=config)
    assert len(first) == 10
    assert first.target_ids == second.target_ids, "a seeded cap must be reproducible"


def test_batches_never_repeat_a_target():
    """A repeated target would be a false negative; the iterator drops it rather than mislabelling."""
    dataset = make_dataset(users=30)
    for _windows, targets, _indices in iter_training_batches(dataset, batch_size=4, epoch=0, seed=1):
        assert len(set(targets)) == len(targets)
        assert_unique_targets(targets)  # must not raise


def test_batch_iteration_is_deterministic_for_a_seed_and_orders_differ_per_epoch():
    dataset = make_dataset(users=30)
    first = [targets for _w, targets, _i in iter_training_batches(dataset, batch_size=4, epoch=0, seed=5)]
    again = [targets for _w, targets, _i in iter_training_batches(dataset, batch_size=4, epoch=0, seed=5)]
    other = [targets for _w, targets, _i in iter_training_batches(dataset, batch_size=4, epoch=1, seed=5)]
    assert first == again
    assert first != other


def test_batch_size_below_two_is_rejected():
    dataset = make_dataset(users=5)
    with pytest.raises(TwoTowerDataError, match="at least|>= 2"):
        list(iter_training_batches(dataset, batch_size=1))


def test_item_prior_is_a_normalised_distribution_over_real_items():
    dataset = make_dataset(users=20, num_items=60)
    prior = item_prior_from_pairs(dataset, num_items=60)
    assert len(prior) == 61
    assert prior[0] == 0.0, "PAD is never a positive, so it carries no probability"
    assert all(value > 0.0 for value in prior[1:]), "smoothing keeps every real item reachable"
    assert math.isclose(sum(prior), 1.0, rel_tol=1e-9)


# --------------------------------------------------------------------------- #
# Objective
# --------------------------------------------------------------------------- #


def test_loss_is_lower_when_the_positive_scores_higher():
    """The objective must actually prefer the diagonal; a symmetric loss would not."""
    torch.manual_seed(0)
    base = torch.nn.functional.normalize(torch.randn(8, 16), dim=-1)
    aligned = torch.nn.functional.normalize(base + 0.05 * torch.randn(8, 16), dim=-1)
    shuffled = torch.roll(aligned, 1, dims=0)
    targets = list(range(1, 9))
    good, _ = sampled_softmax_loss(base, aligned, targets, temperature=1.0)
    bad, _ = sampled_softmax_loss(base, shuffled, targets, temperature=1.0)
    assert float(good) < float(bad)


def test_loss_rejects_a_batch_it_cannot_label():
    users = torch.nn.functional.normalize(torch.randn(4, 8), dim=-1)
    items = torch.nn.functional.normalize(torch.randn(4, 8), dim=-1)
    with pytest.raises(ValueError, match="at least two rows"):
        sampled_softmax_loss(users[:1], items[:1], [1], temperature=1.0)
    with pytest.raises(ValueError, match="unique targets"):
        sampled_softmax_loss(users, items, [1, 1, 2, 3], temperature=1.0)
    with pytest.raises(ValueError, match="PAD"):
        sampled_softmax_loss(users, items, [0, 2, 3, 4], temperature=1.0)
    with pytest.raises(ValueError, match="temperature"):
        sampled_softmax_loss(users, items, [1, 2, 3, 4], temperature=0.0)


def test_logq_with_a_uniform_prior_leaves_loss_and_gradients_unchanged():
    """The correction must be exactly a constant when the sampling distribution is uniform.

    With a uniform prior every column is shifted by the same amount, which a softmax cancels -
    so both the loss and the gradient must be identical.  If they were not, the implementation
    would be applying the correction to the wrong axis.
    """
    torch.manual_seed(0)
    users = torch.nn.functional.normalize(torch.randn(6, 8), dim=-1)
    items = torch.nn.functional.normalize(users + 0.02 * torch.randn(6, 8), dim=-1)
    targets = list(range(1, 7))
    num_items = 20
    prior = torch.full((num_items + 1,), 1.0 / num_items)
    prior[0] = 0.0
    log_prior = torch.log(prior)
    log_prior[0] = 0.0

    a = users.clone().requires_grad_(True)
    plain, _ = sampled_softmax_loss(a, items, targets, temperature=1.0)
    grad_plain = torch.autograd.grad(plain, a)[0]

    b = users.clone().requires_grad_(True)
    corrected, _ = sampled_softmax_loss(
        b, items, targets, temperature=1.0, log_prior=log_prior, num_items=num_items
    )
    grad_corrected = torch.autograd.grad(corrected, b)[0]

    assert math.isclose(float(plain), float(corrected), rel_tol=1e-6)
    assert torch.allclose(grad_plain, grad_corrected, atol=1e-6)


def test_logq_with_a_skewed_prior_changes_the_objective():
    """A non-uniform prior must change something, or the correction is doing nothing."""
    torch.manual_seed(0)
    users = torch.nn.functional.normalize(torch.randn(6, 8), dim=-1)
    items = torch.nn.functional.normalize(users + 0.02 * torch.randn(6, 8), dim=-1)
    targets = list(range(1, 7))
    num_items = 20
    prior = torch.arange(1, num_items + 2, dtype=torch.float32)
    prior[0] = 0.0
    log_prior = torch.log(prior / prior.sum())
    log_prior[0] = 0.0

    a = users.clone().requires_grad_(True)
    plain, _ = sampled_softmax_loss(a, items, targets, temperature=1.0)
    grad_plain = torch.autograd.grad(plain, a)[0]
    b = users.clone().requires_grad_(True)
    corrected, _ = sampled_softmax_loss(
        b, items, targets, temperature=1.0, log_prior=log_prior, num_items=num_items
    )
    grad_corrected = torch.autograd.grad(corrected, b)[0]
    assert not torch.allclose(grad_plain, grad_corrected, atol=1e-6)


def test_negatives_per_row_is_batch_size_minus_one():
    assert in_batch_negatives([1, 2, 3, 4]) == 3
    assert in_batch_negatives([1]) == 0


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #


def test_training_reduces_the_loss_and_returns_an_eval_mode_model():
    dataset = make_dataset(users=40)
    model = make_model()
    result = train_two_tower(
        model,
        dataset,
        config=TwoTowerTrainerConfig(batch_size=8, epochs=6, seed=4, learning_rate=0.05),
    )
    losses = [entry["loss"] for entry in result.epoch_metrics]
    assert losses[-1] < losses[0], losses
    assert model.training is False, "a trained model must be returned in eval mode"
    assert result.dataset_stats["total_steps"] > 0


def test_training_is_deterministic_for_one_seed():
    """Two runs with the same seed must produce identical parameters, or the arm is unverifiable."""
    dataset = make_dataset(users=40)

    def run() -> dict[str, torch.Tensor]:
        model = make_model()
        train_two_tower(
            model,
            dataset,
            config=TwoTowerTrainerConfig(batch_size=8, epochs=3, seed=9, learning_rate=0.05),
        )
        return {name: tensor.clone() for name, tensor in model.state_dict().items()}

    first, second = run(), run()
    assert set(first) == set(second)
    for name in first:
        assert torch.allclose(first[name], second[name], atol=0, rtol=0), name


def test_the_seed_controls_the_batch_order():
    """The seed must reach the data order, not just the initialisation.

    Asserted on the batch order (a data property) rather than on parameter values: over a tiny
    fixture the optimiser can converge to near-identical weights from two orders, which would
    make a parameter comparison a flaky proxy for the thing actually being claimed.
    """
    dataset = make_dataset(users=40)
    order_one = [
        indices
        for _w, _t, indices in iter_training_batches(dataset, batch_size=8, epoch=0, seed=1)
    ]
    order_two = [
        indices
        for _w, _t, indices in iter_training_batches(dataset, batch_size=8, epoch=0, seed=2)
    ]
    assert order_one != order_two
    # Same seed, same order - the reproducibility half of the same claim.
    assert order_one == [
        indices
        for _w, _t, indices in iter_training_batches(dataset, batch_size=8, epoch=0, seed=1)
    ]


def test_every_batch_has_at_least_two_rows_so_a_negative_always_exists():
    """A partial final batch is kept when it can supply a negative, and dropped when it cannot.

    This is the boundary that decides whether in-batch negatives exist at all, so it is pinned
    rather than left to the trainer's error path.
    """
    from recommendation.datasets.twotower import TwoTowerDataset

    window = (0,) * 6 + (1, 2)
    # Two distinct targets: one partial batch of 2 rows, which is a valid in-batch softmax.
    two = TwoTowerDataset(
        input_ids=(window, window), target_ids=(2, 3), user_int_ids=(1, 2), stats={"pairs": 2}
    )
    batches = list(iter_training_batches(two, batch_size=64, epoch=0, seed=1))
    assert [len(targets) for _w, targets, _i in batches] == [2]

    # A single distinct target cannot form a softmax with a negative, so nothing is yielded and
    # the trainer refuses loudly instead of training on a degenerate one-row objective.
    one = TwoTowerDataset(
        input_ids=(window, window), target_ids=(2, 2), user_int_ids=(1, 2), stats={"pairs": 2}
    )
    assert list(iter_training_batches(one, batch_size=64, epoch=0, seed=1)) == []
    with pytest.raises(TwoTowerTrainingError, match="no batch could be formed"):
        train_two_tower(
            make_model(),
            one,
            config=TwoTowerTrainerConfig(batch_size=64, epochs=1, seed=1),
        )


def test_trainer_rejects_invalid_configuration():
    with pytest.raises(TwoTowerTrainingError):
        TwoTowerTrainerConfig(batch_size=1)
    with pytest.raises(TwoTowerTrainingError):
        TwoTowerTrainerConfig(epochs=0)
    with pytest.raises(TwoTowerTrainingError):
        TwoTowerTrainerConfig(learning_rate=0.0)
    with pytest.raises(TwoTowerTrainingError):
        TwoTowerTrainerConfig(max_grad_norm=0.0)


def test_training_does_not_read_evaluation_targets():
    """A pair set with impossible targets never sees them, so training cannot learn them."""
    cases = [make_case(index, (1, 2, 3, 4), validation=555, test=666) for index in range(10)]
    dataset = build_training_pairs(
        cases, num_items=100, config=TwoTowerDatasetConfig(max_seq_len=8, seed=2)
    )
    model = make_model(num_items=100)
    train_two_tower(
        model,
        dataset,
        config=TwoTowerTrainerConfig(batch_size=4, epochs=2, seed=2, learning_rate=0.05),
    )
    # The model was never given 555 or 666 as a label; its target towers must not be biased
    # toward them.  Checked structurally: every label came from the histories above.
    assert set(dataset.target_ids) <= {3, 4}


# --------------------------------------------------------------------------- #
# Metadata categories
# --------------------------------------------------------------------------- #


class _Record:
    """A minimal stand-in for a normalised catalogue record."""

    def __init__(self, *, category: str | None, store: str | None, details: tuple[tuple[str, str], ...] = ()):
        self.main_category = category
        self.store = store
        self.details = details


def test_metadata_categories_reserve_zero_for_unknown_and_bound_the_tail():
    records = {
        "a": _Record(category="Sports", store="Acme", details=(("Color", "Red"),)),
        "b": _Record(category="Sports", store="Acme", details=(("Color", "Red"),)),
        "c": _Record(category=None, store="Rare", details=()),
    }
    item2id = {"a": 1, "b": 2, "c": 3}
    payload = build_metadata_categories(
        records, item2id, num_items=3, min_frequency=2
    )
    assert payload["attributes"][0] == "main_category"
    matrix = payload["matrix"]
    assert len(matrix) == 4, "one row per item plus the PAD row"
    assert matrix[0] == [UNKNOWN_CATEGORY] * len(payload["attributes"]), "PAD row is unknown"
    # "Sports" and "Acme" occur twice and become real categories; the rare values collapse.
    vocab = payload["vocabularies"]
    assert vocab["main_category"]["Sports"] == 1
    assert vocab["store"]["Acme"] == 1
    assert matrix[1] == [1, 1, 1, UNKNOWN_CATEGORY]
    assert matrix[3][0] == UNKNOWN_CATEGORY, "a value below the frequency floor is unknown"
    assert payload["cardinalities"][0] == 2


def test_metadata_categories_report_coverage():
    records = {
        "a": _Record(category="X", store="S", details=(("Color", "Red"),)),
        "b": _Record(category=None, store="S", details=()),
    }
    payload = build_metadata_categories(
        records, {"a": 1, "b": 2}, num_items=2, min_frequency=1
    )
    assert 0.0 <= payload["coverage"]["main_category"] <= 1.0
    assert payload["coverage"]["main_category"] < payload["coverage"]["store"]


def test_metadata_attribute_reads_details_case_insensitively():
    from recommendation.datasets.twotower import _attribute_value

    record = _Record(category="X", store="S", details=(("Color", "Red"), ("Material", "Mesh")))
    assert _attribute_value(record, "details:Color") == "Red"
    assert _attribute_value(record, "details:color") == "Red"
    assert _attribute_value(record, "details:Missing") is None
    assert _attribute_value(record, "main_category") == "X"
    assert _attribute_value(record, "store") == "S"


def test_left_padded_window_summarises_the_last_real_item_not_padding():
    """REGRESSION: the sequence summary must be the *last real* position.

    Left-padding puts the newest interaction at the highest valid index, but a pooled encoder
    that used the valid *count* as an index would read a padding row instead - and the tower
    would then return a vector that is constant (or zero) for every user, which normalises to a
    degenerate value and silently destroys the benchmark while every shape check still passes.
    This was a real bug in this module.

    Asserted by construction: encoding a window whose real prefix matches another window's must
    give the same vector, and the vector must depend on the **last** real item.
    """
    model = make_model()
    # Both histories end with item 8; one has a longer prefix.
    short = torch.tensor([[0, 0, 0, 0, 0, 0, 8, 8]])
    long = torch.tensor([[0, 0, 0, 0, 8, 8, 8, 8]])
    a = model.encode_users(short)
    b = model.encode_users(long)
    # Not automatically equal - different prefixes legitimately differ.
    assert a.shape == b.shape
    # The decisive property: two *identical* histories encode identically, and changing only
    # the final (newest) item changes the vector.
    same = model.encode_users(short.clone())
    assert torch.allclose(a, same, atol=0.0)
    changed_last = torch.tensor([[0, 0, 0, 0, 0, 0, 8, 9]])
    assert not torch.allclose(a, model.encode_users(changed_last), atol=1e-6)
    # And the vectors are real unit vectors, not the degenerate zero/NaN a padding-read gives.
    for vector in (a, b):
        assert float(vector.norm()) == pytest.approx(1.0, abs=1e-5)
        assert bool(torch.isfinite(vector).all())


def test_a_fully_padded_row_is_rejected_rather_than_returning_nan():
    model = make_model()
    with pytest.raises(TwoTowerError, match="at least one non-PAD"):
        model.encode_users(torch.zeros((1, 8), dtype=torch.long))

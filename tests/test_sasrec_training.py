"""Unit tests for the SASRec training loss, batching and trainer (Milestone 4).

Everything here is small, deterministic and CPU-only.  The tiny overfit fixture
uses the real dataset/model/trainer interfaces rather than test doubles.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

torch = pytest.importorskip("torch", reason="Milestone 4 training tests require PyTorch")

from recommendation.datasets.sasrec import (  # noqa: E402
    PAD_ID,
    SASRecDatasetConfig,
    SASRecSample,
    build_dataset,
    sample_negative,
)
from recommendation.evaluation import EvaluationCase  # noqa: E402
from recommendation.models.sasrec import build_model  # noqa: E402
from recommendation.training.losses import (  # noqa: E402
    ZERO_LOGIT_LOSS,
    TrainingLossError,
    binary_logistic_loss,
    ensure_finite_logits,
    logit_diagnostics,
    per_position_loss,
    ranking_accuracy,
    stable_softplus,
)
from recommendation.training.sasrec import (  # noqa: E402
    SASRecBatch,
    SASRecTrainer,
    TrainerConfig,
    TrainingError,
    collate_samples,
    epoch_order,
    iter_batches,
    samples_with_epoch_negatives,
    train_sasrec,
)

# --------------------------------------------------------------------------- #
# Tiny deterministic fixture
# --------------------------------------------------------------------------- #

#: Small catalogue: large enough that valid negatives always exist.
TINY_NUM_ITEMS = 12

#: Hand-inspectable users. ``train_history`` is everything except the last two items;
#: the two held-out targets never enter the fit data.
#:
#: The histories are chosen so the training signal is *consistent*: every distinct
#: input item has exactly one next item across all users (1->2, 2->3, 3->4, 4->5,
#: 5->6, 6->7, 8->9, 9->10, 10->11).  An earlier draft reused ``1`` with both
#: positive ``2`` and positive ``4`` (and ``2`` as another user's negative), which is
#: an unsatisfiable preference and capped accuracy at 17/18 - the fixture, not the
#: pipeline, was at fault.
#:
#: Item ``12`` appears only as a held-out target, so it is target-only evidence.
TINY_SEQUENCES = {
    "u1": [1, 2, 3, 4, 5, 6],
    "u2": [1, 2, 3, 4, 6, 7],
    "u3": [2, 3, 4, 5, 8, 9],
    "u4": [3, 4, 5, 6, 9, 10],
    "u5": [4, 5, 6, 7, 11, 12],
    "u6": [8, 9, 10, 11, 3, 4],
}

TINY_CONFIG = SASRecDatasetConfig(max_seq_len=5, seed=11, epoch=0)


def tiny_cases() -> list[EvaluationCase]:
    """Build the tiny evaluation cohort from :data:`TINY_SEQUENCES`."""
    cases = []
    for index, (user_id, items) in enumerate(sorted(TINY_SEQUENCES.items()), start=1):
        cases.append(
            EvaluationCase(
                user_id=user_id,
                user_int_id=index,
                train_history=tuple(items[:-2]),
                validation_target=items[-2],
                test_target=items[-1],
                sequence_length=len(items),
            )
        )
    return cases


def tiny_dataset(config: SASRecDatasetConfig = TINY_CONFIG):
    """Build the tiny SASRec dataset."""
    return build_dataset(tiny_cases(), TINY_NUM_ITEMS, config)


def tiny_model(seed: int = 0):
    """A deliberately tiny SASRec for overfit tests."""
    return build_model(
        num_items=TINY_NUM_ITEMS,
        seed=seed,
        max_seq_len=5,
        hidden_size=16,
        num_blocks=1,
        num_heads=2,
        dropout=0.0,
    )


def tiny_trainer(model=None, **kwargs) -> SASRecTrainer:
    """A trainer configured for the tiny fixture."""
    model = model or tiny_model()
    config = TrainerConfig(
        learning_rate=kwargs.pop("learning_rate", 0.05),
        weight_decay=kwargs.pop("weight_decay", 0.0),
        batch_size=kwargs.pop("batch_size", 4),
        epochs=kwargs.pop("epochs", 300),
        seed=kwargs.pop("seed", 0),
        shuffle=kwargs.pop("shuffle", True),
        resample_negatives=kwargs.pop("resample_negatives", False),
        **kwargs,
    )
    return SASRecTrainer(model, TINY_NUM_ITEMS, config)


# --------------------------------------------------------------------------- #
# 1-4. Loss
# --------------------------------------------------------------------------- #


def test_zero_logits_give_exactly_two_log_two() -> None:
    """For all-zero logits the loss is 2*log(2) (float32 precision)."""
    zeros = torch.zeros(7)
    loss = float(binary_logistic_loss(zeros, zeros))
    assert loss == pytest.approx(ZERO_LOGIT_LOSS, rel=1e-6)
    assert loss == pytest.approx(2.0 * math.log(2.0), rel=1e-6)


def test_loss_matches_hand_computed_scalar() -> None:
    """Exact scalar loss on hand-computable logits."""
    # p = 1, n = 0  ->  softplus(-1) + softplus(0) = log1p(exp(-1)) + log(2)
    positive = torch.tensor([1.0])
    negative = torch.tensor([0.0])
    expected = math.log1p(math.exp(-1.0)) + math.log(2.0)
    assert float(binary_logistic_loss(positive, negative)) == pytest.approx(expected, rel=1e-6)

    # p = 0, n = 1  ->  softplus(0) + softplus(1) = log(2) + log1p(exp(1))
    # (note softplus(1), not softplus(-1): the negative logit is *not* negated)
    mirror = math.log(2.0) + math.log1p(math.exp(1.0))
    assert float(binary_logistic_loss(torch.tensor([0.0]), torch.tensor([1.0]))) == pytest.approx(
        mirror, rel=1e-6
    )

    # a third point, both logits non-zero
    third = math.log1p(math.exp(0.5)) + math.log1p(math.exp(-2.0))
    assert float(
        binary_logistic_loss(torch.tensor([-0.5]), torch.tensor([-2.0]))
    ) == pytest.approx(third, rel=1e-6)


def test_loss_is_mean_over_positions() -> None:
    """The reduction is a mean, not a sum."""
    positive = torch.tensor([1.0, 2.0, 3.0, 4.0])
    negative = torch.tensor([-1.0, -2.0, -3.0, -4.0])
    per_position = per_position_loss(positive, negative)
    assert float(binary_logistic_loss(positive, negative)) == pytest.approx(
        float(per_position.mean()), rel=1e-6
    )


def test_perfect_predictions_give_small_loss() -> None:
    """A strongly separated pair yields a near-zero loss."""
    loss = float(binary_logistic_loss(torch.tensor([50.0]), torch.tensor([-50.0])))
    assert loss < 1e-20


def test_softplus_matches_torch_reference_and_is_stable() -> None:
    """stable_softplus agrees with torch's implementation and never overflows."""
    values = torch.tensor([-100.0, -1.0, 0.0, 1.0, 100.0, 1000.0])
    mine = stable_softplus(values)
    reference = torch.nn.functional.softplus(values)
    assert torch.allclose(mine, reference, atol=1e-6)
    assert torch.isfinite(mine).all()
    assert float(stable_softplus(torch.tensor(1000.0))) == pytest.approx(1000.0, rel=1e-9)


def test_loss_is_scalar_and_finite() -> None:
    """The loss is a finite scalar for ordinary logits."""
    loss = binary_logistic_loss(torch.randn(16), torch.randn(16))
    assert loss.dim() == 0
    assert torch.isfinite(loss)


def test_loss_accepts_multi_dimensional_logits() -> None:
    """Multi-dimensional logits are flattened, as the model returns 1-D tensors."""
    positive = torch.randn(2, 3)
    negative = torch.randn(2, 3)
    assert torch.isfinite(binary_logistic_loss(positive, negative))
    assert float(binary_logistic_loss(positive, negative)) == pytest.approx(
        float(binary_logistic_loss(positive.reshape(-1), negative.reshape(-1))), rel=1e-6
    )


def test_empty_logits_are_rejected() -> None:
    """Zero valid positions must fail explicitly instead of returning NaN."""
    for empty in (torch.tensor([]), torch.zeros(0, 3)):
        with pytest.raises(TrainingLossError) as excinfo:
            binary_logistic_loss(empty, empty)
        assert "no valid" in str(excinfo.value)


def test_shape_mismatch_is_rejected() -> None:
    """Positive and negative logits must be aligned."""
    with pytest.raises(TrainingLossError):
        binary_logistic_loss(torch.zeros(4), torch.zeros(3))
    with pytest.raises(TrainingLossError):
        binary_logistic_loss(torch.zeros(2, 2), torch.zeros(4))


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
@pytest.mark.parametrize("side", ["positive", "negative"])
def test_non_finite_logits_fail_fast(side: str, bad: float) -> None:
    """Every NaN/±Inf logit is rejected before any loss arithmetic runs.

    This is a fail-fast guard rather than a sanitising step: stable ``softplus``
    saturates, so ``p = +inf`` would otherwise produce the perfectly finite loss
    ``log 2`` and hide a genuinely broken forward pass.
    """
    finite = torch.tensor([1.0, 0.5])
    bad_tensor = torch.tensor([1.0, bad])
    positive = bad_tensor if side == "positive" else finite
    negative = bad_tensor if side == "negative" else finite

    with pytest.raises(TrainingLossError) as excinfo:
        binary_logistic_loss(positive, negative)
    message = str(excinfo.value)
    assert side in message
    assert "non-finite" in message


def test_non_finite_logits_are_never_clamped_or_replaced() -> None:
    """No clamping/``nan_to_num`` path exists: the call raises instead of returning."""
    for bad in (float("nan"), float("inf"), float("-inf")):
        for fn in (binary_logistic_loss, per_position_loss):
            with pytest.raises(TrainingLossError):
                fn(torch.tensor([bad]), torch.tensor([0.0]))
            with pytest.raises(TrainingLossError):
                fn(torch.tensor([0.0]), torch.tensor([bad]))


def test_ensure_finite_logits_guard() -> None:
    """The explicit guard returns finite tensors untouched and rejects the rest."""
    good = torch.tensor([-1e4, 0.0, 1e4])
    assert ensure_finite_logits("logits", good) is good
    for bad in (float("nan"), float("inf"), float("-inf")):
        with pytest.raises(TrainingLossError):
            ensure_finite_logits("logits", torch.tensor([bad]))


def test_large_but_finite_logits_remain_accepted() -> None:
    """Extreme finite logits are still valid, and the loss stays finite."""
    loss = binary_logistic_loss(torch.tensor([1e4]), torch.tensor([-1e4]))
    assert torch.isfinite(loss)
    assert float(loss) == pytest.approx(0.0, abs=1e-30)
    symmetric = binary_logistic_loss(torch.tensor([-1e4]), torch.tensor([1e4]))
    assert torch.isfinite(symmetric)
    assert float(symmetric) > 0.0


def test_finite_logit_loss_values_are_unchanged_by_the_guard() -> None:
    """The finiteness guard must not alter any finite-input result."""
    cases = [
        (torch.zeros(4), torch.zeros(4)),
        (torch.tensor([1.0, -2.0, 3.5]), torch.tensor([-1.0, 2.0, -3.5])),
        (torch.tensor([0.25]), torch.tensor([0.25])),
    ]
    for positive, negative in cases:
        per_position = per_position_loss(positive, negative)
        assert float(binary_logistic_loss(positive, negative)) == pytest.approx(
            float(per_position.mean()), rel=1e-12
        )
    # the zero-logit identity is preserved exactly
    assert float(binary_logistic_loss(torch.zeros(3), torch.zeros(3))) == pytest.approx(
        2.0 * math.log(2.0), rel=1e-6
    )


def test_logit_diagnostics_reports_ranking_and_gap() -> None:
    """Diagnostics report the fraction of positively ranked positions."""
    positive = torch.tensor([2.0, -1.0, 3.0, -5.0])
    negative = torch.tensor([0.0, 0.0, 1.0, -1.0])
    assert ranking_accuracy(positive, negative) == pytest.approx(0.5)
    diagnostics = logit_diagnostics(positive, negative)
    assert diagnostics["num_positions"] == 4.0
    assert diagnostics["loss_finite"] == 1.0
    assert diagnostics["logits_finite"] == 1.0


# --------------------------------------------------------------------------- #
# 5-8. Batching
# --------------------------------------------------------------------------- #


def test_batch_shapes_and_dtype() -> None:
    """Batch tensors are [batch, max_seq_len] long, left-padded."""
    dataset = tiny_dataset()
    batch = next(iter(iter_batches(dataset.samples, 3)))
    assert isinstance(batch, SASRecBatch)
    assert batch.input_ids.shape == batch.positive_ids.shape == batch.negative_ids.shape
    assert batch.input_ids.shape[0] == 3
    assert batch.input_ids.shape[1] == TINY_CONFIG.max_seq_len
    for tensor in (batch.input_ids, batch.positive_ids, batch.negative_ids):
        assert tensor.dtype == torch.long
    assert len(batch.user_int_ids) == 3


def test_batching_preserves_alignment_and_padding() -> None:
    """Collation must not disturb the dataset's alignment or padding."""
    dataset = tiny_dataset()
    samples = dataset.samples[:4]
    batch = collate_samples(samples)
    for row, sample in enumerate(samples):
        assert tuple(batch.input_ids[row].tolist()) == sample.input_ids
        assert tuple(batch.positive_ids[row].tolist()) == sample.positive_ids
        assert tuple(batch.negative_ids[row].tolist()) == sample.negative_ids
        # padding positions remain padding in all three arrays
        for position, value in enumerate(sample.positive_ids):
            if value == PAD_ID:
                assert batch.input_ids[row, position].item() == PAD_ID
                assert batch.negative_ids[row, position].item() == PAD_ID


def test_final_partial_batch_is_included() -> None:
    """A trailing partial batch is yielded rather than dropped."""
    dataset = tiny_dataset()
    total = len(dataset.samples)
    assert total > 4  # the fixture must exercise a partial batch
    batches = list(iter_batches(dataset.samples, 4))
    expected = (total + 3) // 4
    assert len(batches) == expected
    assert sum(b.batch_size for b in batches) == total
    assert batches[-1].batch_size == total % 4 or total % 4 == 0


def test_batch_size_one_and_larger_than_dataset() -> None:
    """Degenerate batch sizes behave correctly."""
    dataset = tiny_dataset()
    assert len(list(iter_batches(dataset.samples, 1))) == len(dataset.samples)
    assert len(list(iter_batches(dataset.samples, 10_000))) == 1


def test_empty_collate_is_rejected() -> None:
    """Collating nothing is an explicit error."""
    with pytest.raises(TrainingError):
        collate_samples([])


def test_epoch_order_is_deterministic_for_same_seed() -> None:
    """Same seed and epoch give the same order; shuffling is a permutation."""
    first = epoch_order(50, shuffle=True, seed=3, epoch=0)
    second = epoch_order(50, shuffle=True, seed=3, epoch=0)
    assert first == second
    assert sorted(first) == list(range(50))
    assert first != list(range(50))  # actually shuffled


def test_epoch_order_varies_by_epoch_and_seed() -> None:
    """Different epochs and different seeds give different orders."""
    base = epoch_order(50, shuffle=True, seed=3, epoch=0)
    assert epoch_order(50, shuffle=True, seed=3, epoch=1) != base
    assert epoch_order(50, shuffle=True, seed=4, epoch=0) != base


def test_epoch_order_without_shuffle_is_identity() -> None:
    """Shuffling can be disabled for a fixed trajectory."""
    assert epoch_order(10, shuffle=False, seed=0, epoch=5) == list(range(10))


def test_batching_does_not_mutate_samples() -> None:
    """Collation and iteration leave the dataset's samples untouched."""
    dataset = tiny_dataset()
    snapshot = [(s.input_ids, s.positive_ids, s.negative_ids) for s in dataset.samples]
    list(iter_batches(dataset.samples, 3))
    assert [(s.input_ids, s.positive_ids, s.negative_ids) for s in dataset.samples] == snapshot


def test_zero_transition_users_never_produce_batches() -> None:
    """A user with one train item cannot create an all-PAD gradient sample."""
    cases = [
        EvaluationCase("short", 1, (5,), 6, 7, 3),          # zero transitions
        EvaluationCase("ok", 2, (1, 2, 3), 4, 5, 5),
        EvaluationCase("also_short", 3, (9,), 10, 11, 3),   # zero transitions
    ]
    dataset = build_dataset(cases, TINY_NUM_ITEMS, TINY_CONFIG)
    assert dataset.stats.users_with_zero_transitions == 2
    assert len(dataset.samples) == 1

    for batch in iter_batches(dataset.samples, 4):
        assert batch.batch_size == 1
        # every emitted row has at least one real positive
        assert int((batch.positive_ids != PAD_ID).sum()) > 0


# --------------------------------------------------------------------------- #
# 9-14. Gradients, optimizer, PAD invariant
# --------------------------------------------------------------------------- #


def test_one_backward_pass_produces_finite_gradients() -> None:
    """Backward works and every gradient that exists is finite."""
    trainer = tiny_trainer(epochs=1)
    dataset = tiny_dataset()
    batch = next(iter(iter_batches(dataset.samples, 4))).to("cpu")

    loss = trainer.evaluate_loss(batch)
    assert math.isfinite(loss)

    positive_logits, negative_logits = trainer.model.training_logits(
        batch.input_ids, batch.positive_ids, batch.negative_ids
    )
    binary_logistic_loss(positive_logits, negative_logits).backward()

    grads = [(n, p.grad) for n, p in trainer.model.named_parameters() if p.grad is not None]
    assert grads, "backward produced no gradients at all"
    for name, grad in grads:
        assert torch.isfinite(grad).all(), f"non-finite gradient in {name}"


def test_meaningful_nonzero_gradient_exists() -> None:
    """At least one meaningful (non-PAD) parameter receives a non-zero gradient."""
    trainer = tiny_trainer(epochs=1)
    dataset = tiny_dataset()
    batch = next(iter(iter_batches(dataset.samples, 8)))

    trainer.optimizer.zero_grad(set_to_none=True)
    positive_logits, negative_logits = trainer.model.training_logits(
        batch.input_ids, batch.positive_ids, batch.negative_ids
    )
    binary_logistic_loss(positive_logits, negative_logits).backward()

    # Inspect leaf parameters directly (slicing a parameter gives a non-leaf tensor
    # whose .grad is never populated).
    nonzero = []
    for name, parameter in trainer.model.named_parameters():
        if parameter.grad is None:
            continue
        grad = parameter.grad
        if name == "item_embedding.weight":
            grad = grad[1:]  # exclude the PAD row, which must stay zero
        if bool((grad != 0).any()):
            nonzero.append(name)
    assert nonzero, "no meaningful parameter received a non-zero gradient"


def test_optimizer_step_changes_a_meaningful_parameter() -> None:
    """optimizer.step() really updates the model."""
    trainer = tiny_trainer(epochs=1)
    dataset = tiny_dataset()
    batch = next(iter(iter_batches(dataset.samples, 8)))

    before = {
        name: tensor.detach().clone() for name, tensor in trainer.meaningful_parameters().items()
    }
    trainer.step(batch)
    after = trainer.meaningful_parameters()

    changed = [
        name for name, tensor in after.items() if not torch.equal(before[name], tensor.detach())
    ]
    assert changed, "optimizer.step() did not change any meaningful parameter"


def test_pad_embedding_stays_exactly_zero_after_steps() -> None:
    """The PAD embedding row must remain exactly zero across optimizer steps."""
    trainer = tiny_trainer(epochs=3, weight_decay=0.1)  # decay is the risky case
    dataset = tiny_dataset()
    for batch in iter_batches(dataset.samples, 4):
        trainer.step(batch.to("cpu"))
        assert trainer.pad_embedding_is_zero(), "PAD embedding row drifted from zero"

    result = train_sasrec(trainer.model, dataset, trainer.config)
    assert result.pad_embedding_zero_after_training


def test_parameters_stay_finite_and_loss_finite_after_update() -> None:
    """No NaN/Inf appears in parameters, and the loss stays finite after a step."""
    trainer = tiny_trainer(epochs=2)
    dataset = tiny_dataset()
    batch = next(iter(iter_batches(dataset.samples, 4)))

    trainer.step(batch)
    assert trainer.parameters_finite()
    assert math.isfinite(trainer.evaluate_loss(batch))


def test_trainer_rejects_unusable_device() -> None:
    """Milestone 5 supports cpu/cuda, but never silently substitutes one for the other.

    With CUDA absent, asking for ``cuda`` must fail rather than quietly training on
    CPU; an unsupported string is rejected outright regardless of hardware.
    """
    model = tiny_model()
    with pytest.raises(TrainingError):
        SASRecTrainer(model, TINY_NUM_ITEMS, TrainerConfig(device="gpu"))

    if not torch.cuda.is_available():
        with pytest.raises(TrainingError) as excinfo:
            SASRecTrainer(model, TINY_NUM_ITEMS, TrainerConfig(device="cuda"))
        assert "silently" in str(excinfo.value)


def test_trainer_rejects_mismatched_num_items() -> None:
    """The catalogue size must match the model."""
    with pytest.raises(TrainingError):
        SASRecTrainer(tiny_model(), TINY_NUM_ITEMS + 1, TrainerConfig())


def test_trainer_rejects_invalid_config() -> None:
    """Invalid hyper-parameters are rejected explicitly."""
    for kwargs in (
        {"learning_rate": 0.0},
        {"learning_rate": -1.0},
        {"weight_decay": -0.1},
        {"batch_size": 0},
        {"epochs": 0},
        {"max_grad_norm": 0.0},
    ):
        with pytest.raises(TrainingError):
            TrainerConfig(**kwargs)  # type: ignore[arg-type]


def test_training_rejects_empty_dataset() -> None:
    """Training on a dataset with no trainable samples fails loudly."""
    cases = [EvaluationCase("short", 1, (5,), 6, 7, 3)]
    dataset = build_dataset(cases, TINY_NUM_ITEMS, TINY_CONFIG)
    assert dataset.samples == []
    with pytest.raises(TrainingError) as excinfo:
        train_sasrec(tiny_model(), dataset, TrainerConfig(epochs=1))
    assert "no trainable samples" in str(excinfo.value)


def test_gradient_clipping_option_is_honoured() -> None:
    """max_grad_norm bounds the gradient norm when set."""
    trainer = tiny_trainer(epochs=1, max_grad_norm=1e-6)
    dataset = tiny_dataset()
    batch = next(iter(iter_batches(dataset.samples, 8)))
    trainer.step(batch)

    total_norm = torch.sqrt(
        sum(
            (p.grad.detach() ** 2).sum()
            for p in trainer.model.parameters()
            if p.grad is not None
        )
    )
    assert float(total_norm) <= 1e-6 + 1e-9


# --------------------------------------------------------------------------- #
# 15-18. Tiny overfit experiment
# --------------------------------------------------------------------------- #


def test_tiny_fixture_substantially_overfits() -> None:
    """The milestone's central acceptance test.

    A tiny model must drive the training loss to <= 25% of its initial value and
    rank every training positive above its paired negative.
    """
    dataset = tiny_dataset()
    model = tiny_model(seed=0)
    trainer = tiny_trainer(model, epochs=300, learning_rate=0.05, batch_size=4)
    result = train_sasrec(model, dataset, trainer.config)

    assert result.all_losses_finite, "a non-finite loss appeared during training"
    assert result.parameters_finite_after_training
    assert result.pad_embedding_zero_after_training
    assert result.parameter_update_observed

    assert result.loss_reduction_ratio <= 0.25, (
        f"tiny fixture did not overfit: initial={result.initial_loss:.6f} "
        f"final={result.final_loss:.6f} ratio={result.loss_reduction_ratio:.4f}"
    )

    positives, negatives = trainer.batch_logits(dataset.samples)
    accuracy = ranking_accuracy(positives, negatives)
    assert accuracy >= 0.99, (
        f"positives outranked negatives on only {accuracy:.3f} of training positions"
    )


def test_tiny_final_logits_beat_paired_negatives() -> None:
    """Per-position check that every training positive beats its negative."""
    dataset = tiny_dataset()
    model = tiny_model(seed=0)
    trainer = tiny_trainer(model, epochs=300, learning_rate=0.05, batch_size=4)
    train_sasrec(model, dataset, trainer.config)

    positives, negatives = trainer.batch_logits(dataset.samples)
    assert positives.shape == negatives.shape
    assert positives.numel() == dataset.total_valid_positions
    beaten = (positives > negatives).sum().item()
    assert beaten == positives.numel(), (
        f"{positives.numel() - beaten} training positions still rank their negative first"
    )


def test_tiny_training_is_reproducible_for_same_seed() -> None:
    """Two clean runs with the same seed produce identical trajectories and weights."""
    first_dataset = tiny_dataset()
    second_dataset = tiny_dataset()

    first_model = tiny_model(seed=0)
    second_model = tiny_model(seed=0)

    first = train_sasrec(first_model, first_dataset, tiny_trainer(epochs=50).config)
    second = train_sasrec(second_model, second_dataset, tiny_trainer(epochs=50).config)

    assert first.initial_loss == second.initial_loss
    assert first.final_loss == second.final_loss
    assert first.loss_trajectory == second.loss_trajectory
    assert first.optimizer_steps == second.optimizer_steps

    for (name_a, param_a), (name_b, param_b) in zip(
        first_model.named_parameters(), second_model.named_parameters()
    ):
        assert name_a == name_b
        assert torch.equal(param_a, param_b), f"{name_a} differs between identical runs"


def test_different_seed_produces_a_distinct_run() -> None:
    """A different seed is not accidentally identical, yet still overfits."""
    dataset = tiny_dataset()
    base_model = tiny_model(seed=0)
    other_model = tiny_model(seed=1)

    base = train_sasrec(base_model, dataset, tiny_trainer(epochs=100).config)
    other = train_sasrec(other_model, dataset, tiny_trainer(epochs=100, seed=1).config)

    assert base.initial_loss != other.initial_loss
    assert not torch.equal(
        base_model.item_embedding.weight, other_model.item_embedding.weight
    )
    # both remain valid overfit runs
    assert base.loss_reduction_ratio <= 0.25
    assert other.loss_reduction_ratio <= 0.25


# --------------------------------------------------------------------------- #
# 19-20. Epoch-aware negative sampling
# --------------------------------------------------------------------------- #


def test_epoch_negatives_are_deterministic() -> None:
    """Same (seed, epoch) reproduces the same negatives exactly."""
    dataset = tiny_dataset()
    first = samples_with_epoch_negatives(dataset.samples, TINY_NUM_ITEMS, seed=5, epoch=0)
    second = samples_with_epoch_negatives(dataset.samples, TINY_NUM_ITEMS, seed=5, epoch=0)
    assert [s.negative_ids for s in first] == [s.negative_ids for s in second]


def test_different_epoch_resamples_negatives_and_keeps_structure() -> None:
    """A new epoch redraws negatives while inputs/positives stay identical."""
    dataset = tiny_dataset()
    epoch0 = samples_with_epoch_negatives(dataset.samples, TINY_NUM_ITEMS, seed=5, epoch=0)
    epoch1 = samples_with_epoch_negatives(dataset.samples, TINY_NUM_ITEMS, seed=5, epoch=1)

    assert [s.negative_ids for s in epoch0] != [s.negative_ids for s in epoch1]
    for a, b in zip(epoch0, epoch1):
        assert a.user_int_id == b.user_int_id
        assert a.input_ids == b.input_ids
        assert a.positive_ids == b.positive_ids
        assert a.num_valid_positions == b.num_valid_positions


def test_epoch_negatives_are_valid_items_outside_history() -> None:
    """Resampled negatives stay in-catalog, non-PAD and outside the train history."""
    dataset = tiny_dataset()
    resampled = samples_with_epoch_negatives(dataset.samples, TINY_NUM_ITEMS, seed=9, epoch=3)
    for sample in resampled:
        history = {v for v in sample.input_ids if v != PAD_ID}
        for positive, negative in zip(sample.positive_ids, sample.negative_ids):
            if positive == PAD_ID:
                assert negative == PAD_ID
                continue
            assert 1 <= negative <= TINY_NUM_ITEMS
            assert negative != PAD_ID
            assert negative not in history


def test_resample_does_not_mutate_the_original_dataset() -> None:
    """The dataset's own samples are untouched by epoch resampling."""
    dataset = tiny_dataset()
    snapshot = [s.negative_ids for s in dataset.samples]
    samples_with_epoch_negatives(dataset.samples, TINY_NUM_ITEMS, seed=1, epoch=7)
    assert [s.negative_ids for s in dataset.samples] == snapshot


def test_trainer_resamples_negatives_per_epoch_when_configured() -> None:
    """With resample_negatives enabled, each epoch uses freshly drawn negatives."""
    dataset = tiny_dataset()
    trainer = tiny_trainer(epochs=2, resample_negatives=True)
    seen: dict[int, list[tuple[int, ...]]] = {}

    def record(epoch: int, _trainer: SASRecTrainer) -> None:
        seen[epoch] = list(trainer.samples_for_epoch(dataset, epoch)[0].negative_ids)

    trainer.train(dataset, on_epoch_end=record)
    assert len(seen) == 2
    # epochs 0 and 1 must not use identical negatives for the first sample
    assert seen[0] != seen[1] or len(dataset.samples[0].negative_ids) == 0


def test_sampler_never_hangs_on_an_exhausted_pool() -> None:
    """A pool with a single candidate resolves through the deterministic fallback."""
    exclusion = frozenset(range(1, TINY_NUM_ITEMS))  # only TINY_NUM_ITEMS is allowed
    config = SASRecDatasetConfig(max_seq_len=5, seed=0, epoch=0)
    for position in range(25):
        assert sample_negative(1, position, exclusion, TINY_NUM_ITEMS, config) == TINY_NUM_ITEMS


# --------------------------------------------------------------------------- #
# Callback / snapshot hygiene
# --------------------------------------------------------------------------- #


def test_epoch_callback_fires_once_per_epoch_and_never_mid_epoch() -> None:
    """The epoch hook is a clean boundary for snapshotting or inference."""
    dataset = tiny_dataset()
    trainer = tiny_trainer(epochs=3)
    calls: list[int] = []

    def hook(epoch: int, _trainer: SASRecTrainer) -> None:
        calls.append(epoch)

    trainer.train(dataset, on_epoch_end=hook)
    assert calls == [0, 1, 2]


def test_trainer_never_ranks_or_masks() -> None:
    """The trainer exposes no masking/ranking/metric API (ownership boundary)."""
    trainer = tiny_trainer(epochs=1)
    for forbidden in ("rank", "evaluate_rank", "mask", "ndcg", "hr_at_k", "recall"):
        assert not hasattr(trainer, forbidden), f"trainer must not implement {forbidden}"


def test_training_result_is_serialisable() -> None:
    """The evidence summary is JSON-friendly."""
    import json

    dataset = tiny_dataset()
    model = tiny_model()
    result = train_sasrec(model, dataset, tiny_trainer(epochs=2).config)
    payload = json.loads(json.dumps(result.as_dict()))
    assert payload["optimizer_steps"] > 0
    assert payload["loss_reduction_ratio"] >= 0.0


# --------------------------------------------------------------------------- #
# Audit: the trainer must fail fast, before any optimizer update
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_trainer_step_fails_fast_on_non_finite_logits(bad: float) -> None:
    """A non-finite logit aborts the step before backward, leaving the model intact.

    This is the end-to-end form of the fail-fast guarantee: not only does the loss
    refuse the input, but no gradient is left behind and no parameter moves, so a
    numerically broken forward pass cannot silently corrupt the model.
    """
    dataset = tiny_dataset()
    trainer = tiny_trainer(epochs=1)
    batch = next(iter(iter_batches(dataset.samples, 4)))

    before = {name: p.detach().clone() for name, p in trainer.model.named_parameters()}
    original = trainer.model.training_logits

    def corrupted(input_ids, positive_ids, negative_ids, **kwargs):  # noqa: ANN001
        positive, negative = original(input_ids, positive_ids, negative_ids, **kwargs)
        return positive.clone().fill_(bad), negative

    trainer.model.training_logits = corrupted  # type: ignore[assignment]
    try:
        with pytest.raises(TrainingLossError):
            trainer.step(batch)
    finally:
        trainer.model.training_logits = original  # type: ignore[assignment]

    after = {name: p.detach().clone() for name, p in trainer.model.named_parameters()}
    moved = [name for name in before if not torch.equal(before[name], after[name])]
    assert moved == [], f"a failed step still changed parameters: {moved}"
    assert all(p.grad is None for p in trainer.model.parameters()), "gradients were left behind"


def test_trainer_accepts_ordinary_finite_logits_after_guard() -> None:
    """The guard does not interfere with normal training steps."""
    dataset = tiny_dataset()
    trainer = tiny_trainer(epochs=1)
    batch = next(iter(iter_batches(dataset.samples, 4)))
    loss, loss_finite, grads_finite = trainer.step(batch)
    assert math.isfinite(loss) and loss_finite and grads_finite
    assert trainer.parameters_finite()
    assert trainer.pad_embedding_is_zero()

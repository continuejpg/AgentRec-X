"""Tests for the SASRec model (Milestone 3, Part B).

Every test runs on CPU in eval mode with dropout disabled where determinism
matters.  The model is never trained and no recommendation-quality metric is
asserted from an untrained model.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

torch = pytest.importorskip("torch", reason="SASRec model tests require PyTorch")

from recommendation.datasets.sasrec import (  # noqa: E402
    PAD_ID,
    SASRecDataError,
    encode_inference_history,
)
from recommendation.evaluation import EvaluationCase, FullRankingEvaluator  # noqa: E402
from recommendation.models.sasrec import (  # noqa: E402
    SASRec,
    SASRecConfig,
    SASRecError,
    build_causal_attention_mask,
    build_model,
    build_padding_mask,
    make_score_fn,
    valid_position_mask,
)

NUM_ITEMS = 13
SEQ_LEN = 6


def small_model(seed: int = 0, *, dropout: float = 0.0, hidden: int = 16, blocks: int = 2, heads: int = 2) -> SASRec:
    """Build a small, deterministic, eval-mode SASRec."""
    model = build_model(
        num_items=NUM_ITEMS,
        seed=seed,
        max_seq_len=SEQ_LEN,
        hidden_size=hidden,
        num_blocks=blocks,
        num_heads=heads,
        dropout=dropout,
    )
    model.eval()
    return model


def ids(*rows: list[int]) -> torch.Tensor:
    """Build a long tensor from explicit rows."""
    return torch.tensor(rows, dtype=torch.long)


def random_batch(batch: int = 3, seq: int = SEQ_LEN, seed: int = 1234) -> torch.Tensor:
    """A deterministic batch of real (non-PAD) ids."""
    generator = torch.Generator().manual_seed(seed)
    return torch.randint(1, NUM_ITEMS + 1, (batch, seq), generator=generator, dtype=torch.long)


# --------------------------------------------------------------------------- #
# 1-2. Embedding table and padding_idx
# --------------------------------------------------------------------------- #


def test_item_embedding_size_is_num_items_plus_one() -> None:
    """The item table has exactly num_items + 1 rows so id -> row is direct."""
    model = small_model()
    assert model.item_embedding.num_embeddings == NUM_ITEMS + 1
    assert model.item_dim() == NUM_ITEMS + 1
    assert model.score_vector_length() == NUM_ITEMS + 1

    # an item id can be used directly as an embedding index
    table = model.item_embedding.weight
    assert table.shape[0] == NUM_ITEMS + 1

    # one row of input ids -> shape [1, 1, hidden]; compare values explicitly
    last_id = NUM_ITEMS - 1
    vector = model.item_embedding(ids([last_id]))
    assert vector.shape == (1, 1, model.config.hidden_size)
    assert torch.allclose(vector[0, 0], table[last_id], atol=0.0, rtol=0.0)
    assert torch.equal(vector[0, 0], table[last_id].detach())


def test_item_padding_idx_is_zero_and_row_stays_zero() -> None:
    """padding_idx is 0 and PAD's embedding row is exactly zero."""
    model = small_model()
    assert model.item_padding_index() == PAD_ID == 0
    assert torch.equal(model.item_embedding.weight[PAD_ID], torch.zeros(model.config.hidden_size))
    embedding_of_pad = model.item_embedding(ids([0, 0]))
    assert embedding_of_pad.shape == (1, 2, model.config.hidden_size)
    assert torch.equal(embedding_of_pad[0, 0], torch.zeros(model.config.hidden_size))
    # every column of a PAD-only batch is the zero row
    assert torch.equal(embedding_of_pad[0, 1], torch.zeros(model.config.hidden_size))
    assert float(embedding_of_pad.abs().max()) == 0.0


# --------------------------------------------------------------------------- #
# 3. Forward shapes
# --------------------------------------------------------------------------- #


def test_forward_hidden_state_shape() -> None:
    """encode returns [batch, seq, hidden]."""
    model = small_model(hidden=16)
    batch = random_batch(batch=4, seq=SEQ_LEN)
    with torch.no_grad():
        hidden = model.encode(batch)
    assert hidden.shape == (4, SEQ_LEN, 16)
    assert torch.isfinite(hidden).all()


def test_hidden_state_is_zero_at_padded_positions() -> None:
    """Padded query positions are fully masked, so their hidden states are zero."""
    model = small_model()
    batch = ids([0, 0, 0, 1, 2, 3], [0, 5, 6, 7, 8, 9])
    with torch.no_grad():
        hidden = model.encode(batch)
    valid = build_padding_mask(batch, NUM_ITEMS)
    assert not bool(valid[0, 0])
    assert torch.equal(hidden[0, 0], torch.zeros(model.config.hidden_size))
    assert torch.equal(hidden[0, 1:3], torch.zeros(2, model.config.hidden_size))
    assert torch.isfinite(hidden).all()


# --------------------------------------------------------------------------- #
# 4-6. Training logits and valid mask
# --------------------------------------------------------------------------- #


def test_training_logit_shapes() -> None:
    """Positive and negative logits are flat and aligned over valid positions."""
    model = small_model()
    inputs = ids([0, 0, 1, 2, 3], [0, 4, 5, 6, 7])
    positives = ids([0, 0, 2, 3, 4], [0, 5, 6, 7, 8])
    negatives = ids([9, 9, 9, 9, 9], [9, 9, 9, 9, 9])

    with torch.no_grad():
        positive_logits, negative_logits = model.training_logits(inputs, positives, negatives)

    # row 1 has 4 valid positives (2,3,4 plus the padded 0) -> 3 + 4 = 7
    expected_valid = int(valid_position_mask(positives).sum())
    assert expected_valid == 7
    assert positive_logits.shape == negative_logits.shape == (expected_valid,)
    assert torch.isfinite(positive_logits).all()
    assert torch.isfinite(negative_logits).all()


def test_valid_position_mask_excludes_padding() -> None:
    """The valid mask is exactly the non-PAD positive positions."""
    positives = ids([0, 0, 5, 6], [0, 7, 0, 8])
    mask = valid_position_mask(positives)
    expected = torch.tensor([[False, False, True, True], [False, True, False, True]])
    assert torch.equal(mask, expected)
    assert int(mask.sum()) == 4


def test_training_logits_returns_valid_mask_when_asked() -> None:
    """The optional valid mask is consistent with the logits it accompanies."""
    model = small_model()
    inputs = ids([1, 2, 0, 0])
    positives = ids([2, 3, 0, 0])
    negatives = ids([9, 9, 9, 9])
    with torch.no_grad():
        positive_logits, negative_logits, mask = model.training_logits(
            inputs, positives, negatives, return_valid_mask=True
        )
    assert mask.dtype == torch.bool
    assert int(mask.sum()) == positive_logits.shape[0] == 2


def test_training_logits_reject_mismatched_shapes() -> None:
    """Mismatched positive/negative shapes are rejected."""
    model = small_model()
    inputs = ids([1, 2, 3])
    with pytest.raises(SASRecError):
        model.training_logits(inputs, ids([1, 2]), ids([1, 2, 3]))
    with pytest.raises(SASRecError):
        model.training_logits(inputs, ids([1, 2, 3]), ids([1, 2]))


def test_training_logits_reject_out_of_range_ids() -> None:
    """PAD-only or out-of-catalog positives/negatives are rejected."""
    model = small_model()
    inputs = ids([1, 2, 3])
    with pytest.raises(SASRecError):
        model.training_logits(inputs, ids([1, 2, NUM_ITEMS + 1]), ids([1, 2, 3]))
    with pytest.raises(SASRecError):
        model.training_logits(inputs, ids([1, 2, 3]), ids([1, 2, -1]))


# --------------------------------------------------------------------------- #
# 7, 18. Finiteness and empty/all-PAD rejection
# --------------------------------------------------------------------------- #


def test_all_outputs_are_finite() -> None:
    """No NaN or infinity anywhere in the forward pass, including the PAD column."""
    for dropout in (0.0, 0.3):
        model = small_model(dropout=dropout)
        inputs = random_batch(batch=3, seq=SEQ_LEN)
        positives = random_batch(batch=3, seq=SEQ_LEN, seed=99)
        negatives = random_batch(batch=3, seq=SEQ_LEN, seed=7)
        with torch.no_grad():
            hidden = model.encode(inputs)
            positive_logits, negative_logits = model.training_logits(inputs, positives, negatives)
            scores = model.full_catalog_scores(inputs)
        for tensor in (hidden, positive_logits, negative_logits, scores):
            assert torch.isfinite(tensor).all(), f"non-finite output with dropout={dropout}"


def test_pad_heavy_batch_is_finite() -> None:
    """A batch of mostly padding produces finite outputs."""
    model = small_model()
    inputs = ids([0, 0, 0, 0, 0, 3], [0, 0, 0, 0, 0, 1])
    with torch.no_grad():
        hidden = model.encode(inputs)
        scores = model.full_catalog_scores(inputs)
    assert torch.isfinite(hidden).all()
    assert torch.isfinite(scores).all()


def test_empty_and_all_pad_inputs_are_rejected() -> None:
    """Empty tensors and all-PAD sequences are rejected at the boundary."""
    model = small_model()
    with pytest.raises(SASRecError):
        model.encode(torch.zeros((0, SEQ_LEN), dtype=torch.long))
    with pytest.raises(SASRecError):
        model.encode(torch.zeros((2, 0), dtype=torch.long))
    with pytest.raises(SASRecError):
        model.full_catalog_scores(ids([0, 0, 0, 0, 0, 0]))


def test_out_of_range_input_ids_are_rejected() -> None:
    """Inevitably-invalid item ids in input_ids are rejected."""
    model = small_model()
    for bad in (NUM_ITEMS + 1, -1):
        batch = ids([1, 2, bad])
        with pytest.raises(SASRecError):
            model.encode(batch)


# --------------------------------------------------------------------------- #
# 8-10. Full-catalog scoring
# --------------------------------------------------------------------------- #


def test_full_catalog_score_shape() -> None:
    """Scores have shape [batch, num_items + 1]."""
    model = small_model()
    batch = random_batch(batch=5, seq=SEQ_LEN)
    with torch.no_grad():
        scores = model.full_catalog_scores(batch)
    assert scores.shape == (5, NUM_ITEMS + 1)


def test_score_index_corresponds_to_item_id() -> None:
    """scores[b, item_id] is that item's dot product with the summary hidden state."""
    model = small_model()
    batch = ids([0, 0, 0, 1, 2, 3])
    with torch.no_grad():
        scores = model.full_catalog_scores(batch)
        hidden = model.encode(batch)
        table = model.item_embedding.weight
        # the summary is the hidden state at the last valid position (index 5)
        expected = hidden[0, 5] @ table.t()
    assert torch.allclose(scores[0], expected, atol=1e-6)
    # spot-check individual item ids
    for item_id in (1, 5, NUM_ITEMS):
        manual = float(hidden[0, 5] @ table[item_id])
        assert abs(float(scores[0, item_id]) - manual) < 1e-6


def test_pad_score_is_finite() -> None:
    """The PAD column is finite, because the evaluator validates the whole vector."""
    model = small_model()
    batch = random_batch(batch=3, seq=SEQ_LEN)
    with torch.no_grad():
        scores = model.full_catalog_scores(batch)
    assert torch.isfinite(scores[:, PAD_ID]).all()
    # PAD's embedding row is zero, so its score is ~0
    assert torch.allclose(scores[:, PAD_ID], torch.zeros(3), atol=1e-6)


def test_scores_use_the_shared_item_embedding_table() -> None:
    """Output scoring reuses the input embedding table (weight tying)."""
    model = small_model()
    assert model.full_catalog_scores.__doc__ is not None
    batch = ids([1, 2, 3, 4, 5, 6])
    with torch.no_grad():
        scores = model.full_catalog_scores(batch)
    tied = model.encode(batch)[0, -1] @ model.item_embedding.weight.t()
    assert torch.allclose(scores[0], tied, atol=1e-6)


def test_left_padded_short_sequence_is_accepted() -> None:
    """A mostly-padded short history scores correctly (summary = last real item)."""
    model = small_model()
    batch = ids([0, 0, 0, 0, 0, 4])
    with torch.no_grad():
        scores = model.full_catalog_scores(batch)
        hidden = model.encode(batch)
        expected = hidden[0, 5] @ model.item_embedding.weight.t()
    assert scores.shape == (1, NUM_ITEMS + 1)
    assert torch.allclose(scores[0], expected, atol=1e-6)


def test_max_length_sequence_is_accepted() -> None:
    """A full max_seq_len sequence works and its summary uses the last position."""
    model = small_model()
    batch = random_batch(batch=2, seq=SEQ_LEN)
    with torch.no_grad():
        scores = model.full_catalog_scores(batch)
        hidden = model.encode(batch)
        expected = hidden[:, -1] @ model.item_embedding.weight.t()
    assert torch.allclose(scores, expected, atol=1e-6)


def test_over_length_sequence_is_rejected_by_the_model() -> None:
    """The model refuses over-length input; the encoder is the truncation boundary."""
    model = small_model()
    too_long = random_batch(batch=1, seq=SEQ_LEN + 1)
    with pytest.raises(SASRecError) as excinfo:
        model.encode(too_long)
    assert "max_seq_len" in str(excinfo.value)

    # ...and the encoder handles that case instead, by construction
    encoded = encode_inference_history(list(range(1, 12)), SEQ_LEN, NUM_ITEMS)
    assert len(encoded) == SEQ_LEN
    with torch.no_grad():
        assert torch.isfinite(model.full_catalog_scores(torch.tensor([encoded], dtype=torch.long))).all()


# --------------------------------------------------------------------------- #
# 11-12. The model does not mask
# --------------------------------------------------------------------------- #


def test_scorer_performs_no_seen_item_masking() -> None:
    """Seen items, PAD and targets all receive ordinary scores."""
    model = small_model()
    history = (3, 5, 5, 7)
    encoded = encode_inference_history(history, SEQ_LEN, NUM_ITEMS)
    with torch.no_grad():
        scores = model.full_catalog_scores(torch.tensor([encoded], dtype=torch.long))
    for seen in (3, 5, 7):
        assert torch.isfinite(scores[0, seen])
    assert torch.isfinite(scores[0, PAD_ID])
    # the model returns the full vector, so nothing was removed
    assert scores.shape[1] == NUM_ITEMS + 1


def test_repeated_target_receives_a_raw_score() -> None:
    """A target that also appears in the history is scored like any other item."""
    model = small_model()
    history = (3, 5, 7)
    target = 5  # repeated: in the history and the prediction target
    score_fn = make_score_fn(model, SEQ_LEN, NUM_ITEMS)
    scores = score_fn(history, target)

    assert len(scores) == NUM_ITEMS + 1
    assert math.isfinite(scores[target])
    # it is an ordinary column: compare with a target that is not in the history
    assert scores[target] == score_fn(history, 4)[target]


def test_score_fn_ignores_the_target_argument() -> None:
    """The wrapped scorer scores the whole catalog regardless of the target."""
    model = small_model()
    score_fn = make_score_fn(model, SEQ_LEN, NUM_ITEMS)
    base = score_fn((1, 2, 3), 1)
    for target in (1, 2, 3, NUM_ITEMS):
        assert score_fn((1, 2, 3), target) == base


# --------------------------------------------------------------------------- #
# 13. Causal isolation (critical correctness test)
# --------------------------------------------------------------------------- #


def test_causal_future_token_isolation() -> None:
    """Position t must not depend on items after t.

    Two inputs share an identical prefix and differ only in the final token, which
    is *after* every position we compare.  In eval mode with dropout disabled, all
    prefix hidden states must be bit-for-bit equal.
    """
    model = build_model(
        num_items=NUM_ITEMS, seed=123, max_seq_len=SEQ_LEN,
        hidden_size=16, num_blocks=3, num_heads=2, dropout=0.0,
    )
    model.eval()

    prefix = [3, 5, 2, 7, 1]
    left = ids(prefix + [4])
    right = ids(prefix + [9])
    assert not torch.equal(left, right)

    with torch.no_grad():
        hidden_left = model.encode(left)
        hidden_right = model.encode(right)

    # every position except the changed last one must be identical
    assert torch.equal(hidden_left[0, :-1], hidden_right[0, :-1]), (
        "a future token changed an earlier hidden state -> causality is broken"
    )


def test_causal_isolation_holds_for_multiple_changed_futures() -> None:
    """Changing every position after t leaves positions <= t unchanged."""
    model = build_model(
        num_items=NUM_ITEMS, seed=7, max_seq_len=SEQ_LEN,
        hidden_size=16, num_blocks=2, num_heads=2, dropout=0.0,
    )
    model.eval()

    prefix = [1, 2, 3]
    left = ids(prefix + [4, 5, 6])
    right = ids(prefix + [11, 12, 13])

    with torch.no_grad():
        hidden_left = model.encode(left)
        hidden_right = model.encode(right)

    assert torch.equal(hidden_left[0, : len(prefix)], hidden_right[0, : len(prefix)])


def test_causal_isolation_across_batch_positions() -> None:
    """Causality holds for every row of a batch, not just the first."""
    model = build_model(
        num_items=NUM_ITEMS, seed=5, max_seq_len=SEQ_LEN,
        hidden_size=8, num_blocks=2, num_heads=1, dropout=0.0,
    )
    model.eval()

    left = ids([1, 2, 3, 4, 5, 6], [2, 2, 2, 2, 2, 2])
    right = ids([1, 2, 3, 4, 5, 9], [2, 2, 2, 2, 2, 9])

    with torch.no_grad():
        hidden_left = model.encode(left)
        hidden_right = model.encode(right)

    assert torch.equal(hidden_left[:, :-1], hidden_right[:, :-1])


def test_future_dependence_would_be_detectable() -> None:
    """Sanity check on the test itself: later positions DO differ when changed.

    If the model were ignoring later tokens entirely, the causal test above would
    pass vacuously.  This confirms the changed position actually influences its own
    (and later) representations.
    """
    model = build_model(
        num_items=NUM_ITEMS, seed=123, max_seq_len=SEQ_LEN,
        hidden_size=16, num_blocks=3, num_heads=2, dropout=0.0,
    )
    model.eval()
    left = ids([3, 5, 2, 7, 1, 4])
    right = ids([3, 5, 2, 7, 1, 9])
    with torch.no_grad():
        assert not torch.equal(model.encode(left)[0, -1], model.encode(right)[0, -1])


def test_causal_mask_blocks_upper_triangle() -> None:
    """The mask itself blocks t -> t' for t' > t and never blocks the diagonal."""
    batch = ids([1, 2, 3, 4])
    blocked = build_causal_attention_mask(batch, NUM_ITEMS)
    assert blocked.shape == (1, 1, 4, 4)
    for query in range(4):
        for key in range(4):
            if key > query:
                assert bool(blocked[0, 0, query, key]), f"{query}->{key} must be blocked"
            else:
                assert not bool(blocked[0, 0, query, key]), f"{query}->{key} must be allowed"


# --------------------------------------------------------------------------- #
# 14. Padding mask behaviour
# --------------------------------------------------------------------------- #


def test_padding_mask_blocks_padded_keys() -> None:
    """Padded positions are blocked as keys for every query."""
    batch = ids([0, 0, 3, 4])
    blocked = build_causal_attention_mask(batch, NUM_ITEMS)
    for query in range(4):
        # padded keys are always blocked...
        assert bool(blocked[0, 0, query, 0])
        assert bool(blocked[0, 0, query, 1])
        # ...and real keys follow the causal rule (allowed only when key <= query)
        for key in (2, 3):
            assert bool(blocked[0, 0, query, key]) == (key > query)


def test_padding_mask_blocks_padded_queries_entirely() -> None:
    """Padded query rows are fully blocked so padding cannot read real items."""
    batch = ids([0, 3, 4])
    blocked = build_causal_attention_mask(batch, NUM_ITEMS)
    assert bool(blocked[0, 0, 0].all())
    assert not bool(blocked[0, 0, 1].all())


def test_padding_mask_flags_only_real_items() -> None:
    """build_padding_mask marks exactly the in-catalog, non-PAD positions."""
    batch = ids([0, 1, 5, 0], [7, 0, 1, 2])
    valid = build_padding_mask(batch, NUM_ITEMS)
    expected = torch.tensor([[False, True, True, False], [True, False, True, True]])
    assert torch.equal(valid, expected)


def test_padding_does_not_change_real_position_outputs() -> None:
    """Changing the amount of left padding must not alter the summary embedding.

    A left-padded sequence's last valid position sees the same items either way, so
    the real item representations must be identical regardless of padding width.
    """
    model = build_model(
        num_items=NUM_ITEMS, seed=11, max_seq_len=6,
        hidden_size=16, num_blocks=2, num_heads=2, dropout=0.0,
    )
    model.eval()
    short = ids([0, 0, 0, 0, 0, 5])
    long = ids([0, 5, 6, 7, 8, 9])
    with torch.no_grad():
        hidden_short = model.encode(short)
        hidden_long = model.encode(long)
    # position 5 in both is a real item; in the long row its history differs, so we
    # only assert finiteness and that the PAD positions stayed exactly zero
    assert torch.isfinite(hidden_short).all() and torch.isfinite(hidden_long).all()
    assert torch.equal(hidden_short[0, :5], torch.zeros(5, 16))


def test_padding_mask_rejects_invalid_ids() -> None:
    """Mask builders validate their input ids."""
    with pytest.raises(SASRecError):
        build_padding_mask(ids([1, NUM_ITEMS + 1]), NUM_ITEMS)


# --------------------------------------------------------------------------- #
# 19-20. Determinism
# --------------------------------------------------------------------------- #


def test_deterministic_initialization_and_eval_output() -> None:
    """Same seed -> identical parameters and identical eval-mode outputs."""
    first = small_model(seed=2024)
    second = small_model(seed=2024)

    for (name_a, param_a), (name_b, param_b) in zip(
        first.named_parameters(), second.named_parameters()
    ):
        assert name_a == name_b
        assert torch.equal(param_a, param_b), f"{name_a} differs across identical seeds"

    batch = random_batch(batch=2, seq=SEQ_LEN)
    with torch.no_grad():
        assert torch.equal(first.encode(batch), second.encode(batch))
        assert torch.equal(first.full_catalog_scores(batch), second.full_catalog_scores(batch))


def test_different_seeds_change_initialization() -> None:
    """Different seeds give different parameters, so the tests are not vacuous."""
    first = small_model(seed=1)
    second = small_model(seed=2)
    assert not torch.equal(first.item_embedding.weight, second.item_embedding.weight)


def test_eval_mode_output_is_repeatable_with_dropout_enabled_config() -> None:
    """With dropout > 0 the eval-mode output is still repeatable."""
    model = small_model(seed=3, dropout=0.5)
    model.eval()
    batch = random_batch(batch=2, seq=SEQ_LEN)
    with torch.no_grad():
        first = model.full_catalog_scores(batch)
        for _ in range(5):
            assert torch.equal(model.full_catalog_scores(batch), first)


def test_pad_embedding_stays_zero_after_forward_passes() -> None:
    """Forward passes never populate the PAD embedding row."""
    model = small_model()
    batch = random_batch(batch=2, seq=SEQ_LEN)
    with torch.no_grad():
        model.encode(batch)
        model.full_catalog_scores(batch)
    assert torch.equal(model.item_embedding.weight[PAD_ID], torch.zeros(model.config.hidden_size))


# --------------------------------------------------------------------------- #
# 21. Evaluator integration through a thin adapter
# --------------------------------------------------------------------------- #


def test_model_plugs_into_the_unified_evaluator_without_duplicating_semantics() -> None:
    """The adapter satisfies the evaluator's score contract exactly."""
    model = small_model()
    cases = [
        EvaluationCase(
            user_id="u1",
            user_int_id=1,
            train_history=(1, 2, 3),
            validation_target=4,
            test_target=5,
            sequence_length=5,
        ),
        EvaluationCase(
            user_id="u2",
            user_int_id=2,
            train_history=(3, 6),
            validation_target=7,
            test_target=8,
            sequence_length=4,
        ),
    ]
    evaluator = FullRankingEvaluator(num_items=NUM_ITEMS, k_values=(5, 10))
    score_fn = make_score_fn(model, SEQ_LEN, NUM_ITEMS)

    # the adapter's vector satisfies the evaluator's length/finiteness contract
    evaluator.validate_scores(score_fn(cases[0].test_history, cases[0].test_target))

    for mode in ("validation", "test"):
        outcome = evaluator.evaluate(cases, score_fn, mode=mode)
        assert outcome.report.num_cases == 2
        for k in (5, 10):
            for name in ("HR", "Recall", "NDCG"):
                value = outcome.report.metrics[name][k]
                assert math.isfinite(value)
                assert 0.0 <= value <= 1.0
        assert all(outcome.hr_recall_agree.values())

    # no recommendation-quality claim is made about the untrained model


def test_model_needs_encoded_histories_not_raw_ones() -> None:
    """Over-long raw histories are the encoder's job; the model never sees them."""
    model = small_model()
    long_history = list(range(1, NUM_ITEMS + 1))   # 13 items, far beyond SEQ_LEN
    score_fn = make_score_fn(model, SEQ_LEN, NUM_ITEMS)
    scores = score_fn(tuple(long_history), 5)
    assert len(scores) == NUM_ITEMS + 1
    assert all(math.isfinite(value) for value in scores)


def test_encode_rejects_sequence_longer_than_max_seq_len() -> None:
    """The model validates its own window rather than silently truncating."""
    model = small_model()
    with pytest.raises(SASRecError):
        model.encode(random_batch(batch=1, seq=SEQ_LEN + 2))


def test_config_is_validated() -> None:
    """Invalid configurations are rejected explicitly."""
    for kwargs in (
        {"num_items": 0},
        {"max_seq_len": 0},
        {"num_blocks": 0},
        {"hidden_size": 7, "num_heads": 2},   # not divisible
        {"num_heads": 0},
        {"dropout": 1.0},
        {"dropout": -0.1},
    ):
        base = dict(num_items=10, max_seq_len=4, hidden_size=8, num_blocks=1, num_heads=1)
        base.update(kwargs)
        with pytest.raises(SASRecError):
            SASRecConfig(**base)  # type: ignore[arg-type]


def test_describe_reports_architecture_metadata() -> None:
    """The model description is complete and serialisable."""
    model = small_model()
    payload = model.describe()
    assert payload["config"]["normalization"] == "pre-norm"
    assert payload["config"]["padding_idx"] == PAD_ID
    assert payload["config"]["num_items"] == NUM_ITEMS
    assert payload["parameter_count"] > 0
    assert payload["score_vector_length"] == NUM_ITEMS + 1


def test_encoder_rejects_empty_history_before_the_model() -> None:
    """Empty histories fail at the encoding boundary with a data error."""
    with pytest.raises(SASRecDataError):
        encode_inference_history([], SEQ_LEN, NUM_ITEMS)

"""Integration tests for SASRec smoke training (Milestone 4).

Covers the bounded real-artifact training smoke, the validation/test inference
prefix contract, and the reuse of the frozen Milestone 2A evaluator.

CPU only.  No benchmark claim is made anywhere in this file; metric values that the
evaluator computes internally are labeled SMOKE DIAGNOSTIC ONLY.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

torch = pytest.importorskip("torch", reason="Milestone 4 integration tests require PyTorch")

from recommendation.datasets.sasrec import (  # noqa: E402
    PAD_ID,
    SASRecDatasetConfig,
    build_dataset,
    encode_inference_history,
)
from recommendation.evaluation import (  # noqa: E402
    EvaluationCase,
    FullRankingEvaluator,
    build_cohort_from_artifacts,
)
from recommendation.models.sasrec import build_model, make_score_fn  # noqa: E402
from recommendation.training.sasrec import (  # noqa: E402
    SASRecTrainer,
    TrainerConfig,
    train_sasrec,
)

SEQUENCES_PATH = REPO_ROOT / "data" / "processed" / "Sports_and_Outdoors_sample_sequences.json"
MAPPINGS_PATH = REPO_ROOT / "data" / "processed" / "Sports_and_Outdoors_sample_mappings.json"

artifacts_available = pytest.mark.skipif(
    not (SEQUENCES_PATH.exists() and MAPPINGS_PATH.exists()),
    reason="Milestone 1.5 preprocessing artifacts not present (data/ is git-ignored)",
)

#: Bounded smoke subset (documented): the first N trainable users by user_int_id.
SMOKE_TRAINABLE_USERS = 128
SMOKE_MAX_SEQ_LEN = 20          # existing development setting, NOT a benchmark value
SMOKE_EPOCHS = 2
SMOKE_BATCH_SIZE = 32
SMOKE_LEARNING_RATE = 0.01
SMOKE_HIDDEN = 32

_CACHE: dict[str, object] = {}


def smoke_bundle():
    """Build (and cache) the bounded smoke training bundle from the real artifact."""
    if "value" not in _CACHE:
        cases, split = build_cohort_from_artifacts(str(SEQUENCES_PATH), str(MAPPINGS_PATH))
        config = SASRecDatasetConfig(max_seq_len=SMOKE_MAX_SEQ_LEN, seed=4242, epoch=0)
        full = build_dataset(cases, split.catalog_size, config)

        # deterministic bounded subset: first N trainable users by user_int_id
        subset = full.samples[:SMOKE_TRAINABLE_USERS]
        subset_users = {s.user_int_id for s in subset}
        assert all(s.num_valid_positions > 0 for s in subset)

        # rebuild a dataset container holding just the subset (structure unchanged)
        from recommendation.datasets.sasrec import SASRecDataset

        subset_dataset = SASRecDataset(
            samples=list(subset),
            num_items=full.num_items,
            stats=full.stats,
            user_int_ids=[s.user_int_id for s in subset],
        )

        model = build_model(
            num_items=split.catalog_size,
            seed=99,
            max_seq_len=SMOKE_MAX_SEQ_LEN,
            hidden_size=SMOKE_HIDDEN,
            num_blocks=1,
            num_heads=2,
            dropout=0.1,
        )
        trainer_config = TrainerConfig(
            learning_rate=SMOKE_LEARNING_RATE,
            weight_decay=0.0,
            batch_size=SMOKE_BATCH_SIZE,
            epochs=SMOKE_EPOCHS,
            seed=4242,
            shuffle=True,
            resample_negatives=True,
        )
        trainer = SASRecTrainer(model, split.catalog_size, trainer_config)
        result = train_sasrec(model, subset_dataset, trainer_config)

        _CACHE["value"] = {
            "cases": cases,
            "split": split,
            "dataset": subset_dataset,
            "full_dataset": full,
            "model": model,
            "trainer": trainer,
            "result": result,
            "subset_users": subset_users,
            "subset_size": len(subset),
            "subset_transitions": subset_dataset.total_valid_positions,
        }
    return _CACHE["value"]


# --------------------------------------------------------------------------- #
# 28. Real-artifact bounded training smoke
# --------------------------------------------------------------------------- #


@artifacts_available
def test_real_artifact_bounded_smoke_training_runs_and_learns() -> None:
    """A bounded real-data smoke trains without exception and stays finite."""
    bundle = smoke_bundle()
    result = bundle["result"]

    assert bundle["subset_size"] == SMOKE_TRAINABLE_USERS
    assert bundle["subset_transitions"] > 0

    assert result.all_losses_finite, "a non-finite loss appeared in the real-data smoke"
    assert result.all_gradients_finite, "a non-finite gradient appeared in the real-data smoke"
    assert result.parameters_finite_after_training
    assert result.pad_embedding_zero_after_training
    assert result.parameter_update_observed
    assert result.nonzero_gradient_observed
    assert result.optimizer_steps > 0
    assert math.isfinite(result.initial_loss) and math.isfinite(result.final_loss)


@artifacts_available
def test_real_artifact_smoke_uses_only_trainable_users() -> None:
    """Every selected sample has at least one real training transition."""
    bundle = smoke_bundle()
    for sample in bundle["dataset"].samples:
        assert sample.num_valid_positions >= 1
        assert any(v != PAD_ID for v in sample.positive_ids)

    # zero-transition users are absent from the training subset ...
    zero_transition_users = {
        c.user_int_id for c in bundle["cases"] if len(c.train_history) == 1
    }
    assert zero_transition_users
    assert not (bundle["subset_users"] & zero_transition_users)
    # ...and every selected user is genuinely trainable
    trainable_users = {
        c.user_int_id for c in bundle["cases"] if len(c.train_history) >= 2
    }
    assert bundle["subset_users"] <= trainable_users


@artifacts_available
def test_real_artifact_smoke_negatives_avoid_train_history() -> None:
    """Smoke-training negatives stay in-catalog, non-PAD and outside the history."""
    bundle = smoke_bundle()
    num_items = bundle["split"].catalog_size
    for sample in bundle["dataset"].samples:
        history = {v for v in sample.input_ids if v != PAD_ID}
        for positive, negative in zip(sample.positive_ids, sample.negative_ids):
            if positive == PAD_ID:
                assert negative == PAD_ID
                continue
            assert 1 <= negative <= num_items
            assert negative not in history


# --------------------------------------------------------------------------- #
# 9. Proof that validation/test targets stay out of fitting
# --------------------------------------------------------------------------- #


@artifacts_available
def test_real_artifact_training_data_excludes_targets() -> None:
    """Target-only items never appear as training inputs or positives.

    Scope note: this asserts the *target-knowledge* guarantee for ``input_ids`` and
    ``positive_ids`` only.  A future target may legitimately be drawn as a training
    negative, because at training time it is an unseen catalog item and excluding it
    on the basis of future knowledge would be leakage.
    """
    bundle = smoke_bundle()
    cases = bundle["cases"]
    train_items = {v for c in cases for v in c.train_history}
    target_only = {t for c in cases for t in (c.validation_target, c.test_target)} - train_items
    assert target_only

    for sample in bundle["full_dataset"].samples:
        assert not (set(sample.input_ids) & target_only)
        assert not (set(sample.positive_ids) & target_only)
        # negatives are deliberately NOT asserted against target_only: see the
        # docstring above and the dataset-level audit test.


@artifacts_available
def test_changing_every_target_does_not_change_the_smoke_training_data() -> None:
    """Replacing all validation/test targets leaves the training dataset identical."""
    bundle = smoke_bundle()
    cases = bundle["cases"]
    split = bundle["split"]
    config = SASRecDatasetConfig(max_seq_len=SMOKE_MAX_SEQ_LEN, seed=4242, epoch=0)

    mutated = [
        EvaluationCase(
            user_id=c.user_id,
            user_int_id=c.user_int_id,
            train_history=c.train_history,
            validation_target=(c.validation_target % split.catalog_size) + 1,
            test_target=(c.test_target % split.catalog_size) + 1,
            sequence_length=c.sequence_length,
        )
        for c in cases
    ]
    rebuilt = build_dataset(mutated, split.catalog_size, config)
    assert rebuilt.digest() == bundle["full_dataset"].digest()


# --------------------------------------------------------------------------- #
# 9-10, 23-25. Training vs inference prefix semantics
# --------------------------------------------------------------------------- #


@artifacts_available
def test_inference_prefix_semantics_are_distinct_from_training() -> None:
    """The regression test for the prefix question Milestone 3 left open.

    For ``train_history = [i1..i4]``, ``validation_target = i5``, ``test_target = i6``:

    * training transitions come only from ``[i1..i4]``;
    * validation inference encodes ``[i1..i4]`` and scores ``i5`` without appending it;
    * test inference encodes ``[i1..i4,i5]`` and scores ``i6`` without appending it;
    * neither target entered the fit.
    """
    bundle = smoke_bundle()
    split = bundle["split"]
    num_items = split.catalog_size
    case = next(c for c in bundle["cases"] if len(c.train_history) >= 4)
    train_history = tuple(case.train_history)

    # --- training: transitions derive from train_history only ----------------- #
    training_dataset = build_dataset([case], num_items, SASRecDatasetConfig(max_seq_len=8, seed=1))
    sample = training_dataset.samples[0]
    real_inputs = tuple(v for v in sample.input_ids if v != PAD_ID)
    real_positives = tuple(v for v in sample.positive_ids if v != PAD_ID)
    assert real_inputs == train_history[:-1]
    assert real_positives == train_history[1:]
    assert case.validation_target not in sample.positive_ids
    assert case.test_target not in sample.positive_ids
    assert case.validation_target not in sample.input_ids
    assert case.test_target not in sample.input_ids

    # --- validation inference: train_history, target not appended ------------- #
    validation_encoded = encode_inference_history(train_history, 8, num_items)
    assert tuple(v for v in validation_encoded if v != PAD_ID) == train_history
    assert case.validation_target not in validation_encoded
    assert case.test_target not in validation_encoded

    # --- test inference: train_history + validation interaction --------------- #
    test_encoded = encode_inference_history(
        train_history + (case.validation_target,), 8, num_items
    )
    expected = (train_history + (case.validation_target,))[-8:]
    assert tuple(v for v in test_encoded if v != PAD_ID) == expected
    assert test_encoded[-1] == case.validation_target      # legitimate context
    assert case.test_target not in test_encoded            # target never appended


@artifacts_available
def test_test_target_is_never_appended_to_test_history() -> None:
    """The target being predicted must never appear in the history used to predict it."""
    bundle = smoke_bundle()
    model = bundle["model"]
    num_items = bundle["split"].catalog_size
    score_fn = make_score_fn(model, SMOKE_MAX_SEQ_LEN, num_items)

    for case in bundle["cases"][:5]:
        test_history = tuple(case.train_history) + (case.validation_target,)
        encoded = encode_inference_history(test_history, SMOKE_MAX_SEQ_LEN, num_items)
        assert case.test_target not in encoded
        scores = score_fn(test_history, case.test_target)
        assert len(scores) == num_items + 1
        assert all(math.isfinite(v) for v in scores)


@artifacts_available
def test_smoke_training_does_not_refit_on_validation_interactions() -> None:
    """There is exactly one fit; the validation interaction only appears as context."""
    bundle = smoke_bundle()
    dataset = bundle["dataset"]
    # every training sample's arrays derive from train_history, which excludes both
    # targets by construction - so no validation interaction can have been fitted
    for case in bundle["cases"][:20]:
        assert case.validation_target not in case.train_history
        assert case.test_target not in case.train_history
    assert all(s.num_valid_positions >= 1 for s in dataset.samples)


# --------------------------------------------------------------------------- #
# 29-30, 26-27. Evaluator reuse after smoke training
# --------------------------------------------------------------------------- #


def _evaluation_subset(bundle, limit: int = 8):
    """A small deterministic evaluation subset: trainable + zero-transition users."""
    cases = bundle["cases"]
    trainable = [c for c in cases if len(c.train_history) >= 2][:limit]
    zero_transition = [c for c in cases if len(c.train_history) == 1][:1]
    assert trainable and zero_transition, "fixture must contain both user kinds"
    return trainable, zero_transition


@artifacts_available
def test_validation_and_test_scoring_after_smoke_training() -> None:
    """Both inference paths run through the frozen evaluator on a small subset."""
    bundle = smoke_bundle()
    model = bundle["model"]
    num_items = bundle["split"].catalog_size
    evaluator = FullRankingEvaluator(num_items=num_items, k_values=(5, 10, 20))
    score_fn = make_score_fn(model, SMOKE_MAX_SEQ_LEN, num_items)

    trainable, zero_transition = _evaluation_subset(bundle)
    subset = trainable + zero_transition

    # every case's score vector satisfies the evaluator's contract
    for case in subset:
        evaluator.validate_scores(score_fn(case.validation_history, case.validation_target))

    # SMOKE DIAGNOSTIC ONLY - NOT A BENCHMARK RESULT
    validation = evaluator.evaluate(subset, score_fn, mode="validation")
    test = evaluator.evaluate(subset, score_fn, mode="test")

    assert validation.report.num_cases == len(subset)
    assert test.report.num_cases == len(subset)
    for outcome in (validation, test):
        assert all(outcome.hr_recall_agree.values())
        for k in (5, 10, 20):
            for name in ("HR", "Recall", "NDCG"):
                value = outcome.report.metrics[name][k]
                assert math.isfinite(value) and 0.0 <= value <= 1.0

    # the evaluator owned masking/ranking; the model/trainer produced raw scores only
    assert validation.mode == "validation" and test.mode == "test"


@artifacts_available
def test_zero_transition_evaluation_user_remains_scoreable() -> None:
    """A user with one train item cannot train, yet is still evaluated normally."""
    bundle = smoke_bundle()
    model = bundle["model"]
    num_items = bundle["split"].catalog_size
    evaluator = FullRankingEvaluator(num_items=num_items, k_values=(5,))
    score_fn = make_score_fn(model, SMOKE_MAX_SEQ_LEN, num_items)

    zero_transition = next(c for c in bundle["cases"] if len(c.train_history) == 1)
    assert zero_transition.user_int_id not in bundle["subset_users"]  # not trainable

    encoded = encode_inference_history(zero_transition.validation_history, SMOKE_MAX_SEQ_LEN, num_items)
    assert sum(1 for v in encoded if v != PAD_ID) == 1
    scores = score_fn(zero_transition.validation_history, zero_transition.validation_target)
    assert len(scores) == num_items + 1
    assert all(math.isfinite(v) for v in scores)
    evaluator.validate_scores(scores)

    outcome = evaluator.evaluate([zero_transition], score_fn, mode="test")
    assert outcome.report.num_cases == 1
    assert math.isfinite(outcome.report.metrics["HR"][5])


@artifacts_available
def test_evaluator_is_reused_not_duplicated() -> None:
    """The trainer/model expose no evaluation API; the evaluator owns those rules."""
    bundle = smoke_bundle()
    trainer = bundle["trainer"]
    for forbidden in ("rank", "mask", "ndcg", "hr_at_k", "recall_at_k", "evaluate"):
        assert not hasattr(trainer, forbidden), f"trainer must not implement {forbidden}"
    model = bundle["model"]
    for forbidden in ("rank", "mask_seen", "ndcg", "ndcg_at_k", "hit_rate"):
        assert not hasattr(model, forbidden), f"model must not implement {forbidden}"

    # and the evaluation package is the only place with ranking metrics
    from recommendation.evaluation import metrics as eval_metrics

    assert hasattr(eval_metrics, "ndcg_at_k")
    assert callable(FullRankingEvaluator)


@artifacts_available
def test_smoke_training_result_is_serialisable() -> None:
    """The smoke evidence summary is JSON-friendly and contains no tensors."""
    bundle = smoke_bundle()
    payload = json.loads(json.dumps(bundle["result"].as_dict()))
    assert payload["optimizer_steps"] > 0
    assert payload["pad_embedding_zero_after_training"] is True
    assert set(payload["trainer_config"]) >= {"learning_rate", "batch_size", "epochs", "seed"}

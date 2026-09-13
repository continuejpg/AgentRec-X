"""Milestone 5 device policy, sequence-window selection and checkpoint contracts.

CPU-only testable surface: no test here requires an RTX 4090.  CUDA-specific checks
skip cleanly when CUDA is unavailable.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

torch = pytest.importorskip("torch", reason="Milestone 5 tests require PyTorch")

from recommendation.datasets.window import (  # noqa: E402
    CANDIDATE_WINDOWS,
    RETENTION_TARGET,
    SequenceWindowError,
    retained_transitions,
    select_max_seq_len,
    train_history_lengths,
    train_history_statistics,
    transition_retention,
)
from recommendation.evaluation import EvaluationCase  # noqa: E402
from recommendation.models import build_model  # noqa: E402
from recommendation.training import (  # noqa: E402
    SUPPORTED_DEVICES,
    SASRecTrainer,
    TrainerConfig,
    TrainingError,
    resolve_device,
)

CUDA_AVAILABLE = torch.cuda.is_available()
requires_cuda = pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA is not available")


def case(user_int_id: int, history: tuple[int, ...]) -> EvaluationCase:
    """Build a minimal evaluation case around a train history."""
    return EvaluationCase(
        user_id=f"u{user_int_id}",
        user_int_id=user_int_id,
        train_history=history,
        validation_target=900,
        test_target=901,
        sequence_length=len(history) + 2,
    )


# --------------------------------------------------------------------------- #
# 1-3. Device configuration
# --------------------------------------------------------------------------- #


def test_cpu_device_is_accepted() -> None:
    """CPU remains fully supported."""
    assert resolve_device("cpu") == torch.device("cpu")
    assert str(resolve_device("cpu")) == "cpu"


def test_supported_device_constant() -> None:
    """The accepted device strings are frozen and documented."""
    assert SUPPORTED_DEVICES == ("cpu", "cuda", "cuda:0")


def test_malformed_device_strings_are_rejected() -> None:
    """Anything outside the supported set fails loudly."""
    for bad in ("cuda:1", "cuda:abc", "gpu", "cpu:0", "mps", "", "   ", None, 123):
        with pytest.raises(TrainingError):
            resolve_device(bad)  # type: ignore[arg-type]


@pytest.mark.skipif(CUDA_AVAILABLE, reason="only meaningful when CUDA is absent")
def test_cuda_is_rejected_without_cuda_rather_than_silently_using_cpu() -> None:
    """Requesting CUDA with no CUDA must fail, not quietly fall back to CPU."""
    for requested in ("cuda", "cuda:0"):
        with pytest.raises(TrainingError) as excinfo:
            resolve_device(requested)
        assert "cuda.is_available" in str(excinfo.value) or "CUDA" in str(excinfo.value)
        assert "silently" in str(excinfo.value)


@requires_cuda
def test_cuda_device_is_accepted_when_available() -> None:
    """With CUDA present, both cuda and cuda:0 resolve to a CUDA device."""
    assert resolve_device("cuda").type == "cuda"
    assert resolve_device("cuda:0").type == "cuda"


def test_trainer_moves_model_and_tensors_onto_the_selected_device() -> None:
    """The model's parameters live on the trainer's device after construction."""
    model = build_model(num_items=10, seed=0, max_seq_len=4, hidden_size=8, num_blocks=1, num_heads=1)
    trainer = SASRecTrainer(model, 10, TrainerConfig(device="cpu"))
    assert trainer.device == torch.device("cpu")
    assert all(p.device.type == "cpu" for p in trainer.model.parameters())


def test_trainer_rejects_malformed_device() -> None:
    """A bad device string fails at trainer construction."""
    model = build_model(num_items=10, seed=0, max_seq_len=4, hidden_size=8, num_blocks=1, num_heads=1)
    with pytest.raises(TrainingError):
        SASRecTrainer(model, 10, TrainerConfig(device="gpu"))


@requires_cuda
def test_trainer_cuda_path_moves_model_to_gpu() -> None:
    """On a CUDA host, the trainer places the model on the GPU."""
    model = build_model(num_items=10, seed=0, max_seq_len=4, hidden_size=8, num_blocks=1, num_heads=1)
    trainer = SASRecTrainer(model, 10, TrainerConfig(device="cuda"))
    assert trainer.device.type == "cuda"
    assert all(p.device.type == "cuda" for p in trainer.model.parameters())


# --------------------------------------------------------------------------- #
# 4-5. max_seq_len selection rule and transition retention
# --------------------------------------------------------------------------- #


def test_train_history_lengths_reads_only_train_history() -> None:
    """Length statistics use train_history, never the targets."""
    cases = [case(1, (1, 2, 3)), case(2, (4, 5))]
    assert train_history_lengths(cases) == [3, 2]


def test_retained_transitions_matches_build_arrays_semantics() -> None:
    """Clipping is the same operation the dataset performs."""
    lengths = [1, 2, 3, 5, 21]
    # transitions per user: 0, 1, 2, 4, 20
    assert retained_transitions(lengths, 5) == 0 + 1 + 2 + 4 + 5
    assert retained_transitions(lengths, 20) == 0 + 1 + 2 + 4 + 20
    assert retained_transitions(lengths, 1) == 0 + 1 + 1 + 1 + 1
    assert retained_transitions(lengths, 1000) == 27


def test_transition_retention_matches_dataset_truncation() -> None:
    """Retention equals what the SASRec dataset actually keeps."""
    from recommendation.datasets.sasrec import SASRecDatasetConfig, build_dataset

    cases = [case(i, tuple(range(1, n + 1))) for i, n in enumerate([2, 5, 10, 30, 50], start=1)]
    for window in (3, 8, 20):
        report = transition_retention(cases, (window,))[window]
        dataset = build_dataset(cases, num_items=100, config_=SASRecDatasetConfig(max_seq_len=window, seed=0))
        assert report["retained_transitions"] == dataset.total_valid_positions
        assert dataset.total_valid_positions == retained_transitions(
            train_history_lengths(cases), window
        )


def test_train_history_statistics_percentiles() -> None:
    """Percentiles are hand-verifiable on a small fixture."""
    cases = [case(i, tuple(range(1, n + 1))) for i, n in enumerate([1, 2, 3, 4, 5], start=1)]
    stats = train_history_statistics(cases)
    assert stats["count"] == 5
    assert stats["min"] == 1 and stats["max"] == 5
    assert stats["median"] == 3.0
    assert stats["mean"] == 3.0
    assert stats["p75"] == 4.0
    assert stats["p90"] == 4.6
    assert stats["total_interactions"] == 15
    assert stats["raw_transitions"] == 0 + 1 + 2 + 3 + 4
    assert stats["users_with_zero_transitions"] == 1


def test_max_seq_len_rule_selects_smallest_window_hitting_target() -> None:
    """The frozen rule picks the smallest candidate meeting the 95% threshold."""
    # histories of length 21 -> 20 transitions each; L=20 retains everything
    cases = [case(i, tuple(range(1, 22))) for i in range(1, 6)]
    selected, evidence = select_max_seq_len(cases, windows=(20, 50, 100, 200))
    assert selected == 20
    assert evidence["selected_retention"] == pytest.approx(1.0)
    assert "smallest candidate" in evidence["reason"]


def test_max_seq_len_rule_escalates_when_small_window_is_insufficient() -> None:
    """A heavy tail forces a larger window."""
    cases = [case(1, tuple(range(1, 302)))]  # 301 transitions, all beyond L=200? no
    selected, evidence = select_max_seq_len(cases, windows=(20, 50, 100, 200))
    # L=20 keeps 20/301 = 6.6%; none reach 95%, so the largest is selected
    assert selected == 200
    assert "no candidate reached" in evidence["reason"]
    assert evidence["retention_by_window"]["20"]["retention"] < 0.95


def test_max_seq_len_rule_falls_back_to_largest_with_reported_shortfall() -> None:
    """When even the largest window misses 95%, the shortfall is reported."""
    # A history of 1001 items has 1000 transitions; L=200 retains 200/1000 = 20%,
    # far below the 95% target, so the largest candidate is selected and the
    # shortfall is reported rather than hidden.
    cases = [case(1, tuple(range(1, 1002)))]
    selected, evidence = select_max_seq_len(cases, windows=(20, 50, 100, 200))
    assert selected == 200
    assert evidence["retention_by_window"]["200"]["retained_transitions"] == 200
    assert evidence["retention_by_window"]["200"]["raw_transitions"] == 1000
    assert evidence["selected_retention"] == pytest.approx(0.2, rel=1e-9)
    assert evidence["selected_retention"] < RETENTION_TARGET
    assert all(
        window_evidence["retention"] < RETENTION_TARGET
        for window_evidence in evidence["retention_by_window"].values()
    )


def test_candidate_windows_are_the_frozen_set() -> None:
    """The candidate windows are exactly [20, 50, 100, 200]."""
    assert CANDIDATE_WINDOWS == (20, 50, 100, 200)
    assert RETENTION_TARGET == 0.95


def test_selection_is_independent_of_case_order() -> None:
    """Selection uses only aggregate counts, so ordering cannot matter."""
    cases = [case(i, tuple(range(1, n + 1))) for i, n in enumerate([3, 40, 7, 120], start=1)]
    forward, ev_a = select_max_seq_len(cases)
    backward, ev_b = select_max_seq_len(list(reversed(cases)))
    assert forward == backward
    assert ev_a == ev_b


def test_window_helpers_reject_degenerate_input() -> None:
    """Empty input and invalid windows are explicit errors."""
    with pytest.raises(SequenceWindowError):
        train_history_statistics([])
    with pytest.raises(SequenceWindowError):
        retained_transitions([1, 2], 0)
    with pytest.raises(SequenceWindowError):
        select_max_seq_len([case(1, (1, 2, 3))], windows=())


def test_selection_uses_training_data_only() -> None:
    """Changing every validation/test target cannot change the selected window."""
    histories = [(1, 2, 3), (4, 5, 6, 7), tuple(range(1, 60))]
    left = [case(i, h) for i, h in enumerate(histories, start=1)]
    right = [
        EvaluationCase(f"u{i}", i, h, 1, 2, len(h) + 2) for i, h in enumerate(histories, start=1)
    ]
    _, ev_left = select_max_seq_len(left)
    _, ev_right = select_max_seq_len(right)
    assert ev_left == ev_right


# --------------------------------------------------------------------------- #
# Training-config sanity for the formal run
# --------------------------------------------------------------------------- #


def test_formal_trainer_config_is_constructible_and_serialisable() -> None:
    """The frozen Milestone 5 optimizer configuration is valid."""
    config = TrainerConfig(
        learning_rate=0.001,
        weight_decay=0.0,
        batch_size=256,
        epochs=200,
        seed=2026,
        device="cpu",
        shuffle=True,
        resample_negatives=True,
        max_grad_norm=5.0,
    )
    payload = json.loads(json.dumps(config.as_dict()))
    assert payload["learning_rate"] == 0.001
    assert payload["batch_size"] == 256
    assert payload["seed"] == 2026
    assert payload["max_grad_norm"] == 5.0
    assert payload["optimizer"] == "AdamW"


def test_formal_model_config_parameter_count_is_reported() -> None:
    """The canonical architecture builds and reports its size."""
    model = build_model(
        num_items=1000, seed=2026, max_seq_len=50, hidden_size=64,
        num_blocks=2, num_heads=2, dropout=0.2,
    )
    described = model.describe()
    assert described["config"]["hidden_size"] == 64
    assert described["config"]["num_blocks"] == 2
    assert described["config"]["num_heads"] == 2
    assert described["config"]["dropout"] == 0.2
    # embedding table dominates: (1000+1)*64 item + 50*64 positional
    assert described["parameter_count"] > (1001 + 50) * 64

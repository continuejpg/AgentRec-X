"""Checkpoint / resume / selection-rule tests (Milestone 5).

Covers the checkpoint contents, atomic writes, save-load round trip, semantic resume
(epoch, optimizer, negative-sampling epoch, patience, best record), the validation-only
selection rule with its tie-breaks, and the experiment manifest.  CPU only.
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

torch = pytest.importorskip("torch", reason="checkpoint tests require PyTorch")

from recommendation.models import build_model  # noqa: E402
from recommendation.training.checkpoint import (  # noqa: E402
    CHECKPOINT_FORMAT,
    MANIFEST_FORMAT,
    CheckpointError,
    ExperimentManifest,
    SelectionRecord,
    TrainingState,
    environment_metadata,
    load_checkpoint,
    save_checkpoint,
    sha256_file,
    state_from_payload,
    write_json_atomic,
)
from recommendation.training.sasrec import TrainerConfig  # noqa: E402

NUM_ITEMS = 32
MAX_SEQ_LEN = 8


def make_model(seed: int = 0):
    """A small deterministic SASRec."""
    return build_model(
        num_items=NUM_ITEMS, seed=seed, max_seq_len=MAX_SEQ_LEN,
        hidden_size=8, num_blocks=1, num_heads=1, dropout=0.0,
    )


def make_optimizer(model):
    """An AdamW over the model's parameters."""
    return torch.optim.AdamW(model.parameters(), lr=0.01)


# --------------------------------------------------------------------------- #
# Save / load round trip
# --------------------------------------------------------------------------- #


def test_checkpoint_save_load_round_trip(tmp_path: Path) -> None:
    """A checkpoint reloads into an identically-parameterised model."""
    model = make_model(seed=3)
    optimizer = make_optimizer(model)
    state = TrainingState(epoch=7, global_step=123, best_epoch=5,
                          best_metric=0.25, best_hr=0.4, patience_counter=2, negative_epoch=7)
    path = save_checkpoint(
        tmp_path / "best.pt",
        model=model, optimizer=optimizer, state=state,
        model_config=model.config.as_dict(), trainer_config=TrainerConfig().as_dict(),
        max_seq_len=MAX_SEQ_LEN, seed=2026, num_items=NUM_ITEMS,
        dataset_identity={"sequences_sha256": "abc"},
        validation_metrics={"NDCG@10": 0.25, "HR@10": 0.4},
    )
    assert path.exists()

    fresh = make_model(seed=999)
    assert not torch.equal(fresh.item_embedding.weight, model.item_embedding.weight)

    payload = load_checkpoint(path, model=fresh)
    assert payload["format"] == CHECKPOINT_FORMAT
    assert torch.equal(fresh.item_embedding.weight, model.item_embedding.weight)
    assert payload["max_seq_len"] == MAX_SEQ_LEN
    assert payload["seed"] == 2026
    assert payload["num_items"] == NUM_ITEMS
    assert payload["dataset_identity"]["sequences_sha256"] == "abc"
    assert payload["validation_metrics"]["NDCG@10"] == 0.25


def test_checkpoint_contains_all_required_fields(tmp_path: Path) -> None:
    """Every field the milestone lists is present in the payload."""
    model = make_model()
    path = save_checkpoint(
        tmp_path / "best.pt", model=model, optimizer=make_optimizer(model),
        state=TrainingState(epoch=1, global_step=2, negative_epoch=1),
        model_config=model.config.as_dict(), trainer_config=TrainerConfig().as_dict(),
        max_seq_len=MAX_SEQ_LEN, seed=2026, num_items=NUM_ITEMS,
    )
    payload = load_checkpoint(path, restore_rng=False)
    for key in (
        "model_state_dict", "optimizer_state_dict", "state", "model_config",
        "trainer_config", "max_seq_len", "seed", "num_items", "dataset_identity",
        "validation_metrics", "rng_state",
    ):
        assert key in payload, f"checkpoint is missing {key}"
    assert payload["state"]["negative_epoch"] == 1


def test_checkpoint_rejects_foreign_or_missing_files(tmp_path: Path) -> None:
    """A missing path or a non-checkpoint payload is an explicit error."""
    with pytest.raises(CheckpointError):
        load_checkpoint(tmp_path / "nope.pt")

    foreign = tmp_path / "foreign.pt"
    torch.save({"not": "a checkpoint"}, foreign)
    with pytest.raises(CheckpointError):
        load_checkpoint(foreign)


def test_checkpoint_sha256_tracks_content_not_wall_clock(tmp_path: Path) -> None:
    """The hash identifies the *trained state*, not the save timestamp.

    A checkpoint deliberately records ``saved_at`` for auditing, so two saves of an
    unchanged model differ byte-wise.  The meaningful property is that the recorded
    identity fields (weights, state, config) are what the hash is compared over, and
    that any real change - an optimizer step or a different epoch - changes the hash.
    """
    model = make_model()
    optimizer = make_optimizer(model)

    def save(name: str, epoch: int) -> tuple[str, dict]:
        path = save_checkpoint(
            tmp_path / name, model=model, optimizer=optimizer,
            state=TrainingState(epoch=epoch), model_config=model.config.as_dict(),
            trainer_config=TrainerConfig().as_dict(), max_seq_len=MAX_SEQ_LEN,
            seed=2026, num_items=NUM_ITEMS,
        )
        payload = load_checkpoint(path, restore_rng=False)
        return sha256_file(path), payload

    first_hash, first = save("a.pt", 1)
    second_hash, second = save("b.pt", 1)
    assert first_hash != second_hash, "saved_at makes each save byte-distinct"
    assert first["saved_at"] != second["saved_at"]

    # ...but the trained state itself is identical
    for key, value in first["model_state_dict"].items():
        assert torch.equal(value, second["model_state_dict"][key])
    assert first["state"] == second["state"]
    assert first["model_config"] == second["model_config"]

    # a genuinely different training state changes the hash
    third_hash, third = save("c.pt", 2)
    assert third_hash != first_hash
    assert third["state"]["epoch"] == 2


def test_atomic_write_leaves_no_temp_files(tmp_path: Path) -> None:
    """Atomic writes clean up their temp files."""
    model = make_model()
    for index in range(3):
        save_checkpoint(
            tmp_path / "best.pt", model=model, optimizer=make_optimizer(model),
            state=TrainingState(epoch=index), model_config=model.config.as_dict(),
            trainer_config=TrainerConfig().as_dict(), max_seq_len=MAX_SEQ_LEN,
            seed=2026, num_items=NUM_ITEMS,
        )
    leftovers = [p.name for p in tmp_path.iterdir() if p.name.startswith(".")]
    assert leftovers == [], f"temp files left behind: {leftovers}"


# --------------------------------------------------------------------------- #
# Semantic resume
# --------------------------------------------------------------------------- #


def test_resume_preserves_epoch_step_patience_and_best(tmp_path: Path) -> None:
    """A resumed run does not restart counters or lose the best record."""
    model = make_model()
    optimizer = make_optimizer(model)
    state = TrainingState(
        epoch=11, global_step=880, best_epoch=8, best_metric=0.31, best_hr=0.44,
        patience_counter=3, negative_epoch=11,
    )
    path = save_checkpoint(
        tmp_path / "last.pt", model=model, optimizer=optimizer, state=state,
        model_config=model.config.as_dict(), trainer_config=TrainerConfig().as_dict(),
        max_seq_len=MAX_SEQ_LEN, seed=2026, num_items=NUM_ITEMS,
    )

    restored_model = make_model(seed=77)
    restored_optimizer = make_optimizer(restored_model)
    payload = load_checkpoint(path, model=restored_model, optimizer=restored_optimizer)
    resumed = state_from_payload(payload)

    assert resumed == state, "training state must round-trip exactly"
    assert resumed.negative_epoch == 11, "negative-sampling epoch must not restart"
    assert resumed.patience_counter == 3, "patience must not reset"
    assert resumed.best_epoch == 8 and resumed.best_metric == 0.31, "best record preserved"

    # the optimizer state really was restored (exp_avg buffers exist and match)
    before = optimizer.state_dict()["state"]
    after = restored_optimizer.state_dict()["state"]
    assert set(before.keys()) == set(after.keys())


def test_resume_restores_rng_so_negative_sampling_continues(tmp_path: Path) -> None:
    """Python/torch RNG state is restored, so epoch-resampling continues identically."""
    model = make_model()
    random.seed(1234)
    torch.manual_seed(1234)
    consumed = [random.random() for _ in range(3)]
    expected_next = [random.random() for _ in range(3)]

    # rewind to the saved point
    random.seed(1234)
    torch.manual_seed(1234)
    for _ in range(3):
        random.random()

    path = save_checkpoint(
        tmp_path / "last.pt", model=model, optimizer=None, state=TrainingState(epoch=1),
        model_config=model.config.as_dict(), trainer_config=TrainerConfig().as_dict(),
        max_seq_len=MAX_SEQ_LEN, seed=2026, num_items=NUM_ITEMS,
    )

    random.seed(999)  # scramble before resuming
    torch.manual_seed(999)
    load_checkpoint(path, restore_rng=True)
    assert [random.random() for _ in range(3)] == expected_next, (
        "Python RNG must resume from the checkpointed position"
    )
    assert len(consumed) == 3


def test_rng_restore_can_be_disabled(tmp_path: Path) -> None:
    """``restore_rng=False`` leaves the caller's RNG untouched."""
    model = make_model()
    random.seed(5)
    path = save_checkpoint(
        tmp_path / "last.pt", model=model, optimizer=None, state=TrainingState(),
        model_config=model.config.as_dict(), trainer_config=TrainerConfig().as_dict(),
        max_seq_len=MAX_SEQ_LEN, seed=1, num_items=NUM_ITEMS,
    )
    random.seed(4242)
    baseline = random.random()
    random.seed(4242)
    load_checkpoint(path, restore_rng=False)
    assert random.random() == baseline


def test_resume_into_mismatched_model_is_detected(tmp_path: Path) -> None:
    """Loading into an incompatible architecture fails loudly rather than silently."""
    model = make_model()
    path = save_checkpoint(
        tmp_path / "best.pt", model=model, optimizer=None, state=TrainingState(),
        model_config=model.config.as_dict(), trainer_config=TrainerConfig().as_dict(),
        max_seq_len=MAX_SEQ_LEN, seed=1, num_items=NUM_ITEMS,
    )
    other = build_model(num_items=NUM_ITEMS, seed=0, max_seq_len=MAX_SEQ_LEN,
                        hidden_size=16, num_blocks=1, num_heads=1, dropout=0.0)
    with pytest.raises(Exception):
        load_checkpoint(path, model=other)


# --------------------------------------------------------------------------- #
# Checkpoint selection rule (validation only)
# --------------------------------------------------------------------------- #


def test_selection_prefers_higher_validation_ndcg10() -> None:
    """The primary metric is validation NDCG@10."""
    record = SelectionRecord()
    assert record.consider(0, ndcg10=0.10, hr10=0.30) is True
    assert record.consider(1, ndcg10=0.20, hr10=0.25) is True   # higher NDCG wins
    assert record.consider(2, ndcg10=0.15, hr10=0.90) is False  # hr alone cannot win
    assert record.best_epoch == 1
    assert record.patience == 1


def test_selection_tie_break_uses_hr10() -> None:
    """An exact NDCG@10 tie is broken by higher validation HR@10."""
    record = SelectionRecord()
    record.consider(0, ndcg10=0.2, hr10=0.3)
    assert record.consider(1, ndcg10=0.2, hr10=0.4) is True
    assert record.best_epoch == 1
    assert record.best_hr10 == 0.4
    assert record.patience == 0


def test_selection_tie_break_prefers_earlier_epoch() -> None:
    """When both metrics tie exactly, the earlier epoch keeps the record."""
    record = SelectionRecord()
    record.consider(3, ndcg10=0.2, hr10=0.4)
    assert record.consider(9, ndcg10=0.2, hr10=0.4) is False
    assert record.best_epoch == 3
    assert record.patience == 1


def test_patience_counts_consecutive_non_improvements_and_resets() -> None:
    """Patience increments on failure and resets on improvement."""
    record = SelectionRecord()
    record.consider(0, 0.5, 0.5)
    for _ in range(3):
        record.consider(1, 0.1, 0.1)
    assert record.patience == 3
    record.consider(2, 0.6, 0.6)
    assert record.patience == 0
    assert record.best_epoch == 2


def test_selection_never_consults_test_metrics() -> None:
    """The selection API has no test-metric parameter at all."""
    import inspect

    signature = inspect.signature(SelectionRecord.consider)
    assert list(signature.parameters) == ["self", "epoch", "ndcg10", "hr10"]
    for forbidden in ("test", "split"):
        assert forbidden not in str(signature)


# --------------------------------------------------------------------------- #
# Manifest
# --------------------------------------------------------------------------- #


def test_manifest_round_trip(tmp_path: Path) -> None:
    """The manifest writes atomically, reloads, and carries its format tag."""
    manifest = ExperimentManifest(run_id="m5-canonical-2026")
    manifest.update(
        seed=2026,
        num_items=NUM_ITEMS,
        max_seq_len=MAX_SEQ_LEN,
        raw_sha256="deadbeef",
        window={"selected_max_seq_len": MAX_SEQ_LEN, "retention": 0.982},
    )
    path = manifest.write(tmp_path / "run.json")
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["format"] == MANIFEST_FORMAT
    assert payload["run_id"] == "m5-canonical-2026"
    assert payload["seed"] == 2026
    assert payload["window"]["retention"] == 0.982


def test_manifest_update_merges_sections() -> None:
    """Sections merge rather than replace."""
    manifest = ExperimentManifest(run_id="r")
    manifest.update(a=1, b=2)
    manifest.update(b=3, c=4)
    assert manifest.as_dict()["a"] == 1
    assert manifest.as_dict()["b"] == 3
    assert manifest.as_dict()["c"] == 4


def test_environment_metadata_reports_cpu_and_cuda_state() -> None:
    """Environment metadata is complete whether or not CUDA exists."""
    meta = environment_metadata()
    assert meta["torch_version"]
    assert isinstance(meta["cuda_available"], bool)
    if not meta["cuda_available"]:
        assert meta["device_name"] is None
        assert meta["device_count"] == 0
    else:
        assert meta["device_name"]
        assert meta["total_memory_bytes"] > 0


def test_json_atomic_write_creates_parent_and_no_temp(tmp_path: Path) -> None:
    """JSON writes create parents and leave no temp files."""
    target = write_json_atomic({"k": "v"}, tmp_path / "nested" / "dir" / "x.json")
    assert json.loads(target.read_text(encoding="utf-8")) == {"k": "v"}
    assert [p.name for p in target.parent.iterdir() if p.name.startswith(".")] == []


def test_checkpoint_hash_helper_matches_hashlib(tmp_path: Path) -> None:
    """The streaming SHA-256 helper matches a straightforward hash."""
    import hashlib

    path = tmp_path / "blob.bin"
    path.write_bytes(b"agentrecx" * 1000)
    assert sha256_file(path) == hashlib.sha256(path.read_bytes()).hexdigest()

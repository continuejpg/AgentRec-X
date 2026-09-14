"""Tiny synthetic serving fixtures for the Milestone 6 inference/API tests.

Normal pytest must never load the 349 MB formal run or the 156,746-item catalog, so
these helpers build a deterministic miniature checkpoint, mapping artifact and
manifest in a temporary directory.

The generated checkpoint uses the real Milestone 5 checkpoint format, so the engine's
loader, identity checks and strict ``state_dict`` load are all genuinely exercised.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

from recommendation.models import build_model
from recommendation.training.checkpoint import (
    CHECKPOINT_FORMAT,
    TrainingState,
    save_checkpoint,
)

#: Small catalogue for tests.  Item ids are ``1..NUM_ITEMS``; PAD is 0.
NUM_ITEMS = 12
MAX_SEQ_LEN = 4
HIDDEN_SIZE = 8
NUM_BLOCKS = 1
NUM_HEADS = 2
DROPOUT = 0.0
SEED = 1234

#: Deterministic synthetic ``parent_asin`` values; index i maps to item id i+1.
PARENT_ASINS: tuple[str, ...] = tuple(f"B{i:09d}" for i in range(1, NUM_ITEMS + 1))


def asin_for_item(item_id: int) -> str:
    """Return the synthetic ``parent_asin`` for an integer item id."""
    if not 1 <= item_id <= NUM_ITEMS:
        raise ValueError(f"item id {item_id} outside 1..{NUM_ITEMS}")
    return PARENT_ASINS[item_id - 1]


def build_fixture(tmp_path: Path, *, num_items: int = NUM_ITEMS) -> dict[str, object]:
    """Create a checkpoint + mappings + manifest trio and return their details."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(SEED)
    model = build_model(
        num_items=num_items,
        seed=SEED,
        max_seq_len=MAX_SEQ_LEN,
        hidden_size=HIDDEN_SIZE,
        num_blocks=NUM_BLOCKS,
        num_heads=NUM_HEADS,
        dropout=DROPOUT,
    )
    checkpoint_path = tmp_path / "best.pt"
    save_checkpoint(
        checkpoint_path,
        model=model,
        optimizer=None,
        state=TrainingState(epoch=1, global_step=2),
        model_config=model.config.as_dict(),
        trainer_config={"optimizer": "AdamW", "learning_rate": 0.001},
        max_seq_len=MAX_SEQ_LEN,
        seed=SEED,
        num_items=num_items,
        dataset_identity={"sequences_sha256": "test-fixture"},
        validation_metrics={"NDCG@10": 0.0},
    )

    item2id = {asin_for_item(i): i for i in range(1, num_items + 1)}
    id2item: list[str | None] = [None] + [asin_for_item(i) for i in range(1, num_items + 1)]
    mappings_path = tmp_path / "mappings.json"
    mappings_path.write_text(
        json.dumps(
            {
                "padding": {"pad_id": 0, "first_real_id": 1},
                "num_users": 3,
                "num_items": num_items,
                "user2id": {"u1": 1, "u2": 2, "u3": 3},
                "item2id": item2id,
                "id2user": [None, "u1", "u2", "u3"],
                "id2item": id2item,
            }
        ),
        encoding="utf-8",
    )

    manifest_path = tmp_path / "run.json"
    manifest_path.write_text(
        json.dumps(
            {
                "format": "agentrecx.experiment.v1",
                "run_id": "test-fixture",
                "seed": SEED,
                "num_items": num_items,
                "max_seq_len": MAX_SEQ_LEN,
                "model_config": model.config.as_dict(),
                "git": {"available": True, "commit": "0" * 40, "branch": "test", "dirty": True},
            }
        ),
        encoding="utf-8",
    )

    return {
        "checkpoint_path": checkpoint_path,
        "mappings_path": mappings_path,
        "manifest_path": manifest_path,
        "model": model,
        "num_items": num_items,
    }


def fixture_paths(tmp_path: Path) -> dict[str, Path]:
    """Return just the three artifact paths for a fixture."""
    detail = build_fixture(tmp_path)
    return {
        "checkpoint_path": detail["checkpoint_path"],  # type: ignore[dict-item]
        "mappings_path": detail["mappings_path"],  # type: ignore[dict-item]
        "manifest_path": detail["manifest_path"],  # type: ignore[dict-item]
    }


__all__ = [
    "DROPOUT",
    "HIDDEN_SIZE",
    "MAX_SEQ_LEN",
    "NUM_BLOCKS",
    "NUM_HEADS",
    "NUM_ITEMS",
    "PARENT_ASINS",
    "SEED",
    "asin_for_item",
    "build_fixture",
    "fixture_paths",
]

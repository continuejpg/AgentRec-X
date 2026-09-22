"""Training examples for the Semantic-ID generator.

An example is one **next-item prediction**: a user's history expressed as Semantic IDs, and the
Semantic ID of the item that came next.

Leakage discipline
------------------
Examples are built from ``EvaluationCase.train_history`` **only**.  A history of ``n`` items
contributes the ``n - 2`` transitions whose target is an item *before* the validation window;
``validation_target`` and ``test_target`` are never read, which is the same boundary the accepted
SASRec builder and the corrected Two-Tower builder respect.

Why the window is a whole number of items
-----------------------------------------
The generator reads a fixed window of item slots, each slot being ``[SEP, c1..cL]``.  A history is
therefore truncated to whole items, never mid-way through a Semantic ID, because a partial code
would describe an item that does not exist.

Determinism
-----------
Everything here is a pure function of the cohort and the explicit configuration.  Batch order is a
seeded shuffle on an integer seed (not a tuple, whose ``hash()`` seeding is randomised per process
and would break cross-process reproducibility).
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Iterator, Sequence

from recommendation.evaluation.split import EvaluationCase

__all__ = [
    "GeneratorDataset",
    "GeneratorDatasetConfig",
    "GeneratorDataError",
    "build_generator_dataset",
    "iter_batches",
]


class GeneratorDataError(ValueError):
    """Raised when the generator's training data cannot be built as configured."""


@dataclass(frozen=True)
class GeneratorDatasetConfig:
    """How histories are turned into examples.

    Attributes
    ----------
    max_items:
        History window, in items.  Must match the model's ``max_items``.
    seed:
        Shuffle seed.
    max_examples:
        Optional seeded cap on the number of examples.
    """

    max_items: int = 10
    seed: int = 2026
    max_examples: int | None = None

    def __post_init__(self) -> None:
        if isinstance(self.max_items, bool) or not isinstance(self.max_items, int):
            raise GeneratorDataError("max_items must be an int")
        if self.max_items < 1:
            raise GeneratorDataError(f"max_items must be >= 1, got {self.max_items}")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise GeneratorDataError("seed must be an int")
        if self.max_examples is not None and (
            isinstance(self.max_examples, bool)
            or not isinstance(self.max_examples, int)
            or self.max_examples < 1
        ):
            raise GeneratorDataError(
                f"max_examples must be a positive int, got {self.max_examples!r}"
            )


@dataclass(frozen=True)
class GeneratorDataset:
    """The built examples plus the auditable description of how they were built."""

    #: Flat Semantic-ID codes of each history, one entry per example.
    histories: tuple[tuple[int, ...], ...]
    #: Flat Semantic-ID codes of each target item.
    targets: tuple[tuple[int, ...], ...]
    user_int_ids: tuple[int, ...]
    stats: dict[str, Any]

    def __len__(self) -> int:
        return len(self.targets)

    def as_dict(self) -> dict[str, Any]:
        return dict(self.stats)


def build_generator_dataset(
    cases: Sequence[EvaluationCase],
    *,
    item_codes: Sequence[Sequence[int]],
    levels: int,
    config: GeneratorDatasetConfig | None = None,
) -> GeneratorDataset:
    """Build ``(history codes, target codes)`` examples from train histories.

    ``item_codes`` maps an item id to its Semantic ID, so the histories and targets are expressed
    entirely in code space.  A user whose train history is shorter than three items contributes
    nothing: the first item cannot be a target (nothing precedes it to predict from) and the second
    would give the model a one-item context.
    """
    resolved = config or GeneratorDatasetConfig()
    if levels < 1:
        raise GeneratorDataError(f"levels must be >= 1, got {levels}")
    if len(item_codes) < 2:
        raise GeneratorDataError("item_codes must hold at least PAD and one item")

    def codes_for(item_id: int) -> tuple[int, ...]:
        if not 0 < item_id < len(item_codes):
            raise GeneratorDataError(
                f"item id {item_id!r} is outside the assignment's range 1..{len(item_codes) - 1}"
            )
        codes = tuple(int(value) for value in item_codes[item_id])
        if len(codes) != levels:
            raise GeneratorDataError(
                f"item {item_id} has {len(codes)} codes, expected levels={levels}"
            )
        return codes

    histories: list[tuple[int, ...]] = []
    targets: list[tuple[int, ...]] = []
    users: list[int] = []
    skipped_short = 0
    for case in cases:
        history = list(case.train_history)
        if len(history) < 3:
            skipped_short += 1
            continue
        for position in range(2, len(history)):
            prefix = history[:position]
            target = history[position]
            flat: list[int] = []
            for item_id in prefix[-resolved.max_items :]:
                flat.extend(codes_for(item_id))
            histories.append(tuple(flat))
            targets.append(codes_for(target))
            users.append(case.user_int_id)

    if not targets:
        raise GeneratorDataError(
            "no training examples were built; every case had a train history shorter than 3"
        )

    if resolved.max_examples is not None and len(targets) > resolved.max_examples:
        rng = random.Random(resolved.seed)
        chosen = sorted(rng.sample(range(len(targets)), resolved.max_examples))
        histories = [histories[index] for index in chosen]
        targets = [targets[index] for index in chosen]
        users = [users[index] for index in chosen]

    stats = {
        "cases_considered": len(cases),
        "cases_skipped_short_history": skipped_short,
        "examples": len(targets),
        "distinct_users": len(set(users)),
        "distinct_target_items": len(set(targets)),
        "levels": levels,
        "max_items": resolved.max_items,
        "seed": resolved.seed,
        "max_examples": resolved.max_examples,
        "source": "EvaluationCase.train_history only (no validation/test targets read)",
    }
    return GeneratorDataset(
        histories=tuple(histories),
        targets=tuple(targets),
        user_int_ids=tuple(users),
        stats=stats,
    )


def iter_batches(
    dataset: GeneratorDataset, *, batch_size: int, epoch: int = 0, seed: int = 0
) -> Iterator[list[int]]:
    """Yield batches of example indices in this epoch's deterministic order.

    Unlike the two-tower iterator this does **not** skip repeated targets: the generator's loss is a
    per-position cross-entropy over code tokens, not an in-batch softmax, so two examples sharing a
    target are two legitimate observations rather than each other's false negative.
    """
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
        raise GeneratorDataError(f"batch_size must be a positive int, got {batch_size!r}")
    order = list(range(len(dataset)))
    # Integer seed: a tuple would be hashed, and Python randomises string hashing per process.
    random.Random(seed * 1_000_003 + epoch).shuffle(order)
    for start in range(0, len(order), batch_size):
        yield order[start : start + batch_size]

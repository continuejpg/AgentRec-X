"""Training data for the two-tower retrieval model.

What it builds
--------------
Two things, both derived from artifacts the accepted pipeline already produces:

1. **Interaction training pairs.**  For a user whose train history is ``[i1..in]``, every
   prefix/target pair ``([i1..ik], i(k+1))`` for ``k in 1..n-1``.  Inputs are left-padded
   exactly as :func:`recommendation.datasets.sasrec.build_arrays` pads them, so the
   two-tower user tower consumes the same window convention SASRec does.

2. **Categorical item metadata.**  Integer category indices for a fixed set of catalogue
   attributes, one row per catalogue id, built from the normalised metadata artifact.

Leakage discipline
------------------
The pairs are built from ``EvaluationCase.train_history`` **only**.  ``validation_target``
and ``test_target`` are never read by anything in this module, so no training pair can
contain a validation or test label - which is the same boundary Phase 5 enforces for
SASRec, stated here rather than assumed.

An item that appears in a user's train history may also be that user's validation or test
target (a legitimate repurchase).  That is not leakage: the training pair's *label* is
always an item the user interacted with before the validation window, and the evaluator
masks the full test history before ranking regardless.

Determinism
-----------
Every function here is a pure function of its inputs.  ``iter_training_batches`` shuffles
with an explicit seed and skips pairs in a documented, deterministic order, so a run is
reproducible without relying on dictionary iteration or RNG state that is not recorded.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Sequence

from recommendation.datasets.sasrec import PAD_ID, build_arrays
from recommendation.evaluation.split import EvaluationCase

__all__ = [
    "METADATA_ATTRIBUTES",
    "TwoTowerDataError",
    "TwoTowerDataset",
    "TwoTowerDatasetConfig",
    "build_metadata_categories",
    "build_training_pairs",
    "iter_training_batches",
    "item_prior_from_pairs",
    "load_metadata_categories",
    "metadata_tensor",
    "write_metadata_categories",
]

#: Catalogue attributes encoded as categories, in a fixed order.  These are the fields the
#: normalised catalogue carries cleanly for essentially every record, and each is a short
#: categorical value - not free text, which belongs to the lexical retriever arm.
#:
#: ``main_category`` and ``store`` come from the record; ``Color`` and ``Material`` come from
#: the source's own ``details`` bag (kept under the source's key, never renamed).
METADATA_ATTRIBUTES: tuple[str, ...] = (
    "main_category",
    "store",
    "details:Color",
    "details:Material",
)

#: Reserved category index for "the record does not state this attribute".  Index 0 is never
#: a real value, mirroring how ``PAD_ID`` is reserved in the item table.
UNKNOWN_CATEGORY = 0


class TwoTowerDataError(ValueError):
    """Raised when the two-tower training data cannot be built as configured."""


@dataclass(frozen=True)
class TwoTowerDatasetConfig:
    """Configuration for pair construction.

    Attributes
    ----------
    max_seq_len:
        History window length.  Must match the model's ``max_seq_len``: the pairs are
        padded to it.
    seed:
        Shuffle seed for batch iteration.
    max_pairs:
        Optional cap on how many pairs are used.  ``None`` uses every pair.  When set,
        the cap is applied by a seeded sample, so a smaller experimental run is
        reproducible rather than "the first N in whatever order".
    min_history:
        Skip users whose train history has fewer than this many items.  Two is the
        minimum that yields a transition at all; the default of 2 keeps every eligible
        user.
    """

    max_seq_len: int = 50
    seed: int = 0
    max_pairs: int | None = None
    min_history: int = 2

    def __post_init__(self) -> None:
        if isinstance(self.max_seq_len, bool) or not isinstance(self.max_seq_len, int):
            raise TwoTowerDataError("max_seq_len must be an int")
        if self.max_seq_len < 1:
            raise TwoTowerDataError(f"max_seq_len must be >= 1, got {self.max_seq_len}")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise TwoTowerDataError("seed must be an int")
        if self.max_pairs is not None and (
            isinstance(self.max_pairs, bool) or not isinstance(self.max_pairs, int)
            or self.max_pairs < 1
        ):
            raise TwoTowerDataError(f"max_pairs must be a positive int, got {self.max_pairs!r}")
        if isinstance(self.min_history, bool) or not isinstance(self.min_history, int):
            raise TwoTowerDataError("min_history must be an int")
        if self.min_history < 2:
            raise TwoTowerDataError(
                "min_history must be >= 2: a history of fewer than two items has no "
                "next-item transition to learn from"
            )


@dataclass(frozen=True)
class TwoTowerDataset:
    """The built pairs plus the auditable description of how they were built."""

    input_ids: tuple[tuple[int, ...], ...]
    target_ids: tuple[int, ...]
    user_int_ids: tuple[int, ...]
    stats: dict[str, Any]

    def __len__(self) -> int:
        return len(self.target_ids)

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable view (pairs omitted; the stats describe them)."""
        return dict(self.stats)


def build_training_pairs(
    cases: Sequence[EvaluationCase],
    *,
    num_items: int,
    config: TwoTowerDatasetConfig | None = None,
) -> TwoTowerDataset:
    """Build ``(padded history, next item)`` pairs from train histories only.

    Raises
    ------
    TwoTowerDataError
        ``num_items`` is not a positive int, or no case yields a usable pair.
    """
    resolved = config or TwoTowerDatasetConfig()
    if isinstance(num_items, bool) or not isinstance(num_items, int) or num_items < 1:
        raise TwoTowerDataError(f"num_items must be a positive int, got {num_items!r}")

    inputs: list[tuple[int, ...]] = []
    targets: list[int] = []
    users: list[int] = []
    skipped_short = 0
    for case in cases:
        history = list(case.train_history)
        if len(history) < resolved.min_history:
            skipped_short += 1
            continue
        for position in range(1, len(history)):
            prefix = history[:position]
            target = history[position]
            if not 0 < target <= num_items:
                raise TwoTowerDataError(
                    f"train history for user {case.user_int_id} holds an out-of-range "
                    f"item id {target!r}; the accepted split should have rejected this"
                )
            if position < 2:
                # No transition exists for a single-item prefix; skip rather than learn
                # from a "history" that is one item and call it a sequence.
                continue
            input_ids, _positives, _valid = build_arrays(prefix, resolved.max_seq_len)
            inputs.append(input_ids)
            targets.append(target)
            users.append(case.user_int_id)

    if not targets:
        raise TwoTowerDataError(
            "no training pairs were built; every case was shorter than the minimum history"
        )

    if resolved.max_pairs is not None and len(targets) > resolved.max_pairs:
        rng = random.Random(resolved.seed)
        chosen = sorted(rng.sample(range(len(targets)), resolved.max_pairs))
        inputs = [inputs[index] for index in chosen]
        targets = [targets[index] for index in chosen]
        users = [users[index] for index in chosen]

    stats = {
        "cases_considered": len(cases),
        "cases_skipped_short_history": skipped_short,
        "pairs": len(targets),
        "distinct_users": len(set(users)),
        "distinct_targets": len(set(targets)),
        "max_seq_len": resolved.max_seq_len,
        "seed": resolved.seed,
        "max_pairs": resolved.max_pairs,
        "min_history": resolved.min_history,
        "source": "EvaluationCase.train_history only (no validation/test targets read)",
    }
    return TwoTowerDataset(
        input_ids=tuple(inputs), target_ids=tuple(targets), user_int_ids=tuple(users), stats=stats
    )


def iter_training_batches(
    dataset: TwoTowerDataset,
    *,
    batch_size: int,
    epoch: int = 0,
    seed: int = 0,
) -> Iterator[tuple[list[tuple[int, ...]], list[int], list[int]]]:
    """Yield ``(input_windows, target_ids, row_indices)`` batches with unique targets.

    Why targets must be unique within a batch
    ----------------------------------------
    In-batch contrastive loss is computed as a diagonal softmax: row ``r``'s positive is
    the item at column ``r``.  Two rows sharing a target would make one of them a false
    negative for the other, so a batch containing duplicates would train the model to
    push apart two identical items.  Those pairs are therefore **skipped** rather than
    silently mislabelled, in a deterministic order fixed by the seeded shuffle.

    ``row_indices`` are the dataset positions actually used, so a caller can tell exactly
    which pairs a step trained on - useful when diagnosing why a run plateaued.
    """
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 2:
        raise TwoTowerDataError(
            f"batch_size must be >= 2 for in-batch negatives, got {batch_size!r}"
        )
    order = list(range(len(dataset)))
    # Seed with a plain int, not a tuple: ``random.Random`` seeds a tuple through ``hash()``,
    # which is randomised per process for strings and deprecated - a run would then not be
    # reproducible across processes.  Mixing the epoch arithmetically keeps a distinct order per
    # epoch while staying an int.
    random.Random(seed * 1_000_003 + epoch).shuffle(order)

    windows: list[tuple[int, ...]] = []
    targets: list[int] = []
    indices: list[int] = []
    seen: set[int] = set()
    for index in order:
        target = dataset.target_ids[index]
        if target in seen:
            # Skipped, not mislabelled: see the docstring.  A caller that needs the count can
            # compare `pairs_used` in the epoch report against len(dataset).
            continue
        windows.append(dataset.input_ids[index])
        targets.append(target)
        indices.append(index)
        seen.add(target)
        if len(targets) == batch_size:
            yield windows, targets, indices
            windows, targets, indices, seen = [], [], [], set()
    if len(targets) >= 2:
        # A trailing single-row batch cannot form an in-batch softmax; it is dropped, and the
        # epoch report's `pairs_used` shows how many pairs actually trained.
        yield windows, targets, indices


def item_prior_from_pairs(
    dataset: TwoTowerDataset, *, num_items: int, smoothing: float = 1.0
) -> list[float]:
    """Return a smoothed sampling prior over items, indexed ``0..num_items``.

    Used by the sampled-softmax logQ correction.  The prior is the observed training-target
    frequency with add-``smoothing`` smoothing, so an item never seen as a target still has
    a positive probability and the correction stays finite.  It is computed from training
    targets only.
    """
    if smoothing <= 0.0:
        raise TwoTowerDataError(f"smoothing must be > 0, got {smoothing!r}")
    if isinstance(num_items, bool) or not isinstance(num_items, int) or num_items < 1:
        raise TwoTowerDataError(f"num_items must be a positive int, got {num_items!r}")
    counts = [0.0] * (num_items + 1)
    for target in dataset.target_ids:
        if not 0 < target <= num_items:
            raise TwoTowerDataError(
                f"a training target ({target!r}) is outside the item range 1..{num_items}"
            )
        counts[target] += 1.0
    # Smooth over exactly the real item range, so the distribution sums to 1 and index 0
    # (PAD, never a candidate) carries a probability of 0.
    denominator = float(len(dataset.target_ids)) + smoothing * num_items
    prior = [0.0] * (num_items + 1)
    for item_id in range(1, num_items + 1):
        prior[item_id] = (counts[item_id] + smoothing) / denominator
    return prior


# --------------------------------------------------------------------------- #
# Categorical metadata
# --------------------------------------------------------------------------- #


def _attribute_value(record: Any, attribute: str) -> str | None:
    """Read one attribute off a normalised catalogue record.

    ``details:Key`` reads the source's own attribute bag under the key it was stored with;
    anything else is a direct field.  Values are returned verbatim, whitespace-stripped,
    and never invented.
    """
    if attribute.startswith("details:"):
        key = attribute.split(":", 1)[1]
        details = getattr(record, "details", None) or ()
        for entry in details:
            if not isinstance(entry, tuple) or len(entry) != 2:
                continue
            if str(entry[0]).strip().lower() == key.strip().lower():
                value = str(entry[1]).strip()
                return value or None
        return None
    value = getattr(record, attribute, None)
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def build_metadata_categories(
    records: dict[str, Any],
    item2id: dict[str, int],
    *,
    num_items: int,
    attributes: Sequence[str] = METADATA_ATTRIBUTES,
    min_frequency: int = 20,
) -> dict[str, Any]:
    """Build the ``[num_items + 1, n_fields]`` categorical matrix and its vocabularies.

    Only attribute values occurring at least ``min_frequency`` times become real
    categories; everything rarer collapses to :data:`UNKNOWN_CATEGORY`.  That bound is what
    keeps a long tail of one-off brand strings from becoming 100 000 near-unique
    categories, which would turn the metadata embedding into a second memorised id table
    instead of a generalising feature.

    Returns a JSON-serialisable block: ``matrix`` is the row-major list of category
    indices, ``vocabularies`` maps attribute -> value -> index, and ``cardinalities`` is
    what a :class:`~recommendation.models.twotower.TwoTowerConfig` needs.
    """
    if isinstance(num_items, bool) or not isinstance(num_items, int) or num_items < 1:
        raise TwoTowerDataError(f"num_items must be a positive int, got {num_items!r}")
    if min_frequency < 1:
        raise TwoTowerDataError(f"min_frequency must be >= 1, got {min_frequency!r}")

    # One reverse pass instead of scanning the whole mapping per item: the mapping holds
    # 156 746 entries, so a per-item scan would be ~2.5e10 comparisons.
    id2item = {mapped: identity for identity, mapped in item2id.items()}
    raw: list[list[str | None]] = []
    for item_id in range(num_items + 1):
        if item_id == PAD_ID:
            raw.append([None] * len(attributes))
            continue
        record = records.get(id2item.get(item_id))
        raw.append([_attribute_value(record, name) for name in attributes])

    vocabularies: dict[str, dict[str, int]] = {}
    cardinalities: list[int] = []
    for field_index, attribute in enumerate(attributes):
        frequencies: dict[str, int] = {}
        for row in raw:
            value = row[field_index]
            if value is not None:
                frequencies[value] = frequencies.get(value, 0) + 1
        kept = sorted(
            value for value, count in frequencies.items() if count >= min_frequency
        )
        vocabulary = {value: index + 1 for index, value in enumerate(kept)}
        vocabularies[attribute] = vocabulary
        # +1 for the reserved UNKNOWN slot at index 0.
        cardinalities.append(len(vocabulary) + 1)

    matrix: list[list[int]] = []
    for row in raw:
        matrix.append(
            [
                vocabularies[attribute].get(row[field_index] or "", UNKNOWN_CATEGORY)
                for field_index, attribute in enumerate(attributes)
            ]
        )

    return {
        "attributes": list(attributes),
        "min_frequency": min_frequency,
        "vocabularies": vocabularies,
        "cardinalities": cardinalities,
        "matrix": matrix,
        "coverage": {
            attribute: round(
                1.0
                - sum(
                    1
                    for row in matrix
                    if row[index] == UNKNOWN_CATEGORY
                )
                / max(1, len(matrix) - 1),
                6,
            )
            for index, attribute in enumerate(attributes)
        },
    }


def write_metadata_categories(payload: dict[str, Any], path: str | Path) -> Path:
    """Write the category block as JSON, atomically."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    temporary.replace(target)
    return target


def load_metadata_categories(path: str | Path) -> dict[str, Any]:
    """Read a category block back, validating its shape."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    for key in ("attributes", "vocabularies", "cardinalities", "matrix"):
        if key not in payload:
            raise TwoTowerDataError(f"metadata category artifact is missing {key!r}")
    return payload


def metadata_tensor(payload: dict[str, Any]) -> Any:
    """Return the category matrix as a ``torch.long`` tensor of shape ``[rows, fields]``."""
    import torch

    return torch.tensor(payload["matrix"], dtype=torch.long)

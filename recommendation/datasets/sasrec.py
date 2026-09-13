"""SASRec training dataset and inference encoding (Milestone 3, Part A).

This module owns the SASRec **data contract**: how a Milestone 2A evaluation case
becomes a fixed-length training sample, and how an inference history becomes a
fixed-length input.  It contains no model and no evaluation logic.

Source-of-truth split
---------------------
The only permitted input is ``EvaluationCase.train_history``::

    for [i1, ..., i(n-2), i(n-1), in]  with n >= 3:
        train_history      = [i1, ..., i(n-2)]     <-- the ONLY training input
        validation_target  = i(n-1)                <-- held out
        test_target        = in                    <-- held out

``validation_target`` and ``test_target`` are never consulted - not for sequence
construction, not for truncation, not for the negative-sampling exclusion set, not
for statistics.  The builders below take ``EvaluationCase`` objects and read only
``train_history``, which makes that guarantee structural rather than a convention.
This matters because excluding a *future* target from negatives would itself be
temporal leakage.

Trainable-user policy
---------------------
Evaluation eligibility is ``sequence length >= 3``, so every evaluation user has
``len(train_history) >= 1``.  A SASRec sample needs at least one next-item
transition, i.e. ``len(train_history) >= 2``:

* ``len(train_history) == 1`` -> zero transitions, contributes **no** gradient
  sample, and is reported as such;
* such a user is **not** removed from the validation/test evaluation cohort.

Sequence construction
---------------------
For a (possibly truncated) history ``[h1, ..., hm]`` with ``m >= 2``::

    input_ids    = [h1, ..., h(m-1)]
    positive_ids = [h2, ..., hm]

so ``positive_ids[t]`` is the item that followed ``input_ids[t]``.  Arrays are
``max_seq_len`` long and **left-padded** with PAD (0); when truncation is needed the
most recent transitions are kept.  ``positive_ids == 0`` marks a padding position,
which is exactly the mask later training uses to exclude padded positions from the
loss.
"""

from __future__ import annotations

import hashlib
import random
from functools import lru_cache
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

from recommendation import config
from recommendation.evaluation.split import EvaluationCase

#: PAD item id; not a real catalog item.
PAD_ID = config.PAD_ID

#: Minimum train-history length that yields at least one transition.
MIN_TRAIN_HISTORY_FOR_TRANSITION = 2


class SASRecDataError(ValueError):
    """Raised when histories or ids violate the SASRec data contract."""


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class SASRecDatasetConfig:
    """Configuration for SASRec sequence construction and negative sampling.

    Attributes
    ----------
    max_seq_len:
        Fixed window length for both training and inference arrays.
    seed:
        Master seed for negative sampling.
    epoch:
        Optional epoch component.  Negative sampling hashes
        ``(user_int_id, position, epoch, seed)``, so bumping ``epoch`` resamples
        negatives deterministically without touching anything else.  No training
        loop is implemented in this milestone.
    negative_retry_limit:
        Number of rejection-sampling draws attempted before falling back to a
        deterministic scan over the candidate pool.  The fallback guarantees an
        explicit, terminating outcome when valid negatives are extremely scarce.
    """

    max_seq_len: int = 50
    seed: int = 0
    epoch: int = 0
    negative_retry_limit: int = 32

    def __post_init__(self) -> None:
        for name in ("max_seq_len", "negative_retry_limit"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise SASRecDataError(f"{name} must be a positive int, got {value!r}")
        for name in ("seed", "epoch"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise SASRecDataError(f"{name} must be an int, got {type(value).__name__}")


# --------------------------------------------------------------------------- #
# Sample / report containers
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class SASRecSample:
    """One user's fixed-length SASRec training sample.

    Contains no validation/test target information by construction.
    """

    user_int_id: int
    input_ids: tuple[int, ...]
    positive_ids: tuple[int, ...]
    negative_ids: tuple[int, ...]
    num_valid_positions: int

    @property
    def valid_mask(self) -> tuple[bool, ...]:
        """True where the position carries a real (non-padding) transition."""
        return tuple(positive != PAD_ID for positive in self.positive_ids)

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable view of the sample."""
        return {
            "user_int_id": self.user_int_id,
            "input_ids": list(self.input_ids),
            "positive_ids": list(self.positive_ids),
            "negative_ids": list(self.negative_ids),
            "num_valid_positions": self.num_valid_positions,
        }


@dataclass
class SASRecDatasetStats:
    """Auditable description of how the dataset was built."""

    config: SASRecDatasetConfig
    evaluation_users: int
    trainable_users: int
    users_with_zero_transitions: int
    users_with_one_train_item: int
    raw_train_interactions: int
    raw_next_item_transitions: int
    effective_transitions: int
    users_truncated: int
    train_interactions_in_window: int
    num_items: int

    @property
    def transition_retention(self) -> float:
        """Fraction of raw transitions that survive the ``max_seq_len`` window."""
        if self.raw_next_item_transitions == 0:
            return 0.0
        return self.effective_transitions / self.raw_next_item_transitions

    @property
    def interaction_retention(self) -> float:
        """Fraction of raw train interactions inside the window."""
        if self.raw_train_interactions == 0:
            return 0.0
        return self.train_interactions_in_window / self.raw_train_interactions

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable view of the statistics."""
        return {
            "max_seq_len": self.config.max_seq_len,
            "seed": self.config.seed,
            "epoch": self.config.epoch,
            "num_items": self.num_items,
            "evaluation_users": self.evaluation_users,
            "trainable_users": self.trainable_users,
            "users_with_zero_transitions": self.users_with_zero_transitions,
            "users_with_one_train_item": self.users_with_one_train_item,
            "raw_train_interactions": self.raw_train_interactions,
            "raw_next_item_transitions": self.raw_next_item_transitions,
            "effective_transitions": self.effective_transitions,
            "users_truncated": self.users_truncated,
            "train_interactions_in_window": self.train_interactions_in_window,
            "transition_retention": round(self.transition_retention, 6),
            "interaction_retention": round(self.interaction_retention, 6),
        }


@dataclass
class SASRecDataset:
    """A built SASRec training dataset."""

    samples: list[SASRecSample]
    num_items: int
    stats: SASRecDatasetStats
    user_int_ids: list[int] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.samples)

    def __iter__(self):
        return iter(self.samples)

    @property
    def total_valid_positions(self) -> int:
        """Total gradient-producing positions across the dataset."""
        return sum(sample.num_valid_positions for sample in self.samples)

    def digest(self) -> str:
        """Stable SHA-256 over the sample contents (for reproducibility checks)."""
        hasher = hashlib.sha256()
        for sample in self.samples:
            hasher.update(
                f"{sample.user_int_id}|{sample.input_ids}|{sample.positive_ids}|"
                f"{sample.negative_ids}\n".encode()
            )
        return hasher.hexdigest()


# --------------------------------------------------------------------------- #
# Behavioural / structural digest of source cases
# --------------------------------------------------------------------------- #


def cohort_structure_digest(cases: Sequence[EvaluationCase]) -> str:
    """Hash the *structural* cohort fields used by training.

    Deliberately hashes only ``train_history`` - never the targets - so a digest
    match proves the dataset depends on nothing else.
    """
    hasher = hashlib.sha256()
    for case in cases:
        hasher.update(f"{case.user_int_id}|{case.train_history}\n".encode())
    return hasher.hexdigest()


def _validate_train_history(history: Sequence[int], num_items: int, context: str) -> tuple[int, ...]:
    """Validate a training history; PAD and out-of-catalog ids are rejected."""
    if isinstance(num_items, bool) or not isinstance(num_items, int) or num_items < 1:
        raise SASRecDataError(f"num_items must be a positive int, got {num_items!r}")
    items = tuple(history)
    for item_id in items:
        if isinstance(item_id, bool) or not isinstance(item_id, int):
            raise SASRecDataError(
                f"{context} entry must be an int, got {type(item_id).__name__}: {item_id!r}"
            )
        if item_id == PAD_ID:
            raise SASRecDataError(
                f"{context} contains PAD {PAD_ID}; PAD is not a real catalog item"
            )
        if not config.FIRST_REAL_ID <= item_id <= num_items:
            raise SASRecDataError(
                f"{context} entry {item_id} outside catalog "
                f"[{config.FIRST_REAL_ID}, {num_items}]"
            )
    return items


# --------------------------------------------------------------------------- #
# Negative sampling
# --------------------------------------------------------------------------- #


@lru_cache(maxsize=64)
def _catalog_set(num_items: int) -> frozenset[int]:
    """Cached ``{1..num_items}`` used only to validate exclusion sets."""
    return frozenset(range(config.FIRST_REAL_ID, num_items + 1))


@lru_cache(maxsize=200_000)
def _negative_seed(user_int_id: int, position: int, epoch: int, seed: int) -> int:
    """Derive a stable per-(user, position, epoch, seed) RNG seed.

    A hash rather than arithmetic mixing, so neighbouring positions do not produce
    correlated draws and the result is identical on any platform/Python version for
    a given sample identity.
    """
    payload = f"agentrecx.sasrec.neg|{seed}|{epoch}|{user_int_id}|{position}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def sample_negative(
    user_int_id: int,
    position: int,
    exclusion: frozenset[int],
    num_items: int,
    config_: SASRecDatasetConfig,
) -> int:
    """Draw one deterministic negative item for a training position.

    The candidate pool is ``{1..num_items} - exclusion``, where ``exclusion`` is the
    user's **train history**.  Validation/test targets are not part of the
    exclusion set: they are unknown at training time, and excluding them would leak
    the future into training.

    Sampling is a function of ``(user_int_id, position, config.epoch, config.seed,
    exclusion)`` only, so it is reproducible and can be resampled per epoch.

    Raises
    ------
    SASRecDataError
        If the pool is empty (every catalog item is in the exclusion set).  The
        fallback scan means the function always terminates; it never spins.
    """
    if exclusion and not exclusion <= _catalog_set(num_items):
        raise SASRecDataError("exclusion set contains ids outside the catalog")

    pool_size = num_items - len(exclusion)
    if pool_size <= 0:
        raise SASRecDataError(
            f"no valid negative item exists for user {user_int_id}: all "
            f"{num_items} catalog items are in the train history"
        )

    rng = random.Random(_negative_seed(user_int_id, position, config_.epoch, config_.seed))
    # Fast path: rejection sampling straight off the full id range.  Building the
    # candidate list here would cost O(num_items) for every training position, which
    # dominates dataset construction; it is only needed for the rare fallback.
    for _ in range(config_.negative_retry_limit):
        candidate = rng.randrange(config.FIRST_REAL_ID, num_items + 1)
        if candidate not in exclusion:
            return candidate

    # Deterministic fallback for pathologically small pools: a uniformly random
    # offset into the allowed list.  Still a pure function of the sample identity,
    # and guaranteed to terminate instead of spinning.
    allowed = [i for i in range(config.FIRST_REAL_ID, num_items + 1) if i not in exclusion]
    return allowed[rng.randrange(len(allowed))]


# --------------------------------------------------------------------------- #
# Training sequence construction
# --------------------------------------------------------------------------- #


def build_arrays(
    history: Sequence[int],
    max_seq_len: int,
) -> tuple[tuple[int, ...], tuple[int, ...], int]:
    """Build aligned, left-padded ``(input_ids, positive_ids, num_valid)``.

    For ``history = [h1, ..., hm]`` with ``m >= 2``::

        input_ids    = left_pad([h1, ..., h(m-1)])
        positive_ids = left_pad([h2, ..., hm])

    Truncation keeps the most recent transitions by applying the window to the
    *shifted* arrays, so an input and its next-item positive always stay aligned::

        history = [1,2,3,4], max_seq_len = 5
        -> input_ids    = [0,0,1,2,3]
           positive_ids = [0,0,2,3,4]

        history = [1,2,3,4], max_seq_len = 3
        -> most recent transitions are (2->3) and (3->4)
           input_ids    = [0,2,3]
           positive_ids = [0,3,4]

    Raises
    ------
    SASRecDataError
        If ``max_seq_len < 1`` or the history has fewer than two items (no
        transition exists).
    """
    if isinstance(max_seq_len, bool) or not isinstance(max_seq_len, int) or max_seq_len < 1:
        raise SASRecDataError(f"max_seq_len must be a positive int, got {max_seq_len!r}")

    items = tuple(history)
    if len(items) < MIN_TRAIN_HISTORY_FOR_TRANSITION:
        raise SASRecDataError(
            f"history of length {len(items)} has no next-item transition "
            f"(need >= {MIN_TRAIN_HISTORY_FOR_TRANSITION})"
        )

    inputs = items[:-1]
    positives = items[1:]

    if len(inputs) > max_seq_len:
        inputs = inputs[-max_seq_len:]
        positives = positives[-max_seq_len:]

    pad_count = max_seq_len - len(inputs)
    input_ids = (PAD_ID,) * pad_count + inputs
    positive_ids = (PAD_ID,) * pad_count + positives
    return input_ids, positive_ids, len(inputs)


def build_sample(
    user_int_id: int,
    train_history: Sequence[int],
    num_items: int,
    config_: SASRecDatasetConfig,
) -> SASRecSample:
    """Build one user's sample: arrays plus one negative per valid position."""
    history = _validate_train_history(train_history, num_items, f"train history for user {user_int_id}")
    input_ids, positive_ids, num_valid = build_arrays(history, config_.max_seq_len)

    exclusion = frozenset(history)
    negatives: list[int] = []
    valid_index = 0
    for position, positive in enumerate(positive_ids):
        if positive == PAD_ID:
            negatives.append(PAD_ID)  # padding slot stays padding
            continue
        negatives.append(
            sample_negative(user_int_id, valid_index, exclusion, num_items, config_)
        )
        valid_index += 1

    return SASRecSample(
        user_int_id=user_int_id,
        input_ids=input_ids,
        positive_ids=positive_ids,
        negative_ids=tuple(negatives),
        num_valid_positions=num_valid,
    )


def build_dataset(
    cases: Sequence[EvaluationCase],
    num_items: int,
    config_: SASRecDatasetConfig | None = None,
) -> SASRecDataset:
    """Build a SASRec training dataset from a Milestone 2A cohort.

    Reads **only** ``case.train_history``.  Users with fewer than two train items
    produce no sample (zero transitions) but remain in the evaluation cohort.

    Deterministic: samples follow ascending ``user_int_id`` regardless of input
    order, and negative sampling depends only on explicit seeds and sample identity.
    """
    config_ = config_ or SASRecDatasetConfig()
    if isinstance(num_items, bool) or not isinstance(num_items, int) or num_items < 1:
        raise SASRecDataError(f"num_items must be a positive int, got {num_items!r}")

    ordered = sorted(cases, key=lambda case: (case.user_int_id, case.user_id))

    samples: list[SASRecSample] = []
    raw_interactions = 0
    raw_transitions = 0
    effective_transitions = 0
    in_window = 0
    truncated = 0
    zero_transition = 0
    one_train_item = 0

    for case in ordered:
        history = _validate_train_history(
            case.train_history, num_items, f"train history for user {case.user_int_id}"
        )
        raw_interactions += len(history)

        if len(history) == 1:
            one_train_item += 1
        if len(history) < MIN_TRAIN_HISTORY_FOR_TRANSITION:
            zero_transition += 1
            continue

        raw_transitions += len(history) - 1
        sample = build_sample(case.user_int_id, history, num_items, config_)
        samples.append(sample)
        effective_transitions += sample.num_valid_positions
        # Interactions inside the window: the model sees at most the newest
        # (max_seq_len + 1) history items (max_seq_len inputs plus their targets).
        in_window += min(len(history), config_.max_seq_len + 1)
        if sample.num_valid_positions < len(history) - 1:
            truncated += 1

    stats = SASRecDatasetStats(
        config=config_,
        evaluation_users=len(cases),
        trainable_users=len(samples),
        users_with_zero_transitions=zero_transition,
        users_with_one_train_item=one_train_item,
        raw_train_interactions=raw_interactions,
        raw_next_item_transitions=raw_transitions,
        effective_transitions=effective_transitions,
        users_truncated=truncated,
        train_interactions_in_window=in_window,
        num_items=num_items,
    )
    return SASRecDataset(
        samples=samples,
        num_items=num_items,
        stats=stats,
        user_int_ids=[sample.user_int_id for sample in samples],
    )


# --------------------------------------------------------------------------- #
# Inference encoding
# --------------------------------------------------------------------------- #


def encode_inference_history(
    history: Sequence[int],
    max_seq_len: int,
    num_items: int,
) -> tuple[int, ...]:
    """Encode an inference history into a fixed-length, left-padded input.

    * keeps the **most recent** ``max_seq_len`` items;
    * left-pads shorter histories with PAD ``0``;
    * preserves chronological order;
    * rejects an empty history and any invalid item id;
    * never appends, reads or inspects a target item - the caller supplies history
      only, so no target can enter the encoding.

    This is deliberately a *different* operation from :func:`build_arrays`: training
    needs a shifted input/target pair, inference needs a single next-item input.
    """
    if isinstance(max_seq_len, bool) or not isinstance(max_seq_len, int) or max_seq_len < 1:
        raise SASRecDataError(f"max_seq_len must be a positive int, got {max_seq_len!r}")

    items = _validate_train_history(history, num_items, "inference history")
    if not items:
        raise SASRecDataError("inference history is empty; at least one item is required")

    if len(items) > max_seq_len:
        items = items[-max_seq_len:]
    pad_count = max_seq_len - len(items)
    return (PAD_ID,) * pad_count + items


def encode_batch(
    histories: Iterable[Sequence[int]],
    max_seq_len: int,
    num_items: int,
) -> list[tuple[int, ...]]:
    """Encode several inference histories, preserving input order."""
    return [encode_inference_history(h, max_seq_len, num_items) for h in histories]


def validation_history(case: EvaluationCase) -> tuple[int, ...]:
    """Return the validation-mode inference history (``train_history``)."""
    return tuple(case.train_history)


def test_history(case: EvaluationCase) -> tuple[int, ...]:
    """Return the test-mode inference history (``train_history + [validation_target]``).

    The validation interaction precedes the test target, so it is legitimate history
    at test time.  The test target itself is never included.
    """
    return tuple(case.train_history) + (case.validation_target,)

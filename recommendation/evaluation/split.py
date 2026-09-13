"""Deterministic temporal leave-two-out splitting.

This module owns the *split* half of the unified evaluation protocol described in
``AGENTS.md`` section 5/6.  It is model independent: no recommender logic lives
here.

Protocol
--------
For a chronological user sequence ``[i1, i2, ..., i(n-1), in]`` with ``n >= 3``::

    train_history       = [i1, ..., i(n-2)]
    validation          : history = train_history,               target = i(n-1)
    test                : history = train_history + [i(n-1)],    target = in

Both targets are the *last two* interactions in time, which is why this is called
leave-**two**-out: validation is evaluated against the second-to-last interaction
and test against the last, so the test case's history contains only events that
happened strictly before its target.

Users whose sequence is shorter than 3 are excluded from the evaluation cohort
here, at the evaluation layer.  Preprocessing thresholds and artifacts are never
touched to satisfy this rule.

Determinism
-----------
Input sequences are never mutated: every split copies its items into new tuples.
Cases are emitted in ascending order of ``(user_int_id, user_id)``, so the cohort
and its ordering do not depend on dictionary insertion order or on the order in
which the caller supplied the sequences.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from recommendation import config

#: Sequence length required for a user to be evaluation-eligible.
MIN_SEQUENCE_LENGTH = 3

#: Human-readable label for the evaluation protocol implemented here.
PROTOCOL_NAME = "temporal_leave_two_out"
PROTOCOL_VERSION = "agentrecx.eval_protocol.v1"


# --------------------------------------------------------------------------- #
# Errors
# --------------------------------------------------------------------------- #


class SplitError(ValueError):
    """Raised when input sequences or item ids violate the evaluation contract."""


# --------------------------------------------------------------------------- #
# Cases
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class EvaluationCase:
    """One user's leave-two-out evaluation case.

    Attributes
    ----------
    user_id:
        Raw user identifier as stored in the preprocessing artifact.
    user_int_id:
        Deterministic integer user id from ``user2id``.
    train_history:
        Chronological items preceding both targets (``i1..i(n-2)``).
    validation_target:
        ``i(n-1)`` - the second-to-last interaction.
    test_target:
        ``in`` - the last interaction.
    sequence_length:
        ``n``, the length of the original chronological sequence.
    """

    user_id: str
    user_int_id: int
    train_history: tuple[int, ...]
    validation_target: int
    test_target: int
    sequence_length: int

    @property
    def validation_history(self) -> tuple[int, ...]:
        """History available when predicting :attr:`validation_target`."""
        return self.train_history

    @property
    def test_history(self) -> tuple[int, ...]:
        """History available when predicting :attr:`test_target`.

        This is ``train_history + (validation_target,)`` - the validation
        interaction happened before the test interaction, so it is legitimately
        visible at test time.
        """
        return self.train_history + (self.validation_target,)

    @property
    def validation_seen(self) -> frozenset[int]:
        """Items the user already consumed before the validation target."""
        return frozenset(self.train_history)

    @property
    def test_seen(self) -> frozenset[int]:
        """Items the user already consumed before the test target."""
        return frozenset(self.test_history)


@dataclass(frozen=True)
class SplitReport:
    """Summary of one split run, suitable for reporting and artifact logging."""

    protocol: str
    num_users_total: int
    num_users_eligible: int
    num_users_excluded: int
    min_sequence_length: int
    num_validation_cases: int
    num_test_cases: int
    catalog_size: int

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable view of the split summary."""
        return {
            "protocol": self.protocol,
            "protocol_version": PROTOCOL_VERSION,
            "min_sequence_length": self.min_sequence_length,
            "num_users_total": self.num_users_total,
            "num_users_eligible": self.num_users_eligible,
            "num_users_excluded": self.num_users_excluded,
            "num_validation_cases": self.num_validation_cases,
            "num_test_cases": self.num_test_cases,
            "catalog_size": self.catalog_size,
        }


# --------------------------------------------------------------------------- #
# Catalog
# --------------------------------------------------------------------------- #


def build_catalog(num_items: int) -> tuple[int, ...]:
    """Return the evaluation catalog ``(1, 2, ..., num_items)``.

    Item id :data:`recommendation.config.PAD_ID` (0) is *never* part of the
    catalog: it is a padding slot, not a rankable item.

    Raises
    ------
    SplitError
        If ``num_items`` is not a positive integer.
    """
    if isinstance(num_items, bool) or not isinstance(num_items, int):
        raise SplitError(f"num_items must be an int, got {type(num_items).__name__}")
    if num_items < 1:
        raise SplitError(f"num_items must be >= 1, got {num_items}")
    return tuple(range(config.FIRST_REAL_ID, config.FIRST_REAL_ID + num_items))


def validate_item_ids(
    items: Iterable[int],
    num_items: int,
    *,
    context: str = "item id",
) -> None:
    """Raise :class:`SplitError` unless every id is a real catalog item.

    ``0`` (PAD), negatives, ``> num_items`` and non-integers are all rejected
    explicitly rather than silently ignored, so a corrupt artifact cannot quietly
    shrink or distort the evaluation cohort.
    """
    if isinstance(num_items, bool) or not isinstance(num_items, int) or num_items < 1:
        raise SplitError(f"num_items must be a positive int, got {num_items!r}")
    for item_id in items:
        if isinstance(item_id, bool) or not isinstance(item_id, int):
            raise SplitError(f"{context} must be an int, got {type(item_id).__name__}: {item_id!r}")
        if item_id == config.PAD_ID:
            raise SplitError(
                f"{context} {item_id} is the PAD slot and must never appear in an "
                "evaluation sequence"
            )
        if not config.FIRST_REAL_ID <= item_id <= num_items:
            raise SplitError(
                f"{context} {item_id} outside the catalog "
                f"[{config.FIRST_REAL_ID}, {num_items}]"
            )


# --------------------------------------------------------------------------- #
# Splitting
# --------------------------------------------------------------------------- #


def split_sequence(
    user_id: str,
    items: Sequence[int],
    num_items: int,
    user_int_id: int = -1,
) -> EvaluationCase | None:
    """Split one chronological sequence into a leave-two-out case.

    Returns ``None`` when the sequence is too short to be evaluation-eligible
    (``len(items) < MIN_SEQUENCE_LENGTH``).  Longer sequences are never truncated,
    so a future protocol variant can reuse the full history.

    Raises
    ------
    SplitError
        If any item id violates the catalog contract (in particular PAD ``0``).
    """
    if len(items) < MIN_SEQUENCE_LENGTH:
        return None

    ordered = tuple(items)
    validate_item_ids(ordered, num_items, context=f"item id for user {user_id!r}")

    return EvaluationCase(
        user_id=user_id,
        user_int_id=user_int_id,
        train_history=ordered[:-2],
        validation_target=ordered[-2],
        test_target=ordered[-1],
        sequence_length=len(ordered),
    )


def split_cohort(
    sequences: Mapping[str, Sequence[int]] | Iterable[tuple[str, Sequence[int]]],
    num_items: int,
    user_int_ids: Mapping[str, int] | None = None,
) -> tuple[list[EvaluationCase], SplitReport]:
    """Split a whole cohort of users, returning ``(cases, report)``.

    Parameters
    ----------
    sequences:
        Either a mapping ``user_id -> chronological item ids`` or an iterable of
        ``(user_id, item ids)`` pairs.
    num_items:
        Catalogue size from the preprocessing artifact; defines the catalog as
        ``1..num_items``.
    user_int_ids:
        Optional ``user_id -> int`` mapping (from the artifact's ``user2id``).
        Users missing from it receive ``-1``.

    The returned cases are sorted by ``(user_int_id, user_id)`` so the result is
    independent of input order and of dictionary insertion order.
    """
    build_catalog(num_items)  # validates num_items up front (and defines the PAD rule)

    items_iter: Iterable[tuple[str, Sequence[int]]]
    if isinstance(sequences, Mapping):
        items_iter = sequences.items()
    else:
        items_iter = sequences

    cases: list[EvaluationCase] = []
    total = 0
    eligible = 0
    for user_id, items in items_iter:
        total += 1
        case = split_sequence(
            user_id,
            items,
            num_items,
            user_int_id=(user_int_ids or {}).get(user_id, -1),
        )
        if case is None:
            continue
        eligible += 1
        cases.append(case)

    cases.sort(key=lambda case: (case.user_int_id, case.user_id))

    report = SplitReport(
        protocol=PROTOCOL_NAME,
        num_users_total=total,
        num_users_eligible=eligible,
        num_users_excluded=total - eligible,
        min_sequence_length=MIN_SEQUENCE_LENGTH,
        num_validation_cases=len(cases),
        num_test_cases=len(cases),
        catalog_size=num_items,
    )
    return cases, report


def train_history_statistics(cases: Sequence[EvaluationCase]) -> dict[str, Any]:
    """Return min/max/mean/total length statistics for train histories."""
    lengths = [len(case.train_history) for case in cases]
    if not lengths:
        return {"count": 0, "min": None, "max": None, "mean": None, "total": 0}
    return {
        "count": len(lengths),
        "min": min(lengths),
        "max": max(lengths),
        "mean": round(sum(lengths) / len(lengths), 6),
        "total": sum(lengths),
    }


# --------------------------------------------------------------------------- #
# Loading from preprocessing artifacts
# --------------------------------------------------------------------------- #


def load_sequences_artifact(path: str | Path) -> dict[str, Any]:
    """Load a ``*_sequences.json`` file produced by the preprocessing pipeline."""
    with open(Path(path), "rt", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict) or "sequences" not in payload:
        raise SplitError(f"{path} is not an agentrecx sequences artifact")
    return payload


def sequences_from_artifact(payload: Mapping[str, Any]) -> dict[str, list[int]]:
    """Extract ``user_id -> item_ids`` from a sequences artifact payload."""
    out: dict[str, list[int]] = {}
    for record in payload["sequences"]:
        out[record["user_id"]] = list(record["item_ids"])
    return out


def user_int_ids_from_artifact(payload: Mapping[str, Any]) -> dict[str, int]:
    """Extract ``user_id -> user_int_id`` from a sequences artifact payload."""
    return {record["user_id"]: record["user_int_id"] for record in payload["sequences"]}


def load_cohort_from_artifacts(
    sequences_path: str | Path,
    mappings_path: str | Path | None = None,
) -> tuple[list[EvaluationCase], SplitReport]:
    """Load a preprocessing run and split it into evaluation cases.

    ``num_items`` comes from the sequences artifact when present, otherwise from
    the mappings artifact.  When both are supplied they must agree.
    """
    payload = load_sequences_artifact(sequences_path)
    sequences = sequences_from_artifact(payload)
    user_int_ids = user_int_ids_from_artifact(payload)

    num_items = payload.get("num_items")
    if mappings_path is not None:
        with open(Path(mappings_path), "rt", encoding="utf-8") as handle:
            mappings = json.load(handle)
        mapping_num_items = mappings.get("num_items")
        if num_items is not None and mapping_num_items is not None and num_items != mapping_num_items:
            raise SplitError(
                f"num_items disagreement: sequences says {num_items}, "
                f"mappings says {mapping_num_items}"
            )
        num_items = num_items if num_items is not None else mapping_num_items

    if num_items is None:
        raise SplitError("could not determine num_items from the supplied artifacts")

    return split_cohort(sequences, int(num_items), user_int_ids=user_int_ids)

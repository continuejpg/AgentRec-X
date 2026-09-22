"""Deterministic Semantic-ID collision de-duplication (Step 2.4).

The quantizer maps items onto ``K ** levels`` codes, which is far smaller than a real
catalogue, so several items routinely share one pre-dedup code.  A shared code is a real limit
on what the code space can distinguish, and it must be resolved before a code can be treated
as an item identity.

The resolution is one extra digit, assigned as a **collision ordinal**:

```text
(c1, c2, c3)  ->  (c1, c2, c3, ordinal)
```

Three properties are load-bearing, and each is asserted rather than assumed:

* **first member gets 0.**  The first item in ascending ``item_id`` order within a group is
  ordinal ``0``; later members get ``1, 2, ...``.  (GRID's reference implementation assigns
  ``1..k`` to *every* member including the first.  That is still collision-free, but it wastes
  a code and makes the ordinal depend on an implementation detail rather than on position, so
  this module normalises it.)
* **the ordering is deterministic**, by ``item_id``, so the same catalogue and the same codes
  always produce the same assignment.
* **overflow refuses.**  A group larger than the dedup vocabulary cannot be disambiguated.  The
  build stops with an explicit error naming the group size; it never wraps modulo, truncates,
  or leaves two items sharing a final identity.

The ordinal is **not** a semantic token: it carries no learned meaning, it is not produced by
the quantizer, and it is not part of the reconstruction objective.  It exists solely so that a
Semantic ID can identify one item.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

__all__ = [
    "DedupOverflowError",
    "DedupResult",
    "assign_dedup_digits",
    "largest_collision_group",
]

#: Default width of the dedup level.  A power of two so its token range is a clean offset block
#: above the code space; the build may widen it if the data requires.
DEFAULT_DEDUP_VOCAB_SIZE = 256


class DedupOverflowError(ValueError):
    """Raised when a collision group cannot be disambiguated within the dedup vocabulary."""


@dataclass(frozen=True)
class DedupResult:
    """The extended assignment plus what the pass observed.

    ``pre_codes`` keeps the quantizer's own codes untouched, so the representation-quality
    diagnostic survives the dedup step rather than being overwritten by it.
    """

    pre_codes: tuple[tuple[int, ...], ...]
    #: ``assignment[row]`` is the dedup-extended code; row order matches ``pre_codes``.
    assignment: tuple[tuple[int, ...], ...]
    #: Ordinals in the same order, for callers that want the digit alone.
    ordinals: tuple[int, ...]
    dedup_vocab_size: int
    pre_largest_group: int
    pre_collision_groups: int

    @property
    def levels(self) -> int:
        return len(self.pre_codes[0]) if self.pre_codes else 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "dedup_vocab_size": self.dedup_vocab_size,
            "pre_largest_group": self.pre_largest_group,
            "pre_collision_groups": self.pre_collision_groups,
            "first_member_zero": True,
            "scheme": "collision_ordinal",
        }


def largest_collision_group(codes: Sequence[Sequence[int]]) -> int:
    """The size of the biggest group sharing one pre-dedup code (1 when there are none)."""
    counts: dict[tuple[int, ...], int] = {}
    for code in codes:
        key = tuple(int(value) for value in code)
        counts[key] = counts.get(key, 0) + 1
    return max(counts.values()) if counts else 0


def assign_dedup_digits(
    pre_codes: Sequence[Sequence[int]],
    *,
    dedup_vocab_size: int = DEFAULT_DEDUP_VOCAB_SIZE,
) -> DedupResult:
    """Extend each pre-dedup code with a deterministic collision ordinal.

    Parameters
    ----------
    pre_codes:
        The quantizer's codes, **indexed by item id and excluding the reserved PAD row**.  Row
        ``i`` here is item ``i + 1`` of the catalogue handoff.
    dedup_vocab_size:
        How many distinct ordinals the dedup level can express.  A group larger than this
        refuses the build; it is never wrapped or shared.

    Raises
    ------
    DedupOverflowError
        When the largest pre-dedup collision group exceeds ``dedup_vocab_size``.
    """
    if dedup_vocab_size < 1:
        raise DedupOverflowError(f"dedup_vocab_size must be >= 1, got {dedup_vocab_size}")
    if not pre_codes:
        raise DedupOverflowError("cannot assign dedup digits to an empty assignment")
    width = len(pre_codes[0])
    normalised: list[tuple[int, ...]] = []
    for index, code in enumerate(pre_codes):
        if len(code) != width:
            raise DedupOverflowError(
                f"code {index} has {len(code)} digits; every code must have {width}"
            )
        key = tuple(int(value) for value in code)
        if any(value < 0 for value in key):
            raise DedupOverflowError(
                f"code {index} holds a negative digit; only the reserved PAD row may carry the "
                "sentinel, and PAD is never quantised"
            )
        normalised.append(key)

    counts: dict[tuple[int, ...], int] = {}
    for key in normalised:
        counts[key] = counts.get(key, 0) + 1
    largest = max(counts.values())
    collision_groups = sum(1 for count in counts.values() if count > 1)

    if largest > dedup_vocab_size:
        raise DedupOverflowError(
            f"largest pre-dedup collision group {largest} exceeds dedup_vocab_size "
            f"{dedup_vocab_size}; re-fit with a larger codebook, more levels, or a wider dedup "
            "level - a shared final Semantic ID is not an acceptable fallback"
        )

    seen: dict[tuple[int, ...], int] = {}
    assignment: list[tuple[int, ...]] = []
    ordinals: list[int] = []
    for key in normalised:
        ordinal = seen.get(key, 0)
        seen[key] = ordinal + 1
        ordinals.append(ordinal)
        assignment.append(key + (ordinal,))

    return DedupResult(
        pre_codes=tuple(normalised),
        assignment=tuple(assignment),
        ordinals=tuple(ordinals),
        dedup_vocab_size=int(dedup_vocab_size),
        pre_largest_group=largest,
        pre_collision_groups=collision_groups,
    )

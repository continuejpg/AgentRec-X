"""Deterministic Semantic-ID -> catalogue item resolution.

Why this module exists separately from the generator
----------------------------------------------------
A generative model produces **tokens**, and tokens are not products.  If the model's output were
treated as an item identity directly, then a hallucinated code would become a recommendable
product and a plausible-looking answer could reference something that does not exist.  The
resolution step is therefore:

* **separate** - a different module with no reference to the model, so it cannot be "trained
  around" and its behaviour is auditable on its own;
* **deterministic** - a pure function of the Semantic ID and the trusted catalogue, with no
  sampling, no scores and no learned component;
* **fail-closed** - a code that no catalogue item occupies resolves to *nothing*.  It never
  returns a nearest neighbour, a partial match or a default item, because every one of those
  would invent a recommendation the evidence does not support.

What it resolves *from*
-----------------------
The **trusted catalogue**, expressed as the set of item ids the tokenizer assigned codes to.  The
resolver is built from that assignment and never mutates it, so the set of resolvable codes is
exactly the set of codes the catalogue produced - no more, no less.

Collisions
----------
When several items share one Semantic ID the resolver must still be deterministic.  The rule is
:data:`COLLISION_RULE`: prefer the item that is **eligible**, and among those the **lowest item
id**.  A caller may supply an eligibility predicate (for example "not already consumed"), which is
how the benchmark keeps seen-item masking in the evaluator's hands rather than in the resolver's.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable, Sequence

__all__ = [
    "COLLISION_RULE",
    "SidResolver",
    "Resolution",
]

#: The documented tie-break for a colliding Semantic ID.  Stated as a constant so a reader can
#: find it, and asserted by a test so it cannot drift silently.
COLLISION_RULE = "eligible items first, then ascending item id"


@dataclass(frozen=True)
class Resolution:
    """The outcome of resolving one generated Semantic ID."""

    semantic_id: tuple[int, ...]
    #: The resolved catalogue item id, or ``None`` when the code is not a catalogue code.
    item_id: int | None
    #: How many catalogue items share this code (1 means unambiguous).
    collision_size: int
    #: ``True`` when the code occupies no catalogue item and resolution failed closed.
    invalid: bool

    def as_dict(self) -> dict[str, Any]:
        return {
            "semantic_id": list(self.semantic_id),
            "item_id": self.item_id,
            "collision_size": self.collision_size,
            "invalid": self.invalid,
        }


class SidResolver:
    """Resolve Semantic IDs against the trusted catalogue, deterministically.

    Parameters
    ----------
    assignment:
        ``codes[item_id]`` is that item's Semantic ID, indexed by item id with index 0 reserved
        for PAD.  Item id 0 is never resolvable: it is padding, not a product.
    """

    def __init__(self, assignment: Sequence[Sequence[int]]) -> None:
        table: dict[tuple[int, ...], list[int]] = {}
        for item_id, codes in enumerate(assignment):
            if item_id == 0:
                continue
            key = tuple(int(value) for value in codes)
            table.setdefault(key, []).append(item_id)
        # Deterministic ordering once, at construction, so resolution is a lookup and not a sort.
        self._table: dict[tuple[int, ...], tuple[int, ...]] = {
            key: tuple(sorted(members)) for key, members in table.items()
        }

    @property
    def code_count(self) -> int:
        """How many distinct Semantic IDs the catalogue occupies."""
        return len(self._table)

    def is_catalogue_code(self, semantic_id: Sequence[int]) -> bool:
        """True when at least one catalogue item occupies this code."""
        return tuple(int(value) for value in semantic_id) in self._table

    def members(self, semantic_id: Sequence[int]) -> tuple[int, ...]:
        """Every catalogue item sharing this code, ascending; empty when the code is unoccupied."""
        return self._table.get(tuple(int(value) for value in semantic_id), ())

    def resolve(
        self,
        semantic_id: Sequence[int],
        *,
        eligible: Callable[[int], bool] | None = None,
    ) -> Resolution:
        """Resolve one Semantic ID to a catalogue item, failing closed on an unknown code.

        ``eligible`` is an optional predicate applied inside a collision group.  It is a *filter*,
        never a fallback: if it excludes every member, the result is ``item_id=None`` rather than a
        member that failed the filter.
        """
        key = tuple(int(value) for value in semantic_id)
        members = self._table.get(key, ())
        if not members:
            return Resolution(semantic_id=key, item_id=None, collision_size=0, invalid=True)
        if eligible is None:
            return Resolution(
                semantic_id=key, item_id=members[0], collision_size=len(members), invalid=False
            )
        for member in members:
            if eligible(member):
                return Resolution(
                    semantic_id=key,
                    item_id=member,
                    collision_size=len(members),
                    invalid=False,
                )
        # The code exists but nothing in it is usable for this caller.  That is not a licence to
        # return an ineligible item.
        return Resolution(
            semantic_id=key, item_id=None, collision_size=len(members), invalid=False
        )

    def resolve_many(
        self,
        semantic_ids: Iterable[Sequence[int]],
        *,
        eligible: Callable[[int], bool] | None = None,
    ) -> tuple[Resolution, ...]:
        """Resolve a sequence of Semantic IDs, preserving order and duplicates."""
        return tuple(self.resolve(sid, eligible=eligible) for sid in semantic_ids)

    def assignment(self) -> dict[int, tuple[int, ...]]:
        """The item -> Semantic-ID assignment this resolver was built from."""
        return {
            member: key for key, members in self._table.items() for member in members
        }

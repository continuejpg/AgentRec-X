"""Catalogue-constrained decoding trie (Step 2.5, stage 3).

A prefix trie over the **accepted catalogue's** Semantic-ID token paths, used as a *search
constraint only*.

What it constrains, and what it does not
----------------------------------------
The trie answers one question: given the tokens generated so far, which next tokens keep the
path extendable to a real catalogue item?  Because Step 2.4F guarantees the catalogue's final
SIDs are unique (156 746 distinct, post-dedup collisions 0), a complete path through the trie
resolves to **exactly one** backend item.

What the trie is *not*:

* it is not an identity authority.  It maps a token path to a **backend item id** - an opaque
  integer tag.  Canonical identity (`parent_asin`) is AgentRec-X's and never appears here;
* it is not a scoring rule.  The path's score is the model's own log-probability under the
  frozen full-vocabulary rule in :mod:`tiger_public.scoring`; the trie only removes illegal
  continuations, so the ranking semantics are unchanged by it;
* it is not a lookup that can invent an item.  An unknown prefix has no children, so generation
  stops rather than falling back to a nearest match.

Token semantics come from the accepted ``layout.json``, never from a hardcoded assumption: the
per-item path is ``[level_offsets[l] + code_l for l in range(levels)] + [dedup_offset + ordinal]``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from tiger_public.contracts import PAD_SENTINEL

__all__ = [
    "CatalogueTrie",
    "TrieError",
    "TrieStats",
    "item_token_path",
]


class TrieError(ValueError):
    """Raised when the trie is built from an unusable catalogue assignment."""


@dataclass(frozen=True)
class TrieStats:
    """Shape of the built trie, reported so a reader can see its cost and completeness."""

    nodes: int
    leaves: int
    levels: int
    per_item_tokens: int
    catalogue_items: int
    leaf_items: int
    depth_histogram: Mapping[int, int]

    def as_dict(self) -> dict[str, Any]:
        return {
            "nodes": self.nodes,
            "leaves": self.leaves,
            "levels": self.levels,
            "per_item_tokens": self.per_item_tokens,
            "catalogue_items": self.catalogue_items,
            "leaf_items": self.leaf_items,
            "depth_histogram": {str(k): v for k, v in sorted(self.depth_histogram.items())},
        }


def item_token_path(codes: Sequence[int], layout: Mapping[str, Any]) -> tuple[int, ...]:
    """Map one item's Semantic ID to its token path, using the layout's own offsets.

    The dedup level is a **distinct token level** with its own offset block, not an extra code in
    the last semantic level: its range is ``dedup_vocab_size`` wide and independent of
    ``codebook_size``, so conflating the two would alias different items onto one token.

    Raises
    ------
    TrieError
        If the digits are out of range for their level, or the sentinel is present.  A sentinel
        is never tokenisable: mapping it would produce an out-of-range token that aliases a real
        code.
    """
    levels = int(layout["levels"])
    dedup_levels = int(layout["dedup_levels"])
    offsets = list(layout["level_offsets"])
    width = levels + dedup_levels
    if len(codes) != width:
        raise TrieError(f"a Semantic ID needs {width} digits, got {len(codes)}")
    if len(offsets) != width:
        raise TrieError(f"the layout declares {len(offsets)} offsets for {width} levels")
    tokens: list[int] = []
    for index, digit in enumerate(codes):
        value = int(digit)
        if value < 0:
            raise TrieError(
                f"digit {index} is {value}; the PAD sentinel is not a real catalogue code"
            )
        size = int(layout["codebook_size"]) if index < levels else int(layout["dedup_vocab_size"])
        if value >= size:
            raise TrieError(f"digit {index} = {value} is outside [0, {size})")
        tokens.append(int(offsets[index]) + value)
    return tuple(tokens)


class CatalogueTrie:
    """Prefix trie over catalogue SID token paths.

    Parameters
    ----------
    assignment:
        ``assignment[item_id]`` is that item's Semantic ID, indexed by item id with row 0 the
        reserved PAD sentinel row.  Row 0 is skipped by construction, so PAD can never be a
        target.
    layout:
        The accepted ``layout.json`` payload.
    """

    def __init__(self, assignment: Sequence[Sequence[int]], *, layout: Mapping[str, Any]) -> None:
        self.layout = dict(layout)
        self._levels = int(layout["levels"])
        self._dedup_levels = int(layout["dedup_levels"])
        self._per_item = self._levels + self._dedup_levels
        self._vocab_size = int(layout["vocab_size"])
        self._sentinel_tokenisable = bool(layout.get("sentinel_tokenisable", False))
        if self._sentinel_tokenisable:
            raise TrieError(
                "the layout declares the sentinel tokenisable; refusing to build a trie that "
                "could admit PAD"
            )
        if len(assignment) < 2:
            raise TrieError("the assignment holds no real items")

        # children[node][token] -> child node; items[node] -> item ids terminating at that node
        self._children: list[dict[int, int]] = [{}]
        self._items: dict[int, list[int]] = {}
        self._depth: list[int] = [0]
        self._catalogue_items = 0
        for item_id in range(1, len(assignment)):
            codes = assignment[item_id]
            path = item_token_path(codes, self.layout)
            if len(path) != self._per_item:
                raise TrieError(
                    f"item {item_id} produced {len(path)} tokens, expected {self._per_item}"
                )
            node = 0
            for token in path:
                if not 0 <= token < self._vocab_size:
                    raise TrieError(
                        f"item {item_id} produced token {token} outside the vocabulary "
                        f"[0, {self._vocab_size})"
                    )
                child = self._children[node].get(token)
                if child is None:
                    child = len(self._children)
                    self._children.append({})
                    self._depth.append(self._depth[node] + 1)
                    self._children[node][token] = child
                node = child
            self._items.setdefault(node, []).append(item_id)
            self._catalogue_items += 1

        leaves = [node for node, members in self._items.items() if members]
        if len(leaves) != len(set(leaves)):
            raise TrieError("internal error: a leaf node appears twice")
        histogram: dict[int, int] = {}
        for depth in self._depth:
            histogram[depth] = histogram.get(depth, 0) + 1
        self._stats = TrieStats(
            nodes=len(self._children),
            leaves=len(leaves),
            levels=self._levels,
            per_item_tokens=self._per_item,
            catalogue_items=self._catalogue_items,
            leaf_items=sum(len(members) for members in self._items.values()),
            depth_histogram=histogram,
        )

    # -- introspection ----------------------------------------------------- #

    @property
    def stats(self) -> TrieStats:
        return self._stats

    @property
    def levels(self) -> int:
        return self._levels

    @property
    def per_item_tokens(self) -> int:
        return self._per_item

    @property
    def vocab_size(self) -> int:
        return self._vocab_size

    @property
    def items(self) -> int:
        return self._catalogue_items

    def children(self, prefix: Sequence[int]) -> tuple[int, ...]:
        """Legal next tokens after ``prefix`` (ascending); empty when the path cannot extend."""
        node = self.node_for(prefix)
        if node is None:
            return ()
        return tuple(sorted(self._children[node]))

    def is_prefix(self, prefix: Sequence[int]) -> bool:
        """True when some catalogue item's token path starts with ``prefix``.

        The empty prefix is a legal prefix of every catalogue item, which is what lets decoding
        start.
        """
        return self.node_for(prefix) is not None

    def is_complete(self, path: Sequence[int]) -> bool:
        """True when ``path`` is exactly a catalogue item's token path."""
        node = self.node_for(path)
        return node is not None and bool(self._items.get(node))

    def node_for(self, prefix: Sequence[int]) -> int | None:
        node = 0
        for token in prefix:
            child = self._children[node].get(int(token))
            if child is None:
                return None
            node = child
        return node

    def item_for_path(self, path: Sequence[int]) -> int | None:
        """The catalogue item id a complete path resolves to, or ``None`` if it resolves to none.

        **Fails closed.**  Step 2.4F guarantees post-dedup uniqueness, so a complete catalogue
        path has exactly one member; if more than one ever appeared, the smallest id is returned
        deterministically and the ambiguity is visible through :meth:`members_for_path`.
        """
        node = self.node_for(path)
        if node is None:
            return None
        members = self._items.get(node)
        return members[0] if members else None

    def members_for_path(self, path: Sequence[int]) -> tuple[int, ...]:
        """Every item sharing this complete path; empty when the path is not a catalogue path."""
        node = self.node_for(path)
        if node is None:
            return ()
        return tuple(self._items.get(node, ()))

    def collision_free(self) -> bool:
        """True when every catalogue path terminates at exactly one item."""
        return all(len(members) == 1 for members in self._items.values())

    def contains_sentinel(self) -> bool:
        """True when the sentinel leaked into any token.  Always False for a valid build."""
        sentinel_token = PAD_SENTINEL
        return any(sentinel_token in table for table in self._children)

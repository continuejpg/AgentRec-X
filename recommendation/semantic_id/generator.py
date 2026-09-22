"""Autoregressive Semantic-ID generator with constrained decoding.

What it is
----------
A small causal Transformer that consumes a user's history expressed as Semantic IDs and predicts
the **next item's Semantic ID**, one code at a time::

    [SEP, c1 c2 c3, SEP, c1 c2 c3, ...]        history, flattened
        -> causal Transformer
        -> P(next token) at every position
        -> [c1 c2 c3]                          the generated Semantic ID

Each codebook level gets its own level embedding, because code ``5`` at level 1 and code ``5`` at
level 2 mean different things in a residual hierarchy; sharing one table would conflate the coarse
and fine levels.

Constrained decoding, and why it lives here
-------------------------------------------
Generation walks a **prefix trie built from the catalogue's own Semantic IDs**.  A code is only
extendable if some catalogue item occupies that prefix, so the model cannot emit a code path no
product has.  This is the decode-side half of the guarantee whose resolve-side half lives in
:mod:`recommendation.semantic_id.resolver`: decoding keeps generation inside the catalogue's code
set, and resolution turns a code into an item - failing closed when it is not there.

The two mechanisms are independent on purpose.  If the trie were built from the model's own output
distribution, a hallucinated prefix could widen the allowed set; because it is built from the
catalogue assignment, the allowed set is a property of the data.

Scoring the whole catalogue, and the honest cost of it
------------------------------------------------------
The benchmark needs a score for **every** catalogue item, not just the generated one.  The score
used here is::

    score(item) = sum over levels  log P(code_level | history, code_<level)

which is the model's own sequence log-probability.  Computing it **exactly** for all 156 746 items
requires a forward pass per distinct prefix, and the trie has 256 level-1 prefixes, ~5 400
level-2 and ~65 000 level-3 ones on this catalogue - tens of thousands of forward passes per
cohort, which is not affordable on CPU.

So the arm scores items by a **breadth-limited prefix search**: at each level only the
``beams_per_level`` most probable codes are expanded, and an item whose prefix was never expanded
keeps ``-inf``.  This is stated plainly because it is the honest description: the reported metric
is a *retrieval* metric over the expanded prefixes, not an exact likelihood ranking of the whole
catalogue.  The evaluator still owns PAD exclusion, seen-item masking, tie-breaking and ranking,
and the tail convention (unexpanded items below every expanded one) is the same one the Phase-5
retrieval arms already document.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import torch
from torch import nn

__all__ = [
    "GenerationResult",
    "SemanticIdGenerator",
    "SemanticPrefixTrie",
    "TokenLayout",
    "build_token_layout",
]


# --------------------------------------------------------------------------- #
# Trie
# --------------------------------------------------------------------------- #


class SemanticPrefixTrie:
    """A prefix trie over the catalogue's Semantic IDs, built from the catalogue assignment."""

    def __init__(self, assignment: Sequence[Sequence[int]], *, levels: int) -> None:
        self._levels = int(levels)
        self._children: list[dict[int, int]] = [{}]
        self._items: dict[int, list[int]] = {}
        self._parent: dict[int, tuple[int, int]] = {}
        self._prefix: dict[int, tuple[int, ...]] = {0: ()}
        for item_id, codes in enumerate(assignment):
            if item_id == 0:
                continue
            if len(codes) != self._levels:
                raise ValueError(
                    f"item {item_id} has {len(codes)} codes, expected {self._levels}"
                )
            node = 0
            path: list[int] = []
            for code in codes:
                child = self._children[node].get(int(code))
                if child is None:
                    child = len(self._children)
                    self._children.append({})
                    self._children[node][int(code)] = child
                    self._parent[child] = (node, int(code))
                    path.append(int(code))
                    self._prefix[child] = tuple(path)
                else:
                    path.append(int(code))
                node = child
            self._items.setdefault(node, []).append(item_id)

    # -- metadata ---------------------------------------------------------- #

    @property
    def node_count(self) -> int:
        """How many trie nodes exist, including the root."""
        return len(self._children)

    @property
    def level_count(self) -> int:
        """Semantic-ID length."""
        return self._levels

    @property
    def leaf_count(self) -> int:
        """How many nodes terminate at least one item's Semantic ID."""
        return len(self._items)

    def children(self, node: int) -> dict[int, int]:
        """The code -> child-node map at ``node``."""
        return self._children[node]

    def prefix_of(self, node: int) -> tuple[int, ...]:
        """The code path from the root to this node."""
        return self._prefix[node]

    def node_for(self, prefix: Sequence[int]) -> int | None:
        """The node at this code path, or ``None`` when the path is not in the trie."""
        node = 0
        for code in prefix:
            child = self._children[node].get(int(code))
            if child is None:
                return None
            node = child
        return node

    def items_at(self, node: int) -> tuple[int, ...]:
        """Catalogue items whose Semantic ID is exactly this node's path."""
        return tuple(self._items.get(node, ()))

    def nodes_at_depth(self, depth: int) -> tuple[int, ...]:
        """Every node at a given depth, ascending (depth 0 is the root)."""
        if depth == 0:
            return (0,)
        return tuple(
            sorted(node for node, prefix in self._prefix.items() if len(prefix) == depth)
        )

    def depth_histogram(self) -> dict[int, int]:
        """How many nodes exist at each depth; the cost driver for prefix search."""
        counts: dict[int, int] = {}
        for prefix in self._prefix.values():
            counts[len(prefix)] = counts.get(len(prefix), 0) + 1
        return counts


# --------------------------------------------------------------------------- #
# Token layout
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class TokenLayout:
    """How Semantic IDs are laid out as generator tokens.

    ``SEP`` marks "a new item starts here", so one flat vocabulary covers both the sequence
    boundaries and the codes.
    """

    levels: int
    codebook_size: int
    sep_token: int
    vocab_size: int

    @property
    def codes_per_item(self) -> int:
        return self.levels


def build_token_layout(*, levels: int, codebook_size: int) -> TokenLayout:
    """Build the token layout: codes ``0..K-1`` plus one separator token."""
    if levels < 1 or codebook_size < 1:
        raise ValueError("levels and codebook_size must be >= 1")
    return TokenLayout(
        levels=int(levels),
        codebook_size=int(codebook_size),
        sep_token=int(codebook_size),
        vocab_size=int(codebook_size) + 1,
    )


@dataclass(frozen=True)
class GenerationResult:
    """One generated Semantic ID, with the evidence needed to judge it."""

    codes: tuple[int, ...]
    log_probability: float
    #: How many catalogue prefixes were legal at each level during the constrained walk.
    legal_options: tuple[int, ...]

    @property
    def complete(self) -> bool:
        """True when the walk produced a full-length code path."""
        return len(self.codes) > 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "codes": list(self.codes),
            "log_probability": self.log_probability,
            "legal_options": list(self.legal_options),
        }


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #


class SemanticIdGenerator(nn.Module):
    """Causal Transformer over flattened Semantic-ID sequences.

    Parameters
    ----------
    layout:
        Token layout (codes plus the separator).
    max_items:
        Longest history, in items, that the model reads.  The training sequence holds
        ``max_items`` history items plus one target slot, i.e. ``(max_items + 1) * (levels + 1)``
        tokens.  Scoring a conditional prefix needs a few more slots, so the positional table is
        sized with headroom rather than exactly.
    d_model, n_heads, n_layers, dropout:
        Standard Transformer sizes, kept small on purpose: this arm has to train on CPU.
    """

    #: Extra item slots reserved for a condition prefix during scoring/decoding.
    PREFIX_SLOTS = 1

    def __init__(
        self,
        *,
        layout: TokenLayout,
        max_items: int = 10,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if max_items < 1:
            raise ValueError(f"max_items must be >= 1, got {max_items!r}")
        if d_model % n_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by n_heads ({n_heads})")
        self.layout = layout
        self.max_items = int(max_items)

        self.token_embedding = nn.Embedding(layout.vocab_size, d_model)
        self.level_embedding = nn.Embedding(layout.levels + 1, d_model)
        self.position_embedding = nn.Embedding(self.max_sequence_length, d_model)
        self.dropout = nn.Dropout(dropout)

        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.output = nn.Linear(d_model, layout.vocab_size)

    # -- shapes ------------------------------------------------------------- #

    @property
    def per_item(self) -> int:
        """Tokens per item slot: one separator plus the codes."""
        return self.layout.levels + 1

    @property
    def max_sequence_length(self) -> int:
        """Longest sequence the model can embed: the training window plus prefix headroom."""
        return (self.max_items + 1 + self.PREFIX_SLOTS) * self.per_item

    # -- sequence construction --------------------------------------------- #

    def build_sequences(self, histories: Sequence[Sequence[int]]) -> torch.Tensor:
        """Flatten per-user Semantic-ID histories into the fixed-width training window.

        Each row holds the most recent ``max_items`` items - each as ``[SEP, c1..cL]`` - left-padded
        so the target slot is the final item slot.  Left-padding with the separator absent means a
        padded slot is a zero token, which the model never sees at a real position.
        """
        width = (self.max_items + 1) * self.per_item
        rows = torch.zeros((len(histories), width), dtype=torch.long)
        sep = self.layout.sep_token
        levels = self.layout.levels
        for row, history in enumerate(histories):
            codes = list(history)
            if len(codes) % levels != 0:
                raise ValueError(
                    f"history of {len(codes)} codes is not a whole number of Semantic IDs "
                    f"(levels={levels})"
                )
            items = [
                codes[start : start + levels] for start in range(0, len(codes), levels)
            ][-self.max_items :]
            tokens: list[int] = []
            for item in items:
                tokens.append(sep)
                tokens.extend(int(code) for code in item)
            if tokens:
                rows[row, self.per_item - 1 : self.per_item - 1 + len(tokens)] = torch.tensor(
                    tokens, dtype=torch.long
                )
        return rows

    def _embed(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Token + level + position embeddings; the level id follows from the slot layout."""
        positions = torch.arange(token_ids.shape[1], device=token_ids.device).unsqueeze(0)
        level_ids = (positions % self.per_item).expand(token_ids.shape[0], -1)
        return (
            self.token_embedding(token_ids)
            + self.level_embedding(level_ids)
            + self.position_embedding(positions)
        )

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Return next-token logits ``[batch, seq, vocab]`` for every position."""
        if token_ids.dim() != 2:
            raise ValueError(f"token_ids must be 2-D, got {tuple(token_ids.shape)}")
        if token_ids.shape[1] > self.max_sequence_length:
            raise ValueError(
                f"token_ids width must be <= {self.max_sequence_length}, got "
                f"{token_ids.shape[1]}"
            )
        hidden = self.dropout(self._embed(token_ids))
        mask = nn.Transformer.generate_square_subsequent_mask(
            token_ids.shape[1], device=token_ids.device
        )
        encoded = self.encoder(hidden, mask=mask, is_causal=True)
        return self.output(encoded)

    # -- conditional context ------------------------------------------------ #

    def conditional_tokens(
        self, histories: Sequence[Sequence[int]], prefix: Sequence[int]
    ) -> torch.Tensor:
        """Build the window for scoring/generating the code after ``prefix``.

        The window is the training window with its **oldest** ``len(prefix) + 1`` tokens replaced by
        the condition ``[SEP, codes...]``.  That keeps the sequence length constant (so the model
        always sees a well-formed window) while putting the condition at the newest positions,
        which is what makes the walk autoregressive: the next code is predicted from the history
        *and* from what has already been generated.
        """
        history_tokens = self.build_sequences(histories)
        appended = [self.layout.sep_token, *(int(code) for code in prefix)]
        if len(appended) > history_tokens.shape[1]:
            raise ValueError(
                f"the condition prefix needs {len(appended)} tokens but the window holds "
                f"{history_tokens.shape[1]}; raise max_items or PREFIX_SLOTS"
            )
        tail = torch.tensor(appended, dtype=torch.long).unsqueeze(0).expand(
            history_tokens.shape[0], -1
        )
        return torch.cat([history_tokens[:, : history_tokens.shape[1] - len(appended)], tail], dim=1)

    def next_code_logits(
        self, histories: Sequence[Sequence[int]], prefix: Sequence[int]
    ) -> torch.Tensor:
        """Code logits for the next level after ``prefix`` (no separator in the vocabulary)."""
        tokens = self.conditional_tokens(histories, prefix)
        logits = self.forward(tokens)[:, -1, : self.layout.codebook_size]
        return logits

    # -- generation --------------------------------------------------------- #

    @torch.no_grad()
    def generate(
        self,
        histories: Sequence[Sequence[int]],
        *,
        trie: SemanticPrefixTrie,
        beam: int = 1,
    ) -> list[GenerationResult]:
        """Generate one Semantic ID per history, constrained to catalogue prefixes.

        At every level only codes that extend a **catalogue** prefix are considered, so the walk
        cannot leave the catalogue's code set.  If a level has no legal continuation the result is
        an incomplete code rather than an invented one, and the resolver then fails closed.
        """
        self.eval()
        device = next(self.parameters()).device
        results: list[GenerationResult] = []
        for index in range(len(histories)):
            history = [histories[index]]
            frontier: list[tuple[float, tuple[int, ...], tuple[int, ...]]] = [(0.0, (), ())]
            for _level in range(trie.level_count):
                candidates: list[tuple[float, tuple[int, ...], tuple[int, ...]]] = []
                for score, prefix, options in frontier:
                    node = trie.node_for(prefix)
                    legal = sorted(trie.children(node or 0).keys()) if node is not None else []
                    if not legal:
                        continue
                    logits = self.next_code_logits(history, prefix).to(device)
                    log_probs = torch.log_softmax(logits, dim=-1)[0]
                    for code in legal:
                        candidates.append(
                            (score + float(log_probs[code]), prefix + (code,), options + (len(legal),))
                        )
                candidates.sort(key=lambda item: (-item[0], item[1]))
                frontier = candidates[: max(1, beam)]
                if not frontier:
                    break
            if not frontier:
                results.append(
                    GenerationResult(codes=(), log_probability=float("-inf"), legal_options=())
                )
                continue
            best_score, best_prefix, best_options = frontier[0]
            results.append(
                GenerationResult(
                    codes=best_prefix, log_probability=best_score, legal_options=best_options
                )
            )
        return results

    # -- prefix-search scoring ---------------------------------------------- #

    @torch.no_grad()
    def full_catalog_scores(
        self,
        histories: Sequence[Sequence[int]],
        *,
        trie: SemanticPrefixTrie,
        assignment: Sequence[Sequence[int]],
        beams_per_level: int = 512,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Return ``([batch, num_items + 1] scores, diagnostics)``.

        The score of item ``i`` is the model's sequence log-probability of that item's Semantic ID,
        computed by a **breadth-limited prefix search**: at each level only the
        ``beams_per_level`` most probable legal codes are expanded.  Items whose prefix was never
        expanded keep ``-inf``, which the caller turns into the documented tail score; the returned
        diagnostics record how many items were reachable, so the search's coverage is visible rather
        than implied.

        Score bookkeeping is per **trie node**, because one forward pass serves a whole parent
        prefix; the node scores are scattered to items only at the end.  Complexity is
        ``levels × beams_per_level`` forward passes for the whole batch, independent of catalogue
        size - that is what makes this affordable, and the bounded breadth is its price.
        """
        self.eval()
        device = next(self.parameters()).device
        num_items = len(assignment) - 1
        # One column per **terminal node** (plus the root at column 0), not per trie node: only
        # terminal nodes can hold an item, so sizing by ``node_count`` would allocate tens of
        # thousands of unreachable columns and make the caller's scatter ambiguous.
        terminal_nodes = sorted(
            node for node in range(trie.node_count) if trie.items_at(node)
        )
        self._terminal_nodes = terminal_nodes
        self._node_column = {node: position + 1 for position, node in enumerate(terminal_nodes)}
        node_scores = torch.full(
            (len(histories), len(terminal_nodes) + 1), float("-inf"), device=device
        )
        node_scores[:, 0] = 0.0
        frontier: list[tuple[int, tuple[int, ...]]] = [(0, ())]
        for _level in range(trie.level_count):
            candidates: list[tuple[float, int, tuple[int, ...]]] = []
            for node, prefix in frontier:
                legal = sorted(trie.children(node).keys())
                if not legal:
                    continue
                log_probs = torch.log_softmax(
                    self.next_code_logits(histories, prefix).to(device), dim=-1
                )
                for code in legal:
                    child = trie.children(node)[code]
                    if child not in self._node_column:
                        # A non-terminal child cannot hold an item, so nothing can be scored
                        # through it - but it may still lead to terminals, so it stays in the
                        # frontier and its own score is simply not stored.
                        candidates.append(
                            (float(log_probs[:, code].mean()), child, prefix + (code,))
                        )
                        continue
                    column = self._node_column[child]
                    parent_column = self._node_column.get(node, 0)
                    node_scores[:, column] = (
                        node_scores[:, parent_column] + log_probs[:, code]
                    )
                    candidates.append(
                        (float(log_probs[:, code].mean()), child, prefix + (code,))
                    )
            candidates.sort(key=lambda item: (-item[0], item[2]))
            frontier = [(node, prefix) for _s, node, prefix in candidates[:beams_per_level]]
            if not frontier:
                break

        totals = torch.full(
            (len(histories), num_items + 1), float("-inf"), device=device, dtype=torch.float32
        )
        totals[:, 0] = 0.0  # PAD is never a candidate; the evaluator masks it positionally.
        scored_items: set[int] = set()
        for node, _prefix in frontier:
            members = trie.items_at(node)
            column = self._node_column.get(node)
            if not members or column is None:
                continue
            scored_items.update(members)
            # A collision group shares one score, which is the model's honest statement: it
            # cannot distinguish the items that share a code.
            totals[:, list(members)] = node_scores[:, column].unsqueeze(1)
        diagnostics = {
            "levels": trie.level_count,
            "beams_per_level": beams_per_level,
            "nodes_reached": len(frontier),
            "trie_nodes": trie.node_count,
            "scored_items": len(scored_items),
            "scored_share": round(len(scored_items) / num_items, 6) if num_items else 0.0,
            "trie_depth_histogram": trie.depth_histogram(),
        }
        return totals, diagnostics

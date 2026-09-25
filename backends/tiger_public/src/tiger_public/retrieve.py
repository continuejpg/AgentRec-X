"""Step 2.6 - TIGER catalogue-constrained retrieval: APPROXIMATE and CERTIFIED.

This module is the Step-2.6 deliverable named in ``docs/TIGER_BACKEND.md`` section 11.  It
implements two *explicitly different* search modes over the accepted catalogue trie, both
ranking under the **identical frozen item-score rule** of :mod:`tiger_public.scoring`:

APPROXIMATE
    Level-synchronous bounded beam.  Fast and bounded, and only ever a diagnostic: its results
    carry ``status == "APPROXIMATE"`` and must never be reported as a canonical TIGER metric.

CERTIFIED
    Exact branch-and-bound over the catalogue trie with an admissible upper bound, so the
    returned top-``k_prime`` is the true top-``k_prime`` over the **entire valid catalogue**.
    The bound is admissible because every term of the score is a full-vocabulary
    log-probability and therefore ``<= 0``::

        S(q) = S(p) + SUM_{l>m} log P(.)  <=  S(p)      for every extension q of p

    so ``S(p)`` bounds every item in the subtree rooted at ``p``.  A branch is terminated
    exactly when ``S(p) < T``, where ``T`` is the current ``k_prime``-th best *completed* item
    score, and the search terminates globally when the best remaining prefix has ``S(p) < T``.
    No unvisited item can then enter the certified frontier.

Why the search is separated from the model
------------------------------------------
Both search functions take a :class:`NextTokenScorer` - a plain callable returning a
full-vocabulary log-softmax for a batch of prefixes - and never touch the generator.  That is
what makes CERTIFIED's correctness mechanically testable against a brute-force oracle on small
catalogues with no model in the loop, and it keeps "model scoring semantics" (spec 5.1) and
"search algorithm" (spec 5.2) in separate places on purpose.

What this module deliberately does not do
-----------------------------------------
* It never renormalises over valid children.  The trie only removes illegal continuations.
* It never masks seen items, never excludes PAD for the caller, never ranks for presentation
  and never reports a metric.  Those belong to AgentRec-X.
* It never carries ``parent_asin``.  ``contracts.py`` states that absence structurally and
  AGENTS.md section 19.2 rule 1 forbids canonical identity anywhere under ``backends/``.
  :class:`SidResolver` therefore returns **integer item ids** in AgentRec-X's item-id space;
  canonical identity is re-attached by AgentRec-X outside the boundary.
"""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass
from typing import Any, Mapping, Protocol, Sequence

import numpy as np

from tiger_public.scoring import SCORE_RULE
from tiger_public.trie import CatalogueTrie, item_token_path

__all__ = [
    "ADMISSIBLE_BOUND",
    "APPROXIMATE",
    "CERTIFIED",
    "Candidate",
    "CaseRetrieval",
    "CertificationBlock",
    "METHOD_APPROXIMATE",
    "METHOD_CERTIFIED",
    "NextTokenScorer",
    "ResolverRecord",
    "RetrievalConfig",
    "RetrievalError",
    "SidResolver",
    "TigerCaseScorer",
    "aggregate_certificate",
    "build_history_input",
    "assert_next_log_probs_admissible",
    "path_to_semantic_id",
    "retrieve",
    "retrieve_approximate",
    "retrieve_certified",
    "tail_ranking",
]

#: The two retrieval modes.  They are not interchangeable and are never conflated.
APPROXIMATE = "APPROXIMATE"
CERTIFIED = "CERTIFIED"

#: The admissible bound's stated justification, echoed into every certificate (spec 15.2).
ADMISSIBLE_BOUND = "full-vocabulary log-probability prefix sum <= 0 for all continuations"

METHOD_APPROXIMATE = "level_beam"
METHOD_CERTIFIED = "branch_and_bound"


class RetrievalError(ValueError):
    """Raised when retrieval cannot proceed soundly.  Always fails closed."""


# --------------------------------------------------------------------------- #
# Scorer protocol
# --------------------------------------------------------------------------- #


class NextTokenScorer(Protocol):
    """Full-vocabulary log-softmax for the next token after each of ``prefixes``.

    Returns ``[len(prefixes), vocab_size]``.  Implementations must be pure functions of the
    prefix for a fixed history, so the search stays deterministic.
    """

    def __call__(self, prefixes: Sequence[tuple[int, ...]]) -> np.ndarray:  # pragma: no cover
        ...


def assert_next_log_probs_admissible(log_probabilities: np.ndarray) -> None:
    """Assert the precondition that makes the certified bound sound.

    Every entry must be a finite log-probability ``<= 0``.  A positive value means the caller
    handed us something that is not a full-vocabulary log-softmax (a renormalised or
    temperature-scaled distribution, say), and certifying against it would be unsound - so this
    raises rather than silently proceeding.
    """
    table = np.asarray(log_probabilities, dtype=np.float64)
    if table.ndim != 2:
        raise RetrievalError(
            f"next-token log-probabilities must be [batch, vocab], got {tuple(table.shape)}"
        )
    if not np.all(np.isfinite(table)):
        count = int((~np.isfinite(table)).sum())
        raise RetrievalError(
            f"{count} non-finite next-token log-probabilities; the certified bound requires "
            "finite log-probabilities"
        )
    if np.any(table > 1e-6):
        raise RetrievalError(
            f"next-token log-probabilities must be <= 0, got max {float(table.max())!r}; the "
            "input is not a full-vocabulary log-softmax and the prefix bound would not be "
            "admissible"
        )


# --------------------------------------------------------------------------- #
# Records
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class ResolverRecord:
    """Provenance of one SID -> item-id resolution.  Never carries canonical identity."""

    resolver: str
    layout_levels: int
    per_item_tokens: int
    collision_policy: str
    collisions_observed: int
    path_is_complete: bool

    def as_dict(self) -> dict[str, Any]:
        return {
            "resolver": self.resolver,
            "layout_levels": self.layout_levels,
            "per_item_tokens": self.per_item_tokens,
            "collision_policy": self.collision_policy,
            "collisions_observed": self.collisions_observed,
            "path_is_complete": self.path_is_complete,
        }


@dataclass(frozen=True)
class Candidate:
    """One retrieved catalogue item, in integer item-id space.

    ``item_id`` is AgentRec-X's item id (1..156746).  ``semantic_id`` is the digit form of the
    token path, kept for provenance; it is not an identity.
    """

    rank: int
    item_id: int
    semantic_id: tuple[int, ...]
    token_path: tuple[int, ...]
    score: float
    retrieval_mode: str
    resolver: ResolverRecord

    def as_dict(self) -> dict[str, Any]:
        return {
            "rank": self.rank,
            "item_id": self.item_id,
            "semantic_id": list(self.semantic_id),
            "token_path": list(self.token_path),
            "score": self.score,
            "retrieval_mode": self.retrieval_mode,
            "resolver": self.resolver.as_dict(),
        }


@dataclass(frozen=True)
class CertificationBlock:
    """The machine-readable certificate, shaped like spec section 15.2."""

    status: str
    method: str
    score_rule_version: int
    admissible_bound: str
    k_prime_field: str
    threshold_final: float
    prefixes_expanded: int
    nodes_pruned_by_bound: int
    certified_items_per_case: Mapping[str, float]
    certificate_holds_for_all_cases: bool
    budget_exhausted_cases: int
    requested_frontier: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "method": self.method,
            "score_rule_version": self.score_rule_version,
            "admissible_bound": self.admissible_bound,
            "k_prime_field": self.k_prime_field,
            "threshold_final": self.threshold_final,
            "prefixes_expanded": self.prefixes_expanded,
            "nodes_pruned_by_bound": self.nodes_pruned_by_bound,
            "certified_items_per_case": dict(self.certified_items_per_case),
            "certificate_holds_for_all_cases": self.certificate_holds_for_all_cases,
            "budget_exhausted_cases": self.budget_exhausted_cases,
            "requested_frontier": self.requested_frontier,
        }


@dataclass(frozen=True)
class CaseRetrieval:
    """Retrieval result for one case."""

    case_id: int
    candidates: tuple[Candidate, ...]
    status: str
    prefixes_expanded: int
    nodes_pruned_by_bound: int
    budget_exhausted: bool
    threshold_final: float
    requested_frontier: int
    exhaustion_reason: str | None = None

    @property
    def item_ids(self) -> tuple[int, ...]:
        return tuple(c.item_id for c in self.candidates)

    def as_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "status": self.status,
            "prefixes_expanded": self.prefixes_expanded,
            "nodes_pruned_by_bound": self.nodes_pruned_by_bound,
            "budget_exhausted": self.budget_exhausted,
            "threshold_final": self.threshold_final,
            "requested_frontier": self.requested_frontier,
            "exhaustion_reason": self.exhaustion_reason,
            "candidates": [c.as_dict() for c in self.candidates],
        }


@dataclass(frozen=True)
class RetrievalConfig:
    """Bounded-search budget.  ``prefix_budget`` is the certification budget of spec 15.2."""

    mode: str = CERTIFIED
    k: int = 20
    beam: int = 512
    prefix_budget: int = 200_000
    batch_prefixes: int = 256

    def __post_init__(self) -> None:
        if self.mode not in (APPROXIMATE, CERTIFIED):
            raise RetrievalError(f"mode must be {APPROXIMATE} or {CERTIFIED}, got {self.mode!r}")
        if self.k < 1:
            raise RetrievalError(f"k must be >= 1, got {self.k}")
        if self.beam < 1:
            raise RetrievalError(f"beam must be >= 1, got {self.beam}")
        if self.prefix_budget < 1:
            raise RetrievalError(f"prefix_budget must be >= 1, got {self.prefix_budget}")
        if self.batch_prefixes < 1:
            raise RetrievalError(f"batch_prefixes must be >= 1, got {self.batch_prefixes}")


# --------------------------------------------------------------------------- #
# Resolver
# --------------------------------------------------------------------------- #


def path_to_semantic_id(token_path: Sequence[int], layout: Mapping[str, Any]) -> tuple[int, ...]:
    """Invert ``token = level_offsets[l] + code_l`` to recover the digit form.

    Purely presentational provenance: the trie is the authority on what is a catalogue path,
    and this never invents a digit - an out-of-range token is an error, not a clamp.
    """
    offsets = list(layout["level_offsets"])
    width = int(layout["levels"]) + int(layout["dedup_levels"])
    if len(offsets) != width:
        raise RetrievalError(f"the layout declares {len(offsets)} offsets for {width} levels")
    if len(token_path) != width:
        raise RetrievalError(f"a complete Semantic ID needs {width} digits, got {len(token_path)}")
    digits: list[int] = []
    for level, token in enumerate(token_path):
        digit = int(token) - int(offsets[level])
        if digit < 0:
            raise RetrievalError(
                f"token {token} at level {level} is below its level offset {offsets[level]}"
            )
        digits.append(digit)
    return tuple(digits)


class SidResolver:
    """Deterministic, fail-closed token-path -> integer item-id resolver.

    The trie is the sole authority.  A path that is not a complete catalogue path resolves to
    nothing; a path with more than one member resolves to the **smallest** id and counts a
    collision.  The frozen Step-2.4F assignment has zero post-dedup collisions, so a nonzero
    count here is a hard failure upstream rather than something this class papers over.
    """

    name = "tiger_public.retrieve.SidResolver"
    collision_policy = "smallest_item_id_wins"

    def __init__(self, trie: CatalogueTrie) -> None:
        self._trie = trie
        self._layout = dict(trie.layout)
        self._per_item = int(self._layout["levels"]) + int(self._layout["dedup_levels"])
        self.collisions_observed = 0

    @property
    def layout(self) -> Mapping[str, Any]:
        return self._layout

    @property
    def trie(self) -> CatalogueTrie:
        return self._trie

    def is_complete(self, token_path: Sequence[int]) -> bool:
        return bool(self._trie.is_complete(token_path))

    def is_prefix(self, token_path: Sequence[int]) -> bool:
        return bool(self._trie.is_prefix(token_path))

    def resolve(self, token_path: Sequence[int]) -> tuple[int, ResolverRecord] | None:
        """Resolve one complete path.  Returns ``None`` when it is not a catalogue item."""
        path = tuple(int(t) for t in token_path)
        members = self._trie.members_for_path(path)
        if not members:
            return None
        if len(members) > 1:
            self.collisions_observed += 1
        record = ResolverRecord(
            resolver=self.name,
            layout_levels=int(self._layout["levels"]),
            per_item_tokens=self._per_item,
            collision_policy=self.collision_policy,
            collisions_observed=len(members) - 1,
            path_is_complete=True,
        )
        return int(members[0]), record


# --------------------------------------------------------------------------- #
# Search primitives
# --------------------------------------------------------------------------- #


def _order_key(score: float, path: tuple[int, ...]) -> tuple[float, tuple[int, ...]]:
    """Total, deterministic ordering: descending score, then ascending token path."""
    return (-score, path)


def _take_available(available: int, requested: int) -> int:
    """``min(requested, available)``; asking for more candidates than exist is not an error."""
    return max(0, min(int(requested), int(available)))


def _validate_prefix_batch(log_probabilities: np.ndarray, expected: int, vocab: int) -> None:
    if log_probabilities.shape != (expected, vocab):
        raise RetrievalError(
            f"the scorer returned {tuple(log_probabilities.shape)} for {expected} prefixes; "
            f"expected ({expected}, {vocab})"
        )


def _materialise(
    scored: Sequence[tuple[float, tuple[int, ...]]], mode: str, resolver: SidResolver
) -> tuple[Candidate, ...]:
    """Turn ``(score, path)`` pairs into deduplicated, ranked :class:`Candidate` records.

    Deduplication is by resolved ``item_id``: the highest score wins and the earliest rank is
    kept.  The input is already in deterministic order, so the output is a pure function of it.
    """
    out: list[Candidate] = []
    seen: set[int] = set()
    for score, path in scored:
        resolved = resolver.resolve(path)
        if resolved is None:
            continue
        item_id, record = resolved
        if item_id in seen:
            continue
        seen.add(item_id)
        out.append(
            Candidate(
                rank=len(out) + 1,
                item_id=item_id,
                semantic_id=path_to_semantic_id(path, resolver.layout),
                token_path=tuple(path),
                score=float(score),
                retrieval_mode=mode,
                resolver=record,
            )
        )
    return tuple(out)


def retrieve_approximate(
    trie: CatalogueTrie,
    scorer: NextTokenScorer,
    *,
    k: int,
    beam: int = 512,
    prefix_budget: int = 200_000,
    batch_prefixes: int = 256,
    case_id: int = -1,
    resolver: SidResolver | None = None,
) -> CaseRetrieval:
    """Bounded level-synchronous beam.  Always labelled ``APPROXIMATE``."""
    if k < 1:
        raise RetrievalError(f"k must be >= 1, got {k}")
    if beam < 1:
        raise RetrievalError(f"beam must be >= 1, got {beam}")
    resolver = resolver or SidResolver(trie)
    vocab = trie.vocab_size
    frontier: list[tuple[float, tuple[int, ...]]] = [(0.0, ())]
    expanded = 0
    budget_exhausted = False
    exhaustion_reason: str | None = None

    for _level in range(trie.per_item_tokens):
        if not frontier:
            break
        if expanded + len(frontier) > prefix_budget:
            budget_exhausted = True
            exhaustion_reason = "prefix_budget"
            break
        paths = [path for _, path in frontier]
        scores = [score for score, _ in frontier]
        log_probs = np.asarray(scorer(paths), dtype=np.float64)
        _validate_prefix_batch(log_probs, len(paths), vocab)
        assert_next_log_probs_admissible(log_probs)
        expanded += len(paths)
        children: list[tuple[float, tuple[int, ...]]] = []
        for index, path in enumerate(paths):
            base = scores[index]
            for token in trie.children(path):
                value = base + float(log_probs[index, token])
                if not math.isfinite(value):
                    raise RetrievalError(f"non-finite score {value!r} for prefix {path + (token,)}")
                children.append((value, path + (token,)))
        if not children:
            frontier = []
            break
        children.sort(key=lambda item: _order_key(item[0], item[1]))
        frontier = children[:beam]

    completed = [(score, path) for score, path in frontier if trie.is_complete(path)]
    completed.sort(key=lambda item: _order_key(item[0], item[1]))
    candidates = _materialise(completed[:_take_available(len(completed), k)], APPROXIMATE, resolver)
    return CaseRetrieval(
        case_id=case_id,
        candidates=candidates,
        status=APPROXIMATE,
        prefixes_expanded=expanded,
        nodes_pruned_by_bound=0,
        budget_exhausted=budget_exhausted,
        threshold_final=math.nan,
        requested_frontier=k,
        exhaustion_reason=exhaustion_reason,
    )


def retrieve_certified(
    trie: CatalogueTrie,
    scorer: NextTokenScorer,
    *,
    k_prime: int,
    prefix_budget: int = 200_000,
    batch_prefixes: int = 256,
    case_id: int = -1,
    resolver: SidResolver | None = None,
) -> CaseRetrieval:
    """Exact branch-and-bound.  The returned top-``k_prime`` is provably the true top-``k_prime``.

    ``k_prime`` is AgentRec-X's ``required_frontier`` (``K_max + |seen|``), not a display
    cut-off: returning the true top-``required_frontier`` is what lets the shared evaluator mask
    seen items and still see the true top-K.

    ``prefixes_expanded`` counts nodes that were **popped and had their children scored**.  A
    complete path is *recorded* as a candidate and is only expanded further if it also has
    children (possible in a variable-depth layout, never in the accepted fixed-depth one).  With
    a frontier of at least the catalogue size the search is exhaustive, so
    ``prefixes_expanded == nodes - leaves`` and every catalogue item is returned.
    """
    if k_prime < 1:
        raise RetrievalError(f"k_prime must be >= 1, got {k_prime}")
    resolver = resolver or SidResolver(trie)
    vocab = trie.vocab_size
    heap: list[tuple[float, tuple[int, ...]]] = [(-0.0, ())]
    completed: list[tuple[float, tuple[int, ...]]] = []
    threshold = math.nan
    expanded = 0
    pruned = 0
    budget_exhausted = False
    exhaustion_reason: str | None = None

    while heap:
        if not math.isnan(threshold) and -heap[0][0] < threshold:
            break

        batch: list[tuple[float, tuple[int, ...]]] = []
        while heap and len(batch) < batch_prefixes:
            if not math.isnan(threshold) and -heap[0][0] < threshold:
                break
            batch.append(heapq.heappop(heap))
        if not batch:
            break
        if expanded + len(batch) > prefix_budget:
            budget_exhausted = True
            exhaustion_reason = "prefix_budget"
            break

        paths = [path for _, path in batch]
        scores = [-neg for neg, _ in batch]
        log_probs = np.asarray(scorer(paths), dtype=np.float64)
        _validate_prefix_batch(log_probs, len(paths), vocab)
        assert_next_log_probs_admissible(log_probs)

        for index, path in enumerate(paths):
            expanded += 1
            base = scores[index]
            for token in trie.children(path):
                value = base + float(log_probs[index, token])
                if not math.isfinite(value):
                    raise RetrievalError(f"non-finite score {value!r} for prefix {path + (token,)}")
                child = path + (token,)
                if trie.is_complete(child):
                    completed.append((value, child))
                    if len(completed) >= k_prime:
                        ordered = sorted((s for s, _ in completed), reverse=True)
                        threshold = ordered[k_prime - 1]
                    # A complete path may still be a prefix of deeper items in a layout with
                    # variable depth; expanding it too keeps the search exhaustive rather than
                    # silently stopping at the first terminal node.
                    if not trie.children(child):
                        continue
                    if not math.isnan(threshold) and value < threshold:
                        pruned += 1
                        continue
                    heapq.heappush(heap, (-value, child))
                elif not math.isnan(threshold) and value < threshold:
                    pruned += 1
                else:
                    heapq.heappush(heap, (-value, child))

    completed.sort(key=lambda item: _order_key(item[0], item[1]))
    candidates = _materialise(completed[:_take_available(len(completed), k_prime)],
                              CERTIFIED, resolver)
    return CaseRetrieval(
        case_id=case_id,
        candidates=candidates,
        status=CERTIFIED,
        prefixes_expanded=expanded,
        nodes_pruned_by_bound=pruned,
        budget_exhausted=budget_exhausted,
        threshold_final=threshold,
        requested_frontier=k_prime,
        exhaustion_reason=exhaustion_reason,
    )


def retrieve(
    trie: CatalogueTrie,
    scorer: NextTokenScorer,
    *,
    config: RetrievalConfig,
    required_frontier: int,
    case_id: int = -1,
    resolver: SidResolver | None = None,
) -> CaseRetrieval:
    """Dispatch on ``config.mode``.

    CERTIFIED searches for ``required_frontier`` items; APPROXIMATE searches for ``config.k``.
    The status of the result is the mode that actually ran and is never upgraded.
    """
    if config.mode == CERTIFIED:
        return retrieve_certified(
            trie, scorer, k_prime=required_frontier, prefix_budget=config.prefix_budget,
            batch_prefixes=config.batch_prefixes, case_id=case_id, resolver=resolver,
        )
    return retrieve_approximate(
        trie, scorer, k=config.k, beam=config.beam, prefix_budget=config.prefix_budget,
        batch_prefixes=config.batch_prefixes, case_id=case_id, resolver=resolver,
    )


def tail_ranking(retrieved: Sequence[Candidate], num_items: int) -> tuple[int, ...]:
    """A full-catalogue ranking: the retrieved candidates in order, then ascending item id.

    The documented tail exists because AgentRec-X's evaluator consumes a complete ranking while
    the certified search deliberately scores only the frontier.  When the certified frontier is
    the true top-``required_frontier``, the tail can never enter the evaluator's top-K after
    seen-item masking - which is exactly what ``required_frontier = K_max + |seen|`` buys.
    """
    head = [c.item_id for c in retrieved]
    seen = set(head)
    tail = [item for item in range(1, int(num_items) + 1) if item not in seen]
    return tuple(head + tail)


# --------------------------------------------------------------------------- #
# Model binding (the only part that needs torch; imported lazily)
# --------------------------------------------------------------------------- #


def _torch():  # pragma: no cover - exercised only where torch is installed
    try:
        import torch
    except Exception as error:  # pragma: no cover
        raise RetrievalError(
            f"the model-backed scorer needs torch, which is not importable: {error}"
        ) from error
    return torch


def build_history_input(
    history: Sequence[int],
    *,
    assignment: Sequence[Sequence[int]],
    layout: Mapping[str, Any],
    max_hist_items: int = 20,
) -> tuple[int, ...]:
    """The encoder token sequence for one case's train history.

    Byte-for-byte the construction in :func:`tiger_public.tiger.build_examples`::

        [BOS] + ( block(item) + SEP ) for item in history[-max_hist_items:]

    Parity here is a correctness requirement, not a convenience: a differently built encoder
    input would condition the model differently and would not be the accepted model's evaluation.
    """
    if max_hist_items < 1:
        raise RetrievalError(f"max_hist_items must be >= 1, got {max_hist_items}")
    special = layout.get("special") or {}
    for name in ("bos", "sep"):
        if name not in special:
            raise RetrievalError(
                f"the generator layout declares no {name!r} token; refusing to guess an id that "
                "could alias a code"
            )
    bos = int(special["bos"])
    sep = int(special["sep"])
    window = [int(item) for item in history][-max_hist_items:]
    flat: list[int] = [bos]
    for item_id in window:
        if not 0 < item_id < len(assignment):
            raise RetrievalError(
                f"history item {item_id} is outside the assignment range 1..{len(assignment) - 1}"
            )
        flat.extend(item_token_path(assignment[item_id], layout))
        flat.append(sep)
    return tuple(flat)


class TigerCaseScorer:
    """Binds the accepted generator to one case's history and scores prefix batches.

    The encoder runs **once** per case; each :meth:`__call__` runs only the decoder over the
    supplied prefixes, reading the logit at position ``len(prefix)`` (the decoder input is
    ``[BOS] + prefix``).  ``model.eval()`` is set, so scoring is dropout-free and deterministic.

    Values are full-vocabulary log-softmaxes over the generator's whole vocabulary, exactly as
    :mod:`tiger_public.scoring` requires.  Nothing is renormalised over valid children.
    """

    def __init__(
        self,
        generator: Any,
        *,
        encoder_tokens: Sequence[int],
        batch_prefixes: int = 256,
        device: Any | None = None,
    ) -> None:
        torch = _torch()
        self.batch_prefixes = int(batch_prefixes)
        if self.batch_prefixes < 1:
            raise RetrievalError(f"batch_prefixes must be >= 1, got {batch_prefixes}")
        self.generator = generator
        self.vocab_size = int(generator.vocab_size)
        special = generator.layout["special"]
        self._bos = int(special["bos"])
        self._pad = int(special["pad"])
        generator.model.eval()
        self._device = device if device is not None else next(generator.parameters()).device
        ids = torch.tensor([list(encoder_tokens)], dtype=torch.long, device=self._device)
        mask = torch.ones_like(ids)
        self.encoder_tokens = tuple(int(t) for t in encoder_tokens)
        with torch.no_grad():
            self._encoder = generator.model.encoder(input_ids=ids, attention_mask=mask)
        self._mask = mask
        self.calls = 0
        self.prefixes_scored = 0

    def __call__(self, prefixes: Sequence[tuple[int, ...]]) -> np.ndarray:
        torch = _torch()
        paths = [tuple(int(t) for t in p) for p in prefixes]
        if not paths:
            return np.zeros((0, self.vocab_size), dtype=np.float64)
        for path in paths:
            if any(not 0 <= token < self.vocab_size for token in path):
                raise RetrievalError(f"prefix {path} holds a token outside the vocabulary")
        total = len(paths)
        out = np.empty((total, self.vocab_size), dtype=np.float64)
        base_state = self._encoder.last_hidden_state
        for start in range(0, total, self.batch_prefixes):
            chunk = paths[start : start + self.batch_prefixes]
            width = max(len(p) for p in chunk) + 1
            decoder_ids = torch.full(
                (len(chunk), width), self._pad, dtype=torch.long, device=self._device
            )
            decoder_mask = torch.zeros_like(decoder_ids)
            for row, path in enumerate(chunk):
                decoder_ids[row, 0] = self._bos
                if path:
                    decoder_ids[row, 1 : 1 + len(path)] = torch.tensor(
                        list(path), dtype=torch.long, device=self._device
                    )
                decoder_mask[row, : 1 + len(path)] = 1
            # The encoder ran once for this case (batch 1).  Every prefix in the chunk is
            # conditioned on that SAME history, so the single encoder row is repeated - it must
            # never be indexed by the prefix offset, which would silently swap histories once a
            # chunk starts beyond row 0.
            if base_state.shape[0] != 1:
                raise RetrievalError(
                    f"the case scorer holds {base_state.shape[0]} encoder rows; expected exactly 1"
                )
            encoder_outputs = type(self._encoder)(
                last_hidden_state=base_state.expand(len(chunk), -1, -1).contiguous()
            )
            with torch.no_grad():
                logits = self.generator.model(
                    encoder_outputs=encoder_outputs,
                    attention_mask=self._mask.expand(len(chunk), -1).contiguous(),
                    decoder_input_ids=decoder_ids,
                    decoder_attention_mask=decoder_mask,
                ).logits
            for row, path in enumerate(chunk):
                step = len(path)
                out[start + row] = torch.log_softmax(
                    logits[row, step, :].float(), dim=-1
                ).cpu().numpy()
            self.calls += 1
            self.prefixes_scored += len(chunk)
        return out


def aggregate_certificate(
    results: Sequence[CaseRetrieval], *, requested_frontier: int, method: str, status: str
) -> CertificationBlock:
    """Aggregate per-case diagnostics into the run-level certificate block."""
    if not results:
        raise RetrievalError("cannot certify a run with no cases")
    counts = [len(r.candidates) for r in results]
    thresholds = [r.threshold_final for r in results if not math.isnan(r.threshold_final)]
    budget_exhausted = sum(1 for r in results if r.budget_exhausted)
    return CertificationBlock(
        status=status,
        method=method,
        score_rule_version=int(SCORE_RULE["version"]),
        admissible_bound=ADMISSIBLE_BOUND,
        k_prime_field="required_frontier",
        threshold_final=float(max(thresholds)) if thresholds else math.nan,
        prefixes_expanded=int(sum(r.prefixes_expanded for r in results)),
        nodes_pruned_by_bound=int(sum(r.nodes_pruned_by_bound for r in results)),
        certified_items_per_case={
            "min": int(min(counts)),
            "max": int(max(counts)),
            "mean": round(sum(counts) / len(counts), 6),
        },
        certificate_holds_for_all_cases=bool(budget_exhausted == 0),
        budget_exhausted_cases=int(budget_exhausted),
        requested_frontier=int(requested_frontier),
    )

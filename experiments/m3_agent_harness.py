"""Real-catalogue M3 harness factory.

Composes the existing bounded control plane over the **canonical 156,746-item catalogue**,
sharing every expensive component across users so a 20,000-user evaluation does not rebuild the
catalogue index 20,000 times.

What is shared (built once, reused for every user)
--------------------------------------------------
``MetadataIndex`` over ``data/processed/Sports_and_Outdoors_products.jsonl``; the derived
``ProductEnricher``; the ``GroundingVerifier`` (same metadata, same identity map); the
``CatalogSearchSource``; and the identity maps.

What is per user (cheap)
------------------------
a ``LoopController``, a fresh ``CandidateLedger``, a ``CandidatePlane`` bound to that ledger, and
a history engine backed by **this user's frozen sequential head** — the accepted SASrec source as
materialised in M3 Blocker-2. The heavy catalogue objects are passed in, not rebuilt.

Nothing here modifies policy, the ``CandidateSource`` enum, the ``fixed_fusion`` comparator or the
evaluator. ``SELECT_SOURCE`` stays unreachable: no similar-item tool is registered, and the frozen
offering rule in ``loop.py::_plane_actions`` therefore never offers it.
"""

from __future__ import annotations

import hashlib
import json
import pathlib
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

from recommendation.catalog.metadata import MetadataIndex
from recommendation.control import (
    CandidatePlane,
    CatalogSearchSource,
    GroundingVerifier,
    LoopController,
    RecommendFromHistoryCapability,
)
from recommendation.control.candidate_ledger import CandidateLedger
from recommendation.control.schemas import LoopLimits
from recommendation.preference_matching import PreferenceCandidateMatcher
from recommendation.rag import ProductEnricher
from recommendation.reranking import PreferenceReranker
from recommendation.tools import RecommendationTool
from recommendation.tools.schemas import RecommendationToolResult, ToolRecommendation

REPO = pathlib.Path(__file__).resolve().parents[1]

__all__ = [
    "FROZEN_HEAD_SHA256",
    "FrozenHeadEngine",
    "IdentityMap",
    "SharedCatalogue",
    "RealCatalogueHarnessFactory",
    "UserHarness",
]

#: The frozen M3 Blocker-2 artifact holding the accepted source heads, verified before use.
FROZEN_HEAD_SHA256 = "c4c0fdd6bc1e70fd4d1ffc0387c3afbe6e0642d2d0c6c0184ee440bdef5165d4"


def _sha256(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


class IdentityMap:
    """The trusted ``parent_asin`` <-> ``item_id`` map the catalogue is keyed by."""

    def __init__(self, item2id: Mapping[str, int], id2item: Sequence[str | None]) -> None:
        self._item2id = dict(item2id)
        self._id2item = tuple(id2item)

    def has_parent_asin(self, parent_asin: str) -> bool:
        return parent_asin in self._item2id

    def parent_asin_to_item_id(self, parent_asin: str) -> int:
        return self._item2id[parent_asin]

    def item_id_to_parent_asin(self, item_id: int) -> str:
        identity = self._id2item[item_id]
        if identity is None:
            raise KeyError(item_id)
        return identity


class FrozenHeadEngine:
    """The accepted sequential source, expressed as the accepted Tool's engine.

    Returns this user's frozen head. The one subtlety that matters: the Tool result must declare
    the **true** ``history_length``. The repository's test engine hardcodes ``3``, which is exactly
    what makes a non-3-length trusted history fail verification with ``history_length_mismatch``.
    """

    def __init__(self, rows: Sequence[tuple[str, int, float]]) -> None:
        self._rows = tuple(rows)
        self.calls: list[dict[str, Any]] = []

    @property
    def last_history(self) -> list[str] | None:
        return self.calls[-1]["history"] if self.calls else None

    def recommend(self, history_parent_asins: Any, k: int = 10) -> RecommendationToolResult:
        history = list(history_parent_asins)
        self.calls.append({"history": history, "k": k})
        rows = self._rows[:k]
        return RecommendationToolResult(
            recommendations=tuple(
                ToolRecommendation(rank=position, parent_asin=row[0], item_id=row[1], score=row[2])
                for position, row in enumerate(rows, start=1)
            ),
            requested_k=len(rows),
            returned_k=len(rows),
            # the real length, never a constant
            history_length=len(history),
            effective_history_length=len(history),
            history_truncated=False,
            eligible_candidates=len(self._rows),
            timings_ms={"scoring": 0.0, "ranking": 0.0},
        )


@dataclass(frozen=True)
class SharedCatalogue:
    """The once-built catalogue components, reused by every user harness."""

    metadata: Any
    identity: IdentityMap
    enricher: Any
    grounding: Any
    catalog_search: Any
    item2id: Mapping[str, int]
    records: int


@dataclass(frozen=True)
class UserHarness:
    """One user's control plane plus the ledger this factory owns."""

    controller: Any
    ledger: Any
    trusted_history: tuple[str, ...]
    engine: FrozenHeadEngine


class RealCatalogueHarnessFactory:
    """Build a per-user control plane over the shared canonical catalogue.

    Parameters
    ----------
    policy_factory:
        Called once per user and must return a policy. Tests pass a fresh
        ``LLMAgentPolicy(ScriptedModelClient(...))``; the evaluation passes the live policy.
    user_order:
        ``user_int_id`` values in the **same order as the frozen head rows**, which is the frozen
        cohort order. The mapping is explicit because positional alignment with frozen evidence is
        not something this class may guess at.
    """

    def __init__(
        self,
        *,
        policy_factory: Callable[[], Any],
        user_order: Sequence[int],
        products: pathlib.Path | None = None,
        mappings_path: pathlib.Path | None = None,
        heads_path: pathlib.Path | None = None,
        driver: str = "direct",
        limits: LoopLimits | None = None,
        top_n: int = 1000,
        verify_heads_sha256: bool = True,
    ) -> None:
        self._policy_factory = policy_factory
        self._position = {int(uid): i for i, uid in enumerate(user_order)}
        if len(self._position) != len(user_order):
            raise ValueError("user_order contains duplicate user ids")
        self._products = products or REPO / "data/processed/Sports_and_Outdoors_products.jsonl"
        self._mappings_path = (
            mappings_path or REPO / "data/processed/Sports_and_Outdoors_mappings.json"
        )
        self._heads_path = heads_path or REPO / "runs/m3_evidence/phase5_fusion_source_heads.npz"
        self._driver = driver
        self._limits = limits or LoopLimits()
        self._top_n = top_n
        self._verify_heads = verify_heads_sha256
        self._shared: SharedCatalogue | None = None
        self._heads: list[list[int]] | None = None
        self.shared_builds = 0
        self.heads_loads = 0

    # -- once-only construction -------------------------------------------- #

    def shared(self) -> SharedCatalogue:
        """Build the catalogue components on first use and reuse them thereafter."""
        if self._shared is None:
            mappings = json.loads(self._mappings_path.read_text(encoding="utf-8"))
            item2id = mappings["item2id"]
            id2item = mappings["id2item"]
            metadata = MetadataIndex.load(self._products)
            identity = IdentityMap(item2id, id2item)
            self._shared = SharedCatalogue(
                metadata=metadata,
                identity=identity,
                enricher=ProductEnricher(metadata),
                grounding=GroundingVerifier(identity, metadata),
                catalog_search=CatalogSearchSource(metadata),
                item2id=item2id,
                records=len(metadata.records),
            )
            self.shared_builds += 1
        return self._shared

    def _load_heads(self) -> list[list[int]]:
        if self._heads is None:
            if self._verify_heads:
                got = _sha256(self._heads_path)
                if got != FROZEN_HEAD_SHA256:
                    raise ValueError(
                        f"frozen head artifact sha256 {got} != recorded {FROZEN_HEAD_SHA256}"
                    )
            import numpy as np

            z = np.load(self._heads_path)
            matrix, lengths = z["sequential"], z["sequential_lengths"]
            self._heads = [matrix[i, : int(lengths[i])].tolist() for i in range(len(lengths))]
            self.heads_loads += 1
        return self._heads

    def head_rows(self, case: Any) -> list[tuple[str, int, float]]:
        """This case's frozen sequential head as engine rows, best first."""
        position = self._position.get(int(getattr(case, "user_int_id", -1)))
        if position is None:
            raise KeyError(f"user_int_id {getattr(case, 'user_int_id', None)!r} is not in user_order")
        shared = self.shared()
        head = self._load_heads()[position][: self._top_n]
        id2item_seq = shared.identity._id2item  # noqa: SLF001 - identity map owns the sequence
        rows: list[tuple[str, int, float]] = []
        for rank, item_id in enumerate(head):
            identity = id2item_seq[item_id] if 0 < item_id < len(id2item_seq) else None
            if identity is None:
                continue
            rows.append((identity, int(item_id), 1.0 / (rank + 1)))
        return rows

    # -- per-user construction --------------------------------------------- #

    def __call__(self, case: Any) -> UserHarness:
        shared = self.shared()
        engine = FrozenHeadEngine(self.head_rows(case))
        tool = RecommendationTool(engine)
        capability = RecommendFromHistoryCapability(
            tool,
            product_enricher=shared.enricher,
            preference_matcher=PreferenceCandidateMatcher(),
            preference_reranker=PreferenceReranker(),
        )
        controller = LoopController(
            self._policy_factory(),
            capability,
            driver=self._driver,
            limits=self._limits,
        )
        ledger = CandidateLedger()
        controller._candidate_plane = CandidatePlane(  # noqa: SLF001
            ledger=ledger,
            grounding=shared.grounding,
            history_tool=tool,
            catalog_search=shared.catalog_search,
        )
        history: tuple[str, ...] = tuple(
            shared.identity.item_id_to_parent_asin(int(i))
            for i in getattr(case, "test_history", ())
            if 0 < int(i) < len(shared.identity._id2item)  # noqa: SLF001
            and shared.identity._id2item[int(i)] is not None  # noqa: SLF001
        )
        return UserHarness(
            controller=controller, ledger=ledger, trusted_history=history, engine=engine
        )

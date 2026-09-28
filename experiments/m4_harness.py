"""M4 harness: one guarded control plane per user over pre-materialised source heads.

Preregistration references: §2 (source universe), §3 (depth), §6 (budget), §7.2 (adaptive loop).

Memory design
-------------
Once the heads are materialised, the benchmark needs **no catalogue and no live retriever**: only
the identity mapping, the frozen head table and the cohort. Grounding therefore uses the
repository's explicit :class:`~recommendation.control.grounding.MappingOnlyCatalog`, which reports
``checks_catalog`` honestly rather than pretending a catalogue was consulted. On the verified
6.9 GiB budget that is the difference between fitting and being OOM-killed (preregistration §12.3).

What is shared vs per user
--------------------------
**Shared, built once:** the identity map, the grounding verifier, the frozen head table.
**Per user:** a ``LoopController``, a fresh ``CandidateLedger``, a ``CandidatePlane`` bound to that
ledger, and a fresh source tool per source -- the last of which is what makes the §9 one-call rule
enforceable with no cross-user state.

One menu narrowing, no frozen edit
----------------------------------
``available_actions()`` always offers ``RECOMMEND_FROM_HISTORY`` while the tool budget lasts
(``loop.py:491-492``). M4 registers ``HISTORY`` as a **plane source** so that ``SELECT_SOURCE``
covers all four sources uniformly, and it narrows the controller's *menu* to drop the separate
Stage-1 action. That keeps "choose a source" the single control variable instead of giving history
two entry points. The narrowing is applied to the controller **instance**, so it is visible in one
place and ``loop.py`` is untouched.
"""

from __future__ import annotations

import pathlib
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

from recommendation.control import (
    CandidatePlane,
    GroundingVerifier,
    LoopController,
    RecommendFromHistoryCapability,
)
from recommendation.control.arguments import CandidateSource
from recommendation.control.candidate_ledger import CandidateLedger
from recommendation.control.grounding import MappingOnlyCatalog
from recommendation.control.schemas import ActionKind, LoopLimits
from recommendation.tools import RecommendationTool
from recommendation.tools.schemas import (
    RecommendationToolResult,
    ToolRecommendation,
)

from experiments.m4_conformance import FrozenQueryCatalogSearch, TruthfulSourcePlane
from experiments.m4_frozen_sources import (
    AccountedSources,
    DuplicateSourceQuery,
    FrozenHeadTool,
)

__all__ = [
    "M4_DEFAULT_LIMITS",
    "M4HeadTable",
    "M4IdentityMap",
    "M4HarnessFactory",
    "UserHarness",
    "build_identity_map",
]

#: Frozen budget (preregistration §6): four source retrievals, with FINISH free.
M4_DEFAULT_LIMITS = LoopLimits(max_tool_calls=4, max_steps=6)

#: The Stage-1 action M4 removes from the menu, so source selection is the only control variable.
_NARROWED_ACTION = ActionKind.RECOMMEND_FROM_HISTORY


class M4IdentityMap:
    """The trusted ``parent_asin`` <-> ``item_id`` map, loaded from the frozen mappings artifact.

    Implements the repository's ``IdentityMapLike`` protocol so it can be handed to
    :class:`~recommendation.control.grounding.GroundingVerifier` unchanged.
    """

    def __init__(self, item2id: Mapping[str, int], id2item: Sequence[str | None]) -> None:
        self._item2id = dict(item2id)
        self._id2item = tuple(id2item)

    def has_parent_asin(self, parent_asin: str) -> bool:
        """True when the mapping knows this identity."""
        return parent_asin in self._item2id

    def parent_asin_to_item_id(self, parent_asin: str) -> int:
        """Return the integer id, raising for an unknown identity."""
        try:
            return int(self._item2id[parent_asin])
        except KeyError as exc:
            raise KeyError(f"unknown parent_asin {parent_asin!r}") from exc

    def item_id_to_parent_asin(self, item_id: int) -> str:
        """Return the identity for an integer id, raising for an unknown id."""
        if not (0 < int(item_id) < len(self._id2item)):
            raise KeyError(f"item_id {item_id!r} is out of range")
        identity = self._id2item[int(item_id)]
        if identity is None:
            raise KeyError(f"item_id {item_id!r} has no identity")
        return str(identity)

    @property
    def item2id(self) -> dict[str, int]:
        """The identity -> id map (a copy, for read-only use)."""
        return dict(self._item2id)

    def size(self) -> int:
        """How many identities the mapping holds."""
        return len(self._item2id)


def build_identity_map(mappings_path: pathlib.Path | str) -> M4IdentityMap:
    """Load the frozen ``*_mappings.json`` artifact into an :class:`M4IdentityMap`.

    ``id2item`` is positional with ``None`` at index 0 (PAD), which the repository's preprocessing
    guarantees; this loader preserves that layout rather than renumbering it.
    """
    import json

    payload = json.loads(pathlib.Path(mappings_path).read_text(encoding="utf-8"))
    return M4IdentityMap(payload["item2id"], payload["id2item"])


class _FrozenRowsEngine:
    """The minimal engine the accepted :class:`RecommendationTool` expects.

    ``RecommendationTool.run`` owns request validation and result construction
    (``tools/recommendation.py:191``), so M4 supplies only the engine and lets the accepted tool do
    everything else. That keeps the M4 history source on the accepted code path instead of
    reimplementing it, and it reports the **true** history length -- the repository's own test
    engine hardcodes ``3``, which is exactly what makes a non-3-length history fail verification
    with ``history_length_mismatch``.
    """

    def __init__(
        self, rows: Sequence[tuple[str, int, float]], state: list[bool] | None = None
    ) -> None:
        self._rows = tuple(rows)
        self.calls: list[dict[str, Any]] = []
        #: Shared with the HISTORY adapter so the capability path cannot bypass the §9 one-call rule.
        self.state: list[bool] = state if state is not None else [False]

    @property
    def last_history(self) -> list[str] | None:
        """The history supplied to the most recent call, or ``None``."""
        return self.calls[-1]["history"] if self.calls else None

    def recommend(self, history_parent_asins: Any, k: int = 10) -> RecommendationToolResult:
        """Return the frozen head truncated to ``k``, with the true history length."""
        history = list(history_parent_asins)
        self.calls.append({"history": history, "k": int(k)})
        if self.state[0]:
            raise DuplicateSourceQuery(
                "source 'history' has already been queried for this user; "
                "preregistration §9 forbids a second query because it double-counts in RRF"
            )
        self.state[0] = True
        rows = self._rows[: max(0, int(k))]
        return RecommendationToolResult(
            recommendations=tuple(
                ToolRecommendation(
                    rank=position, parent_asin=row[0], item_id=int(row[1]), score=float(row[2])
                )
                for position, row in enumerate(rows, start=1)
            ),
            requested_k=int(k),
            returned_k=len(rows),
            history_length=len(history),
            effective_history_length=len(history),
            history_truncated=False,
            eligible_candidates=len(self._rows),
            timings_ms={"scoring": 0.0, "ranking": 0.0},
        )


class _HistoryToolShim(FrozenHeadTool):
    """The HISTORY adapter, sharing one ``RecommendationTool`` with the Stage-1 capability.

    ``CandidatePlane`` wraps its ``history_tool`` in ``_HistorySourceTool`` and calls the accepted
    ``RecommendationTool`` (``candidate_plane.py:185-190``), while the capability calls the same
    ``RecommendationTool`` directly. Both must therefore be the *same* instance, so that whichever
    path runs consumes the single permitted call and the §9 one-call rule cannot be bypassed.

    The one-shot bookkeeping lives on this shim, whose ``propose`` is never reached for HISTORY --
    the plane goes through ``RecommendationTool`` -- so the shim forwards consumption to the engine
    instead.
    """

    def __init__(self, tool: RecommendationTool, original: FrozenHeadTool | None) -> None:
        super().__init__(
            source=CandidateSource.HISTORY,
            score_kind=(original.score_kind if original is not None else "sasrec_logit"),
            head=(original.head if original is not None else ()),
            state=(original.state if original is not None else [False]),
        )
        self._tool = tool

    def propose(self, *, arguments: Any = None, limit: int = 0, history: Sequence[str] = ()):
        """Serve the frozen head through the accepted tool, enforcing the one-call rule."""
        self._claim()
        if limit <= 0:
            return []
        return [tuple(row) for row in self.head[:limit]]  # type: ignore[misc]

    @property
    def tool(self) -> RecommendationTool:
        """The single shared accepted tool."""
        return self._tool


@dataclass(frozen=True)
class M4HeadTable:
    """The materialised heads, their ``score_kind``, and the query each head came from."""

    heads: Mapping[CandidateSource, Mapping[int, tuple[tuple[str, int, float], ...]]]
    score_kinds: Mapping[CandidateSource, str]
    #: source -> user_int_id -> the exact query/seed recorded during materialisation. Empty for
    #: sources driven by the run's trusted history rather than by a caller-supplied query.
    queries: Mapping[CandidateSource, Mapping[int, str]] = field(default_factory=dict)
    #: How many catalogue records the materialising BM25 index held, reported by the conformance
    #: shim's ``status`` instead of claiming an index it does not carry.
    indexed_records: int = 0
    #: The fields the materialising index searched, for the same reason.
    search_fields: tuple[str, ...] = ()

    @property
    def sources(self) -> tuple[CandidateSource, ...]:
        """Which sources were materialised, in a deterministic order."""
        return tuple(sorted(self.heads, key=lambda s: s.value))

    def users(self, source: CandidateSource) -> int:
        """How many users this source has a head for."""
        return len(self.heads.get(source, {}))


@dataclass
class UserHarness:
    """Everything one user's run needs, already wired to a fresh ledger."""

    controller: LoopController
    ledger: CandidateLedger
    trusted_history: tuple[str, ...]
    tools: dict[CandidateSource, FrozenHeadTool]

    def queried_sources(self) -> tuple[str, ...]:
        """Which sources this user's run actually queried, in a deterministic source order."""
        return tuple(
            tool.source.value
            for _source, tool in sorted(self.tools.items(), key=lambda kv: kv[0].value)
            if tool.consumed
        )


@dataclass
class M4HarnessFactory:
    """Build one :class:`UserHarness` per user over a shared, pre-materialised head table."""

    identity_map: M4IdentityMap
    table: M4HeadTable
    policy_factory: Callable[[], Any]
    limits: LoopLimits = M4_DEFAULT_LIMITS
    _grounding: GroundingVerifier = field(init=False, repr=False)
    _accounted: AccountedSources = field(init=False, repr=False)

    def __post_init__(self) -> None:
        # MappingOnlyCatalog is the explicit "the mapping is the catalogue" adapter, so
        # `checks_catalog` stays honest about what was actually verified.
        self._grounding = GroundingVerifier(
            self.identity_map, MappingOnlyCatalog(self.identity_map)
        )
        self._accounted = AccountedSources(self.table.heads, self.table.score_kinds)

    # -- helpers ----------------------------------------------------------- #

    def trusted_history(self, case: Any) -> tuple[str, ...]:
        """Map ``case.test_history`` to parent ASIN identities, preserving order.

        This is preregistration §5.1's definition of ``h``: the identities of
        ``train_history + (validation_target,)``, with out-of-range and PAD entries dropped.
        """
        out: list[str] = []
        for item_id in getattr(case, "test_history", ()) or ():
            try:
                out.append(self.identity_map.item_id_to_parent_asin(int(item_id)))
            except (KeyError, TypeError, ValueError):
                continue
        return tuple(out)

    # -- per user ---------------------------------------------------------- #

    def __call__(self, case: Any) -> UserHarness:
        """Build a fresh harness for one cohort case."""
        user_int_id = int(getattr(case, "user_int_id", -1))
        tools = self._accounted.for_user(user_int_id)
        ledger = CandidateLedger()

        # ONE history tool instance serves both paths: the plane's `history_tool` and the Stage-1
        # capability. Building a second instance would let the capability produce the frozen head
        # without consuming the one permitted call, bypassing the §9 enforcement.
        history_rows = (
            tools[CandidateSource.HISTORY].head if CandidateSource.HISTORY in tools else ()
        )
        history_adapter = tools.get(CandidateSource.HISTORY)
        history_tool = RecommendationTool(
            _FrozenRowsEngine(
                history_rows,
                state=(history_adapter.state if history_adapter is not None else None),
            )
        )
        tools[CandidateSource.HISTORY] = _HistoryToolShim(history_tool, tools.get(CandidateSource.HISTORY))

        # The plane dispatches every candidate action to the registered tool. Identity-keyed
        # sources (HISTORY, TWO_TOWER) receive trusted history and no arguments; argument-keyed
        # sources (CATALOG_SEARCH, SIMILAR_ITEM) receive arguments and no history.
        # `FrozenHeadTool.propose` accepts both shapes and ignores arguments by design.
        # The `catalog_search` slot is filled by FrozenQueryCatalogSearch, which serves this
        # user's materialised head and whose `propose` is the method the plane dispatches to. The
        # frozen plane wraps the slot in `_CatalogSearchSourceTool`, whose own `propose` returns []
        # when the action carries no `terms` (candidate_plane.py:216-218) -- and
        # `SelectSourceArguments` has no `terms`. Registration here is unconditional, so
        # `plane._tools[CATALOG_SEARCH]` is this object and that early return is never reached.
        # The query is frozen trusted input (preregistration §5); SELECT_SOURCE still chooses only
        # the source. See experiments/m4_conformance for the full rationale.
        catalog_adapter = tools.get(CandidateSource.CATALOG_SEARCH)
        catalog_search = FrozenQueryCatalogSearch(
            head=(catalog_adapter.head if catalog_adapter is not None else ()),
            query=self.table.queries.get(CandidateSource.CATALOG_SEARCH, {}).get(
                user_int_id, ""
            ),
            indexed_records=int(self.table.indexed_records),
            fields=self.table.search_fields,
        )
        plane = CandidatePlane(
            ledger=ledger,
            grounding=self._grounding,
            history_tool=history_tool,
            catalog_search=catalog_search,
            similar_item_tool=tools.get(CandidateSource.SIMILAR_ITEM),
            two_tower_tool=tools.get(CandidateSource.TWO_TOWER),
        )
        # Truthful source labels: the plane assigns a plain `str` while the loop's `_source_for`
        # requires a `CandidateSource` instance (loop.py:2005-2007), so without this every
        # SELECT_SOURCE observation is labelled `catalog_search`. Metadata only -- candidates,
        # scores, ranking, policy decisions and completion are untouched.
        plane = TruthfulSourcePlane(plane)
        capability = RecommendFromHistoryCapability(history_tool)
        controller = LoopController(
            self.policy_factory(),
            capability,
            limits=self.limits,
            candidate_plane=plane,
        )
        _narrow_menu(controller)
        return UserHarness(
            controller=controller,
            ledger=ledger,
            trusted_history=self.trusted_history(case),
            tools=tools,
        )


def _narrow_menu(controller: LoopController) -> None:
    """Drop the Stage-1 action from the menu of every engine this controller builds.

    ``available_actions()`` lives on ``_LoopEngine`` (``loop.py:481``), and the controller builds a
    **fresh engine per run** through its ``new_engine`` factory (``loop.py:1804``, whose only call
    site is ``loop.py:1854``). That factory is therefore the single seam. It is wrapped on the
    **instance**, so the frozen method bodies are untouched, and the narrowing is visible at the one
    place M4 builds a controller.

    Every other action the frozen rule offers is passed through unchanged: this removes exactly one
    action and adds none.
    """
    original_new_engine = controller.new_engine

    def narrowed_new_engine(*args: Any, **kwargs: Any) -> Any:
        engine = original_new_engine(*args, **kwargs)
        original_available = engine.available_actions

        def narrowed_available() -> tuple[ActionKind, ...]:
            return tuple(a for a in original_available() if a is not _NARROWED_ACTION)

        engine.available_actions = narrowed_available  # type: ignore[method-assign]
        return engine

    controller.new_engine = narrowed_new_engine  # type: ignore[method-assign]

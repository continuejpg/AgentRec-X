"""Source-universe conformance: all four frozen M4 sources must be genuinely usable.

M4 requires the adaptive and fixed arms to share one source universe, with every source reachable
through ``SELECT_SOURCE``. Enumerating the frozen ``CandidateSource`` members is not evidence of
that, so these tests **execute** every source through the real ``CandidatePlane.execute`` path and
assert that each returns candidates, through the implementation intended for it, with a truthful
observation label.

The defect these tests were written against
-------------------------------------------
``CandidatePlane`` wraps whatever fills the ``catalog_search`` slot in its own
:class:`~recommendation.control.candidate_plane._CatalogSearchSourceTool`, which assembles a query
from ``arguments.terms`` and returned ``[]`` when there were none -- before calling
``self._search.candidates(...)``. ``SelectSourceArguments`` carries ``source`` and ``limit`` but no
``terms``, so ``SELECT_SOURCE{catalog_search}`` silently yielded zero candidates while the same
source stayed reachable through its dedicated ``SEARCH_CATALOG`` action.

The repair is confined to the query's provenance: a search object that declares a non-empty
``frozen_query`` has its blank query delegated rather than short-circuited, and only M4's slot
object declares it. The invariant is unchanged and is asserted below:

    ``SELECT_SOURCE`` chooses the **source only**; it never generates or alters the query.
"""

from __future__ import annotations

from typing import Any, Sequence

import pytest

from recommendation.control.arguments import CandidateSource, SelectSourceArguments

from experiments.m4_conformance import (
    FrozenQueryCatalogSearch,
    SourceLabelMismatch,
    TruthfulSourcePlane,
)
from experiments.m4_harness import M4HarnessFactory, M4HeadTable, M4IdentityMap
from experiments.m4_policies import FIXED_SOURCE_ORDER, FixedTraversalPolicy

NUM_ITEMS = 40
USERS = 8

#: The frozen four-source universe (preregistration §2).
UNIVERSE = (
    CandidateSource.CATALOG_SEARCH,
    CandidateSource.HISTORY,
    CandidateSource.SIMILAR_ITEM,
    CandidateSource.TWO_TOWER,
)

#: The query every test user's catalogue head was materialised with.
FROZEN_QUERY = "camping lantern waterproof"


def _identities() -> list[str]:
    return [f"B{index:04d}" for index in range(1, NUM_ITEMS + 1)]


def _identity_map() -> M4IdentityMap:
    ids = _identities()
    return M4IdentityMap({value: i + 1 for i, value in enumerate(ids)}, [None, *ids])


def _head(source: CandidateSource, user_int_id: int) -> tuple[tuple[str, int, float], ...]:
    ids = _identities()
    offset = {s: i for i, s in enumerate(UNIVERSE)}[source] * 3
    return tuple(
        (ids[(user_int_id * 2 + offset + rank) % len(ids)], rank + 1, 1.0 / (rank + 1))
        for rank in range(20)
    )


def _table() -> M4HeadTable:
    return M4HeadTable(
        heads={
            source: {uid: _head(source, uid) for uid in range(1, USERS + 1)}
            for source in UNIVERSE
        },
        score_kinds={
            CandidateSource.CATALOG_SEARCH: "bm25",
            CandidateSource.HISTORY: "sasrec_logit",
            CandidateSource.SIMILAR_ITEM: "item_item_similarity",
            CandidateSource.TWO_TOWER: "two_tower_cosine",
        },
        queries={
            CandidateSource.CATALOG_SEARCH: {uid: FROZEN_QUERY for uid in range(1, USERS + 1)},
            CandidateSource.SIMILAR_ITEM: {uid: "B0001" for uid in range(1, USERS + 1)},
        },
        indexed_records=NUM_ITEMS,
        search_fields=("title", "store", "main_category", "categories", "features", "description"),
    )


class _Case:
    """A minimal ``EvaluationCase``-shaped object."""

    def __init__(self, uid: int) -> None:
        self.user_int_id = uid
        self.user_id = f"user{uid:03d}"
        self.test_history = (uid, uid + 1)
        self.test_target = uid + 2


def _harness(uid: int = 1) -> Any:
    factory = M4HarnessFactory(
        identity_map=_identity_map(), table=_table(), policy_factory=FixedTraversalPolicy
    )
    return factory(_Case(uid))


def _plane_tool(harness: Any, source: CandidateSource) -> Any:
    return harness.controller._candidate_plane._tools[source]


def _validated(harness: Any, source: CandidateSource) -> Any:
    """Build a validated ``SELECT_SOURCE`` action for one source."""
    from recommendation.control.schemas import ActionKind, ActionProposal

    proposal = ActionProposal(
        action=ActionKind.SELECT_SOURCE,
        arguments=SelectSourceArguments(source=source, limit=100),
        rationale="conformance test",
    )
    validated, verification = harness.controller._validator.validate(
        proposal,
        available_actions=(ActionKind.SELECT_SOURCE,),
        run_id="conformance",
        turn_id=None,
        step_index=0,
    )
    assert verification.verified, f"proposal was refused: {verification.code}"
    assert validated is not None
    return validated


# --------------------------------------------------------------------------- #
# 1. execution-level source-universe equality
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("source", UNIVERSE, ids=lambda s: s.value)
def test_every_universe_source_returns_candidates_through_select_source(
    source: CandidateSource,
) -> None:
    """The gate: all four sources must yield candidates through the real plane path."""
    harness = _harness(uid=1)
    plane = harness.controller._candidate_plane
    result = plane.execute(
        _validated(harness, source),
        read_trusted_history=lambda: harness.trusted_history,
        step_index=0,
    )
    assert result.returned_k > 0, f"{source.value} returned no candidates"
    assert result.status == "ok"
    assert result.source is source, f"{source.value} was labelled {result.source}"


def test_the_adaptive_universe_equals_the_fixed_universe() -> None:
    """Same registered set, and now the same *reachable* behaviour."""
    harness = _harness(uid=1)
    plane = harness.controller._candidate_plane
    assert set(plane.available_sources()) == set(UNIVERSE)
    assert set(FIXED_SOURCE_ORDER) == set(UNIVERSE)

    reachable = set()
    for source in UNIVERSE:
        fresh = _harness(uid=1)
        result = fresh.controller._candidate_plane.execute(
            _validated(fresh, source),
            read_trusted_history=lambda: fresh.trusted_history,
            step_index=0,
        )
        if result.returned_k > 0:
            reachable.add(source)
    assert reachable == set(UNIVERSE), f"unreachable sources: {set(UNIVERSE) - reachable}"


def test_no_source_contributes_zero_candidates_in_a_full_traversal() -> None:
    """The silent-zero path is what made the defect invisible; assert it is gone."""
    harness = _harness(uid=6)
    result = harness.controller.run("recommend something", harness.trusted_history)
    counts: dict[str, int] = {}
    for step in getattr(result.trajectory, "steps", ()) or ():
        observation = getattr(step, "observation", None)
        if observation and observation.get("kind") == "candidate_set":
            counts[observation["source"]] = int(observation.get("returned_k", 0))
    assert set(counts) == {source.value for source in UNIVERSE}
    for name, returned in counts.items():
        assert returned > 0, f"{name} contributed zero candidates"


# --------------------------------------------------------------------------- #
# 2. the frozen query is trusted input, and identity is preserved
# --------------------------------------------------------------------------- #


def test_select_source_uses_the_materialised_query_not_a_derived_one() -> None:
    """The query served equals the one persisted in the head artifact for that user."""
    harness = _harness(uid=3)
    tool = _plane_tool(harness, CandidateSource.CATALOG_SEARCH)
    expected = _table().heads[CandidateSource.CATALOG_SEARCH][3]
    served = tool.propose(
        arguments=SelectSourceArguments(source=CandidateSource.CATALOG_SEARCH, limit=100),
        limit=100,
    )
    assert tuple(served) == expected
    assert tool._search.frozen_query == FROZEN_QUERY


def test_a_different_query_is_refused_rather_than_served() -> None:
    """A divergent query is an error, never a silently different retrieval."""
    harness = _harness(uid=1)
    shim = _plane_tool(harness, CandidateSource.CATALOG_SEARCH)._search
    with pytest.raises(SourceLabelMismatch):
        shim.candidates("a completely different query", limit=5)


def test_the_shim_does_not_build_a_live_index() -> None:
    """It must serve the materialised head, not re-run retrieval."""
    harness = _harness(uid=1)
    shim = _plane_tool(harness, CandidateSource.CATALOG_SEARCH)._search
    status = shim.status()
    assert status["live_index"] is False
    assert status["head_size"] == 20
    assert status["materialised_query"] == FROZEN_QUERY
    assert status["indexed_records"] == NUM_ITEMS
    assert status["score_kind"] == "bm25"


def test_the_shim_reports_the_source_and_score_kind_the_ledger_expects() -> None:
    harness = _harness(uid=1)
    shim = _plane_tool(harness, CandidateSource.CATALOG_SEARCH)._search
    assert shim.source is CandidateSource.CATALOG_SEARCH
    assert shim.score_kind == "bm25"
    assert shim.name == "catalog_search"


def test_an_action_carrying_terms_must_match_the_materialised_query() -> None:
    """If an action ever does carry terms, they cannot silently select a different retrieval."""
    shim = FrozenQueryCatalogSearch(head=(("A", 1, 9.0),), query=FROZEN_QUERY)
    assert list(shim.candidates(FROZEN_QUERY, limit=5)) == [("A", 1, 9.0)]
    with pytest.raises(SourceLabelMismatch):
        shim.candidates("other terms", limit=5)


# --------------------------------------------------------------------------- #
# 3. the frozen plane's blank-query guard is unchanged for everyone else
# --------------------------------------------------------------------------- #


def test_a_plain_search_object_still_yields_nothing_for_a_blank_query() -> None:
    """The shared adapter's behaviour is unchanged when no frozen query is declared.

    This is what keeps the repair from becoming a repository-wide "blank query means every query"
    relaxation: only an object exposing a non-empty ``frozen_query`` is delegated to.
    """
    from recommendation.control.candidate_plane import _CatalogSearchSourceTool

    class _PlainSearch:
        """Stands in for a real ``CatalogSearchSource``: no ``frozen_query`` attribute."""

        calls: list[str] = []

        def candidates(self, query: str, *, limit: int = 10) -> list[tuple[str, int, float]]:
            type(self).calls.append(query)
            return [("X", 1, 1.0)]

    adapter = _CatalogSearchSourceTool(_PlainSearch())  # type: ignore[arg-type]
    blank = adapter.propose(
        arguments=SelectSourceArguments(source=CandidateSource.CATALOG_SEARCH, limit=5), limit=5
    )
    assert blank == []
    assert _PlainSearch.calls == [], "a plain search object must not be consulted for a blank query"


def test_a_plain_search_object_still_serves_a_real_query() -> None:
    """The non-blank path is untouched."""
    from recommendation.control.arguments import SearchCatalogArguments
    from recommendation.control.candidate_plane import _CatalogSearchSourceTool

    class _PlainSearch:
        calls: list[str] = []

        def candidates(self, query: str, *, limit: int = 10) -> list[tuple[str, int, float]]:
            type(self).calls.append(query)
            return [("X", 1, 1.0)]

    adapter = _CatalogSearchSourceTool(_PlainSearch())  # type: ignore[arg-type]
    rows = adapter.propose(
        arguments=SearchCatalogArguments(terms=("camping", "lantern"), limit=5), limit=5
    )
    assert rows == [("X", 1, 1.0)]
    assert _PlainSearch.calls == ["camping lantern"]


def test_the_m4_slot_is_the_shim() -> None:
    """The plane holds the shim, wrapped in its own adapter, so the blank query is delegated."""
    harness = _harness(uid=1)
    tool = _plane_tool(harness, CandidateSource.CATALOG_SEARCH)
    assert type(tool).__name__ == "_CatalogSearchSourceTool"
    assert isinstance(tool._search, FrozenQueryCatalogSearch)


# --------------------------------------------------------------------------- #
# 4. truthful observation and provenance labels
# --------------------------------------------------------------------------- #


def test_the_plane_declares_the_real_source_for_every_universe_member() -> None:
    """The label the observation carries must name the source that ran."""
    for source in UNIVERSE:
        harness = _harness(uid=1)
        result = harness.controller._candidate_plane.execute(
            _validated(harness, source),
            read_trusted_history=lambda: harness.trusted_history,
            step_index=0,
        )
        declared = getattr(result, "source", None)
        assert isinstance(declared, CandidateSource), (
            f"{source.value} declared {declared!r} ({type(declared).__name__}); the loop's "
            "_source_for would mislabel it"
        )
        assert declared is source


def test_a_full_traversal_labels_all_four_sources_truthfully() -> None:
    """Run the fixed arm and assert the observation named every source correctly."""
    harness = _harness(uid=4)
    result = harness.controller.run("recommend something", harness.trusted_history)
    labelled = [
        step.observation.get("source")
        for step in (getattr(result.trajectory, "steps", ()) or ())
        if getattr(step, "observation", None)
        and step.observation.get("kind") == "candidate_set"
    ]
    assert sorted(labelled) == sorted(source.value for source in UNIVERSE), (
        f"observations named {sorted(labelled)}, expected all four sources"
    )


def test_the_ledger_records_all_four_sources_in_a_full_traversal() -> None:
    """Provenance, not just the label: the ledger must hold all four sources."""
    harness = _harness(uid=5)
    harness.controller.run("recommend something", harness.trusted_history)
    present = {source.value for source in harness.ledger.sources_present()}
    assert present == {source.value for source in UNIVERSE}, (
        f"ledger holds {sorted(present)}, expected all four"
    )


def test_the_truthful_wrapper_restamps_a_string_source() -> None:
    """The type mismatch that mislabelled observations, fixed non-destructively."""

    class _Result:
        def __init__(self, source: Any) -> None:
            self.source = source
            self.other = "untouched"

        def model_copy(self, update: dict[str, Any]) -> "_Result":
            clone = _Result(update.get("source", self.source))
            clone.other = self.other
            return clone

    class _Plane:
        def __init__(self, source: Any) -> None:
            self._source = source

        def execute(self, action: Any, **kwargs: Any) -> _Result:
            return _Result(self._source)

        def has_source(self, source: CandidateSource) -> bool:
            return True

        def available_sources(self) -> tuple[CandidateSource, ...]:
            return UNIVERSE

    result = TruthfulSourcePlane(_Plane("two_tower")).execute(object())
    assert result.source is CandidateSource.TWO_TOWER
    assert result.other == "untouched"

    already = TruthfulSourcePlane(_Plane(CandidateSource.SIMILAR_ITEM))
    assert already.execute(object()).source is CandidateSource.SIMILAR_ITEM


def test_an_unknown_string_source_is_left_alone() -> None:
    """A value that is not a CandidateSource must not be silently rewritten."""

    class _Result:
        source = "not_a_real_source"

        def model_copy(self, update: dict[str, Any]) -> "_Result":  # pragma: no cover
            raise AssertionError("must not be called")

    class _Plane:
        def execute(self, action: Any, **kwargs: Any) -> _Result:
            return _Result()

        def has_source(self, source: CandidateSource) -> bool:
            return True

        def available_sources(self) -> tuple[CandidateSource, ...]:
            return UNIVERSE

    assert TruthfulSourcePlane(_Plane()).execute(object()).source == "not_a_real_source"


def test_the_label_repair_does_not_change_candidates_or_scores() -> None:
    """Metadata only: the same rows, ranks and scores with and without the wrapper."""
    for source in UNIVERSE:
        fresh_a = _harness(uid=7)
        fresh_b = _harness(uid=7)
        raw = fresh_a.controller._candidate_plane._plane.execute(
            _validated(fresh_a, source),
            read_trusted_history=lambda: fresh_a.trusted_history,
            step_index=0,
        )
        stamped = fresh_b.controller._candidate_plane.execute(
            _validated(fresh_b, source),
            read_trusted_history=lambda: fresh_b.trusted_history,
            step_index=0,
        )
        assert list(raw.provenance) == list(stamped.provenance)
        assert raw.returned_k == stamped.returned_k
        assert raw.grounded_parent_asins == stamped.grounded_parent_asins


# --------------------------------------------------------------------------- #
# 5. heads are real, not stubs
# --------------------------------------------------------------------------- #


def test_every_universe_source_has_a_materialised_head() -> None:
    """No source is an empty stub: all four carry real heads."""
    table = _table()
    for source in UNIVERSE:
        assert table.heads[source], f"{source.value} has no head"
        assert all(rows for rows in table.heads[source].values())

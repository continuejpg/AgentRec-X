"""TIGER's wiring into the Agent control plane (sections B.3-B.5, B.10 of T-02).

Three things are pinned here, all of them fail if the wiring regresses:

1. the plan vocabulary accepts ``tiger`` and routes it through ``SELECT_SOURCE``;
2. the controller's menu offers ``SELECT_SOURCE`` for a plane whose only selectable source is
   TIGER - the defect this section fixes, where the action was gated on item-item similar-item
   registration and therefore was never offered to a deployment without the 6.21 GB index;
3. the action a ``tiger`` plan builds actually names the TIGER source.

Execution of the action itself is covered by ``tests/test_tiger_source.py`` (plane-level) and
``tests/test_tiger_source_contract.py`` (source contract); this file is about reachability.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from recommendation.control.arguments import CandidateSource
from recommendation.control.candidate_plane import CandidatePlane
from recommendation.control.loop import _LoopEngine
from recommendation.control.schemas import ActionKind
from recommendation.demo.agent_policy import (
    DEFAULT_SOURCE_PLAN,
    MAX_SOURCE_PLAN,
    SOURCE_ARGUMENTS,
    SOURCE_PLANS,
    DemoSourcePlanPolicy,
    normalise_source_plan,
)


class _Tool:
    """Minimal plane source: enough for registration, never called here."""

    def __init__(self, source: CandidateSource) -> None:
        self._source = source

    @property
    def source(self) -> CandidateSource:
        return self._source

    @property
    def score_kind(self) -> str:
        return "test"

    def propose(self, **_kwargs: Any) -> list[tuple[str, int, float]]:
        return []


def menu_for(**tools: Any) -> tuple[ActionKind, ...]:
    """The controller's candidate actions for a plane built from the given tools."""
    plane = CandidatePlane(ledger=None, grounding=None, **tools)  # type: ignore[arg-type]
    return _LoopEngine._plane_candidate_actions(SimpleNamespace(candidate_plane=plane))  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# 1. plan vocabulary
# --------------------------------------------------------------------------- #


def test_tiger_is_a_known_plan_source_routed_through_select_source() -> None:
    assert SOURCE_PLANS["tiger"] is ActionKind.SELECT_SOURCE
    assert SOURCE_ARGUMENTS["tiger"] == "tiger"
    assert normalise_source_plan(["tiger"]) == ("tiger",)
    assert normalise_source_plan(["two_tower", "tiger"]) == ("two_tower", "tiger")
    assert MAX_SOURCE_PLAN == len(SOURCE_PLANS) == 4
    assert DEFAULT_SOURCE_PLAN == ("history",), "the default path must not change"


def test_unknown_and_duplicate_sources_are_still_rejected() -> None:
    with pytest.raises(ValueError):
        normalise_source_plan(["tiger", "tiger"])
    with pytest.raises(ValueError):
        normalise_source_plan(["tiger", "nope"])


def test_a_tiger_plan_builds_a_select_source_action_naming_tiger() -> None:
    policy = DemoSourcePlanPolicy(("tiger",), default_k=7)
    action = policy._select_arguments("tiger")  # noqa: SLF001 - the seam under test
    assert action.source == "tiger"
    assert action.limit == 7


# --------------------------------------------------------------------------- #
# 2. the menu rule (the defect fixed in T-02)
# --------------------------------------------------------------------------- #


def test_select_source_is_offered_when_tiger_is_the_only_selectable_source() -> None:
    offered = menu_for(tiger_tool=_Tool(CandidateSource.TIGER))
    assert ActionKind.SELECT_SOURCE in offered
    assert ActionKind.FIND_SIMILAR not in offered
    assert ActionKind.SEARCH_CATALOG not in offered


def test_select_source_is_offered_for_two_tower_without_similar_item() -> None:
    offered = menu_for(two_tower_tool=_Tool(CandidateSource.TWO_TOWER))
    assert ActionKind.SELECT_SOURCE in offered
    assert ActionKind.FIND_SIMILAR not in offered


def test_a_catalog_only_plane_still_offers_no_select_source() -> None:
    """The accepted single-source deployment's menu must not grow."""
    offered = _LoopEngine._plane_candidate_actions(  # type: ignore[arg-type]
        SimpleNamespace(candidate_plane=_CatalogOnlyPlane())
    )
    assert ActionKind.SELECT_SOURCE not in offered
    assert ActionKind.SEARCH_CATALOG in offered


class _CatalogOnlyPlane:
    """A plane stub reporting only the catalog source, as the accepted deployment does."""

    def has_source(self, source: CandidateSource) -> bool:
        return source is CandidateSource.CATALOG_SEARCH


def test_similar_item_deployments_keep_the_original_order() -> None:
    """Archived recordings replay on this tuple: FIND_SIMILAR first, then SELECT_SOURCE."""
    offered = menu_for(similar_item_tool=_Tool(CandidateSource.SIMILAR_ITEM))
    assert offered == (ActionKind.FIND_SIMILAR, ActionKind.SELECT_SOURCE)

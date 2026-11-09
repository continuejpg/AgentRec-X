from __future__ import annotations

from types import SimpleNamespace

from recommendation.demo.product_role import (
    RoleStatus,
    TargetRole,
    assess_candidate,
    infer_target,
    select_grounded_recommendations,
)


def _record(title: str, category: str):
    return SimpleNamespace(title=title, categories=("Sports & Outdoors", category))


def _item(asin: str):
    return SimpleNamespace(parent_asin=asin)


def test_complete_filter_does_not_accept_accessory_or_filtered_bottle() -> None:
    records = {
        "main": _record("LoGest Portable Water Filter Pump", "Water Filters"),
        "part": _record("Survivor Filter Replacement Mouthpiece", "Water Filters"),
        "bottle": _record("Clear2GO Water Bottle with Filter", "Sports Water Bottles"),
        "unknown": _record("Perko Gas Tank Vent", "Fuel Tanks"),
    }
    selection = select_grounded_recommendations(
        "I need a complete water filter.",
        [_item("unknown"), _item("part"), _item("bottle"), _item("main")],
        records,
    )
    assert selection.target.type == "water_filter"
    assert selection.target.role is TargetRole.MAIN
    assert [item.parent_asin for item in selection.candidates] == ["main", "unknown"]
    assert {a.parent_asin: a.status for a in selection.assessments} == {
        "unknown": RoleStatus.UNKNOWN,
        "part": RoleStatus.MISMATCH,
        "bottle": RoleStatus.MISMATCH,
        "main": RoleStatus.MATCH,
    }


def test_correct_main_product_below_base_rank_is_moved_first() -> None:
    records = {
        "main": _record("LifeStraw Personal Water Filter", "Water Filters"),
        "bottle": _record("Filtered Water Bottle", "Sports Water Bottles"),
    }
    selection = select_grounded_recommendations(
        "complete water filter",
        [_item("bottle"), _item("main")],
        records,
        target=infer_target("complete water filter"),
    )
    assert [item.parent_asin for item in selection.candidates] == ["main"]


def test_all_mismatch_is_not_filled_with_irrelevant_candidates() -> None:
    records = {
        "food": _record("Freeze Dried Camping Food", "Freeze-Dried Food"),
        "tent": _record("Four Person Dome Tent", "Tents"),
        "cookware": _record("Camping Cookware Set", "Open Fire Cookware"),
    }
    selection = select_grounded_recommendations(
        "I need a complete water filter.", [_item("food"), _item("tent"), _item("cookware")], records
    )
    # MISMATCH is removed; with no MATCH, UNKNOWN evidence is reported but not
    # presented as if it satisfied the explicit target.
    assert selection.candidates == ()
    assert selection.mismatch_count == 1
    assert selection.unknown_count == 2


def test_unknown_evidence_is_retained_after_matches_but_not_called_match() -> None:
    records = {
        "main": _record("Portable Water Filtration System", "Water Filters"),
        "bare": _record("Mystery Outdoor Item", "Miscellaneous"),
    }
    selection = select_grounded_recommendations(
        "water filter", [_item("bare"), _item("main")], records
    )
    assert [item.parent_asin for item in selection.candidates] == ["main", "bare"]
    assert selection.assessments[0].status is RoleStatus.UNKNOWN
    assert selection.assessments[1].status is RoleStatus.MATCH


def test_explicit_accessory_request_accepts_only_compatible_accessory() -> None:
    records = {
        "part": _record("Survivor Filter Replacement Mouthpiece", "Water Filters"),
        "main": _record("LoGest Water Filter Pump", "Water Filters"),
        "bottle_part": _record("Water Bottle Bite Valve Replacement", "Sports Water Bottles"),
    }
    target = infer_target("I need a replacement mouthpiece for a water filter")
    assert target.role is TargetRole.ACCESSORY
    selection = select_grounded_recommendations(
        "replacement mouthpiece", [_item("main"), _item("bottle_part"), _item("part")], records,
        target=target,
    )
    assert [item.parent_asin for item in selection.candidates] == ["part"]
    assert assess_candidate("main", records["main"], target).status is RoleStatus.MISMATCH
    assert assess_candidate("bottle_part", records["bottle_part"], target).status is RoleStatus.MISMATCH


def test_broad_request_has_no_role_filter() -> None:
    records = {"food": _record("Camping Food", "Freeze-Dried Food")}
    selection = select_grounded_recommendations(
        "I need camping gear", [_item("food")], records
    )
    assert selection.target.type is None
    assert [item.parent_asin for item in selection.candidates] == ["food"]

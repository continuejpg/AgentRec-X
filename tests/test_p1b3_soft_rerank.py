from __future__ import annotations

from types import SimpleNamespace

from recommendation.demo.preference_construction import TaskPreference
from recommendation.demo.soft_preference_ranking import (
    merge_soft_preferences,
    rank_with_m5_soft_preferences,
    soft_preferences_from_ranker_projection,
)


def _record(weight_kg: str | None):
    details = {} if weight_kg is None else {"Item Weight": weight_kg}
    return SimpleNamespace(title="camping cookware", details=details)


def _item(identity: str):
    return SimpleNamespace(parent_asin=identity, item_id=hash(identity) % 1000 + 1)


def _feature(**overrides):
    values = {
        "preference_id": "m5-weight-lower",
        "kind": "feature",
        "value": "weight",
        "direction": "minimize",
        "lifecycle": "explicit",
        "is_soft": False,
        "scope": "task",
        "polarity": "prefer",
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _projection(*features):
    return SimpleNamespace(features=tuple(features))


def test_unconfirmed_or_unsupported_preference_never_enters_m61() -> None:
    assert soft_preferences_from_ranker_projection(
        _projection(_feature(is_soft=True))
    ) == ()
    assert soft_preferences_from_ranker_projection(
        _projection(_feature(kind="feature", value="lighter option", direction=None))
    ) == ()
    assert soft_preferences_from_ranker_projection(
        _projection(_feature(kind="category", value="cookware"))
    ) == ()


def test_confirmed_weight_preference_is_consumed_and_changes_same_candidate_order() -> None:
    records = {"heavy": _record("1.5 kg"), "light": _record("0.2 kg")}
    candidates = [_item("heavy"), _item("light")]
    ranking, combined, soft = rank_with_m5_soft_preferences(
        candidates, records, _projection(_feature())
    )
    assert len(soft) == 1
    assert combined[0].attribute == "weight"
    assert ranking is not None
    assert ranking.changed_order is True
    assert ranking.ranked_order == ("light", "heavy")
    assert ranking.baseline_order == ("heavy", "light")
    assert [item.parent_asin for item in candidates] == ["heavy", "light"]


def test_missing_weight_evidence_gets_no_fabricated_advantage() -> None:
    records = {"known": _record("1.0 kg"), "unknown": _record(None)}
    ranking, _, _ = rank_with_m5_soft_preferences(
        [_item("known"), _item("unknown")],
        records,
        _projection(_feature()),
    )
    assert ranking is not None
    assert ranking.ranked_order == ("known", "unknown")
    assert ranking.by_identity()["unknown"].unknown_count == 1


def test_retracted_preference_yields_no_soft_evidence() -> None:
    records = {"heavy": _record("1.5 kg"), "light": _record("0.2 kg")}
    ranking, combined, soft = rank_with_m5_soft_preferences(
        [_item("heavy"), _item("light")],
        records,
        _projection(),
    )
    assert ranking is None
    assert combined == ()
    assert soft == ()


def test_duplicate_legacy_and_m5_weight_sources_are_not_double_weighted() -> None:
    existing = (
        TaskPreference(
            attribute="weight",
            action="lower",
            value=None,
            option_id="legacy:weight:lower",
            label="legacy lighter",
        ),
    )
    merged = merge_soft_preferences(existing, soft_preferences_from_ranker_projection(_projection(_feature())))
    assert len(merged) == 1
    assert merged[0].option_id == "legacy:weight:lower"


def test_m5_material_categorical_preference_maps_to_supported_attribute() -> None:
    feature = _feature(preference_id="m5-material", kind="material", value="Titanium", direction=None)
    records = {"ti": SimpleNamespace(title="ti", details={"Material": "Titanium"}), "al": SimpleNamespace(title="al", details={"Material": "Aluminum"})}
    ranking, _, soft = rank_with_m5_soft_preferences(
        [_item("al"), _item("ti")], records, _projection(feature)
    )
    assert soft[0].attribute == "material"
    assert ranking is not None
    assert ranking.ranked_order == ("ti", "al")


def test_soft_ranking_never_adds_a_candidate_removed_by_hard_constraints() -> None:
    # The caller passes the hard-filtered set.  Soft ordering cannot resurrect the
    # excluded identity or change the candidate identity set.
    records = {"allowed": _record("1.0 kg"), "violator": _record("0.1 kg")}
    hard_eligible = [_item("allowed")]
    ranking, _, _ = rank_with_m5_soft_preferences(
        hard_eligible, records, _projection(_feature())
    )
    assert ranking is tuple() or ranking is None or ranking.ranked_order == ("allowed",)
    assert [item.parent_asin for item in hard_eligible] == ["allowed"]
    assert "violator" not in {item.parent_asin for item in hard_eligible}

"""Candidate-aware preference construction tests (shopping-only)."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from recommendation.demo.preference_construction import (  # noqa: E402
    augment_intent,
    blocked_attributes_from_history,
    build_preference_prompt,
    preference_from_option,
    rank_with_task_preference,
    resolve_preference_option,
)
from recommendation.demo.shopping_intent import ShoppingIntent, Strength  # noqa: E402
from tests.agent_reranking_fixture import (  # noqa: E402
    CANDIDATE_ROWS,
    build_index,
    build_tool_result,
)
from tests.demo_fixture import PROFILE_A, build_harness  # noqa: E402


ENDPOINT = "/v1/demo/sessions/{session}/agent/shopping"


def _records() -> dict[str, Any]:
    return dict(build_index().records)


def test_prompt_uses_real_price_difference_and_real_values() -> None:
    records = _records()
    prompt = build_preference_prompt(
        ("cand-red", "cand-blue", "cand-black"), records
    )
    assert prompt is not None
    assert prompt.attribute == "price"
    assert prompt.comparison_kind == "numeric"
    assert prompt.values == {"cand-red": 25.0, "cand-blue": 40.0, "cand-black": 15.0}
    assert {option.option_id for option in prompt.options} == {
        "pref:price:lower",
        "pref:price:higher",
        "pref:none",
        "pref:skip",
    }
    lower = next(option for option in prompt.options if option.action == "lower")
    assert "Lower price" in lower.label


def test_prompt_skips_an_explicitly_blocked_attribute() -> None:
    records = _records()
    prompt = build_preference_prompt(
        ("cand-red", "cand-blue", "cand-black"),
        records,
        blocked_attributes=("price", "weight"),
    )
    assert prompt is not None
    assert prompt.attribute == "color"
    assert prompt.values == {
        "cand-red": "red",
        "cand-blue": "blue",
        "cand-black": "black",
    }


def test_no_prompt_when_all_supported_attributes_are_equal_or_missing() -> None:
    base = _records()
    records: dict[str, Any] = {}
    for identity in ("cand-red", "cand-blue"):
        record = base[identity]
        records[identity] = record.model_copy(
            update={
                "price_text": "25.0",
                "details": (
                    ("Color", "red"),
                    ("Item Weight", "1.0 kg"),
                    ("Material", "cotton"),
                ),
            }
        )
    assert build_preference_prompt(("cand-red", "cand-blue"), records) is None


def test_prompt_never_infers_a_missing_attribute_from_titles() -> None:
    records = _records()
    # All titles differ, but title is not a supported preference dimension.
    prompt = build_preference_prompt(
        ("cand-red", "cand-blue"),
        records,
        blocked_attributes=("price", "color", "weight", "brand", "material"),
    )
    assert prompt is None


def test_supported_categorical_and_weight_attributes() -> None:
    base = _records()

    def make(identity: str, details: tuple[tuple[str, str], ...]) -> Any:
        record = base[identity]
        return record.model_copy(
            update={"price_text": "25.0", "details": details}
        )

    shared = {
        "Color": "red",
        "Item Weight": "1.0 kg",
        "Material": "cotton",
        "Brand Name": "Acme",
    }
    cases = {
        "weight": (
            make("cand-red", (("Color", "red"), ("Item Weight", "0.8 kg"), ("Material", "cotton"), ("Brand Name", "Acme"))),
            make("cand-blue", (("Color", "red"), ("Item Weight", "1.2 kg"), ("Material", "cotton"), ("Brand Name", "Acme"))),
        ),
        "material": (
            make("cand-red", (("Color", "red"), ("Item Weight", "1.0 kg"), ("Material", "cotton"), ("Brand Name", "Acme"))),
            make("cand-blue", (("Color", "red"), ("Item Weight", "1.0 kg"), ("Material", "leather"), ("Brand Name", "Acme"))),
        ),
        "brand": (
            make("cand-red", (("Color", "red"), ("Item Weight", "1.0 kg"), ("Material", "cotton"), ("Brand Name", "Acme"))),
            make("cand-blue", (("Color", "red"), ("Item Weight", "1.0 kg"), ("Material", "cotton"), ("Brand Name", "Beta"))),
        ),
    }
    for attribute, (red, blue) in cases.items():
        records = {"cand-red": red, "cand-blue": blue}
        prompt = build_preference_prompt(("cand-red", "cand-blue"), records)
        assert prompt is not None, attribute
        assert prompt.attribute == attribute, (attribute, prompt)
        if attribute == "weight":
            assert {option.action for option in prompt.options} >= {"lower", "higher"}
        else:
            assert any(option.action == "value" for option in prompt.options)


def test_lower_and_higher_preferences_change_order_without_changing_set() -> None:
    records = _records()
    candidates = build_tool_result().recommendations
    prompt = build_preference_prompt(
        tuple(item.parent_asin for item in candidates), records
    )
    assert prompt is not None
    lower_option = next(option for option in prompt.options if option.action == "lower")
    higher_option = next(option for option in prompt.options if option.action == "higher")
    lower = preference_from_option(prompt, lower_option)
    higher = preference_from_option(prompt, higher_option)
    lower_rank = rank_with_task_preference(candidates, records, lower)
    higher_rank = rank_with_task_preference(candidates, records, higher)
    assert lower_rank is not None and higher_rank is not None
    assert lower_rank.ranked_order == (
        "cand-black", "cand-red", "cand-blue", "cand-green"
    )
    assert higher_rank.ranked_order == (
        "cand-green", "cand-blue", "cand-red", "cand-black"
    )
    assert set(lower_rank.ranked_order) == {
        item.parent_asin for item in candidates
    }
    assert [item.rank for item in lower_rank.candidates] == [1, 2, 3, 4]


def test_categorical_preference_keeps_non_matching_candidates() -> None:
    records = _records()
    candidates = build_tool_result().recommendations
    prompt = build_preference_prompt(
        tuple(item.parent_asin for item in candidates), records,
        blocked_attributes=("price", "weight")
    )
    assert prompt is not None and prompt.attribute == "color"
    red = next(option for option in prompt.options if option.value == "red")
    ranking = rank_with_task_preference(
        candidates, records, preference_from_option(prompt, red)
    )
    assert ranking is not None
    assert ranking.ranked_order[0] == "cand-red"
    assert set(ranking.ranked_order) == {item.parent_asin for item in candidates}


def test_augment_intent_adds_a_soft_preference_not_a_hard_constraint() -> None:
    records = _records()
    prompt = build_preference_prompt(("cand-red", "cand-blue"), records)
    assert prompt is not None
    lower = next(option for option in prompt.options if option.action == "lower")
    preference = preference_from_option(prompt, lower)
    intent = augment_intent(ShoppingIntent(), preference)
    assert intent.hard == ()
    assert len(intent.soft) == 1
    entry = intent.soft[0]
    assert entry.kind == "price"
    assert entry.value == "lower"
    assert entry.strength is Strength.SOFT
    assert intent.active_preference_count == 1


def test_revoked_preference_attribute_is_blocked() -> None:
    class Revoked:
        is_active = False
        kind = type("Kind", (), {"value": "color"})()

    assert blocked_attributes_from_history([Revoked()]) == {"color"}


def _first_preference(
    harness: Any, *, constraints: list[dict[str, str]] | None = None
) -> tuple[str, dict[str, Any]]:
    session = harness.create_session(PROFILE_A)["session_id"]
    body: dict[str, Any] = {
        "message": "recommend outdoor gear",
        "k": 3,
        "sources": ["history"],
        "preference_discovery": True,
    }
    if constraints:
        body["constraints"] = constraints
    response = harness.client.post(ENDPOINT.format(session=session), json=body)
    assert response.status_code == 200, response.text
    return session, response.json()


def test_shopping_question_exposes_evidence_and_no_final_list(tmp_path: Path) -> None:
    harness = build_harness(tmp_path)
    try:
        _, body = _first_preference(harness)
        assert body["terminal"]["status"] == "waiting_for_user"
        assert body["terminal"]["succeeded"] is False
        assert body["recommendations"] == []
        assert body["clarification"]["kind"] == "preference"
        assert body["preference"]["attribute"] == "price"
        assert body["preference"]["values"] == {
            "cand-red": 25.0,
            "cand-blue": 40.0,
            "cand-black": 15.0,
        }
        assert body["clarification"]["options"]
        assert body["clarification"]["preference"] == body["preference"]
    finally:
        harness.runtime.close()


def test_selection_changes_final_order_and_intent(tmp_path: Path) -> None:
    harness = build_harness(tmp_path)
    try:
        # Baseline without opt-in.
        baseline_session = harness.create_session(PROFILE_A)["session_id"]
        baseline = harness.client.post(
            ENDPOINT.format(session=baseline_session),
            json={"message": "recommend outdoor gear", "k": 3, "sources": ["history"]},
        ).json()
        assert [item["parent_asin"] for item in baseline["recommendations"]] == [
            "cand-red", "cand-blue", "cand-black"
        ]

        session, question = _first_preference(harness)
        response = harness.client.post(
            ENDPOINT.format(session=session),
            json={
                "message": "lower price",
                "k": 3,
                "sources": ["history"],
                "clarification_id": question["clarification"]["clarification_id"],
                "clarification_option": "pref:price:lower",
            },
        )
        assert response.status_code == 200, response.text
        body = response.json()
        assert body["terminal"]["status"] == "finished"
        assert [item["parent_asin"] for item in body["recommendations"]] == [
            "cand-black", "cand-red", "cand-blue"
        ]
        assert body["intent"]["soft"] == [
            {
                "kind": "price",
                "value": "lower",
                "polarity": "prefer",
                "strength": "soft",
                "source_turn_id": "task-preference",
            }
        ]
        assert all(
            "price preference applied" in item["reason"]
            for item in body["recommendations"]
        )
        assert body["preference"] is None
    finally:
        harness.runtime.close()


def test_skip_creates_no_preference_and_keeps_baseline_order(tmp_path: Path) -> None:
    harness = build_harness(tmp_path)
    try:
        session, question = _first_preference(harness)
        response = harness.client.post(
            ENDPOINT.format(session=session),
            json={
                "message": "skip",
                "k": 3,
                "sources": ["history"],
                "clarification_id": question["clarification"]["clarification_id"],
                "clarification_option": "pref:skip",
            },
        )
        assert response.status_code == 200, response.text
        body = response.json()
        assert body["terminal"]["status"] == "finished"
        assert [item["parent_asin"] for item in body["recommendations"]] == [
            "cand-red", "cand-blue", "cand-black"
        ]
        assert body["intent"]["soft"] == []
        assert body["intent"]["active_preference_count"] == 0
        # The one-prompt budget is spent: a later request in the same task must not ask again.
        later = harness.client.post(
            ENDPOINT.format(session=session),
            json={
                "message": "recommend outdoor gear again",
                "k": 3,
                "sources": ["history"],
                "preference_discovery": True,
            },
        )
        assert later.status_code == 200, later.text
        assert later.json()["clarification"] is None
    finally:
        harness.runtime.close()


def test_hard_constraint_is_not_reasked_as_preference(tmp_path: Path) -> None:
    harness = build_harness(tmp_path)
    try:
        session, question = _first_preference(
            harness, constraints=[{"kind": "price_max", "value": "30"}]
        )
        assert question["preference"] is not None
        assert question["preference"]["attribute"] != "price"
        # Hard filtering already removed cand-blue before the preference question.
        assert "cand-blue" not in question["preference"]["products"]
        option = next(
            item for item in question["preference"]["options"]
            if item["action"] in {"lower", "higher", "value"}
        )
        response = harness.client.post(
            ENDPOINT.format(session=session),
            json={
                "message": "prefer red",
                "k": 3,
                "sources": ["history"],
                "clarification_id": question["clarification"]["clarification_id"],
                "clarification_option": option["option_id"],
                "constraints": [{"kind": "price_max", "value": "30"}],
            },
        )
        assert response.status_code == 200, response.text
        body = response.json()
        ids = [item["parent_asin"] for item in body["recommendations"]]
        assert ids == ["cand-red", "cand-black"]
        assert "cand-blue" not in ids
        assert all(item["reason"] for item in body["recommendations"])
    finally:
        harness.runtime.close()


def test_no_question_when_candidates_do_not_differ(tmp_path: Path) -> None:
    harness = build_harness(tmp_path, rows=CANDIDATE_ROWS[:2])
    try:
        records = harness.runtime.metadata.records
        for identity in ("cand-red", "cand-blue"):
            record = records[identity]
            records[identity] = record.model_copy(
                update={
                    "price_text": "25.0",
                    "details": (("Color", "red"),),
                }
            )
        session = harness.create_session(PROFILE_A)["session_id"]
        response = harness.client.post(
            ENDPOINT.format(session=session),
            json={
                "message": "recommend outdoor gear",
                "k": 2,
                "sources": ["history"],
                "preference_discovery": True,
            },
        )
        assert response.status_code == 200, response.text
        body = response.json()
        assert body["terminal"]["status"] == "finished"
        assert body["clarification"] is None
        assert body["recommendations"]
    finally:
        harness.runtime.close()


def test_invalid_option_does_not_consume_and_duplicate_is_rejected(tmp_path: Path) -> None:
    harness = build_harness(tmp_path)
    try:
        session, question = _first_preference(harness)
        clarification_id = question["clarification"]["clarification_id"]
        invalid = harness.client.post(
            ENDPOINT.format(session=session),
            json={
                "message": "bad",
                "clarification_id": clarification_id,
                "clarification_option": "pref:not-real",
            },
        )
        assert invalid.status_code == 422
        valid = harness.client.post(
            ENDPOINT.format(session=session),
            json={
                "message": "skip",
                "clarification_id": clarification_id,
                "clarification_option": "pref:skip",
            },
        )
        assert valid.status_code == 200, valid.text
        duplicate = harness.client.post(
            ENDPOINT.format(session=session),
            json={
                "message": "skip",
                "clarification_id": clarification_id,
                "clarification_option": "pref:skip",
            },
        )
        assert duplicate.status_code == 404
    finally:
        harness.runtime.close()


def test_cross_session_cannot_consume_preference_question(tmp_path: Path) -> None:
    harness = build_harness(tmp_path)
    try:
        first_session, question = _first_preference(harness)
        other_session = harness.create_session(PROFILE_A)["session_id"]
        cross = harness.client.post(
            ENDPOINT.format(session=other_session),
            json={
                "message": "skip",
                "clarification_id": question["clarification"]["clarification_id"],
                "clarification_option": "pref:skip",
            },
        )
        assert cross.status_code == 404
        valid = harness.client.post(
            ENDPOINT.format(session=first_session),
            json={
                "message": "skip",
                "clarification_id": question["clarification"]["clarification_id"],
                "clarification_option": "pref:skip",
            },
        )
        assert valid.status_code == 200, valid.text
    finally:
        harness.runtime.close()


def test_expired_preference_question_is_410(
    tmp_path: Path, monkeypatch: Any
) -> None:
    harness = build_harness(tmp_path)
    try:
        session, question = _first_preference(harness)
        from recommendation.demo import agent_service

        monkeypatch.setattr(agent_service, "CLARIFICATION_TTL_SECONDS", -1)
        expired = harness.client.post(
            ENDPOINT.format(session=session),
            json={
                "message": "skip",
                "clarification_id": question["clarification"]["clarification_id"],
                "clarification_option": "pref:skip",
            },
        )
        assert expired.status_code == 410
        again = harness.client.post(
            ENDPOINT.format(session=session),
            json={
                "message": "skip",
                "clarification_id": question["clarification"]["clarification_id"],
                "clarification_option": "pref:skip",
            },
        )
        assert again.status_code == 404
    finally:
        harness.runtime.close()

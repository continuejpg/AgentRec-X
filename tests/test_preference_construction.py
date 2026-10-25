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
    choose_preference_action,
    preference_from_option,
    rank_with_task_preference,
    resolve_preference_option,
)
from recommendation.demo.preference_construction import TaskPreference  # noqa: E402
from recommendation.demo.preference_construction import PreferenceState  # noqa: E402
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
    decision = choose_preference_action(
        ("cand-red", "cand-blue", "cand-black"),
        records,
        message="gear under a tight budget",
        top_k=3,
    )
    prompt = decision.dimension
    assert prompt is not None
    assert decision.action == "SHOW_CONTRAST"
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
    assert prompt.impact > 0


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
    decision = choose_preference_action(
        tuple(item.parent_asin for item in candidates),
        records,
        message="gear under a tight budget",
        top_k=4,
    )
    prompt = decision.dimension
    assert prompt is not None and prompt.attribute == "price"
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
    decision = choose_preference_action(
        ("cand-red", "cand-blue"),
        records,
        message="gear under a tight budget",
        top_k=2,
    )
    prompt = decision.dimension
    assert prompt is not None and prompt.attribute == "price"
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


def test_context_changes_which_dimension_is_asked() -> None:
    records = _records()
    ids = ("cand-red", "cand-blue", "cand-black")
    budget = choose_preference_action(
        ids, records, message="gear under a tight budget", top_k=3
    )
    lightweight = choose_preference_action(
        ids, records, message="lightweight camping gear", top_k=3
    )
    color = choose_preference_action(
        ids, records, message="colorful camping gear", top_k=3
    )
    assert budget.dimension is not None and budget.dimension.attribute == "price"
    assert lightweight.dimension is not None and lightweight.dimension.attribute == "weight"
    assert color.dimension is not None and color.dimension.attribute == "color"


def test_interaction_modes_follow_evidence_and_context() -> None:
    records = _records()
    two = choose_preference_action(
        ("cand-red", "cand-blue"),
        records,
        message="gear under a tight budget",
        top_k=2,
    )
    three = choose_preference_action(
        ("cand-red", "cand-blue", "cand-black"),
        records,
        message="gear under a tight budget",
        top_k=3,
    )
    prior = (
        TaskPreference(
            attribute="price",
            action="lower",
            value=None,
            option_id="pref:price:lower",
            label="Lower price",
        ),
    )
    explained = choose_preference_action(
        ("cand-red", "cand-blue", "cand-black"),
        records,
        current_preferences=prior,
        message="lightweight gear",
        top_k=3,
        round_no=2,
    )
    assert two.action == "ASK_DIRECT"
    assert three.action == "SHOW_CONTRAST"
    assert explained.action == "EXPLAIN"
    assert explained.dimension is not None
    assert explained.dimension.explanation


def test_skip_when_no_presented_top_k_can_change() -> None:
    records = _records()
    decision = choose_preference_action(
        ("cand-red", "cand-blue", "cand-black"),
        records,
        message="gear under a tight budget",
        top_k=0,
    )
    assert decision.action == "SKIP"
    assert decision.dimension is None


def test_no_second_round_when_only_the_confirmed_dimension_differs() -> None:
    base = _records()
    records: dict[str, Any] = {}
    for identity in ("cand-red", "cand-blue", "cand-black"):
        record = base[identity]
        records[identity] = record.model_copy(
            update={
                "price_text": "25.0" if identity != "cand-blue" else "40.0",
                "details": (("Color", "red"), ("Item Weight", "1.0 kg")),
            }
        )
    prior = (
        TaskPreference(
            attribute="price",
            action="lower",
            value=None,
            option_id="pref:price:lower",
            label="Lower price",
        ),
    )
    decision = choose_preference_action(
        ("cand-red", "cand-blue", "cand-black"),
        records,
        current_preferences=prior,
        message="gear under a tight budget",
        top_k=3,
        round_no=2,
    )
    assert decision.action == "SKIP"
    assert decision.dimension is None


def test_preference_state_blocks_known_and_revoked_dimensions() -> None:
    records = _records()
    state = PreferenceState(
        explicit=frozenset({"price"}),
        revoked=frozenset({"color"}),
    )
    decision = choose_preference_action(
        ("cand-red", "cand-blue", "cand-black"),
        records,
        message="gear under a tight budget",
        state=state,
        top_k=3,
    )
    assert decision.dimension is not None
    assert decision.dimension.attribute not in {"price", "color"}


def _first_preference(
    harness: Any,
    *,
    constraints: list[dict[str, str]] | None = None,
    message: str = "recommend gear under a tight budget",
) -> tuple[str, dict[str, Any]]:
    session = harness.create_session(PROFILE_A)["session_id"]
    body: dict[str, Any] = {
        "message": message,
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
            json={
                "message": "recommend gear under a tight budget",
                "k": 3,
                "sources": ["history"],
            },
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
        first_body = response.json()
        assert first_body["terminal"]["status"] == "waiting_for_user"
        assert first_body["preference"]["attribute"] != "price"
        assert first_body["preference_outcome"]["selected_option_id"] == "pref:price:lower"
        assert first_body["preference_outcome"]["verified"] is True
        assert first_body["preference_outcome"]["actual_top_k"] == [
            "cand-black", "cand-red", "cand-blue"
        ]
        second_option = next(
            item for item in first_body["preference"]["options"]
            if item["action"] in {"lower", "higher", "value"}
        )
        second = harness.client.post(
            ENDPOINT.format(session=session),
            json={
                "message": "choose",
                "k": 3,
                "sources": ["history"],
                "clarification_id": first_body["clarification"]["clarification_id"],
                "clarification_option": second_option["option_id"],
            },
        )
        assert second.status_code == 200, second.text
        body = second.json()
        assert body["terminal"]["status"] == "finished"
        assert [item["parent_asin"] for item in body["recommendations"]] == [
            "cand-red", "cand-blue", "cand-black"
        ]
        assert [entry["kind"] for entry in body["intent"]["soft"]] == [
            "price", "weight"
        ]
        assert body["preference_outcome"]["selected_option_id"] == second_option["option_id"]
        assert body["preference_outcome"]["verified"] is True
        assert all(
            "preference applied" in item["reason"]
            for item in body["recommendations"]
        )
        assert body["preference"] is None
        # A task-scoped answer is not a long-term preference write.
        assert ("price", "prefer", "lower") not in harness.active_values(session)
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
        if body.get("clarification") is not None:
            # A second, still valuable dimension may remain after the first answer.  Stop it
            # explicitly; it must not relax the hard price constraint.
            stopped = harness.client.post(
                ENDPOINT.format(session=session),
                json={
                    "message": "skip",
                    "k": 3,
                    "sources": ["history"],
                    "clarification_id": body["clarification"]["clarification_id"],
                    "clarification_option": "pref:skip",
                    "constraints": [{"kind": "price_max", "value": "30"}],
                },
            )
            assert stopped.status_code == 200, stopped.text
            body = stopped.json()
        assert body["terminal"]["status"] == "finished"
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


def test_preference_task_state_is_isolated_per_demo_session(tmp_path: Path) -> None:
    harness = build_harness(tmp_path)
    try:
        first_session, first_question = _first_preference(harness)
        second_session, second_question = _first_preference(harness)
        assert first_question["clarification"]["clarification_id"] != (
            second_question["clarification"]["clarification_id"]
        )
        # Answering the first session's prompt must not consume or alter the second's.
        first_answer = harness.client.post(
            ENDPOINT.format(session=first_session),
            json={
                "message": "skip",
                "clarification_id": first_question["clarification"]["clarification_id"],
                "clarification_option": "pref:skip",
            },
        )
        assert first_answer.status_code == 200, first_answer.text
        second_answer = harness.client.post(
            ENDPOINT.format(session=second_session),
            json={
                "message": "skip",
                "clarification_id": second_question["clarification"]["clarification_id"],
                "clarification_option": "pref:skip",
            },
        )
        assert second_answer.status_code == 200, second_answer.text
        assert second_answer.json()["terminal"]["status"] == "finished"
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

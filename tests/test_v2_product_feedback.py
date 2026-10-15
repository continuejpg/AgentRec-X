"""E-03C: product-level feedback bound to real presented cards, and its effect on the next turn.

Every test drives the real Chat API through the real chain (session, M9 store, controller, identity
mapping, M10A/M10B, RRF and the final response).  The model is a fake that answers from the payload it
is handed, so nothing here depends on model prose: identity comes from the cards the session really
presented, and a write is believed only when the store's own snapshot says so.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from recommendation.demo.feedback import (
    PresentedCard,
    feedback_candidate,
    resolve_reference,
)
from recommendation.memory.schemas import PreferenceKind, PreferencePolarity
from tests.test_v2_bounded_replanning import (  # the E-03B fixture: two sources, differing products
    ALL_ROWS,
    DIRECT,
    PLAN_CATALOG,
    PLAN_HISTORY,
    REQUEST,
    _harness,
    _ReactiveClient,
    _turn,
)
from tests.demo_fixture import close_harness

PROVIDER_ENV = {
    "AGENTRECX_LLM_BASE_URL": "http://127.0.0.1:9/v1",
    "AGENTRECX_LLM_MODEL": "deepseek-chat",
    "AGENTRECX_LLM_API_KEY": "unit-test-placeholder-not-a-credential",
    "AGENTRECX_LLM_PROFILE": "deepseek",
}


@pytest.fixture()
def configured(monkeypatch: pytest.MonkeyPatch) -> None:
    for name, value in PROVIDER_ENV.items():
        monkeypatch.setenv(name, value)


def _ids(body: dict[str, Any]) -> list[str]:
    return [card["parent_asin"] for card in body["recommendations"]]


def _marks(body: dict[str, Any]) -> dict[str, str]:
    return {
        card["parent_asin"]: card["feedback"]
        for card in body["recommendations"]
        if card.get("feedback")
    }


def _feedback(body: dict[str, Any]) -> dict[str, Any]:
    feedback = body["audit"]["feedback"]
    assert feedback is not None, "a feedback block is always present on the llm_agent path"
    return feedback


def _recommend(harness: Any, session: str, *, k: int = 4) -> dict[str, Any]:
    return _turn(
        harness,
        _ReactiveClient(PLAN_HISTORY, replan_source="catalog_search"),
        REQUEST,
        k=k,
        session=session,
    )


def _say(harness: Any, session: str, message: str) -> dict[str, Any]:
    return _turn(harness, _ReactiveClient(DIRECT), message, session=session)


# --------------------------------------------------------------------------- #
# A. an explicit positive statement about one presented card
# --------------------------------------------------------------------------- #


def test_liking_a_presented_card_is_bound_promoted_and_marked(
    tmp_path: Path, configured: None
) -> None:
    harness = _harness(tmp_path)
    try:
        session = harness.create_session()["session_id"]
        first = _recommend(harness, session)
        assert _ids(first) == ["k-hist", "k-black", "k-blue", "k-red"]
        # The session really recorded what it showed, with the positions the user saw.
        assert [
            (card["position"], card["parent_asin"])
            for card in harness.manager.get(session).last_presentation
        ] == [(1, "k-hist"), (2, "k-black"), (3, "k-blue"), (4, "k-red")]

        statement = _say(harness, session, "I like the second one.")
        feedback = _feedback(statement)
        assert (feedback["status"], feedback["action"]) == ("resolved", "like")
        assert (feedback["parent_asin"], feedback["item_id"], feedback["final_position"]) == (
            "k-black", 804, 2)
        assert feedback["written"] is True and feedback["entry_status"] == "active"
        assert feedback["effect"] == "promote"
        assert feedback["active_product_feedback"] == [
            {"parent_asin": "k-black", "polarity": "prefer"}
        ]
        # The write is exactly the product statement: the reference itself was not also stored as
        # a preference of its own, and the panel shows one fact about one product.
        assert [
            (entry["kind"], entry["polarity"], entry["value"])
            for entry in statement["memory_update"]["added"]
        ] == [("product", "prefer", "k-black")]
        assert [
            (entry["kind"], entry["polarity"], entry["value"])
            for entry in statement["active_preferences"]
        ] == [("product", "prefer", "k-black")]

        second = _recommend(harness, session)
        assert _ids(second)[0] == "k-black", "the liked product now leads the same pool"
        assert _marks(second) == {"k-black": "liked"}
        # Nothing else about the ranking changed: same candidates, same provenance, same scores.
        assert set(_ids(second)) == set(_ids(first))
        assert _cards_scores(first) == _cards_scores(second)
    finally:
        close_harness(harness)


def _cards_scores(body: dict[str, Any]) -> dict[str, tuple[Any, ...]]:
    return {
        card["parent_asin"]: (card["sasrec_score"], card["score_kind"], tuple(card["fusion_sources"]))
        for card in body["recommendations"]
    }


# --------------------------------------------------------------------------- #
# B. an explicit negative statement about one card, and nothing wider
# --------------------------------------------------------------------------- #


def test_disliking_one_card_does_not_generalise_to_its_colour_or_brand(
    tmp_path: Path, configured: None
) -> None:
    harness = _harness(tmp_path)
    try:
        session = harness.create_session()["session_id"]
        _recommend(harness, session)
        statement = _say(harness, session, "I don't like the first one.")
        feedback = _feedback(statement)
        assert (feedback["status"], feedback["action"], feedback["parent_asin"]) == (
            "resolved", "dislike", "k-hist")

        active = statement["active_preferences"]
        assert [(entry["kind"], entry["polarity"], entry["value"]) for entry in active] == [
            ("product", "avoid", "k-hist")
        ], (
            "the only stored fact is about that product: no colour, no brand, no category"
        )
        # The sentence's generic reading was dropped rather than stored: the turn added exactly the
        # product statement and nothing else.
        assert [
            (entry["kind"], entry["value"]) for entry in statement["memory_update"]["added"]
        ] == [("product", "k-hist")]

        second = _recommend(harness, session)
        assert _ids(second)[-1] == "k-hist", "the disliked product sinks to the end"
        assert _marks(second) == {"k-hist": "disliked"}
    finally:
        close_harness(harness)


def test_a_colour_remark_about_a_card_is_still_only_about_that_card(
    tmp_path: Path, configured: None
) -> None:
    """§5's worked example: feedback about "the black one" is not a dislike of black products."""
    harness = _harness(tmp_path)
    try:
        session = harness.create_session()["session_id"]
        _recommend(harness, session)
        statement = _say(harness, session, "I don't like the black shoes you showed me.")
        feedback = _feedback(statement)
        # One card carries the colour in this fixture, so the reference is unambiguous.
        assert feedback["status"] in ("resolved", "ambiguous")
        assert all(
            entry["kind"] == "product"
            for entry in statement["active_preferences"]
        ), "no colour preference may be inferred from product feedback"
        assert not any(
            entry["kind"] == "color" for entry in statement["active_preferences"]
        )
    finally:
        close_harness(harness)


# --------------------------------------------------------------------------- #
# C. an explicit request not to be shown it again
# --------------------------------------------------------------------------- #


def test_an_explicit_exclusion_request_is_reported_as_a_demotion_not_a_removal(
    tmp_path: Path, configured: None
) -> None:
    harness = _harness(tmp_path)
    try:
        session = harness.create_session()["session_id"]
        _recommend(harness, session)
        statement = _say(harness, session, "Please don't recommend the first one again.")
        feedback = _feedback(statement)
        assert (feedback["status"], feedback["action"]) == ("resolved", "exclude")
        assert feedback["effect"] == "demote", (
            "this deployment orders candidates by M10A evidence: an exclusion request is honoured "
            "as a demotion and reported as one, never as a hard removal"
        )
        assert feedback["parent_asin"] == "k-hist"
        # The demotion is real: the product leaves the answer when the pool exceeds k.
        narrow = _turn(
            harness,
            _ReactiveClient(PLAN_HISTORY, replan_source="catalog_search"),
            REQUEST,
            k=2,
            session=session,
        )
        assert "k-hist" not in _ids(narrow)
        assert len(_ids(narrow)) == 2
    finally:
        close_harness(harness)


# --------------------------------------------------------------------------- #
# D. correction and retraction
# --------------------------------------------------------------------------- #


def test_retracting_feedback_restores_the_previous_order(
    tmp_path: Path, configured: None
) -> None:
    harness = _harness(tmp_path)
    try:
        session = harness.create_session()["session_id"]
        baseline = _ids(_recommend(harness, session))
        _say(harness, session, "I like the second one.")
        promoted = _ids(_recommend(harness, session))
        assert promoted[0] == "k-black" and promoted != baseline

        retraction = _say(harness, session, "Forget what I said about the first one.")
        feedback = _feedback(retraction)
        assert (feedback["action"], feedback["written"], feedback["entry_status"]) == (
            "retract", True, "removed")
        assert feedback["active_product_feedback"] == []

        restored = _ids(_recommend(harness, session))
        assert restored == baseline, "the retracted feedback no longer affects the order"
    finally:
        close_harness(harness)


def test_a_correction_replaces_the_attitude_instead_of_adding_one(
    tmp_path: Path, configured: None
) -> None:
    harness = _harness(tmp_path)
    try:
        session = harness.create_session()["session_id"]
        baseline = _ids(_recommend(harness, session))
        _say(harness, session, "I don't like the second one.")     # k-black -> last
        avoided = _ids(_recommend(harness, session))
        assert avoided[-1] == "k-black"
        assert avoided == ["k-hist", "k-blue", "k-red", "k-black"]

        # "the first one" is the card the user now sees first, so this corrects the same product.
        correction = _say(harness, session, "Actually I like the first one instead.")
        feedback = _feedback(correction)
        assert (feedback["action"], feedback["parent_asin"]) == ("correct", "k-hist")
        assert feedback["written"] is True
        active = {
            entry["parent_asin"]: entry["polarity"]
            for entry in feedback["active_product_feedback"]
        }
        assert active["k-hist"] == "prefer", (
            "a correction stores the attitude the wording expressed, not its opposite"
        )
        assert set(baseline) == set(_ids(_recommend(harness, session)))
    finally:
        close_harness(harness)


def test_a_retraction_that_targets_nothing_claims_nothing(
    tmp_path: Path, configured: None
) -> None:
    harness = _harness(tmp_path)
    try:
        session = harness.create_session()["session_id"]
        _recommend(harness, session)
        statement = _say(harness, session, "Forget what I said about the third one.")
        feedback = _feedback(statement)
        assert feedback["status"] == "resolved" and feedback["action"] == "retract"
        assert feedback["written"] is False and feedback["entry_status"] is None, (
            "nothing was stored for that product, so nothing was removed - and the response says so"
        )
    finally:
        close_harness(harness)


# --------------------------------------------------------------------------- #
# E. references follow what the user saw, not the raw order
# --------------------------------------------------------------------------- #


def test_the_first_card_is_the_one_the_user_saw_after_ranking(
    tmp_path: Path, configured: None
) -> None:
    harness = _harness(tmp_path)
    try:
        session = harness.create_session()["session_id"]
        baseline = _ids(_recommend(harness, session))
        assert baseline[0] == "k-hist"
        _say(harness, session, "I like the second one.")          # k-black -> first
        after = _ids(_recommend(harness, session))
        assert after == ["k-black", "k-hist", "k-blue", "k-red"]

        statement = _say(harness, session, "I don't like the first one.")
        feedback = _feedback(statement)
        assert feedback["parent_asin"] == "k-black", (
            "the first card is the one the ranking put first, not the fused order's first"
        )
        assert feedback["final_position"] == 1
        # The raw (fused) order is still visible separately, so the two can never be confused.
        assert statement["audit"]["original_order"] == []
        assert _ids(_recommend(harness, session))[-1] == "k-black"
    finally:
        close_harness(harness)


# --------------------------------------------------------------------------- #
# F. two sessions
# --------------------------------------------------------------------------- #


def test_feedback_never_crosses_a_session_boundary(tmp_path: Path, configured: None) -> None:
    harness = _harness(tmp_path)
    try:
        first = harness.create_session()["session_id"]
        second = harness.create_session()["session_id"]
        assert harness.manager.get(first).user_key != harness.manager.get(second).user_key
        _recommend(harness, first)
        _recommend(harness, second)

        _say(harness, first, "I like the second one.")
        body_first = _recommend(harness, first)
        body_second = _recommend(harness, second)

        assert body_first["recommendations"][0]["parent_asin"] == "k-black"
        assert body_second["recommendations"][0]["parent_asin"] == "k-hist", (
            "the other session has no feedback at all"
        )
        assert _marks(body_second) == {}
        assert body_second["active_preferences"] == []
        # A session can only refer to cards it was shown itself.
        statement = _say(harness, second, "Forget what I said about the second one.")
        assert _feedback(statement)["written"] is False
    finally:
        close_harness(harness)


# --------------------------------------------------------------------------- #
# G. references that must not write anything
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(("message", "status"), [
    ("I like it.", "ambiguous"),                                 # several cards, no reference
    ("I like the seventh one.", "ambiguous"),                    # no such position
    ("I like B0NOTASHOWN.", "unknown_product"),                  # a real-looking identity, never shown
    ("I like item 999999.", "unknown_product"),                  # an item id that was not shown
    ("hello there", "none"),                                     # not feedback at all
])
def test_unresolvable_references_write_nothing(
    tmp_path: Path, configured: None, message: str, status: str
) -> None:
    harness = _harness(tmp_path)
    try:
        session = harness.create_session()["session_id"]
        _recommend(harness, session)
        before = _recommend(harness, session)
        statement = _say(harness, session, message)
        feedback = _feedback(statement)
        assert feedback["status"] == status, message
        assert feedback["written"] is False and feedback["parent_asin"] is None
        assert statement["active_preferences"] == []
        assert _ids(_recommend(harness, session)) == _ids(before), "no ordering effect either"
    finally:
        close_harness(harness)


def test_feedback_before_anything_was_presented_writes_nothing(
    tmp_path: Path, configured: None
) -> None:
    harness = _harness(tmp_path)
    try:
        session = harness.create_session()["session_id"]
        statement = _say(harness, session, "I like the first one.")
        feedback = _feedback(statement)
        assert feedback["status"] == "no_presentation"
        assert feedback["written"] is False
        assert statement["active_preferences"] == []
    finally:
        close_harness(harness)


def test_a_product_cut_by_k_can_never_be_a_feedback_target(
    tmp_path: Path, configured: None
) -> None:
    harness = _harness(tmp_path)
    try:
        session = harness.create_session()["session_id"]
        narrow = _turn(
            harness,
            _ReactiveClient(PLAN_HISTORY, replan_source="catalog_search"),
            REQUEST,
            k=2,
            session=session,
        )
        assert _ids(narrow) == ["k-black", "k-hist"], "the measured top-2 of the same pool"
        assert [card["position"] for card in harness.manager.get(session).last_presentation] == [
            1, 2]
        # The other two were retrieved but not presented, so they are not referenceable.
        statement = _say(harness, session, "I like the third one.")
        feedback = _feedback(statement)
        assert feedback["status"] == "ambiguous" and feedback["written"] is False
        statement = _say(harness, session, "I like item 802.")   # k-red: retrieved, never shown
        assert _feedback(statement)["status"] == "unknown_product"
        assert _feedback(statement)["written"] is False
    finally:
        close_harness(harness)


def test_the_resolver_itself_refuses_rather_than_guesses() -> None:
    cards = [
        PresentedCard("A" * 10, 801, 1, "Hist Widget"),
        PresentedCard("B" * 10, 802, 2, "Red Widget"),
    ]
    assert resolve_reference("I like the second one.", cards).card.parent_asin == "B" * 10
    assert resolve_reference("I like it.", cards).status == "ambiguous"
    assert resolve_reference("I like the ninth one.", cards).status == "ambiguous"
    assert resolve_reference("I like " + "Z" * 10 + ".", cards).status == "unknown_product"
    assert resolve_reference("Recommend something.", cards).action == "none"
    # A presentation of exactly one card makes a deictic unambiguous.
    assert resolve_reference(
        "I don't like it.", [PresentedCard("C" * 10, 803, 1, "Black Shoes")]
    ).resolved
    # The candidate the resolver produces is an identity-level M9 statement, never free text.
    candidate = feedback_candidate(resolve_reference("I like the first one.", cards))
    assert candidate.kind is PreferenceKind.PRODUCT
    assert candidate.value == "A" * 10
    assert candidate.polarity is PreferencePolarity.PREFER


# --------------------------------------------------------------------------- #
# H. multi-source: feedback applies to the one fused, de-duplicated pool
# --------------------------------------------------------------------------- #


def test_feedback_applies_to_the_fused_pool_without_touching_provenance(
    tmp_path: Path, configured: None
) -> None:
    harness = _harness(tmp_path)
    try:
        session = harness.create_session()["session_id"]
        both = _turn(
            harness,
            _ReactiveClient(
                {"route": "recommend", "selected_sources": ["history", "catalog_search"],
                 "reason": "both"},
                replan_source=None,
            ),
            REQUEST,
            k=4,
            session=session,
        )
        assert set(_ids(both)) == {"k-hist", "k-red", "k-blue", "k-black"}
        provenance = {
            card["parent_asin"]: tuple(card["fusion_sources"])
            for card in both["recommendations"]
        }
        _say(harness, session, "I like the fourth one.")
        after = _turn(
            harness,
            _ReactiveClient(
                {"route": "recommend", "selected_sources": ["history", "catalog_search"],
                 "reason": "both"},
                replan_source=None,
            ),
            REQUEST,
            k=4,
            session=session,
        )
        assert _marks(after) == {"k-red": "liked"}
        assert after["recommendations"][0]["parent_asin"] == "k-red"
        assert {
            card["parent_asin"]: tuple(card["fusion_sources"])
            for card in after["recommendations"]
        } == provenance, "identity, de-duplication and provenance are untouched by feedback"
    finally:
        close_harness(harness)


# --------------------------------------------------------------------------- #
# I. with bounded replanning enabled
# --------------------------------------------------------------------------- #


def test_feedback_still_applies_when_bounded_replanning_runs(
    tmp_path: Path, configured: None
) -> None:
    harness = _harness(tmp_path)
    try:
        session = harness.create_session()["session_id"]
        first = _turn(
            harness,
            _ReactiveClient(PLAN_HISTORY, replan_source="catalog_search"),
            REQUEST,
            k=4,
            session=session,
        )
        assert first["audit"]["replan"]["status"] == "appended_candidates"
        assert _ids(first) == ["k-hist", "k-black", "k-blue", "k-red"]

        _say(harness, session, "I like the fourth one.")
        client = _ReactiveClient(PLAN_HISTORY, replan_source="catalog_search")
        after = _turn(harness, client, REQUEST, k=4, session=session)

        assert after["audit"]["replan"]["status"] == "appended_candidates", (
            "the appended source still runs, under the same gate and the same budget"
        )
        assert client.replan_calls == 1, "feedback never buys a second decision"
        assert client.replan_payloads[0]["observation"]["requested_k"] == 4
        assert _marks(after) == {"k-red": "liked"}
        assert after["recommendations"][0]["parent_asin"] == "k-red", (
            "the appended source's product obeys the same active feedback"
        )
        assert after["agent"]["terminal"]["tool_calls"] == 2
    finally:
        close_harness(harness)


# --------------------------------------------------------------------------- #
# J. the old paths, and honest failure reporting
# --------------------------------------------------------------------------- #


def test_ordinary_preferences_still_work_and_are_not_mistaken_for_feedback(
    tmp_path: Path, configured: None
) -> None:
    harness = _harness(tmp_path)
    try:
        session = harness.create_session()["session_id"]
        _recommend(harness, session)
        statement = _say(harness, session, "I prefer blue.")
        feedback = _feedback(statement)
        assert feedback["action"] == "none" and feedback["written"] is False, (
            "a statement that names no product is ordinary preference memory, not product feedback"
        )
        assert [
            (entry["kind"], entry["polarity"], entry["value"])
            for entry in statement["active_preferences"]
        ] == [("color", "prefer", "blue")]
        body = _recommend(harness, session)
        assert body["recommendations"][0]["parent_asin"] == "k-blue"
        assert _marks(body) == {}, "an ordinary preference is not a product feedback mark"
    finally:
        close_harness(harness)


def test_the_other_paths_do_not_read_session_feedback(tmp_path: Path, configured: None) -> None:
    harness = _harness(tmp_path)
    try:
        session = harness.create_session()["session_id"]
        _recommend(harness, session)
        _say(harness, session, "I like the second one.")

        # The request-declared agent endpoint keeps its own contract: no session feedback block.
        response = harness.client.post(
            "/v1/demo/agent/recommend",
            json={"message": REQUEST, "k": 4, "sources": ["history"],
                  "history": list(harness.manager.get(session).trusted_user_history)},
        )
        assert response.status_code == 200, response.text
        assert response.json()["replan_execution"] is None

        # The deterministic chat mode reports no feedback block at all.
        deterministic = harness.client.post(
            f"/v1/demo/sessions/{session}/chat",
            json={"message": "Recommend something.", "k": 4,
                  "decision_mode": "deterministic"},
        )
        assert deterministic.status_code == 200
        assert deterministic.json()["audit"]["feedback"] is None
    finally:
        close_harness(harness)


def test_a_reference_is_never_stored_as_a_preference_value() -> None:
    """M9 must not read meaning into "the second one": that is a reference, not a property."""
    from recommendation.memory.extraction import RuleBasedPreferenceExtractor

    extractor = RuleBasedPreferenceExtractor()
    for message in (
        "I like the second one.",
        "Please don't recommend the first one again.",
        "Forget what I said about the third one.",
    ):
        assert extractor.extract(message).preferences == (), message
    # An ordinary statement is untouched.
    assert [
        (candidate.kind.value, candidate.value)
        for candidate in extractor.extract("I prefer black.").preferences
    ] == [("color", "black")]


def test_the_feedback_block_never_carries_a_store_handle(
    tmp_path: Path, configured: None
) -> None:
    harness = _harness(tmp_path)
    try:
        session = harness.create_session()["session_id"]
        _recommend(harness, session)
        statement = _say(harness, session, "I like the second one.")
        rendered = json.dumps(statement)
        for forbidden in ("memory_id", "logical_seq", "source_text", "user_key"):
            assert forbidden not in rendered, forbidden
        assert _feedback(statement)["entry_status"] == "active"
    finally:
        close_harness(harness)

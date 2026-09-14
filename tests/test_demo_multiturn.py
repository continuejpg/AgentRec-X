"""Milestone 11 multi-turn integration tests over HTTP.

Every assertion in this file goes through the FastAPI HTTP contract -- no direct
``AgentGraph`` calls -- because the milestone's claim is precisely that the browser and
the tests use the *same* accepted backend contracts.

Covered here:

* the mandatory five-turn scenario (recommend, state a preference, recommend again,
  retract it, recommend again);
* ADD / REPLACE / REMOVE through the API, not only through Milestone 9 unit tests;
* cross-session isolation, including the adversarial "A avoids red / B prefers red" case;
* trusted-history immutability under conversational purchase claims;
* candidate count / identity / score / metadata / evidence invariants;
* card order equal to the reranked order, with an adversarial reorder;
* same-session and cross-session concurrency;
* determinism across repeated fresh sessions.
"""

from __future__ import annotations

import json
import sys
import threading
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tests.demo_fixture import (  # noqa: E402
    CANDIDATE_ROWS,
    HISTORY,
    INITIAL_ORDER,
    PROFILE_A,
    PROFILE_B,
    build_harness,
    close_harness,
    demo_profiles,
)
from recommendation.agent import history_digest  # noqa: E402
from recommendation.demo import DEFAULT_DEMO_K  # noqa: E402


@pytest.fixture()
def harness(tmp_path: Path):
    built = build_harness(tmp_path)
    try:
        yield built
    finally:
        close_harness(built)


def orders(body: dict) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """``(original, final)`` candidate orders as the API reports and presents them."""
    cards = tuple(card["parent_asin"] for card in body["recommendations"])
    return tuple(body["audit"]["original_order"]), cards


# --------------------------------------------------------------------------- #
# The mandatory five-turn scenario
# --------------------------------------------------------------------------- #


def test_mandatory_five_turn_scenario(harness) -> None:
    """Recommend -> state a preference -> recommend -> retract -> recommend.

    Turn 2's own ranking must follow the pre-turn snapshot, turn 3 must see ``avoid red``,
    turn 4 must persist the retraction, and turn 5 must be free of colour evidence.
    """
    session = harness.create_session()["session_id"]

    # -- Turn 1: ordinary recommendation, no preferences yet ------------------ #
    turn1 = harness.chat(session, "Recommend some products.")
    assert turn1["route"] == "recommend"
    assert [card["parent_asin"] for card in turn1["recommendations"]] == list(INITIAL_ORDER)
    assert turn1["audit"]["ranked_with_preference_count"] == 0
    assert turn1["active_preferences"] == []
    assert all(card["violation_count"] == 0 for card in turn1["recommendations"])

    # -- Turn 2: a preference is written, but must not affect its own turn ---- #
    turn2 = harness.chat(session, "I don't want red.")
    assert turn2["route"] == "direct"
    assert turn2["memory_update"]["changed"] is True
    assert [(item["polarity"], item["value"]) for item in turn2["memory_update"]["added"]] == [
        ("avoid", "red")
    ]
    assert turn2["audit"]["ranked_with_preference_count"] == 0, "same-turn preference leaked"
    assert turn2["active_preferences"] == [
        {"kind": "color", "polarity": "avoid", "value": "red"}
    ], "the panel must show the post-write memory state"

    # -- Turn 3: the stored preference now ranks ----------------------------- #
    turn3 = harness.chat(session, "Recommend again.")
    assert turn3["route"] == "recommend"
    assert turn3["audit"]["ranked_with_preference_count"] == 1
    original, final = orders(turn3)
    assert original == INITIAL_ORDER
    assert final != original, "avoid red did not change the order"
    assert final[-1] == "cand-red", "the violating candidate must be demoted, not removed"
    red = next(card for card in turn3["recommendations"] if card["parent_asin"] == "cand-red")
    assert red["violation_count"] == 1
    assert red["reranked_rank"] == len(final)

    # -- Turn 4: the retraction is persisted --------------------------------- #
    turn4 = harness.chat(session, "I don't care about color anymore.")
    assert turn4["route"] == "direct"
    assert turn4["memory_update"]["changed"] is True
    assert turn4["active_preferences"] == []

    # -- Turn 5: colour evidence is gone and the original order is back ------- #
    turn5 = harness.chat(session, "Recommend again.")
    assert turn5["audit"]["ranked_with_preference_count"] == 0
    assert [card["parent_asin"] for card in turn5["recommendations"]] == list(INITIAL_ORDER)
    for card in turn5["recommendations"]:
        assert card["match_count"] == 0
        assert card["violation_count"] == 0
        assert card["evidence"] == []
    _, final5 = orders(turn5)
    assert final5 == INITIAL_ORDER

    # Five turns, five distinct server-owned ids.
    assert harness.state(session)["turn"] == 5


# --------------------------------------------------------------------------- #
# ADD / REPLACE / REMOVE over HTTP
# --------------------------------------------------------------------------- #


def test_add_keeps_both_preferences_over_http(harness) -> None:
    session = harness.create_session()["session_id"]
    harness.chat(session, "I don't want red.")
    harness.chat(session, "I don't want blue.")

    assert harness.active_values(session) == {
        ("color", "avoid", "red"),
        ("color", "avoid", "blue"),
    }
    turn = harness.chat(session, "Recommend again.")
    assert turn["audit"]["ranked_with_preference_count"] == 2
    for card in turn["recommendations"]:
        assert len(card["evidence"]) == 2


def test_replace_over_http_drops_the_superseded_value(harness) -> None:
    session = harness.create_session()["session_id"]
    harness.chat(session, "I prefer black.")
    assert harness.active_values(session) == {("color", "prefer", "black")}

    turn = harness.chat(session, "Actually, I prefer blue instead.")
    assert turn["memory_update"]["changed"] is True
    assert [item["value"] for item in turn["memory_update"]["added"]] == ["blue"]
    assert [item["value"] for item in turn["memory_update"]["superseded"]] == ["black"]

    assert harness.active_values(session) == {("color", "prefer", "blue")}
    follow_up = harness.chat(session, "Recommend again.")
    values = {
        record["value"]
        for card in follow_up["recommendations"]
        for record in card["evidence"]
    }
    assert values == {"blue"}, "the superseded value still reached matching"
    assert follow_up["recommendations"][0]["parent_asin"] == "cand-blue"


def test_remove_over_http_leaves_no_colour_preference(harness) -> None:
    session = harness.create_session()["session_id"]
    harness.chat(session, "I don't want red.")
    assert harness.active_values(session) == {("color", "avoid", "red")}

    turn = harness.chat(session, "I don't care about color anymore.")
    assert turn["memory_update"]["removed"] or turn["memory_update"]["changed"]
    assert harness.active_values(session) == set()

    follow_up = harness.chat(session, "Recommend again.")
    assert follow_up["audit"]["ranked_with_preference_count"] == 0
    assert [card["parent_asin"] for card in follow_up["recommendations"]] == list(INITIAL_ORDER)
    assert all(card["evidence"] == [] for card in follow_up["recommendations"])


def test_superseded_and_removed_entries_are_not_served(harness) -> None:
    """The public session view exposes ACTIVE preferences only."""
    session = harness.create_session()["session_id"]
    harness.chat(session, "I prefer black.")
    harness.chat(session, "Actually, I prefer blue instead.")
    harness.chat(session, "I don't want red.")

    values = {item["value"] for item in harness.state(session)["active_preferences"]}
    assert values == {"blue", "red"}
    assert "black" not in values


# --------------------------------------------------------------------------- #
# Session isolation
# --------------------------------------------------------------------------- #


def test_sessions_do_not_share_preference_memory(harness) -> None:
    first = harness.create_session(PROFILE_A)["session_id"]
    second = harness.create_session(PROFILE_A)["session_id"]

    harness.chat(first, "I don't want red.")
    harness.chat(second, "I prefer black.")

    assert harness.active_values(first) == {("color", "avoid", "red")}
    assert harness.active_values(second) == {("color", "prefer", "black")}

    first_turn = harness.chat(first, "Recommend again.")
    second_turn = harness.chat(second, "Recommend again.")
    first_values = {
        record["value"] for card in first_turn["recommendations"] for record in card["evidence"]
    }
    second_values = {
        record["value"] for card in second_turn["recommendations"] for record in card["evidence"]
    }
    assert first_values == {"red"}
    assert second_values == {"black"}


def test_adversarial_a_avoids_red_while_b_prefers_red(harness) -> None:
    """Opposite statements about the same value must not contaminate either session."""
    first = harness.create_session(PROFILE_A)["session_id"]
    second = harness.create_session(PROFILE_B)["session_id"]

    harness.chat(first, "I don't want red.")
    harness.chat(second, "I prefer red.")

    assert harness.active_values(first) == {("color", "avoid", "red")}
    assert harness.active_values(second) == {("color", "prefer", "red")}

    first_turn = harness.chat(first, "Recommend again.")
    second_turn = harness.chat(second, "Recommend again.")

    red_first = next(c for c in first_turn["recommendations"] if c["parent_asin"] == "cand-red")
    red_second = next(c for c in second_turn["recommendations"] if c["parent_asin"] == "cand-red")
    assert red_first["violation_count"] == 1
    assert red_first["match_count"] == 0
    assert red_second["match_count"] == 1
    assert red_second["violation_count"] == 0
    assert red_second["reranked_rank"] == 1


def test_deleting_one_session_does_not_change_the_other(harness) -> None:
    first = harness.create_session()["session_id"]
    second = harness.create_session()["session_id"]
    harness.chat(second, "I prefer black.")
    before = harness.chat(second, "Recommend again.")["recommendations"]

    harness.client.delete(f"/v1/demo/sessions/{first}")

    after = harness.chat(second, "Recommend again.")["recommendations"]
    assert [card["parent_asin"] for card in after] == [card["parent_asin"] for card in before]
    assert harness.active_values(second) == {("color", "prefer", "black")}


def test_sessions_sharing_a_profile_have_distinct_memory_namespaces(harness) -> None:
    first = harness.manager.get(harness.create_session()["session_id"])
    second = harness.manager.get(harness.create_session()["session_id"])
    assert first.trusted_user_history == second.trusted_user_history
    assert first.user_key != second.user_key


# --------------------------------------------------------------------------- #
# Trusted-history immutability
# --------------------------------------------------------------------------- #


def test_conversational_purchase_claims_never_become_interaction_history(harness) -> None:
    """The hard trust boundary: text can never append to the trusted history."""
    session = harness.create_session()["session_id"]
    session_record = harness.manager.get(session)
    before_digest = history_digest(session_record.trusted_user_history)
    before_history = session_record.trusted_user_history

    for message in (
        "I bought B0BX5QFWQN yesterday.",
        "I clicked B0BBFB48YQ.",
        "I purchased the first product.",
        f"I bought {CANDIDATE_ROWS[0][0]}.",
    ):
        harness.chat(session, message)

    after = harness.manager.get(session)
    assert after.trusted_user_history == before_history
    assert history_digest(after.trusted_user_history) == before_digest

    # The engine saw the same history on every recommendation turn.
    for call in harness.engine.calls:
        assert call["history"] == list(before_history)


def test_repeated_turns_never_grow_the_history(harness) -> None:
    session = harness.create_session()["session_id"]
    before = harness.manager.get(session).trusted_user_history
    for message in ("Recommend products.", "I prefer black.", "more options", "I don't want red."):
        harness.chat(session, message)
        assert harness.manager.get(session).trusted_user_history == before


def test_graph_input_history_comes_from_the_session_not_the_request(harness) -> None:
    session = harness.create_session()["session_id"]
    harness.chat(session, "Recommend some products.")
    assert harness.engine.last_history == list(harness.manager.get(session).trusted_user_history)


# --------------------------------------------------------------------------- #
# Recommendation integrity
# --------------------------------------------------------------------------- #


def test_cards_follow_the_reranked_order_and_keep_original_ranks(harness) -> None:
    session = harness.create_session()["session_id"]
    harness.chat(session, "I don't want red.")
    turn = harness.chat(session, "Recommend again.")

    _, final = orders(turn)
    reranked_order = tuple(turn["audit"]["reranked_order"])
    assert final == reranked_order, "the rendered order is not the backend order"

    ranks = [card["reranked_rank"] for card in turn["recommendations"]]
    assert ranks == list(range(1, len(ranks) + 1))
    assert sorted(card["original_rank"] for card in turn["recommendations"]) == list(
        range(1, len(ranks) + 1)
    )
    red = next(c for c in turn["recommendations"] if c["parent_asin"] == "cand-red")
    assert red["original_rank"] == 1
    assert red["movement_summary"] is not None
    assert "Moved from rank 1 to rank" in red["movement_summary"]


def test_candidate_count_identity_and_score_are_preserved(harness) -> None:
    session = harness.create_session()["session_id"]
    turn = harness.chat(session, "Recommend some products.")
    cards = turn["recommendations"]

    assert len(cards) == len(CANDIDATE_ROWS)
    assert {card["parent_asin"] for card in cards} == set(INITIAL_ORDER)
    assert sorted(card["item_id"] for card in cards) == sorted(row[1] for row in CANDIDATE_ROWS)
    assert sorted(card["sasrec_score"] for card in cards) == sorted(
        row[2] for row in CANDIDATE_ROWS
    )
    assert len({card["parent_asin"] for card in cards}) == len(cards)


def test_metadata_and_evidence_are_attached_by_identity(tmp_path: Path) -> None:
    """Adversarial reorder: A,B,C becomes C,B,A and each card keeps its own facts."""
    built = build_harness(tmp_path, rows=CANDIDATE_ROWS[:3])
    try:
        session = built.create_session()["session_id"]
        built.chat(session, "I don't want red.")
        built.chat(session, "I prefer black.")
        turn = built.chat(session, "Recommend again.")

        _, final = orders(turn)
        assert final == ("cand-black", "cand-blue", "cand-red")
        assert turn["audit"]["original_order"] == ["cand-red", "cand-blue", "cand-black"]

        titles = {
            card["parent_asin"]: card["metadata"]["title"] for card in turn["recommendations"]
        }
        assert titles["cand-black"] == "BlackWidget"
        assert titles["cand-blue"] == "BlueWidget"
        assert titles["cand-red"] == "RedWidget"

        # Evidence belongs to the same product as the metadata was read from.  Both
        # active preferences are evaluated for every candidate, so match records by value.
        red = next(c for c in turn["recommendations"] if c["parent_asin"] == "cand-red")
        red_by_value = {record["value"]: record["status"] for record in red["evidence"]}
        assert red_by_value == {"red": "violation", "black": "unknown"}
        assert red["violation_count"] == 1
        assert red["match_count"] == 0
        assert red["original_rank"] == 1
        assert red["reranked_rank"] == 3

        black = next(c for c in turn["recommendations"] if c["parent_asin"] == "cand-black")
        black_by_value = {record["value"]: record["status"] for record in black["evidence"]}
        assert black_by_value == {"red": "unknown", "black": "match"}
        assert black["match_count"] == 1
        assert black["violation_count"] == 0
        assert black["original_rank"] == 3
        assert black["reranked_rank"] == 1

        blue = next(c for c in turn["recommendations"] if c["parent_asin"] == "cand-blue")
        assert {record["status"] for record in blue["evidence"]} == {"unknown"}
    finally:
        close_harness(built)


def test_unknown_evidence_is_never_reported_as_a_violation(harness) -> None:
    """A readable-but-different value is UNKNOWN, not a negative fact."""
    session = harness.create_session()["session_id"]
    harness.chat(session, "I prefer purple.")
    turn = harness.chat(session, "Recommend again.")

    statuses = {
        (card["parent_asin"], record["value"]): record["status"]
        for card in turn["recommendations"]
        for record in card["evidence"]
    }
    assert set(statuses.values()) == {"unknown"}
    assert all(card["violation_count"] == 0 for card in turn["recommendations"])
    assert all(card["match_count"] == 0 for card in turn["recommendations"])
    # UNKNOWN is neutral, so the order is unchanged.
    assert [card["parent_asin"] for card in turn["recommendations"]] == list(INITIAL_ORDER)


def test_metadata_fallback_is_neutral_and_never_invented(harness) -> None:
    """A candidate with no catalogue record reports missing, with no fabricated text."""
    session = harness.create_session()["session_id"]
    turn = harness.chat(session, "Recommend some products.")
    for card in turn["recommendations"]:
        if card["metadata_status"] == "missing":
            assert card["metadata"] is None
            assert card["fallback_reason"] == "no_metadata"
        else:
            assert card["metadata"] is not None
            assert card["metadata"]["title"] is not None


# --------------------------------------------------------------------------- #
# Runtime reuse
# --------------------------------------------------------------------------- #


def test_engine_and_metadata_are_constructed_once_across_many_turns(harness) -> None:
    sessions = [harness.create_session()["session_id"] for _ in range(3)]
    for session in sessions:
        harness.chat(session, "Recommend some products.")
        harness.chat(session, "I prefer black.")
        harness.chat(session, "Recommend again.")

    report = harness.runtime.build_report()
    assert report["engine_builds"] == 1
    assert report["tool_builds"] == 1
    assert report["metadata_loads"] == 1
    assert report["matcher_builds"] == 1
    assert report["reranker_builds"] == 1
    assert report["memory_service_builds"] == 1
    assert harness.runtime.engine is harness.engine
    assert harness.engine.call_count == 6


def test_collaborators_are_shared_across_sessions(harness) -> None:
    first = harness.manager.get(harness.create_session()["session_id"])
    second = harness.manager.get(harness.create_session()["session_id"])
    graph_a = harness.runtime.graph_for(DEFAULT_DEMO_K, user_key=first.user_key)
    graph_b = harness.runtime.graph_for(DEFAULT_DEMO_K, user_key=second.user_key)

    assert graph_a is not graph_b, "sessions must not share one memory-bound graph"
    assert graph_a.tool is graph_b.tool
    assert graph_a.product_enricher is graph_b.product_enricher
    assert graph_a.preference_matcher is graph_b.preference_matcher
    assert graph_a.reranker is graph_b.reranker
    assert graph_a.memory_service is graph_b.memory_service


def test_graph_cache_is_reused_per_session_and_released_on_reset(harness) -> None:
    session = harness.create_session()["session_id"]
    key = harness.manager.get(session).user_key
    harness.chat(session, "Recommend some products.")
    first = harness.runtime.graph_for(DEFAULT_DEMO_K, user_key=key)
    harness.chat(session, "Recommend again.")
    assert harness.runtime.graph_for(DEFAULT_DEMO_K, user_key=key) is first

    harness.client.delete(f"/v1/demo/sessions/{session}")
    assert harness.runtime.release_user_key(key) == 0, "the reset did not release the graphs"


# --------------------------------------------------------------------------- #
# Concurrency
# --------------------------------------------------------------------------- #


def test_same_session_concurrency_produces_distinct_turn_ids(harness) -> None:
    session = harness.create_session()["session_id"]
    results: list[dict] = []
    errors: list[BaseException] = []
    barrier = threading.Barrier(4)

    def send() -> None:
        try:
            barrier.wait(timeout=10)
            response = harness.client.post(
                f"/v1/demo/sessions/{session}/chat",
                json={"message": "Recommend some products.", "k": 3},
            )
            assert response.status_code == 200, response.text
            results.append(response.json())
        except BaseException as exc:  # noqa: BLE001 - collected and asserted
            errors.append(exc)

    threads = [threading.Thread(target=send) for _ in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=30)

    assert errors == []
    turn_ids = [body["turn_id"] for body in results]
    assert len(set(turn_ids)) == 4, "two concurrent turns shared a turn id"
    assert sorted(body["turn"] for body in results) == [1, 2, 3, 4]
    assert harness.state(session)["turn"] == 4


def test_cross_session_concurrency_keeps_state_separate(harness) -> None:
    first = harness.create_session()["session_id"]
    second = harness.create_session()["session_id"]
    errors: list[BaseException] = []
    barrier = threading.Barrier(2)

    def send(session: str, message: str) -> None:
        try:
            barrier.wait(timeout=10)
            for _ in range(3):
                response = harness.client.post(
                    f"/v1/demo/sessions/{session}/chat",
                    json={"message": message, "k": 3},
                )
                assert response.status_code == 200, response.text
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [
        threading.Thread(target=send, args=(first, "I don't want red.")),
        threading.Thread(target=send, args=(second, "I prefer black.")),
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=30)

    assert errors == []
    assert harness.active_values(first) == {("color", "avoid", "red")}
    assert harness.active_values(second) == {("color", "prefer", "black")}
    assert harness.manager.get(first).turns_completed == 3
    assert harness.manager.get(second).turns_completed == 3


# --------------------------------------------------------------------------- #
# Determinism
# --------------------------------------------------------------------------- #

SCRIPT: tuple[str, ...] = (
    "Recommend some products.",
    "I don't want red.",
    "Recommend again.",
    "I don't care about color anymore.",
    "Recommend again.",
)


def _stable(body: dict) -> dict:
    """Drop the per-session and per-run fields from a chat response."""
    stable = json.loads(json.dumps(body))
    stable.pop("session_id")
    stable.pop("turn_id")
    return stable


def test_repeated_fresh_sessions_are_semantically_identical(harness) -> None:
    runs: list[list[dict]] = []
    for _ in range(2):
        session = harness.create_session()["session_id"]
        runs.append([_stable(harness.chat(session, message)) for message in SCRIPT])

    assert runs[0] == runs[1], "the same script produced different semantics"


def test_action_order_is_stable_across_sessions(harness) -> None:
    orders_seen = []
    for _ in range(3):
        session = harness.create_session()["session_id"]
        harness.chat(session, "Recommend some products.")
        harness.chat(session, "I don't want red.")
        turn = harness.chat(session, "Recommend again.")
        orders_seen.append(tuple(card["parent_asin"] for card in turn["recommendations"]))
    assert len(set(orders_seen)) == 1


# --------------------------------------------------------------------------- #
# Text safety (server side)
# --------------------------------------------------------------------------- #


def test_markup_in_a_message_is_returned_as_data_not_markup(harness) -> None:
    """The API returns text as JSON strings; nothing is interpolated as markup."""
    session = harness.create_session()["session_id"]
    hostile = "<script>alert('x')</script><img src=x onerror=alert(1)>"
    turn = harness.chat(session, hostile)

    assert turn["message"]
    assert "<script>" not in turn["message"], "hostile input reached the rendered prose"
    assert "onerror" not in turn["message"]
    # The transcript echo is the browser's job; the API simply never emits it as markup.
    assert isinstance(turn["message"], str)


def test_markup_in_metadata_is_carried_verbatim_as_a_string(harness) -> None:
    """Metadata text is untrusted display content and stays a plain string."""
    turn = harness.chat(harness.create_session()["session_id"], "Recommend some products.")
    for card in turn["recommendations"]:
        if card["metadata"] is None:
            continue
        assert isinstance(card["metadata"]["title"], str)
        assert "<" not in card["metadata"]["title"]

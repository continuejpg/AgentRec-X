"""E-02C-NARRATION-FIX: the recommendation prose must describe the authoritative cards.

The defect this file pins down: an ``llm_agent`` recommendation turn used to answer with the
controller's *own* narration, which is rendered inside the run - before the multi-source
accumulation, the memory-aware ranking and the ``k`` limit - so the prose could list products the
response does not show, in an order the ranking had already changed.

Every assertion here is made against a real Chat API response: the prose's numbered products are
parsed and compared, by stable identity (``parent_asin`` + ``item_id``), with the response's own
``recommendations``.  Nothing is asserted about the wording beyond the structure the renderer
guarantees, and every original card field is checked to be untouched.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pytest

from recommendation.demo.agent_chat_adapter import NO_VERIFIED_REASON
from tests.demo_fixture import close_harness
from tests.test_v2_phase4_joint_acceptance import (  # the E-02C joint fixture
    BOTH,
    DIRECT,
    FUSED_BASELINE,
    REQUEST,
    _cards,
    _harness,
    _ids,
    _turn,
)

#: ``N. <parent_asin> (item <item_id>) - ...`` - the structure the renderer guarantees.
_LINE = re.compile(r"^(\d+)\. (\S+) \(item (\d+)\)", re.MULTILINE)

#: Phrases the accepted controller narration contains and the card-derived prose must not reuse.
_OLD_NARRATION_MARKERS = (
    "from the sequential recommender",
    "Original SASRec ranks are shown",
    "Facts above are quoted from Amazon Reviews 2023",
)

PROVIDER_ENV_KEYS = (
    "AGENTRECX_LLM_BASE_URL",
    "AGENTRECX_LLM_MODEL",
    "AGENTRECX_LLM_API_KEY",
    "AGENTRECX_LLM_PROFILE",
)


@pytest.fixture()
def configured(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in PROVIDER_ENV_KEYS:
        monkeypatch.setenv(name, "unit-test-placeholder-not-a-credential")
    monkeypatch.setenv("AGENTRECX_LLM_PROFILE", "deepseek")


def narrated(message: str) -> list[tuple[int, str, int]]:
    """The prose's own product list, as ``(position, parent_asin, item_id)``."""
    return [(int(n), asin, int(item)) for n, asin, item in _LINE.findall(message)]


def card_identities(body: dict[str, Any]) -> list[tuple[int, str, int]]:
    return [
        (position, card["parent_asin"], card["item_id"])
        for position, card in enumerate(body["recommendations"], start=1)
    ]


def assert_narration_matches_cards(body: dict[str, Any]) -> list[tuple[int, str, int]]:
    """The contract: same products, same order, same count - nothing more, nothing less."""
    listed = narrated(body["message"])
    assert listed == card_identities(body), (
        f"the prose must describe the cards exactly; prose={listed} cards="
        f"{card_identities(body)}"
    )
    assert len(listed) == len(body["recommendations"])
    assert [position for position, _, _ in listed] == list(
        range(1, len(body["recommendations"]) + 1)
    )
    return listed


# --------------------------------------------------------------------------- #
# A. Memory reordering
# --------------------------------------------------------------------------- #


def test_a_memory_reorder_is_what_the_prose_reports(tmp_path: Path, configured: None) -> None:
    """Three products: the fused order leads with red, ``prefer black`` must lead with black."""
    harness = _harness(tmp_path)
    try:
        session = harness.create_session()["session_id"]
        baseline, _c0 = _turn(harness, session, REQUEST)
        assert _ids(baseline) == FUSED_BASELINE == ["j-red", "j-black", "j-blue", "j-plain"]
        assert_narration_matches_cards(baseline)
        assert narrated(baseline["message"])[0][1] == "j-red", "the fused order leads"

        _turn(harness, session, "I prefer black.", plan=DIRECT)
        ranked, _c1 = _turn(harness, session, REQUEST)
        assert _ids(ranked) == ["j-black", "j-red", "j-blue", "j-plain"]
        listed = assert_narration_matches_cards(ranked)
        assert listed[0] == (1, "j-black", 701), (
            "the prose follows the memory-aware order, not the fused one it replaced"
        )
        # The card fields themselves are untouched by the narration change.
        for card in ranked["recommendations"]:
            assert set(card) >= {
                "item_id", "parent_asin", "sasrec_score", "score_kind", "fusion_sources",
                "original_rank", "reranked_rank", "evidence", "knowledge_evidence", "reason",
            }
        black = _cards(ranked)["j-black"]
        assert (black["original_rank"], black["reranked_rank"]) == (2, 1)
        assert black["sasrec_score"] == pytest.approx(1.137496)
        assert black["score_kind"] == "bm25"
        assert black["fusion_sources"] == ["catalog_search"]
    finally:
        close_harness(harness)


# --------------------------------------------------------------------------- #
# B. Multi-source fusion
# --------------------------------------------------------------------------- #


def test_the_prose_lists_every_fused_product_and_nothing_else(
    tmp_path: Path, configured: None
) -> None:
    harness = _harness(tmp_path)
    try:
        session = harness.create_session()["session_id"]
        body, _client = _turn(harness, session, REQUEST)
        # Both sources really ran, and the fused pool is the four grounded products.
        assert body["agent"]["action_counts"] == {
            "recommend_from_history": 1, "search_catalog": 1, "finish": 1}
        assert body["agent"]["grounded"]["grounded_count"] == 4
        listed = assert_narration_matches_cards(body)
        assert {asin for _, asin, _ in listed} == set(FUSED_BASELINE), (
            "no legally grounded product may be left out of the prose"
        )
        # The overlap product appears once, with its ledger provenance intact.
        assert [asin for _, asin, _ in listed].count("j-red") == 1
        assert _cards(body)["j-red"]["fusion_sources"] == ["catalog_search"]
    finally:
        close_harness(harness)


# --------------------------------------------------------------------------- #
# C. k truncation
# --------------------------------------------------------------------------- #


def test_a_candidate_cut_by_k_is_never_narrated(tmp_path: Path, configured: None) -> None:
    harness = _harness(tmp_path)
    try:
        session = harness.create_session()["session_id"]
        body, _client = _turn(harness, session, REQUEST, k=2)

        # Three products were retrieved and grounded; only two are presented.
        assert body["agent"]["grounded"]["grounded_count"] == 3
        assert _ids(body) == ["j-red", "j-black"]
        assert_narration_matches_cards(body)

        # The cut product is absent by identity, its item id included - not merely by luck of
        # wording, and never as a bare number either.
        assert "j-blue" not in body["message"]
        assert "702" not in body["message"], "the cut product's item id must not appear"
        assert "j-plain" not in body["message"]
        assert [asin for _, asin, _ in narrated(body["message"])] == ["j-red", "j-black"]
    finally:
        close_harness(harness)


# --------------------------------------------------------------------------- #
# D. ADD and REMOVE keep the prose in step
# --------------------------------------------------------------------------- #


def test_add_and_remove_move_the_prose_with_the_cards(
    tmp_path: Path, configured: None
) -> None:
    harness = _harness(tmp_path)
    try:
        session = harness.create_session()["session_id"]
        baseline, _c0 = _turn(harness, session, REQUEST)
        baseline_list = assert_narration_matches_cards(baseline)

        _turn(harness, session, "I prefer black.", plan=DIRECT)
        added, _c1 = _turn(harness, session, REQUEST)
        added_list = assert_narration_matches_cards(added)
        assert added_list != baseline_list, "ADD changes the order, so the prose changes too"
        assert added_list[0][1] == "j-black"

        _turn(harness, session, "I don't care about color anymore.", plan=DIRECT)
        restored, _c2 = _turn(harness, session, REQUEST)
        restored_list = assert_narration_matches_cards(restored)
        assert restored_list == baseline_list, "REMOVE restores the baseline order in the prose"
        assert restored["audit"]["reranking_applied"] is False
    finally:
        close_harness(harness)


# --------------------------------------------------------------------------- #
# E. RAG success, failure and absent evidence
# --------------------------------------------------------------------------- #


def test_a_validated_reason_is_quoted_and_a_rejected_one_is_not(
    tmp_path: Path, configured: None
) -> None:
    harness = _harness(tmp_path)
    try:
        session = harness.create_session()["session_id"]

        # Success: the validated reason for the promoted product appears verbatim in the prose.
        good, _c1 = _turn(harness, session, REQUEST)
        assert_narration_matches_cards(good)
        black = _cards(good)["j-black"]
        if black["reason_status"] == "grounded":
            assert black["reason"] in good["message"]
        # A product the explanation step could not support has no reason anywhere.
        assert _cards(good)["j-blue"]["reason"] is None
        assert "j-blue" in good["message"], "it is still presented"
        assert NO_VERIFIED_REASON in good["message"]
        assert "Recommended because" not in good["message"].split("3. j-blue")[1].split("\n")[0]

        # A lying model: the citations are refused, so no product fact enters the prose.
        bad, _c2 = _turn(harness, session, REQUEST, mode="hallucinate")
        assert_narration_matches_cards(bad)
        assert all(card["reason"] is None for card in bad["recommendations"])
        assert bad["message"].count(NO_VERIFIED_REASON) == len(bad["recommendations"])
        for invented in ("GhostWidget", "j-ghost", "weight", "material", "premium"):
            assert invented not in bad["message"], invented
        # A timeout cannot change the prose either.
        timed_out, _c3 = _turn(harness, session, REQUEST, mode="timeout")
        assert_narration_matches_cards(timed_out)
        assert timed_out["message"].count(NO_VERIFIED_REASON) == len(
            timed_out["recommendations"]
        )
    finally:
        close_harness(harness)


# --------------------------------------------------------------------------- #
# F. the four switch combinations
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(("ranking", "rag"), [
    (False, False), (True, False), (False, True), (True, True),
])
def test_the_prose_matches_the_cards_in_every_flag_combination(
    tmp_path: Path, configured: None, ranking: bool, rag: bool
) -> None:
    harness = _harness(tmp_path, explanations=rag, memory_ranking=ranking)
    try:
        session = harness.create_session()["session_id"]
        _turn(harness, session, "I prefer black.", plan=DIRECT)
        body, _client = _turn(harness, session, REQUEST)

        assert set(_ids(body)) == set(FUSED_BASELINE)
        listed = assert_narration_matches_cards(body)
        assert listed[0][1] == ("j-black" if ranking else "j-red")
        # With the explanation step off, nothing re-authorises the old narration's product facts.
        if not rag:
            for marker in _OLD_NARRATION_MARKERS:
                assert marker not in body["message"], marker
            assert "Recommended because" not in body["message"]
            assert body["message"].count(NO_VERIFIED_REASON) == len(body["recommendations"])
        # The published card fields never depend on either switch.
        black = _cards(body)["j-black"]
        assert (black["sasrec_score"], black["score_kind"], black["fusion_sources"]) == (
            pytest.approx(1.137496), "bm25", ["catalog_search"])
        assert black["match_count"] == (1 if ranking else 0)
    finally:
        close_harness(harness)


# --------------------------------------------------------------------------- #
# G. direct, refused and failed turns keep their own words
# --------------------------------------------------------------------------- #


def test_non_recommendation_turns_keep_their_own_message(
    tmp_path: Path, configured: None
) -> None:
    from recommendation.control.model_client import ScriptedModelClient

    harness = _harness(tmp_path)
    try:
        # direct: the accepted direct capability answers, and no product list is invented.
        session = harness.create_session()["session_id"]
        direct, _c = _turn(harness, session, "hello there", plan=DIRECT)
        assert direct["recommendations"] == []
        assert direct["agent"]["outcome"] == "direct"
        assert not direct["message"].startswith("Top ")
        assert direct["message"].strip()

        # no candidates / refused: the run's own text, never a fabricated product list.
        refused, _c2 = _turn(
            harness, harness.create_session()["session_id"], "zzzqqq nothing here",
            plan={"route": "recommend", "selected_sources": ["catalog_search"], "reason": "x"},
        )
        assert refused["recommendations"] == []
        assert refused["agent"]["outcome"] == "refused"
        assert not refused["message"].startswith("Top ")
        assert refused["message"].strip()

        # failed: a tool that raises leaves the run refused with its own words.
        harness.engine.error = RuntimeError("engine exploded")
        failed, _c3 = _turn(
            harness, harness.create_session()["session_id"], "find gear",
            plan={"route": "recommend", "selected_sources": ["history"], "reason": "x"},
        )
        assert failed["recommendations"] == []
        assert failed["agent"]["outcome"] == "refused"
        assert not failed["message"].startswith("Top ")
        assert failed["message"].strip()
        assert failed["audit"]["candidate_count"] == 0
    finally:
        close_harness(harness)


# --------------------------------------------------------------------------- #
# H. the legacy paths are untouched
# --------------------------------------------------------------------------- #


def test_other_modes_keep_the_accepted_narration(tmp_path: Path, configured: None) -> None:
    """The narration fix is scoped to the new mode's recommendation projection.

    A request the deterministic policy does not treat as a recommendation ("RedWidget BlackWidget")
    answers directly, so the recommendation-asking request is used here deliberately.
    """
    from recommendation.control.model_client import ScriptedModelClient

    harness = _harness(tmp_path)
    try:
        asking = "Recommend something."

        # deterministic chat: the accepted graph narration, unchanged by this fix.
        deterministic = harness.client.post(
            f"/v1/demo/sessions/{harness.create_session()['session_id']}/chat",
            json={"message": asking, "k": 4})
        assert deterministic.status_code == 200, deterministic.text
        deterministic_body = deterministic.json()
        assert deterministic_body["recommendations"], "this request really recommends"
        assert any(marker in deterministic_body["message"] for marker in _OLD_NARRATION_MARKERS), (
            "the accepted rendering is what the old modes still answer with"
        )

        # llm chat: the same accepted renderer.
        client = ScriptedModelClient(
            [{"route": "recommend", "add": [], "remove": []}], repeat_last=True)
        client.profile = type("P", (), {"name": "deepseek"})()
        client.model = "deepseek-chat"
        harness.runtime.use_llm_client(client)
        llm = harness.client.post(
            f"/v1/demo/sessions/{harness.create_session()['session_id']}/chat",
            json={"message": asking, "k": 4, "decision_mode": "llm"})
        assert llm.status_code == 200, llm.text
        assert llm.json()["recommendations"]
        assert any(marker in llm.json()["message"] for marker in _OLD_NARRATION_MARKERS)

        # legacy Agent Recommend: returns the agent response itself, no chat projection at all.
        session = harness.create_session()["session_id"]
        agent_response = harness.client.post(
            "/v1/demo/agent/recommend",
            json={"message": asking, "k": 4, "sources": ["history"],
                  "history": list(harness.manager.get(session).trusted_user_history)},
        )
        assert agent_response.status_code == 200, agent_response.text
        body = agent_response.json()
        assert body["recommendations"]
        assert any(marker in body["message"] for marker in _OLD_NARRATION_MARKERS)
        assert all("reason" not in card for card in body["recommendations"])
    finally:
        close_harness(harness)


def test_the_prose_is_reproducible_and_free_of_old_order_claims(
    tmp_path: Path, configured: None
) -> None:
    """Two identical turns answer with identical prose, and it claims no rank the run had."""
    harness = _harness(tmp_path)
    try:
        first_session = harness.create_session()["session_id"]
        second_session = harness.create_session()["session_id"]
        first, _c1 = _turn(harness, first_session, REQUEST)
        second, _c2 = _turn(harness, second_session, REQUEST)
        assert first["message"] == second["message"], "same inputs, same prose"
        assert_narration_matches_cards(first)
        # Per-line facts are the cards' own published fields.
        for card in first["recommendations"]:
            assert f"ranking score {card['sasrec_score']} ({card['score_kind']})" in first["message"]
            sources = ", ".join(card["fusion_sources"]) or "none recorded"
            assert f"sources: {sources};" in first["message"]
        assert json.dumps(first["message"]).count("original SASRec rank") == 0
    finally:
        close_harness(harness)

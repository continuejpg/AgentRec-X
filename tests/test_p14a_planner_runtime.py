"""P14-A Planner bridge + end-to-end runtime tests on a fixture catalogue.

These tests exercise the *real* bridge code path (existing Planner Action contract ->
executor -> real search/info/ledger/eligibility) with the ``contract_only`` label.
They are not benchmark results and never read the evaluator-only ground truth.
"""

from __future__ import annotations

import pytest

from recommendation.shopping_companion.catalogue import ShoppingCompanionCatalogue
from recommendation.shopping_companion.product_info import ShoppingCompanionProductInfoTool
from recommendation.shopping_companion.runtime import (
    EXECUTION_MODES,
    BlockedMemoryTool,
    RunConfig,
    RuntimeModeError,
    ShoppingCompanionRuntime,
    TaskInput,
    compose_product_query,
    derive_constraints,
)
from recommendation.shopping_companion.search import ShoppingCompanionSearchTool

DOCS = [
    {
        "id": "1001",
        "contents": "Acme Wireless Mouse\nbrand: Acme\ncolor: Black",
        "product": {
            "product_id": "1001",
            "product_name": "Acme Wireless Mouse",
            "seller_id": "s1",
            "seller_name": "Acme Store",
            "category": "Electronics - Computer Accessories - Mice",
            "price": 15.5,
            "attributes": {"brand": ["Acme"], "color": ["Black"]},
            "options": [],
            "description": "A compact wireless mouse.",
        },
    },
    {
        "id": "1002",
        "contents": "Beta Wireless Mouse Pro\nbrand: Beta",
        "product": {
            "product_id": "1002",
            "product_name": "Beta Wireless Mouse Pro",
            "seller_id": "s2",
            "seller_name": "Beta Shop",
            "category": "Electronics - Computer Accessories - Mice",
            "price": 42.0,
            "attributes": {"brand": ["Beta"]},
            "options": [],
            "description": "A premium wireless mouse.",
        },
    },
    {
        "id": "1003",
        "contents": "Gamma Wired Mouse\nbrand: Gamma",
        "product": {
            "product_id": "1003",
            "product_name": "Gamma Wired Mouse",
            "seller_id": "s3",
            "seller_name": "Gamma Store",
            "category": "Electronics - Computer Accessories - Mice",
            "price": 9.0,
            "attributes": {"brand": ["Gamma"]},
            "options": [],
            "description": "A simple wired mouse.",
        },
    },
]


class FakeMemoryTool:
    """Synthetic-fixture memory used only in contract tests (never a benchmark claim)."""

    def __init__(self, preferences, evidence=None) -> None:
        self._preferences = list(preferences)
        self._evidence = list(evidence or [])

    def search(self, query: str, *, scope: str = "long_term", k: int = 10):
        return {
            "available": bool(self._preferences),
            "status": "synthetic_fixture",
            "preferences": self._preferences,
            "evidence": self._evidence,
            "conflicts": [],
            "reason": "synthetic fixture (contract_only test)",
        }

    def history_status(self):
        return {"available": bool(self._preferences), "status": "synthetic_fixture"}


class OracleLeakingMemoryTool(FakeMemoryTool):
    def search(self, query: str, *, scope: str = "long_term", k: int = 10):
        return {
            "available": True,
            "status": "oracle_diagnostic",
            "preferences": [{"kind": "brand", "polarity": "positive", "value": "Acme"}],
            "evidence": [
                {
                    "conversation_id": "c",
                    "turn_index": 0,
                    "date": "2025/01/01 (Wed)",
                    "text": "I love Acme",
                    "source_kind": "released_oracle_diagnostic",
                }
            ],
            "conflicts": [],
            "reason": "oracle",
        }


def _catalogue() -> ShoppingCompanionCatalogue:
    return ShoppingCompanionCatalogue.from_docs(
        DOCS, manifest={"selection_method": "LIMITED catalogue; contract fixture"}
    )


def _runtime(
    memory,
    *,
    mode: str = "contract_only",
    diagnostic_stage2: bool = False,
) -> ShoppingCompanionRuntime:
    catalogue = _catalogue()
    return ShoppingCompanionRuntime(
        catalogue=catalogue,
        search_tool=ShoppingCompanionSearchTool(catalogue),
        info_tool=ShoppingCompanionProductInfoTool(catalogue),
        memory=memory,
        config=RunConfig(
            mode=mode,
            search_limit=3,
            inspect_top_k=3,
            stage2_diagnostic_without_preferences=diagnostic_stage2,
        ),
    )


def _task(question: str = "Current Date: 2026/01/01 (Thu)\nwireless mouse under $20") -> TaskInput:
    return TaskInput(
        task_id="fixture_task_1",
        conversation_id="fixture_conversation",
        question=question,
        question_date="2026/01/01",
    )


def test_derive_constraints_is_conservative_and_deterministic() -> None:
    assert derive_constraints("wireless mouse under $20")[0].kind.value == "price_max"
    assert derive_constraints("wireless mouse under $20")[0].expected == "20"
    assert derive_constraints("a budget of 35 for a kettle")[0].expected == "35"
    assert derive_constraints("find me a nice kettle") == ()
    assert derive_constraints("under $50 and under $30")[0].expected == "30"  # most restrictive wins


def test_compose_query_uses_only_recovered_preferences() -> None:
    question = "Current Date: 2026/01/01 (Thu)\nwireless mouse"
    assert compose_product_query(question, []) == "wireless mouse"
    query = compose_product_query(
        question,
        [{"kind": "brand", "polarity": "positive", "value": "Acme"},
         {"kind": "color", "polarity": "negative", "value": "red"}],
    )
    assert "Acme" in query
    assert "red" not in query


def test_invalid_mode_is_rejected() -> None:
    with pytest.raises(RuntimeModeError):
        RunConfig(mode="totally_made_up")
    assert "real_limited_catalogue" in EXECUTION_MODES


def test_blocked_history_stops_at_awaiting_clarification() -> None:
    """An unresolved stage_1 ASK must NOT be silently continued as a task success."""
    trajectory = _runtime(BlockedMemoryTool()).run_task(_task())
    stage_1 = trajectory["stages"]["stage_1"]
    assert stage_1["result"]["status"] == "history_unavailable"
    assert trajectory["memory"]["status"]["status"] == "history_unavailable"
    assert trajectory["memory"]["preferences"] == []
    assert trajectory["awaiting_clarification"] is True
    assert trajectory["stage_2_executed"] is False
    assert trajectory["terminal"]["status"] == "awaiting_clarification"
    assert trajectory["terminal"]["succeeded"] is False
    assert trajectory["stage_2_diagnostic_only"] is False
    # no product tool ran and no recommendation was claimed
    tools = [call["tool"] for call in trajectory["tool_calls"]]
    assert "memory.search" in tools and "ask" in tools
    assert "product_search" not in tools
    assert trajectory["recommendation"] == []
    assert any("awaiting user clarification" in note for note in trajectory["limitations"])


def test_diagnostic_stage2_runs_real_tools_but_is_not_a_task_success() -> None:
    """The opt-in diagnostic path may exercise stage_2, clearly labelled and single-product."""
    trajectory = _runtime(BlockedMemoryTool(), diagnostic_stage2=True).run_task(_task())
    assert trajectory["awaiting_clarification"] is True
    assert trajectory["stage_2_executed"] is True
    assert trajectory["stage_2_diagnostic_only"] is True
    assert trajectory["terminal"]["status"] == "awaiting_clarification"
    assert trajectory["terminal"]["succeeded"] is False
    tools = [call["tool"] for call in trajectory["tool_calls"]]
    assert "product_search" in tools and "product_view" in tools
    # real ledger entries with provenance
    entries = trajectory["ledger"]["entries"]
    assert entries
    assert all(entry["parent_asin"].startswith("shopping_companion:") for entry in entries)
    for entry in entries:
        for provenance in entry["provenance"]:
            assert provenance["score_kind"] == "bm25"
            assert provenance["source"] == "catalog_search"
    # price constraint was really evaluated and the violating candidate excluded
    eligibility = trajectory["stages"]["stage_2"]["result"]["eligibility"]
    assert eligibility["requirements"][0]["kind"] == "price_max"
    assert eligibility["ineligible_count"] >= 1
    excluded = {item["identity"] for item in eligibility["excluded"]}
    assert "shopping_companion:1002" in excluded
    # single-product task: exactly ONE product is presented, and never an excluded one
    assert len(trajectory["recommendation"]) == 1
    presented = {item["namespaced_id"] for item in trajectory["recommendation"]}
    assert not (presented & excluded)
    assert trajectory["mode"] == "contract_only"
    actions = [step["action"] for step in trajectory["stages"]["stage_2"]["planner_run"]["steps"]]
    assert actions == ["RETRIEVE", "RERANK", "FINALIZE"]


def test_stage_2_query_uses_recovered_preferences_when_memory_available() -> None:
    memory = FakeMemoryTool(
        [{"kind": "brand", "polarity": "positive", "value": "Acme", "status": "active"}],
        [{"conversation_id": "c1", "turn_index": 0, "date": "2025/01/01", "text": "I like Acme",
          "source_kind": "synthetic_fixture"}],
    )
    trajectory = _runtime(memory).run_task(_task("Current Date: 2026/01/01 (Thu)\nwireless mouse under $20"))
    stage_1 = trajectory["stages"]["stage_1"]["result"]
    assert stage_1["status"] == "preferences_recovered"
    assert stage_1["preferences"][0]["value"] == "Acme"
    assert "Acme" in trajectory["stages"]["stage_2"]["search_query"]
    assert trajectory["awaiting_clarification"] is False
    assert trajectory["stage_2_executed"] is True
    assert trajectory["terminal"]["succeeded"] is True
    # the single-product task presents exactly one product (official task semantics)
    assert len(trajectory["recommendation"]) == 1


def test_oracle_evidence_is_refused_outside_oracle_mode() -> None:
    trajectory = _runtime(OracleLeakingMemoryTool([]), mode="contract_only").run_task(_task())
    # the guard turns the leak into an explicit tool error; the pref is not used
    assert trajectory["memory"]["preferences"] == []
    assert trajectory["terminal"]["succeeded"] is False
    tools = [call["tool"] for call in trajectory["tool_calls"]]
    assert any(str(tool).startswith("error:") for tool in tools)
    assert any(call.get("error") for call in trajectory["tool_calls"])


def test_oracle_diagnostic_mode_allows_tagged_oracle_evidence() -> None:
    trajectory = _runtime(OracleLeakingMemoryTool([]), mode="oracle_diagnostic").run_task(_task())
    assert trajectory["mode"] == "oracle_diagnostic"
    assert trajectory["stages"]["stage_1"]["result"]["status"] == "preferences_recovered"
    assert trajectory["terminal"]["succeeded"] is True


def test_add_on_task_is_a_structural_smoke_not_a_bundle_success() -> None:
    """Add-on tasks must never be reported as a completed bundle recommendation."""
    runtime = _runtime(BlockedMemoryTool())
    task = TaskInput(
        task_id="fixture_addon_1",
        conversation_id="fixture_conversation",
        question="Current Date: 2026/01/01 (Thu)\nProduct bundle includes: mouse and pad. Budget: $25",
        question_date="2026/01/01",
        question_type="add_on_deals",
    )
    trajectory = runtime.run_task(task)
    assert trajectory["task"]["question_type"] == "add_on_deals"
    assert trajectory["bundle_structural_only"] is True
    assert any("bundle" in note for note in trajectory["limitations"])
    # no bundle-level budget is silently downgraded to a per-product ceiling
    assert trajectory["constraints"] == []
    # stage_1 has no legal history, so the task awaits clarification instead of faking
    # a completed bundle recommendation
    assert trajectory["awaiting_clarification"] is True
    assert trajectory["stage_2_executed"] is False
    assert trajectory["terminal"]["status"] == "awaiting_clarification"
    assert trajectory["terminal"]["succeeded"] is False


def test_trajectory_carries_execution_provenance() -> None:
    trajectory = _runtime(BlockedMemoryTool()).run_task(_task())
    assert trajectory["schema"] == "agentrecx.p14a.trajectory.v1"
    assert trajectory["namespace"] == "shopping_companion"
    assert trajectory["catalogue"]["limited_catalogue"] is True
    assert trajectory["tool_call_count"] == len(trajectory["tool_calls"])
    assert trajectory["timing"]["total_seconds"] >= 0.0
    for call in trajectory["tool_calls"]:
        assert {"stage", "tool", "inputs", "output", "error", "latency_ms"} <= set(call)
    # the trajectory is JSON-serialisable (it is written to JSONL by the CLI)
    import json

    json.dumps(trajectory)

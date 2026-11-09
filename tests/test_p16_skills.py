"""P16 tests: two registered skills, real evidence comparison, clarification/M9 update/idempotency."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from experiments.p14c0.electronics_executor import prepare_preferences
from experiments.p14c1.search_backend import configure_catalogue
from experiments.p14e.catalogue import ElectronicsCatalogue
from recommendation.preference.history import HistoryStateReader
from recommendation.shopping_companion.memory import SOURCE_KIND_SYNTHETIC_FIXTURE, ShoppingCompanionMemory
from recommendation.skills import build_registry
from recommendation.skills.clarification import read_authorized_history
from recommendation.skills.shopping import shopping_comparison

REPO = Path(__file__).resolve().parents[1]
RUNS = REPO / "runs/p16"
DATA = REPO / "data/p14e2/benchmark"


def _jsonl(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _task(task_id: str):
    return next(row for row in _jsonl(DATA / "dev.agent.jsonl") if row["task_id"] == task_id)


def _memory(task):
    memory = ShoppingCompanionMemory()
    memory.ingest_sessions(task["task_id"], [dict(s) for s in task["history_sessions"]],
                           source_kind=SOURCE_KIND_SYNTHETIC_FIXTURE, source_ref=f"p16:{task['task_id']}")
    history = HistoryStateReader(memory.preference_service).read(
        task["task_id"], conversation_id=task["task_id"], current_query=task["current_query"])
    return memory, history


def test_skill_docs_and_registry_contract():
    comparison = (REPO / "skills/shopping-comparison/SKILL.md").read_text(encoding="utf-8")
    clarification = (REPO / "skills/preference-clarification/SKILL.md").read_text(encoding="utf-8")
    for text, name in ((comparison, "shopping-comparison"), (clarification, "preference-clarification")):
        assert text.startswith("```yaml")
        assert f"name: {name}" in text
        assert "description:" in text
        assert "## Output contract" in text
    registry = build_registry()
    assert registry.names() == ("preference-clarification", "shopping-comparison")
    assert "2" in registry.describe("shopping-comparison")["description"]
    assert "clarification" in registry.describe("preference-clarification")["description"].lower()


def test_comparison_uses_real_products_evidence_and_no_quality_claim():
    catalogue = ElectronicsCatalogue.load()
    configure_catalogue(catalogue)
    task = _task("p14e2_lite_0043")
    memory, history = _memory(task)
    try:
        result = build_registry().invoke("shopping-comparison", catalogue=catalogue, task=task,
                                         history=history, product_refs=["B09828X5WY", "B00EWHGDVC"])
        assert result["status"] == "comparison_ready"
        assert [product["item_id"] for product in result["products"]] == ["B09828X5WY", "B00EWHGDVC"]
        assert all(product["role_status"] == "MATCH" for product in result["products"])
        assert result["qwen_used"] is False
        assert result["claims_valid"] > 0 and result["claims_rejected"] == 0
        assert all(product["unknown_fields"] for product in result["products"])
        for product in result["products"]:
            for evidence in product["evidence_bundle"]["product_evidence"]:
                assert evidence["raw_excerpt"][evidence["span_start"]:evidence["span_end"]] == evidence["span_text"]
        assert "objective quality ranking" not in result["final_answer"].lower()
        assert "UNKNOWN" in result["final_answer"]
    finally:
        memory.close()


def test_comparison_preference_relations_do_not_false_positive_ear():
    catalogue = ElectronicsCatalogue.load()
    configure_catalogue(catalogue)
    task = _task("p14e2_lite_0045")
    memory, history = _memory(task)
    try:
        result = build_registry().invoke("shopping-comparison", catalogue=catalogue, task=task,
                                         history=history,
                                         product_refs=["B0BLSQ5T75", "B08HM3MLJF", "B0BQ38B3NC"])
        assert result["status"] == "comparison_ready"
        over_ear = result["products"][0]
        relations = {rel["value"]: rel for rel in over_ear["preference_relations"]}
        assert relations["over-ear"]["observed_in_catalogue_text"] is True
        assert relations["in-ear"]["observed_in_catalogue_text"] is False
        assert result["conditional_best_supported"] == "B0BLSQ5T75"
    finally:
        memory.close()


def test_comparison_rejects_bad_identity_and_wrong_namespace():
    catalogue = ElectronicsCatalogue.load()
    configure_catalogue(catalogue)
    task = _task("p14e2_lite_0043")
    memory, history = _memory(task)
    try:
        bad = build_registry().invoke("shopping-comparison", catalogue=catalogue, task=task,
                                      history=history, product_refs=["B09828X5WY", "B000NOTREAL"])
        assert bad["status"] == "invalid_product_reference"
        assert bad["products"] == []
        wrong_ns = build_registry().invoke("shopping-comparison", catalogue=catalogue, task=task,
                                           history=history, product_refs=["B09828X5WY", "sports:123"])
        assert wrong_ns["status"] == "invalid_product_reference"
        count = build_registry().invoke("shopping-comparison", catalogue=catalogue, task=task,
                                        history=history, product_refs=["B09828X5WY"])
        assert count["status"] == "invalid_product_count"
        role = build_registry().invoke("shopping-comparison", catalogue=catalogue, task=task,
                                       history=history, product_refs=["B09828X5WY", "B004CRRRB2"])
        assert role["status"] == "role_mismatch"
    finally:
        memory.close()


def test_clarification_ask_reply_update_retraction_and_idempotency():
    catalogue = ElectronicsCatalogue.load()
    task = _task("p14e2_lite_0057")
    memory, history = _memory(task)
    try:
        ask = build_registry().invoke("preference-clarification", operation="ask", task=task,
                                      preferences=prepare_preferences(history), gate=None)
        assert ask["status"] == "awaiting_clarification"
        assert ask["selected_product_id"] is None
        assert ask["finalize_allowed"] is False
        assert ask["requires_user_reply"] is True
        first = build_registry().invoke("preference-clarification", operation="resume", memory=memory,
                                        task=task, user_message="I prefer wireless.",
                                        turn_id="test_dup", now=1767000000.0)
        second = build_registry().invoke("preference-clarification", operation="resume", memory=memory,
                                         task=task, user_message="I prefer wireless.",
                                         turn_id="test_dup", now=1767000001.0)
        assert first["update"]["added"] >= 1
        assert second["already_processed"] is True
        assert len(second["active_preferences"]) == len(first["active_preferences"])
        with pytest.raises(ValueError):
            build_registry().invoke("preference-clarification", operation="resume", memory=memory,
                                    task=task, user_message="", turn_id="empty")
    finally:
        memory.close()


def test_withdrawal_and_cross_user_isolation():
    catalogue = ElectronicsCatalogue.load()
    task = _task("p14e2_lite_0055")
    memory, history = _memory(task)
    try:
        before = len(history["active_preferences"])
        result = build_registry().invoke("preference-clarification", operation="resume", memory=memory,
                                         task=task, user_message="Forget my wireless preference.",
                                         turn_id="test_retract", now=1767000000.0)
        assert result["update"]["removed"] >= 1
        assert any(item["status"] == "removed" for item in result["inactive_preferences"])
        assert len(result["active_preferences"]) < before
        with pytest.raises(PermissionError):
            read_authorized_history(memory=memory, user_key=task["task_id"],
                                    conversation_id="p14e2_lite_0056", current_query=task["current_query"])
    finally:
        memory.close()


def test_e2e_artifacts_and_existing_runtime_contract():
    records = [json.loads(path.read_text(encoding="utf-8")) for path in sorted((RUNS / "e2e").glob("*.json"))]
    assert len(records) == 12
    comparison = [row for row in records if row["skill"] == "shopping-comparison"]
    clarification = [row for row in records if row["skill"] == "preference-clarification"]
    assert len(comparison) == 6 and len(clarification) == 6
    successful = [row for row in comparison if row["result"]["status"] == "comparison_ready"]
    assert len(successful) == 4
    assert all(row["result"]["qwen_used"] is False for row in successful)
    assert all(row["result"]["claims_rejected"] == 0 for row in successful)
    assert {row["case_id"] for row in comparison} >= {"cmp_invalid_id", "cmp_namespace_mismatch"}
    assert {row["case_id"] for row in clarification} >= {
        "clr_unanswered", "clr_duplicate_reply", "clr_cross_user", "clr_withdrawal"}
    assert all(row["ask_result"]["finalize_allowed"] is False for row in clarification
               if "ask_result" in row and row["ask_result"].get("status") == "awaiting_clarification")
    summary = json.loads((RUNS / "summary.json").read_text(encoding="utf-8"))
    assert summary["skill_registry"] == ["preference-clarification", "shopping-comparison"]
    assert summary["search_backend"]["mode"] == "bge"
    reserved = {json.loads(line)["task_id"] for line in
                (REPO / "data/p14e2/benchmark/reserved_test.agent.jsonl").read_text(encoding="utf-8").splitlines()
                if line.strip()}
    for path in RUNS.rglob("*.json*"):
        assert not any(task_id in path.read_text(encoding="utf-8", errors="ignore") for task_id in reserved), path

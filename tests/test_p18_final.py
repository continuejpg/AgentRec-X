"""P18 final integration tests: freeze, demos, performance, review status, reserved run, docs."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
RUNS = REPO / "runs/p18"


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def test_freeze_manifest_and_one_time_reserved_run():
    freeze = load(RUNS / "freeze_manifest.json")
    assert freeze["schema"] == "agentrecx.p18.freeze_manifest.v1"
    assert freeze["policy"]["no_reserved_retune"] is True
    assert freeze["core_file_count"] >= 10
    for rel, meta in freeze["files"].items():
        path = REPO / rel
        assert path.is_file(), rel
        assert hashlib.sha256(path.read_bytes()).hexdigest() == meta["sha256"], rel
    manifest = load(RUNS / "reserved/reserved_run_manifest.json")
    assert manifest["run_index"] == 1 and manifest["one_time"] is True
    assert manifest["post_result_tuning"] is False
    assert manifest["reserved_rows"] == 20
    summary = load(RUNS / "reserved/summary.json")["summary"]
    assert summary["recommend_tasks"] == 12
    assert summary["correct_top1"] == 11
    assert summary["false_ask"] == 0 and summary["false_refusal"] == 0
    assert summary["hard_violations"] == 0 and summary["errors"] == 0
    assert "simulated rule labels" in summary["reserved_probability"]["labels"]
    results = [json.loads(line) for line in (RUNS / "reserved/reserved_results.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(results) == 20
    source = (REPO / "experiments/p18/run_reserved.py").read_text(encoding="utf-8")
    assert "already run once; refusing a second run" in source


def test_three_cpu_mcp_demos_have_real_trajectories():
    transcript = load(RUNS / "demos/demo_transcript.json")
    assert transcript["mode"] == "CPU_MCP"
    assert transcript["qwen_used"] is False
    assert len(transcript["demos"]) == 3
    assert transcript["demos"][0]["recommendation"]["selected"]["item_id"]
    assert transcript["demos"][1]["comparison"]["status"] == "comparison_ready"
    assert transcript["demos"][2]["active_preferences_after"]
    for demo in transcript["demos"]:
        assert demo["mode"] == "CPU_MCP"
        assert demo.get("steps") or demo.get("recommendation")
    assert all(not demo.get("qwen_used", False) for demo in transcript["demos"])
    assert load(RUNS / "demos/search_and_recommend.json")["demo"] == "search_and_recommend"
    assert load(RUNS / "demos/multi_product_comparison.json")["demo"] == "multi_product_comparison"
    assert load(RUNS / "demos/clarification_and_memory.json")["demo"] == "clarification_and_memory"


def test_performance_cold_and_hot_are_separated():
    performance = load(RUNS / "performance.json")
    assert performance["gpu_used"] is False and performance["qwen_used"] is False
    assert performance["components"]["catalogue_load_ms"] > 0
    assert performance["components"]["bge_encoder_load_ms"] > 0
    assert performance["bge_first_search_ms"] > performance["bge_hot_search_ms"]
    assert performance["mcp_server_startup_ms"]
    assert performance["tool_discovery_ms"] > 0


def test_qwen_mode_is_replay_not_live():
    replay = load(RUNS / "qwen_replay.json")
    assert replay["live_call"] is False and replay["replay"] is True
    assert replay["examples"]
    assert all(item["mode"] == "Qwen_Agent_Replay" and item["replay"] is True for item in replay["examples"])
    docs = (REPO / "P18_FINAL_REPORT.md").read_text(encoding="utf-8")
    assert "Replay" in docs or "replay" in docs
    assert "Qwen Agent Mode" in docs


def test_human_review_materials_are_pending_and_not_claimed_reviewed():
    status = load(RUNS / "human_review_status.json")
    assert status["materials"] == 20
    assert status["reviewed"] == 0 and status["pending"] == 20
    assert status["human_review_performed"] is False
    assert status["semantic_truth_verified"] is False
    materials = [json.loads(line) for line in (RUNS / "human_review_materials.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(materials) == 20
    assert all(item["human_review"]["review_status"] == "pending_human_review" for item in materials)
    assert all(item["claim_validation"]["semantic_truth_verified"] is False for item in materials)


def test_final_deliverables_and_no_reserved_leakage():
    for rel in ("P18_FINAL_REPORT.md", "PROJECT_ARCHITECTURE.md", "DEMO_GUIDE.md", "EXPERIMENT_SUMMARY.md"):
        assert (REPO / rel).is_file(), rel
    assert (REPO / "README.md").is_file()
    assert "P18" in (REPO / "README.md").read_text(encoding="utf-8")
    reserved = {json.loads(line)["task_id"] for line in
                (REPO / "data/p14e2/benchmark/reserved_test.agent.jsonl").read_text(encoding="utf-8").splitlines()
                if line.strip()}
    for path in (RUNS / "demos").glob("*.json"):
        assert not any(task_id in path.read_text(encoding="utf-8") for task_id in reserved), path

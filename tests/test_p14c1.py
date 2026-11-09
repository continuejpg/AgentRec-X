"""P14-C1 tests: BGE integration, SFT data legality/leakage, training/Dev comparison."""

from __future__ import annotations

import json
from pathlib import Path

from experiments.p14e.catalogue import ElectronicsCatalogue

REPO = Path(__file__).resolve().parents[1]
RUNS = REPO / "runs/p14c1"


def _jsonl(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_bge_backend_is_integrated_and_default_serving_is_bm25():
    from experiments.p14c1.search_backend import configure_catalogue
    catalogue = ElectronicsCatalogue.load()
    assert catalogue.search_source.mode == "bm25"
    config = configure_catalogue(catalogue)
    assert config["status"] == "active" and config["mode"] == "bge"
    result = catalogue.search_source.search(
        "Which headphones would work well for my commute? It needs to be wireless.", limit=10)
    assert result.score_kind == "bge_dense"
    assert result.hits and all(hit.parent_asin.startswith("amazon_electronics_c4:") for hit in result.hits)
    catalogue_ids = {product.item_id for product in catalogue._products.values()}
    assert all(hit.parent_asin.split(":", 1)[-1] in catalogue_ids for hit in result.hits)


def test_bge_fallback_is_explicit_not_fake():
    from experiments.p14c1.search_backend import configure_catalogue
    from experiments.p14c1.search_backend import REPO as BASE
    catalogue = ElectronicsCatalogue.load()
    config = configure_catalogue(catalogue, index_path=BASE / "runs/p14c2_missing.npz",
                                 model_path=BASE / "models/bge-base-en-v1.5")
    assert config["status"] == "fallback" and config["mode"] == "bm25"
    assert catalogue.search_source.mode == "bm25"


def test_sft_data_is_legal_unleaked_and_train_only():
    data = json.loads((RUNS / "sft_data_manifest.json").read_text(encoding="utf-8"))
    rows = _jsonl(RUNS / "sft_train.jsonl")
    assert data["split"].startswith("train only")
    assert data["samples"] == len(rows) == 134
    assert data["tasks"] == 40 and data["tasks_with_samples"] == 40
    assert data["evaluator_fields_in_input"] is False
    assert data["target_source"].startswith("validated Rule PlannerAction")
    assert set(data["action_counts"]) == {"RETRIEVE", "RERANK", "ASK", "FINALIZE"}
    for row in rows:
        action = json.loads(row["completion"])["action"]
        assert action in row["allowed_actions"]
        text = json.dumps(row, ensure_ascii=False).lower()
        assert "target_parent_asin" not in text
        assert "expected_decision" not in text
        assert "verified_candidate" not in text


def test_sft_training_artifact_and_manifest():
    training = json.loads((RUNS / "training_manifest.json").read_text(encoding="utf-8"))
    assert training["method"] == "QLoRA" and training["train_samples"] == 134
    assert training["epochs"] == 2 and training["train_seconds"] > 0
    assert training["train_loss"] < 0.2
    assert (RUNS / "sft_adapter/adapter_model.safetensors").is_file()
    manifest = json.loads((RUNS / "model_manifest.json").read_text(encoding="utf-8"))
    assert manifest["sft_adapter"]["adapter_files"]
    assert manifest["no_reranker_training"] is True


def test_dev_comparison_uses_same_bge_backend():
    comparison = json.loads((RUNS / "comparison.json").read_text(encoding="utf-8"))
    assert comparison["search_backend"]["mode"] == "bge"
    assert comparison["search_backend"]["status"] == "active"
    planners = comparison["planners"]
    assert planners["rule"]["correct_top1"] == 12
    assert planners["qwen_base"]["correct_top1"] == 15
    assert planners["qwen_sft"]["correct_top1"] == 12
    assert planners["qwen_base"]["errors"] == 1
    assert planners["qwen_sft"]["errors"] == 0
    assert planners["qwen_sft"]["avg_model_latency_ms"] > planners["qwen_base"]["avg_model_latency_ms"]


def test_decision_and_reserved_guard():
    decision = json.loads((RUNS / "decision.json").read_text(encoding="utf-8"))
    assert decision["BGE_INTEGRATION"] == "PASS"
    assert decision["SFT_TRAINING"] == "PASS"
    assert decision["BASE_VS_SFT"] == "FAIL"
    assert decision["sft_adopted"] is False
    assert decision["keep_planner"] == "Qwen Base"
    assert (RUNS / "bge_integration.json").is_file()
    reserved = {json.loads(line)["task_id"] for line in
                (REPO / "data/p14e2/benchmark/reserved_test.agent.jsonl").read_text(encoding="utf-8").splitlines()
                if line.strip()}
    for path in RUNS.rglob("*.json*"):
        assert not any(task_id in path.read_text(encoding="utf-8", errors="ignore") for task_id in reserved), path
    assert (REPO / "runs/p14d2/comparison.json").is_file()
    assert (REPO / "runs/p14b3_eval100/comparison.json").is_file()

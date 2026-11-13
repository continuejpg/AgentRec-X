"""Build the machine-readable inventory and human-readable inventory report."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
DOC_MD = REPO / "docs/DATASET_ASSET_INVENTORY.md"
DOC_JSON = REPO / "docs/DATASET_ASSET_INVENTORY.json"


def _sha(path: Path) -> str | None:
    if not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def entry(
    asset_id: str,
    path: str,
    classes: list[str],
    kind: str,
    *,
    rows: int | None = None,
    users: int | None = None,
    items: int | None = None,
    key_fields: str = "",
    source: str = "",
    label_type: str = "",
    human_label: bool | None = None,
    training_use: str = "",
    evaluation_use: str = "",
    code_dependency: bool = False,
    duplicate_of: str | None = None,
    rebuildable: str = "",
    sha: str | None = None,
    location: str = "local",
) -> dict[str, Any]:
    p = REPO / path
    size = p.stat().st_size if p.is_file() else None
    if sha is None and p.is_file():
        sha = _sha(p)
    return {
        "id": asset_id,
        "path": path,
        "location": location,
        "classes": classes,
        "file_type": kind,
        "bytes": size,
        "rows": rows,
        "users": users,
        "items": items,
        "key_fields": key_fields,
        "source": source,
        "label_type": label_type,
        "human_label": human_label,
        "training_use": training_use,
        "evaluation_use": evaluation_use,
        "code_dependency": code_dependency,
        "duplicate_of": duplicate_of,
        "rebuildable": rebuildable,
        "sha256": sha,
    }


def main() -> int:
    assets = [
        entry(
            "electronics_catalog", "data/p13r/electronics/catalog_meta.jsonl", ["A", "C"],
            "JSONL catalogue metadata", rows=76918, items=76918,
            key_fields="item_id, category, text", source="Amazon-C4 sampled Electronics metadata",
            label_type="none", human_label=None, code_dependency=True,
            rebuildable="P13-R prepare_data with authorised C4 raw inputs",
        ),
        entry(
            "metadata_sidecar", "data/enrichment/electronics_metadata_sidecar.jsonl", ["A", "C"],
            "JSONL metadata evidence", rows=76263, items=76263,
            key_fields="item_id, attributes/evidence", source="Amazon Reviews 2023 Electronics raw metadata",
            label_type="none", human_label=None, code_dependency=True,
            rebuildable="scripts/build_electronics_metadata_sidecar.py",
        ),
        entry(
            "enriched_retrieval_docs", "data/enrichment/electronics_retrieval_docs_v2.jsonl", ["A", "C"],
            "JSONL retrieval documents", rows=76263, items=76263,
            key_fields="item_id, title, attributes, specs, features, description",
            source="catalog + sidecar", label_type="none", code_dependency=True,
            rebuildable="experiments/build_enriched_retrieval_docs.py",
        ),
        entry(
            "original_bge_index", "runs/p14d2/bge_index.npz", ["A"],
            "NPZ dense index", rows=76263, items=76263,
            key_fields="identities, vectors[76263,768]", source="BAAI/bge-base-en-v1.5 revision a5beb1e3",
            label_type="none", code_dependency=True,
            rebuildable="encode 76,263 accepted identities; 213MB",
        ),
        entry(
            "bge_index_manifest", "runs/p14d2/dense_index_manifest.json", ["A", "C"],
            "JSON manifest", source="P14-D2", label_type="none", code_dependency=True,
        ),
        entry(
            "p14d1_tasks", "data/p14d1/dev_tasks.jsonl", ["C", "D"],
            "JSONL natural retrieval tasks", rows=135,
            key_fields="task_id, query_original, query_rewritten",
            source="P14-D1", label_type="provisional natural query; one observed positive",
            evaluation_use="retrieval evaluation", code_dependency=True,
            rebuildable="P14-D1 preparation",
        ),
        entry(
            "p14d2_retrieval", "runs/p14d2/bge_retrieval.json", ["C", "D"],
            "JSON retrieval ranks", rows=135,
            key_fields="top100.original/rewritten, score",
            source="original BGE", label_type="provisional one-positive",
            evaluation_use="dense retrieval baseline", code_dependency=True,
            rebuildable="BGE encode + rank over P14-D1",
        ),
        entry(
            "p14e2_dev_agent", "data/p14e2/benchmark/dev.agent.jsonl", ["C"],
            "JSONL Agent tasks", rows=20,
            key_fields="task_id, current_query, persona/history",
            source="P14-E2", label_type="provisional automated checked",
            evaluation_use="Agent development/evaluation", code_dependency=True,
        ),
        entry(
            "p14e2_dev_evaluator", "data/p14e2/benchmark/dev.evaluator.jsonl", ["C"],
            "JSONL evaluator labels", rows=20,
            key_fields="task_id, expected_decision, hard_constraints, effective_preferences",
            source="P14-E2", label_type="provisional acceptable-id evaluator",
            evaluation_use="Agent official dev evaluation", code_dependency=True,
        ),
        entry(
            "p14b3_eval100_agent", "data/p14b3_eval100/eval100.agent.jsonl", ["C"],
            "JSONL Agent tasks", rows=143,
            key_fields="task_id, current_query, persona/history",
            source="P14-B3", label_type="pending human provisional",
            evaluation_use="held-out task source", code_dependency=True,
        ),
        entry(
            "p14b3_eval100_evaluator", "data/p14b3_eval100/eval100.evaluator.jsonl", ["C"],
            "JSONL evaluator labels", rows=143,
            key_fields="task_id, expected_decision, hard_constraints, effective_preferences",
            source="P14-B3", label_type="pending human provisional",
            evaluation_use="held-out acceptable ids", code_dependency=True,
        ),
        entry(
            "p15_evidence", "runs/p15/metrics.json", ["C", "D"],
            "JSON evidence metrics", source="P15 Evidence / claim validation",
            label_type="provisional evidence support", evaluation_use="evidence regression",
        ),
        entry(
            "p18_freeze_manifest", "runs/p18/freeze_manifest.json", ["A", "C"],
            "JSON frozen manifest", source="P18", label_type="frozen; no changes",
            evaluation_use="frozen/reserved; protected", code_dependency=True,
        ),
        entry(
            "p18_reserved", "runs/p18/reserved", ["C"],
            "Reserved task directory", rows=20,
            key_fields="reserved task payloads", source="P18",
            label_type="reserved; no tuning", evaluation_use="frozen reserved",
            code_dependency=False,
        ),
        entry(
            "p20_task_suite", "runs/p20_agent_eval/task_suite.json", ["C"],
            "JSON task suite", rows=30,
            key_fields="task_id, split, expected, hard, categories",
            source="P20 / P14-E2 + P14-B3", label_type="provisional",
            evaluation_use="Agent regression and A/B", code_dependency=True,
        ),
        entry(
            "p20_base_arm", "runs/advanced_retrieval_v2/task_arm_original_bge.json", ["C", "D"],
            "JSON Agent task records", rows=30,
            key_fields="selected_product_id, success, steps, gate",
            source="Planner SFT/ranking evaluation", label_type="provisional Agent outcome",
            evaluation_use="Base Agent baseline",
        ),
        entry(
            "planner_sft_v0", "data/planner_sft_v0/train.jsonl", ["B", "D"],
            "JSONL Planner Action", rows=267,
            key_fields="messages, target_action", source="rule/tool synthetic",
            label_type="rule-derived, not human", human_label=False,
            training_use="old Planner SFT only; not ranking", rebuildable="training/planner_sft/build_v0.py",
        ),
        entry(
            "planner_sft_v1_train", "data/planner_sft_v1/train.jsonl", ["B", "D"],
            "JSONL Planner Action", rows=307,
            key_fields="messages, target_action", source="53 real Qwen Action + 320 rule-derived",
            label_type="actual successful Qwen action or rule", human_label=False,
            training_use="Planner SFT v1 only; not ranking", rebuildable="training/planner_sft/build_v1.py",
        ),
        entry(
            "preference_operation_sft_train", "training/agent_sft/data/train.jsonl", ["B", "D"],
            "JSONL preference operation", rows=2400,
            key_fields="user_message, state, assistant operations",
            source="P2-SFT synthetic rule", label_type="rule-derived; not human",
            human_label=False, training_use="old preference-operation SFT only",
            rebuildable="P2-SFT generator",
        ),
        entry(
            "preference_ranking_v1", "runs/preference_ranking_v1/ranking_dataset.jsonl", ["C"],
            "JSONL fixed-pool ranking", rows=30, items=19,
            key_fields="query, preferences, hard, candidates, acceptable_ids, A/B/C ranks",
            source="P20 real traces + provisional evaluator ids",
            label_type="provisional acceptable ids; unjudged non-acceptable",
            human_label=False, evaluation_use="ranking baseline/report",
            rebuildable="experiments/preference_ranking/build_fixed_pool.py",
        ),
        entry(
            "esci_examples_raw", "data/esci_external/shopping_queries_dataset_examples.parquet",
            ["B", "D"], "Parquet ESCI judgments", rows=2621288,
            key_fields="query, query_id, product_id, locale, esci_label, split, small_version",
            source="amazon-science/esci-data commit 7916cdf6; Apache-2.0",
            label_type="human E/S/C/I", human_label=True,
            training_use="Dataset A source", evaluation_use="Dataset A official test",
            rebuildable="official parquet download",
        ),
        entry(
            "esci_products_raw", "data/esci_external/shopping_queries_dataset_products.parquet",
            ["B", "D"], "Parquet ESCI products", rows=1814924,
            key_fields="product_id, title, description, bullet, brand, color, locale",
            source="amazon-science/esci-data; Apache-2.0", label_type=None,
            training_use="Dataset A product text", rebuildable="official parquet download",
        ),
        entry(
            "dataset_a_esci", "data/gold/esci_relevance_v1/train.jsonl.gz", ["B", "C"],
            "Gzipped JSONL relevance", rows=378413, items=482105,
            key_fields="query, product_id, product_text, esci_label, gain, query_group, split",
            source="ESCI small_version=1 US; official train/test; val from train query groups",
            label_type="human E/S/C/I; mapped gains", human_label=True,
            training_use="external cross-encoder/reranker candidate", evaluation_use="relevance benchmark",
            rebuildable="experiments/dataset_consolidation/build_esci_dataset_a.py",
        ),
        entry(
            "dataset_b_amazon_reviews", "data/gold/amazon_reviews_electronics_v1/train.jsonl.gz",
            ["B", "C"], "Gzipped JSONL preference histories", rows=8270, users=14157, items=27403,
            key_fields="user_id, chronological history, target_item, rating, timestamps, split",
            source="Amazon Reviews 2023 Electronics valid/test ratings; catalog intersection",
            label_type="observed rating; no human pairwise", human_label=False,
            training_use="external pointwise preference signal only", evaluation_use="personalized benchmark",
            rebuildable="experiments/dataset_consolidation/build_amazon_reviews_dataset_b.py",
        ),
        entry(
            "amazon_catalog_ratings_raw", "data/gold/amazon_reviews_electronics_v1/electronics_catalog_ratings.csv",
            ["B", "D"], "CSV real ratings", rows=1104764, users=1005308, items=27403,
            key_fields="user_id, parent_asin, rating, timestamp, source_split",
            source="Electronics.valid.csv + Electronics.test.csv, exact catalog-id intersection",
            label_type="observed rating", human_label=False,
            rebuildable="remote raw ratings + exact catalog IDs",
        ),
        entry(
            "trec_2025_qrels", "data/trec2025/2025-qrels.rec", ["C"],
            "TREC qrels text", rows=7633, items=7103,
            key_fields="query_id, iteration, product_id, graded label",
            source="TREC 2025 Product Recommendation official; Apache-2.0 dataset card",
            label_type="human assessor S/C/NR/UA grades", human_label=True,
            evaluation_use="independent external test only",
            rebuildable="trec.nist.gov/data/product2025.html",
        ),
        entry(
            "trec_2025_corpus", "data/trec2025/product-corpus.json.zst", ["C", "D"],
            "Zstd JSONL product corpus", source="TREC 2025 product corpus; ESCI/M2 derived",
            label_type="none", evaluation_use="external product text",
            rebuildable="HF trec-product-search dataset",
        ),
        entry(
            "sports_processed", "data/processed/Sports_and_Outdoors_sequences.json", ["D"],
            "JSON interaction sequences", users=412445, items=156746,
            key_fields="user histories, temporal leave-two-out", source="Amazon Reviews 2023 Sports & Outdoors",
            label_type="real interactions; known-positive next item", human_label=False,
            evaluation_use="old recommendation baseline; not current Electronics",
            rebuildable="preprocessing pipeline",
        ),
        entry(
            "planner_sft_v0_adapter", "runs/dataset_archive/planner_sft_v0/adapter_model.safetensors",
            ["D"], "Safetensors LoRA adapter", source="Planner SFT v0 (rule-derived)",
            label_type="rule-derived SFT; not human", human_label=False,
            training_use="archive only; not ranking", evaluation_use="old adapter action comparison",
            sha="54797dc5fc8d0aa67bd47facac6732f1a9886f08a289701e72f73572a84e96d8",
        ),
        entry(
            "planner_sft_v1_adapter", "runs/dataset_archive/planner_sft_v1/adapter_model.safetensors",
            ["D"], "Safetensors LoRA adapter", source="Planner SFT v1 QLoRA",
            label_type="53 real successful Qwen actions + rule labels", human_label=False,
            training_use="archive only; rejected for Agent deployment",
            evaluation_use="action/Agent comparison",
            sha="3ed5d5ffbcd41481a4312765a227bf084a78646925cd1bc7cb1d10ca7a2ef222",
        ),
        entry(
            "p14c1_adapter", "runs/p14c1/sft_adapter/adapter_model.safetensors", ["D"],
            "Safetensors LoRA adapter", source="P14-C1 preference-operation QLoRA",
            label_type="preference-operation SFT; different schema", human_label=False,
            training_use="archive only", evaluation_use="not Planner ranking",
            sha="f191c811cab6f1f3da98466c0c0341ccfd0bfa4105b371e1c1af4500631326b8",
        ),
        entry(
            "p10_lightgcn_checkpoint", "runs/p10/lightgcn/checkpoint.pt", ["D"],
            "PyTorch checkpoint", source="P10 old recommendation baseline",
            label_type="real interaction training", human_label=False,
            evaluation_use="historical only", rebuildable="P10 training",
        ),
        entry(
            "p10_bge_sasrec_checkpoint", "runs/p10/bge_sasrec/checkpoint.pt", ["D"],
            "PyTorch checkpoint", source="P10 old BGE+SASRec experiment",
            label_type="real interaction training", human_label=False,
            evaluation_use="historical only", rebuildable="P10 training",
        ),
        entry(
            "p11_benchmark_tasks", "data/p11_benchmark/tasks.jsonl", ["C", "D"],
            "JSONL benchmark tasks", source="P11 benchmark",
            label_type="provisional", evaluation_use="historical benchmark",
        ),
        entry(
            "p13_sports_c4_test", "data/p13/sports/amazon_c4_test.csv", ["D"],
            "CSV query data", source="P13 Sports/C4", label_type="provisional",
            evaluation_use="historical retrieval", rebuildable="P13 preparation",
        ),
        entry(
            "p14a_products_prefix", "data/p14a/catalogue/products_prefix.jsonl", ["D"],
            "JSONL old catalogue prefix", source="P14-A",
            label_type="none", rebuildable="P14-A prep",
        ),
        entry(
            "p14a_train_eval", "data/p14a/dataset/train_eval.jsonl", ["D"],
            "JSONL old train/eval", source="P14-A", label_type="provisional",
            evaluation_use="historical only", rebuildable="P14-A prep",
        ),
        entry(
            "m4_similar_item_index", "runs/m4_evidence/similar_item_index.pkl", ["D"],
            "Pickle similarity index", source="M4 Evidence", label_type="real interactions",
            human_label=False, evaluation_use="historical evidence/ranking",
            rebuildable="M4 build",
        ),
        entry(
            "lr_v1_r2_groups", "runs/lr_v1_r2/rlr01_groups.jsonl", ["D"],
            "JSONL learned-ranker groups", source="LR v1/R2 experiment",
            label_type="proxy labels; not human gold", human_label=False,
            training_use="historical learned ranker", evaluation_use="historical only",
            rebuildable="LR v1/R2 build",
        ),
        entry(
            "sasrec_checkpoint", "runs/sasrec_canonical_2026/best.pt", ["D"],
            "PyTorch checkpoint", source="SASRec canonical 2026", label_type="real interactions",
            human_label=False, evaluation_use="historical recommendation baseline",
            rebuildable="SASRec training",
        ),
        entry(
            "twotower_checkpoint", "runs/twotower_public_2026_full/checkpoint.pt", ["D"],
            "PyTorch checkpoint", source="Two-Tower public 2026", label_type="real interactions",
            human_label=False, evaluation_use="historical recommendation baseline",
            rebuildable="Two-Tower training",
        ),
        entry(
            "tiger_checkpoint", "runs/tiger_public_2026/checkpoint/tiger.pt", ["D"],
            "PyTorch checkpoint", source="TIGER public 2026", label_type="real interactions",
            human_label=False, evaluation_use="historical generative backend",
            rebuildable="TIGER training",
        ),
        entry(
            "p15_e2e_trajectories", "runs/p15/e2e_trajectories.json", ["C", "D"],
            "JSON evidence trajectories", source="P15 E2E", label_type="provisional evidence support",
            evaluation_use="evidence/claim regression",
        ),
    ]
    # Add duplicate vector entries and remote-only model/env assets.
    for asset_id, path, duplicate_of, size in (
        ("esci_product_vectors_gpu_duplicate", "runs/esci_external/bge_product_vectors_gpu.npz",
         "runs/esci_external/bge_product_vectors.npz", 470451487),
        ("esci_query_vectors_gpu_duplicate", "runs/esci_external/bge_query_vectors_gpu.npz",
         "runs/esci_external/bge_query_vectors.npz", 25573984),
        ("esci_queries_shard0_duplicate", "runs/esci_external/bge_queries_shard0.npz",
         "runs/esci_external/bge_query_vectors_cpu.npz", 25587104),
    ):
        p = REPO / path
        assets.append({
            "id": asset_id, "path": path, "location": "local", "classes": ["F"],
            "file_type": "NPZ embedding cache", "bytes": p.stat().st_size if p.is_file() else size,
            "rows": None, "users": None, "items": None,
            "key_fields": "vectors", "source": "ESCI external experiment",
            "label_type": None, "human_label": None, "training_use": "cache only",
            "evaluation_use": "reproducibility", "code_dependency": False,
            "duplicate_of": duplicate_of,
            "rebuildable": "recompute BGE vectors from raw parquet",
            "sha256": _sha(p) if p.is_file() else None,
        })
    assets.extend([
        {"id": "qwen_base", "path": "remote:/root/autodl-tmp/models/Qwen3.5-9B-ms",
         "location": "remote", "classes": ["A", "D"], "file_type": "HF model directory",
         "bytes": 19_000_000_000, "rows": None, "users": None, "items": None,
         "key_fields": "Qwen3_5ForCausalLM, tokenizer, config", "source": "Qwen3.5-9B-ms",
         "label_type": None, "human_label": None, "training_use": "Planner base; no overwrite",
         "evaluation_use": "Agent base", "code_dependency": True, "duplicate_of": None,
         "rebuildable": "download only if lost; do not download now", "sha256": None},
        {"id": "bge_model", "path": "remote:/root/autodl-tmp/models/bge-base-en-v1.5",
         "location": "remote", "classes": ["A"], "file_type": "HF model directory",
         "bytes": 1_300_000_000, "rows": None, "users": None, "items": None,
         "key_fields": "BGE-base-en-v1.5", "source": "BAAI", "label_type": None,
         "human_label": None, "training_use": "retrieval baseline", "evaluation_use": "dense retrieval",
         "code_dependency": True, "duplicate_of": None, "rebuildable": "existing model", "sha256": None},
        {"id": "qwen3_reranker", "path": "remote:/root/autodl-tmp/models/Qwen3-Reranker-4B",
         "location": "remote", "classes": ["A", "D"], "file_type": "HF reranker model",
         "bytes": 7_600_000_000, "rows": None, "users": None, "items": None,
         "key_fields": "Qwen3-Reranker-4B", "source": "Qwen", "label_type": None,
         "human_label": None, "training_use": "frozen reranker", "evaluation_use": "candidate ranking",
         "code_dependency": False, "duplicate_of": None, "rebuildable": "existing; do not download",
         "sha256": None},
        {"id": "agentrecx_sft_env", "path": "remote:/root/autodl-tmp/agentrecx-sft",
         "location": "remote", "classes": ["A"], "file_type": "Conda/Python environment",
         "bytes": 7_300_000_000, "rows": None, "users": None, "items": None,
         "key_fields": "torch 2.11, transformers, peft, bitsandbytes", "source": "AutoDL existing env",
         "label_type": None, "human_label": None, "training_use": "GPU training env",
         "evaluation_use": "server/training", "code_dependency": True, "duplicate_of": None,
         "rebuildable": "do not reinstall without approval", "sha256": None},
        {"id": "agentrecx_phase2", "path": "remote:/root/autodl-tmp/AgentRec-X-phase2",
         "location": "remote", "classes": ["A"], "file_type": "project working copy",
         "bytes": 594_000_000, "rows": None, "users": None, "items": None,
         "key_fields": "code, data, runs", "source": "synced from local Git archive",
         "label_type": None, "human_label": None, "training_use": "GPU runner",
         "evaluation_use": "remote experiments", "code_dependency": True, "duplicate_of": None,
         "rebuildable": "git archive + data sync", "sha256": None},
    ])
    manifest = {
        "schema": "agentrecx.dataset_asset_inventory.v1",
        "generated_by": "experiments/dataset_consolidation/build_inventory.py",
        "classes": {
            "A": "Active runtime assets", "B": "Training candidates",
            "C": "Evaluation / frozen assets", "D": "Reproducibility archives",
            "E": "Regenerable temporary files", "F": "Duplicate / obsolete assets",
        },
        "protected": [
            "data/p13r/electronics/catalog_meta.jsonl",
            "data/enrichment/electronics_metadata_sidecar.jsonl",
            "runs/p14d2/bge_index.npz",
            "runs/p18/freeze_manifest.json",
            "runs/p20_agent_eval/task_suite.json",
            "remote:/root/autodl-tmp/models/Qwen3.5-9B-ms",
            "remote:/root/autodl-tmp/agentrecx-sft",
            "remote:/root/autodl-tmp/AgentRec-X-phase2",
        ],
        "assets": assets,
        "duplicates": [a["id"] for a in assets if a.get("duplicate_of")],
        "old_sft_checkpoints": {
            "status": "deleted in the preceding cleanup round; no new checkpoint deletion here",
            "deleted_paths": [
                "agentrecx/training/agent_sft/output/qwen3_5_9b_lora/checkpoints",
                "agentrecx/training/agent_sft/output/qwen3_5_9b_lora_v2/checkpoints",
                "agentrecx/training/planner_sft/output/qwen3_5_9b_planner_lora_v0/checkpoints",
            ],
            "freed_estimate_gib": 2.9,
        },
        "final_adapter_archives": [
            {"path": "runs/dataset_archive/planner_sft_v0", "sha256": "54797dc5fc8d0aa67bd47facac6732f1a9886f08a289701e72f73572a84e96d8"},
            {"path": "runs/dataset_archive/planner_sft_v1", "sha256": "3ed5d5ffbcd41481a4312765a227bf084a78646925cd1bc7cb1d10ca7a2ef222"},
            {"path": "runs/p14c1/sft_adapter/adapter_model.safetensors", "sha256": "f191c811cab6f1f3da98466c0c0341ccfd0bfa4105b371e1c1af4500631326b8"},
        ],
        "cleanup_plan_note": "No destructive cleanup is performed in this inventory round; duplicate vector cleanup is proposed but not authorised.",
    }
    DOC_JSON.write_text(json.dumps(manifest, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    lines = [
        "# Dataset Asset Inventory",
        "",
        "Generated by `experiments/dataset_consolidation/build_inventory.py`. Machine-readable",
        "manifest: `docs/DATASET_ASSET_INVENTORY.json`.",
        "",
        "| ID | Path | Classes | Type | Bytes | Rows | Users | Items | Labels | Human | Train | Eval | Code dep | Sha256 |",
        "|---|---|---|---|---:|---:|---:|---:|---|---:|---|---|---:|---|",
    ]
    for a in assets:
        lines.append(
            f"| {a['id']} | `{a['path']}` | {','.join(a['classes'])} | {a['file_type']} | "
            f"{a['bytes'] if a['bytes'] is not None else ''} | "
            f"{a['rows'] if a['rows'] is not None else ''} | "
            f"{a['users'] if a['users'] is not None else ''} | "
            f"{a['items'] if a['items'] is not None else ''} | "
            f"{a['label_type'] or ''} | "
            f"{'yes' if a['human_label'] else ('no' if a['human_label'] is False else '')} | "
            f"{a['training_use']} | {a['evaluation_use']} | "
            f"{'yes' if a['code_dependency'] else 'no'} | "
            f"`{a['sha256'] or ''}` |"
        )
    lines += [
        "",
        "## Retention classes",
        "",
        "- **A** Active runtime: catalogue, sidecar, original BGE index, Qwen base, current code, environment.",
        "- **B** Training candidates: ESCI Dataset A, Amazon Dataset B, Planner SFT assets.",
        "- **C** Evaluation/frozen: P14-D1/D2, P14-E2/B3, P15, P18, P20, ranking v1, TREC.",
        "- **D** Reproducibility archives: raw ESCI/TREC/review files, old manifests, final adapters.",
        "- **E** Regenerable temporary: embeddings/caches, scratch pools.",
        "- **F** Duplicate/obsolete: ESCI CPU/GPU vector caches; old SFT checkpoints already removed.",
        "",
        "## Protected",
        "",
    ]
    lines += [f"- `{item}`" for item in manifest["protected"]]
    lines += [
        "",
        "## Final adapter archives",
        "",
    ]
    lines += [f"- `{item['path']}` — SHA256 `{item['sha256']}`" for item in manifest["final_adapter_archives"]]
    lines += [
        "",
        "## Cleanup plan",
        "",
        "- Keep all active runtime and frozen assets untouched.",
        "- Keep at least one final copy of each adapter; the three final adapter archives above are retained.",
        "- Old SFT checkpoints were deleted in the preceding cleanup round; no new checkpoint deletion in this round.",
        "- Duplicate ESCI CPU/GPU vector caches are listed as class F and are not deleted because they are only regenerable caches and no deletion authorisation was given here.",
        "- Raw ESCI, TREC, and Amazon ratings are class D; do not delete unless a verified external backup exists.",
        "- Do not delete any model weights, current catalogue, sidecar, BGE index, P18, or task suite.",
    ]
    DOC_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"assets": len(assets), "md": str(DOC_MD), "json": str(DOC_JSON)}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

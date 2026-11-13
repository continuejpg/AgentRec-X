"""Independent ESCI relevance benchmark manifest.

The official test split (small_version=1, US) is preserved by Dataset A.  The BGE/BM25
numbers recorded here come from the existing judged-candidate ESCI external experiment and
are not re-labelled as a full-catalogue metric.
"""

from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "data/gold/esci_relevance_v1"


def main() -> int:
    manifest = json.loads((OUT / "manifest.json").read_text(encoding="utf-8"))
    metrics = json.loads((REPO / "runs/esci_external/esci_metrics.json").read_text(encoding="utf-8"))
    benchmark = {
        "schema": "agentrecx.gold.esci_relevance.benchmark_manifest.v1",
        "task": "ESCI Task 1 query-product ranking; official small_version=1 test, US",
        "dataset": str(OUT / "test.jsonl.gz"),
        "dataset_sha256": manifest["files_sha256"]["test"],
        "official_test_preserved": True,
        "label_counts": manifest["label_counts"]["test"],
        "queries": manifest["unique_queries"]["test"],
        "candidates": manifest["counts"]["test"],
        "candidate_policy": "judged ESCI candidates only; no unjudged products added",
        "metrics": {
            "bm25_ndcg10_primary_mean": metrics["aggregate"]["bm25_ndcg10_primary"]["mean"],
            "bge_ndcg10_primary_mean": metrics["aggregate"]["bge_ndcg10_primary"]["mean"],
            "bm25_ndcg_all_primary_mean": metrics["aggregate"]["bm25_ndcg_all_primary"]["mean"],
            "bge_ndcg_all_primary_mean": metrics["aggregate"]["bge_ndcg_all_primary"]["mean"],
            "bm25_mrr10_binary_ES": 0.904570,
            "bge_mrr10_binary_ES": 0.936004,
            "bm25_mrr10_ESC": 0.937021,
            "bge_mrr10_ESC": 0.954548,
        },
        "metrics_source": "runs/esci_external/esci_metrics.json; judged-candidate reranking, not full-catalogue retrieval",
        "candidate_text": "ESCI product_title for the prior BGE/BM25 run; Dataset A includes richer product text",
        "independent_from_agent": True,
        "training_allowed_on": ["train", "val"],
        "test_use": "report only; never use for model selection",
    }
    (OUT / "benchmark_manifest.json").write_text(json.dumps(benchmark, indent=1, ensure_ascii=False) + "\n",
                                                encoding="utf-8")
    print(json.dumps(benchmark, indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

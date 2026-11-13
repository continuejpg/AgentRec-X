"""Independent TREC 2025 Product Recommendation benchmark manifest.

Official qrels are used unchanged.  This is an external test-only benchmark; no training
labels are synthesized from it.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
TREC = REPO / "data/trec2025"
OUT = REPO / "data/gold/trec2025_product_recommendation_v1"
LABELS = {
    "S2": "full_substitute",
    "S1": "conceptual_substitute",
    "S0": "related_not_substitute",
    "C2": "full_complement",
    "C1": "complement_unknown_compatibility",
    "C0": "complementary_but_incompatible",
    "NR": "not_related",
    "UA": "unable_to_assess",
}


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _qrels(path: Path) -> list[tuple[str, str, str]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            qid, _iteration, product, label = line.split()[:4]
            rows.append((qid, product, label))
    return rows


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    qrels = _qrels(TREC / "2025-qrels.rec")
    queries = [line.rstrip("\n").split("\t") for line in
               (TREC / "product-recommendation-2025-test-queries.tsv").read_text(encoding="utf-8").splitlines()
               if line.strip()]
    query_ids = {row[0] for row in queries[1:]}
    qrels_qids = {row[0] for row in qrels}
    catalog = {
        json.loads(line)["item_id"]
        for line in (REPO / "data/p13r/electronics/catalog_meta.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    }
    products = {row[1] for row in qrels}
    manifest = {
        "schema": "agentrecx.gold.trec2025_product_recommendation.benchmark_manifest.v1",
        "dataset": "TREC 2025 Product Recommendation",
        "source": "https://trec.nist.gov/data/product2025.html",
        "hf_dataset": "https://huggingface.co/datasets/trec-product-search/product-recommendation-2025",
        "license": "Apache-2.0 (TREC product dataset card; derived from ESCI/M2)",
        "query_count": len(query_ids),
        "qrels_rows": len(qrels),
        "qrels_queries": len(qrels_qids),
        "catalog_products": len(products),
        "catalog_intersection_products": len(products & catalog),
        "catalog_intersection_rate": round(len(products & catalog) / max(1, len(products)), 6),
        "label_counts": dict(Counter(row[2] for row in qrels)),
        "label_meaning": LABELS,
        "queries_match_qrels": bool(query_ids) and query_ids <= qrels_qids,
        "test_queries": len(query_ids),
        "queries_with_qrels": len(query_ids & qrels_qids),
        "queries_without_qrels": sorted(query_ids - qrels_qids),
        "test_only": True,
        "training_use": "forbidden; external independent evaluation only",
        "candidate_policy": "official pooled qrels only; no synthetic negatives",
        "files": {
            "qrels_rec": {"path": "data/trec2025/2025-qrels.rec", "sha256": _sha(TREC / "2025-qrels.rec")},
            "queries": {"path": "data/trec2025/product-recommendation-2025-test-queries.tsv",
                        "sha256": _sha(TREC / "product-recommendation-2025-test-queries.tsv")},
            "corpus": {"path": "data/trec2025/product-corpus.json.zst",
                       "sha256": _sha(TREC / "product-corpus.json.zst")},
        },
        "note": "TREC recommendation labels are relationship categories, not a single relevance scalar; C/S grades must be evaluated per relationship task.",
        "no_p18": True,
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=1, ensure_ascii=False) + "\n",
                                       encoding="utf-8")
    print(json.dumps(manifest, indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

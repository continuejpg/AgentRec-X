"""Build P100 fixed BGE Top-20 candidate pools and metadata on AutoDL."""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

from experiments.p14b1.diagnose import candidate_rows
from experiments.p14c1.search_backend import configure_catalogue
from experiments.p14e.catalogue import ElectronicsCatalogue
from recommendation.preference.scorer import PreferenceAwareScorer

REPO = Path("/root/autodl-tmp/AgentRec-X-phase2")
DATA = REPO / "data/p100_extended_benchmark"
OUT = REPO / "data/p100_extended_benchmark"
TASKS = DATA / "p100_agent_tasks.jsonl"


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main():
    tasks = read_jsonl(TASKS)
    catalogue = ElectronicsCatalogue.load(metadata_mode="enriched")
    retrieval = configure_catalogue(catalogue, mode="bge")
    warmup = catalogue.search_source.search("wireless earbuds for working out", limit=20)
    started = time.perf_counter()
    pools = []
    metadata = {}
    for task in tasks:
        result = catalogue.search_source.search(task["query"], limit=20)
        hits = list(result.hits)
        ids = []
        rows = []
        for index, hit in enumerate(hits):
            item_id = hit.parent_asin.split(":", 1)[1] if ":" in hit.parent_asin else hit.parent_asin
            product = catalogue.get(item_id)
            attrs = {}
            evidence = []
            if product is not None:
                for attr, fact in (product.attributes or {}).items():
                    attrs[attr] = True
                    if isinstance(fact, dict) and fact.get("evidence"):
                        evidence.append({"attribute": attr, "evidence": str(fact["evidence"])[:400]})
            row = {
                "item_id": item_id, "title": getattr(product, "title", None),
                "attributes": sorted(attrs), "evidence": evidence,
                "bge_rank": index + 1,
                "bge_score": round(float(hit.score), 8),
                "metadata_source": "data/enrichment/electronics_metadata_sidecar.jsonl",
            }
            ids.append(item_id)
            rows.append(row)
            metadata[item_id] = row
        # Eligibility is computed by the production row builder; hard requirements are query-derived.
        hard = []
        try:
            from experiments.p14c0.electronics_executor import prepare_hard
            hard = prepare_hard(task["query"])
        except Exception:
            hard = []
        eligibility = []
        if rows:
            base_task = {"task_id": task["base_query_id"], "current_query": task["query"],
                         "candidate_ids": ids, "search_scores": {}}
            try:
                scored = candidate_rows(base_task, catalogue, PreferenceAwareScorer(), [], hard)
                eligibility = [{"item_id": r["product_ref"]["item_id"], "hard_tier": r["hard_tier"],
                                "presentable": bool(r.get("presentable")),
                                "hard_states": r.get("hard_states", {})} for r in scored]
            except Exception as exc:
                eligibility = [{"error": f"{type(exc).__name__}: {exc}"}]
        pool = {
            "base_query_id": task["base_query_id"], "query_id": task["query_id"],
            "query": task["query"], "candidate_source": "original_bge_top20_electronics_catalog",
            "candidate_pool_ids": ids, "bge_scores": [row["bge_score"] for row in rows],
            "bge_ranks": [row["bge_rank"] for row in rows],
            "hard_requirements": hard, "eligibility": eligibility,
            "candidate_pool_hash": hashlib.sha256(json.dumps(ids, sort_keys=True).encode()).hexdigest(),
        }
        pools.append(pool)
    (OUT / "p100_bge_pools.jsonl").write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in pools) + "\n", encoding="utf-8")
    (OUT / "p100_candidate_metadata.jsonl").write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in metadata.values()) + "\n", encoding="utf-8")
    report = {"schema": "agentrecx.p100.bge_pools.v1", "retrieval": retrieval,
              "warmup_first_hit": warmup.hits[0].parent_asin if warmup.hits else None,
              "queries": len(pools), "candidate_rows": sum(len(p["candidate_pool_ids"]) for p in pools),
              "unique_candidates": len(metadata), "wall_seconds": round(time.perf_counter() - started, 3),
              "bge_index": "/root/autodl-tmp/AgentRec-X-phase2/runs/p14d2/bge_index.npz",
              "bge_index_size": 76263}
    (OUT / "p100_bge_pools_manifest.json").write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()

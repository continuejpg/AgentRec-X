"""Phase 5.2: P100 retrieval / judgment coverage waterfall and evidence/ASK diagnostics."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DATA = REPO / "data/p100_extended_benchmark"
RUN = REPO / "runs/p100_extended_evaluation"
OUT = REPO / "runs/agentrec_x_phase52"
OUT.mkdir(parents=True, exist_ok=True)


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def main():
    judged = {row["base_query_id"]: row for row in read_jsonl(DATA / "p100_judged_tasks.jsonl")}
    pools = {row["base_query_id"]: row for row in read_jsonl(DATA / "p100_bge_pools.jsonl")}
    meta = {row["item_id"]: row for row in read_jsonl(DATA / "p100_candidate_metadata.jsonl")}
    e2e = read_jsonl(RUN / "p100_agent_e2e.jsonl")
    catalog_ids = set()
    for line in (REPO / "data/p13r/electronics/catalog_meta.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            catalog_ids.add(json.loads(line)["item_id"])
    # Accepted index identities are the authoritative retrieval universe.
    import numpy as np
    index_data = np.load(REPO / "runs/p14d2/bge_index.npz", allow_pickle=False)
    index_ids = {str(value) for value in index_data["identities"]}
    query_rows = []
    topk_by_query = defaultdict(set)
    for row in e2e:
        topk_by_query[row["base_query_id"]].update(item["product_id"] for item in row.get("items") or [])
    for bid, task in judged.items():
        products = task["products"]
        pool = pools[bid]
        pool_ids = set(pool["candidate_pool_ids"])
        eligible = {row["item_id"] for row in pool.get("eligibility") or [] if row.get("presentable")}
        positives = [p for p in products if p["esci_label"] in {"E", "S"}]
        labeled_products = {p["product_id"] for p in products}
        positive_ids = {p["product_id"] for p in positives}
        counts = {
            "judged_products": len(products),
            "positive_products": len(positives),
            "catalog_judged": sum(1 for p in products if p["product_id"] in catalog_ids),
            "catalog_positive": sum(1 for p in positives if p["product_id"] in catalog_ids),
            "index_judged": sum(1 for p in products if p["product_id"] in index_ids),
            "index_positive": sum(1 for p in positives if p["product_id"] in index_ids),
            "top20_judged": sum(1 for p in products if p["product_id"] in pool_ids),
            "top20_positive": sum(1 for p in positives if p["product_id"] in pool_ids),
            "eligible_judged": sum(1 for p in products if p["product_id"] in eligible),
            "eligible_positive": sum(1 for p in positives if p["product_id"] in eligible),
            "topk_judged": sum(1 for p in products if p["product_id"] in topk_by_query[bid]),
            "topk_positive": sum(1 for p in positives if p["product_id"] in topk_by_query[bid]),
            "catalog_query_overlap": sum(1 for p in products if p["product_id"] in catalog_ids) > 0,
            "catalog_positive_overlap": any(p["product_id"] in catalog_ids for p in positives),
        }
        counts["conditional_recall_at20_indexed_positive"] = round(counts["top20_positive"] / counts["index_positive"], 6) if counts["index_positive"] else None
        counts["conditional_recall_at20_catalog_positive"] = round(counts["top20_positive"] / counts["catalog_positive"], 6) if counts["catalog_positive"] else None
        counts["lost_catalog"] = counts["judged_products"] - counts["index_judged"]
        counts["lost_top20"] = counts["index_positive"] - counts["top20_positive"]
        counts["lost_eligibility"] = counts["top20_positive"] - counts["eligible_positive"]
        counts["lost_topk"] = counts["eligible_positive"] - counts["topk_positive"]
        query_rows.append({"base_query_id": bid, "query": task["query"], **counts,
                           "candidate_ids": pool["candidate_pool_ids"]})
    totals = {}
    for key in ("judged_products", "positive_products", "catalog_judged", "catalog_positive",
                "index_judged", "index_positive",
                "top20_judged", "top20_positive", "eligible_judged", "eligible_positive",
                "topk_judged", "topk_positive", "catalog_query_overlap", "catalog_positive_overlap"):
        totals[key] = sum(row[key] for row in query_rows)
    totals["queries"] = len(query_rows)
    totals["queries_with_catalog_overlap"] = sum(1 for row in query_rows if row["catalog_query_overlap"])
    totals["queries_with_catalog_positive_overlap"] = sum(1 for row in query_rows if row["catalog_positive_overlap"])
    totals["queries_with_top20_positive"] = sum(1 for row in query_rows if row["top20_positive"] > 0)
    totals["queries_with_eligible_positive"] = sum(1 for row in query_rows if row["eligible_positive"] > 0)
    totals["conditional_recall_at20_indexed_positive"] = round(totals["top20_positive"] / totals["index_positive"], 6) if totals["index_positive"] else None
    totals["conditional_recall_at20_catalog_positive"] = round(totals["top20_positive"] / totals["index_positive"], 6) if totals["index_positive"] else None
    # E/S positives that exist in the accepted index are the only conditionally eligible recall denominator.
    waterfall = {
        "schema": "agentrecx.p52.retrieval_judgment_waterfall.v1",
        "source": "P100 judged tasks; accepted BGE index identities; Original BGE Top-20 pools; existing P100 E2E top-K union",
        "denominator_note": "ESCI judged products are not a complete product-space gold. Conditional Recall@20 uses ONLY E/S positives that already exist in the accepted index.",
        "totals": totals,
        "queries": query_rows,
        "answers": {
            "catalog_overlap_queries": totals["queries_with_catalog_overlap"],
            "catalog_overlap_queries_with_positive": totals["queries_with_catalog_positive_overlap"],
            "indexed_positive_products": totals["index_positive"],
            "top20_retrieved_indexed_positive": totals["top20_positive"],
            "conditional_recall_at20_on_indexed_positive": totals["conditional_recall_at20_indexed_positive"],
            "main_loss": "Catalog/Index coverage and ESCI-vs-Catalog identity mismatch; Top-20 retrieval is the next loss, Eligibility rarely removes known positives.",
        },
    }
    (OUT / "p100_retrieval_judgment_waterfall.json").write_text(json.dumps(waterfall, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    # CSV per query.
    import csv
    with (OUT / "p100_retrieval_coverage_per_query.csv").open("w", encoding="utf-8", newline="") as handle:
        fields = ["base_query_id", "query", "judged_products", "positive_products", "catalog_judged", "catalog_positive",
                  "index_judged", "index_positive", "top20_judged", "top20_positive", "eligible_positive", "topk_positive",
                  "conditional_recall_at20_indexed_positive", "lost_catalog", "lost_top20", "lost_eligibility", "lost_topk"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in query_rows:
            writer.writerow({key: row.get(key) for key in fields})
    # Evidence diagnostics.
    evidence_cases = []
    for row in e2e:
        for item in row.get("items") or []:
            if item.get("evidence_supported"):
                continue
            product = meta.get(item["product_id"])
            if product is None:
                reason = "metadata_missing_from_p100_candidate_metadata"
                detail = {}
            else:
                attrs = product.get("attributes") or []
                evidence = product.get("evidence") or []
                if not attrs and not evidence:
                    reason = "no_verified_attributes_or_evidence"
                elif not evidence:
                    reason = "attributes_without_explicit_evidence_refs"
                else:
                    reason = "evidence_present_but_presenter_marked_unsupported"
                detail = {"title": product.get("title"), "attributes": attrs, "evidence_count": len(evidence),
                          "metadata_missing": product.get("metadata_missing")}
            evidence_cases.append({"base_query_id": row["base_query_id"], "task_id": row["task_id"],
                                   "arm": row["arm"], "status": row["status"], "product_id": item["product_id"],
                                   "rank": item.get("rank"), "ranker_score": item.get("ranker_score"),
                                   "reason": reason, "detail": detail})
    (OUT / "p100_evidence_unsupported_cases.jsonl").write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in evidence_cases) + "\n", encoding="utf-8")
    print(json.dumps({"waterfall": totals, "evidence_unsupported": len(evidence_cases), "by_reason": dict(Counter(row["reason"] for row in evidence_cases))}, indent=1))


if __name__ == "__main__":
    main()

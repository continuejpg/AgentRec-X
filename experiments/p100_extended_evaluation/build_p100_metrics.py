"""P100 ranker-only, judged relevance and memory OFF/ON metrics."""

from __future__ import annotations

import csv
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DATA = REPO / "data/p100_extended_benchmark"
RUN = REPO / "runs/p100_extended_evaluation"
BENCH = REPO / "runs/memory_ranking_benchmark_v2"
SEED = 20261114
GAIN = {"E": 1.0, "S": 0.1, "C": 0.01, "I": 0.0}


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def dcg(gains):
    return sum(g / math.log2(i + 2) for i, g in enumerate(gains))


def ndcg(gains, k):
    ideal = sorted(gains, reverse=True)[:k]
    actual = gains[:k]
    denom = dcg(ideal)
    return dcg(actual) / denom if denom else None


def topk(ids, scores, k):
    return sorted(ids, key=lambda item_id: (-float(scores.get(item_id, -1e9)), item_id))[:k]


def main():
    tasks = {row["base_query_id"]: row for row in read_jsonl(DATA / "p100_agent_tasks_memory_on.jsonl")}
    off_tasks = {row["base_query_id"]: row for row in read_jsonl(DATA / "p100_agent_tasks_memory_off.jsonl")}
    pools = {row["base_query_id"]: row for row in read_jsonl(DATA / "p100_bge_pools.jsonl")}
    meta = {row["item_id"]: row for row in read_jsonl(DATA / "p100_candidate_metadata.jsonl")}
    preds = {row["base_query_id"]: row for row in read_jsonl(RUN / "p100_ranker_gpu_predictions.jsonl")}
    judged = {row["base_query_id"]: row for row in read_jsonl(RUN / "p100_judged_gpu_predictions.jsonl")}
    esci_judged = {row["base_query_id"]: row for row in read_jsonl(DATA / "p100_judged_tasks.jsonl")}
    # Agent ranker-only metrics.
    ranker_metrics = {}
    per_query_rows = []
    for arm in ("existing", "natural", "hybrid"):
        top1 = top3 = top5 = 0
        top1_change = top3_change = 0
        positive_top1 = positive_n = 0
        negative_conflict_off = negative_conflict_on = negative_n = 0
        update_consistency = withdrawal_consistency = 0
        judged_queries = judged_candidates = judged_pos = hit_judged = 0
        evidence_items = total_items = 0
        top1_ids = []
        for bid, row in preds.items():
            task = tasks[bid]
            ids = row["candidate_ids"]
            on = row["modes"]["ON"][arm]
            off = row["modes"]["OFF"][arm]
            top1 += 1
            on_top1 = topk(ids, on, 1)
            on_top3 = topk(ids, on, 3)
            on_top5 = topk(ids, on, 5)
            top3 += 1
            top5 += 1
            top1_ids.append(on_top1[0] if on_top1 else None)
            top1_change += int(on_top1 != topk(ids, off, 1))
            top3_change += int(set(on_top3) != set(topk(ids, off, 3)))
            positive = [p["value"] for p in (row["modes"]["ON"].get("preferences") or []) if str(p.get("polarity")) in {"prefer", "positive"}]
            negative = [p["value"] for p in (row["modes"]["ON"].get("preferences") or []) if str(p.get("polarity")) in {"avoid", "negative"}]
            if positive:
                positive_n += 1
                attrs = set((meta.get(on_top1[0]) or {}).get("attributes") or [])
                positive_top1 += int(any(value in attrs for value in positive))
            if negative:
                negative_n += 1
                attrs_off = set((meta.get(topk(ids, off, 1)[0]) or {}).get("attributes") or [])
                attrs_on = set((meta.get(on_top1[0]) or {}).get("attributes") or [])
                negative_conflict_off += int(any(value in attrs_off for value in negative))
                negative_conflict_on += int(any(value in attrs_on for value in negative))
            if task["scenario_type"] == "preference_update":
                update_consistency += int(len(positive) > 0 and len(negative) == 0)
            if task["scenario_type"] == "preference_withdrawal":
                withdrawal_consistency += int(len(positive) == 0 and len(negative) == 0)
            # Exact ESCI product overlap only: partial judgments, not full Agent-pool Gold.
            jrow = esci_judged.get(bid) or {}
            judged_products = {p["product_id"]: p["esci_label"] for p in jrow.get("products") or []}
            overlap = [item_id for item_id in ids if item_id in judged_products]
            if overlap:
                judged_queries += 1
                judged_candidates += len(overlap)
                positives = [item_id for item_id in overlap if judged_products[item_id] in {"E", "S"}]
                judged_pos += len(positives)
                hit_judged += int(any(item_id in positives for item_id in on_top5))
            total_items += len(on_top5)
            evidence_items += sum(1 for item_id in on_top5 if (meta.get(item_id) or {}).get("evidence"))
            per_query_rows.append({"arm": arm, "base_query_id": bid, "query_id": row["query_id"],
                                   "scenario_type": task["scenario_type"], "top1_on": on_top1,
                                   "top3_on": on_top3, "top5_on": on_top5,
                                   "top1_off": topk(ids, off, 1), "top3_off": topk(ids, off, 3),
                                   "top1_changed": on_top1 != topk(ids, off, 1),
                                   "top1_score_on": float(on.get(on_top1[0], 0.0)) if on_top1 else None,
                                   "top1_score_off": float(off.get(topk(ids, off, 1)[0], 0.0)) if topk(ids, off, 1) else None,
                                   "positive_prefs": positive, "negative_prefs": negative,
                                   "judged_overlap": overlap, "query": row["query"]})
        n = len(preds)
        ranker_metrics[arm] = {
            "queries": n, "top1": top1, "top3": top3, "top5": top5,
            "top1_change_rate_on_off": round(top1_change / n, 6),
            "top3_set_change_rate_on_off": round(top3_change / n, 6),
            "positive_top1_match_rate": round(positive_top1 / positive_n, 6) if positive_n else None,
            "negative_conflict_rate_off": round(negative_conflict_off / negative_n, 6) if negative_n else None,
            "negative_conflict_rate_on": round(negative_conflict_on / negative_n, 6) if negative_n else None,
            "update_state_consistency": f"{update_consistency}/{sum(1 for t in tasks.values() if t['scenario_type']=='preference_update')}",
            "withdrawal_state_consistency": f"{withdrawal_consistency}/{sum(1 for t in tasks.values() if t['scenario_type']=='preference_withdrawal')}",
            "judged_queries_with_exact_overlap": judged_queries,
            "judged_candidates_in_agent_pool": judged_candidates,
            "judged_positive_candidates": judged_pos,
            "observed_relevant_hit_at5_partial": f"{hit_judged}/{judged_queries}",
            "evidence_supported_top5": f"{evidence_items}/{total_items}",
            "coverage_note": "Agent pool Hit@K is NOT computed for unjudged candidates; only exact ESCI overlaps.",
        }
    (RUN / "p100_ranker_metrics.json").write_text(json.dumps({
        "schema": "agentrecx.p100.ranker_metrics.v1",
        "arms": ranker_metrics,
        "memory_off_on": "same fixed pool; OFF empty memory; ON synthetic event-sourced writer/reader state",
        "provenance": "partial ESCI judged; no human memory preference Gold",
    }, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    # Judged relevance track.
    judged_metrics = {}
    per_judged = []
    for arm in ("shopranker_query_only",):
        ndcg3 = ndcg5 = ndcg10 = mrr = 0.0
        n = valid = 0
        pair_correct = pair_total = 0
        per_query = []
        for bid, row in judged.items():
            labels = row["labels"]
            scores = row["scores"]
            if len(labels) < 2 or len(set(labels.values())) < 2:
                continue
            n += 1
            ranked = sorted(labels, key=lambda item_id: (-scores.get(item_id, -1e9), item_id))
            gains = [GAIN.get(labels[item_id], 0.0) for item_id in ranked]
            ndcg3 += ndcg(gains, 3) or 0.0
            ndcg5 += ndcg(gains, 5) or 0.0
            ndcg10 += ndcg(gains, 10) or 0.0
            positive_rank = next((i + 1 for i, item_id in enumerate(ranked) if labels[item_id] in {"E", "S"}), None)
            mrr += 1.0 / positive_rank if positive_rank else 0.0
            valid += 1
            for i in range(len(ranked)):
                for j in range(i + 1, len(ranked)):
                    a, b = ranked[i], ranked[j]
                    ga, gb = GAIN.get(labels[a], 0.0), GAIN.get(labels[b], 0.0)
                    if ga == gb:
                        continue
                    pair_total += 1
                    pair_correct += int(ga > gb)
            per_query.append({"base_query_id": bid, "query": row["query"], "n_judged": len(labels),
                              "ndcg@3": ndcg(gains, 3), "ndcg@5": ndcg(gains, 5),
                              "ndcg@10": ndcg(gains, 10), "mrr": (1.0 / positive_rank) if positive_rank else 0.0,
                              "ranking": ranked[:10]})
        judged_metrics[arm] = {"valid_queries": valid, "n_queries": n,
                               "ndcg@3": round(ndcg3 / valid, 6) if valid else None,
                               "ndcg@5": round(ndcg5 / valid, 6) if valid else None,
                               "ndcg@10": round(ndcg10 / valid, 6) if valid else None,
                               "mrr_positive_threshold_E_or_S": round(mrr / valid, 6) if valid else None,
                               "pairwise_accuracy": round(pair_correct / pair_total, 6) if pair_total else None,
                               "pairwise_pairs": pair_total}
        per_judged.extend(per_query)
    # Bootstrap CI for nDCG@10.
    values = [row["ndcg@10"] for row in per_judged if row["ndcg@10"] is not None]
    if values:
        rng = random.Random(SEED)
        boots = []
        for _ in range(1000):
            sample = [values[rng.randrange(len(values))] for _ in range(len(values))]
            boots.append(sum(sample) / len(sample))
        boots.sort()
        judged_metrics["shopranker_query_only"]["ndcg@10_ci95"] = [round(boots[int(0.025 * len(boots))], 6), round(boots[int(0.975 * len(boots)) - 1], 6)]
    (RUN / "p100_judged_metrics.json").write_text(json.dumps({
        "schema": "agentrecx.p100.judged_metrics.v1",
        "gold_source": "official ESCI Task 1 E/S/C/I human labels",
        "gain_mapping": "official gain E=1.0 S=0.1 C=0.01 I=0",
        "candidate_source": "official ESCI judged products (NOT Original BGE Top-20)",
        "arms": judged_metrics,
        "not_comparable": {"existing_scorer": "ESCI judged products do not carry the catalogue structured attribute schema; no fabricated attributes",
                            "natural_memory": "no historical memory in judged track; query-only is the defined non-personalized comparison",
                            "hybrid": "requires structured memory features not defined for this judged query set"},
        "per_query": per_judged,
    }, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    # CSV per-query comparison.
    with (RUN / "p100_per_query_comparison.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["arm", "base_query_id", "query_id", "query", "scenario_type",
                                                    "top1_on", "top3_on", "top5_on", "top1_off", "top3_off", "top1_changed",
                                                    "top1_score_on", "top1_score_off", "positive_prefs", "negative_prefs"])
        writer.writeheader()
        for row in per_query_rows:
            out_row = {k: row.get(k) for k in writer.fieldnames}
            out_row.update({"top1_on": json.dumps(row["top1_on"]), "top3_on": json.dumps(row["top3_on"]),
                            "top5_on": json.dumps(row["top5_on"]), "top1_off": json.dumps(row["top1_off"]),
                            "top3_off": json.dumps(row["top3_off"]),
                            "positive_prefs": json.dumps(row["positive_prefs"]), "negative_prefs": json.dumps(row["negative_prefs"])})
            writer.writerow(out_row)
    # Memory OFF/ON summary.
    offon = {}
    for arm in ("existing", "natural", "hybrid"):
        rows = [r for r in per_query_rows if r["arm"] == arm]
        offon[arm] = {"queries": len(rows), "top1_changed": sum(1 for r in rows if r["top1_changed"]),
                      "top1_change_rate": round(sum(1 for r in rows if r["top1_changed"]) / len(rows), 6)}
    (RUN / "p100_memory_off_on.json").write_text(json.dumps({"schema": "agentrecx.p100.memory_off_on.v1", "arms": offon,
                                                               "note": "same query/pool; OFF empty memory; ON fixed synthetic event sequence"}, indent=1) + "\n", encoding="utf-8")
    # Label coverage.
    agent_judged = sum(1 for r in preds.values() if any(item_id in {p["product_id"] for p in (esci_judged.get(r["base_query_id"]) or {}).get("products", [])} for item_id in r["candidate_ids"]))
    unjudged = sum(1 for r in preds.values() for item_id in r["candidate_ids"] if item_id not in {p["product_id"] for p in (esci_judged.get(r["base_query_id"]) or {}).get("products", [])})
    total = sum(len(r["candidate_ids"]) for r in preds.values())
    coverage = {"schema": "agentrecx.p100.label_coverage.v1",
                "agent_queries": len(preds), "agent_candidate_rows": total,
                "agent_queries_with_exact_esci_judgment": agent_judged,
                "agent_unjudged_candidate_rows": unjudged,
                "agent_unjudged_fraction": round(unjudged / total, 6),
                "judged_queries": len(judged), "judged_products": sum(len(r["labels"]) for r in judged.values()),
                "judged_queries_with_valid_groups": judged_metrics["shopranker_query_only"]["valid_queries"],
                "human_memory_preference_gold": 0,
                "human_relevance_labels": "official ESCI labels on judged products only",
                "note": "Judged pool is NOT Original BGE Top-20; Agent unjudged products are never treated as negatives."}
    (DATA / "p100_label_coverage.json").write_text(json.dumps(coverage, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"ranker": ranker_metrics, "judged": judged_metrics["shopranker_query_only"], "memory_off_on": offon, "coverage": coverage}, indent=1)[:8000])


if __name__ == "__main__":
    main()

"""P100 GPU ShopRanker scores + CPU Existing/Hybrid ranks on fixed BGE pools."""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from experiments.agent_native_ranking.memory_adapter import candidate_features
from experiments.p14b1.diagnose import candidate_rows
from experiments.p14c0.electronics_executor import canonical_prefs, prepare_hard
from experiments.p14c0.run_dev import _memory_read
from experiments.p14e.catalogue import ElectronicsCatalogue
from recommendation.preference.scorer import PreferenceAwareScorer

REPO = Path("/root/autodl-tmp/AgentRec-X-phase2")
DATA = REPO / "data/p100_extended_benchmark"
OUT = REPO / "runs/p100_extended_evaluation"
MODEL = "/root/autodl-tmp/models/zoowork-shopranker-0.6b"
PREFIX = ("<|im_start|>system\nJudge whether the Document meets the requirements based on "
          "the Query and the Instruct provided. Note that the answer can only be \"yes\" "
          "or \"no\".<|im_end|>\n<|im_start|>user\n")
SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
INSTRUCTION = "Given a web search query, retrieve relevant passages that answer the query"


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def product_text(item):
    attrs = item.get("attributes") or []
    evidence = [f"{e.get('attribute')}={str(e.get('evidence'))[:80]}" for e in item.get("evidence") or []]
    text = f"title: {item.get('title') or 'unknown'} | product_type: electronics | verified_attributes: {', '.join(attrs) if attrs else 'none'}"
    if evidence:
        text += f" | evidence: {'; '.join(evidence[:4])}"
    return text


def natural_query(query, positive, negative):
    parts = []
    if positive:
        parts.append("prefer " + ", ".join(positive))
    if negative:
        parts.append("avoid " + ", ".join(negative))
    return f"{query} User preferences: {'; '.join(parts)}." if parts else query


class Scorer:
    def __init__(self):
        self.tok = AutoTokenizer.from_pretrained(MODEL, padding_side="left", local_files_only=True)
        base = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, local_files_only=True)
        self.model = PeftModel.from_pretrained(base, MODEL).cuda().eval()
        self.yes = self.tok.convert_tokens_to_ids("yes")
        self.no = self.tok.convert_tokens_to_ids("no")

    @torch.no_grad()
    def score(self, query, products):
        texts = [PREFIX + f"<Instruct>: {INSTRUCTION}\n<Query>: {query}\n<Document>: {product}" + SUFFIX for product in products]
        enc = self.tok(texts, return_tensors="pt", padding=True, truncation=True, max_length=1536, add_special_tokens=False).to(self.model.device)
        logits = self.model(**enc).logits[:, -1, :]
        return torch.softmax(logits[:, [self.no, self.yes]].float(), dim=-1)[:, 1].cpu().tolist()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    tasks_on = {row["base_query_id"]: row for row in read_jsonl(DATA / "p100_agent_tasks_memory_on.jsonl")}
    tasks_off = {row["base_query_id"]: row for row in read_jsonl(DATA / "p100_agent_tasks_memory_off.jsonl")}
    pools = {row["base_query_id"]: row for row in read_jsonl(DATA / "p100_bge_pools.jsonl")}
    meta = {row["item_id"]: row for row in read_jsonl(DATA / "p100_candidate_metadata.jsonl")}
    catalogue = ElectronicsCatalogue.load(metadata_mode="enriched")
    scorer = Scorer()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    rows = []
    for bid, pool in pools.items():
        ids = pool["candidate_pool_ids"]
        texts = [product_text(meta[item_id]) for item_id in ids]
        q_scores = dict(zip(ids, scorer.score(pool["query"], texts)))
        records = {}
        for mode, task in (("OFF", tasks_off[bid]), ("ON", tasks_on[bid])):
            prefs, _ = _memory_read(catalogue, task)
            positive = [str(p["value"]) for p in prefs if str(p.get("polarity")) in {"prefer", "positive"}]
            negative = [str(p["value"]) for p in prefs if str(p.get("polarity")) in {"avoid", "negative"}]
            n_scores = dict(zip(ids, scorer.score(natural_query(pool["query"], positive, negative), texts)))
            # Existing Scorer scores.
            base_task = {"task_id": bid, "current_query": pool["query"], "candidate_ids": ids, "search_scores": {}}
            base_rows = candidate_rows(base_task, catalogue, PreferenceAwareScorer(), prefs,
                                       pool.get("hard_requirements") or [])
            existing = {r["product_ref"]["item_id"]: float(r["final_score"]) for r in base_rows}
            z = {item_id: (q_scores[item_id] - sum(q_scores.values()) / len(q_scores)) /
                 (sum((v - sum(q_scores.values()) / len(q_scores)) ** 2 for v in q_scores.values()) / len(q_scores)) ** 0.5
                 for item_id in ids}
            memory = {"positive": [{"value": v} for v in positive], "negative": [{"value": v} for v in negative]}
            feats = {item_id: candidate_features(meta[item_id], memory) for item_id in ids}
            hybrid = {item_id: z[item_id] + 0.5 * float(feats[item_id].get("positive_match_rate", 0.0))
                      - 0.75 * float(feats[item_id].get("negative_conflict_rate", 0.0)) for item_id in ids}
            records[mode] = {"query_only": q_scores, "natural": n_scores, "existing": existing, "hybrid": hybrid,
                             "preferences": prefs, "positive": positive, "negative": negative,
                             "adapter_features": {item_id: {k: feats[item_id].get(k) for k in ("positive_match_rate", "negative_conflict_rate", "unknown_rate")} for item_id in ids}}
        rows.append({"base_query_id": bid, "query_id": pool["query_id"], "query": pool["query"],
                     "pool_hash": pool["candidate_pool_hash"], "candidate_ids": ids,
                     "bge_ranks": pool["bge_ranks"], "modes": records})
    # Judged track query-only scores (official ESCI judged products, no structured memory).
    judged = read_jsonl(DATA / "p100_judged_tasks.jsonl")
    judged_rows = []
    for task in judged:
        ids = [p["product_id"] for p in task["products"]]
        texts = []
        for p in task["products"]:
            texts.append(f"title: {p.get('title') or 'unknown'} | product_type: electronics | brand: {p.get('brand') or ''}")
        scored = dict(zip(ids, scorer.score(task["query"], texts))) if ids else {}
        judged_rows.append({"base_query_id": task["base_query_id"], "query_id": task["query_id"], "query": task["query"],
                            "candidate_source": "official_esci_judged_products", "scores": scored,
                            "labels": {p["product_id"]: p["esci_label"] for p in task["products"]}})
    (OUT / "p100_ranker_gpu_predictions.jsonl").write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8")
    (OUT / "p100_judged_gpu_predictions.jsonl").write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in judged_rows) + "\n", encoding="utf-8")
    runtime = {"schema": "agentrecx.p100.ranker_gpu_runtime.v1", "wall_seconds": round(time.perf_counter() - started, 3),
               "peak_memory_mb": round(torch.cuda.max_memory_allocated() / 1024 ** 2, 2), "queries": len(rows),
               "candidate_rows": sum(len(r["candidate_ids"]) for r in rows), "judged_queries": len(judged_rows),
               "model": MODEL, "serialization": "title|product_type|verified_attributes|evidence", "prompt_sha256": hashlib.sha256(PREFIX.encode()).hexdigest()}
    (OUT / "p100_ranker_gpu_runtime.json").write_text(json.dumps(runtime, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(runtime, indent=1))


if __name__ == "__main__":
    main()

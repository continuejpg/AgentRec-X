"""ShopRanker-0.6B extended shadow on Memory Ranking Benchmark v2."""

from __future__ import annotations

import json
import statistics
import time
from contextlib import nullcontext
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "/root/autodl-tmp/models/zoowork-shopranker-0.6b"
BENCH = Path("/root/autodl-tmp/AgentRec-X-phase2/data/memory_ranking_benchmark_v2")
OUT = Path("/root/autodl-tmp/AgentRec-X-phase2/runs/memory_ranking_benchmark_v2")
PREFIX = ("<|im_start|>system\nJudge whether the Document meets the requirements based on "
          "the Query and the Instruct provided. Note that the answer can only be \"yes\" "
          "or \"no\".<|im_end|>\n<|im_start|>user\n")
SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
INSTRUCTION = "Given a web search query, retrieve relevant passages that answer the query"
ATTRS = ("over_ear", "on_ear", "in_ear", "wireless", "wired", "bluetooth", "microphone",
         "rechargeable", "noise_cancelling", "waterproof", "foldable", "portable", "gaming",
         "mechanical", "silent", "ergonomic", "touch", "rgb", "led", "usb", "hdmi", "4k", "remote")


def product_text(item: dict) -> str:
    attrs = list(item.get("attributes") or [])
    categories = item.get("categories") or []
    category = item.get("leaf_category") or item.get("main_category") or (categories[-1] if categories else "unknown")
    evidence = list(item.get("evidence") or [])[:3]
    unknown = [a for a in ATTRS if a not in set(attrs)]
    text = f"title: {item.get('title') or 'unknown'} | category: {category} | verified_attributes: {', '.join(attrs) if attrs else 'none'}"
    if evidence:
        text += f" | evidence: {'; '.join(evidence)}"
    if unknown:
        text += f" | unknown_attributes: {', '.join(unknown)}"
    return text


def memory_text(memory: dict) -> str:
    positive = [str(item.get("value")) for item in memory.get("positive") or [] if item.get("value")]
    negative = [str(item.get("value")) for item in memory.get("negative") or [] if item.get("value")]
    parts = []
    if positive:
        parts.append("prefer " + ", ".join(positive))
    if negative:
        parts.append("avoid " + ", ".join(negative))
    return "; ".join(parts)


def query_with_memory(query: str, memory: dict) -> str:
    text = memory_text(memory)
    return f"{query} User preferences: {text}." if text else query


class Scorer:
    def __init__(self):
        self.tok = AutoTokenizer.from_pretrained(MODEL, padding_side="left", local_files_only=True)
        self.base = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, local_files_only=True)
        self.model = PeftModel.from_pretrained(self.base, MODEL).cuda().eval()
        self.yes = self.tok.convert_tokens_to_ids("yes")
        self.no = self.tok.convert_tokens_to_ids("no")

    @torch.no_grad()
    def score(self, query: str, products: list[str]) -> list[float]:
        texts = [PREFIX + f"<Instruct>: {INSTRUCTION}\n<Query>: {query}\n<Document>: {product}" + SUFFIX for product in products]
        enc = self.tok(texts, return_tensors="pt", padding=True, truncation=True, max_length=4096, add_special_tokens=False).to(self.model.device)
        logits = self.model(**enc).logits[:, -1, :]
        probs = torch.softmax(logits[:, [self.no, self.yes]].float(), dim=-1)[:, 1]
        return probs.cpu().tolist()


def rank(ids: list[str], scores: dict[str, float]) -> list[str]:
    return sorted(ids, key=lambda item_id: (-scores[item_id], item_id))


def main():
    scenarios = [json.loads(line) for line in (BENCH / "scenarios.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    metadata = {row["item_id"]: row for row in (json.loads(line) for line in (BENCH / "candidate_metadata.jsonl").read_text(encoding="utf-8").splitlines() if line.strip())}
    pairs = {row["scenario_id"]: row for row in (json.loads(line) for line in (BENCH / "pair_tasks.jsonl").read_text(encoding="utf-8").splitlines() if line.strip())}
    scorer = Scorer()
    torch.cuda.reset_peak_memory_stats()
    scorer.score("warm up", ["title: warm up | category: test | verified_attributes: none"])
    per_scenario = []
    latencies = []
    for scenario in scenarios:
        ids = list(scenario["candidate_pool_ids"])
        items = [metadata[item_id] for item_id in ids]
        products = [product_text(item) for item in items]
        started = time.perf_counter()
        scores_b = dict(zip(ids, scorer.score(scenario["query"], products)))
        elapsed = (time.perf_counter() - started) * 1000.0
        latencies.append(elapsed)
        scores_c = dict(zip(ids, scorer.score(query_with_memory(scenario["query"], scenario["memory_after"]), products)))
        ranking_b = rank(ids, scores_b)
        ranking_c = rank(ids, scores_c)
        pair = pairs.get(scenario["scenario_id"])
        pair_detail = None
        if pair:
            a, b = pair["candidate_a_id"], pair["candidate_b_id"]
            pair_detail = {
                "candidate_a": a,
                "candidate_b": b,
                "label": pair["choice"],
                "label_provenance": pair["label_provenance"],
                "query_only_a_rank": ranking_b.index(a) + 1,
                "query_only_b_rank": ranking_b.index(b) + 1,
                "memory_a_rank": ranking_c.index(a) + 1,
                "memory_b_rank": ranking_c.index(b) + 1,
                "query_only_a_score": round(scores_b[a], 6),
                "query_only_b_score": round(scores_b[b], 6),
                "memory_a_score": round(scores_c[a], 6),
                "memory_b_score": round(scores_c[b], 6),
                "query_only_prefers_a": scores_b[a] > scores_b[b],
                "memory_prefers_a": scores_c[a] > scores_c[b],
            }
        per_scenario.append({
            "scenario_id": scenario["scenario_id"],
            "base_query_id": scenario["base_query_id"],
            "scenario_type": scenario["scenario_type"],
            "split": scenario["split"],
            "requires_clarification": scenario["requires_clarification"],
            "memory_after": scenario["memory_after"],
            "query_only_ranking": ranking_b,
            "memory_ranking": ranking_c,
            "query_only_scores": {item_id: round(scores_b[item_id], 6) for item_id in ids},
            "memory_scores": {item_id: round(scores_c[item_id], 6) for item_id in ids},
            "pair": pair_detail,
        })
    OUT.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": "agentrecx.memory_ranking_benchmark_v2.shopranker_shadow.v1",
        "model": "srpone/zoowork-shopranker-0.6b",
        "mode": "frozen; official prompt; official yes/no logit scoring; no tuning",
        "scenarios": len(per_scenario),
        "pair_tasks": sum(1 for row in per_scenario if row["pair"]),
        "latency_ms": {
            "mean": round(statistics.mean(latencies), 3),
            "p50": round(statistics.median(latencies), 3),
            "max": round(max(latencies), 3),
            "scope": "20 candidates, batch call, Query Only timing; GPU/Python overhead",
        },
        "gpu_peak_memory_mb": round(torch.cuda.max_memory_allocated() / 1024 ** 2, 2),
        "per_scenario": per_scenario,
    }
    (OUT / "shopranker_shadow.json").write_text(json.dumps(payload, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({key: payload[key] for key in ("scenarios", "pair_tasks", "latency_ms", "gpu_peak_memory_mb")}, indent=1))


if __name__ == "__main__":
    main()

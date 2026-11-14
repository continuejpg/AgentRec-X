"""ShopRanker-0.6B zero-shot fixed-pool benchmark on P20.

Uses the official model card prompt and yes/no logit scoring.  The model is frozen
(LoRA adapter loaded exactly as published).  Memory is rendered as a concise suffix of
the official Query input; no scoring-head modification.

Run remotely on the AutoDL GPU::

    /root/autodl-tmp/agentrecx-sft/bin/python -m experiments.shopranker.benchmark
"""

from __future__ import annotations

import json
import statistics
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "/root/autodl-tmp/models/zoowork-shopranker-0.6b"
DATASET = Path("/root/autodl-tmp/AgentRec-X-phase2/data/p20_shadow/ranking_dataset.jsonl")
PER_TASK = Path("/root/autodl-tmp/AgentRec-X-phase2/data/p20_shadow/per_task_results.jsonl")
COUNTERFACTUAL = Path("/root/autodl-tmp/AgentRec-X-phase2/data/p20_shadow/counterfactual_cases.jsonl")
OUT = Path("/root/autodl-tmp/AgentRec-X-phase2/runs/shopranker_v1")

PREFIX = ("<|im_start|>system\nJudge whether the Document meets the requirements based on "
          "the Query and the Instruct provided. Note that the answer can only be \"yes\" "
          "or \"no\".<|im_end|>\n<|im_start|>user\n")
SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
INSTRUCTION = "Given a web search query, retrieve relevant passages that answer the query"
ATTRS = ("over_ear", "on_ear", "in_ear", "wireless", "wired", "bluetooth", "microphone",
         "rechargeable", "noise_cancelling", "waterproof", "foldable", "portable", "gaming",
         "mechanical", "silent", "ergonomic", "touch", "rgb", "led", "usb", "hdmi", "4k", "remote")


def _product_text(item: dict) -> str:
    attrs = sorted((item.get("attributes") or {}).keys())
    scorer = item.get("scorer") or {}
    product_type = scorer.get("candidate_product_type") or scorer.get("expected_product_type") or "unknown"
    evidence_parts = []
    for name in attrs[:4]:
        fact = (item.get("attributes") or {}).get(name) or {}
        evidence = str(fact.get("evidence") or "")[:80]
        if evidence:
            evidence_parts.append(f"{name}={evidence}")
    unknown = [name for name in ATTRS if name not in attrs]
    text = (
        f"title: {item.get('title') or 'unknown'} | product_type: {product_type} | "
        f"verified_attributes: {', '.join(attrs) if attrs else 'none'}"
    )
    if evidence_parts:
        text += f" | evidence: {'; '.join(evidence_parts)}"
    if unknown:
        text += f" | unknown_attributes: {', '.join(unknown)}"
    return text


def _memory_text(preferences: list[dict]) -> str:
    positive = []
    negative = []
    for preference in preferences:
        value = str(preference.get("value") or "")
        if not value:
            continue
        polarity = str(preference.get("polarity") or "positive").lower()
        (negative if polarity in {"avoid", "negative", "dislike"} else positive).append(value)
    parts = []
    if positive:
        parts.append("prefer " + ", ".join(positive))
    if negative:
        parts.append("avoid " + ", ".join(negative))
    return "; ".join(parts)


def _query_with_memory(query: str, preferences: list[dict]) -> str:
    memory = _memory_text(preferences)
    return f"{query} User preferences: {memory}." if memory else query


def _hits(ranking: list[str], acceptable: set[str]) -> dict:
    first = next((index + 1 for index, item in enumerate(ranking) if item in acceptable), None)
    return {
        "top1": ranking[:1],
        "top3": ranking[:3],
        "ranking": ranking,
        "hit@1": bool(first is not None and first <= 1),
        "hit@3": bool(first is not None and first <= 3),
        "hit@5": bool(first is not None and first <= 5),
        "first_acceptable_rank": first,
        "reciprocal_rank@10": round(1.0 / first, 6) if first and first <= 10 else 0.0,
    }


def _rank_by_scores(ids: list[str], scores: list[float]) -> list[str]:
    order = sorted(range(len(ids)), key=lambda index: (-float(scores[index]), ids[index]))
    return [ids[index] for index in order]


def _null_context():
    return nullcontext()


class Scorer:
    def __init__(self) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL, padding_side="left", local_files_only=True)
        self.base = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, local_files_only=True)
        self.shopranker = PeftModel.from_pretrained(self.base, MODEL).cuda().eval()
        self.yes = self.tokenizer.convert_tokens_to_ids("yes")
        self.no = self.tokenizer.convert_tokens_to_ids("no")

    @torch.no_grad()
    def score(self, query: str, products: list[str], use_shopranker: bool = True) -> list[float]:
        model = self.shopranker
        texts = [
            PREFIX + f"<Instruct>: {INSTRUCTION}\n<Query>: {query}\n<Document>: {product}" + SUFFIX
            for product in products
        ]
        encoded = self.tokenizer(
            texts, return_tensors="pt", padding=True, truncation=True,
            max_length=4096, add_special_tokens=False,
        ).to(model.device)
        context = model.disable_adapter() if not use_shopranker else _null_context()
        with context:
            logits = model(**encoded).logits[:, -1, :]
        probabilities = torch.softmax(logits[:, [self.no, self.yes]].float(), dim=-1)[:, 1]
        return probabilities.cpu().tolist()


def _score_pool(scorer: Scorer, query: str, pool: list[dict], eligible_ids: set[str], shopranker: bool = True):
    all_ids = [item["item_id"] for item in pool]
    products = [_product_text(item) for item in pool]
    scores = scorer.score(query, products, use_shopranker=shopranker)
    by_id = dict(zip(all_ids, scores))
    eligible = [item_id for item_id in all_ids if item_id in eligible_ids]
    ranking = _rank_by_scores(eligible, [by_id[item_id] for item_id in eligible])
    return ranking, by_id


def main() -> int:
    if not torch.cuda.is_available():
        raise RuntimeError("ShopRanker benchmark expects GPU")
    rows = [json.loads(line) for line in DATASET.read_text(encoding="utf-8").splitlines() if line.strip()]
    rows = [row for row in rows if row["expected_decision"] == "recommend"]
    phase1 = {
        row["task_id"]: row
        for row in (json.loads(line) for line in PER_TASK.read_text(encoding="utf-8").splitlines() if line.strip())
    }
    scorer = Scorer()
    torch.cuda.reset_peak_memory_stats()
    # warm up after load
    scorer.score("warm up query", ["title: warm up product | product_type: test"], use_shopranker=True)

    per_task = []
    latencies = []
    for row in rows:
        pool = row["candidate_pool"]
        task_id = row["task_id"]
        acceptable = set(row.get("acceptable_ids_in_pool") or [])
        eligible = set(phase1[task_id]["eligible_candidate_ids"])
        preferences = row.get("historical_memory") or []
        query_memory = _query_with_memory(row["query"], preferences)
        started = time.perf_counter()
        ranking_b, scores_b = _score_pool(scorer, row["query"], pool, eligible, shopranker=True)
        latency = (time.perf_counter() - started) * 1000.0
        latencies.append(latency)
        ranking_c, scores_c = _score_pool(scorer, query_memory, pool, eligible, shopranker=True)
        ranking_d, scores_d = _score_pool(scorer, query_memory, pool, eligible, shopranker=False)
        existing = [item_id for item_id in phase1[task_id]["rankings"]["C_scorer_memory_on"] if item_id in eligible]
        item = {
            "task_id": task_id,
            "query": row["query"],
            "query_with_memory": query_memory,
            "active_memory": preferences,
            "existing_scorer_memory_on": _hits(existing, acceptable),
            "shopranker_query_only": _hits(ranking_b, acceptable),
            "shopranker_query_memory": _hits(ranking_c, acceptable),
            "qwen_base_query_memory": _hits(ranking_d, acceptable),
        }
        per_task.append(item)

    # 0051 detail
    detail = next(item for item in per_task if item["task_id"] == "p14e2_lite_0051")
    row_0051 = next(row for row in rows if row["task_id"] == "p14e2_lite_0051")
    eligible_0051 = set(phase1["p14e2_lite_0051"]["eligible_candidate_ids"])
    ranking_b, scores_b = _score_pool(scorer, row_0051["query"], row_0051["candidate_pool"], eligible_0051)
    ranking_c, scores_c = _score_pool(
        scorer, _query_with_memory(row_0051["query"], row_0051.get("historical_memory") or []),
        row_0051["candidate_pool"], eligible_0051,
    )
    detail["0051"] = {
        "query_only_ranking": ranking_b,
        "query_memory_ranking": ranking_c,
        "query_only_scores": {item_id: round(scores_b[item_id], 6) for item_id in eligible_0051},
        "query_memory_scores": {item_id: round(scores_c[item_id], 6) for item_id in eligible_0051},
        "acceptable_ids": sorted(set(row_0051.get("acceptable_ids_in_pool") or [])),
    }

    # Counterfactual cases: same candidate pools, baseline vs mutated Event-Sourced state.
    cases = [json.loads(line) for line in COUNTERFACTUAL.read_text(encoding="utf-8").splitlines() if line.strip()]
    ranking_rows = {row["task_id"]: row for row in rows}
    counterfactuals = []
    for case in cases:
        pool_row = ranking_rows.get(case["pool_id"])
        if pool_row is None:
            continue
        pool = pool_row["candidate_pool"]
        eligible = set(case.get("eligible_candidate_ids") or [])
        case_query = case["query"]
        baseline_prefs = (case.get("baseline_memory_state") or {}).get("active_preferences") or []
        mutated_prefs = (case.get("mutated_memory_state") or {}).get("active_preferences") or []
        baseline_query = _query_with_memory(case_query, baseline_prefs)
        mutated_query = _query_with_memory(case_query, mutated_prefs)
        baseline_ranking, _ = _score_pool(scorer, baseline_query, pool, eligible)
        mutated_ranking, mutated_scores = _score_pool(scorer, mutated_query, pool, eligible)
        # Candidate rank changes for attributes introduced by the mutation.
        baseline_attrs = {str(pref.get("value")) for pref in baseline_prefs}
        mutated_attrs = {str(pref.get("value")) for pref in mutated_prefs}
        added_attrs = sorted(mutated_attrs - baseline_attrs)
        removed_attrs = sorted(baseline_attrs - mutated_attrs)
        attr_candidates = {}
        for attr in added_attrs + removed_attrs:
            ids = [
                item["item_id"] for item in pool
                if item["item_id"] in eligible and attr in (item.get("attributes") or {})
            ]
            attr_candidates[attr] = {
                "n_candidates": len(ids),
                "baseline_ranks": {item_id: baseline_ranking.index(item_id) + 1 for item_id in ids if item_id in baseline_ranking},
                "mutated_ranks": {item_id: mutated_ranking.index(item_id) + 1 for item_id in ids if item_id in mutated_ranking},
            }
        counterfactuals.append({
            "case_id": case["case_id"],
            "category": case["category"],
            "query": case_query,
            "baseline_active": baseline_prefs,
            "mutated_active": mutated_prefs,
            "added_attrs": added_attrs,
            "removed_attrs": removed_attrs,
            "baseline_top1": baseline_ranking[:1],
            "mutated_top1": mutated_ranking[:1],
            "top1_changed": baseline_ranking[:1] != mutated_ranking[:1],
            "top3_changed": set(baseline_ranking[:3]) != set(mutated_ranking[:3]),
            "full_ranking_changed": baseline_ranking != mutated_ranking,
            "attr_candidates": attr_candidates,
            "requires_clarification": bool(case.get("requires_clarification")),
            "top1_gold_allowed": not bool(case.get("requires_clarification")),
        })

    def aggregate(arm: str) -> dict:
        return {
            "tasks": len(per_task),
            "hit@1": sum(1 for item in per_task if item[arm]["hit@1"]),
            "hit@3": sum(1 for item in per_task if item[arm]["hit@3"]),
            "hit@5": sum(1 for item in per_task if item[arm]["hit@5"]),
            "mrr@10": round(statistics.mean(item[arm]["reciprocal_rank@10"] for item in per_task), 6),
        }

    report = {
        "schema": "agentrecx.shopranker.v1",
        "model": "srpone/zoowork-shopranker-0.6b",
        "model_mode": "frozen LoRA; official yes/no logit scoring; no fine-tuning",
        "data": {
            "tasks": len(per_task),
            "candidate_pool": "P20 Original BGE Top-20",
            "labels": "P20 provisional observed-acceptable; not used in training",
        },
        "aggregate": {
            "existing_scorer_memory_on": aggregate("existing_scorer_memory_on"),
            "shopranker_query_only": aggregate("shopranker_query_only"),
            "shopranker_query_memory": aggregate("shopranker_query_memory"),
            "qwen_base_query_memory": aggregate("qwen_base_query_memory"),
        },
        "latency_ms": {
            "mean": round(float(statistics.mean(latencies)), 3),
            "p50": round(float(statistics.median(latencies)), 3),
            "max": round(float(max(latencies)), 3),
            "scope": "ShopRanker Query+Memory, 20 candidates, batch call, GPU, Python overhead",
        },
        "gpu_peak_memory_mb": round(torch.cuda.max_memory_allocated() / (1024 ** 2), 2),
        "per_task": per_task,
        "counterfactual": {
            "cases": len(counterfactuals),
            "top1_changed": sum(1 for case in counterfactuals if case["top1_changed"]),
            "top3_changed": sum(1 for case in counterfactuals if case["top3_changed"]),
            "full_ranking_changed": sum(1 for case in counterfactuals if case["full_ranking_changed"]),
            "requires_clarification_cases": [
                case["case_id"] for case in counterfactuals if case["requires_clarification"]
            ],
            "details": counterfactuals,
        },
        "caveats": [
            "ShopRanker is a zero-shot general e-commerce preference reranker; it was not trained on AgentRec-X memory semantics.",
            "Event-Sourced preferences are rendered as a Query suffix; the model has no separate memory head.",
            "P20 labels are provisional and were not used for training, prompt tuning, or weight tuning.",
            "A high P20 hit rate would not prove production superiority.",
        ],
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "shopranker_results.json").write_text(json.dumps(report, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("aggregate", "latency_ms", "gpu_peak_memory_mb", "counterfactual")}, indent=1, ensure_ascii=False)[:8000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

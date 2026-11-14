"""GPU ShopRanker controlled memory + Hybrid Shadow for Phase 4.2."""

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
REPO = Path("/root/autodl-tmp/AgentRec-X-phase2")
BM = REPO / "data/memory_ranking_benchmark_v2"
P20 = REPO / "data/p20_shadow/ranking_dataset.jsonl"
P20_TASKS = REPO / "data/p20_shadow/per_task_results.jsonl"
CF = REPO / "data/p20_shadow/counterfactual_cases.jsonl"
ADAPTER = REPO / "runs/agent_native_ranking_phase42/memory_adapter_features.jsonl"
OUT = REPO / "runs/agent_native_ranking_phase42"
PREFIX = ("<|im_start|>system\nJudge whether the Document meets the requirements based on "
          "the Query and the Instruct provided. Note that the answer can only be \"yes\" "
          "or \"no\".<|im_end|>\n<|im_start|>user\n")
SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
INSTRUCTION = "Given a web search query, retrieve relevant passages that answer the query"
ATTRS = ("over_ear", "on_ear", "in_ear", "wireless", "wired", "bluetooth", "microphone",
         "rechargeable", "noise_cancelling", "waterproof", "foldable", "portable", "gaming",
         "mechanical", "silent", "ergonomic", "touch", "rgb", "led", "usb", "hdmi", "4k", "remote")
MEMORY_KEYS = ("positive_match_rate", "negative_conflict_rate", "positive_match_count",
               "negative_conflict_count", "active_preference_count", "preference_evidence_coverage")
HYBRID_WEIGHTS = {"semantic_weight": 1.0, "positive_match_rate": 0.5, "negative_conflict_rate": -0.75}


def read_jsonl(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def product_text(item: dict) -> str:
    raw_attrs = item.get("attributes") or {}
    if isinstance(raw_attrs, dict):
        attrs = list(raw_attrs.keys())
        evidence_parts = []
        for name, fact in list(raw_attrs.items())[:4]:
            if isinstance(fact, dict) and fact.get("evidence"):
                evidence_parts.append(f"{name}={str(fact['evidence'])[:80]}")
    else:
        attrs = list(raw_attrs)
        evidence_parts = [str(value)[:80] for value in (item.get("evidence") or [])[:3]]
    scorer = item.get("scorer") or {}
    product_type = (
        scorer.get("candidate_product_type")
        or scorer.get("expected_product_type")
        or item.get("leaf_category")
        or item.get("main_category")
        or "unknown"
    )
    categories = item.get("categories") or []
    if product_type == "unknown" and categories:
        product_type = categories[-1]
    unknown = [a for a in ATTRS if a not in set(attrs)]
    text = f"title: {item.get('title') or 'unknown'} | product_type: {product_type} | verified_attributes: {', '.join(attrs) if attrs else 'none'}"
    if evidence_parts:
        text += f" | evidence: {'; '.join(evidence_parts)}"
    if unknown:
        text += f" | unknown_attributes: {', '.join(unknown)}"
    return text


def preferences(memory) -> tuple[list[str], list[str]]:
    positive, negative = [], []
    if isinstance(memory, dict):
        positive = [str(entry.get("value")) for entry in memory.get("positive") or [] if entry.get("value")]
        negative = [str(entry.get("value")) for entry in memory.get("negative") or [] if entry.get("value")]
    elif isinstance(memory, list):
        for entry in memory:
            value = str(entry.get("value") or "")
            if not value:
                continue
            if str(entry.get("polarity") or "positive").lower() in {"negative", "avoid", "dislike"}:
                negative.append(value)
            else:
                positive.append(value)
    return positive, negative


def natural_query(query: str, memory) -> str:
    positive, negative = preferences(memory)
    parts = []
    if positive:
        parts.append("prefer " + ", ".join(positive))
    if negative:
        parts.append("avoid " + ", ".join(negative))
    return f"{query} User preferences: {'; '.join(parts)}." if parts else query


def structured_query(query: str, memory) -> str:
    positive, negative = preferences(memory)
    lines = ["Current Shopping Request:", query, "", "Active Long-Term Preferences:", "Positive:"]
    lines.extend([f"- {value}: preferred" for value in positive] or ["- none"])
    lines.append("Negative:")
    lines.extend([f"- {value}: avoided" for value in negative] or ["- none"])
    return "\n".join(lines)


class Scorer:
    def __init__(self):
        self.tok = AutoTokenizer.from_pretrained(MODEL, padding_side="left", local_files_only=True)
        base = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, local_files_only=True)
        self.model = PeftModel.from_pretrained(base, MODEL).cuda().eval()
        self.yes = self.tok.convert_tokens_to_ids("yes")
        self.no = self.tok.convert_tokens_to_ids("no")

    @torch.no_grad()
    def score(self, query: str, products: list[str]) -> list[float]:
        texts = [PREFIX + f"<Instruct>: {INSTRUCTION}\n<Query>: {query}\n<Document>: {product}" + SUFFIX for product in products]
        enc = self.tok(texts, return_tensors="pt", padding=True, truncation=True, max_length=1536, add_special_tokens=False).to(self.model.device)
        logits = self.model(**enc).logits[:, -1, :]
        return torch.softmax(logits[:, [self.no, self.yes]].float(), dim=-1)[:, 1].cpu().tolist()


def rank(ids: list[str], scores: dict[str, float]) -> list[str]:
    return sorted(ids, key=lambda item_id: (-scores[item_id], item_id))


def read_adapter():
    rows = {}
    for row in read_jsonl(ADAPTER):
        rows[(row["source"], row["group_id"], row["candidate_id"], row["memory_snapshot_id"])] = row["features"]
    return rows


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    scenarios = read_jsonl(BM / "scenarios.jsonl")
    metadata = {row["item_id"]: row for row in read_jsonl(BM / "candidate_metadata.jsonl")}
    pairs = {row["scenario_id"]: row for row in read_jsonl(BM / "pair_tasks.jsonl")}
    p20_rows = [row for row in read_jsonl(P20) if row.get("expected_decision") == "recommend"]
    p20_tasks = {row["task_id"]: row for row in read_jsonl(P20_TASKS)}
    cf_cases = read_jsonl(CF)
    adapter = read_adapter()
    scorer = Scorer()
    torch.cuda.reset_peak_memory_stats()
    scorer.score("warmup", ["title: warmup | category: test | verified_attributes: wireless"])
    latencies = {"query": [], "natural": [], "structured": [], "hybrid": []}
    bm_rows = []
    pair_accuracy = {arm: {"correct": 0, "n": 0} for arm in ("query", "natural", "structured", "hybrid")}
    for scenario in scenarios:
        ids = list(scenario["candidate_pool_ids"])
        items = [metadata[item_id] for item_id in ids]
        products = [product_text(item) for item in items]
        memory = scenario["memory_after"]
        queries = {"query": scenario["query"], "natural": natural_query(scenario["query"], memory), "structured": structured_query(scenario["query"], memory)}
        scores = {}
        for arm, query in queries.items():
            started = time.perf_counter()
            scores[arm] = dict(zip(ids, scorer.score(query, products)))
            latencies[arm].append((time.perf_counter() - started) * 1000)
        # GPU hybrid over semantic z-scores and Memory Adapter features.
        semantic = torch.tensor([scores["query"][item_id] for item_id in ids], device="cuda", dtype=torch.float32)
        semantic_z = (semantic - semantic.mean()) / (semantic.std() + 1e-6)
        memory_values = []
        for item_id in ids:
            features = adapter.get(("benchmark_v2", scenario["scenario_id"], item_id, scenario["scenario_id"])) or {}
            memory_values.append(
                HYBRID_WEIGHTS["positive_match_rate"] * float(features.get("positive_match_rate", 0.0))
                + HYBRID_WEIGHTS["negative_conflict_rate"] * float(features.get("negative_conflict_rate", 0.0))
            )
        memory_t = torch.tensor(memory_values, device="cuda", dtype=torch.float32)
        started = time.perf_counter()
        hybrid = semantic_z + memory_t
        _ = torch.argsort(hybrid, descending=True)
        hybrid_tensor_latency = (time.perf_counter() - started) * 1000.0
        latencies["hybrid"].append(hybrid_tensor_latency + latencies["query"][-1])
        scores["hybrid"] = {item_id: float(hybrid[index].detach().cpu()) for index, item_id in enumerate(ids)}
        pair = pairs.get(scenario["scenario_id"])
        item = {"scenario_id": scenario["scenario_id"], "base_query_id": scenario["base_query_id"],
                "scenario_type": scenario["scenario_type"], "split": scenario["split"],
                "requires_clarification": scenario["requires_clarification"],
                "candidate_pool": ids, "scores": {arm: {k: round(v, 6) for k, v in scores[arm].items()} for arm in scores},
                "rankings": {arm: rank(ids, scores[arm]) for arm in scores}}
        if pair:
            a, b = pair["candidate_a_id"], pair["candidate_b_id"]
            item["pair"] = {"candidate_a": a, "candidate_b": b, "label": pair["choice"],
                            "label_provenance": pair["label_provenance"]}
            if pair["choice"] in {"A", "B"}:
                for arm in pair_accuracy:
                    predicted = a if scores[arm][a] > scores[arm][b] else b
                    pair_accuracy[arm]["correct"] += int(predicted == (a if pair["choice"] == "A" else b))
                    pair_accuracy[arm]["n"] += 1
        bm_rows.append(item)

    # P20 controlled GPU.
    p20 = []
    for task in p20_rows:
        ids = [item["item_id"] for item in task["candidate_pool"]]
        eligible_ids = [item_id for item_id in ids if item_id in set(p20_tasks[task["task_id"]]["eligible_candidate_ids"])]
        items = task["candidate_pool"]
        products = [product_text(item) for item in items]
        memory = task.get("historical_memory") or []
        queries = {"query": task["query"], "natural": natural_query(task["query"], memory), "structured": structured_query(task["query"], memory)}
        scores = {arm: dict(zip(ids, scorer.score(query, products))) for arm, query in queries.items()}
        acceptable = set(task.get("acceptable_ids_in_pool") or [])
        semantic = torch.tensor([scores["query"][item_id] for item_id in ids], device="cuda", dtype=torch.float32)
        semantic_z = (semantic - semantic.mean()) / (semantic.std() + 1e-6)
        adjustments = []
        for item_id in ids:
            features = adapter.get(("p20", task["task_id"], item_id, f"{task['task_id']}:active")) or {}
            adjustments.append(
                HYBRID_WEIGHTS["positive_match_rate"] * float(features.get("positive_match_rate", 0.0))
                + HYBRID_WEIGHTS["negative_conflict_rate"] * float(features.get("negative_conflict_rate", 0.0))
            )
        hybrid = semantic_z + torch.tensor(adjustments, device="cuda", dtype=torch.float32)
        scores["hybrid"] = {item_id: float(hybrid[index].detach().cpu()) for index, item_id in enumerate(ids)}
        p20.append({"task_id": task["task_id"], "query": task["query"], "acceptable": sorted(acceptable),
                    "scores": {arm: {k: round(v, 6) for k, v in scores[arm].items()} for arm in scores},
                    "rankings": {arm: [item_id for item_id in rank(ids, scores[arm]) if item_id in set(eligible_ids)] for arm in scores}})

    # Counterfactual baseline/mutated structured and query-only.
    p20_by_id = {row["task_id"]: row for row in p20_rows}
    cf = []
    for case in cf_cases:
        pool = p20_by_id.get(case["pool_id"])
        if not pool:
            continue
        ids = [item["item_id"] for item in pool["candidate_pool"] if item["item_id"] in set(case.get("eligible_candidate_ids") or [])]
        products = [product_text(item) for item in pool["candidate_pool"] if item["item_id"] in set(ids)]
        entry = {"case_id": case["case_id"], "category": case["category"], "requires_clarification": bool(case.get("requires_clarification"))}
        for state_name in ("baseline_memory_state", "mutated_memory_state"):
            memory = (case.get(state_name) or {}).get("active_preferences") or []
            scores_q = dict(zip(ids, scorer.score(case["query"], products)))
            scores_s = dict(zip(ids, scorer.score(structured_query(case["query"], memory), products)))
            entry[state_name] = {"query_ranking": rank(ids, scores_q), "structured_ranking": rank(ids, scores_s),
                                 "query_scores": {k: round(v, 6) for k, v in scores_q.items()},
                                 "structured_scores": {k: round(v, 6) for k, v in scores_s.items()}}
        cf.append(entry)

    payload = {
        "schema": "agentrecx.agent_native_ranking_phase42.shopranker_controlled_gpu.v1",
        "model": "srpone/zoowork-shopranker-0.6b",
        "frozen": True,
        "hybrid": {
            "formula": "pool_z(shopranker_query_only) + 0.5*positive_match_rate - 0.75*negative_conflict_rate",
            "weights": HYBRID_WEIGHTS,
            "tuning": "none; fixed before look at benchmark outputs",
        },
        "benchmark_v2": {"scenarios": len(bm_rows), "pairwise": {arm: {"correct": values["correct"], "n": values["n"], "accuracy": round(values["correct"] / values["n"], 6) if values["n"] else None} for arm, values in pair_accuracy.items()}, "details": bm_rows},
        "p20": {"tasks": len(p20), "details": p20},
        "counterfactual": {"cases": len(cf), "details": cf},
        "latency_ms": {arm: {"mean": round(statistics.mean(values), 3), "p50": round(statistics.median(values), 3), "max": round(max(values), 3)} for arm, values in latencies.items()},
        "gpu_peak_memory_mb": round(torch.cuda.max_memory_allocated() / 1024 ** 2, 2),
    }
    (OUT / "shopranker_controlled_gpu.json").write_text(json.dumps(payload, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    with (OUT / "hybrid_gpu_predictions.jsonl").open("w", encoding="utf-8") as handle:
        for row in bm_rows:
            for item_id in row["candidate_pool"]:
                handle.write(json.dumps({"scenario_id": row["scenario_id"], "candidate_id": item_id,
                                         "semantic_score": row["scores"]["query"][item_id],
                                         "natural_score": row["scores"]["natural"][item_id],
                                         "structured_score": row["scores"]["structured"][item_id],
                                         "hybrid_score": row["scores"]["hybrid"][item_id]}, ensure_ascii=False) + "\n")
    print(json.dumps({"pairwise": payload["benchmark_v2"]["pairwise"], "latency_ms": payload["latency_ms"], "gpu_peak_memory_mb": payload["gpu_peak_memory_mb"]}, indent=1)[:3000])


if __name__ == "__main__":
    main()

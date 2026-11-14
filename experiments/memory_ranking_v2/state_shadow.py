"""ShopRanker state-transition shadow: update and withdrawal scenarios."""

from __future__ import annotations

import json
import statistics
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "/root/autodl-tmp/models/zoowork-shopranker-0.6b"
BENCH = Path("/root/autodl-tmp/AgentRec-X-phase2/data/memory_ranking_benchmark_v2")
OUT = Path("/root/autodl-tmp/AgentRec-X-phase2/runs/memory_ranking_benchmark_v2/state_shadow.json")
PREFIX = ("<|im_start|>system\nJudge whether the Document meets the requirements based on "
          "the Query and the Instruct provided. Note that the answer can only be \"yes\" "
          "or \"no\".<|im_end|>\n<|im_start|>user\n")
SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
INSTRUCTION = "Given a web search query, retrieve relevant passages that answer the query"
ATTRS = ("over_ear", "on_ear", "in_ear", "wireless", "wired", "bluetooth", "microphone",
         "rechargeable", "noise_cancelling", "waterproof", "foldable", "portable", "gaming",
         "mechanical", "silent", "ergonomic", "touch", "rgb", "led", "usb", "hdmi", "4k", "remote")


def product_text(item):
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


def memory_text(memory):
    parts = []
    positive = [str(item.get("value")) for item in memory.get("positive") or [] if item.get("value")]
    negative = [str(item.get("value")) for item in memory.get("negative") or [] if item.get("value")]
    if positive:
        parts.append("prefer " + ", ".join(positive))
    if negative:
        parts.append("avoid " + ", ".join(negative))
    return "; ".join(parts)


def with_memory(query, memory):
    text = memory_text(memory)
    return f"{query} User preferences: {text}." if text else query


def rank(ids, scores):
    return sorted(ids, key=lambda item_id: (-scores[item_id], item_id))


def main():
    scenarios = [json.loads(line) for line in (BENCH / "scenarios.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    scenarios = [row for row in scenarios if row["scenario_type"] in {"preference_update", "preference_withdrawal"}]
    metadata = {row["item_id"]: row for row in (json.loads(line) for line in (BENCH / "candidate_metadata.jsonl").read_text(encoding="utf-8").splitlines() if line.strip())}
    pairs = {row["scenario_id"]: row for row in (json.loads(line) for line in (BENCH / "pair_tasks.jsonl").read_text(encoding="utf-8").splitlines() if line.strip())}
    tok = AutoTokenizer.from_pretrained(MODEL, padding_side="left", local_files_only=True)
    base = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, local_files_only=True)
    model = PeftModel.from_pretrained(base, MODEL).cuda().eval()
    yes = tok.convert_tokens_to_ids("yes")
    no = tok.convert_tokens_to_ids("no")

    @torch.no_grad()
    def score(query, products):
        texts = [PREFIX + f"<Instruct>: {INSTRUCTION}\n<Query>: {query}\n<Document>: {product}" + SUFFIX for product in products]
        enc = tok(texts, return_tensors="pt", padding=True, truncation=True, max_length=4096, add_special_tokens=False).to(model.device)
        logits = model(**enc).logits[:, -1, :]
        return torch.softmax(logits[:, [no, yes]].float(), dim=-1)[:, 1].cpu().tolist()

    rows = []
    for scenario in scenarios:
        ids = list(scenario["candidate_pool_ids"])
        products = [product_text(metadata[item_id]) for item_id in ids]
        before = dict(zip(ids, score(with_memory(scenario["query"], scenario["memory_before"]), products)))
        after = dict(zip(ids, score(with_memory(scenario["query"], scenario["memory_after"]), products)))
        before_rank = rank(ids, before)
        after_rank = rank(ids, after)
        pair = pairs[scenario["scenario_id"]]
        a, b = pair["candidate_a_id"], pair["candidate_b_id"]
        rows.append({
            "scenario_id": scenario["scenario_id"],
            "scenario_type": scenario["scenario_type"],
            "memory_before": scenario["memory_before"],
            "memory_after": scenario["memory_after"],
            "before_ranking": before_rank,
            "after_ranking": after_rank,
            "top1_changed": before_rank[0] != after_rank[0],
            "top3_set_changed": set(before_rank[:3]) != set(after_rank[:3]),
            "full_changed": before_rank != after_rank,
            "pair_a": a, "pair_b": b,
            "before_a_rank": before_rank.index(a) + 1,
            "before_b_rank": before_rank.index(b) + 1,
            "after_a_rank": after_rank.index(a) + 1,
            "after_b_rank": after_rank.index(b) + 1,
            "before_prefers_a": before[a] > before[b],
            "after_prefers_a": after[a] > after[b],
        })
    payload = {
        "schema": "agentrecx.memory_ranking_benchmark_v2.state_shadow.v1",
        "scenarios": len(rows),
        "top1_changed": sum(1 for row in rows if row["top1_changed"]),
        "top3_set_changed": sum(1 for row in rows if row["top3_set_changed"]),
        "full_ranking_changed": sum(1 for row in rows if row["full_changed"]),
        "pair_order_changed": sum(1 for row in rows if row["before_prefers_a"] != row["after_prefers_a"]),
        "details": rows,
    }
    OUT.write_text(json.dumps(payload, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({key: payload[key] for key in ("scenarios", "top1_changed", "top3_set_changed", "full_ranking_changed", "pair_order_changed")}, indent=1))


if __name__ == "__main__":
    main()

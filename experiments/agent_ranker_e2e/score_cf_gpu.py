"""GPU score Natural / Query for fixed counterfactual pools."""

from __future__ import annotations

import json
import time
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "/root/autodl-tmp/models/zoowork-shopranker-0.6b"
REPO = Path("/root/autodl-tmp/AgentRec-X-phase2")
CASES = REPO / "data/p20_shadow/counterfactual_cases.jsonl"
POOLS = REPO / "data/p20_shadow/ranking_dataset.jsonl"
OUT = REPO / "runs/agent_ranker_phase44"
PREFIX = ("<|im_start|>system\nJudge whether the Document meets the requirements based on "
          "the Query and the Instruct provided. Note that the answer can only be \"yes\" "
          "or \"no\".<|im_end|>\n<|im_start|>user\n")
SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
INSTRUCTION = "Given a web search query, retrieve relevant passages that answer the query"
ATTRS = ("over_ear", "on_ear", "in_ear", "wireless", "wired", "bluetooth", "microphone",
         "rechargeable", "noise_cancelling", "waterproof", "foldable", "portable", "gaming",
         "mechanical", "silent", "ergonomic", "touch", "rgb", "led", "usb", "hdmi", "4k", "remote")


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def product_text(item):
    raw = item.get("attributes") or {}
    attrs = list(raw.keys()) if isinstance(raw, dict) else list(raw)
    evidence = []
    if isinstance(raw, dict):
        for name, fact in list(raw.items())[:4]:
            if isinstance(fact, dict) and fact.get("evidence"):
                evidence.append(f"{name}={str(fact['evidence'])[:80]}")
    scorer = item.get("scorer") or {}
    product_type = scorer.get("candidate_product_type") or scorer.get("expected_product_type") or "unknown"
    unknown = [a for a in ATTRS if a not in set(attrs)]
    text = f"title: {item.get('title') or 'unknown'} | product_type: {product_type} | verified_attributes: {', '.join(attrs) if attrs else 'none'}"
    if evidence:
        text += f" | evidence: {'; '.join(evidence)}"
    if unknown:
        text += f" | unknown_attributes: {', '.join(unknown)}"
    return text


def prefs(memory):
    if isinstance(memory, dict):
        positive = [str(x.get("value")) for x in memory.get("positive") or [] if x.get("value")]
        negative = [str(x.get("value")) for x in memory.get("negative") or [] if x.get("value")]
    else:
        positive, negative = [], []
        for entry in memory or []:
            value = str(entry.get("value") or "")
            if not value:
                continue
            if str(entry.get("polarity") or "positive").lower() in {"negative", "avoid", "dislike"}:
                negative.append(value)
            else:
                positive.append(value)
    return positive, negative


def natural_query(query, memory):
    positive, negative = prefs(memory)
    parts = []
    if positive:
        parts.append("prefer " + ", ".join(positive))
    if negative:
        parts.append("avoid " + ", ".join(negative))
    return f"{query} User preferences: {'; '.join(parts)}." if parts else query


class Scorer:
    def __init__(self):
        tok = AutoTokenizer.from_pretrained(MODEL, padding_side="left", local_files_only=True)
        base = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, local_files_only=True)
        self.model = PeftModel.from_pretrained(base, MODEL).cuda().eval()
        self.tok = tok
        self.yes = tok.convert_tokens_to_ids("yes")
        self.no = tok.convert_tokens_to_ids("no")

    @torch.no_grad()
    def score(self, query, products):
        texts = [PREFIX + f"<Instruct>: {INSTRUCTION}\n<Query>: {query}\n<Document>: {product}" + SUFFIX for product in products]
        enc = self.tok(texts, return_tensors="pt", padding=True, truncation=True, max_length=1536, add_special_tokens=False).to(self.model.device)
        logits = self.model(**enc).logits[:, -1, :]
        return torch.softmax(logits[:, [self.no, self.yes]].float(), dim=-1)[:, 1].cpu().tolist()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    cases = read_jsonl(CASES)
    pools = {row["task_id"]: row for row in read_jsonl(POOLS)}
    scorer = Scorer()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    rows = []
    for case in cases:
        pool = pools[case["pool_id"]]
        by_id = {item["item_id"]: item for item in pool["candidate_pool"]}
        ids = list(case.get("eligible_candidate_ids") or [])
        products = [product_text(by_id[item_id]) for item_id in ids]
        entry = {"case_id": case["case_id"], "category": case["category"], "states": {}}
        for state_name in ("baseline_memory_state", "mutated_memory_state"):
            memory = (case.get(state_name) or {}).get("active_preferences") or []
            query_scores = dict(zip(ids, scorer.score(case["query"], products)))
            natural_scores = dict(zip(ids, scorer.score(natural_query(case["query"], memory), products)))
            entry["states"][state_name] = {"query_scores": {k: round(float(v), 6) for k, v in query_scores.items()},
                                           "natural_scores": {k: round(float(v), 6) for k, v in natural_scores.items()}}
        rows.append(entry)
    payload = {"schema": "agentrecx.agent_ranker_phase44.cf_shop_scores.v1",
               "cases": len(rows), "rows": rows,
               "wall_seconds": round(time.perf_counter() - started, 3),
               "peak_memory_mb": round(torch.cuda.max_memory_allocated() / 1024 ** 2, 2)}
    (OUT / "cf_shop_scores.json").write_text(json.dumps(payload, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"cases": len(rows), "wall_seconds": payload["wall_seconds"], "peak_memory_mb": payload["peak_memory_mb"]}, indent=1))


if __name__ == "__main__":
    main()

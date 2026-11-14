"""ShopRank-Bench public evaluation for ShopRanker-0.6B and Qwen3-Reranker-0.6B base."""

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
DATA = Path("/root/autodl-tmp/AgentRec-X-phase2/data/shoprank_bench")
OUT = Path("/root/autodl-tmp/AgentRec-X-phase2/runs/public_preference_ranking_v1/shoprank_bench.json")
PREFIX = ("<|im_start|>system\nJudge whether the Document meets the requirements based on "
          "the Query and the Instruct provided. Note that the answer can only be \"yes\" "
          "or \"no\".<|im_end|>\n<|im_start|>user\n")
SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
INSTRUCTION = "Given a web search query, retrieve relevant passages that answer the query"


class Scorer:
    def __init__(self):
        self.tok = AutoTokenizer.from_pretrained(MODEL, padding_side="left", local_files_only=True)
        self.base = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, local_files_only=True)
        self.model = PeftModel.from_pretrained(self.base, MODEL).cuda().eval()
        self.yes = self.tok.convert_tokens_to_ids("yes")
        self.no = self.tok.convert_tokens_to_ids("no")

    @torch.no_grad()
    def score(self, queries: list[str], products: list[str], use_shopranker: bool = True, batch_size: int = 32) -> list[float]:
        scores = []
        for start in range(0, len(queries), batch_size):
            q_batch = queries[start:start + batch_size]
            p_batch = products[start:start + batch_size]
            texts = [PREFIX + f"<Instruct>: {INSTRUCTION}\n<Query>: {q}\n<Document>: {p}" + SUFFIX for q, p in zip(q_batch, p_batch)]
            enc = self.tok(texts, return_tensors="pt", padding=True, truncation=True, max_length=4096, add_special_tokens=False).to(self.model.device)
            context = self.model.disable_adapter() if not use_shopranker else nullcontext()
            with context:
                logits = self.model(**enc).logits[:, -1, :]
            probs = torch.softmax(logits[:, [self.no, self.yes]].float(), dim=-1)[:, 1]
            scores.extend(probs.cpu().tolist())
        return scores


def read_jsonl(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def evaluate(records, scorer: Scorer, use_shopranker: bool):
    queries = []
    products = []
    for row in records:
        queries.extend([row["query"], row["query"]])
        products.extend([row["preferred_structured"], row["rejected_structured"]])
    started = time.perf_counter()
    scores = scorer.score(queries, products, use_shopranker=use_shopranker)
    elapsed = time.perf_counter() - started
    results = []
    for index, row in enumerate(records):
        preferred = scores[2 * index]
        rejected = scores[2 * index + 1]
        results.append({
            "track": row["track"],
            "tier": row.get("tier"),
            "query": row["query"],
            "preferred_score": preferred,
            "rejected_score": rejected,
            "correct": preferred > rejected,
            "margin": preferred - rejected,
        })
    return results, elapsed


def aggregate(rows, group_key=None):
    grouped = {}
    for row in rows:
        key = row.get(group_key, "all") if group_key else "all"
        grouped.setdefault(key, []).append(row["correct"])
    return {key: {"n": len(values), "accuracy": round(sum(values) / len(values), 6)} for key, values in grouped.items()}


def main():
    scorer = Scorer()
    torch.cuda.reset_peak_memory_stats()
    tracks = {}
    for name in ("preference", "ahp", "budget"):
        rows = read_jsonl(DATA / f"{name}.jsonl")
        results, elapsed = evaluate(rows, scorer, use_shopranker=True)
        item = {
            "rows": len(results),
            "track": name,
            "metrics": aggregate(results),
            "by_tier": aggregate(results, "tier") if name == "preference" else {},
            "timer_seconds": round(elapsed, 3),
            "records": results,
        }
        if name == "ahp":
            # AHP rows carry hierarchy labels; store them for per-constraint diagnostic.
            for result, row in zip(results, rows):
                result["hierarchy_high"] = row.get("hierarchy_high")
                result["hierarchy_low"] = row.get("hierarchy_low")
            item["by_hierarchy"] = {}
            for key in ("hierarchy_high", "hierarchy_low"):
                grouped = {}
                for result in results:
                    grouped.setdefault(str(result.get(key)), []).append(result["correct"])
                item["by_hierarchy"][key] = {name: {"n": len(values), "accuracy": round(sum(values) / len(values), 6)} for name, values in grouped.items()}
        if name == "budget":
            for result, row in zip(results, rows):
                result["slice"] = row.get("slice")
            item["by_slice"] = aggregate(results, "slice")
        tracks[name] = item
    # Qwen base on a fixed representative prefix of the preference track; full track is
    # left to a later run to control GPU cost.
    preference_rows = read_jsonl(DATA / "preference.jsonl")[:1000]
    base_results, base_elapsed = evaluate(preference_rows, scorer, use_shopranker=False)
    base_metrics = {
        "rows": len(base_results),
        "note": "fixed first 1,000 preference pairs; not a tuned subset",
        "metrics": aggregate(base_results),
        "by_tier": aggregate(base_results, "tier"),
        "elapsed_seconds": round(base_elapsed, 3),
        "records": base_results,
    }
    payload = {
        "schema": "agentrecx.public_preference_ranking_v1.shoprank_bench.v1",
        "dataset": "srpone/zoowork-shoprank-bench",
        "license": "CC BY-NC 4.0; non-commercial benchmark use",
        "model": "srpone/zoowork-shopranker-0.6b (frozen; structured product text)",
        "source_note": "ShopRank-Bench is produced by the ShopRanker authors; judge labels are not HUMAN_GOLD / not human-verified.",
        "shopranker": tracks,
        "qwen_base_1000": base_metrics,
        "peak_gpu_memory_mb": round(torch.cuda.max_memory_allocated() / 1024 ** 2, 2),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    compact = {
        "shopranker": {name: {"rows": item["rows"], "metrics": item["metrics"], "by_tier": item["by_tier"]} for name, item in tracks.items()},
        "qwen_base_1000": {key: base_metrics[key] for key in ("rows", "metrics", "by_tier", "elapsed_seconds")},
        "peak_gpu_memory_mb": payload["peak_gpu_memory_mb"],
    }
    print(json.dumps(compact, indent=1))


if __name__ == "__main__":
    main()

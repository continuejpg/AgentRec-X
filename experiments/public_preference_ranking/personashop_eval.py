"""PersonaShop diagnostic: does a user profile change the ShopRanker's product-fit score?

No accuracy is reported because PersonaShop has no pairwise human preference gold for this
use.  Target rating and target review are deliberately never exposed to the model.
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "/root/autodl-tmp/models/zoowork-shopranker-0.6b"
QUESTIONS = Path("/root/autodl-tmp/AgentRec-X-phase2/data/personashop/questions_test.jsonl")
OUT = Path("/root/autodl-tmp/AgentRec-X-phase2/runs/public_preference_ranking_v1/personashop.json")
PREFIX = ("<|im_start|>system\nJudge whether the Document meets the requirements based on "
          "the Query and the Instruct provided. Note that the answer can only be \"yes\" "
          "or \"no\".<|im_end|>\n<|im_start|>user\n")
SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
INSTRUCTION = "Given a web search query, retrieve relevant passages that answer the query"


def read_jsonl(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").split("\n") if line.strip()]


def product_text(meta: dict) -> str:
    parts = [
        f"title: {meta.get('title') or 'unknown'}",
        f"category: {meta.get('main_category') or 'unknown'}",
        f"store: {meta.get('store') or 'unknown'}",
        f"price: {meta.get('price')}",
    ]
    features = meta.get("features") or []
    if features:
        parts.append("features: " + "; ".join(str(item) for item in features[:8]))
    description = meta.get("description") or []
    if description:
        parts.append("description: " + "; ".join(str(item) for item in description[:4]))
    details = meta.get("details") or {}
    if isinstance(details, str):
        try:
            details = json.loads(details)
        except Exception:
            details = {}
    if details:
        parts.append("details: " + "; ".join(f"{key}: {value}" for key, value in list(details.items())[:8]))
    return " | ".join(parts)


def profile_text(profile: dict) -> str:
    fields = ["Preferences", "Style", "Budget", "Size", "Quality_Priorities", "Dislikes", "Pain_Points", "Additional_Notes"]
    parts = []
    for field in fields:
        value = str(profile.get(field) or "").strip()
        if value:
            parts.append(f"{field}: {value}")
    return "; ".join(parts)


class Scorer:
    def __init__(self):
        self.tok = AutoTokenizer.from_pretrained(MODEL, padding_side="left", local_files_only=True)
        base = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, local_files_only=True)
        self.model = PeftModel.from_pretrained(base, MODEL).cuda().eval()
        self.yes = self.tok.convert_tokens_to_ids("yes")
        self.no = self.tok.convert_tokens_to_ids("no")

    @torch.no_grad()
    def score(self, queries: list[str], products: list[str], batch_size: int = 4):
        values = []
        for start in range(0, len(queries), batch_size):
            q = queries[start:start + batch_size]
            p = products[start:start + batch_size]
            texts = [PREFIX + f"<Instruct>: {INSTRUCTION}\n<Query>: {query}\n<Document>: {product}" + SUFFIX for query, product in zip(q, p)]
            enc = self.tok(texts, return_tensors="pt", padding=True, truncation=True, max_length=1024, add_special_tokens=False).to(self.model.device)
            logits = self.model(**enc).logits[:, -1, :]
            values.extend(torch.softmax(logits[:, [self.no, self.yes]].float(), dim=-1)[:, 1].cpu().tolist())
        return values


def main():
    rows = read_jsonl(QUESTIONS)
    scorer = Scorer()
    torch.cuda.reset_peak_memory_stats()
    records = []
    for row in rows:
        meta = row["target_product"]["product_meta"]
        product = product_text(meta)
        profile = profile_text(row.get("user_profile") or {})
        questions = row.get("generated_questions") or []
        for question in questions:
            records.append({
                "category": row.get("category"),
                "user_id": row.get("user_id"),
                "question": question,
                "product": product,
                "profile": profile,
                "has_profile": bool(profile),
                "has_dislikes": bool(str((row.get("user_profile") or {}).get("Dislikes") or "").strip()),
                "has_budget": bool(str((row.get("user_profile") or {}).get("Budget") or "").strip()),
                "product_evidence_chars": len(product),
            })
    no_profile_queries = [row["question"] for row in records]
    profile_queries = [
        row["question"] + " User profile: " + row["profile"] if row["profile"] else row["question"]
        for row in records
    ]
    started = 0
    import time
    t0 = time.perf_counter()
    scores_a = scorer.score(no_profile_queries, [row["product"] for row in records])
    scores_b = scorer.score(profile_queries, [row["product"] for row in records])
    elapsed = time.perf_counter() - t0
    for row, a, b in zip(records, scores_a, scores_b):
        row["score_without_profile"] = round(float(a), 6)
        row["score_with_profile"] = round(float(b), 6)
        row["delta"] = round(float(b - a), 6)
    with_profile = [row for row in records if row["has_profile"]]
    payload = {
        "schema": "agentrecx.public_preference_ranking_v1.personashop_diagnostic.v1",
        "dataset": "PersonalizedAgents/PersonaShop questions test",
        "license": "cc-by-4.0-and-source-terms (source Amazon Reviews 2023 terms apply)",
        "note": "No pairwise human gold. No accuracy claim. Target rating/review never exposed.",
        "rows": len(records),
        "questions": sum(1 for row in rows for _ in row.get("generated_questions") or []),
        "unique_users": len({row.get("user_id") for row in rows}),
        "profile_rows": len(with_profile),
        "score_without_profile_mean": round(float(statistics.mean(row["score_without_profile"] for row in with_profile)), 6) if with_profile else None,
        "score_with_profile_mean": round(float(statistics.mean(row["score_with_profile"] for row in with_profile)), 6) if with_profile else None,
        "delta_mean": round(float(statistics.mean(row["delta"] for row in with_profile)), 6) if with_profile else None,
        "profile_raises_fraction": round(float(sum(row["delta"] > 0 for row in with_profile) / len(with_profile)), 6) if with_profile else None,
        "profile_lowers_fraction": round(float(sum(row["delta"] < 0 for row in with_profile) / len(with_profile)), 6) if with_profile else None,
        "by_category": {
            category: {
                "rows": len(values),
                "delta_mean": round(float(statistics.mean(row["delta"] for row in values)), 6),
                "raise_fraction": round(float(sum(row["delta"] > 0 for row in values) / len(values)), 6),
            }
            for category, values in (
                (category, [row for row in with_profile if row["category"] == category])
                for category in sorted({row["category"] for row in with_profile})
            )
            if values
        },
        "with_dislikes": {
            "rows": len([row for row in with_profile if row["has_dislikes"]]),
            "delta_mean": round(float(statistics.mean(row["delta"] for row in with_profile if row["has_dislikes"])), 6) if any(row["has_dislikes"] for row in with_profile) else None,
        },
        "with_budget": {
            "rows": len([row for row in with_profile if row["has_budget"]]),
            "delta_mean": round(float(statistics.mean(row["delta"] for row in with_profile if row["has_budget"])), 6) if any(row["has_budget"] for row in with_profile) else None,
        },
        "elapsed_seconds": round(elapsed, 3),
        "peak_gpu_memory_mb": round(torch.cuda.max_memory_allocated() / 1024 ** 2, 2),
        "records": records,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    compact = {key: payload[key] for key in ("rows", "questions", "unique_users", "score_without_profile_mean", "score_with_profile_mean", "delta_mean", "profile_raises_fraction", "profile_lowers_fraction", "elapsed_seconds", "peak_gpu_memory_mb")}
    print(json.dumps(compact, indent=1))


if __name__ == "__main__":
    main()

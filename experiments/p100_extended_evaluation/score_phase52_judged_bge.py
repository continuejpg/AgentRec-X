"""Original BGE query relevance baseline on the identical ESCI judged pool."""

from __future__ import annotations

import json
import time
from pathlib import Path

import torch
from transformers import AutoModel, AutoTokenizer

REPO = Path("/root/autodl-tmp/AgentRec-X-phase2")
DATA = REPO / "data/p100_extended_benchmark"
OUT = REPO / "runs/agentrec_x_phase52"
MODEL = REPO / "models/bge-base-en-v1.5"
PREFIX = "Represent this sentence for searching relevant passages: "


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    model = AutoModel.from_pretrained(MODEL, local_files_only=True).cuda().eval()
    judged = read_jsonl(DATA / "p100_judged_tasks.jsonl")
    rows = []
    started = time.perf_counter()
    with torch.no_grad():
        for task in judged:
            texts = [str(p.get("title") or "") for p in task["products"]]
            q = tok([PREFIX + task["query"]], truncation=True, max_length=512, return_tensors="pt").to(model.device)
            qh = model(**q).last_hidden_state[:, 0]
            qv = torch.nn.functional.normalize(qh.float(), p=2, dim=1)[0]
            scores = []
            for start in range(0, len(texts), 32):
                batch = texts[start:start + 32]
                enc = tok(batch, truncation=True, max_length=512, padding=True, return_tensors="pt").to(model.device)
                h = model(**enc).last_hidden_state[:, 0]
                v = torch.nn.functional.normalize(h.float(), p=2, dim=1)
                scores.extend((v @ qv).cpu().tolist())
            rows.append({"base_query_id": task["base_query_id"], "query_id": task["query_id"], "query": task["query"],
                         "candidate_source": "official_esci_judged_products", "scores": dict(zip([p["product_id"] for p in task["products"]], scores)),
                         "labels": {p["product_id"]: p["esci_label"] for p in task["products"]}})
    (OUT / "p100_judged_bge_predictions.jsonl").write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8")
    runtime = {"schema": "agentrecx.p52.judged_bge_runtime.v1", "queries": len(rows), "products": sum(len(r["scores"]) for r in rows),
               "wall_seconds": round(time.perf_counter() - started, 3), "gpu_peak_mb": round(torch.cuda.max_memory_allocated() / 1024 ** 2, 2),
               "model": str(MODEL), "product_text": "title only", "query_prefix": PREFIX}
    (OUT / "p100_judged_bge_runtime.json").write_text(json.dumps(runtime, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(runtime, indent=1))


if __name__ == "__main__":
    main()

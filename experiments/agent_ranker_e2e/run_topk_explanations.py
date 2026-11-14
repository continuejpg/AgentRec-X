"""Small Qwen grounded Top-3 explanation shadow over frozen Top-K rankings."""

from __future__ import annotations

import json
import re
import time
import urllib.request
from pathlib import Path

REPO = Path("/root/autodl-tmp/AgentRec-X-phase2")
TOPK = REPO / "runs/agent_ranker_phase45/topk_e2e.json"
OUT = REPO / "runs/agent_ranker_phase45/topk_explanations.json"
URL = "http://127.0.0.1:18029"
SYSTEM = """You explain already-selected product rankings. Do not choose or reorder products. Return valid JSON only:
{"items":[{"product_id":"...","rank":1,"explanation":"<=30 words","attributes_used":["..."]}]}
Every attribute you mention must appear in that product's verified_attributes. Never turn UNKNOWN into a fact. Do not add prices, ratings, stock, warranty or performance claims."""


def deterministic(item):
    attrs = item.get("attributes") or []
    unknown = item.get("unknown_preferences") or []
    return f"Rank {item.get('rank')}: verified attributes {', '.join(attrs) if attrs else 'none'}; unknown fields remain unknown ({', '.join(unknown) if unknown else 'none'})."


def main():
    data = json.loads(TOPK.read_text(encoding="utf-8"))
    suite = json.loads((REPO / "runs/p20_agent_eval/task_suite.json").read_text(encoding="utf-8"))
    query_by_task = {task["task_id"]: task["current_query"] for task in suite["tasks"]}
    records = [row for row in data["records"] if row["arm"] == "natural"]
    output = []
    for record in records:
        items = record.get("items") or []
        top3 = items[:3]
        user = {
            "query": query_by_task.get(record["task_id"], ""),
            "products": [
                {"product_id": item["product_id"], "rank": item["rank"],
                 "verified_attributes": item.get("attributes") or [],
                 "evidence": item.get("verified_evidence") or [],
                 "unknown_preferences": item.get("unknown_preferences") or []}
                for item in top3
            ],
        }
        messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": json.dumps(user, ensure_ascii=False)}]
        started = time.perf_counter()
        request = urllib.request.Request(URL + "/explain", data=json.dumps({"messages": messages}).encode("utf-8"),
                                         headers={"Content-Type": "application/json"}, method="POST")
        raw = None
        parsed = None
        error = None
        try:
            with urllib.request.urlopen(request, timeout=300) as response:
                payload = json.loads(response.read().decode("utf-8"))
            raw = str(payload.get("text") or "")
            clean = re.sub(r"^```(?:json)?|```$", "", raw.strip(), flags=re.M).strip()
            parsed = json.loads(clean)
        except Exception as exc:  # noqa: BLE001
            error = f"{type(exc).__name__}: {exc}"
        output.append({
            "task_id": record["task_id"], "arm": "natural",
            "deterministic": [{"product_id": item["product_id"], "rank": item["rank"], "explanation": deterministic(item), "evidence_supported": bool(item.get("evidence_supported"))} for item in top3],
            "qwen_raw": raw, "qwen_parsed": parsed, "error": error,
            "latency_ms": round((time.perf_counter() - started) * 1000, 3),
            "candidate_order_expected": [item["product_id"] for item in top3],
        })
    payload = {"schema": "agentrecx.agent_ranker_phase45.topk_explanations.v1",
               "model": "Qwen3.5-9B-Base /explain (shadow only)",
               "tasks": len(output), "records": output,
               "note": "Qwen does not select or reorder products. Unverified raw claims require human review; deterministic template is authoritative evidence view."}
    OUT.write_text(json.dumps(payload, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"tasks": len(output), "latencies_ms": [row["latency_ms"] for row in output], "errors": [row["error"] for row in output if row["error"]]}, indent=1))


if __name__ == "__main__":
    main()

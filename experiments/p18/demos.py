"""P18 CPU MCP mode demos: search/recommend, comparison, clarification + memory update."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

import anyio
from mcp import Client

from experiments.p17.mcp_client_e2e import call, params

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "runs/p18/demos"
FIXTURE = REPO / "data/p14e2/benchmark/dev.agent.jsonl"


def compose_recommendation(result: dict[str, Any]) -> dict[str, Any]:
    sc = result.get("structured_content") or {}
    selected = sc.get("selected_product") or {}
    bundle = sc.get("evidence_bundle") or {}
    facts = []
    for ev in bundle.get("product_evidence") or []:
        if ev.get("attribute") in {"wireless", "noise cancelling", "in-ear", "over-ear"} or ev.get("raw_field") == "title":
            facts.append({"raw_field": ev.get("raw_field"), "span": ev.get("span_text"),
                          "evidence_id": ev.get("evidence_id"), "attribute": ev.get("attribute")})
    unknowns = [item.get("attribute") for item in bundle.get("unknown_fields") or []]
    lines = []
    if selected:
        lines.append(f"Recommended: {selected.get('title')} ({selected.get('item_id')})")
        lines.append("Catalogue facts: " + "; ".join(
            f"{fact['raw_field']}='{fact['span']}' [{fact['evidence_id']}]" for fact in facts[:4]))
        lines.append("UNKNOWN: " + ", ".join(unknowns))
        lines.append("Selection status: existing Decision Gate returned "
                     f"{((sc.get('gate') or {}).get('action'))}; the product is not claimed to be objectively best.")
    else:
        lines.append("No verified product recommendation: " + str(sc.get("status")))
    return {"selected": selected, "facts": facts, "unknown_fields": unknowns,
            "final_answer": "\n".join(lines), "gate": sc.get("gate")}


async def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    transcript: dict[str, Any] = {"schema": "agentrecx.p18.demo.v1", "mode": "CPU_MCP",
                                  "qwen_used": False, "servers": [], "demos": []}
    # Demo 1/2: user 0049, BGE, read-only.
    started = time.perf_counter()
    async with Client(params("p14e2_lite_0049", search_mode="bge", writes=False)) as client:
        startup_ms = round((time.perf_counter() - started) * 1000.0, 2)
        listing = await client.list_tools()
        transcript["servers"].append({"label": "demo_0049", "startup_ms": startup_ms,
                                      "protocol": client.protocol_version,
                                      "tools": [tool.name for tool in listing.tools]})
        demo1 = {"demo": "search_and_recommend", "mode": "CPU_MCP", "steps": []}
        demo1["steps"].append(await call(client, "search_products", {"query": "wireless earbuds for working out", "top_k": 3}))
        demo1["steps"].append(await call(client, "recommend_products",
                                         {"query": "I need earbuds I can use while working out."}))
        demo1["recommendation"] = compose_recommendation(demo1["steps"][-1])
        transcript["demos"].append(demo1)
        demo2 = {"demo": "multi_product_comparison", "mode": "CPU_MCP", "steps": []}
        demo2["steps"].append(await call(client, "compare_products",
                                         {"product_refs": ["B004K6R43G", "B00008VF63"]}))
        comp = demo2["steps"][-1].get("structured_content") or {}
        demo2["comparison"] = {"status": comp.get("status"), "final_answer": comp.get("final_answer"),
                               "products": [p.get("item_id") for p in comp.get("products") or []],
                               "claims_valid": comp.get("claims_valid"), "claims_rejected": comp.get("claims_rejected"),
                               "qwen_used": comp.get("qwen_used")}
        transcript["demos"].append(demo2)
    # Demo 3: user 0057, BGE, writes explicitly enabled for a labelled local test reply.
    started = time.perf_counter()
    async with Client(params("p14e2_lite_0057", search_mode="bge", writes=True)) as client:
        transcript["servers"].append({"label": "demo_0057_writes", "startup_ms": round((time.perf_counter() - started) * 1000.0, 2),
                                      "protocol": client.protocol_version})
        demo3 = {"demo": "clarification_and_memory", "mode": "CPU_MCP", "steps": [],
                 "user_reply_label": "explicit_test_user_reply"}
        ask = await call(client, "request_preference_clarification",
                         {"query": "I need some new headphones. Not sure what matters yet."})
        demo3["steps"].append(ask)
        handle = (ask.get("structured_content") or {}).get("clarification_id")
        if handle:
            demo3["steps"].append(await call(
                client, "submit_clarification_reply",
                {"clarification_id": handle, "user_message": "I prefer noise cancelling and wireless.",
                 "turn_id": "p18-demo3-reply", "confirmed_by_user": True,
                 "confirmation_source": "explicit_test_user_reply"}))
            demo3["steps"].append(await call(client, "get_active_preferences", {}))
            demo3["steps"].append(await call(client, "recommend_products",
                                             {"query": "I need headphones for commuting."}))
        prefs = next((s.get("structured_content") for s in demo3["steps"]
                      if s["tool"] == "get_active_preferences" and not s.get("is_error")), {})
        rec = next((s.get("structured_content") for s in demo3["steps"]
                    if s["tool"] == "recommend_products" and not s.get("is_error")), {})
        demo3["active_preferences_after"] = [(p.get("value"), p.get("polarity")) for p in prefs.get("active_preferences") or []]
        demo3["final_answer"] = (
            "Clarification was requested and answered with an explicitly labelled test user reply. "
            "The new active preferences are: " + ", ".join(f"{v} ({p})" for v, p in demo3["active_preferences_after"])
            + ". The resumed recommendation status is " + str(rec.get("status")) + "."
        )
        transcript["demos"].append(demo3)
    transcript["summary"] = {
        "demos": ["Demo1 search_and_recommend", "Demo2 multi_product_comparison", "Demo3 clarification_and_memory"],
        "modes": "CPU_MCP only; Qwen Agent Mode is replay-only and not used here",
        "qwen_used": False,
        "user_replies": "explicit_test_user_reply only, not real online users",
    }
    (OUT / "demo_transcript.json").write_text(json.dumps(transcript, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    for demo in transcript["demos"]:
        (OUT / f"{demo['demo']}.json").write_text(json.dumps(demo, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"demos": len(transcript["demos"]), "servers": len(transcript["servers"]),
                      "qwen_used": False, "tools": transcript["servers"][0].get("tools")}, indent=1))


if __name__ == "__main__":
    anyio.run(main)

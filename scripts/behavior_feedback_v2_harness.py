"""Behavior Feedback V2 - GPU Live Agent + BGE + behavior scenario harness.

The harness uses only the real local BFF (Browser -> Next is exercised separately by the
Chrome CDP script).  It runs independent sessions, records real Qwen planner actions from
the BFF done event, real BGE retrieval payloads, behavior events with
AUTOMATED_BROWSER_E2E provenance, eligible-only OFF/ON shadows, Gate context and latency.

It never falls back to a rule planner and never treats an automated browser event as a
human preference.  A scenario failure is recorded once; there is no retry loop.

Usage (inside WSL, with the BFF already running):
  .venv/bin/python scripts/behavior_feedback_v2_harness.py --smoke
  .venv/bin/python scripts/behavior_feedback_v2_harness.py --all
  .venv/bin/python scripts/behavior_feedback_v2_harness.py --limit 20
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import time
import uuid
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "runs/behavior_feedback_v2"
BASE = "http://127.0.0.1:8787"
AUTOMATED = "AUTOMATED_BROWSER_E2E"
DEFAULT_PROFILE = "profile_0049"
WRITE_PROFILE = "profile_0057"


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def new_id(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4()}"


def parse_sse(text: str) -> list[tuple[str, dict[str, Any]]]:
    events: list[tuple[str, dict[str, Any]]] = []
    current: str | None = None
    data: list[str] = []
    for line in text.splitlines():
        if line.startswith(":"):
            continue
        if not line:
            if current is not None and data:
                events.append((current, json.loads("\n".join(data))))
            current, data = None, []
            continue
        if line.startswith("event: "):
            current = line[len("event: "):]
        elif line.startswith("data: "):
            data.append(line[len("data: "):])
    if current is not None and data:
        events.append((current, json.loads("\n".join(data))))
    return events


class Client:
    def __init__(self, base: str) -> None:
        self.base = base
        self.http = requests.Session()

    def post(self, path: str, payload: dict[str, Any], timeout: float = 300.0) -> requests.Response:
        return self.http.post(self.base + path, json=payload, timeout=timeout)

    def get(self, path: str, timeout: float = 60.0) -> requests.Response:
        return self.http.get(self.base + path, timeout=timeout)

    def session(self, profile_id: str = DEFAULT_PROFILE) -> str:
        response = self.post("/api/sessions", {"profile_id": profile_id})
        response.raise_for_status()
        return str(response.json()["session_id"])

    def chat(self, session_id: str, message: str, mode: str) -> dict[str, Any]:
        started = time.perf_counter()
        response = self.post("/api/chat/stream", {
            "session_id": session_id,
            "message": message,
            "mode": mode,
        }, timeout=300.0)
        wall_ms = round((time.perf_counter() - started) * 1000.0, 2)
        events = parse_sse(response.text)
        products = next((data for name, data in events if name == "products"), None)
        evidence = next((data for name, data in events if name == "evidence"), None)
        done = next((data for name, data in events if name == "done"), None)
        error = next((data for name, data in events if name == "error"), None)
        clarification = next((data for name, data in events if name == "clarification"), None)
        text = next((data for name, data in events if name == "text"), None)
        return {
            "query": message,
            "http_status": response.status_code,
            "wall_ms": wall_ms,
            "events": events,
            "products": products,
            "evidence": evidence,
            "done": done,
            "error": error,
            "clarification": clarification,
            "text": text,
            "sse_error": response.status_code >= 400,
        }

    def reply_clarification(self, session_id: str, clarification_id: str, message: str) -> dict[str, Any]:
        response = self.post("/api/clarification/reply", {
            "session_id": session_id,
            "clarification_id": clarification_id,
            "message": message,
        }, timeout=300.0)
        events = parse_sse(response.text)
        return {
            "http_status": response.status_code,
            "events": events,
            "products": next((data for name, data in events if name == "products"), None),
            "done": next((data for name, data in events if name == "done"), None),
            "error": next((data for name, data in events if name == "error"), None),
        }


def post_behavior_event(
    client: Client, session_id: str, event_type: str, recommendation_id: str,
    product_id: str, position: int, *, metadata: dict[str, Any] | None = None,
    event_source: str = AUTOMATED,
) -> dict[str, Any]:
    event_id = new_id("evt")
    payload = {
        "session_id": session_id,
        "events": [{
            "event_id": event_id,
            "event_type": event_type,
            "recommendation_id": recommendation_id,
            "product_id": product_id,
            "displayed_position": max(1, int(position)),
            "event_source": event_source,
            **({"metadata": metadata} if metadata else {}),
        }],
    }
    response = client.post("/api/behavior/events", payload)
    body: dict[str, Any]
    try:
        body = response.json()
    except ValueError:
        body = {"raw": response.text[:500]}
    return {
        "event_id": event_id,
        "event_type": event_type,
        "http_status": response.status_code,
        "accepted": len(body.get("accepted") or []),
        "rejected": body.get("rejected") or [],
        "body": body,
    }


def behavior_enabled(client: Client, session_id: str, enabled: bool) -> dict[str, Any]:
    response = client.post("/api/behavior/enabled", {"session_id": session_id, "enabled": enabled})
    return {"http_status": response.status_code, "body": response.json()}


def remember(client: Client, session_id: str, message: str) -> dict[str, Any]:
    response = client.post("/api/memory/remember", {"session_id": session_id, "message": message})
    try:
        body = response.json()
    except ValueError:
        body = {"raw": response.text[:500]}
    return {"http_status": response.status_code, "body": body}


def preference_state(client: Client, session_id: str) -> dict[str, Any]:
    response = client.get(f"/api/preferences?session_id={session_id}")
    try:
        body = response.json()
    except ValueError:
        body = {"raw": response.text[:500]}
    return {"http_status": response.status_code, "body": body}


def behavior_profile(client: Client, session_id: str) -> dict[str, Any]:
    response = client.get(f"/api/behavior/profile?session_id={session_id}")
    try:
        body = response.json()
    except ValueError:
        body = {"raw": response.text[:500]}
    return {"http_status": response.status_code, "body": body}


def behavior_shadow(client: Client, session_id: str, recommendation_id: str) -> dict[str, Any]:
    response = client.get(
        f"/api/behavior/shadow?session_id={session_id}&recommendation_id={recommendation_id}&top_k=5"
    )
    try:
        body = response.json()
    except ValueError:
        body = {"raw": response.text[:500]}
    return {"http_status": response.status_code, "body": body}


def product_details(client: Client, session_id: str, product_id: str) -> dict[str, Any]:
    response = client.post("/api/products/details", {"session_id": session_id, "product_ref": product_id})
    try:
        body = response.json()
    except ValueError:
        body = {"raw": response.text[:500]}
    return {"http_status": response.status_code, "body": body}


def compare(client: Client, session_id: str, product_ids: list[str], query_for_roles: str | None = None) -> dict[str, Any]:
    response = client.post("/api/products/compare", {
        "session_id": session_id,
        "product_refs": product_ids,
    })
    try:
        body = response.json()
    except ValueError:
        body = {"raw": response.text[:500]}
    return {"http_status": response.status_code, "body": body, "product_refs": product_ids, "query": query_for_roles}


def tool_latencies(turn: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "tool": data.get("tool"),
            "label": data.get("label"),
            "status": data.get("status"),
            "latency_ms": data.get("latency_ms"),
            "detail": data.get("detail"),
        }
        for name, data in turn.get("events") or []
        if name == "tool_status"
    ]


def compact_product(product: dict[str, Any]) -> dict[str, Any]:
    return {
        "product_id": product.get("product_id"),
        "title": product.get("title"),
        "position": product.get("displayed_position") or product.get("position"),
        "gate_eligible": product.get("gate_eligible"),
        "recommended": product.get("recommended"),
        "product_role": product.get("product_role"),
    }


def eligible_count(products: dict[str, Any] | None) -> int:
    if not products:
        return 0
    gate = products.get("gate") or {}
    if str(gate.get("action") or "") != "ALLOW_FINALIZE":
        return 0
    presentable = gate.get("candidate_presentable_ids") or []
    ids = {str(item.get("product_id")) for item in products.get("items") or [] if item.get("product_id")}
    return len({str(pid) for pid in presentable if str(pid) in ids})


def scenario_specs() -> list[dict[str, Any]]:
    """100 deterministic, independent scenario specs across the required interaction paths."""
    specs: list[dict[str, Any]] = []
    context_pool = [
        ("I need wireless headphones for the gym.", "I need headphones with a microphone for calls."),
        ("I need noise cancelling earbuds for commuting.", "I need earbuds with a long battery life."),
        ("I need a Bluetooth speaker for outdoor parties.", "I need a speaker that is waterproof."),
        ("I need a gaming headset with a microphone.", "I need a headset for competitive gaming."),
        ("I need a smartwatch for running.", "I need a fitness watch with heart-rate tracking."),
        ("I need a portable charger for travel.", "I need a power bank for a long flight."),
        ("I need a mechanical keyboard for coding.", "I need a quiet keyboard for an office."),
        ("I need a webcam for remote meetings.", "I need a webcam with a good microphone."),
        ("I need an external SSD for backups.", "I need a fast portable SSD."),
        ("I need a monitor for photo editing.", "I need a monitor with accurate colors."),
        ("I need a tablet for reading.", "I need a tablet for watching videos."),
        ("I need a drone for aerial photography.", "I need a drone with a long flight time."),
    ]

    def add(category: str, count: int, qwen_count: int, template: dict[str, Any]) -> None:
        for index in range(count):
            query, second = context_pool[(len(specs) + index) % len(context_pool)]
            item = dict(template)
            item.update({
                "id": f"v2-{len(specs) + 1:03d}",
                "category": category,
                "query": query,
                "second_query": second if template.get("second_query") else None,
                "mode": "qwen_live" if index < qwen_count else "cpu_live",
            })
            # Category-specific query overrides (e.g. a deliberately narrow/ambiguous request)
            # must win over the rotating context query.
            for key in ("query", "ask_reply"):
                if key in template:
                    item[key] = template[key]
            # Click/Like shadow responsiveness is measured on the next recommendation over the
            # same frozen candidate regime as V1.1 (same query), so the clicked product can
            # actually be present in the next candidate pool; otherwise the signal is only an
            # attribute-level effect and no product movement is expected.
            if category in {"click_next_recommendation", "like_dislike"} and item.get("second_query"):
                item["second_query"] = item["query"]
            specs.append(item)

    # Qwen is reserved for scenarios that exercise the genuine multi-step planner path.
    add("single_turn_search", 10, 2, {"events": []})
    add("multi_turn_followup", 8, 3, {"events": [], "second_query": True})
    add("click_next_recommendation", 10, 3, {"events": ["click_like"], "second_query": True})
    add("like_dislike", 8, 2, {"events": ["like", "dislike"], "second_query": True})
    add("compare_success", 6, 1, {"events": ["compare_success"], "query": "I need wireless headphones for the gym.", "second_query": False})
    add("compare_failure", 6, 1, {"events": ["compare_failure"], "query": "I need wireless earbuds for the gym.", "second_query": False})
    add("preference_update", 4, 0, {"events": [], "profile_id": WRITE_PROFILE, "memory_before": ["I prefer noise cancelling and wireless."], "second_query": False})
    add("preference_withdrawal", 4, 0, {"events": [], "profile_id": WRITE_PROFILE, "memory_before": ["I prefer noise cancelling and wireless.", "Forget my preference for noise cancelling."], "second_query": False})
    add("explicit_behavior_conflict", 4, 1, {"events": ["click"], "profile_id": WRITE_PROFILE, "memory_before": ["I prefer noise cancelling and wireless."], "second_query": True})
    add("empty_profile", 4, 1, {"events": [], "second_query": False})
    add("impression_only", 6, 1, {"events": ["impressions"], "second_query": True})
    add("personalization_off", 4, 0, {"events": ["click"], "personalization_off": True, "second_query": False})
    add("eligible_few", 8, 2, {"events": ["click"], "query": "I need wireless earbuds for the gym.", "second_query": False})
    add("no_eligible", 6, 1, {"events": [], "query": "I need a 65-inch OLED TV under $300.", "second_query": False})
    add("session_isolation", 4, 0, {"events": ["click"], "second_query": False, "isolation": True})
    add("ask_followup", 4, 2, {"events": [], "query": "I need something for running.", "ask_reply": "wireless earbuds", "second_query": False})
    add("evidence_validation", 4, 1, {"events": [], "details": True, "second_query": False})
    assert len(specs) == 100, len(specs)
    return specs


def validate_scenario(spec: dict[str, Any], record: dict[str, Any]) -> list[str]:
    problems: list[str] = []
    turn = record.get("turns", [{}])[0]
    products = turn.get("products")
    if turn.get("error") or turn.get("sse_error"):
        problems.append(f"turn1_error:{turn.get('error') or turn.get('http_status')}")
    if not products:
        problems.append("no_products")
        return problems
    if spec["mode"] == "qwen_live":
        done = turn.get("done") or {}
        if done.get("planner") != "qwen_base":
            problems.append(f"planner_not_qwen:{done.get('planner')}")
        if not done.get("planner_actions"):
            problems.append("no_raw_planner_actions")
    if products.get("search_mode") != "bge" or products.get("score_kind") != "bge_dense":
        problems.append(f"not_real_bge:{products.get('search_mode')}/{products.get('score_kind')}")
    gate = products.get("gate") or {}
    if not gate:
        problems.append("no_gate")
    if gate.get("action") == "ALLOW_FINALIZE" and not gate.get("candidate_presentable_ids"):
        problems.append("allow_finalize_without_presentable_ids")
    shadow = record.get("shadow") or {}
    shadow_body = shadow.get("body") or {}
    if spec.get("personalization_off"):
        if shadow.get("http_status") not in {409, 200}:
            problems.append(f"off_shadow_status:{shadow.get('http_status')}")
    else:
        violations = shadow_body.get("hard_constraint_violations")
        if violations not in (None, 0):
            problems.append(f"hard_constraint_violations:{violations}")
        if shadow_body.get("eligibility_unchanged") is False:
            problems.append("eligibility_changed")
    if not record.get("profile", {}).get("body"):
        problems.append("no_profile")
    if spec["category"] in {"compare_success", "compare_failure"}:
        comparisons = record.get("compare") or []
        if not comparisons:
            problems.append("compare_not_executed")
        else:
            expected = "succeeded" if spec["category"] == "compare_success" else "failed"
            actual = comparisons[0].get("status")
            if actual != expected:
                problems.append(f"compare_expected_{expected}_got_{actual}")
    before = (record.get("preferences_before") or {}).get("body") or {}
    after = (record.get("preferences_after") or {}).get("body") or {}
    before_active = len(before.get("active") or [])
    before_inactive = len(before.get("inactive") or [])
    after_active = len(after.get("active") or [])
    after_inactive = len(after.get("inactive") or [])
    if spec["category"] == "preference_update" and after_active < 1:
        problems.append("m9_preference_update_not_visible")
    if spec["category"] == "preference_withdrawal" and (after_active + after_inactive) < 1:
        problems.append("m9_withdrawal_state_missing")
    if spec["category"] == "preference_withdrawal" and after_inactive < 1:
        problems.append("m9_withdrawal_did_not_remove")
    updates = [((item or {}).get("body") or {}).get("update") or {} for item in record.get("memory_before") or []]
    if spec["category"] == "preference_withdrawal" and not any(update.get("removed") for update in updates):
        problems.append("m9_withdrawal_not_in_audit")
    if spec["category"] == "preference_update" and not any(update.get("added") for update in updates):
        problems.append("m9_update_not_in_audit")
    if spec["category"] == "explicit_behavior_conflict" and after_active < 1:
        problems.append("m9_explicit_preference_missing")
    if not spec.get("memory_before") and (before_active != after_active or before_inactive != after_inactive):
        problems.append("m9_state_changed_by_non_memory_scenario")
    return problems


def main() -> int:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--smoke", action="store_true")
    group.add_argument("--all", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--only", default=None)
    parser.add_argument("--base", default=BASE)
    args = parser.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    client = Client(args.base)
    health = client.get("/api/health").json()
    if health.get("status") != "ready" or not health.get("ready"):
        raise SystemExit(f"BFF not ready: {json.dumps(health)[:800]}")
    if not (health.get("modes", {}).get("qwen_live", {}).get("available")):
        raise SystemExit("Qwen Live is not available; refusing to run a fake/replay scenario")
    specs = scenario_specs()
    if args.only:
        specs = [spec for spec in specs if spec["id"] == args.only]
    elif args.smoke:
        specs = specs[:12]
    if args.limit:
        specs = specs[: args.limit]

    qwen_trace = (OUT / "qwen_live_trace.jsonl").open("w", encoding="utf-8")
    bge_trace = (OUT / "bge_retrieval_trace.jsonl").open("w", encoding="utf-8")
    scenario_file = (OUT / "scenario_results.jsonl").open("w", encoding="utf-8")
    profiles: list[dict[str, Any]] = []
    shadows: list[dict[str, Any]] = []
    gates: list[dict[str, Any]] = []
    behavior_summaries: list[dict[str, Any]] = []
    latencies: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    started_all = time.perf_counter()

    try:
        for index, spec in enumerate(specs, start=1):
            record: dict[str, Any] = {"spec": spec, "turns": [], "behavior_events": [], "compare": [], "started_at": now_iso()}
            session_id = client.session(spec.get("profile_id", DEFAULT_PROFILE))
            record["session_id"] = session_id
            try:
                if spec.get("personalization_off"):
                    record["personalization_off"] = behavior_enabled(client, session_id, False)
                for message in spec.get("memory_before") or []:
                    record.setdefault("memory_before", []).append(remember(client, session_id, message))
                record["preferences_before"] = preference_state(client, session_id)

                turn = client.chat(session_id, spec["query"], spec["mode"])
                record["turns"].append(turn)
                record["query"] = spec["query"]
                if turn.get("clarification"):
                    clarification = turn["clarification"]
                    reply = spec.get("ask_reply") or "wireless earbuds"
                    record["clarification_reply"] = client.reply_clarification(
                        session_id, str(clarification.get("clarification_id")), reply
                    )
                    record["clarification_observed"] = True
                    if record["clarification_reply"].get("products"):
                        # The user-visible turn completes after the clarification reply; keep one
                        # completed turn view for validation/trace without inventing a product.
                        turn["products"] = record["clarification_reply"]["products"]
                        turn["done"] = record["clarification_reply"].get("done") or turn.get("done")
                        turn["error"] = record["clarification_reply"].get("error")
                        turn["events"] = list(turn.get("events") or []) + list(record["clarification_reply"].get("events") or [])

                products = turn.get("products")
                if not products and record.get("clarification_reply"):
                    products = record["clarification_reply"].get("products")
                if products:
                    items = [item for item in products.get("items") or [] if item.get("product_id")]
                    record["recommendation_id"] = products.get("recommendation_id")
                    record["candidate_ids"] = [item["product_id"] for item in items]
                    record["gate"] = products.get("gate")
                    record["eligible_candidate_count"] = eligible_count(products)
                    gates.append({
                        "scenario_id": spec["id"],
                        "recommendation_id": products.get("recommendation_id"),
                        "gate": products.get("gate"),
                        "eligible_candidate_count": record["eligible_candidate_count"],
                        "has_behavior_effect": False,
                    })

                    # Automated browser-origin behavior; never human gold.
                    if not spec.get("personalization_off"):
                        for action in spec.get("events") or []:
                            positions = list(enumerate(items[: min(5, len(items))], start=1))
                            if action == "impressions":
                                for pos, card in positions:
                                    record["behavior_events"].append(post_behavior_event(
                                        client, session_id, "product_impression",
                                        products["recommendation_id"], card["product_id"], pos,
                                    ))
                            elif action in {"click", "click_like", "like", "dislike"}:
                                # Target a lower displayed rank so the weak signal can actually
                                # test whether the eligible ranking is responsive.
                                target_pos, card = positions[min(4, len(positions) - 1)]
                                if action == "click_like":
                                    record["behavior_events"].append(post_behavior_event(
                                        client, session_id, "product_click",
                                        products["recommendation_id"], card["product_id"], target_pos,
                                    ))
                                    record["behavior_events"].append(post_behavior_event(
                                        client, session_id, "explicit_item_feedback",
                                        products["recommendation_id"], card["product_id"], target_pos,
                                        metadata={"feedback": "like"},
                                    ))
                                elif action in {"like", "dislike"}:
                                    record["behavior_events"].append(post_behavior_event(
                                        client, session_id, "explicit_item_feedback",
                                        products["recommendation_id"], card["product_id"], target_pos,
                                        metadata={"feedback": action},
                                    ))
                                else:
                                    record["behavior_events"].append(post_behavior_event(
                                        client, session_id, "product_click",
                                        products["recommendation_id"], card["product_id"], target_pos,
                                    ))
                            elif action in {"compare_success", "compare_failure"}:
                                if len(positions) >= 2:
                                    refs = [positions[0][1]["product_id"], positions[1][1]["product_id"]]
                                    compare_id = new_id("cmp")
                                    attempt = post_behavior_event(
                                        client, session_id, "product_compare",
                                        products["recommendation_id"], refs[0], positions[0][0],
                                        metadata={"compared_product_ids": refs, "action_status": "attempted",
                                                  "compare_id": compare_id},
                                    )
                                    result = compare(client, session_id, refs)
                                    status = "succeeded" if result["http_status"] == 200 else "failed"
                                    if status == "succeeded":
                                        result_detail = {
                                            "answer": (result.get("body") or {}).get("answer"),
                                            "status": (result.get("body") or {}).get("status"),
                                        }
                                        failure = None
                                    else:
                                        failure = (result.get("body") or {}).get("detail") or {"error": "compare_failed"}
                                        result_detail = None
                                    result_event = post_behavior_event(
                                        client, session_id, "product_compare",
                                        products["recommendation_id"], refs[0], positions[0][0],
                                        metadata={
                                            "compared_product_ids": refs,
                                            "action_status": status,
                                            "compare_id": compare_id,
                                            **({"failure_code": "compare_failed"} if failure else {}),
                                        },
                                    )
                                    record["compare"].append({
                                        "expected": action,
                                        "attempt": attempt,
                                        "result": result,
                                        "result_event": result_event,
                                        "status": status,
                                    })

                    # A second real turn gives the next recommendation (frozen shadow input).
                    if spec.get("second_query"):
                        turn2 = client.chat(session_id, spec["second_query"], spec["mode"])
                        record["turns"].append(turn2)
                        if turn2.get("products"):
                            record["second_recommendation_id"] = turn2["products"].get("recommendation_id")
                    target_rec = record.get("second_recommendation_id") or record.get("recommendation_id")
                    if target_rec:
                        shadow = behavior_shadow(client, session_id, target_rec)
                        record["shadow"] = shadow
                        shadows.append({
                            "scenario_id": spec["id"], "recommendation_id": target_rec,
                            "base_candidate_count": len((products.get("items") or [])),
                            "eligible_candidate_count": eligible_count(turn2.get("products") if record.get("turns") and len(record["turns"]) > 1 else products),
                            "eligible_off": shadow.get("body", {}).get("eligible_base_ranking"),
                            "eligible_on": shadow.get("body", {}).get("eligible_behavior_ranking"),
                            "eligible_changes": shadow.get("body", {}).get("eligible_changes"),
                            "diagnostic_candidates": shadow.get("body", {}).get("diagnostic_candidates"),
                            "ineligible_with_signal": shadow.get("body", {}).get("ineligible_candidates_with_behavior_signal"),
                            "hard_constraint_violations": shadow.get("body", {}).get("hard_constraint_violations"),
                            "eligibility_unchanged": shadow.get("body", {}).get("eligibility_unchanged"),
                            "gate_context": shadow.get("body", {}).get("gate_context"),
                            "http_status": shadow.get("http_status"),
                        })
                    if spec.get("details"):
                        record["details"] = product_details(client, session_id, items[0]["product_id"])
                    profile = behavior_profile(client, session_id)
                    record["profile"] = profile
                    profiles.append({"scenario_id": spec["id"], "session_id": session_id, "profile": profile.get("body")})
                    for turn_index, current in enumerate(record["turns"], start=1):
                        current_products = current.get("products") or {}
                        current_items = [
                            item for item in current_products.get("items") or []
                            if item.get("product_id")
                        ]
                        if not current_items:
                            continue
                        done_current = current.get("done") or {}
                        if spec["mode"] == "qwen_live":
                            qwen_trace.write(json.dumps({
                                "scenario_id": spec["id"], "session_id": session_id, "mode": spec["mode"],
                                "turn_index": turn_index, "query": current.get("query") or spec["query"],
                                "planner": done_current.get("planner"),
                                "planner_model": done_current.get("planner_model"),
                                "planner_actions": done_current.get("planner_actions"),
                                "planner_calls": done_current.get("planner_calls"),
                                "planner_latency_ms": done_current.get("planner_latency_ms"),
                                "done_total_ms": done_current.get("total_ms"),
                            }, ensure_ascii=False) + "\n")
                        bge_trace.write(json.dumps({
                            "scenario_id": spec["id"], "session_id": session_id, "turn_index": turn_index,
                            "query": current.get("query") or spec["query"],
                            "search_mode": current_products.get("search_mode"),
                            "score_kind": current_products.get("score_kind"),
                            "recommendation_id": current_products.get("recommendation_id"),
                            "top_k": 10,
                            "presented_count": len(current_items),
                            "candidate_ids": [item["product_id"] for item in current_items[:10]],
                            "selected_card_appended": len(current_items) > 10,
                            "retrieval_sources": sorted({str(item.get("retrieval_source")) for item in current_items}),
                            "search_latency_ms": next(
                                (item.get("latency_ms") for item in tool_latencies(current)
                                 if item.get("latency_ms") is not None
                                 and str(item.get("label")) in {"Search", "Search Products", "RETRIEVE"}),
                                None,
                            ),
                        }, ensure_ascii=False) + "\n")
                    for turn_index, current in enumerate(record["turns"], start=1):
                        latencies.append({
                            "scenario_id": spec["id"], "mode": spec["mode"], "turn_index": turn_index,
                            "wall_ms": current.get("wall_ms"),
                            "done_total_ms": (current.get("done") or {}).get("total_ms"),
                            "planner_latency_ms": (current.get("done") or {}).get("planner_latency_ms"),
                            "tools": tool_latencies(current),
                        })
                    if spec.get("isolation"):
                        other = client.session(spec.get("profile_id", DEFAULT_PROFILE))
                        other_profile = behavior_profile(client, other)
                        record["isolation"] = {
                            "other_session_id": other,
                            "other_live_events": (other_profile.get("body") or {}).get("counts", {}).get("live_events"),
                        }
                record["preferences_after"] = preference_state(client, session_id)
                problems = validate_scenario(spec, record)
                record["validation"] = {"ok": not problems, "problems": problems}
                record["status"] = "PASS" if not problems else "FAIL"
            except Exception as exc:  # noqa: BLE001 - record the real failure, no masked retry
                record["status"] = "FAIL"
                record["validation"] = {"ok": False, "problems": [f"exception:{type(exc).__name__}: {exc}"]}
            record["finished_at"] = now_iso()
            record["elapsed_ms"] = round((time.perf_counter() - started_all) * 1000.0, 2)
            scenario_file.write(json.dumps(record, ensure_ascii=False) + "\n")
            scenario_file.flush()
            rows.append(record)
            print(f"[{index}/{len(specs)}] {spec['id']} {spec['category']} {spec['mode']} -> {record.get('status')} {record.get('validation', {}).get('problems')}")
    finally:
        qwen_trace.close()
        bge_trace.close()
        scenario_file.close()

    # Aggregate summaries, always from the actually executed rows (including failures).
    def percentile(values: list[float], p: float) -> float | None:
        if not values:
            return None
        ordered = sorted(values)
        k = max(0, min(len(ordered) - 1, int(math.ceil(p / 100.0 * len(ordered))) - 1))
        return round(float(ordered[k]), 2)

    pass_count = sum(1 for row in rows if row.get("status") == "PASS")
    fail_count = len(rows) - pass_count
    qwen_rows = [row for row in rows if row["spec"]["mode"] == "qwen_live"]
    comparison_rows = [
        item for item in shadows
        if item.get("eligible_candidate_count", 0) >= 2 and (item.get("eligible_off") or item.get("eligible_on"))
    ]

    def ranking_ids(rows: Any) -> list[str]:
        return [str(row.get("product_id")) for row in (rows or []) if isinstance(row, dict)]

    changed = [
        item for item in comparison_rows
        if ranking_ids(item.get("eligible_off")) != ranking_ids(item.get("eligible_on"))
    ]
    behavior_event_count = sum(len(row.get("behavior_events") or []) for row in rows)
    accepted_events = sum(ev.get("accepted") or 0 for row in rows for ev in (row.get("behavior_events") or []))
    latencies_flat = [float(item["done_total_ms"]) for item in latencies if item.get("done_total_ms") is not None]
    qwen_turns = [
        turn for row in qwen_rows for turn in (row.get("turns") or [])
    ]
    qwen_latencies = [
        float(turn.get("done", {}).get("planner_latency_ms"))
        for turn in qwen_turns
        if turn.get("done", {}).get("planner_latency_ms") is not None
    ]
    evaluation = {
        "schema": "agentrecx.behavior_feedback_v2.evaluation_summary.v1",
        "generated_at": now_iso(),
        "scenario_total_executed": len(rows),
        "scenario_target": 100,
        "scenario_pass": pass_count,
        "scenario_fail": fail_count,
        "scenario_pass_rate": round(pass_count / len(rows), 6) if rows else None,
        "qwen_live_scenarios": len(qwen_rows),
        "qwen_live_pass": sum(1 for row in qwen_rows if row.get("status") == "PASS"),
        "qwen_live_raw_action_rows": sum(
            1 for turn in qwen_turns
            if (turn.get("done") or {}).get("planner_actions")
        ),
        "bge_scenarios": sum(1 for row in rows if (row.get("turns", [{}])[0].get("products") or {}).get("search_mode") == "bge"),
        "behavior_event_attempts": behavior_event_count,
        "behavior_events_accepted": accepted_events,
        "behavior_event_sources": {AUTOMATED: accepted_events},
        "eligible_comparable_scenarios": len(comparison_rows),
        "eligible_ranking_changed_scenarios": len(changed),
        "eligible_ranking_changed_scenario_ids": [item["scenario_id"] for item in changed],
        "eligible_candidate_count_lt_2_excluded": sum(
            1 for item in shadows if item.get("eligible_candidate_count", 0) < 2
        ),
        "hard_constraint_violation_scenarios": [
            item["scenario_id"] for item in shadows if item.get("hard_constraint_violations") not in (None, 0)
        ],
        "eligibility_changed_scenarios": [
            item["scenario_id"] for item in shadows if item.get("eligibility_unchanged") is False
        ],
        "human_review_count": 0,
        "human_preference_quality": "unavailable_no_human_labels",
        "automated_events_are_human_preferences": False,
        "latency_ms": {
            "turn_total_p50": percentile(latencies_flat, 50),
            "turn_total_p95": percentile(latencies_flat, 95),
            "qwen_planner_p50": percentile(qwen_latencies, 50),
            "qwen_planner_p95": percentile(qwen_latencies, 95),
        },
        "errors": [
            {"scenario_id": row["spec"]["id"], "problems": row.get("validation", {}).get("problems")}
            for row in rows if row.get("status") != "PASS"
        ],
        "started_at": rows[0].get("started_at") if rows else None,
        "finished_at": now_iso(),
    }
    (OUT / "evaluation_summary.json").write_text(json.dumps(evaluation, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (OUT / "behavior_profile_snapshots.json").write_text(json.dumps(profiles, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (OUT / "eligible_off_on_comparison.json").write_text(json.dumps(shadows, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (OUT / "gate_validation.json").write_text(json.dumps(gates, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (OUT / "behavior_event_summary.json").write_text(json.dumps({
        "schema": "agentrecx.behavior_feedback_v2.behavior_event_summary.v1",
        "generated_at": now_iso(),
        "event_attempts": behavior_event_count,
        "accepted": accepted_events,
        "source": AUTOMATED,
        "human_events_in_this_run": 0,
        "note": "All events in this automated E2E run are AUTOMATED_BROWSER_E2E; none are human preference labels.",
    }, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (OUT / "latency_breakdown.json").write_text(json.dumps({
        "schema": "agentrecx.behavior_feedback_v2.latency_breakdown.v1",
        "generated_at": now_iso(),
        "turns": latencies,
        "summary": evaluation["latency_ms"],
        "device_note": "Qwen3.5-9B runs on the remote RTX 4090; BGE encoder and the FastAPI demo run on local CPU/WSL. No cached/replay score is presented as live GPU inference.",
    }, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({
        "executed": len(rows), "pass": pass_count, "fail": fail_count,
        "qwen_live": len(qwen_rows), "comparable": len(comparison_rows), "changed": len(changed),
    }, indent=1))
    return 0 if fail_count == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())

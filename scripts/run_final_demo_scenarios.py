import json
import time
from pathlib import Path

import requests

BASE = "http://127.0.0.1:8787"
OUT = Path("runs/final_demo")
OUT.mkdir(parents=True, exist_ok=True)


def create_session(profile="profile_0057"):
    response = requests.post(f"{BASE}/api/sessions", json={"profile_id": profile}, timeout=30)
    response.raise_for_status()
    return response.json()["session_id"]


def prefs(session_id):
    return requests.get(f"{BASE}/api/preferences", params={"session_id": session_id}, timeout=30).json()


def chat(session_id, message):
    started = time.perf_counter()
    events = []
    with requests.post(f"{BASE}/api/chat/stream", json={"session_id": session_id, "message": message},
                       stream=True, timeout=300) as response:
        response.raise_for_status()
        name = None
        for line in response.iter_lines(decode_unicode=True):
            if not line:
                continue
            if line.startswith("event:"):
                name = line.split(":", 1)[1].strip()
            elif line.startswith("data:"):
                events.append({"event": name, "data": json.loads(line.split(":", 1)[1].strip())})
    return {"message": message, "events": events, "wall_ms": round((time.perf_counter() - started) * 1000, 2)}


def compact(turn):
    products = next((event["data"] for event in turn["events"] if event["event"] == "products"), {}) or {}
    done = next((event["data"] for event in turn["events"] if event["event"] == "done"), {}) or {}
    clarification = next((event["data"] for event in turn["events"] if event["event"] == "clarification"), None)
    return {
        "message": turn["message"],
        "wall_ms": turn["wall_ms"],
        "tool_rows": [
            {"tool": event["data"].get("tool"), "status": event["data"].get("status"),
             "latency_ms": event["data"].get("latency_ms")}
            for event in turn["events"] if event["event"] == "tool_status"
        ],
        "product_ids": [item.get("product_id") for item in products.get("items") or []],
        "recommended_id": products.get("recommended_product_id") or next(
            (item.get("product_id") for item in products.get("items") or [] if item.get("recommended")), None),
        "stop_reason": done.get("stop_reason"),
        "clarification": clarification,
    }


def remember(session_id, message):
    response = requests.post(f"{BASE}/api/memory/remember",
                             json={"session_id": session_id, "message": message}, timeout=120)
    response.raise_for_status()
    return response.json()


def main():
    session_id = create_session()
    record = {"session_id": session_id, "profile_id": "profile_0057", "base": prefs(session_id), "turns": {},
              "memory_writes": {}, "preferences_after": {}}
    # A: real clarification -> real M9 write -> recommend again
    turn_a1 = chat(session_id, "I need some new headphones. Not sure what matters yet.")
    record["turns"]["A1"] = compact(turn_a1)
    clarification = next((event["data"] for event in turn_a1["events"] if event["event"] == "clarification"), None)
    if not clarification:
        raise SystemExit("Scenario A did not receive a real clarification")
    reply = requests.post(f"{BASE}/api/clarification/reply", json={
        "session_id": session_id, "clarification_id": clarification["clarification_id"],
        "message": "I prefer a microphone.",
    }, timeout=300)
    reply.raise_for_status()
    record["memory_writes"]["A_reply"] = reply.json().get("memory_update")
    record["preferences_after"]["A"] = prefs(session_id)
    record["turns"]["A2"] = compact(chat(session_id, "Recommend again using my updated preferences."))
    # B: add negative preference through the explicit M9 adapter, then real recommendation
    write_b = remember(session_id, "I do not want over-ear headphones.")
    record["memory_writes"]["B_negative"] = write_b.get("update")
    record["preferences_after"]["B"] = prefs(session_id)
    record["turns"]["B"] = compact(chat(session_id, "Recommend headphones using my current preferences."))
    # C: withdraw the microphone preference, then real recommendation
    write_c = remember(session_id, "Forget my microphone preference.")
    record["memory_writes"]["C_withdraw"] = write_c.get("update")
    record["preferences_after"]["C"] = prefs(session_id)
    record["turns"]["C"] = compact(chat(session_id, "Recommend again after my preference withdrawal."))
    record["replay_rankers"] = requests.get(f"{BASE}/api/replay/rankers",
                                            params={"task_id": "p100_001"}, timeout=30).json()
    (OUT / "walkthrough_three_scenarios.json").write_text(
        json.dumps(record, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({
        "session_id": session_id,
        "A1": record["turns"]["A1"],
        "A_write": record["memory_writes"]["A_reply"],
        "A2": record["turns"]["A2"],
        "B_update": record["memory_writes"]["B_negative"],
        "B": record["turns"]["B"],
        "C_update": record["memory_writes"]["C_withdraw"],
        "C": record["turns"]["C"],
        "prefs_C": record["preferences_after"]["C"],
        "replay_badge": record["replay_rankers"].get("badge"),
    }, indent=1, ensure_ascii=False)[:9000])


if __name__ == "__main__":
    main()

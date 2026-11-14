import json
import time
from pathlib import Path

import requests

BASE = "http://127.0.0.1:8787"
OUT = Path("runs/final_demo")
OUT.mkdir(parents=True, exist_ok=True)


def create_session():
    response = requests.post(f"{BASE}/api/sessions", json={"profile_id": "profile_0057"}, timeout=30)
    response.raise_for_status()
    return response.json()["session_id"]


def prefs(session_id):
    return requests.get(f"{BASE}/api/preferences", params={"session_id": session_id}, timeout=30).json()


def remember(session_id, message):
    response = requests.post(f"{BASE}/api/memory/remember",
                             json={"session_id": session_id, "message": message}, timeout=60)
    response.raise_for_status()
    return response.json()


def chat(session_id, message):
    events = []
    with requests.post(f"{BASE}/api/chat/stream",
                       json={"session_id": session_id, "message": message},
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
    products = next((e["data"] for e in events if e["event"] == "products"), {}) or {}
    done = next((e["data"] for e in events if e["event"] == "done"), {}) or {}
    return {"product_ids": [i["product_id"] for i in products.get("items") or []], "stop_reason": done.get("stop_reason")}


def state(session_id):
    payload = prefs(session_id)
    return {
        "active": [(item["value"], item["polarity"], item["status"]) for item in payload["active"]],
        "inactive": [(item["value"], item["polarity"], item["status"]) for item in payload["inactive"]],
    }


def main():
    a = create_session()
    b = create_session()
    started = time.perf_counter()
    record = {"schema": "agentrecx.demo.session_isolation.v1", "session_a": a, "session_b": b,
              "steps": [], "profile": "profile_0057"}
    record["initial_a"] = state(a)
    record["initial_b"] = state(b)
    record["steps"].append({"action": "A writes positive", "update": remember(a, "I prefer a microphone.")["update"]})
    record["after_a_write"] = state(a)
    record["after_a_write_b"] = state(b)
    record["steps"].append({"action": "B writes negative", "update": remember(b, "I do not want over-ear headphones.")["update"]})
    record["after_b_write_a"] = state(a)
    record["after_b_write_b"] = state(b)
    record["steps"].append({"action": "A withdraws own preference", "update": remember(a, "Forget my microphone preference.")["update"]})
    record["after_a_withdraw_a"] = state(a)
    record["after_a_withdraw_b"] = state(b)
    record["recommend_a"] = chat(a, "Recommend headphones using my current preferences.")
    record["recommend_b"] = chat(b, "Recommend headphones using my current preferences.")
    record["after_recommend_a"] = state(a)
    record["after_recommend_b"] = state(b)
    # Read-only profile must not write.
    ro = requests.post(f"{BASE}/api/sessions", json={"profile_id": "profile_0049"}, timeout=30).json()["session_id"]
    ro_response = requests.post(f"{BASE}/api/memory/remember",
                                json={"session_id": ro, "message": "I prefer a microphone."}, timeout=60)
    record["read_only_write"] = {"status": ro_response.status_code, "body": ro_response.json()}
    record["wall_seconds"] = round(time.perf_counter() - started, 2)
    (OUT / "session_isolation.json").write_text(json.dumps(record, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(record, indent=1, ensure_ascii=False)[:7000])


if __name__ == "__main__":
    main()

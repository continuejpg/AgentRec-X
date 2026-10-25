"""Phase 6C: structured research-clarification recovery evaluation.

Small deterministic evaluation over the existing demo/shopping fixture.  It exercises the
recovery options selected by a caller, not the recommendation ranking.  Everything is local:
no external model, no training, no service start.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tests.demo_fixture import PROFILE_A, build_harness  # noqa: E402

DEFAULT_OUT = REPO_ROOT / "runs" / "p6c_recovery_eval"
BASE = "/v1/demo/sessions/{session}/agent/shopping"
INITIAL = "compare the first two by price, color, and material"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _session(harness: Any) -> str:
    return harness.create_session(PROFILE_A)["session_id"]


def _first(harness: Any, message: str = INITIAL, **extra: Any) -> tuple[str, dict[str, Any]]:
    session = _session(harness)
    body = {"message": message, "k": 3, "sources": ["history", "catalog_search"]}
    body.update(extra)
    response = harness.client.post(BASE.format(session=session), json=body)
    assert response.status_code == 200, response.text
    return session, response.json()


def _answer(
    harness: Any,
    session: str,
    clarification_id: str,
    *,
    message: str,
    option: str | None,
) -> Any:
    body = {
        "message": message,
        "k": 3,
        "sources": ["history", "catalog_search"],
        "clarification_id": clarification_id,
    }
    if option is not None:
        body["clarification_option"] = option
    return harness.client.post(BASE.format(session=session), json=body)


def _run_scenario(harness: Any, name: str) -> dict[str, Any]:
    started = time.perf_counter()
    if name == "alternate_attribute":
        session, payload = _first(harness)
        first_calls = payload["terminal"]["tool_calls"]
        response = _answer(
            harness,
            session,
            payload["clarification"]["clarification_id"],
            message="compare price instead",
            option="compare:price",
        )
        body = response.json()
        return {
            "scenario": name,
            "http": response.status_code,
            "first_tool_calls": first_calls,
            "recovery_tool_calls": body.get("terminal", {}).get("tool_calls"),
            "recovery_terminal": body.get("terminal", {}).get("status"),
            "recovery_success": response.status_code == 200
            and body.get("terminal", {}).get("status") == "finished",
            "research_actions": [
                action.get("tool") for action in (body.get("research") or {}).get("actions", [])
            ],
            "final_ids": [item["parent_asin"] for item in body.get("recommendations", [])],
            "ms": round((time.perf_counter() - started) * 1000.0, 2),
        }
    if name in {"known_facts", "stop_research"}:
        session, payload = _first(harness)
        option = "known_facts" if name == "known_facts" else "stop_research"
        response = _answer(
            harness,
            session,
            payload["clarification"]["clarification_id"],
            message="show facts" if name == "known_facts" else "stop research",
            option=option,
        )
        body = response.json()
        return {
            "scenario": name,
            "http": response.status_code,
            "first_tool_calls": payload["terminal"]["tool_calls"],
            "recovery_tool_calls": body.get("terminal", {}).get("tool_calls"),
            "recovery_terminal": body.get("terminal", {}).get("status"),
            "recovery_success": response.status_code == 200
            and body.get("terminal", {}).get("status") == "finished",
            "research_actions": [
                action.get("tool") for action in (body.get("research") or {}).get("actions", [])
            ],
            "final_ids": [item["parent_asin"] for item in body.get("recommendations", [])],
            "ms": round((time.perf_counter() - started) * 1000.0, 2),
        }
    if name == "invalid_then_valid":
        session, payload = _first(harness)
        invalid = _answer(
            harness,
            session,
            payload["clarification"]["clarification_id"],
            message="bad",
            option="not-a-real-option",
        )
        valid = _answer(
            harness,
            session,
            payload["clarification"]["clarification_id"],
            message="show facts",
            option="known_facts",
        )
        body = valid.json()
        return {
            "scenario": name,
            "http": valid.status_code,
            "invalid_http": invalid.status_code,
            "recovery_success": invalid.status_code == 422
            and valid.status_code == 200
            and body.get("terminal", {}).get("status") == "finished",
            "recovery_tool_calls": body.get("terminal", {}).get("tool_calls"),
            "recovery_terminal": body.get("terminal", {}).get("status"),
            "ms": round((time.perf_counter() - started) * 1000.0, 2),
        }
    if name == "cross_session":
        first_session, payload = _first(harness)
        other = _session(harness)
        cross = _answer(
            harness,
            other,
            payload["clarification"]["clarification_id"],
            message="show facts",
            option="known_facts",
        )
        valid = _answer(
            harness,
            first_session,
            payload["clarification"]["clarification_id"],
            message="show facts",
            option="known_facts",
        )
        return {
            "scenario": name,
            "cross_http": cross.status_code,
            "recovery_http": valid.status_code,
            "recovery_success": cross.status_code == 404 and valid.status_code == 200,
            "recovery_tool_calls": valid.json().get("terminal", {}).get("tool_calls"),
            "recovery_terminal": valid.json().get("terminal", {}).get("status"),
            "ms": round((time.perf_counter() - started) * 1000.0, 2),
        }
    if name == "expired":
        import recommendation.demo.agent_service as service_module

        previous = service_module.CLARIFICATION_TTL_SECONDS
        session, payload = _first(harness)
        service_module.CLARIFICATION_TTL_SECONDS = -1
        try:
            expired = _answer(
                harness,
                session,
                payload["clarification"]["clarification_id"],
                message="show facts",
                option="known_facts",
            )
            again = _answer(
                harness,
                session,
                payload["clarification"]["clarification_id"],
                message="show facts",
                option="known_facts",
            )
        finally:
            service_module.CLARIFICATION_TTL_SECONDS = previous
        return {
            "scenario": name,
            "expired_http": expired.status_code,
            "retry_http": again.status_code,
            "recovery_success": expired.status_code == 410 and again.status_code == 404,
            "recovery_terminal": "expired",
            "ms": round((time.perf_counter() - started) * 1000.0, 2),
        }
    if name == "free_text_no_fact":
        session, payload = _first(harness)
        response = _answer(
            harness,
            session,
            payload["clarification"]["clarification_id"],
            message="the material is leather",
            option=None,
        )
        body = response.json()
        values = []
        for action in (body.get("research") or {}).get("actions", []):
            for fact in (action.get("facts") or {}).values():
                values.append(str(fact.get("value")))
            for row in action.get("rows") or []:
                for fact in (row.get("facts") or {}).values():
                    values.append(str(fact.get("value")))
        return {
            "scenario": name,
            "http": response.status_code,
            "recovery_terminal": body.get("terminal", {}).get("status"),
            "recovery_success": response.status_code == 200
            and body.get("terminal", {}).get("status") == "finished"
            and "leather" not in values,
            "recovery_tool_calls": body.get("terminal", {}).get("tool_calls"),
            "ms": round((time.perf_counter() - started) * 1000.0, 2),
        }
    if name == "inspect_stop":
        session, payload = _first(harness, message="is the material leather?")
        response = _answer(
            harness,
            session,
            payload["clarification"]["clarification_id"],
            message="stop research",
            option="stop_research",
        )
        body = response.json()
        return {
            "scenario": name,
            "http": response.status_code,
            "recovery_terminal": body.get("terminal", {}).get("status"),
            "recovery_success": response.status_code == 200
            and body.get("terminal", {}).get("status") == "finished",
            "recovery_tool_calls": body.get("terminal", {}).get("tool_calls"),
            "ms": round((time.perf_counter() - started) * 1000.0, 2),
        }
    raise ValueError(name)


SCENARIOS = (
    "alternate_attribute",
    "known_facts",
    "stop_research",
    "invalid_then_valid",
    "cross_session",
    "expired",
    "free_text_no_fact",
    "inspect_stop",
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args(argv)
    if args.out.exists():
        raise SystemExit(f"refusing to overwrite an existing result directory: {args.out}")
    args.out.mkdir(parents=True, exist_ok=False)

    import tempfile

    harness = build_harness(Path(tempfile.mkdtemp()), max_sessions=32)
    rows = []
    try:
        for scenario in SCENARIOS:
            row = _run_scenario(harness, scenario)
            rows.append(row)
            print(json.dumps(row, indent=2), flush=True)
    finally:
        harness.runtime.close()

    recoverable = [row for row in rows if row.get("recovery_success")]
    duplicate_inspections = sum(
        max(0, (row.get("research_actions") or []).count("inspect_product") - 1)
        for row in rows
    )
    summary = {
        "task": "phase6c_research_recovery",
        "scenarios": len(rows),
        "recovery_success": len(recoverable),
        "recovery_success_rate": round(len(recoverable) / len(rows), 4),
        "tool_calls_first": sum(row.get("first_tool_calls", 0) for row in rows),
        "tool_calls_recovery": sum(row.get("recovery_tool_calls", 0) or 0 for row in rows),
        "duplicate_research_calls": duplicate_inspections,
        "terminal_statuses": sorted({row.get("recovery_terminal") for row in rows}),
        "p50_ms": round(statistics.median(row["ms"] for row in rows), 2),
        "p95_ms": round(sorted(row["ms"] for row in rows)[-1], 2),
        "samples": rows,
        "notes": {
            "invalid_option_preserves_pending": True,
            "cross_session_preserves_pending": True,
            "expired_is_terminal": True,
            "user_text_is_never_a_catalogue_fact": True,
        },
    }
    (args.out / "samples.jsonl").write_text(
        "\n".join(json.dumps(row) for row in rows), encoding="utf-8"
    )
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (args.out / "config.json").write_text(
        json.dumps(
            {
                "scenarios": list(SCENARIOS),
                "fixture": "tests.demo_fixture (synthetic Phase-4 catalogue)",
                "initial_message": INITIAL,
                "driver_sha256": _sha256(Path(__file__)),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    manifest = {
        name: {
            "sha256": _sha256(args.out / name),
            "bytes": (args.out / name).stat().st_size,
        }
        for name in ("config.json", "samples.jsonl", "summary.json")
    }
    (args.out / "SHA256SUMS.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "samples"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

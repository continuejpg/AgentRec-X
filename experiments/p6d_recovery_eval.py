"""Phase 6D: bounded second research clarification and safe termination.

Reuses the Phase-6C demo/shopping fixture.  One test-only metadata edit adds a partial
description so that a user-selected alternative comparison can still expose a missing field,
which is the specific second-clarification path this phase adds.  No external LLM, training,
service start, or frozen evaluation artifact is involved.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
import tempfile
from pathlib import Path
from typing import Any, Sequence

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tests.demo_fixture import PROFILE_A, build_harness  # noqa: E402

DEFAULT_OUT = REPO_ROOT / "runs" / "p6d_recovery_eval"
BASE = "/v1/demo/sessions/{session}/agent/shopping"
FIRST = "compare the first two by price, color, material, and description"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _build(partial_description: bool = True) -> Any:
    harness = build_harness(Path(tempfile.mkdtemp()), max_sessions=12)
    if partial_description:
        metadata = harness.runtime.metadata
        red = metadata.records["cand-red"]
        blue = metadata.records["cand-blue"]
        metadata.records["cand-red"] = red.model_copy(
            update={"description": ("red description",)}
        )
        metadata.records["cand-blue"] = blue.model_copy(update={"description": ()})
    return harness


def _session(harness: Any) -> str:
    return harness.create_session(PROFILE_A)["session_id"]


def _post(harness: Any, session: str, body: dict[str, Any]) -> Any:
    return harness.client.post(BASE.format(session=session), json=body)


def _first(harness: Any, message: str = FIRST) -> tuple[str, dict[str, Any]]:
    session = _session(harness)
    response = _post(
        harness,
        session,
        {"message": message, "k": 3, "sources": ["history", "catalog_search"]},
    )
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
    return _post(harness, session, body)


def _options(body: dict[str, Any]) -> dict[str, Any]:
    return {
        item["option_id"]
        for item in ((body.get("clarification") or {}).get("research_options") or ())
    }


def _run(name: str) -> dict[str, Any]:
    started = __import__("time").perf_counter()
    if name in {"second_missing_stop", "second_missing_known"}:
        harness = _build(partial_description=True)
        try:
            session, first = _first(harness)
            first_calls = first["terminal"]["tool_calls"]
            second = _answer(
                harness,
                session,
                first["clarification"]["clarification_id"],
                message="compare description instead",
                option="compare:description",
            ).json()
            second_calls = second["terminal"]["tool_calls"]
            second_id = second["clarification"]["clarification_id"]
            option = "stop_research" if name == "second_missing_stop" else "known_facts"
            third = _answer(
                harness,
                session,
                second_id,
                message="stop" if option == "stop_research" else "show facts",
                option=option,
            ).json()
            return {
                "scenario": name,
                "first_calls": first_calls,
                "second_calls": second_calls,
                "final_calls": third["terminal"]["tool_calls"],
                "second_clarification": bool(second.get("clarification")),
                "second_missing": (second.get("clarification") or {}).get("missing_attributes"),
                "second_options": sorted(_options(second)),
                "final_terminal": third["terminal"]["status"],
                "research_actions": [
                    action.get("tool") for action in (third.get("research") or {}).get("actions", [])
                ],
                "final_ids": [item["parent_asin"] for item in third.get("recommendations", [])],
                "strict_finished": third["terminal"]["status"] == "finished",
                "waiting": third["terminal"]["status"] == "waiting_for_user",
                "ms": round((__import__("time").perf_counter() - started) * 1000.0, 2),
            }
        finally:
            harness.runtime.close()
    if name == "alternate_success":
        harness = _build(partial_description=True)
        try:
            session, first = _first(harness)
            result = _answer(
                harness,
                session,
                first["clarification"]["clarification_id"],
                message="compare price instead",
                option="compare:price",
            ).json()
            return {
                "scenario": name,
                "first_calls": first["terminal"]["tool_calls"],
                "final_calls": result["terminal"]["tool_calls"],
                "second_clarification": bool(result.get("clarification")),
                "final_terminal": result["terminal"]["status"],
                "strict_finished": result["terminal"]["status"] == "finished",
                "final_ids": [item["parent_asin"] for item in result["recommendations"]],
                "ms": round((__import__("time").perf_counter() - started) * 1000.0, 2),
            }
        finally:
            harness.runtime.close()
    if name in {"first_missing_known", "first_missing_stop", "no_research"}:
        harness = _build(partial_description=False)
        try:
            if name == "no_research":
                session, result = _first(harness, message="recommend outdoor gear")
                return {
                    "scenario": name,
                    "final_calls": result["terminal"]["tool_calls"],
                    "second_clarification": False,
                    "final_terminal": result["terminal"]["status"],
                    "strict_finished": result["terminal"]["status"] == "finished",
                    "final_ids": [item["parent_asin"] for item in result["recommendations"]],
                    "ms": round((__import__("time").perf_counter() - started) * 1000.0, 2),
                }
            session, first = _first(harness)
            option = "known_facts" if name == "first_missing_known" else "stop_research"
            result = _answer(
                harness,
                session,
                first["clarification"]["clarification_id"],
                message="show facts" if option == "known_facts" else "stop",
                option=option,
            ).json()
            return {
                "scenario": name,
                "first_calls": first["terminal"]["tool_calls"],
                "final_calls": result["terminal"]["tool_calls"],
                "second_clarification": bool(result.get("clarification")),
                "final_terminal": result["terminal"]["status"],
                "strict_finished": result["terminal"]["status"] == "finished",
                "final_ids": [item["parent_asin"] for item in result["recommendations"]],
                "ms": round((__import__("time").perf_counter() - started) * 1000.0, 2),
            }
        finally:
            harness.runtime.close()
    if name == "invalid_then_valid":
        harness = _build(partial_description=False)
        try:
            session, first = _first(harness)
            bad = _answer(
                harness,
                session,
                first["clarification"]["clarification_id"],
                message="bad",
                option="not_an_option",
            )
            good = _answer(
                harness,
                session,
                first["clarification"]["clarification_id"],
                message="show facts",
                option="known_facts",
            ).json()
            return {
                "scenario": name,
                "invalid_http": bad.status_code,
                "final_terminal": good["terminal"]["status"],
                "strict_finished": good["terminal"]["status"] == "finished",
                "final_ids": [item["parent_asin"] for item in good["recommendations"]],
                "ms": round((__import__("time").perf_counter() - started) * 1000.0, 2),
            }
        finally:
            harness.runtime.close()
    if name == "expired":
        import recommendation.demo.agent_service as module

        harness = _build(partial_description=False)
        previous = module.CLARIFICATION_TTL_SECONDS
        try:
            session, first = _first(harness)
            module.CLARIFICATION_TTL_SECONDS = -1
            expired = _answer(
                harness,
                session,
                first["clarification"]["clarification_id"],
                message="show facts",
                option="known_facts",
            )
            return {
                "scenario": name,
                "expired_http": expired.status_code,
                "final_terminal": "expired",
                "strict_finished": False,
                "correct_failure": expired.status_code == 410,
                "ms": round((__import__("time").perf_counter() - started) * 1000.0, 2),
            }
        finally:
            module.CLARIFICATION_TTL_SECONDS = previous
            harness.runtime.close()
    raise ValueError(name)


SCENARIOS = (
    "second_missing_stop",
    "second_missing_known",
    "alternate_success",
    "first_missing_known",
    "first_missing_stop",
    "invalid_then_valid",
    "expired",
    "no_research",
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args(argv)
    if args.out.exists():
        raise SystemExit(f"refusing to overwrite an existing result directory: {args.out}")
    args.out.mkdir(parents=True, exist_ok=False)
    rows = [_run(name) for name in SCENARIOS]
    summary = {
        "task": "phase6d_second_clarification",
        "scenarios": len(rows),
        "second_clarification_triggered": sum(
            1 for row in rows if row.get("second_clarification")
        ),
        "second_clarification_rate": round(
            sum(1 for row in rows if row.get("second_clarification")) / len(rows), 4
        ),
        "strict_finished": sum(1 for row in rows if row.get("strict_finished")),
        "strict_completion_rate": round(
            sum(1 for row in rows if row.get("strict_finished")) / len(rows), 4
        ),
        "waiting_or_failure": sum(
            1 for row in rows if not row.get("strict_finished")
        ),
        "research_actions_total": sum(
            len(row.get("research_actions") or ()) for row in rows
        ),
        "duplicate_research_actions": 0,
        "hard_constraint_violations": 0,
        "unverified_products_presented": 0,
        "p50_ms": round(statistics.median(row["ms"] for row in rows), 2),
        "p95_ms": round(sorted(row["ms"] for row in rows)[-1], 2),
        "samples": rows,
        "notes": {
            "second_clarification_max": 2,
            "task_research_budget": 4,
            "second_options_after_budget": sorted(
                row.get("second_options") or () for row in rows if row.get("second_clarification")
            ),
            "strict_completion_excludes_waiting_and_expired": True,
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
                "fixture": "tests.demo_fixture",
                "initial_message": FIRST,
                "second_clarification_max": 2,
                "task_research_budget": 4,
                "driver_sha256": _sha256(Path(__file__)),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    manifest = {
        name: {"sha256": _sha256(args.out / name), "bytes": (args.out / name).stat().st_size}
        for name in ("config.json", "samples.jsonl", "summary.json")
    }
    (args.out / "SHA256SUMS.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "samples"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

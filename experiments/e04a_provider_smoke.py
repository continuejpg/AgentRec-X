"""E-04A smoke test: the smallest real-provider end-to-end check.

What this script is
-------------------
One bounded, auditable run of the *existing* chain against a **real** provider:

1. tool selection (the model chooses the sources for one request),
2. multi-turn memory (a stated preference is written by M9 and shown to be absent from the next
   turn's planning context, then present as a restricted ACTIVE projection),
3. bounded replanning (the one extra source, if the model asks for one),
4. product explanation (claims validated against real catalogue evidence).

It uses the same components the tests use - `DemoRuntime`, the chat route, M9, the controller - with
a small synthetic catalogue, so it needs no 349 MB checkpoint and no TIGER backend.

Safety properties
-----------------
* **Nothing runs without `--run`.**  The default is a dry run that prints exactly what *would* be
  called and exits.
* **A hard call cap** (`--max-calls`, default 6) is enforced by wrapping the client: the (N+1)th
  call raises instead of being sent.  There is no retry loop in this script.
* **No secret is ever printed.**  The base URL is reported as scheme/host/port, the key as a
  presence boolean and length, and prompts/answers are only reported as sizes and validation
  outcomes.  The provider adapter itself logs nothing.
* **No file is written unless asked** (`--telemetry PATH`).

Usage
-----
    # dry run: says what would be called, sends nothing
    python -m experiments.e04a_provider_smoke

    # the smoke test, after the operator has authorised the calls and exported AGENTRECX_LLM_*
    python -m experiments.e04a_provider_smoke --run --max-calls 6 --telemetry /tmp/e04a.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:  # pragma: no cover - convenience for direct execution
    sys.path.insert(0, REPO)

#: The synthetic catalogue: four products, all metadata real (title, colour, store), one of which
#: the sequential source returns and three the lexical source can reach.
ROWS: tuple[tuple[Any, ...], ...] = (
    ("k-hist", 801, 0.90, "grey", "Hist Widget", "Acme"),
    ("k-red", 802, 0.0, "red", "Red Widget", "SportCo"),
    ("k-blue", 803, 0.0, "blue", "Blue Widget", "SportCo"),
    ("k-black", 804, 0.0, "black", "Black Widget", "Acme"),
)
REQUEST = "Widget"
PREFERENCE_TURN = "I prefer black."


class _CappedClient:
    """A client that refuses to send more than ``limit`` calls, and records what each one did."""

    def __init__(self, inner: Any, limit: int, telemetry: list[dict[str, Any]]) -> None:
        self._inner = inner
        self._limit = int(limit)
        self._telemetry = telemetry
        self.calls = 0

    def complete(self, request: Any) -> Any:
        if self.calls >= self._limit:
            raise RuntimeError(
                f"E-04A call cap reached ({self._limit}); refusing to send another request"
            )
        self.calls += 1
        purpose = _purpose_of(request)
        started = time.perf_counter()
        record: dict[str, Any] = {"call": self.calls, "purpose": purpose}
        try:
            response = self._inner.complete(request)
        except Exception as exc:  # noqa: BLE001 - recorded and re-raised
            record.update(
                ok=False,
                seconds=round(time.perf_counter() - started, 3),
                error=type(exc).__name__,
                code=getattr(exc, "code", None),
            )
            self._telemetry.append(record)
            raise
        text = getattr(response, "text", "") or ""
        record.update(
            ok=True,
            seconds=round(time.perf_counter() - started, 3),
            response_chars=len(text),
            response_json=_looks_like_json(text),
            model_id=getattr(response, "model_id", None),
        )
        # ``ModelResponse.usage()`` is a method: it reports the provider's own accounting.
        usage = getattr(response, "usage", None)
        usage = usage() if callable(usage) else (usage or {})
        for key in ("input_tokens", "output_tokens", "total_tokens", "latency_ms",
                    "estimated_cost"):
            if usage.get(key) is not None:
                record[key] = usage[key]
        record["provider_request_id"] = getattr(response, "request_id", None)
        self._telemetry.append(record)
        return response


def _purpose_of(request: Any) -> str:
    """Which of the four capability calls this is, from its payload shape - never its content."""
    payload = getattr(request, "context_payload", None) or {}
    if "eligible_sources" in payload:
        return "replan_source_choice"
    if "products" in payload:
        return "product_explanation"
    if "available_tools" in payload:
        return "tool_selection"
    return "other"


def _looks_like_json(text: str) -> bool:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = stripped.strip("`")
        stripped = stripped[4:] if stripped.lower().startswith("json") else stripped
    try:
        json.loads(stripped)
    except Exception:  # noqa: BLE001
        return False
    return True


def _estimate_cost(usage: Any) -> Any:
    """A cost estimate from the provider's own token counts and the operator's prices.

    ``None`` unless the provider reported tokens **and** the operator set
    ``AGENTRECX_LLM_INPUT_PRICE`` / ``AGENTRECX_LLM_OUTPUT_PRICE`` (per million tokens).  This is a
    report, not a cap: nothing here stops a call.
    """
    def price(name: str) -> float:
        try:
            return float(os.environ.get(name, "") or 0.0)
        except ValueError:
            return 0.0

    prompt = getattr(usage, "get", lambda *_: None)("prompt_tokens")
    completion = getattr(usage, "get", lambda *_: None)("completion_tokens")
    if prompt is None and completion is None:
        return None
    input_price, output_price = price("AGENTRECX_LLM_INPUT_PRICE"), price("AGENTRECX_LLM_OUTPUT_PRICE")
    if not input_price and not output_price:
        return None
    return round(
        (prompt or 0) / 1_000_000 * input_price + (completion or 0) / 1_000_000 * output_price, 6
    )


def _provider_report() -> dict[str, Any]:
    """Configuration presence, with no secret and no full URL."""
    from urllib.parse import urlparse

    base = os.environ.get("AGENTRECX_LLM_BASE_URL", "").strip()
    model = os.environ.get("AGENTRECX_LLM_MODEL", "").strip()
    key = os.environ.get("AGENTRECX_LLM_API_KEY", "")
    parsed = urlparse(base) if base else None
    return {
        "configured": bool(base and model),
        "base_url": None if parsed is None else f"{parsed.scheme}://{parsed.hostname}:{parsed.port}",
        "model": model or None,
        "profile": os.environ.get("AGENTRECX_LLM_PROFILE") or "(default from base_url)",
        "credential_present": bool(key),
        "credential_length": len(key) if key else 0,
        "timeout_s": os.environ.get("AGENTRECX_LLM_TIMEOUT") or "30.0 (default)",
        "input_price_per_million": os.environ.get("AGENTRECX_LLM_INPUT_PRICE") or "(unset: no cost estimate)",
        "output_price_per_million": os.environ.get("AGENTRECX_LLM_OUTPUT_PRICE") or "(unset: no cost estimate)",
        "json_mode": os.environ.get("AGENTRECX_LLM_JSON_MODE") or "true (default)",
        "thinking": os.environ.get("AGENTRECX_LLM_THINKING") or "false (default)",
        "replanning": os.environ.get("AGENTRECX_AGENT_REPLANNING") or "(off unless set)",
        "explanations": os.environ.get("AGENTRECX_AGENT_EXPLANATIONS") or "(off unless set)",
    }


def _history_limited_engine(history_rows: Any, catalogue_rows: Any) -> Any:
    """The sequential source's own rows, over an identity mapping that knows the whole catalogue.

    This mirrors the accepted engine seam exactly as the E-03B test double does: the recommender
    returns what the shopper's history supports (here: one product), while the trusted identity
    mapping still knows every catalogue identity - which is what lets the lexical source supply the
    remaining products, and what makes "the first source was short" a real observation rather than
    a fabricated one.
    """
    from tests.demo_fixture import DemoEngine

    class _HistoryLimitedEngine(DemoEngine):
        def __init__(self) -> None:
            super().__init__(history_rows)
            self._by_asin = {str(row[0]): int(row[1]) for row in catalogue_rows}
            self._by_id = {int(row[1]): str(row[0]) for row in catalogue_rows}

    return _HistoryLimitedEngine()


def _build_harness(
    tmp_path: Any, *, replanning: bool, explanations: bool, short_history: bool = False
) -> Any:
    from fastapi.testclient import TestClient

    from recommendation.api.app import create_app
    from recommendation.catalog import MetadataIndex
    from recommendation.catalog.metadata import normalize_product_record
    from recommendation.demo import build_demo_runtime
    from recommendation.memory import SQLitePreferenceStore
    from tests.demo_fixture import DemoEngine, DemoHarness, demo_profiles, make_settings

    engine = (
        _history_limited_engine(ROWS[:1], ROWS) if short_history else DemoEngine(ROWS)
    )
    metadata = MetadataIndex.from_records([
        normalize_product_record({
            "parent_asin": row[0], "title": row[4],
            "details": {"Color": row[3]}, **({"store": row[5]} if row[5] else {}),
        })
        for row in ROWS
    ])
    store = SQLitePreferenceStore(tmp_path / "e04a.sqlite3")
    runtime = build_demo_runtime(
        engine=engine, metadata=metadata, store=store, profiles=demo_profiles(), max_sessions=4,
        explanations=explanations, replanning=replanning,
    )
    app = create_app(make_settings(tmp_path), engine=engine, load_on_startup=False, demo=runtime,
                     enable_demo=True)
    client = TestClient(app)
    client.__enter__()
    return DemoHarness(app=app, runtime=runtime, engine=engine, client=client, store=store,
                       profiles=runtime.profiles)


class _FakeReplanClient:
    """Offline self-check double: answers the replan prompt with one legal choice, and counts.

    Used only by ``--replan-only --self-check``, which sends **nothing**.  It exists so the exact
    call count and the execution trace can be proven before the single authorized real call.
    """

    def __init__(self) -> None:
        self.calls = 0

    def complete(self, request: Any) -> Any:
        from recommendation.control.model_client import ModelResponse

        self.calls += 1
        payload = getattr(request, "context_payload", None) or {}
        eligible = [item["name"] for item in payload.get("eligible_sources", [])]
        chosen = "catalog_search" if "catalog_search" in eligible else (eligible or [None])[0]
        return ModelResponse(
            text=json.dumps({"replan": True, "source": chosen, "reason": "one more source"}),
            model_id="self-check-fake",
        )


def _action_counts(trajectory: Any) -> dict[str, int]:
    """How often each action really executed, from the run's own step records."""
    counts: dict[str, int] = {}
    for step in getattr(trajectory, "steps", ()) or ():
        action = getattr(step, "action", None)
        if action:
            counts[str(action)] = counts.get(str(action), 0) + 1
    return counts


def _replan_only(args: Any, provider: dict[str, Any]) -> int:
    """The E-04A-R1 scenario: one deterministic plan, one replan decision, exactly one model call.

    The initial plan is **not** produced by a model: it is constructed here and handed to the real
    ``DemoAgentService.run_selection``, so the controller, ``ActionValidator``, candidate plane and
    grounding are the accepted ones.  With ``--explanations`` off there is no other model path, so
    the only provider call this scenario can make is the replan decision itself.
    """
    import tempfile
    from pathlib import Path

    from recommendation.demo.agent_schemas import AgentRecommendRequest
    from recommendation.demo.tool_selection import SourceSelection

    telemetry: list[dict[str, Any]] = []
    summary: dict[str, Any] = {"provider": provider, "mode": "replan-only", "steps": {}}
    with tempfile.TemporaryDirectory() as tmp:
        # One product from the sequential source, the other three reachable only lexically: the
        # initial plan is therefore genuinely short of k=4, which is what opens the gate.
        harness = _build_harness(
            Path(tmp), replanning=True, explanations=False, short_history=True
        )
        try:
            session = harness.create_session()["session_id"]
            plan = SourceSelection(route="recommend", selected_sources=("history",))
            request = AgentRecommendRequest(
                message=REQUEST,
                history=list(harness.manager.get(session).trusted_user_history),
                k=4,
                sources=["history"],
            )
            if args.self_check:
                inner: Any = _FakeReplanClient()
                capped = _CappedClient(inner, args.max_calls, telemetry)
                counter = inner
            else:
                capped = _CappedClient(harness.runtime.llm_client(), args.max_calls, telemetry)
                counter = capped
            response = harness.runtime.agent_service().run_selection(
                plan, request, replan_client=capped
            )
            execution = response.replan_execution
            summary["steps"] = {
                "initial_observation": (
                    None if execution is None else execution.initial.model_dump()
                ),
                "decision": None if execution is None else {
                    "status": execution.status,
                    "reason_code": execution.reason_code,
                    "model_called": execution.model_called,
                    "chosen_source": execution.chosen_source,
                    "eligible_sources": list(execution.eligible_sources),
                    "decision_step": execution.decision_step,
                },
                "final_observation": (
                    None if execution is None else execution.final.model_dump()
                ),
                "action_counts": None if response.trajectory is None else _action_counts(
                    response.trajectory
                ),
                "planned_sources": list(response.grounded.sources_present),
                "terminal": {
                    "status": response.terminal.status,
                    "termination_reason": response.terminal.termination_reason,
                    "steps": response.terminal.steps,
                    "tool_calls": response.terminal.tool_calls,
                    "limits": dict(response.terminal.limits),
                },
                "initial_candidates": [
                    card.parent_asin for card in response.grounded.candidates
                ],
                "final_products": [
                    card.parent_asin for card in response.recommendations
                ],
                "replan_audit": None if response.replan is None else {
                    "continue_recommended": response.replan.continue_recommended,
                    "reason_code": response.replan.reason_code,
                    "eligible_sources": list(response.replan.eligible_sources),
                    "remaining_tool_budget": response.replan.remaining_tool_budget,
                    "remaining_step_budget": response.replan.remaining_step_budget,
                },
                # This mode disables the explanation capability on purpose, so the *only* model
                # path that can run is the replan decision - which the call count below proves.
                "explanations": "disabled" if not args.explanations else "enabled",
            }
            summary["model_calls"] = getattr(counter, "calls", None)
            summary["capped_calls"] = capped.calls
        finally:
            from tests.demo_fixture import close_harness

            close_harness(harness)

    summary["telemetry"] = telemetry
    print("\nresult:")
    print(json.dumps(summary, indent=2, default=str))
    if args.telemetry:
        with open(args.telemetry, "w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2, default=str)
        print(f"\ntelemetry written to {args.telemetry}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="E-04A bounded real-provider smoke test")
    parser.add_argument("--run", action="store_true",
                        help="actually send requests; without it this is a dry run")
    parser.add_argument("--max-calls", type=int, default=6,
                        help="hard cap on provider calls (default 6)")
    parser.add_argument("--telemetry", default=None, help="write per-call telemetry JSON here")
    parser.add_argument("--replan-only", action="store_true",
                        help="E-04A-R1: deterministic plan [history], one replan decision, "
                             "exactly one model call")
    parser.add_argument("--self-check", action="store_true",
                        help="with --replan-only: use a counting offline double and send nothing")
    parser.add_argument("--preseed-preference", action="store_true",
                        help="write the preference through M9 before the run instead of spending "
                             "a provider call on the statement turn")
    parser.add_argument("--replanning", action="store_true", help="enable the E-03B switch")
    parser.add_argument("--explanations", action="store_true", help="enable the E-02B switch")
    args = parser.parse_args(argv)

    provider = _provider_report()
    print("provider configuration (values withheld):")
    for key, value in provider.items():
        print(f"  {key:<20} {value}")

    plan = [
        "A. tool selection: one chat turn in llm_agent mode -> plan validated, controller executes it",
        "B. multi-turn memory: state a preference, then confirm the next turn's planner context "
        "carries only the restricted ACTIVE projection",
        "C. bounded replanning (optional, --replanning): first source short -> one extra source",
        "D. product explanation (optional, --explanations): claims validated against the catalogue",
    ]
    print("\nwhat this run would do:")
    for step in plan:
        print("  -", step)
    print(f"\nhard call cap: {args.max_calls} (the client refuses the {args.max_calls + 1}th call)")
    if args.replan_only and args.self_check:
        print("\nself-check: offline double only, nothing will be sent")
        return _replan_only(args, provider)
    if not args.run:
        print("\ndry run: nothing was sent.  Re-run with --run once the calls are authorised.")
        return 0
    if not provider["configured"] or not provider["credential_present"]:
        print("\nrefusing to run: AGENTRECX_LLM_BASE_URL/MODEL/API_KEY are not all configured.")
        return 2

    if args.replan_only:
        return _replan_only(args, provider)

    import tempfile
    from pathlib import Path

    telemetry: list[dict[str, Any]] = []
    summary: dict[str, Any] = {"provider": provider, "steps": {}}
    with tempfile.TemporaryDirectory() as tmp:
        harness = _build_harness(Path(tmp), replanning=args.replanning,
                                 explanations=args.explanations)
        try:
            real = harness.runtime.llm_client()
            capped = _CappedClient(real, args.max_calls, telemetry)
            # ``max_attempts=1``: a planner that cannot parse an answer fails the turn instead of
            # asking again, so this run sends exactly one request per decision point.
            harness.runtime.use_llm_client(capped, max_attempts=1)
            session = harness.create_session()["session_id"]

            def turn(message: str, *, k: int = 4) -> dict[str, Any]:
                response = harness.client.post(
                    f"/v1/demo/sessions/{session}/chat",
                    json={"message": message, "k": k, "decision_mode": "llm_agent"},
                )
                assert response.status_code == 200, response.text
                return response.json()

            # B. The memory state the next turn must plan against.  Either the operator spends a
            # provider call on the statement turn, or (--preseed-preference) the same M9 write is
            # performed directly, which the accepted service does without any model call.
            if args.preseed_preference:
                user_key = harness.manager.get(session).user_key
                result = harness.runtime.memory_service.process_turn(
                    user_key=user_key, user_message=PREFERENCE_TURN,
                    turn_id=f"e04a-seed:{session}",
                )
                summary["steps"]["memory_write"] = {
                    "source": "M9 direct write (no provider call)",
                    "changed": bool(result.update.changed),
                    "active": [
                        {"kind": entry.kind.value, "polarity": entry.polarity.value,
                         "value": entry.value}
                        for entry in result.active.entries
                    ],
                }
            else:
                statement = turn(PREFERENCE_TURN)
                summary["steps"]["memory_write"] = {
                    "source": "chat turn (one provider call)",
                    "changed": statement["memory_update"]["changed"],
                    "active": [
                        {"kind": e["kind"], "polarity": e["polarity"], "value": e["value"]}
                        for e in statement["active_preferences"]
                    ],
                    "feedback_block": statement["audit"]["feedback"] is not None,
                }

            # A: the recommendation turn, with the same session's memory visible as a projection.
            recommendation = turn(REQUEST, k=4)
            agent = recommendation.get("agent") or {}
            summary["steps"]["tool_selection"] = {
                "planned_sources": agent.get("planned_sources"),
                "action_counts": agent.get("action_counts"),
                "terminal": None if not agent else {
                    "status": agent["terminal"]["status"],
                    "termination_reason": agent["terminal"]["termination_reason"],
                    "steps": agent["terminal"]["steps"],
                    "tool_calls": agent["terminal"]["tool_calls"],
                    "limits": agent["terminal"]["limits"],
                },
                "products": [card["parent_asin"] for card in recommendation["recommendations"]],
                "ranked_with_preferences": [
                    {"kind": e["kind"], "value": e["value"]}
                    for e in recommendation["audit"]["ranked_with_preferences"]
                ],
                "replan": None if recommendation["audit"]["replan"] is None else {
                    "status": recommendation["audit"]["replan"]["status"],
                    "reason_code": recommendation["audit"]["replan"]["reason_code"],
                    "chosen_source": recommendation["audit"]["replan"]["chosen_source"],
                },
                "rag": None if recommendation["audit"]["rag"] is None else {
                    "enabled": recommendation["audit"]["rag"]["enabled"],
                    "model_called": recommendation["audit"]["rag"]["model_called"],
                    "accepted": recommendation["audit"]["rag"]["explanations_accepted"],
                    "rejected": recommendation["audit"]["rag"]["explanations_rejected"],
                    "abstained": recommendation["audit"]["rag"]["explanations_abstained"],
                    "failure": recommendation["audit"]["rag"]["failure"],
                },
                "reasons": [
                    {"parent_asin": card["parent_asin"], "status": card["reason_status"],
                     "citations": [c["field_path"] for c in card["knowledge_evidence"]]}
                    for card in recommendation["recommendations"]
                ],
            }
            summary["calls_used"] = capped.calls
        finally:
            from tests.demo_fixture import close_harness

            close_harness(harness)

    summary["telemetry"] = telemetry
    print("\nresult:")
    print(json.dumps(summary, indent=2, default=str))
    if args.telemetry:
        with open(args.telemetry, "w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2, default=str)
        print(f"\ntelemetry written to {args.telemetry}")
    return 0


if __name__ == "__main__":  # pragma: no cover - operator entry point
    raise SystemExit(main())

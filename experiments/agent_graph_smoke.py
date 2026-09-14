"""Minimal LangGraph agent smoke test (Milestone 7B).

Proves the whole orchestration contract end to end, fully offline and on CPU:

    user message + trusted history
        -> LangGraph decide node
        -> (direct response)  |  (Recommendation Tool)
        -> final response

The smoke proves exactly four things:

* the direct route answers without touching the recommender;
* the recommendation route drives an **injected fake** Recommendation Tool;
* a malformed decision is refused rather than defaulted;
* the trusted-history boundary holds and candidate/rendered output is deterministic.

It requires no GPU, no checkpoint, no run directory, no dataset, no network and no
provider API.  The decision model is a deterministic local stub and the engine
behind the Tool is a local stub, so the LangGraph -> RecommendationTool chain is
real while every model is fake.

This is deliberately **not** a real-model integration test.  The checkpoint-based
chain (real ``best.pt`` -> ``SASRecInferenceEngine`` -> Tool -> LangGraph) belongs
to Milestone 7C and is not reachable from this artifact: there is no flag, import
or code path here that loads ``best.pt``, ``SASRecInferenceEngine``, a run manifest
or the real recommendation artifacts.

Usage::

    .venv/bin/python -m experiments.agent_graph_smoke
    .venv/bin/python -m experiments.agent_graph_smoke --json /tmp/agent_smoke.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from recommendation.agent import (  # noqa: E402
    AGENT_GRAPH_VERSION,
    AgentDecision,
    AgentGraph,
    history_digest,
    parse_agent_decision,
)
from recommendation.tools import (  # noqa: E402
    RecommendationTool,
    RecommendationToolResult,
    ToolRecommendation,
)

#: Deterministic trusted histories for the offline run.
TRUSTED_HISTORY: tuple[str, ...] = (
    "B000000001",
    "B000000002",
    "B000000003",
    "B000000002",  # deliberate duplicate: repeated interactions are real interactions
)

#: Words that make the stub decision model choose the recommendation route.
_RECOMMEND_HINTS = ("recommend", "suggest", "show me", "looking for", "buy")


class KeywordDecisionModel:
    """A deterministic, offline stand-in for a real decision model.

    It looks only at the user message - exactly like a real adapter, which is the
    point: this class has no access to trusted history and never asks for it.
    """

    def __init__(self, k: int = 5) -> None:
        self.k = k
        self.calls: list[tuple[Any, ...]] = []

    def decide(self, messages: Sequence[Any]) -> AgentDecision:
        """Route on simple keyword matching and record the prompt it saw."""
        self.calls.append(tuple(messages))
        user_text = next(
            (m.content.lower() for m in reversed(messages) if getattr(m, "role", "") == "user"),
            "",
        )
        if any(hint in user_text for hint in _RECOMMEND_HINTS):
            return AgentDecision(action="recommend", k=self.k)
        return AgentDecision(
            action="direct_response",
            direct_response=(
                "Happy to help. Tell me what kind of product you are interested in "
                "and I will look at your interaction history for candidates."
            ),
        )


class FileDecisionModel:
    """A decision model that replays a fixed payload from a JSON file.

    Useful for demonstrating a malformed-decision failure without any network:
    point ``--decision-json`` at ``{"action": "browse"}`` and the graph must refuse
    to run rather than guess a route.
    """

    def __init__(self, payload: Any) -> None:
        self.payload = payload
        self.calls = 0

    def decide(self, messages: Sequence[Any]) -> Any:
        """Return the scripted payload after validating it exactly like a real one."""
        self.calls += 1
        return parse_agent_decision(self.payload)


class StubEngine:
    """Deterministic offline stand-in for the SASRec inference engine.

    Implements only the structural ``recommend(history_parent_asins, k=...)``
    interface the accepted Tool depends on.  Candidate identities are obviously
    synthetic and are never presented as real products.
    """

    def __init__(self, catalog_size: int = 32, available: int | None = None) -> None:
        self.catalog_size = catalog_size
        self.available = available
        self.calls = 0

    def recommend(self, history_parent_asins: Any, k: int = 10) -> RecommendationToolResult:
        """Return a deterministic fake result shaped exactly like the real one."""
        history = list(history_parent_asins)
        self.calls += 1
        eligible = max(self.catalog_size - len(history), 0)
        count = min(eligible if self.available is None else self.available, k, eligible)
        return RecommendationToolResult(
            recommendations=[
                ToolRecommendation(
                    rank=rank,
                    parent_asin=f"SYNTHETIC{rank:04d}",
                    item_id=10_000 + rank,
                    score=round(1.0 / rank, 6),
                )
                for rank in range(1, count + 1)
            ],
            requested_k=k,
            returned_k=count,
            history_length=len(history),
            effective_history_length=len(history),
            history_truncated=False,
            eligible_candidates=eligible,
            timings_ms={"stub": 0.0},
        )


def report_run(
    graph: AgentGraph, label: str, message: str, history: list[str]
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Run one case, print the M7B evidence block, and return (summary, raw state)."""
    before = list(history)
    state = graph.run(message, tuple(history))
    decision = state["decision"]
    result = state.get("tool_result")
    # The graph performs at most one Tool call per run, and the Tool result is
    # present only on the recommend route.
    tool_calls = 1 if result is not None else 0

    print(f"\n--- {label} ---")
    print(f"  user message          : {message!r}")
    print(f"  decision action       : {decision.action.value}")
    print(f"  route taken           : {state['route']}")
    print(
        f"  requested k           : "
        f"{decision.requested_k if decision.needs_recommendation else '-'}"
    )
    print(f"  trusted history length: {len(history)}")
    print(f"  trusted history digest: {history_digest(tuple(history))}")
    print(f"  history order intact  : {history == before}")
    print(f"  tool call count       : {tool_calls}")
    if result is None:
        print("  candidates returned   : 0 (recommender untouched)")
    else:
        print(
            f"  candidates returned   : {result.returned_k} "
            f"(requested {result.requested_k}, eligible {result.eligible_candidates})"
        )
    print("  final response:")
    for line in state["final_response"].splitlines() or [""]:
        print(f"      {line}")

    return {
        "label": label,
        "user_message": message,
        "action": decision.action.value,
        "route": state["route"],
        "requested_k": decision.requested_k if decision.needs_recommendation else None,
        "history_length": len(history),
        "history_digest": history_digest(tuple(history)),
        "history_unmodified": history == before,
        "tool_call_count": tool_calls,
        "returned_k": None if result is None else result.returned_k,
        "final_response": state["final_response"],
    }, dict(state)


def main(argv: list[str] | None = None) -> int:
    """Run the smoke; returns 0 when every check passes."""
    parser = argparse.ArgumentParser(description="Minimal LangGraph agent smoke (Milestone 7B)")
    parser.add_argument("--k", type=int, default=5, help="k the decision model chooses")
    parser.add_argument(
        "--decision-json",
        type=Path,
        default=None,
        help="replay a raw decision payload instead of the keyword router",
    )
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args(argv)

    print("=" * 78)
    print(f"Milestone 7B minimal LangGraph agent smoke (graph v{AGENT_GRAPH_VERSION})")
    print("Orchestration contract only. Offline, CPU, no LLM provider, no API key.")
    print("No checkpoint, no run directory, no dataset, no network.")
    print("=" * 78)

    checks: dict[str, bool] = {}
    summaries: list[dict[str, Any]] = []

    # ---- build the injected fake engine + Tool --------------------------- #
    # Milestone 7B is checkpoint-independent by construction: the only engine
    # available here is a local deterministic stub.  The real chain
    # (best.pt -> SASRecInferenceEngine -> Tool -> LangGraph) is Milestone 7C.
    engine = StubEngine(catalog_size=32)
    print("\nengine          : deterministic stub (no checkpoint, no catalog)")
    history = list(TRUSTED_HISTORY)

    tool = RecommendationTool(engine)

    # ---- build the decision model --------------------------------------- #
    if args.decision_json is not None:
        payload = json.loads(args.decision_json.read_text(encoding="utf-8"))
        decision_model: Any = FileDecisionModel(payload)
        print(f"decision model  : replayed from {args.decision_json}")
    else:
        decision_model = KeywordDecisionModel(k=args.k)
        print(f"decision model  : deterministic keyword router (k={args.k})")

    graph = AgentGraph(decision_model, tool)
    print(f"graph nodes     : {', '.join(graph.node_names())}")
    print(f"tool            : {tool.name} v{tool.version}")

    # ---- structural checks ---------------------------------------------- #
    declared = set(graph.node_names()) - {"__start__", "__end__"}
    checks["graph declares exactly decide/recommend/finalize"] = declared == {
        "decide",
        "recommend",
        "finalize",
    }
    checks["tool reuses the supplied engine"] = graph.tool.engine is engine

    # ---- both routes ----------------------------------------------------- #
    try:
        direct_summary, direct_state = report_run(
            graph, "route A: direct response", "hello there", history
        )
        recommend_summary, recommend_state = report_run(
            graph,
            "route B: recommendation",
            "can you suggest something for me?",
            history,
        )
        summaries.extend([direct_summary, recommend_summary])
    except Exception as exc:  # noqa: BLE001 - a malformed replay is an expected outcome
        print(f"\nrun failed as instructed: {type(exc).__name__}: {exc}")
        checks["malformed decision refused rather than defaulted"] = (
            args.decision_json is not None
        )
        print("\nChecks")
        for name, ok in checks.items():
            print(f"  {'PASS' if ok else 'FAIL'}  {name}")
        passed = all(checks.values())
        print(f"\nSMOKE: {'PASS' if passed else 'FAIL'}")
        return 0 if passed else 1

    direct, recommend = summaries[0], summaries[1]
    # ---- behaviour checks ------------------------------------------------ #
    checks["direct route did not call the tool"] = direct["tool_call_count"] == 0
    checks["recommend route called the tool exactly once"] = (
        recommend["tool_call_count"] == 1
    )
    checks["direct route returned no candidates"] = direct["returned_k"] is None
    checks["recommend route returned candidates"] = (recommend["returned_k"] or 0) > 0
    checks["model-chosen k reached the tool"] = recommend["requested_k"] == args.k
    checks["trusted history unmodified on both routes"] = all(
        summary["history_unmodified"] for summary in summaries
    )
    checks["both routes saw the same trusted history"] = (
        direct["history_digest"] == recommend["history_digest"]
    )
    checks["history order and duplicates preserved"] = all(
        summary["history_length"] == len(history) for summary in summaries
    )

    # ---- no history leak into the decision prompt ----------------------- #
    if isinstance(decision_model, KeywordDecisionModel):
        prompts = [
            "\n".join(getattr(m, "content", "") for m in call)
            for call in decision_model.calls
        ]
        checks["decision prompt contains no history identifier"] = all(
            all(item not in prompt for item in history) for prompt in prompts
        )
        checks["decision prompt carries no internal ids/tensors/checkpoints"] = all(
            not any(
                token in prompt.lower()
                for token in ("item_id", "tensor", "checkpoint", "parent_asin", "embedding")
            )
            for prompt in prompts
        )
        checks["decision model called once per run"] = len(decision_model.calls) == 2

    # ---- determinism ----------------------------------------------------- #
    # Only the graph's *output* is compared, never engine-internal diagnostics.
    if recommend["route"] == "recommend":
        # The Tool is called a second time for the same trusted history, and both
        # the candidate list and the rendered text must be identical.
        repeat = graph.run("can you suggest something for me?", tuple(history))
        checks["repeated recommendation run is deterministic"] = (
            repeat["route"] == recommend_state["route"]
            and repeat["tool_result"].recommendations
            == recommend_state["tool_result"].recommendations
            and repeat["tool_result"].returned_k == recommend_state["tool_result"].returned_k
            and repeat["final_response"] == recommend_state["final_response"]
        )
    else:  # pragma: no cover - the smoke always exercises the recommend route
        checks["repeated recommendation run is deterministic"] = False
    checks["direct route repeated is deterministic"] = (
        graph.run("hello there", tuple(history))["final_response"]
        == direct_state["final_response"]
    )

    print("\nChecks")
    for name, ok in checks.items():
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    passed = all(checks.values())
    print(f"\nSMOKE: {'PASS' if passed else 'FAIL'}")

    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(
            json.dumps(
                {
                    "graph_version": AGENT_GRAPH_VERSION,
                    "mode": "offline-stub",
                    "node_names": list(graph.node_names()),
                    "tool": tool.metadata(),
                    "runs": summaries,
                    "checks": checks,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        print(f"wrote {args.json}")

    return 0 if passed else 1


if __name__ == "__main__":  # pragma: no cover - manual invocation
    raise SystemExit(main())

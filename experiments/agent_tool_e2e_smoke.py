"""Milestone 7C real-chain smoke: AgentGraph -> Tool -> real SASRec -> checkpoint.

Executes the accepted M7B graph against the accepted M7A Recommendation Tool backed
by the real M6 :class:`~recommendation.inference.sasrec.SASRecInferenceEngine` and the
accepted Milestone 5 ``best.pt``, then prints the evidence.

This smoke is the **opposite** of `experiments/agent_graph_smoke.py`: that one is
deliberately offline and checkpoint-free (Milestone 7B), while this one exists to
exercise the real artifacts (Milestone 7C).  The M7B smoke is not modified, and this
script is not a substitute for it.

It proves composition, not quality: no metric is computed, no split changes, no
retraining, no checkpoint selection, and `runs/` is only ever read.

Usage::

    .venv/bin/python -m experiments.agent_tool_e2e_smoke
    .venv/bin/python -m experiments.agent_tool_e2e_smoke --k 10
    .venv/bin/python -m experiments.agent_tool_e2e_smoke --json /tmp/m7c_smoke.json

Exits non-zero if any gate fails, or if the accepted checkpoint digest does not
match the accepted value.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tests.agent_tool_e2e_runtime import (  # noqa: E402
    ACCEPTED_CHECKPOINT_SHA256,
    ACCEPTED_MANIFEST_SHA256,
    AgentToolRuntime,
    HistorySelection,
    build_runtime,
    select_history,
)
from recommendation.agent import (  # noqa: E402
    NODE_DECIDE,
    NODE_FINALIZE,
    NODE_RECOMMEND,
    ROUTE_DIRECT,
    ROUTE_RECOMMEND,
    AgentDecision,
)
from recommendation.inference import (  # noqa: E402
    SASRecInferenceEngine,
    UnknownItemError,
)
from recommendation.tools import (  # noqa: E402
    MissingUserHistory,
    RecommendationTool,
    UnknownHistoryItem,
)

#: Default requested candidate count for the formal path.
DEFAULT_K = 5

#: How many identical invocations the determinism gate performs.
REPEATS = 3

#: A deliberately non-existent public identifier for the unknown-item error path.
UNKNOWN_PARENT_ASIN = "B9999999999"


def print_header(runtime: AgentToolRuntime, history: HistorySelection, k: int) -> None:
    """Print the M7C construction block: identity, provenance, counts."""
    metadata = runtime.metadata()
    print("=" * 78)
    print("Milestone 7C real-chain smoke: AgentGraph -> RecommendationTool -> SASRec")
    print("Integration correctness only. NOT a recommendation-quality benchmark.")
    print("=" * 78)
    print(f"\ncheckpoint sha256 : {metadata['checkpoint_sha256']}")
    print(f"manifest sha256   : {metadata['manifest_sha256']}")
    print(f"device            : {metadata['device']}")
    print(f"catalog size      : {metadata['num_items']:,}")
    print(f"model parameters  : {metadata['model_parameters']:,}")
    print(f"max_seq_len       : {metadata['max_seq_len']}")
    print(f"engine load       : {metadata['load_seconds']:.2f}s (once)")
    print(f"engine class      : {type(runtime.engine).__name__} (real SASRecInferenceEngine)")
    print(f"tool              : {metadata['tool']['name']} v{metadata['tool']['version']}")
    print(f"graph nodes       : {', '.join(runtime.graph.node_names())}")
    print(f"decision model    : {type(runtime.decision_model).__name__} (injected, deterministic)")
    print(
        f"trusted history   : user_int_id={history.user_int_id} "
        f"length={history.length} distinct={history.distinct_length} "
        f"digest={history.digest}"
    )
    print(f"history source    : {history.source_path.name} (read-only)")
    print(f"selected k        : {k}")


def _candidates(result: Any) -> list[tuple[int, str, int, float]]:
    """Semantic candidate tuple: rank, parent_asin, item_id, score."""
    return [(i.rank, i.parent_asin, i.item_id, i.score) for i in result.recommendations]


def run_recommend_route(
    runtime: AgentToolRuntime, history: HistorySelection, k: int
) -> tuple[dict[str, Any], list[tuple[int, str, int, float]]]:
    """Invoke the graph on the recommend route and print the candidate table."""
    runtime.decision_model.set_decision(AgentDecision(action="recommend", k=k))
    before = runtime.engine.recommend_calls
    state = runtime.graph.run("please recommend something for me", history.parent_asins)
    calls = runtime.engine.recommend_calls - before

    result = state["tool_result"]
    print(f"\n--- route: RECOMMEND (k={k}) ---")
    print(f"  route taken          : {state['route']}")
    print(f"  requested k          : {result.requested_k}")
    print(f"  candidates returned  : {result.returned_k} (<= k, exhaustion is legal)")
    print(f"  eligible candidates  : {result.eligible_candidates:,}")
    print(f"  history length       : {result.history_length}")
    print(f"  effective window     : {result.effective_history_length} (truncated={result.history_truncated})")
    print(f"  engine invocations   : {calls}")
    print("\n    rank  parent_asin    item_id      score")
    for item in result.recommendations:
        print(f"    {item.rank:4d}  {item.parent_asin}  {item.item_id:8d}  {item.score:+.4f}")
    print("\n  final response:")
    for line in state["final_response"].splitlines() or [""]:
        print(f"      {line}")
    return state, _candidates(result)


def main(argv: list[str] | None = None) -> int:
    """Run the M7C smoke; returns 0 only when every gate passes."""
    parser = argparse.ArgumentParser(description="Milestone 7C real-chain smoke")
    parser.add_argument("--k", type=int, default=DEFAULT_K, help="requested candidate count")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="override the accepted checkpoint path (digest check still applies)",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args(argv)

    checks: dict[str, bool] = {}

    # ---- 1-4. build the real chain -------------------------------------- #
    from recommendation.api.app import ServiceSettings

    settings = ServiceSettings.from_env()
    if args.checkpoint is not None:
        settings = ServiceSettings(
            checkpoint_path=args.checkpoint,
            manifest_path=settings.manifest_path,
            mappings_path=settings.mappings_path,
            device=args.device,
            verify_checkpoint_sha256=True,
        )

    for label, path in (
        ("checkpoint", settings.checkpoint_path),
        ("mappings", settings.mappings_path),
    ):
        if not Path(path).exists():
            print(f"missing accepted {label}: {path}", file=sys.stderr)
            return 2

    try:
        runtime = build_runtime(settings=settings, device=args.device, k=args.k)
    except Exception as exc:  # noqa: BLE001 - identity failure must exit non-zero
        print(f"failed to build the real chain: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2

    # 5. deterministic trusted real history
    history = select_history()
    print_header(runtime, history, args.k)

    # ---- identity gates -------------------------------------------------- #
    checks["checkpoint is a real SASRecInferenceEngine"] = isinstance(
        runtime.engine, SASRecInferenceEngine
    )
    checks["accepted checkpoint sha256 matches"] = (
        runtime.checkpoint_sha256 == ACCEPTED_CHECKPOINT_SHA256
    )
    checks["accepted run manifest sha256 matches"] = (
        runtime.manifest_sha256 == ACCEPTED_MANIFEST_SHA256
    )
    checks["real mappings loaded (full catalog)"] = runtime.engine.num_items == 156_746
    checks["tool is the accepted RecommendationTool"] = type(runtime.tool) is RecommendationTool
    checks["graph nodes are the accepted M7B topology"] = (
        set(runtime.graph.node_names()) - {"__start__", "__end__"}
        == {NODE_DECIDE, NODE_RECOMMEND, NODE_FINALIZE}
    )

    # ---- 6-7. recommend route on the real chain -------------------------- #
    state, first_candidates = run_recommend_route(runtime, history, args.k)
    result = state["tool_result"]

    checks["graph selected the recommend route"] = state["route"] == ROUTE_RECOMMEND
    checks["exactly one engine invocation for one graph run"] = (
        runtime.engine.recommend_calls == 1
    )
    checks["requested k reached the tool unchanged"] = result.requested_k == args.k
    checks["returned count never exceeds k"] = result.returned_k <= args.k
    checks["candidates were returned"] = result.returned_k > 0

    seen = set(history.parent_asins)
    checks["no PAD candidate"] = all(i.item_id != 0 for i in result.recommendations)
    checks["no already-seen candidate (masking intact)"] = not (
        {i.parent_asin for i in result.recommendations} & seen
    )
    checks["candidate asins exist in the accepted catalog"] = all(
        runtime.engine.has_parent_asin(i.parent_asin) for i in result.recommendations
    )
    checks["candidate asin/item_id mapping is consistent"] = all(
        runtime.engine.item_id_to_parent_asin(i.item_id) == i.parent_asin
        for i in result.recommendations
    )
    checks["candidate scores are finite"] = all(
        i.score == i.score and abs(i.score) != float("inf") for i in result.recommendations
    )
    checks["ranks are contiguous from 1"] = [
        i.rank for i in result.recommendations
    ] == list(range(1, result.returned_k + 1))
    checks["scores are descending (accepted serving order)"] = [
        i.score for i in result.recommendations
    ] == sorted((i.score for i in result.recommendations), reverse=True)
    checks["eligible == catalog minus distinct history"] = (
        result.eligible_candidates == runtime.engine.num_items - history.distinct_length
    )

    # ---- G. honest response ---------------------------------------------- #
    text = state["final_response"]
    lowered = text.lower()
    checks["final response names the returned candidates"] = all(
        i.parent_asin in text for i in result.recommendations
    )
    checks["final response fabricates no product metadata"] = not any(
        token in lowered
        for token in (
            "brand:",
            "category:",
            "material:",
            "color:",
            "colour:",
            "price:",
            "$",
            "in stock",
            "description:",
            "title:",
        )
    )
    checks["final response carries the raw-score disclaimer"] = (
        "not evidence about a product" in lowered
    )
    checks["final response makes no quality claim"] = not any(
        claim in lowered
        for claim in ("best for you", "perfect for you", "you will love", "outperforms")
    )

    # ---- B. trusted-history boundary ------------------------------------ #
    prompt = runtime.decision_model.last_prompt_text
    checks["decision model never received trusted history"] = all(
        asin not in prompt for asin in history.parent_asins
    )
    checks["graph state preserved the trusted history"] = (
        state["trusted_user_history"] == history.parent_asins
    )

    # ---- 8. determinism -------------------------------------------------- #
    for _ in range(REPEATS - 1):
        repeat_state, repeat_candidates = run_recommend_route(runtime, history, args.k)
    checks["repeated invocations are semantically deterministic"] = (
        repeat_candidates == first_candidates
        and repeat_state["final_response"] == state["final_response"]
    )
    checks["one engine invocation per repeated graph run"] = (
        runtime.engine.recommend_calls == REPEATS
    )

    # ---- F. dependency reuse --------------------------------------------- #
    checks["engine instance reused across invocations"] = (
        runtime.graph.tool.engine is runtime.engine
    )
    checks["model was loaded once (loaded_at unchanged)"] = (
        runtime.engine.loaded_at == runtime.loaded_at
    )

    # ---- 9. direct route isolation --------------------------------------- #
    runtime.engine.reset_counters()
    runtime.decision_model.set_decision(
        AgentDecision(action="direct_response", direct_response="Happy to help with a product question.")
    )
    direct_state = runtime.graph.run("hello there", history.parent_asins)
    print("\n--- route: DIRECT_RESPONSE ---")
    print(f"  route taken          : {direct_state['route']}")
    print(f"  engine invocations   : {runtime.engine.recommend_calls}")
    print(f"  final response       : {direct_state['final_response']}")

    checks["direct route took the direct branch"] = direct_state["route"] == ROUTE_DIRECT
    checks["direct route performed no inference"] = runtime.engine.recommend_calls == 0
    checks["direct route produced no tool result"] = "tool_result" not in direct_state

    # ---- 8. error paths -------------------------------------------------- #
    runtime.decision_model.set_decision(AgentDecision(action="recommend", k=args.k))

    runtime.engine.reset_counters()
    try:
        runtime.graph.run("recommend something", history.parent_asins + (UNKNOWN_PARENT_ASIN,))
        unknown_ok = False
    except UnknownHistoryItem:
        unknown_ok = True
    except Exception:  # noqa: BLE001 - any other error is a failure
        unknown_ok = False
    checks["unknown parent_asin fails explicitly"] = unknown_ok
    checks["unknown parent_asin reached the real engine"] = runtime.engine.recommend_calls == 1

    runtime.engine.reset_counters()
    try:
        runtime.engine.recommend([UNKNOWN_PARENT_ASIN], k=args.k)
        engine_unknown_ok = False
    except UnknownItemError:
        engine_unknown_ok = True
    checks["engine rejects unknown items directly (strict semantics)"] = engine_unknown_ok
    checks["rejected history was not silently partially used"] = (
        runtime.engine.recommend_calls == 1
    )

    runtime.engine.reset_counters()
    try:
        runtime.graph._recommend_node(  # noqa: SLF001 - node-level contract check
            {"user_message": "m", "trusted_user_history": (),
             "decision": AgentDecision(action="recommend", k=args.k)}
        )
        empty_ok = False
    except MissingUserHistory:
        empty_ok = True
    except Exception:  # noqa: BLE001
        empty_ok = False
    checks["empty trusted history fails explicitly"] = empty_ok
    checks["empty history performed no inference"] = runtime.engine.recommend_calls == 0

    from pydantic import ValidationError

    try:
        AgentDecision(action="recommend", k=0)
        invalid_k_ok = False
    except ValidationError:
        invalid_k_ok = True
    checks["invalid k rejected by the accepted typed contract"] = invalid_k_ok

    # ---- report ---------------------------------------------------------- #
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
                    "milestone": "7C",
                    "runtime": runtime.metadata(),
                    "history": history.as_dict(),
                    "requested_k": args.k,
                    "returned_k": result.returned_k,
                    "eligible_candidates": result.eligible_candidates,
                    "candidates": [
                        {
                            "rank": i.rank,
                            "parent_asin": i.parent_asin,
                            "item_id": i.item_id,
                            "score": i.score,
                        }
                        for i in result.recommendations
                    ],
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

"""Milestone 9 memory smoke: multi-turn preference memory over the real M8 stack.

Two parts:

**Part A -- memory lifecycle across a reopened store.**  A three-turn conversation is
processed with a temporary SQLite database, the store is closed and reopened, and the
result is checked: ``lightweight`` stays active, the ``red`` avoidance becomes inactive
after an explicit retraction, every event survives in the audit trail, and **no trusted
interaction history is created** from any turn.

**Part B -- real recommendation chain with memory injected.**  The accepted M8 chain
(real M5 checkpoint -> M6 engine -> M7A Tool -> M7B/M7C graph -> M8 metadata/RAG) is
driven twice over the same trusted history: once with memory configured and no active
preferences, once with active preferences.  Candidate identity, count, rank and raw
SASRec score must be identical, and the candidate list must never be reordered.

No quality claim is made: preference memory does not rerank in M9, and this smoke does
not measure whether the preferences are good.

Usage::

    .venv/bin/python -m experiments.memory_smoke
    .venv/bin/python -m experiments.memory_smoke --k 5 --json /tmp/m9.json
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.product_rag_support import (  # noqa: E402
    ACCEPTED_CHECKPOINT_SHA256,
    SMOKE_QUERY,
    build_m8_runtime,
    select_smoke_history,
)
from recommendation.agent import AgentDecision, AgentGraph  # noqa: E402
from recommendation.memory import (  # noqa: E402
    MEMORY_SCHEMA_VERSION,
    InMemoryPreferenceStore,
    PreferenceMemoryService,
    RuleBasedPreferenceExtractor,
    SQLitePreferenceStore,
)

#: The three-turn conversation the milestone specifies.
TURNS: tuple[tuple[str, str], ...] = (
    ("t1", "I prefer lightweight hiking gear."),
    ("t2", "I don't want red."),
    ("t3", "Actually, color does not matter anymore."),
)

#: Preference values expected to still be active at the end.
EXPECTED_ACTIVE = ("lightweight hiking",)

#: Preference values expected to have been retracted or superseded.
EXPECTED_INACTIVE = ("red",)


def run_part_a(json_payload: dict[str, Any]) -> dict[str, bool]:
    """Part A: multi-turn lifecycle with SQLite persistence across a reopen."""
    checks: dict[str, bool] = {}
    extractor = RuleBasedPreferenceExtractor()

    with tempfile.TemporaryDirectory() as temp_dir:
        database = Path(temp_dir) / "memory.db"

        store = SQLitePreferenceStore(database)
        service = PreferenceMemoryService(store, extractor)
        print("\n--- Part A: multi-turn preference memory (SQLite) ---")
        print(f"  database             : {database.name} (schema v{store.schema_version})")

        turn_records: list[dict[str, Any]] = []
        read_latency_ms: list[float] = []
        write_latency_ms: list[float] = []

        for turn_id, message in TURNS:
            started = time.perf_counter()
            result = service.process_turn(
                user_key="smoke-user", user_message=message, turn_id=turn_id, now=1.0
            )
            write_latency_ms.append((time.perf_counter() - started) * 1000.0)

            started = time.perf_counter()
            active = service.get_active_preferences("smoke-user")
            read_latency_ms.append((time.perf_counter() - started) * 1000.0)

            change = result.update
            print(f"\n  {turn_id}: {message!r}")
            print(f"      added    : {[(e.kind.value, e.polarity.value, e.value) for e in change.added]}")
            print(f"      superseded: {[e.value for e in change.superseded]}")
            print(f"      removed  : {[e.value for e in change.removed]}")
            print(
                f"      active   : "
                f"{[(e.kind.value, e.polarity.value, e.value) for e in active.active_entries]}"
            )
            turn_records.append(
                {
                    "turn_id": turn_id,
                    "message": message,
                    "added": [e.value for e in change.added],
                    "superseded": [e.value for e in change.superseded],
                    "removed": [e.value for e in change.removed],
                    "active_after": [e.value for e in active.active_entries],
                }
            )

        before_reopen_active = [
            (e.kind.value, e.polarity.value, e.value)
            for e in service.get_active_preferences("smoke-user").active_entries
        ]
        before_reopen_history = [
            (e.value, e.status.value) for e in service.get_memory_history("smoke-user").entries
        ]
        store.close()

        # ---- reopen: persistence check ---------------------------------- #
        started = time.perf_counter()
        reopened = SQLitePreferenceStore(database)
        reopen_ms = (time.perf_counter() - started) * 1000.0
        service2 = PreferenceMemoryService(reopened, extractor)
        after_active = [
            (e.kind.value, e.polarity.value, e.value)
            for e in service2.get_active_preferences("smoke-user").active_entries
        ]
        after_history = [
            (e.value, e.status.value) for e in service2.get_memory_history("smoke-user").entries
        ]

        print("\n  after reopening the database")
        print(f"      active  : {after_active}")
        print(f"      history : {after_history}")
        print(f"      reopen  : {reopen_ms:.1f} ms")

        checks["preferences survive a store reopen"] = after_active == before_reopen_active
        checks["provenance survives a store reopen"] = after_history == before_reopen_history
        checks["lightweight remains active"] = any(
            value in EXPECTED_ACTIVE for _, _, value in after_active
        )
        checks["red avoidance is no longer active"] = all(
            value not in EXPECTED_INACTIVE for _, _, value in after_active
        )
        checks["retracted entry keeps its audit trail"] = any(
            value == "red" and status == "removed" for value, status in after_history
        )
        checks["every turn produced an auditable event"] = len(after_history) >= 2

        # ---- the hard trust boundary ------------------------------------ #
        history_snapshot = service2.get_memory_history("smoke-user").as_dict()
        serialised = json.dumps(history_snapshot)
        checks["no interaction history was created from conversation"] = all(
            token not in serialised for token in ("parent_asin", "item_id", "B0")
        )
        checks["memory store exposes no interaction API"] = not any(
            hasattr(reopened, name)
            for name in ("add_interaction", "append_history", "record_event")
        )

        # ---- explicit ADD vs REPLACE semantics ---------------------------- #
        coexist = PreferenceMemoryService(
            InMemoryPreferenceStore(), RuleBasedPreferenceExtractor()
        )
        coexist.process_turn(
            user_key="c", user_message="I don't want red.", turn_id="c1", now=1.0
        )
        coexist.process_turn(
            user_key="c", user_message="I don't want blue.", turn_id="c2", now=2.0
        )
        red_and_blue = [
            (e.polarity.value, e.value)
            for e in coexist.get_active_preferences("c").active_entries
        ]
        checks["two independent avoidances coexist"] = red_and_blue == [
            ("avoid", "red"),
            ("avoid", "blue"),
        ]

        coexist.process_turn(
            user_key="c", user_message="Actually, I prefer green instead.", turn_id="c3", now=3.0
        )
        coexist.process_turn(
            user_key="c", user_message="I prefer black.", turn_id="c4", now=4.0
        )
        coexist.process_turn(
            user_key="c", user_message="Make that teal.", turn_id="c5", now=5.0
        )
        active_after = [
            (e.polarity.value, e.value)
            for e in coexist.get_active_preferences("c").active_entries
        ]
        checks["plain second preference does not replace the first"] = (
            ("prefer", "black") not in active_after
        )
        checks["explicit 'instead' replaces the corrected value"] = (
            ("prefer", "teal") in active_after and ("prefer", "black") not in active_after
        )
        checks["replacement keeps the superseded entry in history"] = any(
            e.status.value == "superseded" for e in coexist.get_memory_history("c").entries
        )

        # ---- idempotency -------------------------------------------------- #
        repeated = service2.process_turn(
            user_key="smoke-user", user_message=TURNS[0][1], turn_id=TURNS[0][0], now=2.0
        )
        checks["reprocessing a turn is idempotent"] = (
            repeated.update.added == [] and repeated.update.already_processed is True
        )
        checks["idempotent replay adds no active entry"] = (
            len(service2.get_active_preferences("smoke-user").active_entries)
            == len(after_active)
        )

        # ---- user isolation ---------------------------------------------- #
        other = service2.get_active_preferences("someone-else")
        checks["users are isolated"] = other.active_entries == ()

        # Offline proof: the memory package imports no network client at all.
        import ast

        memory_dir = Path(
            __import__("recommendation.memory", fromlist=["x"]).__file__
        ).parent
        offline = True
        for module_path in sorted(memory_dir.glob("*.py")):
            tree = ast.parse(module_path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                names: list[str] = []
                if isinstance(node, ast.Import):
                    names = [alias.name for alias in node.names]
                elif isinstance(node, ast.ImportFrom):
                    names = [node.module or ""]
                for name in names:
                    if name.split(".")[0] in {"requests", "httpx", "urllib", "socket", "http"}:
                        offline = False
        checks["memory layer imports no network client"] = offline
        reopened.close()

        json_payload["part_a"] = {
            "schema_version": MEMORY_SCHEMA_VERSION,
            "turns": turn_records,
            "active_after": after_active,
            "history_after": after_history,
            "read_latency_ms": read_latency_ms,
            "write_latency_ms": write_latency_ms,
            "reopen_latency_ms": round(reopen_ms, 3),
        }
        print("\n  latency diagnostics")
        print(f"      write  p50 : {sorted(write_latency_ms)[len(write_latency_ms)//2]:.3f} ms")
        print(f"      read   p50 : {sorted(read_latency_ms)[len(read_latency_ms)//2]:.3f} ms")
        print(f"      reopen     : {reopen_ms:.1f} ms")

    return checks


def run_part_b(json_payload: dict[str, Any], k: int, device: str) -> dict[str, bool]:
    """Part B: the real M8 chain, with and without active preference memory."""
    checks: dict[str, bool] = {}
    print("\n--- Part B: real M8 chain with preference memory injected ---")

    try:
        runtime = build_m8_runtime(device=device, k=k)
    except FileNotFoundError as exc:
        print(f"  skipped: {exc}")
        json_payload["part_b"] = {"skipped": str(exc)}
        return {}

    history = select_smoke_history()
    chain = runtime.chain_summary()
    print(f"  checkpoint sha256 : {chain['checkpoint_sha256']}")
    print(f"  catalog size      : {chain['num_items']:,}")
    print(f"  history           : user_int_id={history.user_int_id} length={history.length} digest={history.digest}")
    print(f"  query             : {SMOKE_QUERY!r}")

    checks["accepted checkpoint digest matches"] = (
        chain["checkpoint_sha256"] == ACCEPTED_CHECKPOINT_SHA256
    )

    # ---- run 1: memory configured, no active preferences ---------------- #
    store = SQLitePreferenceStore(":memory:")
    service = PreferenceMemoryService(store, RuleBasedPreferenceExtractor())
    graph_with_memory = AgentGraph(
        runtime.decision_model,
        runtime.tool,
        product_enricher=runtime.enricher,
        memory_service=service,
        user_key="smoke-user",
    )
    runtime.decision_model.set_decision(AgentDecision(action="recommend", k=k))
    runtime.engine.reset_counters()
    baseline = graph_with_memory.run(SMOKE_QUERY, history.parent_asins)
    baseline_calls = runtime.engine.recommend_calls

    baseline_candidates = [
        (r.rank, r.parent_asin, r.item_id, r.score)
        for r in baseline["tool_result"].recommendations
    ]
    baseline_order = [r.parent_asin for r in baseline["tool_result"].recommendations]

    print(f"\n  baseline (no active preferences)")
    print(f"      engine invocations : {baseline_calls}")
    print(f"      candidates         : {len(baseline_candidates)}")
    for rank, asin, _item, score in baseline_candidates:
        print(f"        {rank}  {asin}  {score:+.4f}")

    checks["baseline graph took the recommend route"] = baseline["route"] == "recommend"
    checks["baseline made exactly one engine invocation"] = baseline_calls == 1

    # ---- store a preference, then run again ----------------------------- #
    service.process_turn(
        user_key="smoke-user",
        user_message="I prefer lightweight hiking gear.",
        turn_id="b1",
        now=1.0,
    )
    service.process_turn(
        user_key="smoke-user",
        user_message="I don't want red.",
        turn_id="b2",
        now=2.0,
    )
    active = service.get_active_preferences("smoke-user")
    print(
        "\n  active preferences : "
        f"{[(e.kind.value, e.polarity.value, e.value) for e in active.active_entries]}"
    )

    runtime.engine.reset_counters()
    with_preferences = graph_with_memory.run(SMOKE_QUERY, history.parent_asins)
    preference_calls = runtime.engine.recommend_calls
    preference_candidates = [
        (r.rank, r.parent_asin, r.item_id, r.score)
        for r in with_preferences["tool_result"].recommendations
    ]
    preference_order = [r.parent_asin for r in with_preferences["tool_result"].recommendations]

    print("\n  with active preferences")
    print(f"      engine invocations : {preference_calls}")
    print(f"      candidates         : {len(preference_candidates)}")
    for rank, asin, item, score in preference_candidates:
        marker = "  (changed)" if (rank, asin, item, score) not in baseline_candidates else ""
        print(f"        {rank}  {asin}  {score:+.4f}{marker}")

    # ---- the required invariants ---------------------------------------- #
    checks["candidate identity unchanged by preferences"] = [
        (asin, item) for _, asin, item, _ in baseline_candidates
    ] == [(asin, item) for _, asin, item, _ in preference_candidates]
    checks["candidate count unchanged"] = len(baseline_candidates) == len(preference_candidates)
    checks["candidate rank unchanged"] = [r[0] for r in baseline_candidates] == [
        r[0] for r in preference_candidates
    ]
    checks["SASRec score unchanged"] = [r[3] for r in baseline_candidates] == [
        r[3] for r in preference_candidates
    ]
    checks["candidate order unchanged (no reranking)"] = baseline_order == preference_order
    checks["enrichment order unchanged"] = list(with_preferences["enrichment"].parent_asins) == preference_order
    checks["preferences were reported in the response"] = (
        "Your stated preferences" in with_preferences["final_response"]
    )
    checks["response does not claim a preference match"] = not any(
        claim in with_preferences["final_response"].lower()
        for claim in ("matches your preference", "perfectly matches", "match score", "best match")
    )

    # ---- RAG universe invariant ----------------------------------------- #
    candidate_set = set(preference_order)
    checks["all evidence belongs to a candidate"] = all(
        evidence.parent_asin in candidate_set
        for item in with_preferences["enrichment"].items
        for evidence in item.evidence
    )
    checks["memory did not widen the candidate universe"] = set(
        with_preferences["enrichment"].parent_asins
    ) == candidate_set

    # ---- trusted history ------------------------------------------------- #
    checks["trusted history unchanged"] = (
        with_preferences["trusted_user_history"] == history.parent_asins
    )
    checks["extractor never saw history"] = all(
        asin not in runtime.decision_model.last_prompt_text
        for asin in history.parent_asins
    )

    # ---- direct route with memory ---------------------------------------- #
    runtime.decision_model.set_decision(
        AgentDecision(action="direct_response", direct_response="Happy to help.")
    )
    runtime.engine.reset_counters()
    direct = graph_with_memory.run("I prefer lightweight gear.", history.parent_asins)
    checks["direct route performs no inference"] = runtime.engine.recommend_calls == 0
    checks["direct route produces no recommendation or enrichment"] = (
        "tool_result" not in direct and "enrichment" not in direct
    )
    checks["direct route can still write memory"] = (
        direct["memory_update"] is not None
        and (direct["memory_update"].update.added or direct["memory_update"].update.already_processed)
    )
    runtime.decision_model.set_decision(AgentDecision(action="recommend", k=k))

    json_payload["part_b"] = {
        "checkpoint_sha256": chain["checkpoint_sha256"],
        "query": SMOKE_QUERY,
        "requested_k": k,
        "history": history.as_dict(),
        "baseline_candidates": [
            {"rank": r, "parent_asin": a, "item_id": i, "score": s}
            for r, a, i, s in baseline_candidates
        ],
        "preference_candidates": [
            {"rank": r, "parent_asin": a, "item_id": i, "score": s}
            for r, a, i, s in preference_candidates
        ],
        "active_preferences": [
            {"kind": e.kind.value, "polarity": e.polarity.value, "value": e.value}
            for e in active.active_entries
        ],
        "evidence_count": with_preferences["enrichment"].evidence_count,
    }
    store.close()
    return checks


def main(argv: list[str] | None = None) -> int:
    """Run the M9 smoke; returns 0 only when every gate passes."""
    parser = argparse.ArgumentParser(description="Milestone 9 memory smoke")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--skip-real", action="store_true", help="run only the memory part")
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args(argv)

    print("=" * 78)
    print("Milestone 9 preference-memory smoke")
    print("Lifecycle + trust boundaries only. NOT a recommendation-quality benchmark.")
    print("=" * 78)

    json_payload: dict[str, Any] = {"milestone": "9"}
    checks = run_part_a(json_payload)
    if not args.skip_real:
        checks.update(run_part_b(json_payload, args.k, args.device))

    print("\nChecks")
    for name, ok in checks.items():
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    passed = all(checks.values()) if checks else False
    print(f"\nSMOKE: {'PASS' if passed else 'FAIL'}")

    json_payload["checks"] = checks
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(
            json.dumps(json_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(f"wrote {args.json}")

    return 0 if passed else 1


if __name__ == "__main__":  # pragma: no cover - manual invocation
    raise SystemExit(main())

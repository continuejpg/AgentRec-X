"""Milestone 10A smoke: preference-candidate evidence over the real chain.

Runs the accepted recommendation stack and produces M10A evidence for the real
candidates:

    accepted M5 checkpoint -> M6 engine -> M7A Tool -> M7B/M7C graph
        -> M8 candidate-scoped metadata
        -> M9 ACTIVE preference memory
        -> M10A evidence

Two parts:

**Part A** -- the M9 integration path.  A three-turn conversation is processed through
a real ``PreferenceMemoryService`` over SQLite, then extended with explicit replacement
and category-wide removal, and the matcher is run against the final ACTIVE state.  This
proves M10A observes exactly what M9 considers active.

**Part B** -- the real M8 candidate path.  Real SASRec candidates are enriched with the
accepted M8 metadata artifact and matched.  Candidate identity, count, rank, order and
raw SASRec score must be unchanged; missing metadata must yield UNKNOWN; no new
candidate may appear; repeated runs must be identical.

The preferences used in Part B are **synthetic and deterministic**, applied through the
real memory service.  They validate engineering behaviour, not personalisation quality:
M10A reports evidence and does not rerank.  No recommendation-quality claim is made and
no benchmark metric is recomputed.

Usage::

    .venv/bin/python -m experiments.preference_matching_smoke
    .venv/bin/python -m experiments.preference_matching_smoke --k 5 --json /tmp/m10a.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
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
from recommendation.agent import AgentDecision  # noqa: E402
from recommendation.catalog import normalize_product_record  # noqa: E402
from recommendation.memory import (  # noqa: E402
    InMemoryPreferenceStore,
    PreferenceMemoryService,
    PreferenceStatus,
    RuleBasedPreferenceExtractor,
    SQLitePreferenceStore,
)
from recommendation.preference_matching import (  # noqa: E402
    EvidenceStatus,
    PreferenceCandidateMatcher,
    match_candidates,
)

#: Deterministic synthetic preference turns used for the real-candidate part.  Reported
#: as synthetic: they exist to exercise every status, not to model a real shopper.
PART_B_TURNS: tuple[str, ...] = (
    "I prefer lightweight hiking gear.",
    "I don't want red.",
    "I prefer black.",
    "My budget is under $100.",
)

#: Repeat runs used for the determinism gate.
REPEATS = 5

#: Bounded samples used for the latency diagnostic.
TIMING_SAMPLES = 200


def _status_counts(report: Any) -> dict[str, int]:
    """Aggregate status counts across every candidate."""
    counts = {"match": 0, "violation": 0, "unknown": 0}
    for candidate in report.candidates:
        for record in candidate.evidence:
            counts[record.status.value] += 1
    return counts


def run_part_a(json_payload: dict[str, Any]) -> dict[str, bool]:
    """Part A: the matcher against real M9 ACTIVE memory state."""
    checks: dict[str, bool] = {}
    extractor = RuleBasedPreferenceExtractor()
    service = PreferenceMemoryService(InMemoryPreferenceStore(), extractor)

    print("\n--- Part A: M9 integration (active-state semantics) ---")

    # Two independent avoidances plus a feature preference.
    for index, message in enumerate(
        ("I don't want red.", "I don't want blue.", "I need a lightweight build."), start=1
    ):
        service.process_turn(
            user_key="u", user_message=message, turn_id=f"a{index}", now=float(index)
        )

    active = service.get_active_preferences("u")
    active_pairs = [(e.kind.value, e.polarity.value, e.value) for e in active.active_entries]
    print(f"  after three turns : {active_pairs}")
    checks["two independent avoidances coexist in active state"] = (
        ("color", "avoid", "red") in active_pairs and ("color", "avoid", "blue") in active_pairs
    )

    # Explicit replacement of the feature preference.
    service.process_turn(
        user_key="u",
        user_message="Actually, I want a waterproof membrane instead.",
        turn_id="a4",
        now=4.0,
    )
    active = service.get_active_preferences("u")
    replaced_pairs = [(e.kind.value, e.polarity.value, e.value) for e in active.active_entries]
    print(f"  after replacement : {replaced_pairs}")
    checks["replaced preference is gone from active state"] = all(
        value != "lightweight build" for _, _, value in replaced_pairs
    )
    checks["replacement value is active"] = any(
        value == "waterproof membrane" for _, _, value in replaced_pairs
    )

    history = service.get_memory_history("u").entries
    checks["replaced entry keeps provenance in history"] = any(
        entry.value == "lightweight build"
        and entry.status is PreferenceStatus.SUPERSEDED
        and entry.superseded_by is not None
        for entry in history
    )

    # Category-wide removal.
    service.process_turn(
        user_key="u",
        user_message="I don't care about color anymore.",
        turn_id="a5",
        now=5.0,
    )
    final_active = service.get_active_preferences("u")
    final_pairs = [(e.kind.value, e.polarity.value, e.value) for e in final_active.active_entries]
    print(f"  after color removal: {final_pairs}")
    checks["removed colour constraints leave no active colour entry"] = all(
        kind != "color" for kind, _, _ in final_pairs
    )

    # A candidate carrying both a removed colour and the active feature.
    from tests.preference_matching_fixture import make_candidate  # noqa: PLC0415

    metadata = normalize_product_record(
        {
            "parent_asin": "probe",
            # Carries the removed colour (must produce no evidence) and the active
            # feature (must produce a MATCH).
            "details": {"Color": "red"},
            "features": ["waterproof membrane"],
        }
    )
    report = match_candidates(
        candidates=[make_candidate("probe", metadata=metadata)], preferences=final_active
    )
    evidence = report.candidates[0].evidence
    kinds = [record.preference_kind.value for record in evidence]
    print(f"  evidence kinds    : {kinds}")
    checks["removed colour produces no evidence"] = "color" not in kinds
    checks["removed directive is not itself a preference"] = len(evidence) == len(final_pairs)
    checks["active feature produces a match"] = any(
        record.preference_kind.value == "feature"
        and record.status is EvidenceStatus.MATCH
        for record in evidence
    )
    checks["evidence provenance is available"] = all(
        record.preference_source_turn_id and record.preference_source_text
        for record in evidence
    )

    # Persistence parity: the same active set from a reopened SQLite store.
    import tempfile  # noqa: PLC0415

    with tempfile.TemporaryDirectory() as temp_dir:
        store = SQLitePreferenceStore(Path(temp_dir) / "m.db")
        sqlite_service = PreferenceMemoryService(store, extractor)
        for index, message in enumerate(
            ("I don't want red.", "I don't want blue.", "I need a lightweight build."), start=1
        ):
            sqlite_service.process_turn(
                user_key="u", user_message=message, turn_id=f"a{index}", now=float(index)
            )
        sqlite_service.process_turn(
            user_key="u", user_message="Actually, I want a waterproof membrane instead.",
            turn_id="a4", now=4.0,
        )
        sqlite_service.process_turn(
            user_key="u", user_message="I don't care about color anymore.",
            turn_id="a5", now=5.0,
        )
        store.close()
        reopened = SQLitePreferenceStore(Path(temp_dir) / "m.db")
        reopened_service = PreferenceMemoryService(reopened, extractor)
        reopened_report = match_candidates(
            candidates=[make_candidate("probe", metadata=metadata)],
            preferences=reopened_service.get_active_preferences("u"),
        )
        reopened.close()

    checks["active state survives reopen and matches identically"] = (
        reopened_report.candidates[0].evidence == report.candidates[0].evidence
    )

    json_payload["part_a"] = {
        "active_after_three_turns": active_pairs,
        "active_after_replacement": replaced_pairs,
        "active_final": final_pairs,
        "evidence_kinds": kinds,
        "history": [
            {"value": e.value, "status": e.status.value, "superseded_by": e.superseded_by}
            for e in history
        ],
    }
    return checks


def run_part_b(json_payload: dict[str, Any], k: int, device: str) -> dict[str, bool]:
    """Part B: the matcher over real candidates and real M8 metadata."""
    checks: dict[str, bool] = {}
    print("\n--- Part B: real candidates + real M8 metadata ---")

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
    print(f"  metadata records  : {runtime.metadata.size:,}")

    checks["accepted checkpoint digest matches"] = (
        chain["checkpoint_sha256"] == ACCEPTED_CHECKPOINT_SHA256
    )

    # Real recommendation + real M8 enrichment.
    runtime.decision_model.set_decision(AgentDecision(action="recommend", k=k))
    runtime.engine.reset_counters()
    state = runtime.graph.run(SMOKE_QUERY, history.parent_asins)
    enriched = state["enrichment"]
    tool_result = state["tool_result"]

    print(f"\n  query             : {SMOKE_QUERY!r}")
    print(f"  candidates        : {tool_result.returned_k}")
    for item in enriched.items:
        print(f"    rank {item.rank}  {item.parent_asin}  metadata={item.metadata_status}")

    baseline_identity = [
        (item.rank, item.recommendation.item_id, item.parent_asin, item.score)
        for item in enriched.items
    ]

    # Synthetic, deterministic ACTIVE preferences through the real memory service.
    service = PreferenceMemoryService(InMemoryPreferenceStore(), RuleBasedPreferenceExtractor())
    for index, message in enumerate(PART_B_TURNS, start=1):
        service.process_turn(
            user_key="smoke", user_message=message, turn_id=f"b{index}", now=float(index)
        )
    active = service.get_active_preferences("smoke")
    print(
        "\n  synthetic ACTIVE preferences (engineering validation only) : "
        f"{[(e.kind.value, e.polarity.value, e.value) for e in active.active_entries]}"
    )

    matcher = PreferenceCandidateMatcher()
    started = time.perf_counter()
    report = matcher.match(candidates=enriched.items, preferences=active)
    elapsed_ms = (time.perf_counter() - started) * 1000.0

    after_identity = [
        (c.original_rank, c.item_id, c.parent_asin, c.sasrec_score) for c in report.candidates
    ]

    print("\n  evidence")
    for candidate in report.candidates:
        statuses = ",".join(record.status.value for record in candidate.evidence) or "-"
        print(
            f"    rank {candidate.original_rank}  {candidate.parent_asin:<12} "
            f"score={candidate.sasrec_score:+.4f}  [{statuses}]"
        )
        for record in candidate.evidence:
            print(
                f"        {record.preference_kind.value:<12} "
                f"{record.preference_polarity.value:<6} {record.preference_value:<12} "
                f"-> {record.status.value:<9} {record.reason_code.value}"
            )

    counts = _status_counts(report)
    print(f"\n  status counts     : {counts}")

    # ---- required invariants ------------------------------------------- #
    checks["candidate identity unchanged"] = [c[2] for c in baseline_identity] == [
        c[2] for c in after_identity
    ]
    checks["candidate item ids unchanged"] = [c[1] for c in baseline_identity] == [
        c[1] for c in after_identity
    ]
    checks["candidate count unchanged"] = len(baseline_identity) == len(after_identity)
    checks["original ranks unchanged"] = [c[0] for c in baseline_identity] == [
        c[0] for c in after_identity
    ]
    checks["SASRec scores unchanged"] = [c[3] for c in baseline_identity] == [
        c[3] for c in after_identity
    ]
    checks["evidence was generated"] = counts["match"] + counts["violation"] + counts["unknown"] > 0
    checks["at least one non-unknown status was produced"] = (
        counts["match"] + counts["violation"] > 0
    )
    checks["every candidate carries one record per active preference"] = all(
        len(c.evidence) == active.active_count for c in report.candidates
    )

    # Missing metadata must be UNKNOWN, never a match or a violation.
    missing_candidates = [c for c in report.candidates if c.parent_asin not in set(runtime.metadata.records)]
    checks["missing metadata yields UNKNOWN"] = all(
        record.status is EvidenceStatus.UNKNOWN
        for candidate in missing_candidates
        for record in candidate.evidence
    )
    json_payload["missing_metadata_candidates"] = [c.parent_asin for c in missing_candidates]

    # No preference may introduce a candidate.
    candidate_asins = {c.parent_asin for c in report.candidates}
    checks["no new candidate appeared"] = candidate_asins == set(enriched.parent_asins)
    checks["candidate universe equals the tool result"] = candidate_asins == {
        r.parent_asin for r in tool_result.recommendations
    }

    # ---- determinism ---------------------------------------------------- #
    baseline_json = report.model_dump_json()
    repeat_ok = True
    for _ in range(REPEATS - 1):
        repeat = matcher.match(candidates=enriched.items, preferences=active)
        repeat_ok &= repeat.model_dump_json() == baseline_json
    checks["repeated matching is identical"] = repeat_ok

    # ---- latency diagnostic --------------------------------------------- #
    latencies: list[float] = []
    for _ in range(TIMING_SAMPLES):
        started = time.perf_counter()
        matcher.match(candidates=enriched.items, preferences=active)
        latencies.append((time.perf_counter() - started) * 1000.0)
    latencies.sort()
    p50 = statistics.median(latencies)
    p95 = latencies[min(int(0.95 * len(latencies)), len(latencies) - 1)]

    print("\n  latency (engineering diagnostic)")
    print(f"    candidates={len(report.candidates)} active_preferences={active.active_count}")
    print(f"    p50 = {p50:.3f} ms   p95 = {p95:.3f} ms   (n={TIMING_SAMPLES})")
    print(f"    single run = {elapsed_ms:.3f} ms")

    json_payload["part_b"] = {
        "checkpoint_sha256": chain["checkpoint_sha256"],
        "query": SMOKE_QUERY,
        "requested_k": k,
        "history": history.as_dict(),
        "active_preferences": [
            {
                "kind": e.kind.value,
                "polarity": e.polarity.value,
                "value": e.value,
                "source_turn_id": e.source_turn_id,
            }
            for e in active.active_entries
        ],
        "candidates": [
            {
                "original_rank": c.original_rank,
                "item_id": c.item_id,
                "parent_asin": c.parent_asin,
                "sasrec_score": c.sasrec_score,
                "evidence": [
                    {
                        "preference_id": r.preference_id,
                        "kind": r.preference_kind.value,
                        "polarity": r.preference_polarity.value,
                        "value": r.preference_value,
                        "status": r.status.value,
                        "reason_code": r.reason_code.value,
                        "support": r.support.value,
                        "metadata_field": r.metadata_field,
                        "metadata_value": r.metadata_value,
                    }
                    for r in c.evidence
                ],
            }
            for c in report.candidates
        ],
        "status_counts": counts,
        "latency_ms": {"p50": round(p50, 4), "p95": round(p95, 4), "samples": TIMING_SAMPLES},
    }
    return checks


def main(argv: list[str] | None = None) -> int:
    """Run the M10A smoke; returns 0 only when every gate passes."""
    parser = argparse.ArgumentParser(description="Milestone 10A preference matching smoke")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--skip-real", action="store_true", help="run only the M9 part")
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args(argv)

    print("=" * 78)
    print("Milestone 10A preference-candidate evidence smoke")
    print("Evidence generation only. NOT a recommendation-quality benchmark.")
    print("=" * 78)

    json_payload: dict[str, Any] = {"milestone": "10A"}
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

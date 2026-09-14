"""Milestone 10B smoke: deterministic preference-aware reranking.

Three parts, each proving a different property:

**Part A -- canonical policy on hand-authored fixtures.**  The exact behaviours the
milestone specifies: no preferences, all-UNKNOWN, violation-vs-match precedence, match
promotion, original-rank ties, and an item_id fallback case.

**Part B -- M9 lifecycle integration.**  Real ``PreferenceMemoryService`` state drives
real M10A evidence: ADD coexistence, explicit REPLACE, category-wide REMOVE, and the
specific "removal restores the original order" behaviour.

**Part C -- the real chain.**  Real accepted checkpoint -> SASRec candidates -> real M8
metadata -> real M10A evidence -> M10B reranking, plus reranking latency at several
candidate counts.  Preferences here are **synthetic and deterministic**: they exist to
exercise the policy, not to model a real shopper.

No recommendation-quality claim is made.  M10B reports order and adherence diagnostics
only; the accepted M5 benchmark remains frozen and is not recomputed.

Usage::

    .venv/bin/python -m experiments.preference_reranking_smoke
    .venv/bin/python -m experiments.preference_reranking_smoke --k 5 --json /tmp/m10b.json
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
    PreferencePolarity,
    PreferenceKind,
    RuleBasedPreferenceExtractor,
)
from recommendation.preference_matching import match_candidates  # noqa: E402
from recommendation.reranking import (  # noqa: E402
    PreferenceReranker,
    RerankReason,
    rerank_candidates,
)

#: Synthetic deterministic preference statements used for the real-candidate part.
SYNTHETIC_TURNS: tuple[str, ...] = (
    "I don't want red.",
    "I don't want blue.",
    "I prefer black.",
    "My budget is under $100.",
)

#: Repeat runs for the determinism gates.
REPEATS = 10

#: Latency sample size per candidate count.
TIMING_SAMPLES = 300


def make_candidate(asin: str, rank: int, item_id: int, score: float, color: str | None):
    """Build an M8 enrichment item with a colour-carrying metadata record."""
    from tests.preference_matching_fixture import make_candidate as build  # noqa: PLC0415

    metadata = None
    if color is not None:
        metadata = normalize_product_record(
            {"parent_asin": asin, "details": {"Color": color}}
        )
    return build(asin, rank=rank, item_id=item_id, score=score, metadata=metadata)


def part_a_fixtures() -> dict[str, bool]:
    """Part A: the canonical policy on exact hand-authored scenarios."""
    checks: dict[str, bool] = {}
    print("\n--- Part A: canonical policy fixtures ---")

    from tests.preference_matching_fixture import (  # noqa: PLC0415
        make_entry,
        make_snapshot,
    )

    def avoid(value: str, seq: int = 1):
        return make_entry(
            memory_id=f"avoid-{value}",
            kind=PreferenceKind.COLOR,
            value=value,
            polarity=PreferencePolarity.AVOID,
            logical_seq=seq,
        )

    def prefer(value: str, seq: int = 1):
        return make_entry(
            memory_id=f"prefer-{value}",
            kind=PreferenceKind.COLOR,
            value=value,
            polarity=PreferencePolarity.PREFER,
            logical_seq=seq,
        )

    def run(candidates, preferences):
        return rerank_candidates(
            report=match_candidates(candidates=candidates, preferences=preferences)
        )

    # A. No preferences -> 1,2,3,4
    plain = [make_candidate(f"c{i}", i, i, 10.0 - i, "red" if i == 1 else "blue")
             for i in range(1, 5)]
    result = run(plain, make_snapshot())
    print(f"  A no preferences      : {result.original_ranks} -> {result.reranked_ranks}")
    checks["A: no preferences preserves 1,2,3,4"] = result.reranked_ranks == (1, 2, 3, 4)
    checks["A: nothing moved"] = result.moved_count == 0

    # B. All UNKNOWN -> unchanged
    unknown = [make_candidate(f"u{i}", i, i, 10.0 - i, None) for i in range(1, 5)]
    result = run(unknown, make_snapshot(avoid("red")))
    print(f"  B all UNKNOWN         : {result.original_ranks} -> {result.reranked_ranks}")
    checks["B: all UNKNOWN preserves order"] = result.reranked_ranks == (1, 2, 3, 4)
    checks["B: every count is UNKNOWN"] = all(c.unknown_count == 1 for c in result.candidates)

    # C. Rank 1 violates, rank 5 matches
    mixed = [
        make_candidate("bad", 1, 1, 9.0, "red"),
        make_candidate("f2", 2, 2, 8.0, "green"),
        make_candidate("f3", 3, 3, 7.0, "green"),
        make_candidate("f4", 4, 4, 6.0, "green"),
        make_candidate("good", 5, 5, 1.0, "blue"),
    ]
    result = run(mixed, make_snapshot(avoid("red"), prefer("blue", seq=2)))
    print(f"  C rank1 violates      : {result.parent_asins}")
    checks["C: violating rank 1 is demoted"] = result.parent_asins[-1] == "bad"
    checks["C: matching rank 5 is promoted"] = result.parent_asins[0] == "good"
    checks["C: no candidate dropped"] = result.candidate_count == 5

    # D. One violation vs many matches -> the clean candidate wins
    many = normalize_product_record(
        {
            "parent_asin": "many",
            "details": {"Color": "red"},
            "features": [f"token{i}" for i in range(10)],
        }
    )
    from tests.preference_matching_fixture import make_candidate as build  # noqa: PLC0415

    many_candidate = build("many", rank=1, item_id=1, score=9.0, metadata=many)
    clean_candidate = make_candidate("clean", 2, 2, 1.0, "green")
    preferences = make_snapshot(
        avoid("red"),
        *[
            make_entry(
                memory_id=f"feat{index}",
                kind=PreferenceKind.FEATURE,
                value=f"token{index}",
                polarity=PreferencePolarity.PREFER,
                logical_seq=index + 2,
            )
            for index in range(10)
        ],
    )
    result = run([many_candidate, clean_candidate], preferences)
    many_counts = next(c for c in result.candidates if c.parent_asin == "many")
    print(
        f"  D 1 violation/10 match: many(V={many_counts.violation_count},"
        f" M={many_counts.match_count}) rank {result.original_ranks} -> {result.parent_asins}"
    )
    checks["D: violations dominate matches"] = result.parent_asins[0] == "clean"
    checks["D: violating candidate keeps 10 matches"] = many_counts.match_count == 10

    # E. Equal violations, more matches wins
    low = [make_candidate("low", 1, 1, 9.0, "green")]
    high = [make_candidate("high", 9, 9, 1.0, "blue")]
    result = run(low + high, make_snapshot(prefer("blue")))
    print(f"  E equal violations    : {result.parent_asins}")
    checks["E: more matches is promoted"] = result.parent_asins == ("high", "low")

    # F. Equal evidence profile -> original rank
    same = [make_candidate("first", 2, 20, 5.0, "blue"), make_candidate("second", 5, 50, 4.0, "blue")]
    result = run(same, make_snapshot(prefer("blue")))
    print(f"  F equal evidence      : {result.parent_asins} (ranks {result.original_ranks})")
    checks["F: original rank decides the tie"] = result.parent_asins == ("first", "second")

    # G. Missing metadata is neutral, so 0 violations still beats 1 violation
    result = run([make_candidate("bad", 1, 1, 9.0, "red"), make_candidate("u", 2, 2, 1.0, None)],
                 make_snapshot(avoid("red")))
    print(f"  G UNKNOWN vs violation: {result.parent_asins}")
    checks["G: UNKNOWN beats an explicit violation"] = result.parent_asins == ("u", "bad")

    # H. Every candidate carries a machine-readable reason
    checks["H: reasons are machine-readable"] = all(
        isinstance(c.rerank_reason, RerankReason) for c in result.candidates
    )
    return checks


def part_b_lifecycle() -> dict[str, bool]:
    """Part B: M9 ADD / REPLACE / REMOVE observed through M10A and M10B."""
    checks: dict[str, bool] = {}
    print("\n--- Part B: M9 lifecycle integration ---")
    service = PreferenceMemoryService(InMemoryPreferenceStore(), RuleBasedPreferenceExtractor())
    candidates = [
        make_candidate("red-item", 1, 1, 9.0, "red"),
        make_candidate("blue-item", 2, 2, 8.0, "blue"),
        make_candidate("black-item", 3, 3, 7.0, "black"),
    ]

    def current_order():
        report = match_candidates(
            candidates=candidates, preferences=service.get_active_preferences("u")
        )
        return rerank_candidates(report=report)

    service.process_turn(user_key="u", user_message="I don't want red.", turn_id="t1", now=1.0)
    after_add = current_order()
    print(f"  ADD avoid red         : {after_add.parent_asins}")
    checks["ADD: violating candidate demoted"] = after_add.parent_asins[-1] == "red-item"

    service.process_turn(user_key="u", user_message="I don't want blue.", turn_id="t2", now=2.0)
    after_second_add = current_order()
    active_values = [e.value for e in service.get_active_preferences("u").active_entries]
    print(f"  ADD avoid blue        : active={active_values} order={after_second_add.parent_asins}")
    checks["ADD: both avoidances remain active"] = set(active_values) == {"red", "blue"}
    red_record = next(c for c in after_second_add.candidates if c.parent_asin == "red-item")
    checks["ADD: first avoidance is not forgotten"] = red_record.violation_count == 1

    service.process_turn(user_key="u", user_message="I prefer black.", turn_id="t3", now=3.0)
    before_replace = current_order()
    print(f"  prefer black          : {before_replace.parent_asins}")

    service.process_turn(
        user_key="u", user_message="Actually, I prefer blue instead.", turn_id="t4", now=4.0
    )
    after_replace = current_order()
    active_after_replace = [
        (e.kind.value, e.value) for e in service.get_active_preferences("u").active_entries
    ]
    black_record = next(c for c in after_replace.candidates if c.parent_asin == "black-item")
    blue_record = next(c for c in after_replace.candidates if c.parent_asin == "blue-item")
    print(
        f"  REPLACE with blue     : active={active_after_replace} "
        f"order={after_replace.parent_asins}"
    )
    checks["REPLACE: black is no longer active"] = ("color", "black") not in active_after_replace
    checks["REPLACE: black keeps no match signal"] = black_record.match_count == 0
    checks["REPLACE: blue carries the match signal"] = blue_record.match_count == 1
    checks["REPLACE: blue is promoted above black"] = (
        after_replace.parent_asins.index("blue-item")
        < after_replace.parent_asins.index("black-item")
    )

    service.process_turn(
        user_key="u", user_message="I don't care about color anymore.", turn_id="t5", now=5.0
    )
    after_remove = current_order()
    print(f"  REMOVE color          : order={after_remove.parent_asins}")
    checks["REMOVE: no colour evidence remains"] = all(
        c.violation_count == 0 and c.match_count == 0 for c in after_remove.candidates
    )
    checks["REMOVE: original order is restored"] = after_remove.original_ranks == (1, 2, 3)
    checks["REMOVE: nothing moves"] = after_remove.moved_count == 0
    checks["REMOVE: directive is not a preference"] = (
        after_remove.diagnostics.active_preference_count == 0
    )

    return checks


def run_latency(ks: tuple[int, ...]) -> dict[str, Any]:
    """Measure reranking latency at several candidate counts."""
    from tests.preference_matching_fixture import (  # noqa: PLC0415
        make_entry,
        make_snapshot,
    )

    preferences = make_snapshot(
        make_entry(memory_id="v", kind=PreferenceKind.COLOR, value="red",
                   polarity=PreferencePolarity.AVOID, logical_seq=1),
        make_entry(memory_id="m", kind=PreferenceKind.COLOR, value="blue",
                   polarity=PreferencePolarity.PREFER, logical_seq=2),
    )
    rows: list[dict[str, Any]] = []
    print("\n  reranking latency (engineering diagnostic)")
    for count in ks:
        candidates = [
            make_candidate(
                f"s{index}",
                index,
                index,
                float(count - index),
                ["red", "blue", "green", None][index % 4],
            )
            for index in range(1, count + 1)
        ]
        report = match_candidates(candidates=candidates, preferences=preferences)
        samples: list[float] = []
        for _ in range(TIMING_SAMPLES):
            started = time.perf_counter()
            rerank_candidates(report=report)
            samples.append((time.perf_counter() - started) * 1000.0)
        samples.sort()
        p50 = statistics.median(samples)
        p95 = samples[min(int(0.95 * len(samples)), len(samples) - 1)]
        print(f"    candidates={count:<4} p50={p50:.4f} ms  p95={p95:.4f} ms  (n={TIMING_SAMPLES})")
        rows.append({"candidates": count, "p50_ms": round(p50, 5), "p95_ms": round(p95, 5)})
    return {"samples_per_size": TIMING_SAMPLES, "rows": rows}


def part_c_real_chain(k: int, device: str) -> tuple[dict[str, bool], dict[str, Any]]:
    """Part C: the real chain end to end."""
    checks: dict[str, bool] = {}
    payload: dict[str, Any] = {}
    print("\n--- Part C: real chain (checkpoint -> SASRec -> M8 -> M10A -> M10B) ---")

    try:
        runtime = build_m8_runtime(device=device, k=k)
    except FileNotFoundError as exc:
        print(f"  skipped: {exc}")
        return {}, {"skipped": str(exc)}

    history = select_smoke_history()
    chain = runtime.chain_summary()
    runtime.decision_model.set_decision(AgentDecision(action="recommend", k=k))
    runtime.engine.reset_counters()
    state = runtime.graph.run(SMOKE_QUERY, history.parent_asins)
    enriched = state["enrichment"]

    print(f"  checkpoint sha256 : {chain['checkpoint_sha256']}")
    print(f"  candidates        : {len(enriched.items)}")

    service = PreferenceMemoryService(InMemoryPreferenceStore(), RuleBasedPreferenceExtractor())
    for index, message in enumerate(SYNTHETIC_TURNS, start=1):
        service.process_turn(
            user_key="smoke", user_message=message, turn_id=f"s{index}", now=float(index)
        )
    active = service.get_active_preferences("smoke")
    print(
        "  synthetic ACTIVE preferences (engineering validation only) : "
        f"{[(e.kind.value, e.polarity.value, e.value) for e in active.active_entries]}"
    )

    evidence = match_candidates(candidates=enriched.items, preferences=active)
    reranker = PreferenceReranker()
    started = time.perf_counter()
    reranked = reranker.rerank(evidence)
    single_ms = (time.perf_counter() - started) * 1000.0

    print("\n  order (original -> reranked)")
    for candidate in reranked.candidates:
        print(
            f"    {candidate.original_rank} -> {candidate.reranked_rank}  "
            f"{candidate.parent_asin:<12} score={candidate.sasrec_score:+.4f}  "
            f"V={candidate.violation_count} M={candidate.match_count} "
            f"U={candidate.unknown_count}  {candidate.rerank_reason.value}"
        )

    before = [
        (c.original_rank, c.item_id, c.parent_asin, c.sasrec_score) for c in evidence.candidates
    ]
    after = [
        (c.original_rank, c.item_id, c.parent_asin, c.sasrec_score) for c in reranked.candidates
    ]
    checks["checkpoint digest matches"] = chain["checkpoint_sha256"] == ACCEPTED_CHECKPOINT_SHA256
    checks["candidate count unchanged"] = len(before) == len(after)
    checks["candidate identities unchanged"] = (
        sorted((i, a) for _, i, a, _ in before) == sorted((i, a) for _, i, a, _ in after)
    )
    checks["SASRec scores unchanged"] = (
        sorted(s for *_, s in before) == sorted(s for *_, s in after)
    )
    checks["original ranks preserved as fields"] = sorted(r for r, *_ in before) == sorted(
        r for r, *_ in after
    )
    checks["reranked ranks contiguous"] = reranked.reranked_ranks == tuple(
        range(1, len(after) + 1)
    )
    checks["no candidate added or removed"] = set(reranked.parent_asins) == {
        c.parent_asin for c in evidence.candidates
    }

    baseline_json = reranked.model_dump_json()
    repeat_ok = True
    for _ in range(REPEATS - 1):
        repeat = reranker.rerank(
            match_candidates(candidates=enriched.items, preferences=active)
        )
        repeat_ok &= repeat.model_dump_json() == baseline_json
    checks["reranking is deterministic"] = repeat_ok

    # Compare by identity, not by position: the reranked list is in the new order, so
    # zipping it against the original list would compare different candidates.
    original_evidence = {c.parent_asin: c.evidence for c in evidence.candidates}
    checks["evidence is passed through unchanged"] = all(
        c.evidence == original_evidence[c.parent_asin] for c in reranked.candidates
    )
    # M10A does not expose an unknown_count field; count the statuses it does expose.
    from recommendation.preference_matching import EvidenceStatus  # noqa: PLC0415

    unknown_before = sum(
        record.status is EvidenceStatus.UNKNOWN
        for c in evidence.candidates
        for record in c.evidence
    )
    unknown_after = sum(
        record.status is EvidenceStatus.UNKNOWN
        for c in reranked.candidates
        for record in c.evidence
    )
    checks["UNKNOWN evidence was not dropped"] = (
        unknown_before == unknown_after and unknown_before > 0
    )

    print(f"\n  single rerank = {single_ms:.4f} ms")
    print(f"  moved {reranked.moved_count} of {reranked.candidate_count}")
    for row in reranked.diagnostics.top_k:
        print(
            f"    top-{row.k}: violations {row.violations_before} -> {row.violations_after}, "
            f"matches {row.matches_before} -> {row.matches_after}"
        )

    payload = {
        "checkpoint_sha256": chain["checkpoint_sha256"],
        "query": SMOKE_QUERY,
        "requested_k": k,
        "history": history.as_dict(),
        "active_preferences": [
            {"kind": e.kind.value, "polarity": e.polarity.value, "value": e.value}
            for e in active.active_entries
        ],
        "order": [
            {
                "original_rank": c.original_rank,
                "reranked_rank": c.reranked_rank,
                "parent_asin": c.parent_asin,
                "sasrec_score": c.sasrec_score,
                "violation_count": c.violation_count,
                "match_count": c.match_count,
                "unknown_count": c.unknown_count,
                "rerank_reason": c.rerank_reason.value,
            }
            for c in reranked.candidates
        ],
        "moved_count": reranked.moved_count,
        "single_rerank_ms": round(single_ms, 5),
        "top_k": [row.as_dict() for row in reranked.diagnostics.top_k],
    }
    return checks, payload


def main(argv: list[str] | None = None) -> int:
    """Run the M10B smoke; returns 0 only when every gate passes."""
    parser = argparse.ArgumentParser(description="Milestone 10B preference reranking smoke")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--skip-real", action="store_true", help="skip the real-chain part")
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args(argv)

    print("=" * 78)
    print("Milestone 10B deterministic preference-aware reranking smoke")
    print("Ordering policy + preservation only. NOT a recommendation-quality benchmark.")
    print("=" * 78)

    checks: dict[str, bool] = {}
    json_payload: dict[str, Any] = {"milestone": "10B"}

    checks.update(part_a_fixtures())
    checks.update(part_b_lifecycle())
    json_payload["latency"] = run_latency((5, 20, 100))

    if not args.skip_real:
        real_checks, real_payload = part_c_real_chain(args.k, args.device)
        checks.update(real_checks)
        json_payload["real_chain"] = real_payload

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

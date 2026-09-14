"""Milestone 10D smoke: preference reranking integrated into the Agent route.

Formal real-chain integration check::

    real trusted history
        -> AgentGraph (M7B/M7C topology + M8 enrich + M9 memory + M10D nodes)
        -> real RecommendationTool
        -> accepted SASRec best.pt
        -> real M8 candidate-scoped metadata / RAG
        -> active synthetic explicit preference memory (M9 service)
        -> real M10A preference-candidate matcher
        -> accepted M10B deterministic reranker
        -> reranked, grounded final response

It is deliberately separate from the M7B, M7C, M8, M9, M10A, M10B and M10C smokes: this
one proves the *integration*, not the policy (M10B) and not the offline diagnostics
(M10C).  Only the accepted M10A matcher and M10B reranker are used; the graph reimplements
neither.

Synthetic preference fixture
----------------------------
``SYNTHETIC_PREFERENCE_TURNS`` is a **globally fixed synthetic fixture for integration
validation**.  It is declared in this module *before* any candidate output is examined,
it is never derived from the user's real interaction history, and the candidate list is
never used to choose it.  It is not a claim about any real shopper's preferences.

No recommendation-quality claim is made.  Policy adherence and rank movement are
engineering diagnostics; the accepted M5 benchmark is sealed and is not recomputed.
M10C is invoked only *after* the graph run, as test-side validation of the produced
report -- never inside a serving node.

Usage::

    .venv/bin/python -m experiments.agent_reranking_smoke
    .venv/bin/python -m experiments.agent_reranking_smoke --k 5 --query "waterproof hiking boots"
    .venv/bin/python -m experiments.agent_reranking_smoke --json /tmp/m10d.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Sequence

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
    InMemoryPreferenceStore,
    PreferenceMemoryService,
    RuleBasedPreferenceExtractor,
)
from recommendation.preference_matching import PreferenceCandidateMatcher  # noqa: E402
from recommendation.rag import ProductEnricher  # noqa: E402
from recommendation.reranking import PreferenceReranker  # noqa: E402
from recommendation.reranking.evaluation import (  # noqa: E402
    aggregate_requests,
    evaluate_request,
)

#: Synthetic preference fixture for integration validation.  Declared here, before any
#: candidate is inspected; expressed as natural language so the real M9 extraction path
#: is exercised rather than a hand-built snapshot.
SYNTHETIC_PREFERENCE_TURNS: tuple[str, ...] = (
    "I don't want red.",
    "I don't want blue.",
    "I prefer black.",
    "I prefer lightweight hiking gear.",
    "My budget is under $100.",
)

#: Human-readable description of the fixture policy, recorded in the JSON envelope.
PREFERENCE_FIXTURE_POLICY = (
    "A globally fixed synthetic preference fixture (5 statements) is installed into the "
    "real M9 memory service before the recommendation route runs. It is declared before "
    "any candidate output is examined, is never derived from the selected user's real "
    "interaction history, and is not a claim about any real shopper's preferences."
)

#: Repeat runs used for the determinism gate.
REPEATS = 3

#: Latency sample size per node, measured by issuing repeated matcher/reranker calls on
#: the already-computed evidence (the upstream inference cost is not duplicated).
TIMING_SAMPLES = 300


class CountingLookup:
    """Metadata lookup wrapper recording every identifier the enricher asked about.

    Delegates to the **already-loaded** index, so the smoke never reloads metadata; it
    only observes which candidates reach the metadata layer.
    """

    def __init__(self, delegate: Any) -> None:
        self._delegate = delegate
        self.requested: list[str] = []

    @property
    def delegate(self) -> Any:
        """The wrapped lookup, so a caller can prove no second index was constructed."""
        return self._delegate

    def lookup(self, parent_asin: str) -> Any:
        """Record and delegate."""
        self.requested.append(parent_asin)
        return self._delegate.lookup(parent_asin)

    def lookup_many(self, parent_asins: Sequence[str]) -> Any:
        """Record and delegate, preserving alignment."""
        self.requested.extend(parent_asins)
        return self._delegate.lookup_many(parent_asins)

    def __contains__(self, parent_asin: object) -> bool:
        """Delegate membership."""
        return parent_asin in self._delegate


class TimedMatcher:
    """The real M10A matcher plus call counting and per-call latency."""

    def __init__(self) -> None:
        self._delegate = PreferenceCandidateMatcher()
        self.calls = 0
        self.timings_ms: list[float] = []
        self.reports: list[Any] = []

    def match(self, *, candidates: Any, preferences: Any) -> Any:
        """Time the real matcher and return its unchanged output."""
        started = time.perf_counter()
        report = self._delegate.match(candidates=candidates, preferences=preferences)
        self.timings_ms.append((time.perf_counter() - started) * 1000.0)
        self.calls += 1
        self.reports.append(report)
        return report


class TimedReranker:
    """The real M10B reranker plus call counting and per-call latency."""

    def __init__(self) -> None:
        self._delegate = PreferenceReranker()
        self.calls = 0
        self.timings_ms: list[float] = []
        #: The exact evidence report handed in on each call (input identity proof).
        self.received: list[Any] = []
        #: The reranked report produced on each call.
        self.produced: list[Any] = []

    def rerank(self, report: Any) -> Any:
        """Time the real reranker and return its unchanged output."""
        started = time.perf_counter()
        reranked = self._delegate.rerank(report)
        self.timings_ms.append((time.perf_counter() - started) * 1000.0)
        self.calls += 1
        self.received.append(report)
        self.produced.append(reranked)
        return reranked


def _percentile(values: Sequence[float], fraction: float) -> float:
    """Nearest-rank percentile of a non-empty sample."""
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round(fraction * (len(ordered) - 1))))
    return ordered[index]


def _order(state: dict[str, Any]) -> tuple[str, ...]:
    """Candidate identities in the order the final response presents them."""
    import re

    pattern = re.compile(r"^\d+\. (\S+) \(", re.MULTILINE)
    return tuple(match.group(1) for match in pattern.finditer(state["final_response"]))


def main(argv: list[str] | None = None) -> int:
    """Run the integrated M10D smoke; returns 0 only when every gate passes."""
    parser = argparse.ArgumentParser(description="Milestone 10D Agent reranking smoke")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--query", default=SMOKE_QUERY)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args(argv)

    print("=" * 78)
    print("Milestone 10D smoke: preference reranking integrated into the Agent route")
    print("=" * 78)
    print(f"query      : {args.query!r}")
    print(f"k          : {args.k}")
    print(f"device     : {args.device}")
    print(f"fixture    : {PREFERENCE_FIXTURE_POLICY}")

    checks: dict[str, bool] = {}

    try:
        runtime = build_m8_runtime(device=args.device, k=args.k)
    except FileNotFoundError as exc:
        print(f"\nSKIPPED: {exc}")
        print("\nSMOKE: SKIP")
        return 0

    history = select_smoke_history()
    chain = runtime.chain_summary()
    print(f"\ncheckpoint : {chain['checkpoint_sha256']}")
    print(f"history    : {history.as_dict()}")

    # --- 1-6. Build the real chain once, with an observable metadata seam. ------- #
    graph_version = runtime.graph.version
    lookup = CountingLookup(runtime.metadata)
    enricher = ProductEnricher(lookup)
    matcher = TimedMatcher()
    reranker = TimedReranker()

    service = PreferenceMemoryService(InMemoryPreferenceStore(), RuleBasedPreferenceExtractor())
    for index, message in enumerate(SYNTHETIC_PREFERENCE_TURNS, start=1):
        service.process_turn(
            user_key="m10d", user_message=message, turn_id=f"s{index}", now=float(index)
        )
    active = service.get_active_preferences("m10d")
    print(
        "fixture ACTIVE preferences (synthetic, engineering validation only):\n  "
        + "\n  ".join(
            f"{entry.kind.value}: {entry.polarity.value} {entry.value}"
            for entry in active.active_entries
        )
    )

    graph = AgentGraph(
        runtime.decision_model,
        runtime.tool,
        product_enricher=enricher,
        memory_service=service,
        user_key="m10d",
        preference_matcher=matcher,
        reranker=reranker,
    )

    checks["checkpoint digest matches the accepted artifact"] = (
        chain["checkpoint_sha256"] == ACCEPTED_CHECKPOINT_SHA256
    )
    checks["engine, Tool and metadata index constructed once and shared"] = (
        graph.tool is runtime.tool
        and graph.product_enricher is enricher
        and enricher.metadata is lookup
        and lookup.delegate is runtime.metadata
    )
    checks["memory service initialised with the synthetic fixture"] = active.active_count >= 1
    checks["graph exposes the M10D nodes"] = graph.matches_preferences and graph.reranks_preferences

    # --- 7-18. Run the recommendation route and assert the integration invariants. #
    runtime.engine.reset_counters()
    runtime.decision_model.set_decision(AgentDecision(action="recommend", k=args.k))

    started = time.perf_counter()
    state = graph.run(args.query, history.parent_asins)
    graph_seconds = time.perf_counter() - started

    result = state["tool_result"]
    enrichment = state["enrichment"]
    evidence = state["preference_evidence"]
    reranked = state["reranking"]
    request, evaluated_report = evaluate_request(
        report=evidence, label="m10d-real-chain", diagnostics_k=(1, 3, 5)
    )

    print("\n--- candidate order: original -> reranked ---")
    print(
        f"  {'orig':>4} {'rerank':>6}  {'parent_asin':<12} {'sasrec_score':>13}  "
        f"{'match':>5} {'viol':>4} {'unk':>4}"
    )
    for candidate in reranked.candidates:
        print(
            f"  {candidate.original_rank:>4} {candidate.reranked_rank:>6}  "
            f"{candidate.parent_asin:<12} {candidate.sasrec_score:>+13.6f}  "
            f"{candidate.match_count:>5} {candidate.violation_count:>4} "
            f"{candidate.unknown_count:>4}"
        )

    order_before = tuple(r.parent_asin for r in result.recommendations)
    order_after = tuple(c.parent_asin for c in reranked.candidates)
    print(f"\n  candidate order before : {order_before}")
    print(f"  candidate order after  : {order_after}")
    print(f"  moved_count            : {reranked.moved_count}")

    rendered = _order(state)

    checks["recommendation route ran and produced a final response"] = bool(
        state["final_response"]
    )
    checks["Tool, matcher and reranker invoked exactly once"] = (
        runtime.engine.recommend_calls == 1 and matcher.calls == 1 and reranker.calls == 1
    )
    checks["metadata lookup universe equals the Tool candidates"] = set(
        lookup.requested
    ) == set(order_before) and len(lookup.requested) == len(order_before)
    checks["M10A evidence covers only the candidates and active preferences"] = (
        evidence.parent_asins == order_before
        and evidence.active_preference_count == active.active_count
    )
    checks["M10B received exactly that evidence report"] = (
        reranker.received[-1] is evidence and reranker.received[-1] is matcher.reports[-1]
    )
    checks["candidate count unchanged"] = len(order_after) == len(order_before)
    checks["candidate identity multiset unchanged"] = sorted(order_after) == sorted(order_before)
    checks["raw SASRec scores unchanged"] = sorted(
        c.sasrec_score for c in reranked.candidates
    ) == sorted(r.score for r in result.recommendations)
    checks["original ranks retained on every candidate"] = sorted(
        c.original_rank for c in reranked.candidates
    ) == sorted(r.rank for r in result.recommendations)
    checks["reranked ranks form 1..N"] = reranked.reranked_ranks == tuple(
        range(1, len(order_after) + 1)
    )
    checks["final rendered order equals the reranked order"] = rendered == order_after
    checks["original ranks remain visible in the response"] = all(
        f"original SASRec rank {c.original_rank}" in state["final_response"]
        for c in reranked.candidates
    )
    enriched_by_identity = {(i.parent_asin, i.recommendation.item_id): i for i in enrichment.items}
    evidence_by_identity = {(c.parent_asin, c.item_id): c for c in evidence.candidates}
    checks["metadata and evidence stay attached by identity, not by position"] = all(
        (c.parent_asin, c.item_id) in enriched_by_identity
        and c.evidence == evidence_by_identity[(c.parent_asin, c.item_id)].evidence
        and c.sasrec_score == evidence_by_identity[(c.parent_asin, c.item_id)].sasrec_score
        for c in reranked.candidates
    )
    checks["M10C re-derivation reproduces the graph's reranked report"] = (
        evaluated_report.model_dump() == reranked.model_dump()
    )

    # --- 19. Direct route isolation. --------------------------------------------- #
    runtime.decision_model.set_decision(
        AgentDecision(action="direct_response", direct_response="Hello there.")
    )
    lookups_before = len(lookup.requested)
    matcher_calls_before = matcher.calls
    reranker_calls_before = reranker.calls
    runtime.engine.reset_counters()

    direct = graph.run("hello", history.parent_asins)

    checks["direct route performs zero recommendation / metadata / rerank work"] = (
        direct["route"] == "direct"
        and runtime.engine.recommend_calls == 0
        and len(lookup.requested) == lookups_before
        and matcher.calls == matcher_calls_before
        and reranker.calls == reranker_calls_before
        and "reranking" not in direct
    )

    # --- 20. Determinism. -------------------------------------------------------- #
    runtime.decision_model.set_decision(AgentDecision(action="recommend", k=args.k))
    payloads = {reranked.model_dump_json()}
    texts = {state["final_response"]}
    for _ in range(REPEATS - 1):
        repeat = graph.run(args.query, history.parent_asins)
        payloads.add(repeat["reranking"].model_dump_json())
        texts.add(repeat["final_response"])
    checks["repeated runs are semantically deterministic"] = len(payloads) == 1 and len(texts) == 1

    # --- M10C validation, test-side only (never in a serving node). ------------- #
    # The offline evaluator is imported and used *after* the graph run, purely to
    # validate the produced report.  No serving node imports or calls it.
    m10c_aggregate = aggregate_requests([request])
    checks["M10C offline evaluator confirms every graph invariant"] = (
        request.invariants.all_hold
        and request.movement_attribution.accounting_consistent
        and m10c_aggregate["duplicate_original_rank_count"] == 0
        and m10c_aggregate["item_id_fallback_reachable"] is False
        and m10c_aggregate["all_invariants_hold"]
    )

    # --- 37. Latency diagnostics (engineering only, never quality). ------------- #
    matcher_samples: list[float] = []
    reranker_samples: list[float] = []
    for _ in range(TIMING_SAMPLES):
        started = time.perf_counter()
        produced = matcher._delegate.match(  # noqa: SLF001 - timed diagnostics
            candidates=enrichment.items, preferences=active
        )
        matcher_samples.append((time.perf_counter() - started) * 1000.0)
        started = time.perf_counter()
        reranker._delegate.rerank(produced)  # noqa: SLF001 - timed diagnostics
        reranker_samples.append((time.perf_counter() - started) * 1000.0)

    matcher_ms = _percentile(matcher_samples, 0.5)
    reranker_ms = _percentile(reranker_samples, 0.5)
    print("\n--- latency diagnostics (engineering only, not quality metrics) ---")
    print(f"  M10A matcher   p50 {matcher_ms:.3f} ms")
    print(f"  M10B reranker  p50 {reranker_ms:.3f} ms")
    print(f"  M10A+M10B added overhead p50 {matcher_ms + reranker_ms:.3f} ms")
    print(f"  total graph latency (one request, CPU) {graph_seconds * 1000.0:.1f} ms")
    print(f"  graph version {graph_version}")

    # --- Report. ---------------------------------------------------------------- #
    passed = all(checks.values())
    print("\n--- gates ---")
    for name, ok in checks.items():
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    print(f"\nSMOKE: {'PASS' if passed else 'FAIL'}")

    if args.json is not None:
        envelope = {
            "milestone": "10D",
            "graph_version": graph_version,
            "query": args.query,
            "k": args.k,
            "history": history.as_dict(),
            "chain": chain,
            "preference_fixture_policy": PREFERENCE_FIXTURE_POLICY,
            "synthetic_preference_turns": list(SYNTHETIC_PREFERENCE_TURNS),
            "active_preferences": [
                [entry.kind.value, entry.polarity.value, entry.value]
                for entry in active.active_entries
            ],
            "candidate_order_before": list(order_before),
            "candidate_order_after": list(order_after),
            "moved_count": reranked.moved_count,
            "candidates": [
                {
                    "original_rank": c.original_rank,
                    "reranked_rank": c.reranked_rank,
                    "parent_asin": c.parent_asin,
                    "sasrec_score": c.sasrec_score,
                    "match_count": c.match_count,
                    "violation_count": c.violation_count,
                    "unknown_count": c.unknown_count,
                }
                for c in reranked.candidates
            ],
            "m10c_invariants": request.invariants.model_dump(),
            "latency_ms": {
                "m10a_p50": matcher_ms,
                "m10b_p50": reranker_ms,
                "added_overhead_p50": matcher_ms + reranker_ms,
                "graph_total": graph_seconds * 1000.0,
            },
            "checks": checks,
            "passed": passed,
        }
        args.json.write_text(json.dumps(envelope, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {args.json}")

    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())

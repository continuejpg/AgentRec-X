"""Milestone 10C smoke: reranking-policy evaluation over the real chain.

Runs the accepted stack and characterises the accepted M10B policy with M10C
diagnostics::

    real trusted history
        -> real RecommendationTool / SASRec          (M6 / M7A)
        -> real M8 candidate-scoped metadata         (frozen boundary)
        -> synthetic deterministic ACTIVE preferences (M9 service)
        -> real M10A evidence
        -> accepted M10B reranker
        -> M10C diagnostics                          (observational)

Two suites:

**Suite S** -- the ten required synthetic policy scenarios (S1-S10), which are where the
policy's behaviour is actually interpretable because the evidence is controlled.

**Suite R** -- a deterministic real cohort: the first N eligible users under the accepted
stable ordering, each matched against a **globally fixed** synthetic preference fixture.
The fixture is declared before any candidate output is examined and is never derived from
user history or from candidate metadata.

No timing or machine metadata enters the deterministic payload, so repeated runs produce
the same digest (the M8 artifact-reproducibility lesson).  The accepted M5 benchmark is
not recomputed, and no relevance/quality claim is made.

Usage::

    .venv/bin/python -m experiments.reranking_evaluation_smoke
    .venv/bin/python -m experiments.reranking_evaluation_smoke --cohort 20 --k 5
    .venv/bin/python -m experiments.reranking_evaluation_smoke --json /tmp/m10c.json
"""

from __future__ import annotations

import argparse
import hashlib
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
    RuleBasedPreferenceExtractor,
)
from recommendation.preference_matching import match_candidates  # noqa: E402
from recommendation.reranking import PreferenceReranker  # noqa: E402
from recommendation.reranking.evaluation import (  # noqa: E402
    build_report,
    evaluate,
    evaluate_request,
)
from recommendation.reranking.evaluation_schemas import (  # noqa: E402
    EVALUATION_SCHEMA_VERSION,
    DEFAULT_DIAGNOSTIC_K,
)

#: Default cohort size.  Chosen for runtime practicality: the expensive part is
#: per-user SASRec inference, and 20 requests is enough to characterise policy
#: behaviour while keeping the smoke under a minute on CPU.
DEFAULT_COHORT = 20

#: The globally fixed synthetic preference fixture applied to EVERY cohort user.
#:
#: Declared here, before any candidate is inspected, and deliberately expressed as
#: natural-language statements so it exercises the real M9 extraction path.  It is not
#: derived from user history or from candidate metadata.
SYNTHETIC_PREFERENCE_TURNS: tuple[str, ...] = (
    "I don't want red.",
    "I don't want blue.",
    "I prefer black.",
    "I prefer lightweight hiking gear.",
    "My budget is under $100.",
)

#: Preference fixture policy description, recorded in the report.
PREFERENCE_FIXTURE_POLICY = (
    "A single globally fixed set of synthetic preference statements is applied to every "
    "cohort user, in the same order, independent of that user's history and of the "
    "returned candidates. No real user preference is inferred."
)

#: Repeat count for the determinism gate.
REPEATS = 3


# --------------------------------------------------------------------------- #
# Suite S -- required synthetic scenarios
# --------------------------------------------------------------------------- #


def _metadata(asin: str, *, color: str | None = None, features: tuple[str, ...] = (),
              price: float | None = None):
    """Normalized metadata with the requested attributes."""
    payload: dict[str, object] = {"parent_asin": asin}
    if color is not None:
        payload["details"] = {"Color": color}
    if features:
        payload["features"] = list(features)
    if price is not None:
        payload["price"] = price
    return normalize_product_record(payload)


def _scenario_candidates(colors: dict[str, str | None], *, features=None, prices=None):
    """Build the standard five-candidate scenario set ``A..E``."""
    from tests.preference_matching_fixture import make_candidate  # noqa: PLC0415

    features = features or {}
    prices = prices or {}
    rows = []
    for position, asin in enumerate(["A", "B", "C", "D", "E"], start=1):
        color = colors.get(asin)
        feature = features.get(asin, ())
        price = prices.get(asin)
        metadata = None
        if color is not None or feature or price is not None:
            metadata = _metadata(asin, color=color, features=feature, price=price)
        rows.append(
            make_candidate(
                asin, rank=position, item_id=position, score=float(10 - position), metadata=metadata
            )
        )
    return rows


def build_scenarios():
    """Build the ten required synthetic scenarios: ``(label, candidates, snapshot)``."""
    from tests.preference_matching_fixture import (  # noqa: PLC0415
        make_candidate,
        make_entry,
        make_snapshot,
    )
    from recommendation.memory.schemas import (  # noqa: PLC0415
        PreferenceKind,
        PreferencePolarity,
    )

    def avoid(value: str, seq: int = 1, kind=PreferenceKind.COLOR):
        return make_entry(
            memory_id=f"avoid-{kind.value}-{value}",
            kind=kind,
            value=value,
            polarity=PreferencePolarity.AVOID,
            logical_seq=seq,
        )

    def prefer(value: str, seq: int = 1, kind=PreferenceKind.COLOR):
        return make_entry(
            memory_id=f"prefer-{kind.value}-{value}",
            kind=kind,
            value=value,
            polarity=PreferencePolarity.PREFER,
            logical_seq=seq,
        )

    scenarios: list[tuple[str, list[Any], Any]] = []

    # S1 -- no preferences
    scenarios.append(
        (
            "S1_no_preferences",
            _scenario_candidates({"A": "red", "B": "blue", "C": "green", "D": None, "E": "green"}),
            make_snapshot(),
        )
    )

    # S2 -- all UNKNOWN (no metadata at all)
    scenarios.append(
        (
            "S2_all_unknown",
            _scenario_candidates({"A": None, "B": None, "C": None, "D": None, "E": None}),
            make_snapshot(avoid("red")),
        )
    )

    # S3 -- one top-ranked violation
    scenarios.append(
        (
            "S3_top_ranked_violation",
            _scenario_candidates({"A": "red", "B": "green", "C": "green", "D": "green", "E": "green"}),
            make_snapshot(avoid("red")),
        )
    )

    # S4 -- lower-ranked strong match (equal violations)
    scenarios.append(
        (
            "S4_lower_ranked_match",
            _scenario_candidates({"A": "green", "B": "green", "C": "green", "D": "green", "E": "blue"}),
            make_snapshot(prefer("blue")),
        )
    )

    # S5 -- violation dominates matches
    many = make_candidate(
        "many",
        rank=1,
        item_id=1,
        score=9.0,
        metadata=_metadata("many", color="red", features=tuple(f"token{i}" for i in range(10))),
    )
    clean = make_candidate(
        "clean", rank=2, item_id=2, score=1.0, metadata=_metadata("clean", color="green")
    )
    scenarios.append(
        (
            "S5_violation_dominates_matches",
            [many, clean],
            make_snapshot(
                avoid("red"),
                *[
                    prefer(f"token{index}", seq=index + 2, kind=PreferenceKind.FEATURE)
                    for index in range(10)
                ],
            ),
        )
    )

    # S6 -- multiple independent avoid constraints
    scenarios.append(
        (
            "S6_independent_avoids",
            _scenario_candidates({"A": "red", "B": "blue", "C": "red", "D": "green", "E": "green"}),
            make_snapshot(avoid("red", seq=1), avoid("blue", seq=2)),
        )
    )

    # S7 -- REPLACE (prefer black -> prefer blue instead)
    service = PreferenceMemoryService(InMemoryPreferenceStore(), RuleBasedPreferenceExtractor())
    service.process_turn(user_key="s7", user_message="I prefer black.", turn_id="t1", now=1.0)
    service.process_turn(
        user_key="s7", user_message="Actually, I prefer blue instead.", turn_id="t2", now=2.0
    )
    scenarios.append(
        (
            "S7_replace_semantics",
            _scenario_candidates({"A": "black", "B": "blue", "C": "green", "D": "green", "E": "green"}),
            service.get_active_preferences("s7"),
        )
    )

    # S8 -- REMOVE (avoid red, then remove colour)
    service8 = PreferenceMemoryService(InMemoryPreferenceStore(), RuleBasedPreferenceExtractor())
    service8.process_turn(user_key="s8", user_message="I don't want red.", turn_id="t1", now=1.0)
    service8.process_turn(
        user_key="s8", user_message="I don't care about color anymore.", turn_id="t2", now=2.0
    )
    scenarios.append(
        (
            "S8_remove_semantics",
            _scenario_candidates({"A": "red", "B": "blue", "C": "green", "D": "green", "E": "green"}),
            service8.get_active_preferences("s8"),
        )
    )

    # S9 -- sparse evidence
    scenarios.append(
        (
            "S9_sparse_evidence",
            _scenario_candidates({"A": "red", "B": None, "C": None, "D": None, "E": None}),
            make_snapshot(avoid("red")),
        )
    )

    # S10 -- exact tie
    scenarios.append(
        (
            "S10_exact_tie",
            _scenario_candidates({"A": "blue", "B": "blue", "C": "blue", "D": "blue", "E": "blue"}),
            make_snapshot(prefer("blue")),
        )
    )

    return scenarios


def run_scenario_suite(diagnostics_k) -> tuple[list[Any], list[str]]:
    """Run the required synthetic scenarios and print a compact table."""
    print("\n--- Suite S: required synthetic policy scenarios ---")
    requests = []
    for label, candidates, preferences in build_scenarios():
        report = match_candidates(candidates=candidates, preferences=preferences)
        request, reranked = evaluate_request(
            report=report, label=label, diagnostics_k=diagnostics_k
        )
        requests.append(request)
        order = "".join(c.parent_asin for c in reranked.candidates)
        top1 = next((row for row in request.adherence if row.k == 1), None)
        print(
            f"  {label:<32} order={order:<6} moved={request.displacement.moved_count}/"
            f"{request.displacement.candidate_count} max_abs={request.displacement.max_abs_delta}"
            + (
                f" top1 V:{top1.violations_before}->{top1.violations_after}"
                f" M:{top1.matches_before}->{top1.matches_after}"
                if top1
                else ""
            )
        )
    return requests, [request.label for request in requests]


# --------------------------------------------------------------------------- #
# Suite R -- deterministic real cohort
# --------------------------------------------------------------------------- #


def cohort_history_digest(histories) -> str:
    """Stable digest over the selected cohort histories (no raw history printed)."""
    joined = "\x1f".join("\x1e".join(history) for history in histories)
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()[:16]


def select_cohort(sequences_path: Path, size: int) -> list[tuple[int, str, list[str]]]:
    """Deterministically select the first ``size`` eligible users.

    Selection walks the accepted sequences artifact in stored order (ascending
    ``user_int_id``) and takes the first ``size`` records with at least three
    interactions.  The history is ``parent_asins[:-2] + [parent_asins[-2]]``, matching
    the accepted M7C convention, and is never chosen based on model output.
    """
    payload = json.loads(sequences_path.read_text(encoding="utf-8"))
    cohort: list[tuple[int, str, list[str]]] = []
    for record in payload["sequences"]:
        asins = record["parent_asins"]
        if len(asins) < 3:
            continue
        cohort.append((int(record["user_int_id"]), str(record["user_id"]), list(asins[:-2]) + [asins[-2]]))
        if len(cohort) >= size:
            break
    return cohort


def fixed_preference_snapshot(user_key: str):
    """Apply the globally fixed synthetic fixture through the real M9 service."""
    service = PreferenceMemoryService(InMemoryPreferenceStore(), RuleBasedPreferenceExtractor())
    for index, message in enumerate(SYNTHETIC_PREFERENCE_TURNS, start=1):
        service.process_turn(
            user_key=user_key, user_message=message, turn_id=f"fx{index}", now=float(index)
        )
    return service.get_active_preferences(user_key)


def run_real_cohort(runtime, cohort_size: int, k: int, diagnostics_k):
    """Run the real cohort: real candidates, real metadata, fixed synthetic preferences."""
    print(f"\n--- Suite R: deterministic real cohort (n={cohort_size}) ---")
    sequences = REPO_ROOT / "data" / "processed" / "Sports_and_Outdoors_sequences.json"
    if not sequences.exists():
        print(f"  skipped: {sequences} not present")
        return [], [], {}

    cohort = select_cohort(sequences, cohort_size)
    histories = [history for _, _, history in cohort]
    digest = cohort_history_digest(histories)
    print(f"  selection  : first {len(cohort)} eligible users in stored order (ascending user_int_id)")
    print(f"  history    : digest={digest} (lengths {min(len(h) for h in histories)}..{max(len(h) for h in histories)})")

    preferences = fixed_preference_snapshot("cohort")
    print(
        "  fixture    : "
        f"{[(e.kind.value, e.polarity.value, e.value) for e in preferences.active_entries]} (synthetic)"
    )

    runtime.decision_model.set_decision(AgentDecision(action="recommend", k=k))
    reranker = PreferenceReranker()
    requests = []
    diagnostics_latency: list[float] = []
    violations_before = violations_after = matches_before = matches_after = 0
    unknown_before = unknown_after = 0

    for user_int_id, user_id, history in cohort:
        runtime.engine.reset_counters()
        state = runtime.graph.run(SMOKE_QUERY, history)
        enriched = state["enrichment"]
        report = match_candidates(candidates=enriched.items, preferences=preferences)
        started = time.perf_counter()
        request, _ = evaluate_request(
            report=report,
            label=f"user_{user_int_id}",
            reranker=reranker,
            diagnostics_k=diagnostics_k,
        )
        diagnostics_latency.append((time.perf_counter() - started) * 1000.0)
        requests.append(request)

        top1 = next((row for row in request.adherence if row.k == 1), None)
        if top1 is not None:
            violations_before += top1.violations_before
            violations_after += top1.violations_after
            matches_before += top1.matches_before
            matches_after += top1.matches_after
            unknown_before += top1.unknown_before
            unknown_after += top1.unknown_after

    moved = sum(1 for request in requests if request.moved)
    print(f"  requests with movement : {moved} / {len(requests)}")
    print(
        "  top-1 adherence        : "
        f"violations {violations_before} -> {violations_after}, "
        f"matches {matches_before} -> {matches_after}, "
        f"UNKNOWN {unknown_before} -> {unknown_after}"
    )

    meta = {
        "cohort_digest": digest,
        "cohort_user_int_ids": [entry[0] for entry in cohort],
        "diagnostics_latency_ms": {
            "p50": round(statistics.median(diagnostics_latency), 5) if diagnostics_latency else 0.0,
            "p95": round(
                sorted(diagnostics_latency)[
                    min(int(0.95 * len(diagnostics_latency)), len(diagnostics_latency) - 1)
                ],
                5,
            )
            if diagnostics_latency
            else 0.0,
            "total": round(sum(diagnostics_latency), 3),
            "samples": len(diagnostics_latency),
        },
        "top1_adherence_totals": {
            "violations_before": violations_before,
            "violations_after": violations_after,
            "matches_before": matches_before,
            "matches_after": matches_after,
            "unknown_before": unknown_before,
            "unknown_after": unknown_after,
        },
    }
    return requests, [request.label for request in requests], meta


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #


def main(argv: list[str] | None = None) -> int:
    """Run the M10C smoke; returns 0 only when every gate passes."""
    parser = argparse.ArgumentParser(description="Milestone 10C reranking evaluation smoke")
    parser.add_argument("--cohort", type=int, default=DEFAULT_COHORT)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--skip-real", action="store_true", help="run only Suite S")
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args(argv)

    print("=" * 78)
    print("Milestone 10C reranking policy evaluation smoke")
    print("Policy/adherence diagnostics only. NOT a recommendation-quality benchmark.")
    print("=" * 78)

    diagnostics_k = DEFAULT_DIAGNOSTIC_K
    checks: dict[str, bool] = {}

    scenario_requests, scenario_labels = run_scenario_suite(diagnostics_k)
    checks["all scenario invariants hold"] = all(
        request.invariants.all_hold for request in scenario_requests
    )
    checks["scenario policy order valid"] = all(
        request.policy_consistency.order_valid for request in scenario_requests
    )
    checks["scenario violation protection holds"] = all(
        request.violation_protection.holds for request in scenario_requests
    )

    checks["scenario movement accounting consistent"] = all(
        request.movement_attribution.accounting_consistent for request in scenario_requests
    )
    checks["scenario item_id fallback never used in movement"] = all(
        request.movement_attribution.item_id_tiebreak == 0 for request in scenario_requests
    )
    checks["scenario item_id fallback never used in positions"] = all(
        request.position_reasons.item_id_tiebreak == 0 for request in scenario_requests
    )
    checks["scenario no duplicate original ranks"] = all(
        len({row.original_rank for row in request.candidate_coverage})
        == len(request.candidate_coverage)
        for request in scenario_requests
    )

    scenario_aggregate = build_report(
        requests=scenario_requests,
        policy="violation_count ASC, match_count DESC, original_rank ASC, item_id ASC",
        cohort_rule="required synthetic policy scenarios S1-S10",
        cohort_size=len(scenario_requests),
        candidate_k=5,
        diagnostics_k=diagnostics_k,
        preference_fixture_policy="Each scenario declares its own active preference set.",
    )
    checks["scenario aggregate invariants hold"] = scenario_aggregate.aggregate[
        "all_invariants_hold"
    ]

    # ---- named scenario assertions ------------------------------------- #
    by_label = {request.label: request for request in scenario_requests}
    checks["S1 no preferences: no movement"] = by_label["S1_no_preferences"].displacement.moved_count == 0
    checks["S2 all UNKNOWN: no movement"] = by_label["S2_all_unknown"].displacement.moved_count == 0
    checks["S3 violation is demoted"] = by_label["S3_top_ranked_violation"].displacement.demoted_count >= 1
    checks["S4 match is promoted"] = by_label["S4_lower_ranked_match"].displacement.promoted_count >= 1
    checks["S5 violation dominates matches"] = (
        by_label["S5_violation_dominates_matches"].adherence[0].violations_after == 0
    )
    checks["S6 independent avoids both active"] = (
        by_label["S6_independent_avoids"].active_preference_count == 2
    )
    checks["S7 replace leaves one preference"] = (
        by_label["S7_replace_semantics"].active_preference_count == 1
    )
    checks["S8 remove leaves no preference"] = (
        by_label["S8_remove_semantics"].active_preference_count == 0
    )
    checks["S8 remove: no movement"] = by_label["S8_remove_semantics"].displacement.moved_count == 0
    checks["S9 sparse: only known violation moves"] = (
        by_label["S9_sparse_evidence"].displacement.demoted_count == 1
    )
    checks["S10 tie: order valid under the canonical key"] = (
        by_label["S10_exact_tie"].policy_consistency.order_valid
    )
    checks["S10 tie: positions explained by ordinal rank"] = (
        by_label["S10_exact_tie"].position_reasons.ordinal_rank
        == by_label["S10_exact_tie"].candidate_count - 1
    )
    checks["S10 tie: item_id never used"] = (
        by_label["S10_exact_tie"].position_reasons.item_id_tiebreak == 0
    )

    real_meta: dict[str, Any] = {}
    real_requests: list[Any] = []
    if not args.skip_real:
        try:
            runtime = build_m8_runtime(device=args.device, k=args.k)
        except FileNotFoundError as exc:
            print(f"  real cohort skipped: {exc}")
            runtime = None
        if runtime is not None:
            chain = runtime.chain_summary()
            checks["accepted checkpoint digest matches"] = (
                chain["checkpoint_sha256"] == ACCEPTED_CHECKPOINT_SHA256
            )
            real_requests, _, real_meta = run_real_cohort(
                runtime, args.cohort, args.k, diagnostics_k
            )
            checks["real cohort invariants hold"] = all(
                request.invariants.all_hold for request in real_requests
            )
            checks["real cohort policy order valid"] = all(
                request.policy_consistency.order_valid for request in real_requests
            )
            checks["real cohort violation protection holds"] = all(
                request.violation_protection.holds for request in real_requests
            )

    real_aggregate = None
    if real_requests:
        real_aggregate = build_report(
            requests=real_requests,
            policy="violation_count ASC, match_count DESC, original_rank ASC, item_id ASC",
            starting_commit="d020d093ccd7dc354291eafad3f6e84dc8943514",
            cohort_rule=(
                "first N eligible users in accepted sequences order (ascending user_int_id); "
                "history = parent_asins[:-2] + [parent_asins[-2]]"
            ),
            cohort_size=len(real_requests),
            candidate_k=args.k,
            diagnostics_k=diagnostics_k,
            preference_fixture_policy=PREFERENCE_FIXTURE_POLICY,
            artifacts={
                "checkpoint_sha256": ACCEPTED_CHECKPOINT_SHA256,
                "cohort_history_digest": real_meta.get("cohort_digest", ""),
                "cohort_user_int_ids": real_meta.get("cohort_user_int_ids", []),
            },
        )
        checks["real cohort aggregate invariants hold"] = real_aggregate.aggregate[
            "all_invariants_hold"
        ]
        checks["real cohort movement accounting consistent"] = all(
            request.movement_attribution.accounting_consistent for request in real_requests
        )
        checks["real cohort item_id fallback never used"] = all(
            request.movement_attribution.item_id_tiebreak == 0
            and request.position_reasons.item_id_tiebreak == 0
            for request in real_requests
        )
        checks["real cohort duplicate original rank count is zero"] = (
            real_aggregate.aggregate["duplicate_original_rank_count"] == 0
        )
        checks["real cohort item_id fallback unreachable"] = (
            real_aggregate.aggregate["item_id_fallback_reachable"] is False
        )

        # ---- determinism over the real cohort ---------------------------- #
        repeat_payloads = []
        for _ in range(REPEATS - 1):
            repeated, _, _ = run_real_cohort(runtime, args.cohort, args.k, diagnostics_k)
            repeat_payloads.append(
                json.dumps([r.model_dump() for r in repeated], sort_keys=True)
            )
        baseline_payload = json.dumps(
            [r.model_dump() for r in real_requests], sort_keys=True
        )
        checks["real cohort evaluation is deterministic"] = all(
            payload == baseline_payload for payload in repeat_payloads
        )
        checks["report digest is stable"] = (
            real_aggregate.digest()
            == build_report(
                requests=real_requests,
                policy="violation_count ASC, match_count DESC, original_rank ASC, item_id ASC",
                starting_commit="d020d093ccd7dc354291eafad3f6e84dc8943514",
                cohort_rule=(
                    "first N eligible users in accepted sequences order (ascending user_int_id); "
                    "history = parent_asins[:-2] + [parent_asins[-2]]"
                ),
                cohort_size=len(real_requests),
                candidate_k=args.k,
                diagnostics_k=diagnostics_k,
                preference_fixture_policy=PREFERENCE_FIXTURE_POLICY,
                artifacts={
                    "checkpoint_sha256": ACCEPTED_CHECKPOINT_SHA256,
                    "cohort_history_digest": real_meta.get("cohort_digest", ""),
                    "cohort_user_int_ids": real_meta.get("cohort_user_int_ids", []),
                },
            ).digest()
        )

    # ---- report ---------------------------------------------------------- #
    print("\nScenario aggregate")
    for key, value in scenario_aggregate.aggregate.items():
        if key in {"adherence_at_k", "top_k_overlap", "attribution"}:
            continue
        print(f"  {key}: {value}")
    print(f"  position_reasons: {scenario_aggregate.aggregate['position_reasons']}")
    print(f"  movement_attribution: {scenario_aggregate.aggregate['movement_attribution']}")
    print(f"  duplicate_original_rank_count: {scenario_aggregate.aggregate['duplicate_original_rank_count']}")

    if real_aggregate is not None:
        print("\nReal cohort aggregate")
        for key, value in real_aggregate.aggregate.items():
            if key in {"adherence_at_k", "top_k_overlap", "attribution"}:
                continue
            print(f"  {key}: {value}")
        print(f"  position_reasons: {real_aggregate.aggregate['position_reasons']}")
        print(f"  movement_attribution: {real_aggregate.aggregate['movement_attribution']}")
        print(f"  duplicate_original_rank_count: {real_aggregate.aggregate['duplicate_original_rank_count']}")
        print(f"  coverage: {real_aggregate.aggregate['coverage']['candidates']}")
        print(f"  top_k_overlap: {real_aggregate.aggregate['top_k_overlap']}")
        print(f"  adherence_at_k: {real_aggregate.aggregate['adherence_at_k']}")
        print(f"  top-1 adherence totals: {real_meta.get('top1_adherence_totals')}")
        print(f"  evaluation latency: {real_meta.get('diagnostics_latency_ms')}")
        print(f"  deterministic digest: {real_aggregate.digest()[:16]}")

    print("\nChecks")
    for name, ok in checks.items():
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    passed = all(checks.values()) if checks else False
    print(f"\nSMOKE: {'PASS' if passed else 'FAIL'}")

    if args.json is not None:
        payload = {
            "schema_version": EVALUATION_SCHEMA_VERSION,
            "milestone": "10C",
            "scenario_report": scenario_aggregate.as_dict(),
            "scenario_digest": scenario_aggregate.digest(),
            "real_cohort_report": None if real_aggregate is None else real_aggregate.as_dict(),
            "real_cohort_digest": None if real_aggregate is None else real_aggregate.digest(),
            "checks": checks,
        }
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(f"wrote {args.json}")

    return 0 if passed else 1


if __name__ == "__main__":  # pragma: no cover - manual invocation
    raise SystemExit(main())

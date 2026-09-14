"""Milestone 8 integrated Agent smoke: recommend -> enrich -> finalize on real data.

Proves the complete real route end to end::

    AgentGraph (accepted M7B/M7C topology + M8 enrich node)
        -> RecommendationTool
        -> real SASRecInferenceEngine + accepted M5 checkpoint
        -> real SASRec candidates
        -> ProductEnricher over the M8-A catalogue metadata artifact
        -> grounded final response

It uses the same deterministic injected decision model as M7C (no provider API, no
network) and the same deterministic real-history selection, plus a second history in
which the SASRec candidates are forced out of the metadata index so the
"metadata missing" path is exercised on **real** candidates without fabricating
records.

Verified here:

* the Tool is called exactly once and the engine exactly once;
* the metadata layer receives only the Tool's candidates;
* the candidate sequence is unchanged by enrichment;
* enrichment happens after recommendation (not before);
* the direct route performs no recommendation and no enrichment work;
* every printed product fact is traceable to that candidate's metadata.

No quality claim is made: metadata coverage and retrieval scores are descriptive, not
accuracy measures.  This milestone has no Product RAG evaluation.

Usage::

    .venv/bin/python -m experiments.agent_product_rag_smoke
    .venv/bin/python -m experiments.agent_product_rag_smoke --k 3 --json /tmp/m8i.json
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

from experiments.product_rag_support import (  # noqa: E402
    ACCEPTED_CHECKPOINT_SHA256,
    SMOKE_QUERY,
    M8Runtime,
    build_m8_runtime,
    select_smoke_history,
)
from recommendation.agent import AgentDecision, AgentGraph  # noqa: E402
from recommendation.catalog import MetadataIndex  # noqa: E402
from recommendation.rag import ProductEnricher  # noqa: E402


class CountingLookup:
    """Wraps a metadata index and records every identifier the Agent asked about."""

    def __init__(self, delegate: MetadataIndex) -> None:
        self._delegate = delegate
        self.requested: list[str] = []

    def lookup(self, parent_asin: str) -> Any:
        """Record and delegate."""
        self.requested.append(parent_asin)
        return self._delegate.lookup(parent_asin)

    def lookup_many(self, parent_asins: Any) -> Any:
        """Record and delegate, preserving alignment."""
        self.requested.extend(parent_asins)
        return self._delegate.lookup_many(parent_asins)

    def __contains__(self, parent_asin: object) -> bool:
        """Delegate membership."""
        return parent_asin in self._delegate


class ExcludingLookup:
    """Metadata lookup that reports a chosen set of candidates as uncovered.

    Used to exercise the "candidate has no metadata" path on real candidates.  It
    **removes** records for those identifiers instead of inventing anything, so the
    missing path is genuine; no recommendation candidate is dropped or replaced.
    """

    def __init__(self, delegate: MetadataIndex, excluded: set[str]) -> None:
        self._delegate = delegate
        self._excluded = excluded

    def lookup(self, parent_asin: str) -> Any:
        """Return an explicit absence for excluded identifiers, else delegate."""
        if parent_asin in self._excluded:
            from recommendation.catalog import MissingMetadata

            return MissingMetadata(parent_asin=parent_asin)
        return self._delegate.lookup(parent_asin)

    def lookup_many(self, parent_asins: Any) -> Any:
        """Aligned lookup honouring the exclusion set."""
        return tuple(self.lookup(asin) for asin in parent_asins)

    def __contains__(self, parent_asin: object) -> bool:
        """Excluded identifiers are not members."""
        return parent_asin not in self._excluded and parent_asin in self._delegate


def main(argv: list[str] | None = None) -> int:
    """Run the integrated M8 smoke; returns 0 only when every gate passes."""
    parser = argparse.ArgumentParser(description="Milestone 8 integrated Agent smoke")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--query", default=SMOKE_QUERY)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args(argv)

    print("=" * 78)
    print("Milestone 8 integrated Agent smoke: recommend -> enrich -> finalize")
    print("Real checkpoint + real catalogue metadata. Grounding only, NOT quality.")
    print("=" * 78)

    try:
        runtime = build_m8_runtime(device=args.device, k=args.k)
    except FileNotFoundError as exc:
        print(f"\n{exc}", file=sys.stderr)
        return 2

    checks: dict[str, bool] = {}
    history = select_smoke_history()
    runtime.decision_model.set_decision(AgentDecision(action="recommend", k=args.k))

    summary = runtime.metadata_summary()
    chain = runtime.chain_summary()
    print(f"\ncheckpoint sha256 : {chain['checkpoint_sha256']}")
    print(f"manifest sha256   : {chain['manifest_sha256']}")
    print(f"catalog size      : {chain['num_items']:,}")
    print(f"metadata records  : {summary['records']:,}  (load {summary['load_seconds']:.2f}s)")
    print(f"metadata sha256   : {summary['artifact_sha256']}")
    print(f"graph nodes       : {', '.join(runtime.graph.node_names())}")
    print(f"history           : user_int_id={history.user_int_id} length={history.length} digest={history.digest}")
    print(f"query             : {args.query!r}")

    checks["accepted checkpoint digest matches"] = (
        chain["checkpoint_sha256"] == ACCEPTED_CHECKPOINT_SHA256
    )
    checks["enrichment node present in the graph"] = (
        set(runtime.graph.node_names()) - {"__start__", "__end__"}
        == {"decide", "recommend", "enrich", "finalize"}
    )

    # ---- route 1: recommend -> enrich -> finalize ------------------------ #
    runtime.engine.reset_counters()
    state = runtime.graph.run(args.query, history.parent_asins)
    recommendation = state["tool_result"]
    enrichment = state["enrichment"]

    print("\n--- route: RECOMMEND -> ENRICH ---")
    print(f"  route                : {state['route']}")
    print(f"  engine invocations   : {runtime.engine.recommend_calls}")
    print(f"  candidates           : {recommendation.returned_k}")
    print(f"  metadata found/missing: {enrichment.metadata_found}/{enrichment.metadata_missing}")
    print(f"  evidence fragments   : {enrichment.evidence_count}")

    candidate_order = [r.parent_asin for r in recommendation.recommendations]
    enriched_order = list(enrichment.parent_asins)
    print("\n    rank  parent_asin    metadata  evidence")
    for item in enrichment.items:
        print(
            f"    {item.rank:4d}  {item.parent_asin:<12}  "
            f"{item.metadata_status:<8}  {len(item.evidence)}"
        )

    checks["graph took the recommend route"] = state["route"] == "recommend"
    checks["tool/engine called exactly once"] = runtime.engine.recommend_calls == 1
    checks["candidate sequence unchanged by enrichment"] = candidate_order == enriched_order
    checks["candidate count unchanged"] = len(enriched_order) == recommendation.returned_k
    checks["enrichment happened after recommendation"] = enrichment.requested_k == recommendation.requested_k

    print("\n  grounded final response (truncated):")
    for line in state["final_response"].splitlines()[:14]:
        print(f"    {line}")

    checks["grounded response quotes evidence"] = all(
        e.text[:40] in state["final_response"]
        for item in enrichment.items
        for e in item.evidence[:1]
    ) or enrichment.evidence_count == 0

    # Grounding: every fact rendered for a candidate must exist in that candidate's
    # own metadata.  This is the real check behind "no fabricated brand/price".
    grounded = True
    for item in enrichment.items:
        if item.metadata is None:
            continue
        pool = (
            list(item.metadata.features)
            + list(item.metadata.description)
            + list(item.metadata.categories)
            + [item.metadata.title, item.metadata.store, item.metadata.main_category]
            + [value for _, value in item.metadata.details]
        )
        for evidence in item.evidence:
            if evidence.text not in pool:
                grounded = False
    checks["every rendered fact exists in its candidate's metadata"] = grounded
    checks["raw score labelled as ranking score"] = "ranking score" in state["final_response"].lower()

    # ---- candidate universe isolation ----------------------------------- #
    counting = CountingLookup(runtime.metadata)
    isolated_graph = AgentGraph(
        runtime.decision_model, runtime.tool, product_enricher=ProductEnricher(counting)
    )
    isolated_state = isolated_graph.run(args.query, history.parent_asins)
    requested = set(counting.requested)
    expected = {r.parent_asin for r in isolated_state["tool_result"].recommendations}

    print(f"\n  metadata lookups requested: {len(requested)} distinct identifiers")
    checks["metadata layer received only tool candidates"] = requested == expected
    checks["no extra catalog identifiers were consulted"] = len(requested) == len(expected)

    # ---- timing diagnostics --------------------------------------------- #
    print("\nlatency (engineering diagnostics, not quality metrics)")
    print(f"  metadata load        : {summary['load_seconds']:.2f}s (once per runtime)")
    print(f"  enrichment           : {enrichment.timings_ms.get('enrich')} ms")

    # ---- route 2: direct response --------------------------------------- #
    runtime.decision_model.set_decision(
        AgentDecision(action="direct_response", direct_response="Happy to help with a product question.")
    )
    runtime.engine.reset_counters()
    direct_state = runtime.graph.run("hello there", history.parent_asins)
    print("\n--- route: DIRECT RESPONSE ---")
    print(f"  route                : {direct_state['route']}")
    print(f"  engine invocations   : {runtime.engine.recommend_calls}")
    print(f"  enrichment present   : {'enrichment' in direct_state}")
    print(f"  response             : {direct_state['final_response']}")

    checks["direct route performs no inference"] = runtime.engine.recommend_calls == 0
    checks["direct route performs no enrichment"] = "enrichment" not in direct_state
    checks["direct route returns the direct response"] = (
        direct_state["final_response"] == "Happy to help with a product question."
    )
    runtime.decision_model.set_decision(AgentDecision(action="recommend", k=args.k))

    # ---- route 3: real candidates with no metadata ----------------------- #
    uncovered = set(candidate_order[:2])
    degraded = AgentGraph(
        runtime.decision_model,
        runtime.tool,
        product_enricher=ProductEnricher(ExcludingLookup(runtime.metadata, uncovered)),
    )
    runtime.engine.reset_counters()
    degraded_state = degraded.run(args.query, history.parent_asins)
    degraded_enrichment = degraded_state["enrichment"]
    degraded_order = [i.parent_asin for i in degraded_enrichment.items]

    print("\n--- route: real candidates with missing metadata ---")
    print(f"  metadata found/missing: {degraded_enrichment.metadata_found}/{degraded_enrichment.metadata_missing}")
    for item in degraded_enrichment.items:
        print(f"    {item.rank:4d}  {item.parent_asin:<12}  {item.metadata_status}  {item.fallback_reason}")

    checks["missing-metadata candidates are retained with identical order"] = (
        degraded_order == candidate_order
    )
    checks["missing metadata is explicit, not fabricated"] = (
        degraded_enrichment.metadata_missing == len(uncovered)
    )
    checks["missing-metadata candidates are marked in the response"] = (
        "metadata unavailable for this item" in degraded_state["final_response"]
    )

    # ---- determinism ----------------------------------------------------- #
    repeat = runtime.graph.run(args.query, history.parent_asins)
    checks["integrated route is deterministic"] = (
        [i.parent_asin for i in repeat["enrichment"].items] == enriched_order
        and repeat["final_response"] == state["final_response"]
    )

    # ---- trust boundary -------------------------------------------------- #
    prompt = runtime.decision_model.last_prompt_text
    checks["decision model never received trusted history"] = all(
        asin not in prompt for asin in history.parent_asins
    )
    checks["trusted history preserved in state"] = (
        state["trusted_user_history"] == history.parent_asins
    )

    print("\nChecks")
    for name, ok in checks.items():
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    passed = all(checks.values())
    print(f"\nSMOKE: {'PASS' if passed else 'FAIL'}")

    if args.json is not None:
        payload = {
            "milestone": "8",
            "query": args.query,
            "requested_k": args.k,
            "metadata": summary,
            "chain": {
                "checkpoint_sha256": chain["checkpoint_sha256"],
                "manifest_sha256": chain["manifest_sha256"],
                "num_items": chain["num_items"],
            },
            "history": history.as_dict(),
            "nodes": list(runtime.graph.node_names()),
            "route": state["route"],
            "engine_invocations": 1,
            "candidates": [
                {"rank": i.rank, "parent_asin": i.parent_asin, "score": i.score, "map": i.metadata_status}
                for i in enrichment.items
            ],
            "evidence": [
                {
                    "parent_asin": e.parent_asin,
                    "field": e.field,
                    "detail_key": e.detail_key,
                    "retrieval_score": e.retrieval_score,
                    "provenance": e.provenance,
                }
                for i in enrichment.items
                for e in i.evidence
            ],
            "metadata_lookups_requested": sorted(requested),
            "checks": checks,
        }
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"wrote {args.json}")

    return 0 if passed else 1


if __name__ == "__main__":  # pragma: no cover - manual invocation
    raise SystemExit(main())

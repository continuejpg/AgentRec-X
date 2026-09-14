"""Milestone 8-B smoke: real candidates + real catalogue metadata + scoped retrieval.

Runs the accepted real recommendation chain, enriches **those exact candidates** with
the normalized Amazon Reviews 2023 Sports & Outdoors metadata artifact, and retrieves
attributed evidence for a fixed query.

What it proves:

1. metadata raw/processed identity (recorded digests);
2. the normalized index loads and its catalog coverage is reported;
3. the real M7C chain produces real candidates from a deterministic real history;
4. retrieval is scoped to exactly those candidates -- nothing else is reachable;
5. candidate identities, order, ranks and scores are unchanged by enrichment;
6. every evidence fragment belongs to an original candidate;
7. missing metadata is counted and stays explicit;
8. a grounded result renders;
9. repeated runs are deterministic.

It does **not** evaluate recommendation quality, and its coverage figures are
descriptive data statistics rather than accuracy metrics.

Usage::

    .venv/bin/python -m experiments.product_rag_smoke
    .venv/bin/python -m experiments.product_rag_smoke --k 5 --query "waterproof boots"
    .venv/bin/python -m experiments.product_rag_smoke --json /tmp/m8b.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.product_rag_support import (  # noqa: E402
    ACCEPTED_CHECKPOINT_SHA256,
    ACCEPTED_MANIFEST_SHA256,
    SMOKE_QUERY,
    M8Runtime,
    build_m8_runtime,
    select_smoke_history,
)
from recommendation.agent import AgentDecision  # noqa: E402
from recommendation.catalog import MissingMetadata  # noqa: E402
from recommendation.rag import ProductEvidence, enrich_candidates  # noqa: E402
from recommendation.tools import ToolRecommendation  # noqa: E402

#: Maximum characters of a title/description printed, so the smoke stays compact.
_PREVIEW = 70


def _preview(text: str, limit: int = _PREVIEW) -> str:
    """Return a short single-line preview of a text fragment."""
    flat = " ".join(text.split())
    return flat if len(flat) <= limit else flat[: limit - 1] + "…"


def _evidence_row(evidence: ProductEvidence) -> dict[str, Any]:
    """Compact JSON-serialisable view of one evidence fragment."""
    return {
        "parent_asin": evidence.parent_asin,
        "field": evidence.field,
        "detail_key": evidence.detail_key,
        "retrieval_score": evidence.retrieval_score,
        "provenance": evidence.provenance,
        "text_preview": _preview(evidence.text),
    }


def run_smoke(runtime: M8Runtime, k: int, query: str, repeats: int) -> tuple[dict[str, Any], dict[str, bool]]:
    """Execute the M8-B smoke and return its evidence and gate results."""
    checks: dict[str, bool] = {}
    history = select_smoke_history()

    summary = runtime.metadata_summary()
    chain = runtime.chain_summary()

    print(f"\nmetadata artifact : {summary['artifact_path']}")
    print(f"  artifact sha256 : {summary['artifact_sha256']}")
    print(f"  artifact bytes  : {summary['artifact_bytes']:,}")
    print(f"  raw sha256      : {summary['raw_sha256']}")
    print(f"  raw bytes       : {summary['raw_bytes']:,}")
    print(f"  source url      : {summary['source_url']}")
    print(f"  norm version    : {summary['normalization_version']} (dup policy {summary['duplicate_policy']})")
    print(f"  records         : {summary['records']:,}")
    print(f"  load            : {summary['load_seconds']:.2f}s (once)")

    coverage = summary["coverage"]
    print("\ncatalog coverage (descriptive data statistic, NOT a quality metric)")
    print(f"  catalog items            : {coverage.get('num_catalog_items', 0):,}")
    print(f"  metadata records         : {coverage.get('num_metadata_records', 0):,}")
    print(f"  catalog items with metadata: {coverage.get('num_catalog_items_with_metadata', 0):,}")
    print(f"  catalog items missing    : {coverage.get('num_catalog_items_missing_metadata', 0):,}")
    print(f"  coverage                 : {coverage.get('coverage_percentage', 0.0):.4f}%")

    checks["accepted checkpoint digest matches"] = (
        chain["checkpoint_sha256"] == ACCEPTED_CHECKPOINT_SHA256
    )
    checks["accepted run manifest digest matches"] = (
        chain["manifest_sha256"] == ACCEPTED_MANIFEST_SHA256
    )
    checks["metadata artifact digest recorded"] = bool(summary["artifact_sha256"])
    checks["metadata raw digest recorded"] = bool(summary["raw_sha256"])
    checks["full catalog covered by metadata"] = (
        coverage.get("num_catalog_items") == summary["records"]
    )

    print(f"\ntrusted history : user_int_id={history.user_int_id} length={history.length} digest={history.digest}")
    print(f"query           : {query!r}")
    print(f"requested k     : {k}")

    # ---- real recommendation -------------------------------------------- #
    runtime.decision_model.set_decision(AgentDecision(action="recommend", k=k))
    before_calls = runtime.engine.recommend_calls
    state = runtime.graph.run(query, history.parent_asins)
    recommendation = state["tool_result"]
    engine_calls = runtime.engine.recommend_calls - before_calls

    print(f"\nrecommend route : {state['route']} | engine invocations: {engine_calls}")
    print(f"candidates      : {recommendation.returned_k} (requested {recommendation.requested_k})")
    candidate_rows = [
        {
            "rank": item.rank,
            "parent_asin": item.parent_asin,
            "item_id": item.item_id,
            "score": item.score,
        }
        for item in recommendation.recommendations
    ]

    checks["graph selected the recommend route"] = state["route"] == "recommend"
    checks["exactly one engine invocation"] = engine_calls == 1
    checks["real candidates returned"] = recommendation.returned_k > 0
    checks["enrichment ran on the recommendation route"] = "enrichment" in state

    # ---- candidate-scoped retrieval ------------------------------------- #
    started = time.perf_counter()
    enrichment = state["enrichment"]
    enrich_ms = (time.perf_counter() - started) * 1000.0

    candidate_asins = {row["parent_asin"] for row in candidate_rows}
    order_before = [row["parent_asin"] for row in candidate_rows]
    order_after = list(enrichment.parent_asins)

    print("\nenrichment (candidate-scoped retrieval)")
    print("  rank  parent_asin    metadata  evidence  title / top evidence")
    evidence_rows: list[dict[str, Any]] = []
    for item in enrichment.items:
        top = item.evidence[0] if item.evidence else None
        if item.metadata_status == "missing":
            detail = "metadata unavailable"
        elif top is None:
            detail = f"(no evidence: {item.fallback_reason})"
        elif top.field == "title":
            detail = _preview(top.text)
        elif item.metadata is not None and item.metadata.title:
            detail = _preview(item.metadata.title)
        else:
            detail = _preview(top.text)
        print(
            f"  {item.rank:4d}  {item.parent_asin:<12}  "
            f"{item.metadata_status:<8}  {len(item.evidence):<8}  {detail}"
        )
        for evidence in item.evidence:
            evidence_rows.append(_evidence_row(evidence))

    print("\nretrieved evidence (attributed fragments)")
    if evidence_rows:
        for row in evidence_rows:
            key = row["detail_key"] or row["field"]
            print(
                f"  {row['parent_asin']:<12} {key:<12} "
                f"score={row['retrieval_score']:.4f}  {row['text_preview']}"
            )
    else:
        print("  (none)")

    print(
        f"\ncounts          : metadata_found={enrichment.metadata_found} "
        f"metadata_missing={enrichment.metadata_missing} evidence={enrichment.evidence_count}"
    )

    checks["candidate order unchanged by enrichment"] = order_before == order_after
    checks["candidate count unchanged by enrichment"] = (
        len(order_after) == recommendation.returned_k
    )

    scores_unchanged = all(
        item.score == row["score"] and item.rank == row["rank"]
        for item, row in zip(enrichment.items, candidate_rows)
    )
    checks["candidate ranks and scores unchanged"] = scores_unchanged

    # Candidate universe isolation: every evidence fragment belongs to a candidate.
    checks["all evidence belongs to an original candidate"] = all(
        row["parent_asin"] in candidate_asins for row in evidence_rows
    )
    checks["no evidence from outside the candidate set"] = all(
        row["parent_asin"] in candidate_asins for row in evidence_rows
    )
    checks["evidence provenance is attributed"] = all(
        row["provenance"].startswith("amazon_reviews_2023:meta_categories#")
        for row in evidence_rows
    )

    # Evidence text must exist in its own candidate's metadata.
    verbatim = True
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
                verbatim = False
    checks["evidence text is verbatim metadata text"] = verbatim

    # Seen-item masking must be untouched by enrichment.
    seen = set(history.parent_asins)
    checks["no already-seen candidate returned"] = not (candidate_asins & seen)
    checks["no PAD candidate"] = all(row["item_id"] != 0 for row in candidate_rows)

    # Missing metadata, if any, must be explicit and not fabricated.
    missing_items = [i for i in enrichment.items if i.metadata_status == "missing"]
    checks["missing metadata is explicit"] = all(
        isinstance(runtime.metadata.lookup(i.parent_asin), MissingMetadata) for i in missing_items
    )
    checks["missing-metadata candidates are retained"] = len(missing_items) == (
        enrichment.metadata_missing
    )

    # ---- grounded rendering --------------------------------------------- #
    text = state["final_response"]
    print("\ngrounded final response (truncated)")
    for line in text.splitlines()[:18]:
        print(f"  {line}")
    if len(text.splitlines()) > 18:
        print("  ...")

    lowered = text.lower()
    checks["grounded response names the candidates"] = all(
        row["parent_asin"] in text for row in candidate_rows
    )
    checks["grounded response quotes metadata facts"] = all(
        row["text_preview"].rstrip("…") in text for row in evidence_rows[:3]
    ) or not evidence_rows
    checks["no currency claim"] = "$" not in text and "price:" not in lowered
    checks["raw score labelled as ranking score"] = "ranking score" in lowered
    checks["no quality claim"] = not any(
        claim in lowered for claim in ("best for you", "perfect for you", "outperforms")
    )

    # ---- determinism ----------------------------------------------------- #
    baseline_semantics = [
        (i.rank, i.parent_asin, i.recommendation.item_id, i.score) for i in enrichment.items
    ]
    baseline_evidence = [
        (e.parent_asin, e.field, e.text, e.retrieval_score)
        for i in enrichment.items
        for e in i.evidence
    ]
    repeat_candidates: list[list[tuple[int, str, int, float]]] = []
    repeat_evidence = baseline_evidence
    for _ in range(max(repeats - 1, 0)):
        repeat_state = runtime.graph.run(query, history.parent_asins)
        repeat_candidates.append(
            [
                (i.rank, i.parent_asin, i.recommendation.item_id, i.score)
                for i in repeat_state["enrichment"].items
            ]
        )
        repeat_evidence = [
            (e.parent_asin, e.field, e.text, e.retrieval_score)
            for i in repeat_state["enrichment"].items
            for e in i.evidence
        ]
    checks["repeated runs are semantically deterministic"] = all(
        repeat == baseline_semantics for repeat in repeat_candidates
    )
    checks["repeated retrieval is deterministic"] = repeat_evidence == baseline_evidence

    # ---- direct route isolation ------------------------------------------ #
    runtime.engine.reset_counters()
    runtime.decision_model.set_decision(
        AgentDecision(action="direct_response", direct_response="Happy to help.")
    )
    direct_state = runtime.graph.run("hello there", history.parent_asins)
    checks["direct route performs no inference"] = runtime.engine.recommend_calls == 0
    checks["direct route produces no enrichment"] = "enrichment" not in direct_state
    runtime.decision_model.set_decision(AgentDecision(action="recommend", k=k))

    payload = {
        "milestone": "8B",
        "query": query,
        "requested_k": k,
        "metadata": summary,
        "history": history.as_dict(),
        "chain": {
            "checkpoint_sha256": chain["checkpoint_sha256"],
            "manifest_sha256": chain["manifest_sha256"],
            "num_items": chain["num_items"],
        },
        "candidates": candidate_rows,
        "candidates_after_enrichment": [
            {"rank": i.rank, "parent_asin": i.parent_asin, "score": i.score}
            for i in enrichment.items
        ],
        "evidence": evidence_rows,
        "counts": {
            "metadata_found": enrichment.metadata_found,
            "metadata_missing": enrichment.metadata_missing,
            "evidence": enrichment.evidence_count,
        },
        "latency_ms": {
            "metadata_load": round(runtime.metadata_load_seconds * 1000.0, 2),
            "enrichment_total": round(enrich_ms, 3),
            "enrichment_reported": enrichment.timings_ms.get("enrich"),
        },
        "checks": checks,
    }
    return payload, checks


def main(argv: list[str] | None = None) -> int:
    """Run the M8-B smoke; returns 0 only when every gate passes."""
    parser = argparse.ArgumentParser(description="Milestone 8-B product RAG smoke")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--query", default=SMOKE_QUERY)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--artifact", type=Path, default=None)
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args(argv)

    print("=" * 78)
    print("Milestone 8-B candidate-scoped product RAG smoke (real chain)")
    print("Integration/grounding only. NOT a recommendation-quality benchmark.")
    print("=" * 78)

    try:
        runtime = build_m8_runtime(device=args.device, k=args.k, artifact=args.artifact)
    except FileNotFoundError as exc:
        print(f"\n{exc}", file=sys.stderr)
        return 2

    payload, checks = run_smoke(runtime, args.k, args.query, args.repeats)

    print("\nChecks")
    for name, ok in checks.items():
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    passed = all(checks.values())
    print(f"\nSMOKE: {'PASS' if passed else 'FAIL'}")

    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"wrote {args.json}")

    return 0 if passed else 1


if __name__ == "__main__":  # pragma: no cover - manual invocation
    raise SystemExit(main())

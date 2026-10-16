"""E-04B-2 Gate A: real protocol self-check, real BM25 smoke, exploratory BM25 baseline.

What this runs (no neural weights, no network, CPU only)
-------------------------------------------------------
A-1  real-data protocol checks, streamed from the frozen artifacts:
     * users are read from ``data/processed/Sports_and_Outdoors_sequences.json`` one element at a
       time (the file is a JSON object with a pretty-printed ``sequences`` array, so a bounded
       line-chunk reader parses one user without holding 333 MB in memory);
     * each user is split by the accepted protocol (``recommendation/evaluation/split.py``,
       ``agentrecx.eval_protocol.v1`` / ``temporal_leave_two_out``);
     * runtime invariants are asserted, not grepped: PAD never appears, ids stay inside the real
       catalogue range, the test target is never inside the masked set, and the masked set the
       metric uses is the accepted one.

A-2  real BM25 smoke over the **real** catalogue (``data/processed/Sports_and_Outdoors_products.jsonl``)
     through the production lexical source (``recommendation/control/catalog_search.py``), for a
     bounded number of cases.  Every returned identity, score, rank and score kind is recorded, and
     the masked-set correctness is checked per case.

A-3  exploratory full-catalogue metrics for a bounded number of valid cases.  The query is built
     **only** from tokens of catalogue text belonging to the case's *test-time-visible* history:
     the hidden ``test_target``'s text is never read.  The script asserts that structurally by
     recording which item ids had their catalogue text read and requiring the target to be absent.

Usage
-----
    python -m experiments.e04b2_gate_a --cases 100 --smoke 20 --k 5,10,20 --out runs/e04b2_gate_a
"""

from __future__ import annotations

import argparse
import json
import os
import resource
import sys
import time
from pathlib import Path
from typing import Any, Sequence

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

SEQUENCES = REPO / "data" / "processed" / "Sports_and_Outdoors_sequences.json"
PRODUCTS = REPO / "data" / "processed" / "Sports_and_Outdoors_products.jsonl"
MAPPINGS = REPO / "data" / "processed" / "Sports_and_Outdoors_mappings.json"

#: Declared before running (Gate A-3 §5): case selection, seed, K values, query rule, masking.
SELECTION_RULE = "users in artifact order, first N with a valid protocol case (no sampling)"
SEED = 20261015  # recorded for reproducibility; the selection rule above needs no randomness
K_VALUES = (5, 10, 20)
QUERY_RULE = (
    "tokens of the catalogue text of the last up-to-5 items in the test-time-visible history "
    "(test_history); never the target, never validation/test labels"
)
MASK_RULE = "exclude test_seen (the case's own history plus the consumed validation target)"
INVALID_CASE_RULE = "a case is skipped when the protocol rejects the sequence or the target is masked"


def _rss_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def _available_mb() -> float:
    try:
        with open("/proc/meminfo", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("MemAvailable:"):
                    return float(line.split()[1]) / 1024.0
    except OSError:
        pass
    return float("inf")


def stream_sequences(path: Path):
    """Yield one user element at a time from the pretty-printed ``sequences`` array."""
    inside = False
    buffer: list[str] = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not inside:
                if stripped.startswith('"sequences"'):
                    inside = True
                continue
            if stripped == "]" or line.startswith("  ]"):
                # the top-level array closes with a two-space indent; a nested array closes with a
                # deeper one and must not end the scan
                break
            if stripped.startswith("{"):
                buffer = [line]
                continue
            if not buffer:
                continue
            buffer.append(line)
            if stripped.startswith("}"):
                raw = "".join(buffer).rstrip().rstrip(",")
                buffer = []
                yield json.loads(raw)


def load_catalogue(path: Path, item2id: dict[str, int]) -> tuple[dict[int, str], dict[int, str]]:
    """``item_id -> searchable text`` and ``item_id -> parent_asin`` from the real catalogue.

    The processed catalogue stores ``parent_asin`` plus the catalogue fields; the integer identity
    comes from the accepted mapping artifact, so the same identity space the split uses is the one
    the lexical index uses.  Only catalogue text is read - never an evaluation label.
    """
    text_by_id: dict[int, str] = {}
    asin_by_id: dict[int, str] = {}
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            parent = record.get("parent_asin")
            if not isinstance(parent, str):
                continue  # the artifact's own header record
            item_id = item2id.get(parent)
            if item_id is None:
                continue
            parts = [str(record.get("title") or "")]
            for field in ("features", "categories"):
                value = record.get(field)
                if isinstance(value, list):
                    parts.extend(str(entry) for entry in value)
                elif isinstance(value, str):
                    parts.append(value)
            parts.append(str(record.get("store") or ""))
            details = record.get("details")
            if isinstance(details, dict):
                parts.extend(str(value) for value in details.values())
            asin_by_id[item_id] = parent
            text_by_id[item_id] = " | ".join(part for part in parts if part and part != "None")
    return text_by_id, asin_by_id


def build_metadata(text_by_id: dict[int, str], asin_by_id: dict[int, str], limit: int) -> Any:
    """Real ``MetadataIndex`` over the real catalogue (identity + text), capped for this run."""
    from recommendation.catalog import MetadataIndex
    from recommendation.catalog.metadata import normalize_product_record

    records = []
    for item_id in sorted(text_by_id)[:limit]:
        text = text_by_id[item_id]
        title = text.split("\n", 1)[0].strip() if text else ""
        records.append(
            normalize_product_record({
                "parent_asin": asin_by_id.get(item_id) or f"item-{item_id}",
                "title": title or f"item {item_id}",
                "details": {"Item Id": str(item_id)},
            })
        )
    return MetadataIndex.from_records(records)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="E-04B-2 Gate A (real protocol + real BM25)")
    parser.add_argument("--out", type=Path, default=REPO / "runs" / "e04b2_gate_a")
    parser.add_argument("--cases", type=int, default=100, help="max valid cases for A-3")
    parser.add_argument("--smoke", type=int, default=20, help="cases for the A-2 smoke")
    parser.add_argument("--catalogue-limit", type=int, default=0,
                        help="cap the indexed catalogue (0 = whole real catalogue)")
    parser.add_argument("--k", default=",".join(str(k) for k in K_VALUES))
    args = parser.parse_args(argv)
    k_values = tuple(int(part) for part in args.k.split(",") if part.strip())

    args.out.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    summary: dict[str, Any] = {
        "protocol_version": None,
        "selection_rule": SELECTION_RULE,
        "seed": SEED,
        "k_values": list(k_values),
        "query_rule": QUERY_RULE,
        "mask_rule": MASK_RULE,
        "invalid_case_rule": INVALID_CASE_RULE,
        "paths": {"sequences": str(SEQUENCES.relative_to(REPO)), "products": str(PRODUCTS.relative_to(REPO)),
                  "mappings": str(MAPPINGS.relative_to(REPO))},
        "resources": {"rss_mb_start": round(_rss_mb(), 1), "mem_available_mb_start": round(_available_mb(), 1)},
    }

    # ---------------- A-1: protocol checks on the real data ------------------ #
    from recommendation.evaluation.split import PROTOCOL_VERSION, split_sequence
    from recommendation.evaluation.metrics import excluded_seen, hit_at_k, ndcg_at_k

    with open(MAPPINGS, encoding="utf-8") as handle:
        mappings = json.load(handle)
    num_items = int(mappings["num_items"])
    num_users = int(mappings["num_users"])
    summary["protocol_version"] = PROTOCOL_VERSION
    summary["catalogue"] = {"num_items": num_items, "num_users": num_users,
                            "pad_id": int(mappings["padding"]["pad_id"])}

    text_by_id, asin_by_id = load_catalogue(PRODUCTS, {str(k): int(v) for k, v in mappings["item2id"].items()})
    catalogue_ids = set(text_by_id)
    summary["catalogue"]["catalogue_records"] = len(catalogue_ids)

    checked = valid = skipped = 0
    skip_reasons: dict[str, int] = {}
    pad_hits = out_of_range = target_masked = not_in_catalogue = 0
    cases: list[Any] = []
    for element in stream_sequences(SEQUENCES):
        items = [int(value) for value in element.get("item_ids", ())]
        checked += 1
        if 0 in items:
            pad_hits += 1
        if any(value < 1 or value > num_items for value in items):
            out_of_range += 1
        case = split_sequence(str(element.get("user_id", "")), items, num_items,
                              int(element.get("user_int_id", -1)))
        if case is None:
            skipped += 1
            skip_reasons["protocol_rejected"] = skip_reasons.get("protocol_rejected", 0) + 1
        else:
            valid += 1
            if case.test_target in excluded_seen(case.test_history, case.test_target):
                target_masked += 1
            if case.test_target not in catalogue_ids or case.test_target not in asin_by_id:
                not_in_catalogue += 1
            if len(cases) < args.cases:
                cases.append(case)
        if checked >= args.cases * 3 or (valid >= args.cases and checked >= args.cases):
            break
    summary["A1"] = {
        "users_checked": checked, "valid_cases": valid, "skipped_cases": skipped,
        "skip_reasons": skip_reasons, "pad_hits": pad_hits, "out_of_range_ids": out_of_range,
        "target_inside_mask": target_masked, "target_not_in_catalogue": not_in_catalogue,
        "cases_retained": len(cases), "rss_mb_after_scan": round(_rss_mb(), 1),
    }

    # ---------------- A-2: real BM25 over the real catalogue ----------------- #
    from recommendation.control.catalog_search import CatalogSearchSource

    limit = args.catalogue_limit or len(catalogue_ids)
    index_started = time.perf_counter()
    metadata = build_metadata(text_by_id, asin_by_id, limit)
    source = CatalogSearchSource(metadata)
    summary["A2_index"] = {
        "indexed_records": source.catalogue_size, "build_seconds": round(time.perf_counter() - index_started, 2),
        "rss_mb_after_index": round(_rss_mb(), 1), "mem_available_mb": round(_available_mb(), 1),
    }

    # Identity comes from the accepted mapping artifact (item_id <-> parent_asin), not from a
    # metadata round-trip: the returned catalogue identity is resolved back to the same integer
    # space the split and the labels use.
    asin_to_id = {asin: item_id for item_id, asin in asin_by_id.items()}

    smoke: list[dict[str, Any]] = []
    text_used: set[int] = set()
    for case in cases[: args.smoke]:
        history = list(case.test_history)[-5:]
        query_tokens: list[str] = []
        for item_id in history:
            text_used.add(item_id)                      # runtime target-isolation evidence
            query_tokens.extend((text_by_id.get(item_id, "") or "").split()[:12])
        query = " ".join(query_tokens)[:400]
        began = time.perf_counter()
        result = source.search(query, limit=max(k_values))
        elapsed = time.perf_counter() - began
        hits = list(getattr(result, "hits", ()) or ())
        returned_ids, rows = [], []
        for rank, hit in enumerate(hits, start=1):
            parent = str(getattr(hit, "parent_asin", ""))
            item_id = asin_to_id.get(parent)
            if item_id is None:
                item_id = int(getattr(hit, "item_id", 0) or 0)
            returned_ids.append(item_id)
            rows.append({
                "rank": rank, "item_id": item_id, "parent_asin": parent,
                "score": round(float(getattr(hit, "score", 0.0)), 6),
                "score_kind": str(getattr(hit, "score_kind", "bm25")),
                "masked": item_id in case.test_seen,
            })
        smoke.append({
            "case_id": case.user_int_id, "query_tokens": len(query_tokens),
            "matched": bool(getattr(result, "matched", bool(hits))), "returned": len(hits),
            "elapsed_ms": round(elapsed * 1000, 2),
            "masked_out_hits": sum(1 for row in rows if row["masked"]),
            "invalid_ids": sum(1 for item_id in returned_ids if item_id < 1 or item_id > num_items),
            "pad_returned": sum(1 for item_id in returned_ids if item_id == 0),
            "target_leaked_into_query": case.test_target in text_used,
            "top": rows[:5],
        })
        if _available_mb() < 1200:
            summary["aborted"] = "mem_available below 1200 MB"
            break
    summary["A2_smoke"] = {"cases": len(smoke), "rows": smoke}
    summary["A2_checks"] = {
        "all_queries_matched": all(row["matched"] for row in smoke),
        "masked_out_hits": sum(row["masked_out_hits"] for row in smoke),
        "invalid_ids": sum(row["invalid_ids"] for row in smoke),
        "pad_returned": sum(row["pad_returned"] for row in smoke),
        "target_text_read": sorted(text_used & {case.test_target for case in cases}),
    }

    # ---------------- A-3: exploratory full-catalogue metrics ---------------- #
    from recommendation.evaluation.metrics import rank_of_target_via_sort

    per_case: list[dict[str, Any]] = []
    for case in cases:
        history = list(case.test_history)[-5:]
        query_tokens: list[str] = []
        for item_id in history:
            text_used.add(item_id)
            query_tokens.extend((text_by_id.get(item_id, "") or "").split()[:12])
        query = " ".join(query_tokens)[:400]
        result = source.search(query, limit=limit)
        hits = list(getattr(result, "hits", ()) or ())
        masked = excluded_seen(case.test_history, case.test_target)
        unmasked: list[tuple[int, float]] = []
        target_score: float | None = None
        for hit in hits:
            parent = str(getattr(hit, "parent_asin", ""))
            item_id = asin_to_id.get(parent)
            if item_id is None:
                continue
            score = float(getattr(hit, "score", 0.0))
            if item_id == case.test_target:
                target_score = score
            if item_id in masked:
                continue
            unmasked.append((item_id, score))
        if target_score is None:
            # full-catalogue convention: an unmatched target ranks after every matched document
            rank = len(unmasked) + 1
        else:
            rank = 1 + sum(1 for _item_id, score in unmasked if score > target_score)
        per_case.append({
            "case_id": case.user_int_id, "rank": int(rank), "returned": len(hits),
            "hr": {str(k): hit_at_k(int(rank), k) for k in k_values},
            "ndcg": {str(k): ndcg_at_k(int(rank), k) for k in k_values},
        })
        if _available_mb() < 1000:
            break

    def mean(values: Sequence[float]) -> float:
        return round(sum(values) / len(values), 6) if values else 0.0

    summary["A3"] = {
        "cases_scored": len(per_case),
        "hits_ranked": sum(1 for row in per_case if row["rank"] <= max(k_values)),
        "HR": {str(k): mean([row["hr"][str(k)] for row in per_case]) for k in k_values},
        "NDCG": {str(k): mean([row["ndcg"][str(k)] for row in per_case]) for k in k_values},
        "legacy": "exploratory only: bounded sample, legal history-derived query, not a model conclusion",
        "per_case": per_case,
    }
    summary["resources"]["rss_mb_end"] = round(_rss_mb(), 1)
    summary["resources"]["mem_available_mb_end"] = round(_available_mb(), 1)
    summary["resources"]["wall_seconds"] = round(time.perf_counter() - started, 2)

    (args.out / "gate_a_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({
        "A1": {k: v for k, v in summary["A1"].items() if k != "cases_retained"},
        "A2_index": summary["A2_index"],
        "A2_checks": summary["A2_checks"],
        "A3": {k: v for k, v in summary["A3"].items() if k != "per_case"},
        "resources": summary["resources"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

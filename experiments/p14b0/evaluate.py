"""P14-B0 pilot diagnosis: fixed-candidate Search Only vs Search + Preferences.

Frozen inputs:
* P14-E ``data/p14e/benchmark`` agent-visible tasks and evaluator labels (read-only);
* P13-R Electronics catalogue via ``experiments.p14e.catalogue``;
* accepted BM25 ``CatalogSearchSource``;
* accepted M9/M5 memory via ``ShoppingCompanionMemory`` + ``HistoryStateReader``;
* ``PreferenceAwareScorer`` V0.

Artifacts go to ``runs/p14b0/``.  No model is trained; no evaluator label ever enters
the scorer's inputs or the history read.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from experiments.p14e.catalogue import ELECTRONICS_NAMESPACE, ElectronicsCatalogue
from experiments.p14e.labeling import assess_candidate
from recommendation.preference.history import HistoryStateReader
from recommendation.preference.scorer import PreferenceAwareScorer, ScorerConfig
from recommendation.shopping_companion.memory import (
    SOURCE_KIND_SYNTHETIC_FIXTURE,
    ShoppingCompanionMemory,
)

REPO = Path(__file__).resolve().parents[2]
DATA = REPO / "data/p14e/benchmark"
RUNS = REPO / "runs/p14b0"
TOPK = 20
SCHEMA = "agentrecx.p14b0.evaluation.v1"
METHODS = ("search_only", "search_preferences", "no_history", "shuffled_history")

ATTR_CANON = {
    "over-ear": "over_ear", "over ear": "over_ear",
    "in-ear": "in_ear", "in ear": "in_ear", "earbud": "in_ear",
    "on-ear": "on_ear", "on ear": "on_ear",
    "noise cancelling": "noise_cancelling", "noise canceling": "noise_cancelling",
    "wireless": "wireless", "wired": "wired",
    "microphone": "microphone", "mic": "microphone",
    "waterproof": "waterproof", "water resistant": "waterproof", "sweatproof": "waterproof",
    "foldable": "foldable", "mechanical": "mechanical",
    "rgb": "rgb", "backlit": "rgb",
    "rechargeable": "rechargeable", "multi": "multidevice",
    "usb": "usb", "silent": "silent", "quiet": "silent",
    "ergonomic": "ergonomic", "gaming": "gaming",
}
FORM_ATTRS = ("over_ear", "on_ear", "in_ear")
HARD_PATTERN = re.compile(
    r"(?:it\s+(?:needs|has|must)\s+to\s+be|it\s+must\s+be|must\s+be|it\s+has\s+to\s+be)\s+([^.!]+)",
    re.IGNORECASE,
)


def _load(name: str) -> list[dict[str, Any]]:
    return [json.loads(line) for line in (DATA / name).read_text(encoding="utf-8").splitlines() if line.strip()]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_attribute(value: Any) -> str | None:
    text = str(value or "").strip().lower().replace("_", "-")
    if not text:
        return None
    for needle, canonical in ATTR_CANON.items():
        if needle in text:
            return canonical
    return None


def canonical_preferences(payload: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[tuple[str, str]] = set()
    out: list[dict[str, Any]] = []
    for item in payload:
        canonical = canonical_attribute(item.get("value"))
        if canonical is None:
            continue
        polarity = str(item.get("polarity") or "positive").lower()
        polarity = "negative" if polarity in {"avoid", "negative", "dislike"} else "positive"
        key = (canonical, polarity)
        if key in seen:
            continue
        seen.add(key)
        out.append(
            {
                "value": canonical,
                "kind": "form_factor" if canonical in FORM_ATTRS else "attribute",
                "polarity": polarity,
                "memory_id": item.get("memory_id"),
                "source_turn_id": item.get("source_turn_id"),
                "source_text": item.get("source_text"),
            }
        )
    return out


def hard_requirements_from_query(query: str) -> list[dict[str, Any]]:
    requirements: list[dict[str, Any]] = []
    if re.search(r"\b(?:budget|price|under\s+\$?\d+)\b", query, re.IGNORECASE) and re.search(r"\$\s?\d+", query):
        requirements.append({"key": "price", "kind": "unverifiable", "value": "price", "source": "query"})
    if re.search(r"warranty", query, re.IGNORECASE):
        requirements.append({"key": "warranty", "kind": "unverifiable", "value": "warranty", "source": "query"})
    for match in HARD_PATTERN.finditer(query):
        phrase = match.group(1).strip().lower()
        canonical = canonical_attribute(phrase)
        if canonical:
            requirements.append(
                {
                    "key": canonical,
                    "kind": "form_factor" if canonical in FORM_ATTRS else "attribute",
                    "value": canonical,
                    "source": phrase,
                }
            )
    # deterministic dedupe
    seen = set()
    unique = []
    for req in requirements:
        if req["key"] in seen:
            continue
        seen.add(req["key"])
        unique.append(req)
    return unique


def candidate_payload(product, rank: int, score: float) -> dict[str, Any]:
    attributes = {
        name: {"evidence": fact.get("evidence"), "source": "catalog_meta:text"}
        for name, fact in (product.attributes or {}).items()
    }
    return {
        "product_ref": {"namespace": ELECTRONICS_NAMESPACE, "item_id": product.item_id},
        "query_rank": rank,
        "query_score": round(score, 6),
        "attributes": attributes,
        "form_factor": product.form_factor,
        "title": product.title,
    }


def verified_ids(ev: dict[str, Any], catalogue: ElectronicsCatalogue) -> set[str]:
    pool = catalogue.by_type(ev["product_type_pool"])
    return {
        product.item_id for product in pool
        if assess_candidate(
            product,
            hard_constraints=ev["hard_constraints"],
            preferences=ev["effective_preferences"],
        ).acceptable
    }


def eval_top1(candidate: dict[str, Any] | None, ev: dict[str, Any]) -> dict[str, Any]:
    """Evaluate one presented top-1 against evaluator-only labels (metrics only)."""
    if candidate is None:
        return {"presented": False, "matched": 0, "violated": 0, "unknown": 0, "hard_violations": 0}
    present = set((candidate.get("attributes") or {}).keys())
    form = candidate.get("form_factor")
    matched = violated = unknown = 0
    for preference in ev["effective_preferences"]:
        canonical = canonical_attribute(preference.get("value"))
        if canonical is None:
            unknown += 1
            continue
        polarity = str(preference.get("polarity"))
        observed = form == canonical if canonical in FORM_ATTRS else canonical in present
        if observed and polarity == "positive":
            matched += 1
        elif observed and polarity == "negative":
            violated += 1
        else:
            unknown += 1
    hard_violations = 0
    for constraint in ev["hard_constraints"]:
        if constraint.get("kind") == "unverifiable":
            continue
        canonical = canonical_attribute(constraint.get("value")) or str(constraint.get("value"))
        if canonical in FORM_ATTRS:
            if form is not None and form != canonical:
                hard_violations += 1
        elif canonical not in present:
            hard_violations += 0  # unknown is not a violation
    return {
        "presented": True,
        "matched": matched,
        "violated": violated,
        "unknown": unknown,
        "hard_violations": hard_violations,
        "hard_verified": all(
            (canonical_attribute(c.get("value")) in present)
            or (c.get("kind") == "form_factor" and form == canonical_attribute(c.get("value")))
            for c in ev["hard_constraints"] if c.get("kind") != "unverifiable"
        ),
    }


def rank_metrics(ranked_ids: list[str], verified: set[str], k_values=(1, 5, 10)) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    for k in k_values:
        metrics[f"hit@{k}"] = int(any(item in verified for item in ranked_ids[:k]))
    mrr = 0.0
    for index, item in enumerate(ranked_ids[:10], start=1):
        if item in verified:
            mrr = 1.0 / index
            break
    metrics["mrr@10"] = round(mrr, 4)
    metrics["first_verified_rank"] = next(
        (index for index, item in enumerate(ranked_ids[:TOPK], start=1) if item in verified), None
    )
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-dir", type=Path, default=RUNS)
    args = parser.parse_args()
    runs = args.runs_dir
    runs.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()

    load_started = time.perf_counter()
    catalogue = ElectronicsCatalogue.load()
    catalogue_load_seconds = round(time.perf_counter() - load_started, 2)
    index_started = time.perf_counter()
    search = catalogue.search_source
    index_seconds = round(time.perf_counter() - index_started, 2)
    agent_tasks = _load("agent_tasks.jsonl")
    evaluator = {ev["task_id"]: ev for ev in _load("evaluator_only.jsonl")}
    scorers = {
        "search_preferences": PreferenceAwareScorer(),
        "no_history": PreferenceAwareScorer(),
        "shuffled_history": PreferenceAwareScorer(),
    }
    reader_config = None

    candidate_pools: list[dict[str, Any]] = []
    memory_reads: list[dict[str, Any]] = []
    results: dict[str, list[dict[str, Any]]] = {method: [] for method in METHODS}
    timings: list[dict[str, Any]] = []

    # First pass: fixed candidates and legal memory reads (one per user/task).
    per_task: list[dict[str, Any]] = []
    for agent in agent_tasks:
        task_id = agent["task_id"]
        ev = evaluator[task_id]
        query = agent["current_query"]
        search_started = time.perf_counter()
        search_result = search.search(query, limit=TOPK)
        search_ms = round((time.perf_counter() - search_started) * 1000, 3)
        candidates = []
        for rank, hit in enumerate(search_result.hits, start=1):
            item_id = hit.parent_asin.split(":", 1)[-1]
            product = catalogue.get(item_id)
            if product is None:
                continue
            candidates.append(candidate_payload(product, rank, hit.score))
        candidates.sort(key=lambda item: item["query_rank"])
        candidate_pools.append(
            {
                "task_id": task_id,
                "current_query": query,
                "top_k": TOPK,
                "catalogue_namespace": ELECTRONICS_NAMESPACE,
                "catalogue_size": catalogue.catalogue_size,
                "candidate_ids": [c["product_ref"]["item_id"] for c in candidates],
                "search_scores": {c["product_ref"]["item_id"]: c["query_score"] for c in candidates},
                "score_kind": "bm25",
                "candidates": candidates,
            }
        )

        memory_started = time.perf_counter()
        with ShoppingCompanionMemory() as memory:
            memory.ingest_sessions(
                task_id,
                [dict(session) for session in agent["history_sessions"]],
                source_kind=SOURCE_KIND_SYNTHETIC_FIXTURE,
                source_ref=f"p14b0:{task_id}",
            )
            legal_read = HistoryStateReader(memory.preference_service).read(
                task_id, conversation_id=task_id, current_query=query
            )
            lexical = memory.search(query, scope="long_term", k=10, conversation_id=task_id)
        memory_ms = round((time.perf_counter() - memory_started) * 1000, 3)
        legal_prefs = canonical_preferences(legal_read["active_preferences"])
        lexical_prefs = canonical_preferences(list(lexical.get("preferences") or []))
        memory_reads.append(
            {
                "task_id": task_id,
                "lexical_read": {
                    "status": lexical.get("status"),
                    "preferences_count": len(lexical_prefs),
                    "preferences": lexical_prefs,
                },
                "history_state_read": {
                    "status": legal_read["status"],
                    "active_count": len(legal_read["active_preferences"]),
                    "inactive_count": len(legal_read["inactive_preferences"]),
                    "canonical_preferences": legal_prefs,
                    "query_relevant": legal_read["relevant"],
                    "read_policy": legal_read["read_policy"],
                    "provenance": legal_read["provenance"],
                },
                "memory_ms": memory_ms,
            }
        )
        per_task.append(
            {
                "agent": agent,
                "evaluator": ev,
                "candidates": candidates,
                "hard_requirements": hard_requirements_from_query(query),
                "true_preferences": legal_prefs,
                "lexical_preferences": lexical_prefs,
                "search_ms": search_ms,
            }
        )

    # Shuffled-history diagnostic: each task uses another legitimate user's legal read.
    true_prefs_by_task = {item["task_id"]: item["history_state_read"]["canonical_preferences"] for item in memory_reads}
    shuffled_for: dict[str, list[dict[str, Any]]] = {}
    task_ids = [agent["task_id"] for agent in agent_tasks]
    for index, task_id in enumerate(task_ids):
        other = task_ids[(index + 1) % len(task_ids)]
        shuffled_for[task_id] = true_prefs_by_task.get(other, [])

    # Second pass: rank the SAME fixed candidate pool four ways and evaluate.
    for item in per_task:
        agent, ev = item["agent"], item["evaluator"]
        task_id = agent["task_id"]
        query = agent["current_query"]
        candidates = item["candidates"]
        hard = item["hard_requirements"]
        verified = verified_ids(ev, catalogue) if ev["expected_decision"] == "recommend" else set()
        ranked_search_only = [c["product_ref"]["item_id"] for c in candidates]
        methods: dict[str, dict[str, Any]] = {}
        scoring_ms_values: list[float] = []
        for method in METHODS:
            if method == "search_only":
                ranked = candidates
                prefs = []
            elif method == "search_preferences":
                prefs = item["true_preferences"]
                scoring_started = time.perf_counter()
                ranked = scorers["search_preferences"].score(
                    preferences=prefs, current_query=query, candidates=candidates,
                    hard_requirements=hard, namespace=ELECTRONICS_NAMESPACE,
                )
                scoring_ms_values.append((time.perf_counter() - scoring_started) * 1000.0)
            elif method == "no_history":
                prefs = []
                scoring_started = time.perf_counter()
                ranked = scorers["no_history"].score(
                    preferences=prefs, current_query=query, candidates=candidates,
                    hard_requirements=hard, namespace=ELECTRONICS_NAMESPACE,
                )
                scoring_ms_values.append((time.perf_counter() - scoring_started) * 1000.0)
            else:
                prefs = shuffled_for[task_id]
                scoring_started = time.perf_counter()
                ranked = scorers["shuffled_history"].score(
                    preferences=prefs, current_query=query, candidates=candidates,
                    hard_requirements=hard, namespace=ELECTRONICS_NAMESPACE,
                )
                scoring_ms_values.append((time.perf_counter() - scoring_started) * 1000.0)
            if method == "search_only":
                ranked_ids = list(ranked_search_only)
                top1 = candidates[0] if candidates else None
                scorer_rows = None
            else:
                ranked_ids = [row["product_ref"]["item_id"] for row in ranked]
                first_presentable = next((row for row in ranked if row["presentable"]), None)
                top1 = None
                if first_presentable is not None:
                    top1 = next(
                        (c for c in candidates if c["product_ref"]["item_id"] == first_presentable["product_ref"]["item_id"]),
                        None,
                    )
                scorer_rows = ranked
            metrics = rank_metrics(ranked_ids, verified) if ev["expected_decision"] == "recommend" else {}
            satisfaction = eval_top1(top1, ev)
            row = {
                "task_id": task_id,
                "task_type": ev["task_type"],
                "expected_decision": ev["expected_decision"],
                "method": method,
                "preferences_used": prefs,
                "hard_requirements": hard,
                "candidate_ids": [c["product_ref"]["item_id"] for c in candidates],
                "ranked_ids": ranked_ids,
                "top1": top1["product_ref"] if top1 else None,
                "metrics": metrics,
                "top1_evaluation": satisfaction,
                "verified_in_pool": len(verified & {c["product_ref"]["item_id"] for c in candidates}) if verified else 0,
                "verified_total": len(verified),
                "scoring_version": "preference_scorer_v0.1" if scorer_rows is not None else "bm25_search_order",
            }
            results[method].append(row)
            methods[method] = row
        timings.append({
            "task_id": task_id,
            "search_ms": item["search_ms"],
            "scoring_ms": round(sum(scoring_ms_values), 3),
        })

    # Aggregates for the 26 recommend tasks.
    def aggregate(rows: list[dict[str, Any]], label: str) -> dict[str, Any]:
        recommend = [row for row in rows if row["expected_decision"] == "recommend"]
        pool_verified = [row for row in recommend if row["verified_in_pool"] > 0]
        out: dict[str, Any] = {
            "label": label,
            "recommend_tasks": len(recommend),
            "recommend_tasks_with_verified_in_pool": len(pool_verified),
            "hit@1": round(sum(row["metrics"].get("hit@1", 0) for row in recommend) / len(recommend), 4) if recommend else 0.0,
            "hit@5": round(sum(row["metrics"].get("hit@5", 0) for row in recommend) / len(recommend), 4) if recommend else 0.0,
            "hit@10": round(sum(row["metrics"].get("hit@10", 0) for row in recommend) / len(recommend), 4) if recommend else 0.0,
            "mrr@10": round(sum(row["metrics"].get("mrr@10", 0.0) for row in recommend) / len(recommend), 4) if recommend else 0.0,
            "top1_matched_preferences": sum(row["top1_evaluation"]["matched"] for row in recommend),
            "top1_conflicting_preferences": sum(row["top1_evaluation"]["violated"] for row in recommend),
            "top1_unknown_preferences": sum(row["top1_evaluation"]["unknown"] for row in recommend),
            "top1_hard_violations": sum(row["top1_evaluation"]["hard_violations"] for row in recommend),
            "hard_verified_top1": sum(1 for row in recommend if row["top1_evaluation"].get("hard_verified")),
        }
        nonrec = [row for row in rows if row["expected_decision"] != "recommend"]
        out["non_recommend_tasks"] = len(nonrec)
        out["non_recommend_false_positive_presentations"] = sum(
            1 for row in nonrec if row["top1"] is not None and row["method"] != "search_only"
        )
        out["non_recommend_hard_violations"] = sum(row["top1_evaluation"]["hard_violations"] for row in nonrec)
        return out

    summary = {
        "schema": SCHEMA,
        "catalogue": {
            "namespace": ELECTRONICS_NAMESPACE,
            "stats": catalogue.stats,
        },
        "config": {
            "top_k": TOPK,
            "score_kind": "bm25",
            "scorer": {
                "query_weight": ScorerConfig().query_weight,
                "positive_weight": ScorerConfig().positive_weight,
                "negative_weight": ScorerConfig().negative_weight,
                "hard_unknown_penalty": ScorerConfig().hard_unknown_penalty,
            },
            "scoring_version": "preference_scorer_v0.1",
            "weights_fixed_before_run": True,
        },
        "aggregates": {method: aggregate(rows, method) for method, rows in results.items()},
        "memory_read_comparison": {
            "lexical_read_tasks_with_preferences": sum(1 for m in memory_reads if m["lexical_read"]["preferences_count"] > 0),
            "history_state_read_tasks_with_preferences": sum(1 for m in memory_reads if m["history_state_read"]["active_count"] > 0),
            "history_state_read_tasks_with_inactive_entries": sum(1 for m in memory_reads if m["history_state_read"]["inactive_count"] > 0),
            "history_state_read_query_relevant_tasks": sum(1 for m in memory_reads if m["history_state_read"]["query_relevant"]),
        },
        "resources": {
            "wall_seconds": round(time.perf_counter() - started, 2),
            "catalogue_load_seconds": catalogue_load_seconds,
            "bm25_index_seconds": index_seconds,
        },
    }

    # Same-query cross-persona contrast (5 groups); ranking difference alone is not improvement.
    rows_by_method = {method: {row["task_id"]: row for row in rows} for method, rows in results.items()}
    pair_groups: dict[str, list[str]] = defaultdict(list)
    for ev in evaluator.values():
        if ev["task_type"] != "same_query_persona_pair":
            continue
        task_id = ev["task_id"]
        agent_query = next(row for row in results["search_preferences"] if row["task_id"] == task_id)
        pairs = pair_groups.setdefault(agent_query["task_id"], agent_query["task_id"])
        pair_groups.setdefault(agent_query["task_id"], agent_query["task_id"])
    # group pairs by the agent-visible query of the pair members
    pair_groups = defaultdict(list)
    for ev in evaluator.values():
        if ev["task_type"] != "same_query_persona_pair":
            continue
        task_id = ev["task_id"]
        query = next(agent["current_query"] for agent in agent_tasks if agent["task_id"] == task_id)
        pair_groups[query].append(task_id)
    cross_persona: list[dict[str, Any]] = []
    for query, group in pair_groups.items():
        if len(group) != 2:
            continue
        left_id, right_id = group
        left = rows_by_method["search_preferences"][left_id]
        right = rows_by_method["search_preferences"][right_id]
        left_verified = verified_ids(evaluator[left_id], catalogue)
        right_verified = verified_ids(evaluator[right_id], catalogue)
        left_rank = next((index for index, item in enumerate(left["ranked_ids"], start=1) if item in left_verified), None)
        right_rank = next((index for index, item in enumerate(right["ranked_ids"], start=1) if item in right_verified), None)
        cross_persona.append({
            "query": query,
            "left_task": left_id,
            "right_task": right_id,
            "left_preferences": left["preferences_used"],
            "right_preferences": right["preferences_used"],
            "left_top1": left["top1"],
            "right_top1": right["top1"],
            "top1_differs": left["top1"] != right["top1"],
            "left_verified_rank": left_rank,
            "right_verified_rank": right_rank,
            "left_matched": left["top1_evaluation"]["matched"],
            "right_matched": right["top1_evaluation"]["matched"],
            "left_violated": left["top1_evaluation"]["violated"],
            "right_violated": right["top1_evaluation"]["violated"],
            "rank_change_has_preference_evidence": bool(
                (left["top1"] != right["top1"])
                and (left["top1_evaluation"]["matched"] + right["top1_evaluation"]["matched"] > 0)
            ),
        })
    summary["same_query_cross_persona"] = cross_persona

    # Per-task error analysis for the 26 recommend tasks.
    error_analysis: list[dict[str, Any]] = []
    for task_id, ev in evaluator.items():
        if ev["expected_decision"] != "recommend":
            continue
        a = rows_by_method["search_only"][task_id]
        b = rows_by_method["search_preferences"][task_id]
        verified = verified_ids(ev, catalogue)
        a_top1 = a["top1"]["item_id"] if a["top1"] else None
        b_top1 = b["top1"]["item_id"] if b["top1"] else None
        category = (
            "retrieval_unreachable" if a["verified_in_pool"] == 0
            else "both_wrong" if a_top1 not in verified and b_top1 not in verified
            else "b_only" if b_top1 in verified and a_top1 not in verified
            else "a_only" if a_top1 in verified and b_top1 not in verified
            else "both_right"
        )
        error_analysis.append({
            "task_id": task_id,
            "task_type": ev["task_type"],
            "category": category,
            "verified_total": len(verified),
            "verified_in_pool": a["verified_in_pool"],
            "search_only_top1": a_top1,
            "preferences_top1": b_top1,
            "search_only_verified_rank": a["metrics"].get("first_verified_rank"),
            "preferences_verified_rank": b["metrics"].get("first_verified_rank"),
            "preferences_used": b["preferences_used"],
            "hard_requirements": b["hard_requirements"],
            "top1_matched": b["top1_evaluation"]["matched"],
            "top1_unknown": b["top1_evaluation"]["unknown"],
        })
    summary["error_analysis"] = dict(Counter(row["category"] for row in error_analysis))
    summary["error_analysis_rows"] = error_analysis

    # Non-recommend decision audit (clarification / no_verified_match / insufficient_evidence).
    nonrecommend_audit: list[dict[str, Any]] = []
    for task_id, ev in evaluator.items():
        if ev["expected_decision"] == "recommend":
            continue
        b = rows_by_method["search_preferences"][task_id]
        nonrecommend_audit.append({
            "task_id": task_id,
            "task_type": ev["task_type"],
            "expected_decision": ev["expected_decision"],
            "would_present": b["top1"] is not None,
            "hard_requirements": b["hard_requirements"],
            "hard_violations": b["top1_evaluation"]["hard_violations"],
            "verified_total": b["verified_total"],
            "scorer_has_decision_gate": False,
            "note": "scorer only reranks; ASK/refusal policy must decide whether to present",
        })
    summary["non_recommend_audit"] = nonrecommend_audit

    # Write artifacts.
    def dump_jsonl(name: str, rows: list[dict[str, Any]]) -> str:
        path = runs / name
        with path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        return str(path.relative_to(REPO)).replace("\\", "/")

    summary["artifacts"] = {
        "candidate_pools": dump_jsonl("candidate_pools.jsonl", [
            {key: pool[key] for key in ("task_id", "current_query", "top_k", "catalogue_namespace",
                                        "catalogue_size", "candidate_ids", "search_scores", "score_kind")}
            for pool in candidate_pools
        ]),
        "search_only": dump_jsonl("search_only.jsonl", results["search_only"]),
        "search_preferences": dump_jsonl("search_preferences.jsonl", results["search_preferences"]),
        "no_history": dump_jsonl("no_history.jsonl", results["no_history"]),
        "shuffled_history": dump_jsonl("shuffled_history.jsonl", results["shuffled_history"]),
        "memory_reads": dump_jsonl("memory_reads.jsonl", memory_reads),
        "timings": dump_jsonl("timings.jsonl", timings),
        "same_query_cross_persona": dump_jsonl("same_query_cross_persona.jsonl", cross_persona),
        "error_analysis": dump_jsonl("error_analysis.jsonl", error_analysis),
        "non_recommend_audit": dump_jsonl("non_recommend_audit.jsonl", nonrecommend_audit),
        "summary": "runs/p14b0/summary.json",
    }
    (runs / "summary.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    (runs / "manifest.json").write_text(json.dumps({
        "schema": "agentrecx.p14b0.manifest.v1",
        "generated_by": "experiments/p14b0/evaluate.py",
        "benchmark": "data/p14e/benchmark (frozen pilot_dev; labels not modified)",
        "catalogue": "data/p13r/electronics/catalog_meta.jsonl",
        "methods": list(METHODS),
        "artifacts": summary["artifacts"],
    }, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({method: summary["aggregates"][method] for method in METHODS}, indent=1, ensure_ascii=False))
    print(json.dumps(summary["memory_read_comparison"], indent=1, ensure_ascii=False))
    print(json.dumps(summary["resources"], indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()

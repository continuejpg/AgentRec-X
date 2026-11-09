"""P14-B1 four-way diagnosis: {Search Only, Search+Preferences} x {Gate Off, Gate On}.

Uses the frozen P14-B0 candidate pools (runs/p14b0/candidate_pools.jsonl), the P14-B0
legal memory reads (runs/p14b0/memory_reads.jsonl), the accepted scorer, and the new
Decision Gate.  Evaluator-only labels are read ONLY after the decision, to score the
outcome; the gate context contains no label fields.

The fixed-candidate-pool comparison never expands the candidates.  When the gate asks
for one more evidence/retrieval round, the harness records the request and evaluates
the gate again with the budget consumed (honest bounded outcome); it does NOT inject
new products.  This is separate from the real Agent loop in runtime_smoke.py.

Artifacts: runs/p14b1/{candidate_decisions,gate_diagnosis,summary}.json(l).
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import replace
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from experiments.p14e.catalogue import ELECTRONICS_NAMESPACE, ElectronicsCatalogue
from experiments.p14e.labeling import assess_candidate
from recommendation.preference.decision_gate import (
    ALLOW_FINALIZE,
    BUDGET_EXHAUSTED,
    NEED_MORE_CANDIDATES,
    NEED_MORE_PRODUCT_EVIDENCE,
    NO_VERIFIED_CANDIDATE_IN_SCOPE,
    NEED_USER_CLARIFICATION,
    DecisionGate,
    GateBudget,
    GateContext,
)
from recommendation.preference.scorer import PreferenceAwareScorer

REPO = Path(__file__).resolve().parents[2]
DATA = REPO / "data/p14e/benchmark"
P14B0 = REPO / "runs/p14b0"
RUNS = REPO / "runs/p14b1"
TOPK = 20

ATTR_CANON = {
    "over-ear": "over_ear", "over ear": "over_ear", "in-ear": "in_ear", "in ear": "in_ear",
    "earbud": "in_ear", "on-ear": "on_ear", "on ear": "on_ear",
    "noise cancelling": "noise_cancelling", "noise canceling": "noise_cancelling",
    "wireless": "wireless", "wired": "wired", "microphone": "microphone",
    "waterproof": "waterproof", "water resistant": "waterproof", "sweatproof": "waterproof",
    "foldable": "foldable", "mechanical": "mechanical", "rgb": "rgb", "backlit": "rgb",
    "rechargeable": "rechargeable", "multidevice": "multidevice", "usb": "usb",
    "silent": "silent", "quiet": "silent", "ergonomic": "ergonomic", "gaming": "gaming",
}
FORM_ATTRS = ("over_ear", "on_ear", "in_ear")
CONFLICT_PAIRS = (("wireless", "wired"), ("over_ear", "in_ear"), ("on_ear", "in_ear"))
QUERY_TOKENS = tuple(name for name in ATTR_CANON if name not in {"earbud"})


def _load(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _compact(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(text or "").lower())


def canonical(value: Any) -> str | None:
    compact = _compact(value)
    if not compact:
        return None
    for needle, attr in ATTR_CANON.items():
        if _compact(needle) and _compact(needle) in compact:
            return attr
    return None


def canonical_prefs(payload: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out, seen = [], set()
    for item in payload:
        attr = canonical(item.get("value"))
        if not attr:
            continue
        polarity = str(item.get("polarity") or "positive").lower()
        polarity = "negative" if polarity in {"avoid", "negative", "dislike"} else "positive"
        key = (attr, polarity)
        if key in seen:
            continue
        seen.add(key)
        out.append({"value": attr, "polarity": polarity, "kind": "form_factor" if attr in FORM_ATTRS else "attribute",
                    "memory_id": item.get("memory_id"), "source_turn_id": item.get("source_turn_id")})
    return out


def conflicts(prefs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    positives = {p["value"] for p in prefs if p["polarity"] == "positive"}
    found = []
    for left, right in CONFLICT_PAIRS:
        if left in positives and right in positives:
            found.append({"left": left, "right": right, "reason": "mutually_exclusive_positive_preferences"})
    return found


def hard_requirements(query: str) -> list[dict[str, Any]]:
    import re

    reqs: list[dict[str, Any]] = []
    if re.search(r"\$\s?\d+", query) and re.search(r"budget|price|under\s+\$", query, re.I):
        reqs.append({"key": "price", "kind": "unverifiable", "value": "price"})
    if re.search(r"warranty", query, re.I):
        reqs.append({"key": "warranty", "kind": "unverifiable", "value": "warranty"})
    for match in re.finditer(r"(?:it\s+(?:needs|has|must)\s+to\s+be|it\s+must\s+be|must\s+be)\s+([^.!]+)", query, re.I):
        attr = canonical(match.group(1))
        if attr:
            reqs.append({"key": attr, "kind": "form_factor" if attr in FORM_ATTRS else "attribute", "value": attr})
    seen, unique = set(), []
    for req in reqs:
        if req["key"] not in seen:
            seen.add(req["key"])
            unique.append(req)
    return unique


def query_specificity(query: str, hard: list[dict[str, Any]], prefs: list[dict[str, Any]]) -> bool:
    if hard or prefs:
        return True
    compact_query = _compact(query)
    return any(_compact(token) in compact_query for token in QUERY_TOKENS)


def candidate_rows(
    task: dict[str, Any],
    catalogue: ElectronicsCatalogue,
    scorer: PreferenceAwareScorer,
    prefs: list[dict[str, Any]],
    hard: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    candidates = []
    for rank, item_id in enumerate(task["candidate_ids"], start=1):
        product = catalogue.get(item_id)
        if product is None:
            continue
        attributes = {
            name: {"evidence": fact.get("evidence"), "source": "catalog_meta:text"}
            for name, fact in (product.attributes or {}).items()
        }
        candidates.append({
            "product_ref": {"namespace": ELECTRONICS_NAMESPACE, "item_id": item_id},
            "query_rank": rank,
            "query_score": task["search_scores"].get(item_id),
            "attributes": attributes,
            "form_factor": product.form_factor,
        })
    return scorer.score(
        preferences=prefs, current_query=task["current_query"], candidates=candidates,
        hard_requirements=hard, namespace=ELECTRONICS_NAMESPACE,
    )


def verified_ids(ev: dict[str, Any], catalogue: ElectronicsCatalogue) -> set[str]:
    pool = catalogue.by_type(ev["product_type_pool"])
    return {
        product.item_id for product in pool
        if assess_candidate(product, hard_constraints=ev["hard_constraints"], preferences=ev["effective_preferences"]).acceptable
    }


def evaluate_top1(top1_id: str | None, ev: dict[str, Any], catalogue: ElectronicsCatalogue, verified: set[str]) -> dict[str, Any]:
    """Outcome evaluation only; never used to build gate context."""
    if top1_id is None:
        return {"presented": False, "correct": None, "verified_total": len(verified)}
    product = catalogue.get(top1_id)
    present = set((product.attributes or {})) if product else set()
    form = product.form_factor if product else None
    matched = violated = unknown = hard_violations = 0
    for preference in ev["effective_preferences"]:
        attr = canonical(preference.get("value"))
        if not attr:
            unknown += 1
            continue
        observed = form == attr if attr in FORM_ATTRS else attr in present
        if observed and preference.get("polarity") == "positive":
            matched += 1
        elif observed and preference.get("polarity") == "negative":
            violated += 1
        else:
            unknown += 1
    for constraint in ev["hard_constraints"]:
        if constraint.get("kind") == "unverifiable":
            continue
        attr = canonical(constraint.get("value")) or str(constraint.get("value"))
        if attr in FORM_ATTRS:
            if form is not None and form != attr:
                hard_violations += 1
    return {
        "presented": True,
        "correct": top1_id in verified,
        "matched": matched,
        "violated": violated,
        "unknown": unknown,
        "hard_violations": hard_violations,
        "verified_total": len(verified),
    }


def run_gate(
    gate: DecisionGate,
    context_kwargs: dict[str, Any],
    max_rounds: int = 2,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Run the gate, consuming one bounded evidence/search round at a time."""
    trace: list[dict[str, Any]] = []
    budget = GateBudget(max_retrievals=1, max_product_evidence_rounds=max_rounds)
    decision = gate.check(GateContext(budget=budget, **context_kwargs))
    trace.append(decision.as_dict())
    for _ in range(4):
        if decision.action == NEED_MORE_PRODUCT_EVIDENCE:
            budget = replace(budget, product_evidence_rounds_used=budget.product_evidence_rounds_used + 1)
        elif decision.action == NEED_MORE_CANDIDATES:
            budget = replace(budget, retrievals_used=budget.retrievals_used + 1)
        else:
            break
        decision = gate.check(GateContext(budget=budget, **context_kwargs))
        trace.append(decision.as_dict())
    return decision.as_dict(), trace


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=RUNS)
    args = parser.parse_args()
    out_dir = args.output
    out_dir.mkdir(parents=True, exist_ok=True)

    catalogue = ElectronicsCatalogue.load()
    scorer = PreferenceAwareScorer()
    gate = DecisionGate()
    agent = {task["task_id"]: task for task in _load(DATA / "agent_tasks.jsonl")}
    evals = {task["task_id"]: task for task in _load(DATA / "evaluator_only.jsonl")}
    pools = {row["task_id"]: row for row in _load(P14B0 / "candidate_pools.jsonl")}
    memory = {row["task_id"]: row for row in _load(P14B0 / "memory_reads.jsonl")}
    # P14-B0 search_preferences rows already contain evidence flags/rank/verified count.
    p14b0 = {row["task_id"]: row for row in _load(P14B0 / "search_preferences.jsonl")}

    rows: list[dict[str, Any]] = []
    tasks: list[dict[str, Any]] = []
    for task_id, task in agent.items():
        ev = evals[task_id]
        pool = pools[task_id]
        true_prefs = canonical_prefs(memory[task_id]["history_state_read"]["canonical_preferences"])
        hard = hard_requirements(task["current_query"])
        specificity = query_specificity(task["current_query"], hard, true_prefs)
        pref_conflicts = conflicts(true_prefs)
        rank_rows = candidate_rows(pool, catalogue, scorer, true_prefs, hard)
        search_rows = sorted(rank_rows, key=lambda row: row["query_rank"])
        verified = verified_ids(ev, catalogue) if ev["expected_decision"] == "recommend" else set()
        common = {"query": task["current_query"], "active_preferences": true_prefs,
                  "hard_requirements": hard, "preference_conflicts": pref_conflicts,
                  "namespace": ELECTRONICS_NAMESPACE, "query_specificity": specificity}
        variants = {
            "search_only_gate_off": (search_rows, None),
            "search_only_gate_on": (search_rows, True),
            "search_preferences_gate_off": (rank_rows, None),
            "search_preferences_gate_on": (rank_rows, True),
        }
        for variant, (rows_for_pool, use_gate) in variants.items():
            top1_id = None
            action = "GATE_OFF"
            trace: list[dict[str, Any]] = []
            if use_gate:
                decision, trace = run_gate(gate, {**common, "candidates": rows_for_pool})
                action = decision["action"]
                if decision["allow_finalize"]:
                    top1_id = decision["selected_candidate_id"]
            else:
                top1_id = rows_for_pool[0]["product_ref"]["item_id"] if rows_for_pool else None
            outcome = evaluate_top1(top1_id, ev, catalogue, verified)
            row = {
                "task_id": task_id,
                "task_type": ev["task_type"],
                "expected_decision": ev["expected_decision"],
                "variant": variant,
                "gate_action": action,
                "top1": top1_id,
                "outcome": outcome,
                "verified_in_pool": len(verified & set(pool["candidate_ids"])),
                "verified_total": len(verified),
                "gate_trace": trace,
            }
            rows.append(row)
            tasks.append({"task_id": task_id, "variant": variant, "hard": hard, "prefs": true_prefs,
                          "conflicts": pref_conflicts, "specificity": specificity})
    # Artifacts
    with (out_dir / "candidate_decisions.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    (out_dir / "gate_diagnosis.json").write_text(json.dumps(rows, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")

    def aggregate(variant: str, subset: set[str] | None = None) -> dict[str, Any]:
        selected = [r for r in rows if r["variant"] == variant and (subset is None or r["task_id"] in subset)]
        recommend = [r for r in selected if r["expected_decision"] == "recommend"]
        nonrec = [r for r in selected if r["expected_decision"] != "recommend"]
        finalized = [r for r in selected if r["top1"] is not None]
        correct = [r for r in recommend if r["outcome"].get("correct")]
        wrong = [r for r in recommend if r["top1"] is not None and not r["outcome"].get("correct")]
        unnecessary_ask = [r for r in recommend if r["gate_action"] == NEED_USER_CLARIFICATION]
        unnecessary_refuse = [r for r in recommend if r["gate_action"] in
                              {NO_VERIFIED_CANDIDATE_IN_SCOPE, BUDGET_EXHAUSTED, NEED_MORE_CANDIDATES, NEED_MORE_PRODUCT_EVIDENCE}]
        false_finalize = [r for r in nonrec if r["top1"] is not None]
        correct_ask = [r for r in nonrec if r["expected_decision"] == "clarification_required" and r["gate_action"] == NEED_USER_CLARIFICATION]
        correct_unknown = [r for r in nonrec if r["expected_decision"] in {"no_verified_match", "insufficient_evidence"}
                           and r["gate_action"] in {NO_VERIFIED_CANDIDATE_IN_SCOPE, BUDGET_EXHAUSTED}]
        wrong_nonrec_action = [r for r in nonrec if r not in correct_ask and r not in correct_unknown and r["top1"] is None]
        return {
            "variant": variant,
            "recommend_tasks": len(recommend),
            "finalized_recommend": len([r for r in recommend if r["top1"] is not None]),
            "correct_top1": len(correct),
            "wrong_top1": len(wrong),
            "unnecessary_ask": len(unnecessary_ask),
            "unnecessary_refusal_or_more_evidence": len(unnecessary_refuse),
            "hard_violations_in_top1": sum(r["outcome"].get("hard_violations", 0) for r in recommend),
            "unknown_evidence_in_top1": sum(r["outcome"].get("unknown", 0) for r in recommend),
            "non_recommend_tasks": len(nonrec),
            "false_finalize": len(false_finalize),
            "correct_ask": len(correct_ask),
            "correct_unknown_handling": len(correct_unknown),
            "other_nonfinalize_actions": len(wrong_nonrec_action),
            "finalize_rate_recommend": round(len([r for r in recommend if r["top1"] is not None]) / max(len(recommend), 1), 4),
            "correct_top1_rate_recommend": round(len(correct) / max(len(recommend), 1), 4),
            "false_finalize_rate_nonrecommend": round(len(false_finalize) / max(len(nonrec), 1), 4),
        }

    all_ids = set(agent)
    pending = {tid for tid, ev in evals.items() if ev["construction_metadata"]["review_status"] == "pending_human_review"}
    nonpending = all_ids - pending
    summary = {
        "schema": "agentrecx.p14b1.gate_diagnosis.v1",
        "gate_version": gate.version,
        "variants": {variant: aggregate(variant) for variant in
                     ("search_only_gate_off", "search_only_gate_on",
                      "search_preferences_gate_off", "search_preferences_gate_on")},
        "pending_sensitivity": {
            "pending_total": len(pending),
            "pending_recommend": len(pending & {tid for tid, ev in evals.items() if ev["expected_decision"] == "recommend"}),
            "excluding_pending": {variant: aggregate(variant, nonpending) for variant in
                                  ("search_only_gate_off", "search_only_gate_on",
                                   "search_preferences_gate_off", "search_preferences_gate_on")},
        },
        "candidate_reachability": {
            "recommend_tasks": sum(1 for ev in evals.values() if ev["expected_decision"] == "recommend"),
            "verified_in_pool": sum(1 for tid, r in p14b0.items() if r["expected_decision"] == "recommend" and r["verified_in_pool"] > 0),
            "retrieval_unreachable": sum(1 for tid, r in p14b0.items() if r["expected_decision"] == "recommend" and r["verified_in_pool"] == 0),
        },
        "labels_untouched": True,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()

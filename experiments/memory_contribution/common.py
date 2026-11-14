"""Shared helpers for the Memory Contribution Phase 1 benchmark.

Phase 1 is a local CPU measurement, not a training run.  It reuses, unchanged:

* the fixed Original-BGE Top-20 pools from the existing Ranking v1 dataset;
* the provisional observed-acceptable labels from that dataset (joined *after*
  ranking, never passed to the scorer);
* :class:`recommendation.preference.scorer.PreferenceAwareScorer`;
* the legal M9 history read path and the deterministic DecisionGate.

The only ablation this module implements is an **input control**: the benchmark
runner tags every preference with its origin (``current_query`` vs ``history``) and
builds a Memory-OFF preference list that keeps current-query preferences while
dropping the historical channel.  The Scorer and the Gate themselves are not
modified.  In the current production pipeline the only soft-preference channel is
historical memory; current-query requirements live in ``hard_requirements`` and are
preserved in every arm.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from experiments.p14b1.diagnose import canonical_prefs, conflicts, query_specificity
from recommendation.memory.extraction import RuleBasedPreferenceExtractor
from recommendation.preference.decision_gate import (
    DecisionGate,
    GateBudget,
    GateContext,
)
from recommendation.preference.scorer import SCORER_VERSION, PreferenceAwareScorer

REPO = Path(__file__).resolve().parents[2]
RANKING_DATASET = REPO / "runs/preference_ranking_v1/ranking_dataset.jsonl"
RANKING_MANIFEST = REPO / "runs/preference_ranking_v1/manifest.json"
P20_SUITE = REPO / "runs/p20_agent_eval/task_suite.json"
RETRIEVAL_TRACE = REPO / "runs/advanced_retrieval_v2/task_arm_original_bge.json"
OUT_DIR = REPO / "runs/memory_contribution_phase1"

SCHEMA_TASK = "agentrecx.memory_contribution.fixed_pool_task.v1"
SCHEMA_COMPARISON = "agentrecx.memory_contribution.fixed_pool_comparison.v1"
SCHEMA_COUNTERFACTUAL = "agentrecx.memory_contribution.counterfactual_case.v1"
SCHEMA_COUNTERFACTUAL_SUMMARY = "agentrecx.memory_contribution.counterfactual_summary.v1"
SCHEMA_MANIFEST = "agentrecx.memory_contribution.manifest.v1"

ARMS = ("A_relevance_only", "B_scorer_memory_off", "C_scorer_memory_on")
ELIGIBLE_ROLE_STATES = frozenset({"MATCH", "NOT_REQUIRED"})
LABEL_STATUS = "provisional_automated_verified_not_human_gold"
TOP_K = 20

_QUERY_SOFT_PREFERENCE = re.compile(
    r"\b(?:i\s+)?(?:prefer|like|love|avoid|don'?t\s+like|do\s+not\s+like)\b",
    re.IGNORECASE,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def load_ranking_rows() -> list[dict[str, Any]]:
    """Load the existing Ranking v1 dataset and assert its task scope."""
    rows = read_jsonl(RANKING_DATASET)
    if len(rows) != 30:
        raise ValueError(f"expected 30 P20 ranking tasks, found {len(rows)}")
    recommend = [row for row in rows if row["expected_decision"] == "recommend"]
    if len(recommend) != 19:
        raise ValueError(f"expected 19 recommend tasks, found {len(recommend)}")
    return recommend


def load_suite_tasks() -> dict[str, dict[str, Any]]:
    suite = read_json(P20_SUITE)
    return {task["task_id"]: task for task in suite["tasks"]}


def history_preferences(row: Mapping[str, Any]) -> list[dict[str, Any]]:
    """History-channel preferences, tagged with an explicit origin."""
    return [
        {**dict(item), "origin": "history"}
        for item in row.get("historical_memory") or ()
    ]


def query_soft_preferences(query: str) -> list[dict[str, Any]]:
    """Current-query soft preferences stated explicitly, if any.

    The current pipeline has no query-soft-preference extraction step: a query such
    as "I need earbuds ..." is a product request, not a preference statement.  This
    helper therefore only fires on explicit preference wording ("I prefer ...",
    "I don't like ...").  It is deliberately conservative: product-category nouns
    are not turned into a synthetic soft preference.  The result is tagged
    ``origin="current_query"`` so the Memory-OFF arm can retain it.
    """
    if not _QUERY_SOFT_PREFERENCE.search(query or ""):
        return []
    extraction = RuleBasedPreferenceExtractor().extract(query)
    payload = [
        {
            "value": candidate.value,
            "polarity": candidate.polarity.value,
            "kind": candidate.kind.value,
            "source_turn_id": "current_query",
            "source_text": candidate.source_text,
        }
        for candidate in extraction.preferences
    ]
    return [{**item, "origin": "current_query"} for item in canonical_prefs(payload)]


def combined_preferences(
    *,
    query_preferences: Sequence[Mapping[str, Any]],
    history_preferences: Sequence[Mapping[str, Any]],
    memory_enabled: bool,
) -> list[dict[str, Any]]:
    """Build the Scorer input for one arm.

    ``memory_enabled=False`` keeps current-query preferences (there are none in this
    task set, but the code path is real) and drops the historical channel.  It does
    **not** clear the hard requirements or any eligibility filter.
    """
    selected: list[dict[str, Any]] = [dict(item) for item in query_preferences]
    if memory_enabled:
        selected.extend(dict(item) for item in history_preferences)
    out: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for item in selected:
        value = str(item.get("value") or "").strip()
        if not value:
            continue
        key = (value.casefold(), str(item.get("polarity") or "positive").casefold())
        if key in seen:
            continue
        seen.add(key)
        out.append(item)
    return out


def candidate_inputs(row: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Convert the stored fixed pool into Scorer candidate records.

    The stored ``scorer`` row already carries the preference-independent facts used
    for eligibility: ``hard_tier``/``hard_status`` and the structural role fields.
    """
    candidates: list[dict[str, Any]] = []
    for item in row["candidate_pool"]:
        scorer_row = item.get("scorer") or {}
        candidates.append(
            {
                "product_ref": {
                    "namespace": "amazon_electronics_c4",
                    "item_id": item["item_id"],
                },
                "query_rank": int(item["original_bge_rank"]),
                "query_score": item.get("original_bge_score"),
                "attributes": dict(item.get("attributes") or {}),
                "form_factor": item.get("form_factor"),
                "_existing_scorer": scorer_row,
            }
        )
    return candidates


def pool_items(row: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    return {item["item_id"]: item for item in row["candidate_pool"]}


def eligibility(row: Mapping[str, Any]) -> tuple[list[str], dict[str, str]]:
    """The shared structural filter for every arm.

    A candidate is eligible only when the existing Scorer gives it hard tier 0 and
    the existing structural role check gives MATCH/NOT_REQUIRED.  This is exactly
    the filter behind the recorded ``C_hard_filtered_scorer`` baseline.  It is
    preference-independent, so A/B/C always rank the same eligible set.
    """
    eligible: list[str] = []
    excluded: dict[str, str] = {}
    for item in row["candidate_pool"]:
        scorer_row = item.get("scorer") or {}
        item_id = item["item_id"]
        hard_tier = int(scorer_row.get("hard_tier") or 0)
        role = str(scorer_row.get("role_status") or "NOT_REQUIRED")
        if hard_tier != 0:
            excluded[item_id] = f"hard_tier_{hard_tier}"
            continue
        if role not in ELIGIBLE_ROLE_STATES:
            excluded[item_id] = f"role_{role}"
            continue
        eligible.append(item_id)
    return eligible, excluded


def score_pool(
    row: Mapping[str, Any],
    preferences: Sequence[Mapping[str, Any]],
    *,
    scorer: PreferenceAwareScorer | None = None,
) -> list[dict[str, Any]]:
    """Score the fixed pool with a given preference list; return Scorer rows."""
    scorer = scorer or PreferenceAwareScorer()
    candidates = candidate_inputs(row)
    clean = [
        {key: value for key, value in candidate.items() if not key.startswith("_")}
        for candidate in candidates
    ]
    scored = scorer.score(
        preferences=preferences,
        current_query=row["query"],
        candidates=clean,
        hard_requirements=row.get("hard_requirements") or (),
        namespace="amazon_electronics_c4",
    )
    pool = pool_items(row)
    for scored_row in scored:
        item_id = scored_row["product_ref"]["item_id"]
        stored = pool.get(item_id) or {}
        scored_row["role_status"] = str(
            (stored.get("scorer") or {}).get("role_status") or "NOT_REQUIRED"
        )
        scored_row["role_evidence"] = list(
            (stored.get("scorer") or {}).get("role_evidence") or ()
        )
        scored_row["query_rank"] = int(stored.get("original_bge_rank") or scored_row["query_rank"])
        scored_row["title"] = stored.get("title")
        scored_row["form_factor"] = stored.get("form_factor")
        scored_row["attributes"] = dict(stored.get("attributes") or {})
        scored_row["label"] = stored.get("label", "unjudged")
        scored_row["task_constraint_conflicts"] = []
    return scored


def ordered_ids(
    scored_rows: Sequence[Mapping[str, Any]], eligible: Iterable[str]
) -> list[str]:
    allowed = set(eligible)
    return [
        row["product_ref"]["item_id"]
        for row in scored_rows
        if row["product_ref"]["item_id"] in allowed
    ]


def rank_map(ranked_ids: Sequence[str]) -> dict[str, int]:
    return {item_id: index + 1 for index, item_id in enumerate(ranked_ids)}


def hit_metrics(ranked_ids: Sequence[str], acceptable: Iterable[str]) -> dict[str, Any]:
    """Observed-acceptable metrics; unlabelled candidates stay unjudged."""
    acceptable = set(acceptable)
    ranks = [
        (index + 1, item_id)
        for index, item_id in enumerate(ranked_ids)
        if item_id in acceptable
    ]
    first = min((rank for rank, _ in ranks), default=None)
    return {
        "acceptable_ranks": {item_id: rank for rank, item_id in ranks},
        "first_acceptable_rank": first,
        "hit@1": bool(first is not None and first <= 1),
        "hit@3": bool(first is not None and first <= 3),
        "hit@5": bool(first is not None and first <= 5),
        "hit@10": bool(first is not None and first <= 10),
        "reciprocal_rank@10": round(1.0 / first, 6) if first is not None and first <= 10 else 0.0,
        "acceptable_in_ranking": len(ranks),
    }


def hard_violations(ranked_ids: Sequence[str], rows_by_id: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    def count(k: int) -> int:
        return sum(
            1
            for item_id in ranked_ids[:k]
            if int((rows_by_id.get(item_id) or {}).get("hard_tier") or 0) == 2
        )

    return {"violations@1": count(1), "violations@3": count(3), "violations@5": count(5)}


def gate_for_arm(
    *,
    row: Mapping[str, Any],
    scored_rows: Sequence[Mapping[str, Any]],
    preferences: Sequence[Mapping[str, Any]],
    order: Sequence[str],
    gate: DecisionGate | None = None,
) -> dict[str, Any]:
    """Run the production DecisionGate on one arm's final ordering.

    The budget is at its post-retrieval/post-inspection state: no further search or
    evidence round is allowed.  This is the same gate configuration for all arms.
    """
    gate = gate or DecisionGate()
    rows_by_id = {item["product_ref"]["item_id"]: item for item in scored_rows}
    ordered_rows = [rows_by_id[item_id] for item_id in order if item_id in rows_by_id]
    # Expected type/role are stored on the fixed-pool scorer row by the original
    # benchmark builder; the Scorer itself does not need them for scoring.
    target_type = _expected_product_type(row)
    context = GateContext(
        query=row["query"],
        candidates=ordered_rows,
        active_preferences=list(preferences),
        hard_requirements=list(row.get("hard_requirements") or ()),
        preference_conflicts=list(conflicts(list(preferences))),
        namespace="amazon_electronics_c4",
        query_specificity=query_specificity(
            row["query"], list(row.get("hard_requirements") or ()), list(preferences)
        ),
        expected_product_type=target_type,
        expected_product_role=_expected_product_role(row, target_type),
        budget=GateBudget(
            max_retrievals=1,
            max_product_evidence_rounds=1,
            retrievals_used=1,
            product_evidence_rounds_used=1,
        ),
        task_constraints=[],
    )
    return gate.check(context).as_dict()


def _expected_product_type(row: Mapping[str, Any]) -> str | None:
    for item in row["candidate_pool"]:
        scorer_row = item.get("scorer") or {}
        value = scorer_row.get("expected_product_type")
        if value:
            return str(value)
    return None


def _expected_product_role(row: Mapping[str, Any], product_type: str | None) -> str | None:
    if not product_type:
        return None
    for item in row["candidate_pool"]:
        scorer_row = item.get("scorer") or {}
        value = scorer_row.get("expected_role")
        if value:
            return str(value)
    return "main"


def preference_summary(preferences: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "value": str(item.get("value") or ""),
            "polarity": str(item.get("polarity") or "positive"),
            "kind": item.get("kind"),
            "origin": item.get("origin"),
            "source_turn_id": item.get("source_turn_id"),
        }
        for item in preferences
    ]


def arm_candidate_rows(
    *,
    scored_rows: Sequence[Mapping[str, Any]],
    order: Sequence[str],
) -> list[dict[str, Any]]:
    """Per-candidate evidence for one arm, in that arm's rank order."""
    by_id = {item["product_ref"]["item_id"]: item for item in scored_rows}
    rows: list[dict[str, Any]] = []
    for rank, item_id in enumerate(order, start=1):
        row = by_id.get(item_id) or {}
        rows.append(
            {
                "rank": rank,
                "item_id": item_id,
                "original_bge_rank": row.get("query_rank"),
                "final_score": row.get("final_score"),
                "query_relevance_score": row.get("query_relevance_score"),
                "preference_match_score": row.get("preference_match_score"),
                "preference_conflict_score": row.get("preference_conflict_score"),
                "hard_requirement_penalty": row.get("hard_requirement_penalty"),
                "hard_tier": row.get("hard_tier"),
                "matched_preferences": [
                    str(item.get("value")) for item in row.get("matched_preferences") or ()
                ],
                "conflicting_preferences": [
                    str(item.get("value")) for item in row.get("conflicting_preferences") or ()
                ],
                "unknown_preferences": [
                    str(item.get("value")) for item in row.get("unknown_preferences") or ()
                ],
                "role_status": row.get("role_status"),
                "presentable": row.get("presentable"),
                "label": row.get("label"),
            }
        )
    return rows


def change_reasons(
    *,
    b_rows: Sequence[Mapping[str, Any]],
    c_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Explain every candidate whose order or score changed between B and C.

    The reason is diagnostic, not a quality claim: a positive match moving a
    candidate up is not evidence that the user likes it more.
    """
    b_rank = {item["item_id"]: item["rank"] for item in b_rows}
    c = {item["item_id"]: item for item in c_rows}
    changes: list[dict[str, Any]] = []
    for item_id, c_row in c.items():
        b_position = b_rank.get(item_id)
        b_score = next(
            (item["final_score"] for item in b_rows if item["item_id"] == item_id), None
        )
        order_changed = b_position != c_row["rank"]
        score_changed = b_score != c_row["final_score"]
        if not order_changed and not score_changed:
            continue
        matched = c_row["matched_preferences"]
        conflicting = c_row["conflicting_preferences"]
        unknown = c_row["unknown_preferences"]
        if matched:
            reason = "positive_memory_match"
        elif conflicting:
            reason = "negative_memory_conflict"
        elif unknown:
            reason = "memory_unknown_no_match"
        else:
            reason = "reordered_by_other_candidates"
        changes.append(
            {
                "item_id": item_id,
                "rank_B": b_position,
                "rank_C": c_row["rank"],
                "rank_delta_B_minus_C": (b_position - c_row["rank"])
                if b_position is not None
                else None,
                "score_B": b_score,
                "score_C": c_row["final_score"],
                "score_delta": (
                    round(float(c_row["final_score"]) - float(b_score), 6)
                    if b_score is not None
                    else None
                ),
                "order_changed": order_changed,
                "score_changed": score_changed,
                "matched_preferences": matched,
                "conflicting_preferences": conflicting,
                "unknown_preferences": unknown,
                "reason": reason,
            }
        )
    return sorted(changes, key=lambda item: (item["rank_C"], item["item_id"]))


def top_k(ranked_ids: Sequence[str], k: int) -> list[str]:
    return list(ranked_ids[:k])


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


SCORER_VERSION_REF = SCORER_VERSION

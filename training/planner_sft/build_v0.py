"""Build the P4A Planner SFT v0 dataset.

The labels are deterministic scenario/state labels (rule + tool semantics), not Qwen
outputs.  Candidate identities and evidence come from the real Amazon catalogue; user
requests are synthetic and marked as such.  The builder does not call any model.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from dataclasses import replace
from typing import Any, Iterable

from recommendation.api.app import ServiceSettings
from recommendation.catalog import MetadataIndex
from recommendation.demo.planner_contract import (
    ActionProposal,
    PlannerAction,
    PlannerObservation,
    allowed_planner_actions,
    validate_action,
)
from training.agent_sft.planner_prompt import planner_messages

SCHEMA_VERSION = "agentrecx.planner_sft.v0"
CONTRACT_VERSION = "agentrecx.planner_action.v1"
OUT = Path("data/planner_sft_v0")
KEYWORDS = ("water filter", "cookware", "lantern", "stove", "sleeping pad", "water bottle")
SCENARIOS = (
    ("retrieve_initial_clear", 60),
    ("retrieve_after_empty", 40),
    ("ask_ambiguous", 30),
    ("rerank_soft_weight", 50),
    ("rerank_soft_feature", 40),
    ("finalize_after_rerank", 50),
    ("finalize_no_supported_soft", 30),
    ("ask_missing_budget", 20),
)
TRIPS = ("backpacking", "car camping", "family camping", "bikepacking", "hiking", "tailgating")
COLORS = ("black", "blue", "green", "orange", "gray", "red")
BUDGETS = (20, 25, 30, 40, 50, 60, 75, 90, 120, 150)


def _details(record: Any) -> dict[str, str]:
    return {str(k): str(v) for k, v in (getattr(record, "details_dict", dict)() or {}).items()}


def _weight(details: dict[str, str]) -> str:
    for key, value in details.items():
        if "weight" in key.casefold():
            return value
    return ""


def _color(details: dict[str, str]) -> str:
    return next((v for k, v in details.items() if "color" in k.casefold()), "")


def _material(details: dict[str, str]) -> str:
    return next((v for k, v in details.items() if "material" in k.casefold()), "")


def _load_pools() -> dict[str, list[Any]]:
    records = list(MetadataIndex.load(ServiceSettings.from_env().catalog_path).records.values())
    pools: dict[str, list[Any]] = {}
    for keyword in KEYWORDS:
        rows = [
            record
            for record in records
            if keyword in str(getattr(record, "title", "") or "").casefold()
            and str(getattr(record, "price_text", "") or "").strip()
            and getattr(record, "parent_asin", None)
        ]
        rows.sort(key=lambda record: str(record.parent_asin))
        # Prefer a deterministic, varied pool; every row is a real catalogue item.
        pools[keyword] = rows[:160]
        if len(pools[keyword]) < 8:
            raise RuntimeError(f"catalogue pool too small for {keyword!r}: {len(pools[keyword])}")
    return pools


def _candidate_evidence(record: Any) -> dict[str, Any]:
    details = _details(record)
    weight = _weight(details)
    color = _color(details)
    material = _material(details)
    pieces = [f"title={record.title}"]
    if weight:
        pieces.append(f"weight={weight}")
    if color:
        pieces.append(f"color={color}")
    if material:
        pieces.append(f"material={material}")
    return {
        "parent_asin": str(record.parent_asin),
        "title": str(record.title),
        "price": str(record.price_text),
        "weight": weight,
        "color": color,
        "material": material,
        "evidence_text": "; ".join(pieces),
    }


def _pool_for(pools: dict[str, list[Any]], scenario: str, index: int, budget: int, count: int = 5) -> tuple[str, list[Any]]:
    keyword = KEYWORDS[index % len(KEYWORDS)]
    pool = pools[keyword]
    if scenario not in {"ask_ambiguous", "ask_missing_budget"}:
        affordable = []
        for row in pool:
            try:
                if float(str(row.price_text).replace(",", "")) <= budget:
                    affordable.append(row)
            except (TypeError, ValueError):
                continue
        if len(affordable) >= count:
            pool = affordable
    if scenario == "rerank_soft_weight":
        weighted = [row for row in pool if _weight(_details(row))]
        if len(weighted) >= count:
            pool = weighted
    offset = (index * 7 + len(scenario)) % max(1, len(pool) - count)
    return keyword, pool[offset:offset + count]


def _budget_for(keyword: str, scenario: str, index: int, pools: dict[str, list[Any]]) -> int:
    """Pick a budget whose real candidate pool is large enough for this sample."""
    pool = pools[keyword]
    start = index % len(BUDGETS)
    for offset in range(len(BUDGETS)):
        budget = BUDGETS[(start + offset) % len(BUDGETS)]
        rows = []
        for row in pool:
            try:
                if float(str(row.price_text).replace(",", "")) > budget:
                    continue
            except (TypeError, ValueError):
                continue
            if scenario == "rerank_soft_weight" and not _weight(_details(row)):
                continue
            rows.append(row)
        if len(rows) >= 5:
            return budget
    return max(BUDGETS)


def _query(scenario: str, keyword: str, index: int, budget: int) -> str:
    trip = TRIPS[index % len(TRIPS)]
    if scenario == "ask_ambiguous":
        return f"Can you help me pick something for {trip}? I have not decided what I need."
    if scenario == "ask_missing_budget":
        return f"I need a {keyword} for {trip}, but I do not know what budget is reasonable."
    if scenario == "retrieve_initial_clear":
        return f"I need a {keyword} under ${budget} for {trip}."
    if scenario == "retrieve_after_empty":
        return f"Try again for a {keyword} under ${budget} for {trip}."
    if scenario == "rerank_soft_weight":
        return f"I need a {keyword} under ${budget} for {trip}; lighter is better."
    if scenario == "rerank_soft_feature":
        return f"I need a {keyword} under ${budget} for {trip}; I prefer {COLORS[index % len(COLORS)]}."
    if scenario == "finalize_after_rerank":
        return f"Show me the best {keyword} under ${budget} for {trip}."
    if scenario == "finalize_no_supported_soft":
        return f"I need a {keyword} under ${budget} for {trip}; I care about noise level."
    return f"I need a {keyword} under ${budget} for {trip}; I do not have a strong color preference."


def _m5_state(scenario: str, keyword: str, index: int, budget: int) -> dict[str, Any]:
    state: dict[str, Any] = {
        "mode": "apply_constraints",
        "goal": keyword,
        "hard_constraint_count": 0 if scenario in {"ask_ambiguous", "ask_missing_budget"} else 1,
        "constraint_kinds": [] if scenario in {"ask_ambiguous", "ask_missing_budget"} else ["price_max"],
        "soft_preference_count": 0,
        "state_digest": hashlib.sha256(f"{scenario}:{keyword}:{index}".encode()).hexdigest()[:24],
    }
    if scenario in {"rerank_soft_weight", "rerank_soft_feature"}:
        state["soft_preference_count"] = 1
        state["soft_preferences"] = [
            {"attribute": "weight" if scenario == "rerank_soft_weight" else "color", "direction": "minimize" if scenario == "rerank_soft_weight" else "equal", "is_soft": True}
        ]
    if scenario == "finalize_no_supported_soft":
        state["soft_preference_count"] = 1
        state["soft_preferences"] = [{"attribute": "noise_level", "direction": "minimize", "is_soft": True}]
    if scenario not in {"ask_ambiguous", "ask_missing_budget"}:
        state["price_max"] = budget
    return state


def _observation(scenario: str, keyword: str, index: int, budget: int, candidates: list[dict[str, Any]]) -> PlannerObservation:
    has_candidates = scenario not in {"ask_ambiguous", "ask_missing_budget", "retrieve_initial_clear", "retrieve_after_empty"}
    reranked = scenario == "finalize_after_rerank"
    needs_input = scenario in {"ask_ambiguous", "ask_missing_budget"}
    candidate_ids = tuple(item["parent_asin"] for item in candidates) if has_candidates else ()
    if scenario == "retrieve_after_empty":
        last_tool = {"tool": "run_shopping_turn", "candidate_count": 0, "terminal": "no_candidates"}
    elif scenario in {"rerank_soft_weight", "rerank_soft_feature", "finalize_no_supported_soft"}:
        last_tool = {"tool": "run_shopping_turn", "candidate_count": len(candidate_ids)}
    elif reranked:
        last_tool = {"tool": "m6_1_soft_rerank", "applied": True, "candidate_count": len(candidate_ids)}
    else:
        last_tool = {}
    observation = PlannerObservation(
        step=index % 3,
        query=_query(scenario, keyword, index, budget),
        candidate_ids=candidate_ids,
        state_version=str(_m5_state(scenario, keyword, index, budget)["state_digest"]),
        needs_user_input=needs_input,
        reranked=reranked,
        m5_state=_m5_state(scenario, keyword, index, budget),
        last_tool_observation=last_tool,
        remaining_steps=3 - (index % 3),
        candidate_evidence=tuple(candidates) if has_candidates else (),
    )
    return observation


def _target(scenario: str, observation: PlannerObservation) -> tuple[PlannerAction, dict[str, Any], str]:
    if scenario in {"retrieve_initial_clear", "retrieve_after_empty"}:
        return PlannerAction.RETRIEVE, {"query": observation.query}, "No current candidates and the request is sufficiently specified; call the retrieval tool."
    if scenario in {"ask_ambiguous", "ask_missing_budget"}:
        return PlannerAction.ASK, {"question": "Which product type, trip style, and budget should I use?"}, "No candidates and a missing decision attribute; ask before retrieval."
    if scenario in {"rerank_soft_weight", "rerank_soft_feature"}:
        return PlannerAction.RERANK, {}, "Candidates exist, have not been reranked, and current catalogue evidence supports the soft preference; consult M6.1 before finalizing."
    if scenario == "finalize_after_rerank":
        return PlannerAction.FINALIZE, {}, "Candidates already passed M6.1 reranking and are validated; finalize without a new retrieval."
    return PlannerAction.FINALIZE, {}, "Soft preference has no supported catalogue evidence; M6.1 cannot consume it, so finalize the validated candidate set."


def _sample(split: str, scenario: str, index: int, pools: dict[str, list[Any]]) -> dict[str, Any]:
    keyword = KEYWORDS[index % len(KEYWORDS)]
    budget = _budget_for(keyword, scenario, index, pools)
    keyword, records = _pool_for(pools, scenario, index, budget)
    evidence = [_candidate_evidence(record) for record in records]
    observation = _observation(scenario, keyword, index, budget, evidence)
    action, arguments, rationale = _target(scenario, observation)
    allowed = allowed_planner_actions(observation)
    observation = replace(observation, allowed_actions=allowed)
    proposal = ActionProposal(action, arguments, rationale)
    validate_action(proposal, observation, step=1, max_steps=4)
    if action.value not in allowed:
        raise RuntimeError(f"target {action} not allowed for {scenario}: {allowed}")
    sample_id = f"planner-sft-v0-{scenario}-{index:03d}"
    request = {
        "contract_version": CONTRACT_VERSION,
        "step": observation.step,
        "query": observation.query,
        "candidate_ids": list(observation.candidate_ids),
        "candidate_count": len(observation.candidate_ids),
        "candidate_evidence": list(observation.candidate_evidence),
        "state_version": observation.state_version,
        "m5_state": dict(observation.m5_state),
        "allowed_actions": list(observation.allowed_actions),
        "last_tool_observation": dict(observation.last_tool_observation),
        "remaining_steps": observation.remaining_steps,
        "needs_user_input": observation.needs_user_input,
        "reranked": observation.reranked,
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "sample_id": sample_id,
        "split": split,
        "source": {
            "request": "synthetic_shopping_scenario",
            "catalogue": "amazon_reviews_2023:meta_categories",
            "label_source": "rule_and_tool_semantics",
            "qwen_output_used": False,
            "task_family": scenario,
        },
        "group_id": f"{scenario}:{keyword}",
        "trace_id": f"synthetic-{scenario}-{index:03d}",
        "user_request": observation.query,
        "observation": request,
        "target": {
            "action": action.value,
            "arguments": arguments,
            "rationale": rationale,
        },
        "label_basis": {
            "candidates_present": bool(observation.candidate_ids),
            "reranked": observation.reranked,
            "needs_user_input": observation.needs_user_input,
            "m5_state_version": observation.state_version,
            "catalogue_evidence_fields": sorted({key for item in observation.candidate_evidence for key in ("price", "weight", "color", "material") if item.get(key)}),
        },
    }


def _chat_row(sample: dict[str, Any]) -> dict[str, Any]:
    request = sample["observation"]
    messages = planner_messages(request)
    target = sample["target"]
    messages.append({
        "role": "assistant",
        "content": json.dumps({"action": target["action"], "arguments": target["arguments"], "rationale": target["rationale"]}, ensure_ascii=False, separators=(",", ":")),
    })
    return {
        "schema_version": "agentrecx.planner_sft.chat.v0",
        "sample_id": sample["sample_id"],
        "split": sample["split"],
        "messages": messages,
        "target_action": target["action"],
        "label_source": sample["source"]["label_source"],
    }


def build() -> dict[str, Any]:
    pools = _load_pools()
    OUT.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for scenario, count in SCENARIOS:
        scenario_offset = int(hashlib.sha256(scenario.encode()).hexdigest(), 16)
        for index in range(count):
            group = f"{scenario}:{KEYWORDS[index % len(KEYWORDS)]}"
            # Split by group, but rotate the held-out keyword per scenario so every
            # action family is represented in validation (no group crosses splits).
            keyword_index = index % len(KEYWORDS)
            split = "val" if (keyword_index + scenario_offset) % len(KEYWORDS) == 0 else "train"
            rows.append(_sample(split, scenario, index, pools))
    for split in ("train", "val"):
        part = [row for row in rows if row["split"] == split]
        (OUT / f"{split}.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in part), encoding="utf-8")
        chat = [_chat_row(row) for row in part]
        (OUT / f"chat_{split}.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in chat), encoding="utf-8")
    action_counts: dict[str, int] = {}
    scenario_counts: dict[str, int] = {}
    for row in rows:
        action_counts[row["target"]["action"]] = action_counts.get(row["target"]["action"], 0) + 1
        scenario_counts[row["source"]["task_family"]] = scenario_counts.get(row["source"]["task_family"], 0) + 1
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "contract_version": CONTRACT_VERSION,
        "generated_by": "training/planner_sft/build_v0.py",
        "samples": len(rows),
        "splits": {split: sum(row["split"] == split for row in rows) for split in ("train", "val")},
        "actions": action_counts,
        "scenarios": scenario_counts,
        "sources": {
            "user_requests": "synthetic_shopping_scenario",
            "catalogue": "amazon_reviews_2023:meta_categories",
            "labels": "rule_and_tool_semantics; no Qwen output",
            "future_observation_or_gold_used": False,
        },
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    schema = {
        "schema_version": "agentrecx.planner_sft.schema.v0",
        "required": ["sample_id", "split", "source", "user_request", "observation", "target", "label_basis"],
        "observation_required": ["query", "candidate_ids", "candidate_evidence", "m5_state", "allowed_actions", "last_tool_observation", "remaining_steps"],
        "target_action_enum": [action.value for action in PlannerAction],
        "no_future_fields": ["final_ids", "future_observation", "gold", "next_observation", "eval_label"],
    }
    (OUT / "schema.json").write_text(json.dumps(schema, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest


if __name__ == "__main__":
    manifest = build()
    print(json.dumps(manifest, ensure_ascii=False, indent=2))

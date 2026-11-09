"""Independent Qwen Planner prompt.

This prompt is deliberately separate from the Preference Operation prompt.  The model
returns one Planner action object; it does not return ``operations`` and never executes a
tool or writes M5.
"""

from __future__ import annotations

import json
from typing import Any, Mapping

PLANNER_SYSTEM_PROMPT = """You are the planning layer of AgentRec-X, a shopping recommendation agent.
Choose exactly one next action from the allowed_actions list:
- ASK: ask the user one clarification question when the request is ambiguous.
- RETRIEVE: ask the existing recommendation tool for candidates.
- RERANK: ask the existing M6.1 reranker to reorder the current candidates.
- FINALIZE: return the current validated candidate set.

Return ONLY one compact JSON object with exactly:
{"action":"RETRIEVE","arguments":{"query":"..."},"rationale":"..."}
Action-specific arguments:
- ASK: {"question":"..."}
- RETRIEVE: {"query":"...","source_policy":"balanced|intent_focus|history_focus"}
  source_policy is optional; omit it for balanced. intent_focus gives catalog_search
  twice the rank-fusion weight; history_focus gives history twice the weight.
- RERANK: {}
- FINALIZE: {}
Do not return Preference Operations, operations, ASINs, URLs, code fences, or prose.
Do not invent candidates or claim a tool was run. The trusted executor validates the
action, role, price constraints and catalogue provenance. If no allowed action is useful,
return a final action only when the observation has candidates; otherwise choose ASK.
"""


def planner_messages(request: Mapping[str, Any]) -> list[dict[str, str]]:
    observation = {
        "contract_version": request.get("contract_version"),
        "step": request.get("step"),
        "query": request.get("query"),
        "candidate_ids": list(request.get("candidate_ids") or ()),
        "candidate_count": request.get("candidate_count"),
        "candidate_evidence": list(request.get("candidate_evidence") or ()),
        "last_retrieve_query": request.get("last_retrieve_query"),
        "retrieval_feedback": request.get("retrieval_feedback") or {},
        "role_target": request.get("role_target") or {},
        "eligibility": request.get("eligibility") or {},
        "source_policy": request.get("source_policy") or "balanced",
        "source_feedback": request.get("source_feedback") or {},
        "state_version": request.get("state_version"),
        "m5_state": request.get("m5_state") or {},
        "allowed_actions": list(request.get("allowed_actions") or ()),
        "last_tool_observation": request.get("last_tool_observation") or {},
        "remaining_steps": request.get("remaining_steps"),
        "needs_user_input": request.get("needs_user_input"),
        "reranked": request.get("reranked"),
    }
    return [
        {"role": "system", "content": PLANNER_SYSTEM_PROMPT},
        {"role": "user", "content": "Current planner observation:\n" + json.dumps(observation, ensure_ascii=False, sort_keys=True)},
    ]


def planner_prompt(tokenizer: Any, request: Mapping[str, Any]) -> str:
    return str(
        tokenizer.apply_chat_template(
            planner_messages(request),
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    )

"""Deterministic task-slot clarification policy for the CPU rule planner.

The P16 preference-clarification skill asks a generic question derived from M9.  A laptop
purchase, however, has two task-scoped slots that materially change the catalogue query and
the budget Gate: the use case (office/gaming) and the budget.  This module adds a *small*,
deterministic policy on top of the existing skill:

* the policy only applies when the task category is explicitly ``laptop``;
* it asks for the use case first, then the budget;
* it never invents a value and never writes M9; the normal reply path still merges the real
  user reply through ``TaskRequirements``;
* every other category keeps the existing P16 question, so the M3 Golden Dataset and the
  single-round headphone/tablet flows are unchanged.

A returned :class:`ClarificationNeed` is a request for one answerable slot.  The MCP server
turns it into a real ``ClarificationState`` and continuation; nothing here is a fake ASK.
"""

from __future__ import annotations

from dataclasses import dataclass

from recommendation.preference.task_requirements import TaskRequirements

__all__ = ["ClarificationNeed", "next_laptop_clarification"]


@dataclass(frozen=True)
class ClarificationNeed:
    """One missing, answerable task slot."""

    slot: str
    question: str
    reason: str


def _has_cjk(text: str) -> bool:
    return any("\u4e00" <= char <= "\u9fff" for char in str(text or ""))


def next_laptop_clarification(
    requirements: TaskRequirements, query: str
) -> ClarificationNeed | None:
    """Return the next missing laptop slot, or ``None`` when the task is ready.

    Only an explicitly extracted ``laptop`` category is handled.  The order is intentional:
    the use case shapes retrieval before a budget can filter candidates, and it matches the
    M1/M4 demo conversation ("办公还是游戏" then "预算").
    """
    if requirements.category != "laptop":
        return None
    chinese = _has_cjk(query)
    if not requirements.use_cases:
        return ClarificationNeed(
            slot="use_case",
            question=(
                "这台笔记本主要用来办公还是游戏？"
                if chinese
                else "Will this laptop be mainly for office work or gaming?"
            ),
            reason="missing_laptop_use_case",
        )
    if requirements.budget is None:
        return ClarificationNeed(
            slot="budget",
            question=(
                "您的预算是多少？"
                if chinese
                else "What is your budget for this laptop?"
            ),
            reason="missing_laptop_budget",
        )
    return None

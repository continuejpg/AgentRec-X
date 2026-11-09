"""P16 registered shopping skills."""
from __future__ import annotations

from recommendation.skills.clarification import (
    ask_clarification,
    read_authorized_history,
    resume_with_user_reply,
)
from recommendation.skills.registry import SkillRegistry, SkillSpec
from recommendation.skills.shopping import shopping_comparison


def build_registry() -> SkillRegistry:
    registry = SkillRegistry()
    registry.register(SkillSpec(
        "shopping-comparison",
        "Compare 2-5 explicit real Electronics products using catalogue Product Info and P15 "
        "Evidence Bundles, including UNKNOWN fields and active preferences; no objective quality claim.",
        shopping_comparison,
    ))
    registry.register(SkillSpec(
        "preference-clarification",
        "Ask one answerable clarification question when the Gate needs it, then persist a real user "
        "reply/retraction through the existing M9 memory writer and re-read state.",
        _clarification_dispatch,
    ))
    return registry


def _clarification_dispatch(**kwargs):
    operation = kwargs.pop("operation", "ask")
    if operation == "ask":
        return ask_clarification(**kwargs)
    if operation == "resume":
        return resume_with_user_reply(**kwargs)
    if operation == "read":
        return read_authorized_history(**kwargs)
    raise ValueError(f"unknown clarification operation {operation!r}")


__all__ = ["build_registry", "shopping_comparison", "ask_clarification", "resume_with_user_reply",
           "read_authorized_history"]

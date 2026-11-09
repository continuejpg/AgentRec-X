"""Declared comparison capabilities for the four benchmark policy arms.

The benchmark does not simulate an unavailable policy.  This manifest is the audit
contract used by the fairness report/tests: it makes every intended difference an
explicit experimental variable instead of an accidental adapter difference.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["PolicyCapability", "POLICY_CAPABILITIES", "fairness_issues"]


@dataclass(frozen=True)
class PolicyCapability:
    policy: str
    execution_entry: str
    stateful_shopping: bool
    requires_llm: bool
    memory_scope: str
    planner: str
    role_eligibility: str
    m5_soft_rerank: str
    same_candidate_service: bool = True
    same_catalogue: bool = True
    same_model_sources: bool = True


POLICY_CAPABILITIES: tuple[PolicyCapability, ...] = (
    PolicyCapability(
        policy="A",
        execution_entry="service.run (stateless request)",
        stateful_shopping=False,
        requires_llm=True,
        memory_scope="request history only",
        planner="LLM query rewrite",
        role_eligibility="adapter-level promote_explicit_type_matches only",
        m5_soft_rerank="unavailable (no task M5 state)",
    ),
    PolicyCapability(
        policy="B",
        execution_entry="service.run (stateless request)",
        stateful_shopping=False,
        requires_llm=True,
        memory_scope="request history only",
        planner="LLM query rewrite + full history context",
        role_eligibility="adapter-level promote_explicit_type_matches only",
        m5_soft_rerank="unavailable (no task M5 state)",
    ),
    PolicyCapability(
        policy="C",
        execution_entry="run_shopping_turn (session/task state)",
        stateful_shopping=True,
        requires_llm=False,
        memory_scope="legacy M9 + M5 state",
        planner="deterministic R1",
        role_eligibility="task-scoped grounded role filter",
        m5_soft_rerank="available in explicit soft_on arm",
    ),
    PolicyCapability(
        policy="D",
        execution_entry="run_shopping_turn (session/task state)",
        stateful_shopping=True,
        requires_llm=True,
        memory_scope="legacy M9 + M5 state",
        planner="LLM preference proposal",
        role_eligibility="task-scoped grounded role filter",
        m5_soft_rerank="available in explicit soft_on arm",
    ),
)


def fairness_issues() -> tuple[str, ...]:
    issues: list[str] = []
    eligibility = {item.policy: item.role_eligibility for item in POLICY_CAPABILITIES}
    if len(set(eligibility.values())) > 1:
        issues.append(
            "A/B do not share C/D's task-scoped role eligibility; a future LLM run "
            "must route both through one shared postprocessor before comparison"
        )
    soft = {item.policy: item.m5_soft_rerank for item in POLICY_CAPABILITIES}
    if len(set(soft.values())) > 1:
        issues.append(
            "M5 soft rerank is task-stateful and therefore available only to C/D; "
            "it is an explicit experimental variable, not parity"
        )
    return tuple(issues)

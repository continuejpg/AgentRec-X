"""E-04B frozen dual-layer evaluation protocol (declarative config, no weights, no network).

This module is the machine-readable half of ``docs/V2_E04B_DUAL_EVALUATION_PROTOCOL.md``: it names the
two layers, the four controlled scenarios, the five comparison arms, the metric registry with the
labels each metric needs, and the leak/budget rules.  Nothing here loads a model, calls a provider or
touches the serving path; the accompanying test module asserts the invariants that keep a later run
honest (same case ids, no target-derived queries, labels before metrics, no budget relabelling).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

#: Frozen identities.  A later run must reproduce these strings in its report.
PROTOCOL_NAME = "agentrecx.eval_protocol.v1"
SPLIT_NAME = "temporal_leave_two_out"
PROTOCOL_FREEZE_SEED = 20261015

LAYER_A = "A.recommendation_quality"
LAYER_B = "B.agentic_recommendation_quality"

#: Where each scenario's input may come from.  ``labels`` says what exists *today*.
SCENARIOS: Mapping[str, Mapping[str, Any]] = {
    "A.history_only": {
        "layer": LAYER_A,
        "inputs": ("train_history", "validation_target", "test_history"),
        "request_text": None,
        "labels": "real: the held-out test target is the single positive",
        "relevance_labels": True,
        "note": "the classic next-item protocol; no natural-language request exists",
    },
    "B.text_only": {
        "layer": LAYER_B,
        "inputs": ("request_text",),
        "request_text": "human-authored request",
        "labels": "MISSING: no real user requests with relevance labels in this repository",
        "relevance_labels": False,
        "note": "DATA GAP - see ANNOTATION_MINIMUM; must not be synthesised from the target",
    },
    "C.history_plus_text": {
        "layer": LAYER_B,
        "inputs": ("test_history", "request_text"),
        "request_text": "human-authored request",
        "labels": "MISSING: same gap as B, plus a preference projection source",
        "relevance_labels": False,
        "note": "DATA GAP - both the request and its relevance labels must be annotated",
    },
    "D.multiturn_feedback": {
        "layer": LAYER_B,
        "inputs": ("test_history", "request_text", "presented_cards", "user_feedback"),
        "request_text": "human-authored multi-turn requests",
        "labels": "PARTIAL: behaviour is checkable from logs; relevance labels are MISSING",
        "relevance_labels": False,
        "note": "feedback identity binding is implemented (E-03C) but no human relevance labels exist",
    },
}

#: Comparison arms.  ``entry`` names the real code path; ``status`` is the honest implementation state.
ARMS: Mapping[str, Mapping[str, Any]] = {
    "B0.single_source": {
        "capabilities": ("one recommendation source",),
        "sources": ("sasrec", "two_tower", "tiger", "similar_item", "bm25_exploratory"),
        "entry": "recommendation/inference/sasrec.py:232, recommendation/inference/twotower.py:70, "
                 "recommendation/backends/tiger_engine.py:90",
        "budget": {"tool_calls": 1, "llm_calls": 0},
        "status": "EXECUTABLE_OFFLINE",
        "selection_rule": "FROZEN before scoring: one source per arm instance; the arm is reported "
                          "per source, never as a per-case best-of",
    },
    "B1.fixed_multi_source_rrf": {
        "capabilities": ("fixed source set", "reciprocal rank fusion"),
        "sources": ("sasrec", "two_tower", "tiger", "similar_item"),
        "entry": "recommendation/inference/fusion.py:226, experiments/benchmark_twotower.py:92, "
                 "experiments/benchmark_genrec.py:104",
        "budget": {"tool_calls": "len(source_set)", "llm_calls": 0},
        "status": "EXECUTABLE_OFFLINE",
        "selection_rule": "FROZEN: the source set is declared before scoring and never adapted per case",
    },
    "B2.rule_tool_selection": {
        "capabilities": ("deterministic policy", "constraint filtering", "empty-result recovery"),
        "sources": ("history", "catalog_search", "similar_item"),
        "entry": "recommendation/evaluation/agent/runner.py:216-513 (policy library), "
                 "recommendation/demo/agent_policy.py (plan policy)",
        "budget": {"tool_calls": "<= 2", "llm_calls": 0},
        "status": "EXECUTABLE_OFFLINE",
        "selection_rule": "FROZEN: policy per scenario, fixed in cases.py; no per-case tuning",
    },
    "B3.llm_tool_selection": {
        "capabilities": ("model-chosen source plan",),
        "sources": ("history", "catalog_search", "similar_item", "two_tower", "tiger"),
        "entry": "recommendation/demo/tool_selection.py:214",
        "budget": {"tool_calls": "<= 2", "llm_calls": 1},
        "status": "BLOCKED_NO_PROVIDER_AUTHORIZATION",
        "selection_rule": "model output validated against the deployment's source menu",
    },
    "B4.full_agent": {
        "capabilities": ("memory-aware ranking", "bounded replanning", "product RAG", "feedback"),
        "sources": ("history", "catalog_search", "similar_item"),
        "entry": "recommendation/demo/runtime.py (llm_agent_turn), recommendation/demo/observation.py, "
                 "recommendation/demo/replan_policy.py, recommendation/demo/explanation.py",
        "budget": {"tool_calls": "<= 2", "llm_calls": "<= 3"},
        "status": "PARTIAL_REPLAY_ONLY",
        "selection_rule": "the accepted planner/gate rules; replay of runs/m3_execution/*.jsonl is "
                          "evidence of a past run, never a fresh measurement",
    },
}

#: ``source`` is how the value is obtained: "log" (execution log decides) or "human" (needs labels).
METRICS: Mapping[str, Mapping[str, Any]] = {
    "hr_at_k": {"layer": LAYER_A, "source": "log", "labels": ("test_target",),
                "definition": "1[rank <= k] / cases", "code": "recommendation/evaluation/metrics.py:311"},
    "ndcg_at_k": {"layer": LAYER_A, "source": "log", "labels": ("test_target",),
                  "definition": "1/log2(rank+1) within k / cases",
                  "code": "recommendation/evaluation/metrics.py:317"},
    "candidate_coverage": {"layer": LAYER_A, "source": "log", "labels": (),
                           "definition": "distinct returned ids / catalogue size",
                           "code": "recommendation/evaluation/evaluator.py"},
    "tool_call_count": {"layer": LAYER_B, "source": "log", "labels": (),
                        "definition": "real tool calls per case",
                        "code": "recommendation/evaluation/agent/metrics.py:64"},
    "unnecessary_tool_calls": {"layer": LAYER_B, "source": "log", "labels": (),
                               "definition": "calls beyond the case's declared minimum",
                               "code": "recommendation/evaluation/agent/metrics.py:64"},
    "legal_tool_selection_rate": {"layer": LAYER_B, "source": "log", "labels": (),
                                  "definition": "validated proposals / proposals",
                                  "code": "recommendation/evaluation/agent/metrics.py:160"},
    "constraint_violation_rate": {"layer": LAYER_B, "source": "log", "labels": (),
                                  "definition": "cases with a verified violation presented / cases",
                                  "code": "recommendation/evaluation/agent/metrics.py:79-100"},
    "constraint_enforcement_rate": {"layer": LAYER_B, "source": "log", "labels": (),
                                    "definition": "cases whose presented set was actually checked",
                                    "code": "recommendation/evaluation/agent/metrics.py:89-100"},
    "preference_follow_rate": {"layer": LAYER_B, "source": "human",
                               "labels": ("annotated request-to-preference mapping",),
                               "definition": "NOT COMPUTABLE TODAY - needs annotation",
                               "code": "recommendation/demo/memory_ranking.py"},
    "task_success_rate": {"layer": LAYER_B, "source": "human",
                          "labels": ("annotated per-case success criterion",),
                          "definition": "PARTIAL: constraint cases are decidable from logs; "
                                        "open-ended requests are not",
                          "code": "recommendation/evaluation/agent/cases.py:46"},
    "rag_claim_validity": {"layer": LAYER_B, "source": "log", "labels": (),
                           "definition": "validated claims / proposed claims",
                           "code": "recommendation/demo/explanation.py:228"},
    "rag_abstention_rate": {"layer": LAYER_B, "source": "log", "labels": (),
                            "definition": "cards with insufficient_evidence / cards",
                            "code": "recommendation/demo/explanation.py:228"},
    "feedback_identity_accuracy": {"layer": LAYER_B, "source": "log", "labels": (),
                                   "definition": "resolved references matching the intended card",
                                   "code": "recommendation/demo/feedback.py:253"},
}

#: The only fields a query may be derived from.  ``test_target`` is deliberately absent.
QUERY_INPUT_FIELDS: tuple[str, ...] = ("test_history", "request_text")

#: Rules a later run must satisfy; the test module asserts them against this config.
LEAK_RULES: tuple[str, ...] = (
    "the request never carries test_target or validation_target",
    "a query is never derived from the hidden target's title, description or category",
    "memory entries are built only from user-authored text and presented cards",
    "a task whose labels are missing cannot produce a relevance metric",
    "replay recordings are past evidence and are labelled as such",
    "results from different tool budgets are never reported as a same-budget comparison",
)


@dataclass(frozen=True)
class ProtocolViolation(ValueError):
    """A requested comparison that the frozen protocol does not allow."""

    reason: str

    def __str__(self) -> str:  # pragma: no cover - display only
        return self.reason


def assert_same_case_ids(case_ids_by_arm: Mapping[str, Sequence[str]]) -> tuple[str, ...]:
    """Every arm compared in one table must score exactly the same case ids."""
    if not case_ids_by_arm:
        raise ProtocolViolation("no arms given")
    sets = {arm: tuple(ids) for arm, ids in case_ids_by_arm.items()}
    reference = next(iter(sets.values()))
    for arm, ids in sets.items():
        if ids != reference:
            raise ProtocolViolation(f"arm {arm!r} scores different case ids than the first arm")
    return reference


def assert_same_budget(budgets: Mapping[str, Mapping[str, Any]], *, label: str) -> None:
    """A comparison may only be called same-budget when every arm declares the same budget."""
    if not budgets:
        raise ProtocolViolation("no budgets given")
    distinct = {tuple(sorted((k, str(v)) for k, v in budget.items())) for budget in budgets.values()}
    if len(distinct) > 1:
        raise ProtocolViolation(
            f"{label!r} is not a same-budget comparison: {len(distinct)} distinct budgets declared"
        )


def metric_is_computable(name: str, available_labels: Sequence[str]) -> bool:
    """A metric runs only when every label it needs is actually available."""
    spec = METRICS.get(name)
    if spec is None:
        raise ProtocolViolation(f"unknown metric {name!r}")
    return set(spec["labels"]) <= set(available_labels)


def assert_query_is_label_free(fields: Sequence[str]) -> None:
    """The query builder may only read fields that are visible when the request is made."""
    forbidden = {"test_target", "validation_target", "required_frontier"}
    overlap = forbidden.intersection(fields)
    if overlap:
        raise ProtocolViolation(f"query inputs contain hidden label fields: {sorted(overlap)}")


def arm_status(arm: str) -> str:
    spec = ARMS.get(arm)
    if spec is None:
        raise ProtocolViolation(f"unknown arm {arm!r}")
    return str(spec["status"])


#: Minimum annotation work needed before Layer B relevance metrics can exist at all.
ANNOTATION_MINIMUM: Mapping[str, Any] = {
    "unit": "one (request, user, judged relevant item ids) triple",
    "scenarios": ("B.text_only", "C.history_plus_text", "D.multiturn_feedback"),
    "size_for_exploratory": 100,
    "size_for_paired_claims": 500,
    "judges": 2,
    "agreement": "report Cohen's kappa on the relevance judgements",
    "provenance": "human-authored requests, written without seeing the held-out target",
    "forbidden": "generating a request from the target's title/description/category",
}

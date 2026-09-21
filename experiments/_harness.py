"""Shared composition of the evaluation control plane (Phase 3).

Every evaluation command - the scripted smoke, the live smoke and the live case-suite runner -
must build the **same** plane, or an ablation across them would compare harnesses instead of
policies.  This module is that one composition, extracted from
``experiments/agent_evaluation_smoke.py`` so the live command cannot drift from it.

What the plane is:

* the **real** accepted pipeline: the Recommendation Tool over a duck-typed engine, the real M8
  enricher, the real M10A matcher, the real M10B reranker;
* the real control plane: `ActionValidator`, `ResultVerifier`, `CompletionGuard`,
  `LoopController`;
* the real Phase-2 constraint kernel (`CandidateEligibilityEvaluator`) over the same catalogue;
* the real Phase-2.1 reference boundary (the same reasoning executor, so run-scoped identity
  enforcement applies identically to a scripted and a real model).

Nothing here is model-aware.  The only thing a caller varies is the *policy*, which is the
ablation.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from recommendation.control import (  # noqa: E402
    CandidateEligibilityEvaluator,
    CandidateLedger,
    CandidatePlane,
    CatalogSearchSource,
    EvidenceExecutor,
    EvidenceStore,
    GroundedReasoner,
    GroundingVerifier,
    LocalEvidenceArtifact,
    ReasoningExecutor,
    SimilarItemSource,
    build_similar_item_index,
)
from recommendation.evaluation.agent import EvaluationCase  # noqa: E402
from tests.agent_reranking_fixture import CANDIDATE_ROWS  # noqa: E402
from tests.control_fixture import build_control_harness  # noqa: E402

__all__ = [
    "EVIDENCE_ARTIFACT",
    "evidence_store_for",
    "phase4_reasoner",
    "IdentityMap",
    "catalog_reasoner",
    "case_controller_factory",
    "evidence_enabled",
    "run_case_suite",
]

#: The repository fixture Phase 4 evidence is read from.  Deliberately **not** the catalogue
#: projection: a fact in this file is unknown to a run until an evidence action reads it, which is
#: what makes the ``UNKNOWN -> acquire -> SATISFIED`` trajectory a real capability test.
EVIDENCE_ARTIFACT = REPO_ROOT / "tests" / "fixtures" / "evidence" / "product_evidence.jsonl"


class IdentityMap:
    """The trusted identity map the fixture catalogue is keyed by."""

    def __init__(self, rows: Any = CANDIDATE_ROWS) -> None:
        self._ids = {row[0]: row[1] for row in rows}

    def has_parent_asin(self, parent_asin: str) -> bool:
        return parent_asin in self._ids

    def parent_asin_to_item_id(self, parent_asin: str) -> int:
        return self._ids[parent_asin]

    def item_id_to_parent_asin(self, item_id: int) -> str:
        for identity, value in self._ids.items():
            if value == item_id:
                return identity
        raise KeyError(item_id)


def catalog_reasoner() -> GroundedReasoner:
    """A reasoner over the **whole** suite catalogue.

    Built from a harness given ``catalog_rows``, because several cases drive the engine with
    nothing: keying the reasoner to the engine's rows would leave it over an empty catalogue for
    exactly the cases that need to read facts.
    """
    return GroundedReasoner(
        build_control_harness(catalog_rows=CANDIDATE_ROWS).parts["enricher"].metadata
    )


def evidence_enabled(case: EvaluationCase) -> bool:
    """True when a case should run with Phase 4's evidence and similar-item sources wired.

    Opt-in **per case**, so a case written before Phase 4 keeps the exact runtime it was written
    for and an ablation can compare Phase-3 and Phase-4 behaviour on the same suite.
    """
    return "evidence" in case.category or "similar" in case.category


#: Per-case evidence stores, keyed by case id.  The controller factory and the reasoner factory
#: must hand the *same* store to the run and to the constraint measurement: an evidence case
#: acquires facts during its turn, and a grader whose reasoner held a different store would not
#: see them, so it would report a constraint the run just resolved as unresolved.  Keyed by case
#: because each case is an independent run, and two cases must never share acquired facts.
_EVIDENCE_STORES: dict[str, EvidenceStore] = {}


def evidence_store_for(case: EvaluationCase) -> EvidenceStore | None:
    """The evidence store for one case, creating it on first request.

    Returns ``None`` for a case that does not opt into Phase 4, so nothing is created for the
    cases that must keep the pre-Phase-4 runtime.
    """
    if not evidence_enabled(case):
        return None
    store = _EVIDENCE_STORES.get(case.case_id)
    if store is None:
        store = EvidenceStore()
        _EVIDENCE_STORES[case.case_id] = store
    return store


def phase4_reasoner(case: EvaluationCase) -> GroundedReasoner:
    """A reasoner over the case's evidence store.

    Used as the runner's ``reasoner_factory``: the runner builds it, hands it to the controller
    factory, and grades with it, so both sides see one object and one overlay.
    """
    return GroundedReasoner(
        build_control_harness(catalog_rows=CANDIDATE_ROWS).parts["enricher"].metadata,
        evidence=evidence_store_for(case),
    )


def case_controller_factory(
    case: EvaluationCase,
    policy: Any,
    limits: Any,
    task_state: Any = None,
    reasoner: Any = None,
) -> Any:
    """Compose the control plane for one case, with the real trusted components wired in.

    ``task_state`` is supplied by the runner and carries the case's declared hard constraints, so
    a case that asserts enforcement runs against a constraint that is genuinely active.
    """
    empty_history = "recovery-required" in case.case_id
    harness = build_control_harness(
        policy=policy,
        limits=limits,
        rows=() if empty_history else CANDIDATE_ROWS,
        catalog_rows=CANDIDATE_ROWS,
        engine_error=(
            RuntimeError("history source unavailable") if case.history_source_fails else None
        ),
    )
    metadata = harness.parts["enricher"].metadata
    # Phase 4: the store is the case-scoped one, so the reasoner the runner grades with and the
    # reasoner driving the run consult a single overlay.  ``reasoner`` is normally handed in by
    # the runner (which built it from ``phase4_reasoner``); the fallback keeps a direct caller -
    # a test that builds the controller itself - working.
    store = evidence_store_for(case)
    if reasoner is None:
        reasoner = GroundedReasoner(metadata, evidence=store)
    harness.controller._reasoning = ReasoningExecutor(reasoner)  # noqa: SLF001 - deliberate wiring
    # Phase 2: the same reasoner drives task-scoped constraint eligibility, so a case that
    # declares a hard constraint is run *and measured* against one enforced constraint set.
    harness.controller._eligibility = CandidateEligibilityEvaluator(reasoner)  # noqa: SLF001
    harness.controller._candidate_plane = CandidatePlane(  # noqa: SLF001
        ledger=CandidateLedger(),
        grounding=GroundingVerifier(IdentityMap(), metadata),
        catalog_search=CatalogSearchSource(metadata),
        # Phase 4: a real lexical item-item index over the same trusted catalogue.  Registered
        # only for the cases that opt in, so ``FIND_SIMILAR`` is offered exactly where a case
        # exists to exercise it.
        similar_item_tool=(
            SimilarItemSource(build_similar_item_index(metadata))
            if evidence_enabled(case)
            else None
        ),
    )
    if store is not None:
        harness.controller._evidence = EvidenceExecutor(  # noqa: SLF001
            [LocalEvidenceArtifact(EVIDENCE_ARTIFACT)], store
        )
    return harness.controller


def run_case_suite(
    *,
    variant: str,
    policy_factory: Any,
    reasoner_factory: Any = catalog_reasoner,
    on_case: Any = None,
) -> Any:
    """Run the whole case set under one policy and return the suite report.

    ``on_case`` is an optional ``(case, outcome) -> None`` callback, so a command can stream
    progress and collect per-case observations without this helper knowing what an experiment
    report looks like.
    """
    from recommendation.evaluation.agent import CaseRunner, load_cases

    runner = CaseRunner(case_controller_factory, variant=variant)
    outcomes = []
    for case in load_cases():
        outcome = runner.run(
            case, policy=policy_factory(case), reasoner=reasoner_factory()
        )
        outcomes.append(outcome)
        if on_case is not None:
            on_case(case, outcome)

    from recommendation.evaluation.agent.runner import SuiteReport

    return SuiteReport(variant, outcomes)

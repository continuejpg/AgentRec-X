"""Offline fixtures for the AgentRec-X 2.0-alpha control-plane tests.

Everything here is deterministic and dependency-free: no checkpoint, no GPU, no network,
no LLM provider SDK and no database.  The collaborators are the **real** accepted
components wherever the behaviour under test is acceptance-relevant:

* candidates come from the accepted Recommendation Tool over a duck-typed engine;
* enrichment runs the real M8 ``ProductEnricher`` over a real ``MetadataIndex``;
* evidence comes from the real M10A ``PreferenceCandidateMatcher``;
* order comes from the real M10B ``PreferenceReranker``;
* memory is the real M9 ``PreferenceMemoryService`` over an in-memory store.

The control plane is driven with these, so the tests assert the real trust boundaries
rather than a mock of them.  The only purpose-built doubles are the *policies*: they exist
because the point of Stage 1 is that the policy is interchangeable.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from recommendation.control import (  # noqa: E402
    ActionKind,
    ActionProposal,
    LoopController,
    LoopLimits,
    PolicyActionError,
    PolicyContext,
    RecommendFromHistoryCapability,
    RuleBasedPolicy,
)
from recommendation.rag import ProductEnricher  # noqa: E402
from recommendation.reranking import PreferenceReranker  # noqa: E402
from recommendation.preference_matching import PreferenceCandidateMatcher  # noqa: E402
from recommendation.tools import RecommendationTool  # noqa: E402
from tests.agent_reranking_fixture import (  # noqa: E402
    CANDIDATE_ROWS,
    QUERY,
    FixedEngine,
    build_index,
    make_service,
)
from tests.agent_fakes import HISTORY  # noqa: E402

__all__ = [
    "CANDIDATE_ROWS",
    "HISTORY",
    "QUERY",
    "ControlHarness",
    "FullCapability",
    "MinimalCapability",
    "RecordingPolicy",
    "ScriptedPolicy",
    "build_control_harness",
    "build_full_capability",
    "build_minimal_capability",
    "engine_identity_set",
]


class RecordingPolicy:
    """A policy that records every context it is handed and delegates one decision rule.

    Used to prove what the policy *sees*: the recorded contexts are the actual objects the
    loop constructed, so a test can assert that trusted history, candidate ids, scores and
    memory contents are simply not reachable from them.
    """

    def __init__(self, delegate: Any | None = None, *, k: int = 3) -> None:
        self._delegate = delegate or RuleBasedPolicy(default_k=k)
        self.contexts: list[PolicyContext] = []

    @property
    def name(self) -> str:
        """Stable identity for trajectory assertions."""
        return "recording"

    @property
    def call_count(self) -> int:
        """How many times the loop asked this policy for an action."""
        return len(self.contexts)

    @property
    def last_context(self) -> PolicyContext | None:
        """The most recent context, or ``None`` before the first call."""
        return self.contexts[-1] if self.contexts else None

    def choose(self, context: PolicyContext) -> ActionProposal:
        """Record the context and delegate."""
        self.contexts.append(context)
        return self._delegate.choose(context)


class ScriptedPolicy:
    """A policy that returns a fixed script of proposals, then repeats the last one.

    Lets a test drive the loop down a path the rule-based policy would never take, which is
    how the protocol guards get exercised (illegal action, illegal k, premature FINISH).
    """

    def __init__(self, script: list[ActionProposal | str | None]) -> None:
        if not script:
            raise ValueError("script must not be empty")
        self._script = list(script)
        self.contexts: list[PolicyContext] = []

    @property
    def name(self) -> str:
        """Stable identity for trajectory assertions."""
        return "scripted"

    @property
    def call_count(self) -> int:
        """How many times the loop asked this policy for an action."""
        return len(self.contexts)

    def choose(self, context: PolicyContext) -> ActionProposal:
        """Return the next scripted proposal (or raise / return junk, if scripted so)."""
        self.contexts.append(context)
        index = min(len(self.contexts) - 1, len(self._script) - 1)
        entry = self._script[index]
        if entry is None:
            raise PolicyActionError("scripted policy has no available action")
        if isinstance(entry, str):
            raise RuntimeError(f"scripted policy failure: {entry}")
        return entry


def build_full_capability(
    *,
    rows: Any = CANDIDATE_ROWS,
    engine_error: Exception | None = None,
    with_enricher: bool = True,
    with_preferences: bool = True,
) -> tuple[RecommendFromHistoryCapability, FixedEngine, dict[str, Any]]:
    """Build the capability over the real accepted pipeline, with optional stages off."""
    engine = FixedEngine(rows, error=engine_error)
    tool = RecommendationTool(engine)
    enricher = ProductEnricher(build_index()) if with_enricher else None
    matcher = PreferenceCandidateMatcher() if (with_enricher and with_preferences) else None
    reranker = PreferenceReranker() if (with_enricher and with_preferences) else None
    capability = RecommendFromHistoryCapability(
        tool,
        product_enricher=enricher,
        preference_matcher=matcher,
        preference_reranker=reranker,
    )
    return capability, engine, {
        "tool": tool,
        "enricher": enricher,
        "matcher": matcher,
        "reranker": reranker,
    }


def build_minimal_capability() -> tuple[RecommendFromHistoryCapability, FixedEngine]:
    """The M7B-shaped capability: recommend only, no enrichment, no preferences."""
    capability, engine, _ = build_full_capability(
        with_enricher=False, with_preferences=False
    )
    return capability, engine


#: Convenience aliases used by the tests.
FullCapability = build_full_capability
MinimalCapability = build_minimal_capability


class ControlHarness:
    """A composed control plane plus everything a test needs to assert about it."""

    def __init__(
        self,
        *,
        capability: RecommendFromHistoryCapability,
        engine: FixedEngine,
        parts: dict[str, Any],
        controller: LoopController,
        policy: Any,
        memory_service: Any = None,
        user_key: str = "control-test-user",
    ) -> None:
        self.capability = capability
        self.engine = engine
        self.parts = parts
        self.controller = controller
        self.policy = policy
        self.memory_service = memory_service
        self.user_key = user_key


def build_control_harness(
    *,
    policy: Any | None = None,
    limits: LoopLimits | None = None,
    driver: str = "graph",
    with_memory: bool = False,
    with_enricher: bool = True,
    with_preferences: bool = True,
    rows: Any = CANDIDATE_ROWS,
    engine_error: Exception | None = None,
    k: int = 3,
) -> ControlHarness:
    """Compose a full control plane over the real accepted pipeline."""
    capability, engine, parts = build_full_capability(
        rows=rows,
        engine_error=engine_error,
        with_enricher=with_enricher,
        with_preferences=with_preferences,
    )
    resolved_policy = policy or RecordingPolicy(k=k)
    memory_service = make_service() if with_memory else None
    controller = LoopController(
        resolved_policy,
        capability,
        memory_service=memory_service,
        user_key="control-test-user" if with_memory else None,
        limits=limits or LoopLimits(),
        driver=driver,
    )
    return ControlHarness(
        capability=capability,
        engine=engine,
        parts=parts,
        controller=controller,
        policy=resolved_policy,
        memory_service=memory_service,
    )


def engine_identity_set(result: Any) -> set[tuple[str, int]]:
    """Collect every candidate identity a run produced, across all its state channels.

    A helper for the identity-preservation assertions: the accepted pipeline must produce
    the *same* identity set through the Tool, the enricher, the evidence report and the
    reranker.
    """
    identities: set[tuple[str, int]] = set()
    state = result.state if hasattr(result, "state") else result
    tool_result = state.get("tool_result")
    if tool_result is not None:
        identities.update(
            (item.parent_asin, item.item_id) for item in tool_result.recommendations
        )
    return identities

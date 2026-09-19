"""Shared trusted stages of the accepted agent pipeline (AgentRec-X 2.0-alpha).

These are the accepted Milestone 7B-10D stage bodies, extracted verbatim from
:mod:`recommendation.agent.graph` so that **both control planes run the same code**:

* the accepted DAG (:class:`~recommendation.agent.graph.AgentGraph`), whose node methods
  now delegate here; and
* the 2.0-alpha bounded loop
  (:class:`~recommendation.control.loop.LoopController`), whose memory-commit and
  finalize steps call the same functions.

Without this, the loop would need its own copy of finalization and memory commit, and the
two control planes would be free to drift - which is precisely what Stage 1 forbids.  The
extraction is **behaviour-preserving**: no logic, message, ordering or side effect was
changed, only its location and its calling convention (explicit collaborators instead of
``self``).

What stays where
----------------
* recommendation, enrichment, preference matching and reranking live in
  :class:`~recommendation.control.capability.RecommendFromHistoryCapability`, which calls
  the same accepted collaborators;
* **memory commit** lives here (:func:`persist_memory_stage`) because memory ownership must
  not move: the Agent layer still writes only the user's own message, still after the
  response text exists, still through the accepted
  :class:`~recommendation.memory.service.PreferenceMemoryService`;
* **finalization** lives here (:func:`finalize_stage`) so both control planes render the
  identical response from identical structured state.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from .decision import AgentDecision, MalformedDecision
from .rendering import (
    build_grounded_response,
    build_recommendation_response,
    build_reranked_response,
)
from .state import AgentGraphState, read_trusted_history

__all__ = [
    "ROUTE_DIRECT",
    "ROUTE_RECOMMEND",
    "MemoryWriterLike",
    "finalize_stage",
    "persist_memory_stage",
]

#: Route labels reported by the finalizing stage.  Duplicated as literals rather than
#: imported from ``graph`` so this module has no import cycle; the graph asserts the same
#: values in its own tests.
ROUTE_DIRECT = "direct"
ROUTE_RECOMMEND = "recommend"


@runtime_checkable
class MemoryWriterLike(Protocol):
    """The memory seam the persistence stage needs, and nothing more."""

    def process_turn(self, **kwargs: Any) -> Any:
        """Extract and persist explicit preferences from one user-authored turn."""
        ...


def persist_memory_stage(
    state: AgentGraphState,
    *,
    memory_service: MemoryWriterLike | None,
    user_key: str | None,
) -> dict[str, Any]:
    """Extract and persist explicit preferences from the user's own message.

    Semantics are the accepted Milestone 9 semantics, unchanged:

    * only ``user_message`` is passed, because only user-authored text is eligible for
      extraction;
    * the write happens **after** the response text has been produced, so a turn can never
      observe its own write;
    * ``preference_snapshot`` is not rewritten, so this turn's candidates cannot have been
      influenced by this turn's statement;
    * the node returns nothing that could influence candidates.
    """
    if memory_service is None or user_key is None:
        return {}

    user_message = state.get("user_message")
    if not isinstance(user_message, str) or not user_message.strip():
        return {}

    turn_id = state.get("turn_id")
    if not isinstance(turn_id, str) or not turn_id.strip():
        # Idempotency needs a turn identity; without one a stable per-message id is
        # derived so re-processing the same text is still deduplicated.
        turn_id = f"auto-{abs(hash(user_message)) % (10 ** 12)}"

    result = memory_service.process_turn(
        user_key=user_key,
        user_message=user_message,
        turn_id=turn_id,
    )
    return {"memory_update": result}


def finalize_stage(state: AgentGraphState) -> dict[str, Any]:
    """Produce the run's final text and route tag.

    The presentation branch is chosen by **which structured stage actually ran**, in order
    of derivation:

    1. ``reranking`` present -- render the M10B order;
    2. else ``enrichment`` present -- render the M8 order;
    3. else -- render the Tool's order.

    It is deliberately not a string check: a candidate list is presented in the reranked
    order only when a real ``RerankingReport`` exists.  A run that produced no
    recommendation never reaches the candidate branches, because ``decision`` is absent;
    the loop records its own route in that case.
    """
    decision = state.get("decision")
    if not isinstance(decision, AgentDecision):
        raise MalformedDecision("no decision is present in the graph state")

    if decision.needs_recommendation:
        # Re-validate the application-owned history here too: the finalize step must never
        # emit a recommendation for a run that had no trusted history.
        read_trusted_history(state)
        result = state.get("tool_result")
        if result is None:
            raise _no_tool_result()
        reranking = state.get("reranking")
        if reranking is not None:
            enrichment = state.get("enrichment")
            if enrichment is None:
                raise _missing_enrichment()
            return {
                "route": ROUTE_RECOMMEND,
                "final_response": build_reranked_response(
                    reranking, enrichment, state.get("preference_snapshot")
                ),
            }
        enrichment = state.get("enrichment")
        if enrichment is not None:
            return {
                "route": ROUTE_RECOMMEND,
                "final_response": build_grounded_response(
                    enrichment, state.get("preference_snapshot")
                ),
            }
        return {
            "route": ROUTE_RECOMMEND,
            "final_response": build_recommendation_response(
                result, state.get("preference_snapshot")
            ),
        }
    return {
        "route": ROUTE_DIRECT,
        "final_response": decision.direct_response or "",
    }


def _no_tool_result() -> Exception:
    """Return the accepted error for a recommend route with no Tool result.

    Imported lazily so this module does not import the graph module at import time, which
    would create a cycle (``graph`` imports this module).
    """
    from .graph import AgentGraphError

    return AgentGraphError("the recommend route produced no Tool result")


def _missing_enrichment() -> Exception:
    """Return the accepted error for reranking without its enriched candidate set."""
    from .graph import AgentGraphError

    return AgentGraphError(
        "reranking is present but the enriched candidate set is missing"
    )

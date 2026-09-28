"""M4 policies: the deterministic fixed arm and the guarded LLM adaptive arm.

Preregistration references: §7.1 (fixed), §7.2 (adaptive), §7.3 (provider-visible projection),
§8 (hard post-response guard), §9 (termination rules), §10 (deviation handling).

Two ideas matter here and nothing else does:

* **The fixed arm is deterministic.** It queries the four frozen sources in a frozen order at the
  frozen depth and finishes. It never calls a provider.
* **The adaptive arm is guarded mechanically.** The provider only ever *sees*
  ``SELECT_SOURCE`` + ``FINISH`` (via :mod:`experiments.m4_schema_projection`), and after parsing
  the guard below re-checks the proposal against the same frozen rules. The projection alone is not
  sufficient, because the repository's parser still knows the controller's original action set
  (``model_policy.py:547``) -- which is exactly why §8 exists.

Neither policy modifies a frozen file: both implement the repository's own policy protocol.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

from recommendation.control.arguments import CandidateSource
from recommendation.control.schemas import ActionKind, ActionProposal

__all__ = [
    "FIXED_SOURCE_ORDER",
    "M4_ALLOWED_ACTIONS",
    "DeviationRecord",
    "FixedTraversalPolicy",
    "GuardedAdaptivePolicy",
    "ProtocolDeviation",
    "PolicyOutcome",
]


#: The frozen traversal order for the fixed arm: ``CandidateSource.value`` ascending
#: (preregistration §7.1). Declared explicitly rather than sorted at call time so the order is
#: visible in one place and cannot drift with enum edits.
FIXED_SOURCE_ORDER: tuple[CandidateSource, ...] = (
    CandidateSource.CATALOG_SEARCH,
    CandidateSource.HISTORY,
    CandidateSource.SIMILAR_ITEM,
    CandidateSource.TWO_TOWER,
)

#: The only parses the adaptive arm may act on (preregistration §7.3, §8 C4-1).
M4_ALLOWED_ACTIONS: frozenset[ActionKind] = frozenset(
    {ActionKind.SELECT_SOURCE, ActionKind.FINISH}
)


class ProtocolDeviation(RuntimeError):
    """A parsed proposal violated a frozen M4 rule.

    Raised by the guard so the run terminates under the frozen deviation rule (§9) without the
    proposal ever executing or reaching ``CandidateLedger`` (§8 C4-1, C4-2).
    """

    def __init__(self, reason: str, *, detail: str = "") -> None:
        super().__init__(f"{reason}: {detail}" if detail else reason)
        self.reason = reason
        self.detail = detail


@dataclass
class DeviationRecord:
    """One recorded deviation, for the per-user deviation list of preregistration §10."""

    reason: str
    detail: str
    step_index: int


@dataclass
class PolicyOutcome:
    """What a policy decided for one step."""

    proposal: ActionProposal | None = None
    deviation: DeviationRecord | None = None
    finish: bool = False


@dataclass
class FixedTraversalPolicy:
    """The deterministic fixed arm (preregistration §7.1).

    Queries every source in :data:`FIXED_SOURCE_ORDER` exactly once at the frozen depth, then
    finishes. Deterministic: the same context always yields the same next action.
    """

    #: Frozen per-source depth (preregistration §3).
    depth: int = 100
    #: The source universe; defaults to the frozen order.
    order: Sequence[CandidateSource] = FIXED_SOURCE_ORDER
    _index: int = field(default=0, init=False, repr=False)

    @property
    def sources(self) -> tuple[CandidateSource, ...]:
        """The sources this arm will query, in order."""
        return tuple(self.order)

    def queries_so_far(self) -> int:
        """How many sources have been requested."""
        return self._index

    def choose(self, context: Any) -> ActionProposal:
        """Return the next action: the next unqueried source, or ``FINISH``."""
        available = tuple(getattr(context, "available_actions", ()) or ())
        while self._index < len(self.order):
            source = self.order[self._index]
            self._index += 1
            if ActionKind.SELECT_SOURCE not in available:
                # The controller does not offer source selection in this state; skip rather than
                # propose something that would be refused.
                continue
            from recommendation.control.arguments import SelectSourceArguments

            return ActionProposal(
                action=ActionKind.SELECT_SOURCE,
                arguments=SelectSourceArguments(source=source, limit=self.depth),
                rationale=f"fixed traversal: consult {source.value}",
            )
        if ActionKind.FINISH not in available:
            raise ProtocolDeviation(
                "fixed arm exhausted its sources but FINISH is not offered",
                detail=", ".join(a.value for a in available),
            )
        return ActionProposal(
            action=ActionKind.FINISH,
            rationale="fixed traversal complete",
        )


@dataclass
class GuardedAdaptivePolicy:
    """Wraps an LLM policy with the §8 guard, the §9 rules and the §10 deviation list.

    Parameters
    ----------
    inner:
        The repository's :class:`~recommendation.control.model_policy.LLMAgentPolicy` (already
        wired to the schema-projection client of :mod:`experiments.m4_schema_projection`).
    universe:
        The frozen four-source universe. A ``SELECT_SOURCE`` outside it is a deviation.
    depth:
        The frozen per-source depth, applied to every accepted ``SELECT_SOURCE``.
    """

    inner: Any
    universe: Sequence[CandidateSource] = FIXED_SOURCE_ORDER
    depth: int = 100
    #: Sources already queried for this user (preregistration §9 mechanism 1).
    queried: set[CandidateSource] = field(default_factory=set)
    #: Every deviation recorded for this user (preregistration §10).
    deviations: list[DeviationRecord] = field(default_factory=list)
    #: True once the run has terminated under the frozen deviation rule.
    terminated: bool = False
    _step: int = field(default=0, init=False, repr=False)

    @property
    def universe_set(self) -> frozenset[CandidateSource]:
        """The frozen universe as a set."""
        return frozenset(self.universe)

    def _deviation(self, reason: str, detail: str) -> ProtocolDeviation:
        self.deviations.append(
            DeviationRecord(reason=reason, detail=detail, step_index=self._step)
        )
        self.terminated = True
        return ProtocolDeviation(reason, detail=detail)

    def _guard(self, proposal: ActionProposal) -> ActionProposal:
        """Apply the hard post-response guard (preregistration §8).

        Raises :class:`ProtocolDeviation` for anything the frozen rules do not permit. The caller
        records it and terminates; nothing here touches the ledger, and the proposal is never
        returned for execution.
        """
        action = proposal.action

        # C4-1: only SELECT_SOURCE and FINISH may proceed.
        if action not in M4_ALLOWED_ACTIONS:
            raise self._deviation(
                "protocol_deviation:action_not_allowed",
                f"parsed action {action.value!r} is outside the frozen M4 action space "
                f"{sorted(a.value for a in M4_ALLOWED_ACTIONS)}",
            )

        if action is ActionKind.FINISH:
            # §9: FINISH with zero sources queried is prohibited. An empty ledger can only produce
            # an empty ranking, which is a guaranteed miss.
            if not self.queried:
                raise self._deviation(
                    "zero_source_finish",
                    "FINISH proposed before any source was queried",
                )
            return proposal

        # action is SELECT_SOURCE from here on.
        arguments = getattr(proposal, "arguments", None)
        raw_source = getattr(arguments, "source", None)
        try:
            source = (
                raw_source
                if isinstance(raw_source, CandidateSource)
                else CandidateSource(str(raw_source))
            )
        except ValueError:
            raise self._deviation(
                "protocol_deviation:source_outside_universe",
                f"source {raw_source!r} is not a CandidateSource member",
            ) from None

        # C4-2(a): the source must be inside the frozen universe.
        if source not in self.universe_set:
            raise self._deviation(
                "protocol_deviation:source_outside_universe",
                f"source {source.value!r} is not in the frozen M4 universe "
                f"{sorted(s.value for s in self.universe)}",
            )

        # C4-2(b): the source must not already have been queried.
        if source in self.queried:
            raise self._deviation(
                "protocol_deviation:duplicate_source",
                f"source {source.value!r} was already queried; a second query double-counts in RRF",
            )

        # Pin the depth to the frozen value regardless of what the model asked for, so depth is an
        # experimental constant rather than a model-controlled quantity (§3).
        from recommendation.control.arguments import SelectSourceArguments

        return ActionProposal(
            action=ActionKind.SELECT_SOURCE,
            arguments=SelectSourceArguments(source=source, limit=self.depth),
            rationale=proposal.rationale,
        )

    def choose(self, context: Any) -> ActionProposal:
        """Ask the inner policy, guard the answer, and record any deviation."""
        self._step += 1
        if self.terminated:
            raise ProtocolDeviation(
                "the run already terminated under the frozen deviation rule"
            )
        proposal = self.inner.choose(context)
        guarded = self._guard(proposal)
        if guarded.action is ActionKind.SELECT_SOURCE:
            self.queried.add(guarded.arguments.source)  # type: ignore[union-attr]
        return guarded

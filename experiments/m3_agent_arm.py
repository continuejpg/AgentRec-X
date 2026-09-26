"""M3 Agent evaluation arm — the real bounded control plane, driven into the shared evaluator.

This module is the missing piece between the accepted evaluator and the accepted control plane.
It does NOT implement an agent: it *drives* the existing one.

What is real here
-----------------
* the policy is ``recommendation.control.model_policy.LLMAgentPolicy`` — the repository's
  existing model-driven policy, not a stand-in;
* the loop is ``recommendation.control.loop.LoopController`` with the run's own ``LoopLimits``;
* candidates come from the existing ``CandidatePlane``, are recorded in the existing
  ``CandidateLedger`` and confirmed by the existing ``GroundingVerifier``;
* the final ranking is scored by the unmodified shared evaluator
  (``recommendation.evaluation.batched.evaluate_batched``), reached exactly the way every other
  arm reaches it.

The arm only does three things of its own: it converts the ledger's own ranked order into an
item-id ranking, it packs that ranking into the score row the evaluator expects, and it records
a per-user behaviour log.

Deliberate non-actions
----------------------
* **No fallback ranking is invented.** If the run proposes nothing, the ranking is empty and the
  user is a miss. Substituting another arm's ranking would fabricate a result.
* **Seen-history masking is not reimplemented.** The arm hands histories to the evaluator, and
  the evaluator masks. A local mask would be a second implementation of the frozen protocol.
* **The score-row convention is copied, not invented**: descending scores over the ranking, ``-1``
  everywhere else, which is byte-for-byte the convention ``arm_fixed_fusion`` uses, so the
  evaluator's own tie-breaking applies unchanged.
* **``popularity`` is not added to ``CandidateSource``.** The Agent cannot reach it; that
  asymmetry against the fixed-fusion comparator is a recorded limitation, not something this code
  papers over.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Mapping, Protocol, Sequence

import torch

__all__ = [
    "DEFAULT_USER_MESSAGE",
    "M3_JSON_FORMAT_SUFFIX",
    "SOURCE_NAME_TO_CANDIDATE_SOURCE",
    "UNREACHABLE_SOURCES",
    "JsonFormatClient",
    "UserRunOutcome",
    "M3AgentRunner",
    "Harness",
    "arm_agent_llm",
    "summarize_behavior",
    "FAILURE_TERMINATIONS",
    "BUDGET_TERMINATIONS",
]

#: How the Phase-5 source names (used by the accepted fusion arms) correspond to the frozen
#: ``CandidateSource`` vocabulary an Agent can actually name.
SOURCE_NAME_TO_CANDIDATE_SOURCE = {
    "sequential": "history",
    "metadata": "catalog_search",
    "two_tower": "two_tower",
    "tiger": "tiger",
}

#: KNOWN, DELIBERATE, UNFIXED LIMITATION — the source-access asymmetry.
#:
#: The accepted ``fixed_fusion`` comparator fuses ``popularity`` + ``sequential`` + ``metadata``.
#: The frozen ``CandidateSource`` enum has ``HISTORY``, ``CATALOG_SEARCH``, ``SIMILAR_ITEM``,
#: ``TWO_TOWER`` and ``TIGER`` — there is **no member for popularity**, so the Agent's action
#: space cannot name that source and the comparison is not source-for-source identical. Extending
#: the enum would edit frozen protocol (AGENTS.md 19.2 rule 7), and restricting the comparator
#: would change the accepted baseline, so M3 records the asymmetry instead of repairing it.
UNREACHABLE_SOURCES = ("popularity",)

#: The frozen user turn for the evaluation. The cohort carries no natural-language request, so
#: the arm supplies one constant string for every user rather than inventing per-user text.
DEFAULT_USER_MESSAGE = "Recommend something for me, based on my history."

#: DeepSeek's JSON Output mode requires the literal word ``json`` in the system or user prompt
#: *and* an example of the desired shape. The policy's own system prompt says "JSON" but not
#: "json", and the user turn is a ``json.dumps`` of context/actions whose keys never contain the
#: word, so neither requirement is met by the existing request. This suffix supplies exactly the
#: two missing, purely presentational things.
#:
#: It is appended at the transport seam and states no routing rule, no budget and no decision
#: criterion: the offered actions, the context payload and the policy's instructions are all
#: unchanged. See ``docs/M3_DEEPSEEK_AMENDMENT.md``.
M3_JSON_FORMAT_SUFFIX = (
    "\n\nOutput format:\n"
    "Answer with exactly one json object and nothing else — no prose, no code fence.\n"
    "The object must be valid json with this shape:\n"
    '{"action": "<one offered action name>", "arguments": {}}\n'
    "Put any action arguments in the arguments object; omit them for actions that take none."
)

#: Terminations that mean the run failed to reach an accepted completion for a bad reason.
FAILURE_TERMINATIONS = frozenset(
    {"invalid_action", "execution_failed", "no_available_action", "completion_refused"}
)
#: Terminations that mean a bounded budget ran out. Recorded, never repaired.
BUDGET_TERMINATIONS = frozenset({"max_steps", "max_tool_calls"})


class JsonFormatClient:
    """Append the frozen JSON-format instruction to every request's system prompt.

    A transport-level formatting decorator. It changes what the model is told about *how to
    format* its answer and nothing about *what to answer*: the action schema, the context
    payload, the retry correction and the policy's instructions pass through untouched.

    Kept separate from the policy so that "the policy" remains exactly the accepted
    ``LLMAgentPolicy``, and so this addition is visible in one place and removable in one line.
    """

    def __init__(self, inner: Any, *, suffix: str = M3_JSON_FORMAT_SUFFIX) -> None:
        self._inner = inner
        self._suffix = suffix

    @property
    def inner(self) -> Any:
        """The wrapped client, exposed so a test can assert passthrough."""
        return self._inner

    def complete(self, request: Any) -> Any:
        augmented = request.model_copy(
            update={"system_prompt": request.system_prompt + self._suffix}
        )
        return self._inner.complete(augmented)


@dataclass(frozen=True)
class UserRunOutcome:
    """One user's Agent result: where it ranked things, and what it did to get there."""

    ranking: tuple[int, ...]
    behavior: dict[str, Any]


class Harness(Protocol):
    """The per-user control plane the runner drives. Supplied by the caller."""

    @property
    def controller(self) -> Any: ...

    @property
    def ledger(self) -> Any: ...

    @property
    def trusted_history(self) -> tuple[str, ...]: ...


def _ranking_from_run_outputs(result: Any, ledger: Any, item2id: Mapping[str, int]) -> tuple[int, ...]:
    """The run's own final ranking, mapped to item ids, order-preserving and deduplicated.

    Two channels belong to the same run and both are read, in a fixed order:

    1. ``ledger.ranked_parent_asins()`` — the candidate plane's accepted cross-source ordering.
       Populated when the run consulted a trusted source through the plane (``SELECT_SOURCE``,
       ``SEARCH_CATALOG``, ``FIND_SIMILAR``); it carries provenance, so it leads.
    2. ``state["tool_result"].recommendations`` — the verified Stage-1 recommendation the loop
       produced. This is the canonical route for ``RECOMMEND_FROM_HISTORY``, which the controller
       executes through the capability rather than the plane.

    Both are outputs of *this run*. Nothing from another arm, and no default ordering, is ever
    added: if the run produced nothing, the ranking is empty and the user is a miss. The ledger's
    own order is not re-sorted or re-scored. Identities with no catalogue id (and the PAD row) are
    dropped rather than guessed at.
    """
    identities: list[str] = []
    if len(ledger):
        identities.extend(ledger.ranked_parent_asins())
    tool_result = result.state.get("tool_result")
    for item in getattr(tool_result, "recommendations", ()) or ():
        identity = getattr(item, "parent_asin", None)
        if isinstance(identity, str):
            identities.append(identity)

    ranked: list[int] = []
    seen: set[int] = set()
    for identity in identities:
        item_id = item2id.get(identity)
        if item_id is None or item_id <= 0 or item_id in seen:
            continue
        seen.add(item_id)
        ranked.append(int(item_id))
    return tuple(ranked)


def _action_sequence(result: Any) -> list[str]:
    """The actions the run actually proposed, in order, straight off the trajectory."""
    sequence: list[str] = []
    for step in getattr(result.trajectory, "steps", ()) or ():
        proposal = getattr(step, "action_proposal", None)
        if not isinstance(proposal, dict):
            continue
        action = proposal.get("action")
        if action is not None:
            sequence.append(str(action))
    return sequence


def _selected_sources(result: Any) -> list[str]:
    """Sources the policy explicitly named via ``select_source``, in order of first naming."""
    named: list[str] = []
    for step in getattr(result.trajectory, "steps", ()) or ():
        proposal = getattr(step, "action_proposal", None)
        if not isinstance(proposal, dict) or proposal.get("action") != "select_source":
            continue
        arguments = proposal.get("arguments") or {}
        source = arguments.get("source") if isinstance(arguments, dict) else None
        if source is None:
            continue
        value = str(getattr(source, "value", source))
        if value not in named:
            named.append(value)
    return named


def _behavior(result: Any, ledger: Any, ranking: Sequence[int]) -> dict[str, Any]:
    """Per-user behaviour record. Every field is read, none is inferred."""
    control = result.control
    reason = control.termination_reason
    reason_value = None if reason is None else str(getattr(reason, "value", reason))
    status_value = str(getattr(control.status, "value", control.status))
    last_action = control.last_action_kind
    sources = tuple(str(getattr(s, "value", s)) for s in ledger.sources_present())
    actions = _action_sequence(result)
    selected = _selected_sources(result)
    empty = len(ranking) == 0
    return {
        "status": status_value,
        "termination_reason": reason_value,
        "succeeded": bool(result.succeeded),
        "steps": int(control.step_count),
        "tool_calls": int(control.tool_call_count),
        "retries": int(control.retry_count),
        "last_action_kind": None if last_action is None else str(getattr(last_action, "value", last_action)),
        # what the run actually did
        "action_sequence": actions,
        # source selection: sources the policy named explicitly...
        "source_selection": selected,
        # ...and sources that actually contributed candidates through the plane
        "sources_present": list(sources),
        "source_count": len(sources),
        # candidate accounting straight off the ledger
        "candidates": len(ledger),
        "grounded_candidates": len(ledger.grounded_entries),
        "multi_source_candidates": len(ledger.multi_source_parent_asins()),
        "ranking_size": len(ranking),
        # fallback / failure, recorded rather than repaired
        "budget_exhausted": reason_value in BUDGET_TERMINATIONS,
        "failure": (not bool(result.succeeded)) or empty,
        "empty_ranking": empty,
    }


class M3AgentRunner:
    """Drive the real control plane for one evaluation case.

    ``harness_factory(case)`` must return a per-user control plane already wired to the shared
    catalogue, with the policy supplied at construction. Everything the loop needs is injected
    rather than built here, so the same runner serves the tests (synthetic catalogue, scripted
    client) and the evaluation (frozen catalogue, live provider) with no branch between them.
    """

    def __init__(
        self,
        *,
        harness_factory: Callable[[Any], Harness],
        item2id: Mapping[str, int],
        user_message: str = DEFAULT_USER_MESSAGE,
    ) -> None:
        self._harness_factory = harness_factory
        self._item2id = item2id
        self._user_message = user_message
        self.calls = 0

    def run(self, case: Any) -> UserRunOutcome:
        harness = self._harness_factory(case)
        result = harness.controller.run(self._user_message, harness.trusted_history)
        self.calls += 1
        ranking = _ranking_from_run_outputs(result, harness.ledger, self._item2id)
        return UserRunOutcome(ranking=ranking, behavior=_behavior(result, harness.ledger, ranking))


def arm_agent_llm(
    *,
    num_items: int,
    runner: Any,
    records: list[dict[str, Any]] | None = None,
    rankings: list[list[int]] | None = None,
) -> Callable[[Sequence[Any], int], Iterable[Any]]:
    """The Agent arm, in the batch-factory form ``evaluate_arm`` expects.

    Yields ``(histories, targets, scores)`` exactly like every other arm, so the shared evaluator
    applies its own seen-history masking, tie-breaking, ranking and metric code unchanged.
    ``records``, when supplied, receives one behaviour record per case in cohort order, and
    ``rankings`` receives that case's final ranking item ids — the Agent's actual output, kept in a
    separate sink because a 1000-deep ranking per user would bloat the behaviour log.
    """
    if num_items <= 0:
        raise ValueError("num_items must be positive")

    def batches(cases: Sequence[Any], batch_size: int) -> Iterable[Any]:
        for start in range(0, len(cases), batch_size):
            chunk = cases[start : start + batch_size]
            scores = torch.zeros((len(chunk), num_items + 1), dtype=torch.float32)
            for row, case in enumerate(chunk):
                outcome = runner.run(case)
                ordered = list(outcome.ranking)
                if ordered:
                    # Identical convention to arm_fixed_fusion: descending scores down the
                    # ranking, and the tail below is set to -1 so the evaluator's own
                    # deterministic tie-breaking orders the remainder.
                    scores[row, torch.tensor(ordered, dtype=torch.long)] = torch.arange(
                        len(ordered), 0, -1, dtype=torch.float32
                    )
                if records is not None:
                    records.append({"index": start + row, **outcome.behavior})
                if rankings is not None:
                    rankings.append(list(outcome.ranking))
            tail = scores[:, 1:]
            tail[tail == 0.0] = -1.0
            yield (
                [tuple(case.test_history) for case in chunk],
                [case.test_target for case in chunk],
                scores,
            )

    return batches


def summarize_behavior(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Aggregate the per-user behaviour records into the M3 behavioural statistics."""
    total = len(records)
    if not total:
        return {"users": 0}

    def rate(pred: Callable[[Mapping[str, Any]], bool]) -> float:
        return sum(1 for r in records if pred(r)) / total

    selection: dict[str, int] = {}
    termination: dict[str, int] = {}
    for record in records:
        chosen = record.get("source_selection") or record.get("sources_present") or []
        key = ",".join(chosen) or "none"
        selection[key] = selection.get(key, 0) + 1
        term = record.get("termination_reason") or "none"
        termination[term] = termination.get(term, 0) + 1

    tool_calls = [int(r.get("tool_calls", 0)) for r in records]
    steps = [int(r.get("steps", 0)) for r in records]
    ranking_sizes = [int(r.get("ranking_size", 0)) for r in records]
    ordered_tools = sorted(tool_calls)
    return {
        "users": total,
        "mean_tool_calls": sum(tool_calls) / total,
        "median_tool_calls": ordered_tools[total // 2],
        "max_tool_calls": max(tool_calls),
        "total_tool_calls": sum(tool_calls),
        "mean_steps": sum(steps) / total,
        "max_steps": max(steps),
        "total_retries": sum(int(r.get("retries", 0)) for r in records),
        "mean_ranking_size": sum(ranking_sizes) / total,
        "empty_ranking_rate": rate(lambda r: bool(r.get("empty_ranking"))),
        "fallback_rate": rate(lambda r: bool(r.get("budget_exhausted"))),
        "failure_rate": rate(lambda r: bool(r.get("failure"))),
        "completion_rate": rate(lambda r: bool(r.get("succeeded"))),
        "source_selection_distribution": selection,
        "termination_reason_distribution": termination,
    }

"""A model-driven next-action policy (Phase 1).

`LLMAgentPolicy` replaces the *deciding* component of the bounded loop with a model, and
changes nothing else.  It returns an untrusted
:class:`~recommendation.control.schemas.ActionProposal`; the existing validator authorises it,
the existing executor runs it, the existing verifier and completion guard judge it, and the
existing controller bounds it.

The authority rule this module is built around:

    The model decides **what action to propose**.
    Trusted code decides whether that action is legal, how it is executed, which product
    identities exist, which catalogue facts are true, whether a hard constraint holds,
    whether memory may be committed, and whether FINISH is permitted.

Nothing here can weaken that, because the policy has no access to any of it.  It receives a
:class:`~recommendation.control.context.PolicyContext` - a projection with no history, no
catalogue, no store and no executor - and it returns a proposal.  It cannot run a tool, read a
fact, write state or declare success.

Structured output, not string matching
--------------------------------------
The model is asked for one JSON object and the answer is parsed into the repository's own
:class:`~recommendation.control.schemas.ActionProposal`.  There is no ``if "search" in text``
anywhere: a malformed answer, an unknown action, or arguments that do not match the action's
declared model are all *rejections*, and the run's failure path is the controller's, not a
guess.

What the model is told
----------------------
A concise system prompt stating the policy contract, plus a bounded JSON payload: the user
request, the offered actions **generated from the real argument models**, the task constraints,
the candidate-state summary, the last observation summary, and the remaining budget.  The
action list is derived from :data:`~recommendation.control.schemas.ARGUMENTS_BY_ACTION`, so a
new action cannot be added to the enum without appearing in the prompt - and the prompt cannot
describe an action that does not exist.

Bounded failure
---------------
A model that answers badly is retried at most ``max_attempts`` times *within one policy call*,
and each policy call is one controller step.  A model that never answers properly therefore
ends the run through the controller's own budget, never through an unbounded loop.  Transport
failures and unusable answers end as :class:`PolicyActionError`, which the controller already
treats as a deterministic abort.
"""

from __future__ import annotations

import json
from typing import Any

from .context import PolicyContext
from .model_client import ModelCallError, ModelRequest, ModelResponse, StructuredModelClient
from .schemas import (
    ARGUMENTS_BY_ACTION,
    CANDIDATE_ACTIONS,
    ActionKind,
    ActionProposal,
    PolicyActionError,
)

__all__ = [
    "LLM_POLICY_NAME",
    "POLICY_PARSE_PROTOCOL_VERSION",
    "POLICY_PROMPT_VERSION",
    "LLMAgentPolicy",
    "build_action_schema",
    "build_policy_context_payload",
    "build_policy_system_prompt",
]

#: Stable policy identity, recorded so a trajectory says which policy decided.
LLM_POLICY_NAME = "llm_agent_policy"

#: Version of the prompt contract, bumped when the payload shape changes.
POLICY_PROMPT_VERSION = 1
#: Version of the *parse protocol* the policy applies to a model answer.  Separate from the
#: prompt version because the two change independently: a parse rule can be corrected without
#: touching the contract the model is given.  It is part of the recording fingerprint, so a
#: trace recorded under one protocol is never replayed as though it were made under another.
#:
#: 1: an empty ``arguments`` object was rejected for an action that declares none.
#: 2: an empty ``arguments`` object is folded onto "no arguments" for such an action.  The
#:    Phase-3 live baseline showed this accounted for 7 of 14 case failures.
POLICY_PARSE_PROTOCOL_VERSION = 2

#: The policy contract.  Deliberately short: the architecture document does not belong in a
#: per-step prompt, and a long prompt is a brittle one.
SYSTEM_PROMPT = """\
You are the next-action policy of a bounded recommendation agent.

Choose exactly ONE action from the offered actions, then stop.

Rules:
1. Use only actions in the offered list. Never invent an action.
2. Never invent a product identity, price, weight, brand, availability or any catalogue fact.
   You do not have them. Facts come from tool observations only.
2a. A reasoning action that names products (for example GET_DETAILS or COMPARE) may reference
   ONLY identities listed in "candidates.candidate_refs". Those are the run's grounded
   candidates. Any other identity is refused, so choose a listed one and read its facts
   instead of guessing. You may choose which listed candidate to target.
2b. "candidate_refs" tells you each candidate's eligibility and whether its evidence needs
   attention. Prefer a candidate whose evidence is unsettled when you need more facts.
3. An action's observation is the only evidence that it succeeded. Do not assume success.
4. If a needed fact or product is missing, choose an action that retrieves or asks for it.
5. Prefer the fewest steps that answer the request. Do not repeat an action whose observation
   already answered it.
6. FINISH only ends the turn as a proposal; it is validated and may be refused.
7. Respect the remaining budget. Do not plan more actions than remain.
8. Output must be a single JSON object and nothing else:
   {"action": "<one of the offered actions>", "arguments": {...}, "rationale": "<short>"}
   Omit "arguments" for an action that takes none. "rationale" is optional and is never trusted.
"""


# --------------------------------------------------------------------------- #
# Prompt construction - derived from the real schemas, never duplicated
# --------------------------------------------------------------------------- #


def _json_type(annotation: Any) -> str:
    """Return a short JSON-ish type name for a field annotation."""
    text = str(annotation)
    if "int" in text:
        return "integer"
    if "float" in text:
        return "number"
    if "bool" in text:
        return "boolean"
    if "tuple" in text or "list" in text:
        return "array"
    if "dict" in text:
        return "object"
    return "string"


def build_action_schema(available: tuple[ActionKind, ...]) -> tuple[dict[str, Any], ...]:
    """Describe each offered action and its arguments, from the declared argument models.

    Generated rather than hand-written, so the prompt cannot claim an action exists that does
    not, cannot omit a newly added action, and cannot describe arguments the validator would
    reject.  Only the *offered* actions appear: a policy must choose from what the system
    permits, and telling the model about actions it may not take would invite a refused
    proposal.
    """
    schema: list[dict[str, Any]] = []
    for action in available:
        model = ARGUMENTS_BY_ACTION.get(action)
        arguments: list[dict[str, Any]] = []
        required: list[str] = []
        if model is not None:
            for name, field in model.model_fields.items():
                arguments.append(
                    {
                        "name": name,
                        "type": _json_type(field.annotation),
                        "required": field.is_required(),
                    }
                )
                if field.is_required():
                    required.append(name)
        entry: dict[str, Any] = {
            "action": action.value,
            "arguments": arguments,
            "required_arguments": required,
        }
        if action in CANDIDATE_ACTIONS:
            entry["produces_candidates"] = True
        schema.append(entry)
    return tuple(schema)



def build_policy_context_payload(context: PolicyContext) -> dict[str, Any]:
    """Build the bounded, structured view of the control state the model receives.

    Only what a next-action decision needs.  What is deliberately absent, and therefore cannot
    leak: trusted behavioural history, the memory store or its ``user_key``, the catalogue, the
    candidate ledger, raw scores, product metadata, and the full trajectory.

    Candidate identity, precisely scoped
    ------------------------------------
    The payload carries the run's **legal reasoning targets** - the bounded
    :attr:`~recommendation.control.context.PolicyContext.grounded_candidates` projection - so a
    model can choose *which* already-grounded candidate to inspect.  That is the difference
    between a model that selects a target and a model that only selects an action while a
    binder quietly picks the product for it.

    What the payload still cannot carry: the catalogue, the ledger, any candidate the run did
    not ground, any product fact (title, price, weight, features), any raw score, and any
    unbounded list.  A reference is identity plus *reasoning position* - the same eligibility
    and evidence summary the policy context already exposes - and nothing else.
    """
    observation = context.last_observation
    payload: dict[str, Any] = {
        "user_request": context.user_request,
        "task_intent": context.task_intent,
        "task_constraints": list(context.task_constraints),
        "has_trusted_history": context.has_trusted_history,
        "active_preference_count": context.active_preference_count,
        "hard_constraint_count": context.hard_constraint_count,
        "inferred_constraint_count": context.inferred_constraint_count,
        "awaiting_user": context.awaiting_user,
        "candidates": {
            "grounded": context.candidate_state.grounded,
            "count": context.candidate_state.candidate_count,
            "verification_status": context.candidate_state.verification_status,
            # The bounded reference list: identity plus reasoning position, per candidate.
            # Name the field "candidate_refs" because that is what the model treats them as -
            # values it may *reference* in a reasoning action's arguments, never invent.
            "candidate_refs": [ref.as_dict() for ref in context.grounded_candidates],
        },
        # Phase 2: the three-state feasibility split, as **counts plus constraint labels**.
        # This is what makes an unresolved candidate actionable: without it the model can tell
        # that candidates exist but not that some are only partly decided, so it cannot know
        # whether reading a missing fact would change the eligible set.  The excluded identities
        # are deliberately absent - the model learns *that* something was excluded and *which
        # constraint* excluded it, which is what choosing a next action needs, and it still
        # cannot name a product outside the grounded allowlist.
        "constraint_feasibility": {
            "active_constraints": list(context.active_constraints),
            "verified_eligible_count": context.verified_eligible_count,
            "ineligible_count": context.ineligible_count,
            "unresolved_count": context.unresolved_count,
        },
        "budget": {
            "step_index": context.step_index,
            "remaining_steps": context.remaining_steps,
            "remaining_tool_calls": context.remaining_tool_calls,
        },
        "previous_proposal_rejected": context.last_proposal_rejected,
        "run_status": context.run_status.value,
    }
    if observation is not None:
        payload["last_observation"] = _observation_summary(observation)
    else:
        payload["last_observation"] = None
    return payload


def _observation_summary(observation: Any) -> dict[str, Any]:
    """Summarise an observation for the prompt, without its payload.

    A reasoning observation's grounded *facts* are deliberately not copied in: the model does not
    need them to choose the next action, and putting catalogue values into a prompt is how a
    model starts treating them as things it may restate.  It learns *that* the facts were read,
    not what they were.
    """
    summary: dict[str, Any] = {
        "kind": getattr(observation, "kind", "unknown"),
        "status": getattr(observation, "status", None),
        "verification_status": getattr(observation, "verification_status", None),
    }
    for field in ("source", "returned_k", "has_candidates", "ungrounded_count"):
        value = getattr(observation, field, None)
        if value is not None:
            summary[field] = value
    # A refused completion or a failed action is the signal that should change the next action.
    for field in ("code", "recoverable", "permitted_actions", "reason", "supported"):
        value = getattr(observation, field, None)
        if value is not None:
            summary[field] = value
    note = getattr(observation, "verification_note", None)
    if note:
        summary["note"] = str(note)[:200]
    return summary


def build_policy_system_prompt() -> str:
    """Return the policy contract prompt."""
    return SYSTEM_PROMPT


# --------------------------------------------------------------------------- #
# The policy
# --------------------------------------------------------------------------- #


class LLMAgentPolicy:
    """A next-action policy that asks a model, then parses its answer into a proposal.

    Parameters
    ----------
    model:
        Any :class:`~recommendation.control.model_client.StructuredModelClient`.  Injected, so
        the test suite uses a scripted client and a deployment supplies a provider adapter.
        The policy never constructs one.
    max_attempts:
        How many times one policy call may ask the model.  The default of two allows a single
        correction after unusable output - valuable, because most malformed answers are one
        formatting mistake - while keeping a policy call bounded.  Each call is one controller
        step, so the run's own step budget bounds the total independently.
    """

    def __init__(
        self,
        model: StructuredModelClient,
        *,
        max_attempts: int = 2,
        name: str = LLM_POLICY_NAME,
    ) -> None:
        if not callable(getattr(model, "complete", None)):
            raise PolicyActionError(
                "a model client must provide a callable complete(request) method"
            )
        if not isinstance(max_attempts, int) or isinstance(max_attempts, bool) or max_attempts < 1:
            raise PolicyActionError("max_attempts must be a positive integer")
        self._model = model
        self._max_attempts = max_attempts
        self._name = name
        self._last_metadata: dict[str, Any] = {}

    # -- metadata ---------------------------------------------------------- #

    @property
    def name(self) -> str:
        """Stable policy identity, recorded in the trajectory."""
        return self._name

    @property
    def model(self) -> StructuredModelClient:
        """The injected model client (exposed for inspection and tests)."""
        return self._model

    @property
    def max_attempts(self) -> int:
        """How many model calls one policy decision may make."""
        return self._max_attempts

    @property
    def model_metadata(self) -> dict[str, Any]:
        """The model's *identity*, as opposed to how a particular decision went.

        Phase 3 has to distinguish three things that all used to look like "a model policy": the
        deterministic scripted double, a recorded provider trace being replayed, and a real
        provider being called.  The client is the only component that knows which of those is
        happening, so it is asked once, by duck-typing a small optional surface:

        * ``mode`` - a :class:`~recommendation.control.model_recorder.ReplayMode` value
          (``live`` / ``record`` / ``replay`` / ``record_if_missing``);
        * ``provider`` - the provider profile name, when the client knows one;
        * ``model`` - the endpoint's model identifier.

        A client with none of that surface is a scripted double, which is reported as ``scripted``
        rather than as an unknown live run.
        """
        metadata: dict[str, Any] = {}
        mode = getattr(self._model, "mode", None)
        execution_mode = getattr(mode, "value", None)
        if isinstance(execution_mode, str) and execution_mode:
            metadata["execution_mode"] = execution_mode
        else:
            metadata["execution_mode"] = "scripted"
        provider = getattr(self._model, "provider", None)
        if isinstance(provider, str) and provider:
            metadata["model_provider"] = provider
        model = getattr(self._model, "model", None)
        if isinstance(model, str) and model:
            metadata["model_endpoint_id"] = model
        return metadata

    @property
    def last_metadata(self) -> dict[str, Any]:
        """Safe diagnostics for the step just decided.

        Answers *which policy decided, what it proposed, whether parsing succeeded, how many
        model calls it took, and whether a correction was needed*.  Contains no prompt text, no
        model chain-of-thought and no secrets: the rationale is truncated because it is
        untrusted diagnostic prose, not reasoning truth.
        """
        return dict(self._last_metadata)

    def _accumulate_usage(self, response: Any) -> None:
        """Fold one response's usage into this decision's metadata.

        Only non-``None`` values are added, so a provider that omits usage never appears to have
        reported zero, and a decision that mixed a reported and an unreported call reports the
        partial sum it actually observed.
        """
        for field in ("input_tokens", "output_tokens", "total_tokens"):
            value = getattr(response, field, None)
            if isinstance(value, int) and not isinstance(value, bool):
                key = f"step_{field}"
                self._last_metadata[key] = self._last_metadata.get(key, 0) + value
        latency = getattr(response, "latency_ms", None)
        if isinstance(latency, (int, float)) and not isinstance(latency, bool):
            key = "step_latency_ms"
            self._last_metadata[key] = round(
                float(self._last_metadata.get(key, 0.0)) + float(latency), 3
            )
        cost = getattr(response, "estimated_cost", None)
        if isinstance(cost, (int, float)) and not isinstance(cost, bool):
            key = "step_estimated_cost"
            self._last_metadata[key] = round(
                float(self._last_metadata.get(key, 0.0)) + float(cost), 8
            )
        finish = getattr(response, "finish_reason", None)
        if finish:
            self._last_metadata["finish_reason"] = str(finish)

    # -- the seam ---------------------------------------------------------- #

    def choose(self, context: PolicyContext) -> ActionProposal:
        """Ask the model for one action and return it as an untrusted proposal.

        Raises
        ------
        PolicyActionError
            The model could not produce a usable action within ``max_attempts``.  The controller
            treats this as a deterministic abort with ``NO_AVAILABLE_ACTION`` rather than a
            crash, which is the existing bounded-failure path.
        """
        self._last_metadata = {
            "policy": self._name,
            "model_calls": 0,
            "attempts": 0,
            **self.model_metadata,
        }
        available = tuple(context.available_actions)
        if not available:
            raise PolicyActionError("the system offered no action for this control state")

        action_schema = build_action_schema(available)
        payload = build_policy_context_payload(context)
        correction: str | None = None
        failures: list[str] = []

        for attempt in range(1, self._max_attempts + 1):
            request = ModelRequest(
                system_prompt=build_policy_system_prompt(),
                context_payload=payload,
                action_schema=action_schema,
                correction=correction,
            )
            response = self._call_model(request)
            self._last_metadata["model_calls"] += 1
            self._last_metadata["attempts"] = attempt
            if response.model_id is not None:
                self._last_metadata["model_id"] = response.model_id
            # Phase 3 diagnostics: accumulate the real provider's own reporting.  Token counts
            # and latency are summed across the calls one *decision* took (a retry after a parse
            # failure costs a second call), so the trajectory can attribute cost per step rather
            # than only per run.  Unknowns stay unknown: a field the provider did not report is
            # left absent instead of being counted as zero.
            self._accumulate_usage(response)

            if response.is_empty:
                failures.append("empty_response")
                correction = "your answer was empty; return one JSON action object"
                self._last_metadata["last_failure"] = "empty_response"
                continue

            proposal, error = self._parse(response, available)
            if proposal is not None:
                self._last_metadata["parse_ok"] = True
                self._last_metadata["proposed_action"] = proposal.action.value
                if proposal.rationale:
                    self._last_metadata["rationale"] = proposal.rationale[:200]
                if failures:
                    self._last_metadata["recovered_after"] = tuple(failures)
                return proposal

            failures.append(error or "unparseable")
            self._last_metadata["parse_ok"] = False
            self._last_metadata["last_failure"] = error
            correction = error

        self._last_metadata["failures"] = tuple(failures)
        raise PolicyActionError(
            "the model did not produce a usable action within "
            f"{self._max_attempts} attempt(s): {', '.join(failures)}"
        )

    # -- internals --------------------------------------------------------- #

    def _call_model(self, request: ModelRequest) -> ModelResponse:
        """Call the model, normalising a transport failure into the bounded failure path."""
        try:
            return self._model.complete(request)
        except ModelCallError as exc:
            self._last_metadata["model_calls"] = self._last_metadata.get("model_calls", 0) + 1
            self._last_metadata["last_failure"] = exc.code
            raise PolicyActionError(f"the model call failed: {exc.code}") from exc
        except Exception as exc:  # noqa: BLE001 - a provider exception must not crash the loop
            self._last_metadata["model_calls"] = self._last_metadata.get("model_calls", 0) + 1
            self._last_metadata["last_failure"] = "provider_error"
            raise PolicyActionError(
                f"the model call failed: {type(exc).__name__}"
            ) from exc

    @staticmethod
    def _parse(
        response: ModelResponse, available: tuple[ActionKind, ...]
    ) -> tuple[ActionProposal | None, str | None]:
        """Parse a model answer into a proposal, or return why it could not.

        The three rejection classes are deliberately distinct, because they mean different
        things about the model and about the run:

        * **not JSON** - a formatting failure;
        * **unknown or unoffered action** - the model proposed something the system did not
          permit, which is a protocol violation rather than a formatting slip;
        * **invalid arguments** - the action is permitted but its arguments do not satisfy the
          action's own declared model, which the construction of ``ActionProposal`` re-checks.

        Returns ``(proposal, None)`` or ``(None, reason)``.  Never guesses a default action.
        """
        text = response.text.strip()
        try:
            decoded = json.loads(text)
        except json.JSONDecodeError:
            return None, "not_json"
        if not isinstance(decoded, dict):
            return None, "not_a_json_object"

        raw_action = decoded.get("action")
        if not isinstance(raw_action, str):
            return None, "missing_action"
        try:
            action = ActionKind(raw_action)
        except ValueError:
            return None, f"unknown_action:{raw_action[:40]}"
        if action not in available:
            return None, f"action_not_offered:{action.value}"

        # ``ActionProposal`` validates that the arguments match the action's declared model and
        # rejects a foreign or malformed payload itself.  The policy adds no coercion: a
        # near-miss is a rejection, not a repair.
        fields: dict[str, Any] = {"action": action}
        arguments = decoded.get("arguments")
        if arguments is not None:
            # ``{}`` and "no arguments" are the same statement for an action that declares no
            # argument fields.  Measured on the Phase-3 live baseline: DeepSeek sent
            # ``"arguments": {}`` for RECOMMEND_FROM_HISTORY on 20 of 44 calls, and the
            # ``ActionProposal`` contract rejects it (``extra="forbid"``), so 7 of 14 cases
            # failed with no usable action after two attempts *because of a JSON convention*
            # rather than because of any policy error.
            #
            # The normalisation belongs here, in the model seam, and not in ``ActionProposal``:
            # that type is also the deterministic policies' construction contract, and it should
            # keep rejecting a foreign or malformed payload outright.  A text model needs the
            # JSON equivalent of "absent" folded onto "absent"; trusted validation is untouched,
            # and an action that *does* take arguments still gets its payload checked in full.
            if arguments == {} and not ARGUMENTS_BY_ACTION[action].model_fields:
                arguments = None
            else:
                fields["arguments"] = arguments
        if action is ActionKind.RECOMMEND_FROM_HISTORY:
            fields["k"] = decoded.get("k", 4)
        rationale = decoded.get("rationale")
        if isinstance(rationale, str) and rationale.strip():
            fields["rationale"] = rationale.strip()[:280]
        try:
            return ActionProposal(**fields), None
        except Exception:  # noqa: BLE001 - pydantic reports several error types
            return None, f"invalid_arguments:{action.value}"

"""Decision contract for the minimal LangGraph agent (Milestone 7B).

The graph does not talk to a language model directly.  It calls an injected
**decision model** - any object satisfying :class:`DecisionModel`::

    decide(messages) -> AgentDecision

and that call must return an :class:`AgentDecision`, i.e. a validated choice
between exactly two routes:

* :attr:`AgentAction.RECOMMEND` - ask the Recommendation Tool for ``k`` items;
* :attr:`AgentAction.DIRECT_RESPONSE` - answer without touching the recommender.

Why this type exists
--------------------
The whole point of the Milestone 7B boundary is that the decision step is
**structurally incapable** of influencing the recommendation history.  Three
properties do that work:

1. ``AgentDecision`` has an ``extra="forbid"`` frozen schema, so a decision
   payload cannot smuggle a ``history``/``trusted_user_history`` field through -
   attempting it is a hard validation error, not a silent merge.
2. :func:`build_decision_messages` accepts *only* the user message, so there is no
   code path that could put trusted history (or item ids, encoded model history,
   SASRec tensors, mapping internals, checkpoint internals) into the decision
   model's prompt.
3. The graph state's ``trusted_user_history`` field is never written by a decision
   node; it is owned by the application and only read by the Tool node.

A malformed decision (unknown action, internal entity, missing/blank
``direct_response``, unparseable payload) raises :class:`MalformedDecision`
rather than guessing a route.  There is deliberately no default action.
"""

from __future__ import annotations

import json
from enum import Enum
from typing import Annotated, Any, Literal, Protocol, Sequence, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field, field_validator

from recommendation.tools.schemas import DEFAULT_K, MAX_K, MIN_K

__all__ = [
    "AGENT_DECISION_VERSION",
    "AgentAction",
    "AgentDecision",
    "DecisionMessage",
    "DecisionModel",
    "MalformedDecision",
    "build_decision_messages",
    "parse_agent_decision",
]


class AgentAction(str, Enum):
    """The two routes the agent graph supports in Milestone 7B."""

    DIRECT_RESPONSE = "direct_response"
    RECOMMEND = "recommend"


#: Decision contract version.  An adapter/model may echo it, but the graph always
#: validates against this value's schema.
AGENT_DECISION_VERSION = 1


class MalformedDecision(Exception):
    """The decision model produced output that violates the decision contract.

    Raised instead of falling back to a default route, so a broken decision model
    is loud rather than silently recommending.
    """


class DecisionMessage(BaseModel):
    """One chat message handed to the decision model.

    A real provider adapter would map this onto its own message type; keeping the
    shape local means Milestone 7B needs no provider SDK (no OpenAI, Anthropic or
    Google client is imported anywhere in this package).
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    role: Literal["system", "user"]
    content: str

    def as_dict(self) -> dict[str, str]:
        """Return the plain ``{"role": ..., "content": ...}`` form."""
        return {"role": self.role, "content": self.content}


#: System prompt for the decision step.  It deliberately says nothing about the
#: user's interaction history, and it forbids the model from producing catalog
#: facts - candidate *facts* come from the Tool, never from the model.
DECISION_SYSTEM_PROMPT = (
    "You are the routing step of a recommendation assistant.\n"
    "Choose exactly one action:\n"
    '  - "recommend": the user wants product recommendations. Set "k" to the '
    f"number of items to return ({MIN_K}..{MAX_K}); if the user did not say how "
    f"many, use {DEFAULT_K}.\n"
    '  - "direct_response": the user is greeting you, thanking you, or asking '
    'something that does not need product recommendations. Set "direct_response" '
    "to the reply text.\n"
    "Output only the action object. Never include product titles, brands, prices, "
    "availability or any other catalog facts; you do not have them. Never include "
    "or refer to the user's interaction history."
)


class AgentDecision(BaseModel):
    """The validated route chosen by the decision model.

    Contract:

    * exactly one of two actions;
    * ``k`` is only meaningful for ``recommend`` - supplying it with
      ``direct_response`` is rejected;
    * ``direct_response`` is required for ``direct_response`` and must be
      non-blank text;
    * ``extra="forbid"``: no other field exists, so a decision cannot carry
      interaction history, candidate lists, item ids or scores;
    * ``frozen=True``: a produced decision cannot be mutated downstream.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    action: AgentAction
    version: Annotated[int, Field(strict=True)] = AGENT_DECISION_VERSION

    # ``strict=True`` matters for the same reason it does in the Tool request:
    # without it pydantic would coerce "10" -> 10 and True -> 1, letting sloppy
    # model output masquerade as a well-formed decision.
    k: (
        Annotated[int, Field(strict=True, ge=MIN_K, le=MAX_K)] | None
    ) = None

    direct_response: str | None = None

    @field_validator("direct_response")
    @classmethod
    def _validate_direct_response(cls, value: str | None) -> str | None:
        """Strip surrounding whitespace and reject an all-whitespace reply."""
        if value is None:
            return None
        stripped = value.strip()
        if not stripped:
            raise ValueError("direct_response must not be blank")
        return stripped

    def model_post_init(self, __context: Any) -> None:
        """Enforce the cross-field rules that a per-field schema cannot express."""
        if self.action is AgentAction.RECOMMEND:
            if self.direct_response is not None:
                raise ValueError(
                    "direct_response must be omitted when action is 'recommend'"
                )
        else:  # AgentAction.DIRECT_RESPONSE
            if self.direct_response is None:
                raise ValueError(
                    "direct_response is required when action is 'direct_response'"
                )
            if self.k is not None:
                raise ValueError(
                    "k must be omitted when action is 'direct_response'"
                )

    # -- derived helpers --------------------------------------------------- #

    @property
    def needs_recommendation(self) -> bool:
        """True when this decision routes to the Recommendation Tool."""
        return self.action is AgentAction.RECOMMEND

    @property
    def requested_k(self) -> int:
        """The number of items to request; only valid on the recommend route."""
        if self.action is not AgentAction.RECOMMEND:
            raise MalformedDecision("requested_k is only defined for action 'recommend'")
        return DEFAULT_K if self.k is None else self.k

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable view, omitting unset optional fields."""
        return self.model_dump(exclude_none=True)


@runtime_checkable
class DecisionModel(Protocol):
    """The single method the agent graph requires from a decision source.

    This is the dependency-injection seam that keeps Milestone 7B offline: tests
    and the smoke experiment supply a deterministic stub, while a later milestone
    can supply a real provider adapter **without changing graph code**.

    The call is synchronous on purpose - the graph is invoked with LangGraph's
    synchronous ``invoke``.  An asynchronous variant is out of scope for M7B.
    """

    def decide(self, messages: Sequence[DecisionMessage]) -> AgentDecision:
        """Return the route for the supplied conversation messages."""
        ...


def build_decision_messages(user_message: str) -> tuple[DecisionMessage, ...]:
    """Build the decision prompt for ``user_message`` **and nothing else**.

    Taking a single string here is the point: the decision step has no argument
    that could carry ``trusted_user_history``, item ids, encoded histories,
    SASRec tensors, mapping internals or checkpoint internals.  Trusted history
    stays in graph state and is read only by the Tool node.
    """
    if not isinstance(user_message, str) or not user_message.strip():
        raise ValueError("user_message must be a non-empty string")
    return (
        DecisionMessage(role="system", content=DECISION_SYSTEM_PROMPT),
        DecisionMessage(role="user", content=user_message.strip()),
    )


def parse_agent_decision(payload: Any) -> AgentDecision:
    """Validate a raw decision payload, raising :class:`MalformedDecision` on failure.

    Accepts an :class:`AgentDecision` (returned unchanged) or a mapping/JSON
    string, which is what a real provider adapter would hand back.  Every
    rejection path raises rather than picking a default route, and the raised
    message never echoes the rejected payload.
    """
    if isinstance(payload, AgentDecision):
        return payload

    if isinstance(payload, (bytes, bytearray)):
        raise MalformedDecision(
            f"decision payload must be a mapping or JSON object, got {type(payload).__name__}"
        )

    if isinstance(payload, str):
        text = payload.strip()
        if not text:
            raise MalformedDecision("decision payload is empty")
        try:
            payload = json.loads(text)
        except json.JSONDecodeError as exc:
            raise MalformedDecision(f"decision payload is not valid JSON: {exc.msg}") from exc

    if not isinstance(payload, dict):
        raise MalformedDecision(
            f"decision payload must be a mapping, got {type(payload).__name__}"
        )

    try:
        return AgentDecision.model_validate(payload)
    except Exception as exc:  # noqa: BLE001 - pydantic wraps several exception types
        raise MalformedDecision(_safe_contract_violation(exc)) from exc


def _safe_contract_violation(exc: Exception) -> str:
    """Summarise a validation failure **without echoing the rejected payload**.

    ``include_input=False`` and ``include_url=False`` matter: a decision payload is
    untrusted model output, and a validation error must not copy it (or a guessed
    history value) into a log line or an exception message.  Only the failing
    field/rule is reported.
    """
    summary = ""
    errors = getattr(exc, "errors", None)
    if callable(errors):
        try:
            details = errors(include_input=False, include_url=False)
        except TypeError:  # pragma: no cover - older pydantic signature
            details = errors()
        parts = []
        for detail in details[:3]:
            location = ".".join(str(part) for part in detail.get("loc", ())) or "<root>"
            parts.append(f"{location}: {detail.get('msg', 'invalid')}")
        summary = "; ".join(parts)
    if not summary:
        summary = type(exc).__name__
    return f"decision payload violates the contract ({summary})"

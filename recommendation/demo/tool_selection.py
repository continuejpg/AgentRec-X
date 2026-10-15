"""LLM recommendation-tool selection with trusted validation (V2 E-01A-1).

One job: given the user's request and the tools **the caller says are available right now**, ask
the model which of them to run, then validate that answer in trusted code before anything can
execute.  The model is told what exists; it never discovers, invents or re-weights a tool.

Deliberate limits, all enforced by validation rather than by prompt wording:

* ``route="direct"`` runs no recommendation tool, and a plan that claims otherwise is refused
  (a contradictory answer is a failure, not something to silently reinterpret);
* every selected name must be in the caller's allowed set, which is how an unavailable or
  not-ready source (TIGER before its handshake, an opted-out ``similar_item``) is unreachable;
* the count may not exceed the caller's tool budget, duplicates are refused, and
  ``route="recommend"`` requires at least one source;
* the answer schema carries **only** route, sources and a short reason - so a product id, a score
  or a fabricated provenance block cannot be expressed, let alone accepted;
* no failure path ever substitutes a default source list.  A refused plan raises
  :class:`ToolSelectionError` with a stable code, and the caller decides what to do about it.

This module does not execute anything: it converts a validated selection into the ordered source
plan the existing ``DemoSourcePlanPolicy`` already accepts.
"""

from __future__ import annotations

import json
from typing import Any, Literal, Mapping, Sequence

from pydantic import BaseModel, ConfigDict, Field

from recommendation.control.model_client import ModelCallError, ModelRequest

ROUTE_RECOMMEND = "recommend"
ROUTE_DIRECT = "direct"

#: Stable codes.  A caller may branch on these; none of them means "run everything".
CODE_EMPTY_REQUEST = "empty_request"
CODE_NO_TOOLS = "no_tools_available"
CODE_INVALID_BUDGET = "invalid_budget"
CODE_UNKNOWN_SOURCE = "unknown_source"
CODE_DUPLICATE_SOURCE = "duplicate_source"
CODE_OVER_BUDGET = "over_budget"
CODE_NO_SOURCES = "no_sources"
CODE_DIRECT_WITH_SOURCES = "direct_with_sources"
CODE_INVALID_PLAN = "invalid_plan"

_SYSTEM_PROMPT = """You choose which product-recommendation tools to run for one shopper turn.

Rules:
1. Choose only from the tools listed as available. Never invent a tool.
2. You may choose one tool, several tools, or every available tool.
3. If the turn does not ask for products (a greeting, a question about the conversation, a
   preference statement with no request), answer with route "direct" and choose no tool.
4. If the turn asks for products, answer with route "recommend" and choose at least one tool.
5. Choose at most the stated number of tools, each at most once.
6. Never output a product id, a score, a ranking or an evidence claim: you are choosing tools,
   not products.
7. You are planning before any tool has run. Do not claim to have seen results.
8. The context may list preferences this shopper stated earlier. They are background, not this
   turn's request: choose tools for what the shopper is asking now, and never treat a stored
   preference as something they just said. Do not repeat a stored preference back as if it were
   the request, and do not append it to the request text.

Answer as JSON with exactly: route ("recommend" or "direct"), selected_sources (list of tool
names, possibly empty), reason (one short sentence)."""

#: Offered actions, in the seam's own shape: ``ModelRequest.action_schema`` is a tuple of
#: action descriptions and is rendered into the user message by ``ModelRequest.as_messages``.
_ACTION_SCHEMA: tuple[dict[str, Any], ...] = (
    {
        "action": "select_tools",
        "description": "the turn asks for products; run the named tools, in this order",
        "arguments": {"route": ROUTE_RECOMMEND, "selected_sources": ["<tool name>", "..."],
                      "reason": "one short sentence"},
    },
    {
        "action": "direct",
        "description": "the turn needs no product search; run no tool",
        "arguments": {"route": ROUTE_DIRECT, "selected_sources": []},
    },
)


class ToolSelectionError(RuntimeError):
    """A refused selection.  ``code`` is stable; the message never carries provider prose."""

    def __init__(self, message: str, *, code: str) -> None:
        super().__init__(message)
        self.code = code


class ToolOption(BaseModel):
    """One tool the caller is offering this turn."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str = Field(..., min_length=1)
    description: str = Field(..., min_length=1)


class SourceSelection(BaseModel):
    """What the model is allowed to decide.  Nothing else can survive validation."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    route: Literal["recommend", "direct"]
    selected_sources: tuple[str, ...] = ()
    reason: str | None = Field(default=None, max_length=200)


def _normalise_options(allowed_sources: Sequence[ToolOption] | Mapping[str, str]) -> tuple[ToolOption, ...]:
    if isinstance(allowed_sources, Mapping):
        options = [ToolOption(name=str(name), description=str(text))
                   for name, text in allowed_sources.items()]
    else:
        options = [ToolOption.model_validate(option) if not isinstance(option, ToolOption) else option
                   for option in allowed_sources]
    names = [option.name for option in options]
    if len(set(names)) != len(names):
        raise ToolSelectionError("the allowed tool set repeats a name", code=CODE_UNKNOWN_SOURCE)
    return tuple(options)


def validate_selection(
    selection: SourceSelection,
    *,
    allowed: tuple[ToolOption, ...],
    budget: int,
) -> SourceSelection:
    """Trusted validation.  Raises :class:`ToolSelectionError`; never rewrites the choice."""
    allowed_names = {option.name for option in allowed}
    chosen = tuple(selection.selected_sources)

    if len(set(chosen)) != len(chosen):
        raise ToolSelectionError("a tool may be selected at most once", code=CODE_DUPLICATE_SOURCE)
    unknown = [name for name in chosen if name not in allowed_names]
    if unknown:
        raise ToolSelectionError(
            f"the model selected a tool that is not available this turn ({len(unknown)} of them)",
            code=CODE_UNKNOWN_SOURCE,
        )
    if len(chosen) > budget:
        raise ToolSelectionError("the plan exceeds the tool budget", code=CODE_OVER_BUDGET)
    if selection.route == ROUTE_RECOMMEND and not chosen:
        raise ToolSelectionError(
            "a recommendation turn must select at least one tool", code=CODE_NO_SOURCES
        )
    if selection.route == ROUTE_DIRECT and chosen:
        raise ToolSelectionError(
            "a direct answer must not select recommendation tools", code=CODE_DIRECT_WITH_SOURCES
        )
    return selection


class ToolSelectionPlanner:
    """Ask the model for a tool plan and validate it.  Mirrors ``LLMTurnPlanner``'s contract."""

    def __init__(self, client: Any, *, max_attempts: int = 2) -> None:
        if max_attempts < 1:
            raise ValueError("max_attempts must be positive")
        self._client = client
        self._max_attempts = int(max_attempts)
        self.calls = 0

    @property
    def client(self) -> Any:
        """The structured-model client this planner asks.

        Read-only, and exposed for the same reason ``LLMTurnPlanner`` exposes it: the caller that
        wants to report *which* provider answered reads the identity of the object that actually
        made the call, rather than assuming one from configuration.
        """
        return self._client

    def plan(
        self,
        *,
        user_request: str,
        allowed_sources: Sequence[ToolOption] | Mapping[str, str],
        budget: int,
        context: Mapping[str, Any] | None = None,
    ) -> SourceSelection:
        """Return a validated selection, or raise :class:`ToolSelectionError`."""
        if not isinstance(user_request, str) or not user_request.strip():
            raise ToolSelectionError("an empty request cannot select tools", code=CODE_EMPTY_REQUEST)
        if isinstance(budget, bool) or not isinstance(budget, int) or budget < 1:
            raise ToolSelectionError("the tool budget must be a positive int",
                                     code=CODE_INVALID_BUDGET)
        allowed = _normalise_options(allowed_sources)
        if not allowed:
            raise ToolSelectionError("no recommendation tool is available this turn",
                                     code=CODE_NO_TOOLS)

        payload: dict[str, Any] = {
            "user_request": user_request,
            "available_tools": [{"name": option.name, "purpose": option.description}
                                for option in allowed],
            "tool_budget": budget,
        }
        if context:
            payload["context"] = dict(context)

        correction: str | None = None
        last_code = CODE_INVALID_PLAN
        for _attempt in range(self._max_attempts):
            request = ModelRequest(
                system_prompt=_SYSTEM_PROMPT,
                context_payload=payload,
                action_schema=_ACTION_SCHEMA,
                correction=correction,
            )
            self.calls += 1
            try:
                response = self._client.complete(request)
            except ModelCallError as exc:  # provider failure is reported, never absorbed
                raise ToolSelectionError(
                    f"the provider call failed ({getattr(exc, 'code', 'model_error')})",
                    code=str(getattr(exc, "code", "model_error")),
                ) from exc

            raw = getattr(response, "text", "")
            try:
                candidate = SourceSelection.model_validate_json(raw)
                return validate_selection(candidate, allowed=allowed, budget=budget)
            except ToolSelectionError:
                raise
            except Exception as exc:  # noqa: BLE001 - malformed is malformed
                last_code = CODE_INVALID_PLAN
                correction = f"your previous answer was rejected: {type(exc).__name__}"
        raise ToolSelectionError("the model did not return a valid tool plan", code=last_code)


def source_plan_from_selection(selection: SourceSelection) -> tuple[str, ...]:
    """The ordered source plan the existing ``DemoSourcePlanPolicy`` accepts.  No execution."""
    if selection.route != ROUTE_RECOMMEND:
        return ()
    return tuple(selection.selected_sources)


def describe_option(name: str, description: str) -> ToolOption:
    """Convenience constructor used by callers that hold plain strings."""
    return ToolOption(name=name, description=description)


__all__ = [
    "ROUTE_DIRECT",
    "ROUTE_RECOMMEND",
    "SourceSelection",
    "ToolOption",
    "ToolSelectionError",
    "ToolSelectionPlanner",
    "describe_option",
    "source_plan_from_selection",
    "validate_selection",
]

"""Optional LLM decision policy for the browser demo.

A thin layer, not a control plane.  The model proposes **one structured turn plan**; trusted
code validates it and the existing demo path executes it.  Nothing here retrieves, scores,
ranks or reports a metric.

What is reused rather than rebuilt
----------------------------------
* the provider seam: :class:`~recommendation.control.model_client.StructuredModelClient` and
  :func:`~recommendation.control.provider_adapter.build_provider_client`.  No second provider
  abstraction exists.
* the M9 mutation seam: :class:`~recommendation.memory.service.PreferenceMemoryService` keeps
  owning validation and persistence, so an LLM-proposed preference travels the same
  validate-then-apply path as a rule-extracted one.

Authority
---------
The plan schema has fields for a route and for preference values, and **no field at all** for a
product id, a score, evidence, a rank or a mutation result (``extra="forbid"``).  The model
therefore cannot express recommendation truth even if it tries; a prompt injection that adds
``parent_asin`` is a validation error, not a silently ignored key.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Sequence

from pydantic import BaseModel, ConfigDict, Field, field_validator

from recommendation.control.model_client import ModelCallError, ModelRequest
from recommendation.memory.schemas import (
    PreferenceCandidate,
    PreferenceExtraction,
    PreferenceKind,
    PreferenceMode,
    PreferencePolarity,
    PreferenceRemoval,
)

__all__ = [
    "LLM_EXTRACTOR_NAME",
    "LLMPolicyError",
    "LLMPreferenceAdd",
    "LLMTurnPlan",
    "LLMTurnPlanner",
    "LLMPlanExtractor",
    "RouteAction",
    "DECISION_MODES",
    "provider_availability",
]

#: Provenance recorded on every preference this layer proposes.
LLM_EXTRACTOR_NAME = "llm_policy_v1"

#: The two decision modes the browser may select.  ``deterministic`` is the default and needs
#: no credential; ``llm`` requires a configured provider.
DECISION_MODES: tuple[str, ...] = ("deterministic", "llm")


class RouteAction(str, Enum):
    """The routing half of a turn plan. Only the two routes the accepted graph already has."""

    RECOMMEND = "recommend"
    DIRECT = "direct"


class LLMPolicyError(RuntimeError):
    """A provider or plan failure. Carries a stable code, never provider prose."""

    def __init__(self, message: str, *, code: str = "llm_policy_error") -> None:
        super().__init__(message)
        self.code = code


class LLMPreferenceAdd(BaseModel):
    """One preference the model proposes to store.

    ``kind`` is a closed enum owned by trusted code, so the model may label *what* it heard
    ("durable" is a feature) without gaining any ability to define how that kind is matched.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    value: str = Field(..., min_length=1, max_length=120)
    kind: PreferenceKind = PreferenceKind.FEATURE

    @field_validator("value")
    @classmethod
    def _clean_value(cls, value: str) -> str:
        """Collapse whitespace and refuse a blank value before trusted code ever sees it."""
        collapsed = " ".join(str(value).split())
        if not collapsed:
            raise ValueError("a preference value must not be blank")
        if len(collapsed) > 120:
            raise ValueError("a preference value must be at most 120 characters")
        return collapsed


class LLMTurnPlan(BaseModel):
    """The complete set of things the model is allowed to decide in one turn.

    There is deliberately no field for a product id, a score, an evidence verdict, a rank or a
    mutation outcome: those are produced by trusted code, and this schema cannot carry them.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    route: RouteAction
    add: tuple[LLMPreferenceAdd, ...] = ()
    remove: tuple[str, ...] = ()

    @field_validator("remove")
    @classmethod
    def _clean_remove(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        cleaned: list[str] = []
        for raw in values:
            value = " ".join(str(raw).split())
            if not value:
                raise ValueError("a removal value must not be blank")
            if len(value) > 120:
                raise ValueError("a removal value must be at most 120 characters")
            if value.casefold() not in {item.casefold() for item in cleaned}:
                cleaned.append(value)
        return tuple(cleaned)


#: The contract shown to the model.  Short, and it names no product and no catalogue fact.
_SYSTEM_PROMPT = (
    "You are the decision policy of a product recommendation demo.\n"
    "Read the user's message and reply with exactly ONE json object and nothing else.\n"
    "The object has these keys:\n"
    '  "route": "recommend" if the user wants product suggestions, otherwise "direct".\n'
    '  "add": a list of preferences to store, each {"value": "<short term>", "kind": '
    '"feature" | "color" | "material" | "brand" | "category"}. Omit or use [] for none.\n'
    '  "remove": a list of preference values to withdraw. Omit or use [] for none.\n'
    "Rules: use short lowercase terms, one concept per entry. Never name a product, a brand you "
    "were not given, a price, an item identifier or a score. You choose actions only; the "
    "application retrieves products and decides what matches."
)

_ACTION_SCHEMA: tuple[dict[str, Any], ...] = (
    {"action": "recommend", "description": "the user wants product suggestions"},
    {"action": "direct", "description": "no recommendation is needed this turn"},
    {"action": "add_preference", "arguments": {"value": "short term", "kind": "feature"}},
    {"action": "remove_preference", "arguments": {"value": "short term"}},
)


class LLMTurnPlanner:
    """Ask a :class:`StructuredModelClient` for one validated turn plan.

    ``max_attempts`` bounds retries on unparseable output only; a provider failure is raised
    immediately so a failed call can never be mistaken for a decision.
    """

    def __init__(self, client: Any, *, max_attempts: int = 2) -> None:
        self._client = client
        self._max_attempts = max(1, int(max_attempts))
        self.calls = 0

    @property
    def client(self) -> Any:
        return self._client

    def plan(self, user_message: str) -> LLMTurnPlan:
        """Return a validated plan, or raise :class:`LLMPolicyError`."""
        if not isinstance(user_message, str) or not user_message.strip():
            raise LLMPolicyError("an empty turn cannot be planned", code="empty_message")

        correction: str | None = None
        for attempt in range(self._max_attempts):
            request = ModelRequest(
                system_prompt=_SYSTEM_PROMPT,
                context_payload={"user_request": user_message},
                action_schema=_ACTION_SCHEMA,
                correction=correction,
            )
            self.calls += 1
            try:
                response = self._client.complete(request)
            except ModelCallError as exc:
                # reported, never swallowed, and never turned into a decision
                raise LLMPolicyError(
                    f"the provider call failed ({getattr(exc, 'code', 'model_error')})",
                    code=str(getattr(exc, "code", "model_error")),
                ) from exc

            try:
                return LLMTurnPlan.model_validate_json(getattr(response, "text", ""))
            except Exception as exc:  # noqa: BLE001 - any malformed plan is the same outcome
                correction = f"your previous answer was rejected: {type(exc).__name__}"
        raise LLMPolicyError(
            "the model did not return a valid plan", code="invalid_plan"
        )


class LLMPlanExtractor:
    """Adapt a planner to the accepted M9 extractor interface.

    This is the whole integration: the memory service still validates and persists, so an
    LLM-proposed preference is subject to exactly the same checks as a rule-extracted one.
    """

    def __init__(self, planner: LLMTurnPlanner) -> None:
        self._planner = planner

    @property
    def name(self) -> str:
        return LLM_EXTRACTOR_NAME

    def extract(self, user_message: str) -> PreferenceExtraction:
        plan = self._planner.plan(user_message)
        return plan_to_extraction(plan, source_text=user_message)


def plan_to_extraction(plan: LLMTurnPlan, *, source_text: str) -> PreferenceExtraction:
    """Convert a validated plan into the accepted M9 extraction payload."""
    source = source_text if isinstance(source_text, str) and source_text.strip() else "llm turn"
    preferences = tuple(
        PreferenceCandidate(
            kind=item.kind,
            value=item.value,
            polarity=PreferencePolarity.PREFER,
            source_text=source,
            extractor=LLM_EXTRACTOR_NAME,
            mode=PreferenceMode.ADD,
        )
        for item in plan.add
    )
    removals = tuple(
        PreferenceRemoval(
            kind=None,
            value=value,
            source_text=source,
            extractor=LLM_EXTRACTOR_NAME,
        )
        for value in plan.remove
    )
    return PreferenceExtraction(preferences=preferences, removals=removals)


def provider_availability() -> dict[str, Any]:
    """Whether LLM mode can run, from execution state only — never a hardcoded label.

    Returns ``available`` plus a machine-readable ``reason`` when it cannot, so the UI can
    disable the mode with an explanation instead of silently falling back to deterministic.
    """
    from recommendation.control.provider_adapter import provider_settings

    try:
        settings = provider_settings()
    except ModelCallError as exc:
        return {
            "available": False,
            "reason": str(getattr(exc, "code", "not_configured")),
            "provider": None,
            "model": None,
        }
    if not settings.api_key_present:
        # DeepSeek rejects unauthenticated calls, so a keyless provider is not usable mode.
        return {
            "available": False,
            "reason": "missing_api_key",
            "provider": settings.profile.name,
            "model": settings.model,
        }
    return {
        "available": True,
        "reason": None,
        "provider": settings.profile.name,
        "model": settings.model,
    }

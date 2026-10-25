"""Recoverable research clarifications (Phase 6C).

A Phase 6B research question was a question only: the next turn could answer it, but the
answer could not select a concrete next research step.  This module turns the **real research
Observation** into a small, executable option set and projects it into the response.

Nothing here reads a value from the user's message as a product fact.  The message only
selects an option; every fact and every offered attribute comes from the recorded inspection /
comparison observation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from .agent_schemas import (
    AgentResearchActionView,
    AgentResearchClarificationView,
    AgentResearchOptionView,
    AgentResearchView,
)

__all__ = [
    "ResearchResume",
    "build_research_clarification",
    "resolve_research_option",
]

_KNOWN_WORDS = ("fact", "facts", "known", "show", "verified", "already")
_STOP_WORDS = ("stop", "enough", "just recommend", "no more", "don't", "do not", "cancel")


@dataclass(frozen=True)
class ResearchResume:
    """The option selected on a previous research clarification."""

    clarification_id: str
    option_id: str
    action: str
    attributes: tuple[str, ...] = ()
    products: tuple[str, ...] = ()
    known_facts: Mapping[str, Mapping[str, Any]] | None = None
    inspected_products: tuple[str, ...] = ()
    compared: bool = False

    @property
    def is_no_research(self) -> bool:
        return self.action in {"show_known_facts", "stop_research"}

    @property
    def is_compare(self) -> bool:
        return self.action == "compare_attributes"


def _known_from_action(
    action: AgentResearchActionView,
    known: dict[str, dict[str, Any]],
) -> None:
    if action.tool == "inspect_product":
        identity = action.products[0] if action.products else ""
        if not identity:
            return
        for attribute, fact in action.facts.items():
            if isinstance(fact, Mapping) and fact.get("status") == "value":
                known.setdefault(identity, {})[attribute] = fact.get("value")
    elif action.tool == "compare_products":
        for row in action.rows:
            if not isinstance(row, Mapping):
                continue
            attribute = str(row.get("attribute") or "")
            facts = row.get("facts") or {}
            if not isinstance(facts, Mapping):
                continue
            for identity, fact in facts.items():
                if isinstance(fact, Mapping) and fact.get("status") == "value":
                    known.setdefault(str(identity), {})[attribute] = fact.get("value")


def build_research_clarification(
    research: AgentResearchView | None,
    *,
    allow_compare: bool = True,
) -> AgentResearchClarificationView | None:
    """Project a real research response into missing facts and executable options.

    Returns ``None`` when no research action ran.  The options are deliberately narrow:

    * ``compare:<attribute>`` only when an actual comparison row for at least two products
      was marked comparable;
    * ``known_facts`` only when at least one catalogue value was verified;
    * ``stop_research`` always.
    """
    if research is None or not research.executed:
        return None
    actions = tuple(research.actions)
    if not actions:
        return None

    missing: list[str] = []
    products: list[str] = []
    inspected: list[str] = []
    known: dict[str, dict[str, Any]] = {}
    comparable: list[str] = []
    partial: list[str] = []
    compared = False
    for action in actions:
        for value in action.missing_attributes:
            if value not in missing:
                missing.append(value)
        for identity in action.products:
            if identity and identity not in products:
                products.append(identity)
        if action.tool == "inspect_product":
            for identity in action.products:
                if identity and identity not in inspected:
                    inspected.append(identity)
        else:
            compared = True
            for row in action.rows:
                if not isinstance(row, Mapping):
                    continue
                status = row.get("status")
                attribute = str(row.get("attribute") or "")
                if status == "comparable" and attribute and attribute not in comparable:
                    comparable.append(attribute)
                elif status == "partial" and attribute and attribute not in partial:
                    partial.append(attribute)
        _known_from_action(action, known)

    options: list[AgentResearchOptionView] = []
    if allow_compare and compared and len(products) >= 2:
        for attribute in comparable:
            if attribute in missing:
                continue
            options.append(
                AgentResearchOptionView(
                    option_id=f"compare:{attribute}",
                    label=f"Compare {attribute} instead",
                    action="compare_attributes",
                    attributes=(attribute,),
                    products=tuple(products[:5]),
                )
            )
        for attribute in partial:
            if attribute in comparable:
                continue
            options.append(
                AgentResearchOptionView(
                    option_id=f"compare:{attribute}",
                    label=f"Compare {attribute} (partial catalogue coverage)",
                    action="compare_attributes",
                    attributes=(attribute,),
                    products=tuple(products[:5]),
                )
            )
    if known:
        options.append(
            AgentResearchOptionView(
                option_id="known_facts",
                label="Show the facts already verified",
                action="show_known_facts",
                products=tuple(products[:5]),
            )
        )
    options.append(
        AgentResearchOptionView(
            option_id="stop_research",
            label="Stop research and recommend with what is known",
            action="stop_research",
            products=tuple(products[:5]),
        )
    )

    reason = "missing_catalogue_attribute"
    return AgentResearchClarificationView(
        reason=reason,
        missing_attributes=tuple(missing),
        products=tuple(products),
        known_facts=known,
        inspected_products=tuple(inspected),
        compared=bool(compared),
        options=tuple(options),
    )


def resolve_research_option(
    clarification: AgentResearchClarificationView,
    *,
    option_id: str | None,
    message: str,
) -> tuple[AgentResearchOptionView | None, str | None]:
    """Resolve an explicit option id or a short user selection.

    An **explicit** id that is not in the clarification is an error, not a silent fallback.
    When no id is supplied, the message is compared with option ids/labels and the attribute
    names; it never becomes a product fact.
    """
    options = tuple(clarification.options)
    if not options:
        return None, None
    if option_id is not None:
        for option in options:
            if option.option_id == option_id:
                return option, None
        return None, f"unknown clarification option: {option_id!r}"

    text = " ".join(str(message or "").casefold().split())
    if not text:
        return None, None
    for option in options:
        if option.option_id.casefold() in text or option.label.casefold() in text:
            return option, None
    for option in options:
        if any(attribute.casefold() in text for attribute in option.attributes):
            return option, None
    if any(word in text for word in _KNOWN_WORDS):
        for option in options:
            if option.action == "show_known_facts":
                return option, None
    if any(word in text for word in _STOP_WORDS):
        for option in options:
            if option.action == "stop_research":
                return option, None
    return None, None

"""Bounded, observation-driven product research for the shopping turn (Phase 6B).

The base source-plan policy retrieves candidates and then finishes.  This wrapper gives the
loop one extra decision point at that finish: if the *current request* contains a comparison
or a catalogue-attribute question that the candidates cannot answer yet, it proposes one
product-research action.  The next decision is made from the resulting observation, not from
a pre-written sequence.

What is deliberately not here
-----------------------------
No planner model, no new source, no catalogue browser and no ranking change.  The wrapper
only proposes the two read-only Phase 6A actions, and every identity it names comes from
``context.grounded_parent_asins`` / ``context.grounded_candidates``.  The loop's validator
still refuses an unoffered action and the reasoning executor still re-checks run membership.

Bounds
------
* at most two research calls per run (and never more than the loop's remaining tool budget);
* one comparison per run;
* each product is inspected at most once;
* a failed research observation is not retried: the run finishes with the facts it has;
* missing facts are never converted into a satisfaction claim.  If a requested comparison
  attribute is still missing after the bounded inspection, the policy asks the existing
  clarification mechanism rather than inventing a value.
"""

from __future__ import annotations

import re
from typing import Any, Sequence

from recommendation.control import ActionKind, ActionProposal
from recommendation.control.arguments import (
    AskClarificationArguments,
    CompareProductsArguments,
    InspectProductArguments,
)
from .research_clarification import ResearchResume

__all__ = [
    "DynamicProductResearchPolicy",
    "requested_attributes",
    "wants_comparison",
]

#: How many read-only research calls one run may make.  Kept independent of the loop's
#: tool budget: the effective bound is ``min(this, remaining_tool_calls)``.
MAX_RESEARCH_CALLS = 2

#: A comparison request without an explicit attribute list uses this small, factual set.
DEFAULT_COMPARISON_ATTRIBUTES: tuple[str, ...] = (
    "title",
    "price",
    "color",
    "material",
    "description",
)

_ATTRIBUTE_TERMS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("price", ("price", "cost", "cheap", "budget", "expensive")),
    ("color", ("color", "colour")),
    ("material", ("material", "leather", "canvas", "suede", "mesh")),
    ("weight", ("weight", "light", "lightweight", "heavy")),
    ("brand", ("brand", "manufacturer")),
    ("store", ("store", "seller")),
    ("main_category", ("category", "kind of")),
    ("features", ("feature", "waterproof", "feature")),
    ("description", ("description", "details")),
    ("average_rating", ("rating", "rated", "review score")),
    ("title", ("title", "name")),
)

_COMPARISON_PATTERN = re.compile(
    r"\b(compare|comparison|versus|vs\.?|side[- ]by[- ]side|difference|which is better)\b",
    re.IGNORECASE,
)


def wants_comparison(message: str) -> bool:
    """True only when the user explicitly asks for a comparison."""
    return bool(_COMPARISON_PATTERN.search(str(message or "")))


def requested_attributes(message: str) -> tuple[str, ...]:
    """Map explicit attribute words in the request to catalogue attributes.

    This is a vocabulary projection, not semantic understanding: a word that does not occur
    produces no attribute, and an absent fact later stays absent.  Order is the fixed
    vocabulary order, so the action arguments are deterministic.
    """
    text = str(message or "").casefold()
    found: list[str] = []
    for attribute, terms in _ATTRIBUTE_TERMS:
        if any(term in text for term in terms):
            found.append(attribute)
    return tuple(dict.fromkeys(found))


def _observation_kind(context: Any) -> str | None:
    observation = getattr(context, "last_observation", None)
    return str(getattr(observation, "kind", None)) if observation is not None else None


def _observation_field(context: Any, name: str, default: Any = None) -> Any:
    observation = getattr(context, "last_observation", None)
    return getattr(observation, name, default)


def _product_ids(context: Any) -> tuple[str, ...]:
    """The only identities this policy may name, in the run's own order."""
    candidates = tuple(getattr(context, "grounded_parent_asins", ()) or ())
    if candidates:
        return candidates
    return tuple(
        reference.parent_asin
        for reference in (getattr(context, "grounded_candidates", ()) or ())
    )


def _first_uninspected(context: Any, inspected: set[str]) -> str | None:
    candidates = getattr(context, "grounded_candidates", ()) or ()
    # Prefer a reference whose evidence position is "attention"/unresolved; this is the
    # signal the context already publishes, not a catalogue fact.
    for reference in candidates:
        identity = str(getattr(reference, "parent_asin", ""))
        evidence = str(getattr(reference, "evidence", ""))
        eligibility = str(getattr(reference, "eligibility", ""))
        if identity and identity not in inspected and (
            evidence == "attention" or eligibility == "unresolved"
        ):
            return identity
    for identity in _product_ids(context):
        if identity not in inspected:
            return identity
    return None


def _missing_after_comparison(context: Any) -> tuple[tuple[str, ...], str | None]:
    """Return (missing attributes, first product missing one) from a comparison observation."""
    rows = _observation_field(context, "rows", ()) or ()
    missing: list[str] = []
    first_product: str | None = None
    for row in rows:
        status = str(row.get("status") if isinstance(row, dict) else getattr(row, "status", ""))
        if status not in {"partial", "not_comparable"}:
            continue
        attribute = str(row.get("attribute") if isinstance(row, dict) else getattr(row, "attribute", ""))
        facts = row.get("facts") if isinstance(row, dict) else getattr(row, "facts", {})
        for identity, fact in (facts or {}).items():
            fact_status = str(fact.get("status") if isinstance(fact, dict) else getattr(fact, "status", ""))
            if fact_status in {"missing", "unreadable"}:
                missing.append(attribute)
                if first_product is None:
                    first_product = str(identity)
                break
    return tuple(dict.fromkeys(missing)), first_product


def _missing_after_inspection(context: Any) -> tuple[str, ...]:
    values = _observation_field(context, "missing_attributes", ()) or ()
    unreadable = _observation_field(context, "unreadable_attributes", ()) or ()
    return tuple(dict.fromkeys(tuple(str(v) for v in values) + tuple(str(v) for v in unreadable)))


def _clarify_question(products: Sequence[str], attributes: Sequence[str]) -> str:
    names = ", ".join(products)
    attrs = ", ".join(attributes) if attributes else "the requested attribute"
    return (
        f"The catalogue does not state {attrs} for {names}. "
        "Should I proceed with the facts it does provide, or would you like to change the comparison?"
    )


class DynamicProductResearchPolicy:
    """Wrap a policy and add a bounded research decision at its finish point.

    ``inner`` is normally the shopping source-plan policy (optionally wrapped by the
    clarification policy).  The wrapper passes every non-FINISH proposal through unchanged;
    it only intervenes when the inner policy is ready to finish and the request plus the most
    recent observation justify exactly one more read.
    """

    name = "demo-source-plan-with-dynamic-research"

    def __init__(
        self,
        inner: Any,
        *,
        max_research_calls: int = MAX_RESEARCH_CALLS,
        resume: ResearchResume | None = None,
        research_budget_remaining: int = 4,
        clarification_budget_remaining: int = 2,
    ) -> None:
        if not callable(getattr(inner, "choose", None)):
            raise TypeError("DynamicProductResearchPolicy needs an inner policy")
        if not 1 <= int(max_research_calls) <= MAX_RESEARCH_CALLS:
            raise ValueError(
                f"max_research_calls must be 1..{MAX_RESEARCH_CALLS}"
            )
        self._inner = inner
        self._max_calls = min(int(max_research_calls), max(0, int(research_budget_remaining)))
        self._clarification_budget = max(0, int(clarification_budget_remaining))
        self._resume = resume
        self._inspected: set[str] = set()
        self._compared = False
        self._research_calls = 0
        self._asked = False
        self._failed = False
        self._last_research_product: str | None = None
        self._resume_compare_pending = False
        self._no_research = False

    @property
    def inner(self) -> Any:
        return self._inner

    @property
    def research_calls(self) -> int:
        return self._research_calls

    def choose(self, context: Any) -> ActionProposal:
        if int(getattr(context, "step_index", 0) or 0) == 0:
            self._inspected = (
                set(self._resume.inspected_products) if self._resume is not None else set()
            )
            self._compared = bool(self._resume.compared) if self._resume is not None else False
            self._research_calls = 0
            self._asked = False
            self._failed = False
            self._last_research_product = None
            self._no_research = bool(
                self._resume is not None and self._resume.is_no_research
            )
            self._resume_compare_pending = bool(
                self._resume is not None and self._resume.is_compare
            )

        proposal = self._inner.choose(context)
        if self._no_research:
            return proposal
        if self._resume_compare_pending and proposal.action in (
            ActionKind.FINISH,
            ActionKind.ASK_CLARIFICATION,
        ):
            products = tuple(
                identity for identity in (self._resume.products if self._resume else ())
                if identity in set(_product_ids(context))
            )
            if len(products) < 2:
                products = tuple(_product_ids(context)[:2])
            if (
                self._can_research(context)
                and len(products) >= 2
                and context.action_available(ActionKind.COMPARE_PRODUCTS)
            ):
                attributes = tuple(self._resume.attributes if self._resume else ())
                self._resume_compare_pending = False
                self._compared = True
                self._research_calls += 1
                return ActionProposal(
                    action=ActionKind.COMPARE_PRODUCTS,
                    rationale=(
                        "the user selected a different comparable attribute; compare only "
                        "the still-authorized products without repeating the missing field"
                    ),
                    arguments=CompareProductsArguments(
                        parent_asins=products,
                        attributes=attributes or DEFAULT_COMPARISON_ATTRIBUTES,
                    ),
                )
            self._resume_compare_pending = False
        if proposal.action is ActionKind.ASK_CLARIFICATION:
            # A pending hard-constraint question must not block a cheap, read-only inspection
            # that could resolve it.  Let the existing clarification fall through if no
            # request-relevant research remains.
            if int(getattr(context, "step_index", 0) or 0) > 0:
                early = self._early_research(context)
                if early is not None:
                    return early
            if not self._can_clarify(context):
                return ActionProposal(action=ActionKind.FINISH)
            return proposal
        if proposal.action is not ActionKind.FINISH:
            return proposal

        # A research failure is an observation, not an invitation to repeat the same call.
        if _observation_kind(context) == "product_research_failure":
            self._failed = True
            return proposal

        message = str(getattr(context, "user_request", "") or "")
        requested = requested_attributes(message)
        observation_kind = _observation_kind(context)

        # A failed fact check is the one case where clarification must be reached even after
        # the research-call cap: the absence is now verified, and inventing a value is worse
        # than asking.
        if observation_kind == "product_inspection":
            missing = _missing_after_inspection(context)
            if (
                missing
                and not self._asked
                and self._can_clarify(context)
                and context.action_available(ActionKind.ASK_CLARIFICATION)
            ):
                self._asked = True
                products = tuple(
                    p for p in _product_ids(context) if p == self._last_research_product
                )
                return self._clarify(products or tuple(_product_ids(context)[:1]), missing)
            return proposal

        if observation_kind == "product_comparison":
            missing, missing_product = _missing_after_comparison(context)
            if missing:
                target = missing_product
                if target and target not in self._inspected and self._inspect(
                    context, target, missing
                ):
                    return self._inspect_proposal(target, missing)
                if (
                    not self._asked
                    and self._can_clarify(context)
                    and context.action_available(ActionKind.ASK_CLARIFICATION)
                ):
                    self._asked = True
                    return self._clarify(tuple(_product_ids(context)[:2]), missing)
            return proposal

        if not self._can_research(context):
            return proposal

        # 1) The user explicitly asked to compare.  Do it once, over the run's own candidates.
        if wants_comparison(message) and not self._compared:
            products = _product_ids(context)[:2]
            if len(products) >= 2 and context.action_available(ActionKind.COMPARE_PRODUCTS):
                self._compared = True
                self._research_calls += 1
                attributes = requested or DEFAULT_COMPARISON_ATTRIBUTES
                return ActionProposal(
                    action=ActionKind.COMPARE_PRODUCTS,
                    rationale=(
                        "the user explicitly requested a comparison; compare the run's "
                        "first two grounded candidates with the requested attributes"
                    ),
                    arguments=CompareProductsArguments(
                        parent_asins=tuple(products), attributes=tuple(attributes)
                    ),
                )

        # 2) The request names a catalogue attribute, but no comparison was requested.  Inspect
        #    one under-evidenced candidate; then decide from what comes back.
        if requested and not self._inspected:
            target = _first_uninspected(context, self._inspected)
            if target and self._inspect(context, target, requested):
                return self._inspect_proposal(target, requested)

        return proposal

    # -- helpers ----------------------------------------------------------- #

    def _early_research(self, context: Any) -> ActionProposal | None:
        """Research that should precede a clarification question, if any remains.

        This is deliberately limited to the two request-driven signals: an explicit comparison
        and a named attribute.  It never turns an unrelated unresolved constraint into a
        catalogue read.
        """
        if not self._can_research(context):
            return None
        message = str(getattr(context, "user_request", "") or "")
        requested = requested_attributes(message)
        if wants_comparison(message) and not self._compared:
            products = _product_ids(context)[:2]
            if len(products) >= 2 and context.action_available(ActionKind.COMPARE_PRODUCTS):
                self._compared = True
                self._research_calls += 1
                return ActionProposal(
                    action=ActionKind.COMPARE_PRODUCTS,
                    rationale=(
                        "the user explicitly requested a comparison; compare the run's first "
                        "two grounded candidates before asking a constraint question"
                    ),
                    arguments=CompareProductsArguments(
                        parent_asins=tuple(products),
                        attributes=tuple(requested or DEFAULT_COMPARISON_ATTRIBUTES),
                    ),
                )
        if _observation_kind(context) == "product_comparison":
            missing, missing_product = _missing_after_comparison(context)
            if missing and missing_product and missing_product not in self._inspected:
                if self._inspect(context, missing_product, missing):
                    return self._inspect_proposal(missing_product, missing)
        if requested and not self._inspected:
            target = _first_uninspected(context, self._inspected)
            if target and self._inspect(context, target, requested):
                return self._inspect_proposal(target, requested)
        return None

    def _can_clarify(self, context: Any) -> bool:
        return self._clarification_budget > 0 and not self._asked

    def _can_research(self, context: Any) -> bool:
        if self._failed or self._research_calls >= self._max_calls:
            return False
        remaining = int(getattr(context, "remaining_tool_calls", 0) or 0)
        return remaining > 0

    def _inspect(self, context: Any, target: str, attributes: Sequence[str]) -> bool:
        if not context.action_available(ActionKind.INSPECT_PRODUCT):
            return False
        if target in self._inspected or self._research_calls >= self._max_calls:
            return False
        remaining = int(getattr(context, "remaining_tool_calls", 0) or 0)
        if remaining <= 0:
            return False
        self._inspected.add(target)
        self._research_calls += 1
        self._last_research_product = target
        return True

    def _inspect_proposal(self, target: str, attributes: Sequence[str]) -> ActionProposal:
        return ActionProposal(
            action=ActionKind.INSPECT_PRODUCT,
            rationale=(
                "the request names a catalogue attribute the current candidates have not "
                "provided; read the missing fact before deciding"
            ),
            arguments=InspectProductArguments(
                parent_asin=target, attributes=tuple(attributes)
            ),
        )

    def _clarify(self, products: Sequence[str], attributes: Sequence[str]) -> ActionProposal:
        return ActionProposal(
            action=ActionKind.ASK_CLARIFICATION,
            rationale=(
                "the requested attribute is still unverified after the bounded research; "
                "ask instead of presenting an unsupported claim"
            ),
            arguments=AskClarificationArguments(
                question=_clarify_question(products, attributes),
                blocks="candidate_set",
            ),
        )

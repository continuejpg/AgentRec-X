"""The trusted reasoning executor (AgentRec-X 2.0-alpha Stage 4 wiring).

`GroundedReasoner` is the component that reads catalogue facts; this module is what turns a
*validated action* into a call on it and an *observation* back.  Keeping the two apart is
deliberate:

* the reasoner knows nothing about actions, policies or observations - it is a pure,
  read-only view of the catalogue;
* the executor knows nothing about how a fact is read - it maps action to capability, checks
  the identities it was given, and shapes the result into an observation.

Why this is the last step of the reasoning plane
------------------------------------------------
A capability that exists but cannot be invoked is not a capability a policy can use.  Until
this module existed, `COMPARE` and its siblings were declared members of the action enum with
argument contracts, but no executor mapped them onto the reasoner, so a proposal naming one
was refused.  This module closes that gap - and nothing more.  It does not add a planner, a
critic or a second agent.

The one rule that matters here
------------------------------
**Reasoning actions are read-only, and the executor enforces it structurally.**  Every action
this module accepts is in :data:`~recommendation.control.schemas.READ_ONLY_ACTIONS`, and the
executor has no access to the candidate plane, the ledger or the memory service.  It cannot
add a candidate even if a policy asks it to; it can only read facts about candidates the run
already holds, which is what makes "reasoning cannot widen the candidate set" true by
construction rather than by review.

Identity handling
-----------------
A reasoning action names ``parent_asins``.  The executor grounds every one against the
catalogue and reports the ones it cannot find as ``unknown`` rather than silently dropping
them: a policy that asked about a product needs to know the product was not found, and a
dropped identity would look like a successful, complete comparison.
"""

from __future__ import annotations

from typing import Any

from .grounded_reasoning import (
    UNKNOWN,
    CompatibilityRequirement,
    CompatibilityVerdict,
    GroundedReasoner,
)
from .schemas import (
    ActionKind,
    BundleObservation,
    ComparisonObservation,
    CompatibilityObservation,
    DetailObservation,
    PolicyActionError,
    READ_ONLY_ACTIONS,
    TradeOffObservation,
    ValidatedAction,
)

__all__ = [
    "REASONING_ACTIONS",
    "ReasoningExecutor",
]

#: The actions this executor is authorised to run.  Derived from the control-plane contract
#: rather than restated, so a new read-only action cannot be added to the enum without this
#: executor's authority set moving with it.
REASONING_ACTIONS: frozenset[ActionKind] = frozenset(READ_ONLY_ACTIONS)

#: Attributes copied into a details observation.  A whitelist, so a future catalogue field
#: cannot leak into a policy-visible payload simply by existing.
_DETAIL_ATTRIBUTES: tuple[str, ...] = (
    "title",
    "brand",
    "color",
    "material",
    "main_category",
    "categories",
    "features",
    "price_text",
    "weight_text",
    "average_rating",
)


class ReasoningExecutor:
    """Execute one read-only reasoning action and return a policy-visible observation.

    Parameters
    ----------
    reasoner:
        The :class:`~recommendation.control.grounded_reasoning.GroundedReasoner` over the
        trusted catalogue.  Required: without a catalogue there are no facts, and the executor
        will not answer a fact question from anywhere else.
    """

    def __init__(self, reasoner: GroundedReasoner) -> None:
        if not isinstance(reasoner, GroundedReasoner):
            raise PolicyActionError(
                "reasoning executor needs a GroundedReasoner over the trusted catalogue"
            )
        self._reasoner = reasoner
        self._executed = 0

    # -- metadata ---------------------------------------------------------- #

    @property
    def reasoner(self) -> GroundedReasoner:
        """The reasoner this executor drives."""
        return self._reasoner

    @property
    def executed_count(self) -> int:
        """How many reasoning actions have run (diagnostics only)."""
        return self._executed

    def handles(self, action: ActionKind) -> bool:
        """True when this executor is authorised to run ``action``."""
        return action in REASONING_ACTIONS

    def available_actions(self) -> tuple[ActionKind, ...]:
        """The reasoning actions this deployment offers a policy."""
        return tuple(sorted(REASONING_ACTIONS, key=lambda a: a.value))

    # -- execution --------------------------------------------------------- #

    def execute(self, action: ValidatedAction) -> Any:
        """Run one reasoning action and return its observation.

        Raises
        ------
        PolicyActionError
            The action is not a read-only reasoning action.  The executor refuses rather than
            guessing, which is what keeps a candidate-producing action from being routed here.
        """
        if action.action not in REASONING_ACTIONS:
            raise PolicyActionError(
                f"'{action.action.value}' is not a read-only reasoning action; "
                f"allowed: {', '.join(a.value for a in self.available_actions())}"
            )
        self._executed += 1
        arguments = action.arguments

        if action.action is ActionKind.GET_DETAILS:
            return self._details(action, arguments)
        if action.action is ActionKind.COMPARE:
            return self._compare(action, arguments)
        if action.action is ActionKind.TRADE_OFF:
            return self._trade_off(action, arguments)
        if action.action is ActionKind.CHECK_COMPATIBILITY:
            return self._compatibility(action, arguments)
        if action.action is ActionKind.BUNDLE:
            return self._bundle(action, arguments)
        # ActionKind.VERIFY: re-check the run's own candidate set against its constraints.
        return self._verify(action, arguments)

    # -- per-action implementations ---------------------------------------- #

    def _requested(self, arguments: Any) -> tuple[tuple[str, ...], tuple[str, ...]]:
        """Split requested identities into grounded and unknown, preserving order.

        An action whose arguments carry no identities (``VERIFY`` takes none) yields two empty
        tuples rather than an error: "nothing to read" is a legitimate answer, and inventing a
        candidate set for it would be the fabrication this plane forbids.
        """
        requested = tuple(getattr(arguments, "parent_asins", ()) or ())
        known = tuple(a for a in requested if self._reasoner.facts(a) is not None)
        unknown = tuple(a for a in requested if self._reasoner.facts(a) is None)
        return known, unknown

    def _details(self, action: ValidatedAction, arguments: Any) -> DetailObservation:
        """Return whitelisted grounded facts for the requested identities."""
        known, unknown = self._requested(arguments)
        facts: dict[str, dict[str, Any]] = {}
        for parent_asin in known:
            profile = self._reasoner.facts(parent_asin)
            if profile is None:  # pragma: no cover - guarded by _requested
                continue
            facts[parent_asin] = {
                attribute: self._read(profile, attribute)
                for attribute in _DETAIL_ATTRIBUTES
            }
        return DetailObservation(
            action_id=action.action_id,
            step_index=action.step_index,
            action=action.action,
            verification_status="verified",
            status="ok" if facts else "empty",
            facts=facts,
            requested=tuple(getattr(arguments, "parent_asins", ()) or ()),
            unknown=unknown,
            verification_note=(
                f"{len(facts)} grounded, {len(unknown)} not in the catalogue"
            ),
        )

    def _compare(self, action: ValidatedAction, arguments: Any) -> ComparisonObservation:
        """Compare grounded products attribute by attribute."""
        known, unknown = self._requested(arguments)
        attributes = tuple(getattr(arguments, "attributes", ()) or ())
        result = self._reasoner.compare(known, attributes=attributes)
        return ComparisonObservation(
            action_id=action.action_id,
            step_index=action.step_index,
            action=action.action,
            verification_status="verified",
            status="ok" if result.rows else "empty",
            compared=result.parent_asins,
            rows=result.rows,
            unsupported_attributes=result.unknown_attributes,
            verification_note=(
                f"{len(result.comparable_attributes)} of {len(result.rows)} attributes "
                f"comparable; {len(unknown)} identity/identities not in the catalogue"
            ),
        )

    def _trade_off(self, action: ValidatedAction, arguments: Any) -> TradeOffObservation:
        """Order grounded products by a stated priority."""
        known, unknown = self._requested(arguments)
        priority = str(getattr(arguments, "priority", "") or "")
        result = self._reasoner.trade_off(known, priority=priority)
        return TradeOffObservation(
            action_id=action.action_id,
            step_index=action.step_index,
            action=action.action,
            verification_status="verified",
            status="ok" if result["order"] else "empty",
            priority=priority,
            attribute=result["attribute"],
            supported=bool(result["supported"]),
            order=tuple(result["order"]),
            unknown=tuple(result["unknown"]) + unknown,
            reason=result["reason"],
            verification_note=(
                f"ordered {len(result['order'])} product(s) by "
                f"{result['attribute'] or 'an unsupported priority'}"
            ),
        )

    def _compatibility(
        self, action: ValidatedAction, arguments: Any
    ) -> CompatibilityObservation:
        """Check a requirement against grounded products."""
        known, unknown = self._requested(arguments)
        attribute = str(getattr(arguments, "requirement", "") or "")
        # The argument contract carries the requirement as free text of the form
        # "attribute: value"; the split is structural, not semantic interpretation.
        name, _, value = attribute.partition(":")
        requirement = CompatibilityRequirement(name.strip() or attribute, value.strip())
        verdict = self._reasoner.check_compatibility(known, requirement=requirement)
        performed = self._checks_performed(requirement)
        return CompatibilityObservation(
            action_id=action.action_id,
            step_index=action.step_index,
            action=action.action,
            verification_status="verified",
            status="ok" if known else "empty",
            verdict=verdict.value,
            requirement_attribute=requirement.attribute,
            requirement_value=requirement.value,
            members=known,
            checks_performed=performed,
            verification_note=(
                "no check could be performed against this catalogue"
                if verdict is CompatibilityVerdict.UNKNOWN
                else f"checked: {', '.join(performed) or 'none'}"
                + (f"; {len(unknown)} identity/identities not in the catalogue" if unknown else "")
            ),
        )

    def _checks_performed(self, requirement: CompatibilityRequirement) -> tuple[str, ...]:
        """Return which deterministic checks the catalogue can actually support.

        Reported so a reader can tell "we checked and it fits" from "we could not check".
        An empty tuple is the honest answer for an interface or fitment requirement.
        """
        attribute = requirement.attribute.strip().casefold()
        mapping = {
            "brand": "brand_match",
            "category": "category_membership",
            "categories": "category_membership",
            "color": "color_match",
            "colour": "color_match",
        }
        check = mapping.get(attribute)
        return (check,) if check else ()

    def _bundle(self, action: ValidatedAction, arguments: Any) -> BundleObservation:
        """Report what a set of grounded products supports as a bundle."""
        known, unknown = self._requested(arguments)
        result = self._reasoner.bundle(known)
        return BundleObservation(
            action_id=action.action_id,
            step_index=action.step_index,
            action=action.action,
            verification_status="verified",
            status="ok" if result["member_count"] else "empty",
            members=tuple(m["parent_asin"] for m in result["members"]),
            member_count=int(result["member_count"]),
            total_price=dict(result["total_price"]),
            total_weight_kg=dict(result["total_weight_kg"]),
            shared_categories=tuple(result["shared_categories"]),
            verification_note=(
                f"{result['member_count']} grounded member(s)"
                + (f"; {len(unknown)} not in the catalogue" if unknown else "")
            ),
        )

    def _verify(self, action: ValidatedAction, arguments: Any) -> DetailObservation:
        """Re-read grounded facts for the identities supplied, for a constraint re-check.

        ``VERIFY`` is the re-check action: it answers "what does the catalogue now say about
        these products", so a policy can confirm a hard constraint before proposing FINISH.  It
        is a **read, never an approval** - completion is authorised by ``CompletionGuard``
        alone, and this observation carries no completion signal.
        """
        return self._details(action, arguments)

    # -- helpers ----------------------------------------------------------- #

    @staticmethod
    def _read(profile: Any, attribute: str) -> str:
        """Read one whitelisted attribute as a display string, or ``UNKNOWN``."""
        if attribute == "price_text":
            return profile.price_text()
        if attribute == "weight_text":
            return profile.weight_text()
        if attribute == "average_rating":
            rating = profile.average_rating
            return UNKNOWN if rating is None else f"{rating:.2f}"
        value = getattr(profile, attribute, None)
        if value is None:
            return UNKNOWN
        if isinstance(value, (tuple, list)):
            return " | ".join(str(v) for v in value) if value else UNKNOWN
        return str(value)

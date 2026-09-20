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
A reasoning action names ``parent_asins``.  Before anything is read, the executor checks every
one of them against the **run's grounded candidate allowlist**, which the loop supplies on each
call.  An identity that is not a grounded candidate of this run is a refusal - not a lookup -
and it fails the whole action rather than the individual identity, so a partially authorised
request cannot be partially answered.

That is the boundary this plane rests on, and it is the reason catalogue membership is not
consulted first: *existing in the catalogue* is what makes a fact groundable, and it is not
what makes a candidate the run's.  Only after membership is established does the reasoner
ground each identity against the catalogue, where a candidate whose fact is absent reports
``UNKNOWN`` rather than a fabricated value.
"""

from __future__ import annotations

from typing import Any, Sequence

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

    Run-scoped identity authority
    -----------------------------
    Catalogue membership authorizes **grounding**, not **attention**.  A product existing in
    the trusted catalogue says the system could read it; it says nothing about whether *this
    run* retrieved it, and a reasoning action exists to inspect the run's own candidates.

    :meth:`execute` therefore takes ``authorized_candidates`` - the run's live grounded
    allowlist - and every identity a reasoning action names must be a member.  The check runs
    **before** the catalogue is consulted, so an unauthorized reference never causes a read.
    Catalogue membership is then applied as a *second*, narrower check inside the reasoner,
    where a missing fact still reports ``UNKNOWN`` rather than being fabricated.
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

    def execute(
        self, action: ValidatedAction, *, authorized_candidates: Sequence[str] | None = None
    ) -> Any:
        """Run one reasoning action and return its observation.

        Parameters
        ----------
        action:
            The validated action to run.
        authorized_candidates:
            The identities this run may reason about - its live grounded allowlist.  **Mandatory
            in practice**: the loop always supplies it, and passing ``None`` is only for a
            caller that has no run to scope to (a direct executor test).  A caller that omits
            it is asserting there is no run boundary to enforce, which is why the omission is
            explicit rather than a permissive default.

        Raises
        ------
        PolicyActionError
            The action is not a read-only reasoning action, or it names an identity that is not
            a grounded candidate of this run.  Both are refusals rather than repairs: the
            executor will not substitute an identity, look one up, or drop an unauthorized
            member of a multi-identity action.
        """
        if action.action not in REASONING_ACTIONS:
            raise PolicyActionError(
                f"'{action.action.value}' is not a read-only reasoning action; "
                f"allowed: {', '.join(a.value for a in self.available_actions())}"
            )
        arguments = action.arguments
        # Authority before evidence: the run-membership check happens first, so an
        # unauthorized reference is refused without the catalogue ever being consulted for it.
        self._require_run_membership(arguments, authorized_candidates)
        self._executed += 1

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

    def _require_run_membership(
        self, arguments: Any, authorized_candidates: Sequence[str] | None
    ) -> None:
        """Refuse the whole action when any identity it names is not a run candidate.

        The rule is all-or-nothing on purpose.  A ``COMPARE(A, X)`` where ``X`` is not in the
        run is not "a comparison of A with something unknown" - it is a request the run has no
        authority to answer, and answering it partially would leak which of the named
        identities the run happens to hold.  Refusing the action also keeps the failure
        recoverable: the policy gets a stable code and can propose a legal target next step.

        When ``authorized_candidates`` is ``None`` the caller has asserted there is no run
        scope to enforce, and this returns without checking.  The loop never does that.
        """
        if authorized_candidates is None:
            return
        allowed = frozenset(authorized_candidates)
        if not allowed:
            # An empty allowlist is a real state: a run that holds no grounded candidate.  Any
            # identity is then unauthorized, and saying so is more honest than treating "no
            # candidates yet" as "no restriction".
            requested = self._requested_identities(arguments)
            if requested:
                raise PolicyActionError(
                    "the action names an identity but this run holds no grounded candidate",
                    code="candidate_not_in_run",
                )
            return
        unauthorized = tuple(
            identity
            for identity in self._requested_identities(arguments)
            if identity not in allowed
        )
        if unauthorized:
            raise PolicyActionError(
                f"{len(unauthorized)} requested identity/identities are not grounded "
                f"candidates of this run",
                code="candidate_not_in_run",
            )

    @staticmethod
    def _requested_identities(arguments: Any) -> tuple[str, ...]:
        """The identities an action's arguments name, preserving order and duplicates.

        Read from the declared argument model rather than by guessing at attributes, so an
        action that grows an identity field cannot slip past the membership check.
        """
        return tuple(str(a) for a in (getattr(arguments, "parent_asins", ()) or ()))

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
        # The aggregate verdict alone cannot say *which* candidate failed, so the
        # per-candidate reports it was derived from are carried through instead of being
        # discarded.  Empty when the requirement maps onto no deterministic check, which is
        # the honest answer rather than a fabricated per-candidate verdict.
        assessments = self._assessments(known, requirement)
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
            assessments=assessments,
            verification_note=(
                "no check could be performed against this catalogue"
                if verdict is CompatibilityVerdict.UNKNOWN
                else f"checked: {', '.join(performed) or 'none'}"
                + (f"; {len(unknown)} identity/identities not in the catalogue" if unknown else "")
            ),
        )

    def _assessments(
        self, known: tuple[str, ...], requirement: CompatibilityRequirement
    ) -> tuple[dict[str, Any], ...]:
        """Per-candidate verdicts behind a compatibility aggregate, in ``known`` order.

        Reuses the reasoner's own constraint check, so the per-candidate verdict and the
        aggregate can never disagree: the aggregate is defined as the fold of exactly these
        verdicts.  A requirement naming a dimension this catalogue cannot check yields no
        per-candidate entries - there is no verdict to report, and inventing ``satisfied``
        for one would be the fabrication the reasoning plane exists to prevent.
        """
        kind = self._constraint_kind(requirement.attribute)
        if kind is None:
            return ()
        return tuple(
            self._reasoner.check_constraint(
                parent_asin, kind=kind, expected=requirement.value
            ).as_dict()
            for parent_asin in known
        )

    @staticmethod
    def _constraint_kind(attribute: str) -> Any:
        """The constraint dimension a compatibility attribute names, or ``None``.

        Mirrors the mapping :meth:`GroundedReasoner.check_compatibility` uses, kept explicit
        so a newly checkable dimension cannot silently produce an aggregate without a
        per-candidate breakdown.
        """
        from .grounded_reasoning import ConstraintKind

        return {
            "brand": ConstraintKind.BRAND,
            "category": ConstraintKind.CATEGORY,
            "categories": ConstraintKind.CATEGORY,
            "color": ConstraintKind.COLOR,
            "colour": ConstraintKind.COLOR,
        }.get(attribute.strip().casefold())

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

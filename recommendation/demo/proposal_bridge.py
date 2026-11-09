"""Minimal, validated LLM operation -> M5 adapter.

The model output is never allowed to write M5 directly.  This module validates the
versioned operation schema against the current user message, converts only explicit
operations to the existing M5 event vocabulary, and returns diagnostics.  The service
passes the result to ``_integrate_state_turn`` before R1/M6.1 runs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

@dataclass(frozen=True)
class ProposalActions:
    preferences: tuple[Any, ...] = ()
    constraints: tuple[Any, ...] = ()
    retractions: tuple[Any, ...] = ()
    role_target: Any | None = None
    accepted: tuple[dict[str, Any], ...] = ()
    rejected: tuple[dict[str, Any], ...] = ()
    errors: tuple[str, ...] = ()
    noop_kinds: tuple[str, ...] = ()

    @property
    def valid(self) -> bool:
        return not self.errors


def _pref_direction(value: Any) -> Any | None:
    from recommendation.state.schemas import PreferenceDirection

    text = str(value or "").strip().lower()
    if text == "minimize":
        return PreferenceDirection.MINIMIZE
    if text == "maximize":
        return PreferenceDirection.MAXIMIZE
    return None


def _pref_kind(attribute: str) -> Any | None:
    from recommendation.state.schemas import PreferenceKind

    return {
        "weight": PreferenceKind.FEATURE,
        "color": PreferenceKind.COLOR,
        "material": PreferenceKind.MATERIAL,
        "brand": PreferenceKind.BRAND,
        "feature": PreferenceKind.FEATURE,
    }.get(attribute)


def _constraint_kind(attribute: str) -> Any | None:
    from recommendation.state.schemas import PreferenceKind

    return {
        "price_max": PreferenceKind.PRICE_MAX,
        "price_min": PreferenceKind.PRICE_MIN,
    }.get(attribute)


def map_proposal(payload: Any, user_message: str) -> ProposalActions:
    """Validate and map one operation payload; no M5 write happens here."""
    from training.agent_sft.schema import validate_proposal

    errors = list(validate_proposal(payload, user_message))
    if errors:
        return ProposalActions(errors=tuple(errors))
    from recommendation.state import shopping_adapter
    from recommendation.state.schemas import PreferenceKind

    preferences: list[Any] = []
    constraints: list[Any] = []
    retractions: list[Any] = []
    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    noop_kinds: list[str] = []
    role_target = None
    for index, operation in enumerate(payload.get("operations", []) or []):
        op = str(operation.get("op", ""))
        attribute = str(operation.get("attribute") or "")
        value = operation.get("value")
        direction = operation.get("direction")
        evidence = str(operation.get("evidence") or "")
        source = {"proposal_index": index, "op": op, "evidence": evidence}
        if op in {"set_goal", "update_goal"}:
            if attribute == "product_type" and value is not None:
                text = str(value).strip().lower()
                from .product_role import ProductTarget, TargetRole

                accessory_terms = ("accessory", "mouthpiece", "cartridge", "replacement", "spare")
                role = TargetRole.ACCESSORY if any(term in text for term in accessory_terms) else TargetRole.MAIN
                role_target = ProductTarget(None, role)
                preferences.append(shopping_adapter.ExplicitPreference(
                    kind=PreferenceKind.FEATURE,
                    value=f"goal:{text}",
                    source_text=evidence,
                ))
            elif attribute == "role" and value is not None:
                from .product_role import ProductTarget, TargetRole

                try:
                    role = TargetRole(str(value))
                except ValueError:
                    rejected.append({**source, "reason": f"unknown role {value!r}"})
                    continue
                role_target = ProductTarget(None, role)
                # A role-only goal has no product type; the current task target supplies it.
                preferences.append(shopping_adapter.ExplicitPreference(
                    kind=PreferenceKind.FEATURE,
                    value=f"role:{role.value}",
                    source_text=evidence,
                ))
            else:
                rejected.append({**source, "reason": "goal has no supported product_type/role"})
                continue
        elif op in {"set_constraint", "update_constraint"}:
            kind = _constraint_kind(attribute)
            if kind is None or value is None:
                rejected.append({**source, "reason": "constraint requires price_max/price_min and value"})
                continue
            constraints.append(shopping_adapter.ExplicitConstraint(
                kind=kind, value=str(value), source_text=evidence
            ))
        elif op == "retract_constraint":
            kind = _constraint_kind(attribute)
            if kind is None:
                rejected.append({**source, "reason": "retract_constraint requires price_max/price_min"})
                continue
            retractions.append(shopping_adapter.ExplicitRetraction(kind=kind, source_text=evidence))
        elif op in {"add_preference", "update_preference"}:
            kind = _pref_kind(attribute)
            if kind is None or value is None and direction is None:
                rejected.append({**source, "reason": "preference has unsupported attribute/value"})
                continue
            preferences.append(shopping_adapter.ExplicitPreference(
                kind=kind,
                value=str(value) if value is not None else attribute,
                source_text=evidence,
                direction=_pref_direction(direction),
                replace_same_kind=op == "update_preference",
            ))
        elif op == "retract_preference":
            kind = _pref_kind(attribute)
            if kind is None:
                rejected.append({**source, "reason": "retract_preference has unsupported attribute"})
                continue
            retractions.append(shopping_adapter.ExplicitRetraction(kind=kind, source_text=evidence))
        elif op in {"clarify", "noop", "abstain", "resolve_reference"}:
            noop_kinds.append(op)
        else:
            rejected.append({**source, "reason": f"unsupported op {op!r}"})
            continue
        accepted.append(dict(operation))
    return ProposalActions(
        preferences=tuple(preferences),
        constraints=tuple(constraints),
        retractions=tuple(retractions),
        role_target=role_target,
        accepted=tuple(accepted),
        rejected=tuple(rejected),
        noop_kinds=tuple(noop_kinds),
    )


def validate_and_map(text: str, user_message: str) -> tuple[ProposalActions, Any | None]:
    from training.agent_sft.schema import parse_and_validate

    payload, errors = parse_and_validate(text, user_message)
    if payload is None:
        return ProposalActions(errors=tuple(errors)), None
    actions = map_proposal(payload, user_message)
    return actions, payload

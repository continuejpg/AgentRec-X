"""Versioned preference-operation proposal schema and independent validator.

The model is only allowed to emit operations.  Product identities (ASINs) are never a
legal proposal field, and every operation must quote a substring of the current user
message as evidence.  M5 lifecycle/replay remains the authority for changing state.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Mapping

SCHEMA_VERSION = "agentrecx.preference_operation.v1"
SCHEMA_PATH = Path(__file__).with_name("schema") / "preference_operation.schema.json"
_ASIN = re.compile(r"\bB[0-9A-Z]{9}\b")
_OPS = {
    "set_goal",
    "update_goal",
    "add_preference",
    "update_preference",
    "retract_preference",
    "set_constraint",
    "update_constraint",
    "retract_constraint",
    "resolve_reference",
    "clarify",
    "noop",
    "abstain",
}
_ATTRS = {
    "product_type",
    "role",
    "price_max",
    "price_min",
    "weight",
    "color",
    "material",
    "brand",
    "feature",
}
_PREF_ATTRS = {"weight", "color", "material", "brand", "feature"}
_CONSTRAINT_ATTRS = {"price_max", "price_min"}
_REFERENCES = {f"previous_candidate_{index}" for index in range(1, 6)} | {"previous_state"}
_REASONS = {
    "no_change",
    "declined_preference",
    "ambiguous_goal",
    "unsupported_request",
    "cross_turn_reference",
}


def schema_payload() -> dict[str, Any]:
    return json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))


def _nonempty_text(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def validate_proposal(payload: Any, user_message: str) -> list[str]:
    """Return human-readable violations; an empty list means structurally valid."""
    errors: list[str] = []
    if not isinstance(payload, Mapping):
        return ["proposal must be a JSON object"]
    if payload.get("schema_version") != SCHEMA_VERSION:
        errors.append("wrong schema_version")
    allowed_top = {"schema_version", "operations", "clarification"}
    extra = set(payload) - allowed_top
    if extra:
        errors.append(f"unexpected top-level fields: {sorted(extra)}")
    operations = payload.get("operations")
    if not isinstance(operations, list):
        return errors + ["operations must be a list"]
    if len(operations) > 4:
        errors.append("at most 4 operations are allowed per turn")
    serialized = json.dumps(payload, ensure_ascii=False)
    if _ASIN.search(serialized):
        errors.append("product identity/ASIN output is forbidden")
    for index, operation in enumerate(operations):
        prefix = f"operations[{index}]"
        if not isinstance(operation, Mapping):
            errors.append(f"{prefix} must be an object")
            continue
        allowed = {"op", "attribute", "value", "direction", "scope", "evidence", "reference", "reason"}
        unexpected = set(operation) - allowed
        if unexpected:
            errors.append(f"{prefix} has unexpected fields: {sorted(unexpected)}")
        for field in ("op", "attribute", "value", "direction", "scope", "evidence"):
            if field not in operation:
                errors.append(f"{prefix} missing {field}")
        op = operation.get("op")
        if op not in _OPS:
            errors.append(f"{prefix} invalid op {op!r}")
            continue
        evidence = operation.get("evidence")
        if not _nonempty_text(evidence):
            errors.append(f"{prefix} evidence must be non-empty")
        elif evidence not in user_message:
            errors.append(f"{prefix} evidence is not a substring of the user message")
        attribute = operation.get("attribute")
        value = operation.get("value")
        direction = operation.get("direction")
        scope = operation.get("scope")
        reference = operation.get("reference")
        reason = operation.get("reason")
        if attribute is not None and attribute not in _ATTRS:
            errors.append(f"{prefix} invalid attribute {attribute!r}")
        if direction not in {None, "minimize", "maximize", "exact"}:
            errors.append(f"{prefix} invalid direction {direction!r}")
        if scope not in {None, "task", "long_term"}:
            errors.append(f"{prefix} invalid scope {scope!r}")
        if op in {"set_goal", "update_goal"}:
            if attribute not in {"product_type", "role"}:
                errors.append(f"{prefix} goal requires product_type/role")
            if value is None or isinstance(value, (dict, list)):
                errors.append(f"{prefix} goal requires a scalar value")
            if op == "update_goal" and not _nonempty_text(reason or ""):
                # update_goal may omit reason; semantic validation is done by state tests.
                pass
        elif op in {"add_preference", "update_preference"}:
            if attribute not in _PREF_ATTRS:
                errors.append(f"{prefix} preference requires a soft attribute")
            if direction not in {"minimize", "maximize", "exact"} and value is None:
                errors.append(f"{prefix} preference needs value or direction")
        elif op == "retract_preference":
            if attribute not in _PREF_ATTRS:
                errors.append(f"{prefix} retraction requires a soft attribute")
        elif op in {"set_constraint", "update_constraint"}:
            if attribute not in _CONSTRAINT_ATTRS:
                errors.append(f"{prefix} constraint requires price_max/price_min")
            if value is None:
                errors.append(f"{prefix} constraint requires a value")
        elif op == "retract_constraint":
            if attribute not in _CONSTRAINT_ATTRS:
                errors.append(f"{prefix} constraint retraction requires price_max/price_min")
        elif op == "resolve_reference":
            if reference not in _REFERENCES:
                errors.append(f"{prefix} invalid reference {reference!r}")
            if attribute is not None or value is not None:
                errors.append(f"{prefix} reference op must not carry attribute/value")
        elif op in {"noop", "abstain"}:
            if reason not in _REASONS:
                errors.append(f"{prefix} {op} requires a known reason")
            if attribute is not None or value is not None or direction is not None:
                errors.append(f"{prefix} {op} must not carry preference fields")
        elif op == "clarify":
            clarification = payload.get("clarification")
            if not isinstance(clarification, Mapping):
                errors.append("clarify requires top-level clarification")
            elif not clarification.get("choices"):
                errors.append("clarification needs choices")
    clarify_ops = [op for op in operations if isinstance(op, Mapping) and op.get("op") == "clarify"]
    if clarify_ops and not isinstance(payload.get("clarification"), Mapping):
        errors.append("clarify op without clarification object")
    if not clarify_ops and payload.get("clarification") is not None:
        errors.append("clarification object without clarify op")
    noop_like = [op for op in operations if isinstance(op, Mapping) and op.get("op") in {"noop", "abstain"}]
    if noop_like and len(operations) > 1:
        errors.append("noop/abstain must be the only operation")
    return errors


def parse_and_validate(text: str, user_message: str) -> tuple[Any | None, list[str]]:
    try:
        payload = json.loads(text)
    except (TypeError, ValueError) as exc:
        return None, [f"invalid JSON: {type(exc).__name__}: {exc}"]
    return payload, validate_proposal(payload, user_message)


def canonical_json(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))

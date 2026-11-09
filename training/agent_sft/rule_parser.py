"""Deterministic rule baseline for the preference-operation parser.

This is deliberately a separate, inspectable baseline.  It does not use dataset labels,
runtime C outputs, model scores or Gold identities.  It only demonstrates the lower
bound that the SFT model must beat on the independent parser test.
"""

from __future__ import annotations

import json
import re
from typing import Any

from .schema import SCHEMA_VERSION, validate_proposal

_TARGETS = [
    ("water filter", "water_filter"),
    ("water purifier", "water_filter"),
    ("water filtration", "water_filter"),
    ("filter straw", "water_filter"),
    ("camping cookware", "cookware"),
    ("cookware", "cookware"),
    ("lantern", "lantern"),
    ("camping stove", "stove"),
    ("stove", "stove"),
    ("sleeping pad", "sleeping_pad"),
]


def _proposal(ops: list[dict[str, Any]], clarification: dict[str, Any] | None = None) -> dict[str, Any]:
    return {"schema_version": SCHEMA_VERSION, "operations": ops, "clarification": clarification}


def _op(op: str, evidence: str, **fields: Any) -> dict[str, Any]:
    base = {
        "op": op,
        "attribute": fields.pop("attribute", None),
        "value": fields.pop("value", None),
        "direction": fields.pop("direction", None),
        "scope": fields.pop("scope", "task"),
        "evidence": evidence,
    }
    base.update(fields)
    return base


def rule_proposal(message: str) -> dict[str, Any]:
    text = " ".join(str(message or "").split())
    lowered = text.casefold()

    if "do not know" in lowered or "not sure what kind" in lowered or "help me choose a category" in lowered or "no idea yet" in lowered:
        return _proposal(
            [_op("clarify", text, attribute="product_type")],
            {
                "attribute": "product_type",
                "question": "Which outdoor product category do you want?",
                "choices": ["water filter", "camping cookware", "lantern", "camping stove"],
            },
        )
    if "spaceship" in lowered or "private jet" in lowered or "teleportation" in lowered:
        return _proposal([_op("abstain", text, reason="unsupported_request")])
    if "no additional preference" in lowered or "nothing else to add" in lowered or "no further requirement" in lowered or "that is all" in lowered:
        return _proposal([_op("noop", text, reason="declined_preference")])
    if "second option" in lowered or "third one" in lowered or "fourth result" in lowered or "compare the third" in lowered:
        reference = "previous_candidate_2"
        if "third" in lowered:
            reference = "previous_candidate_3"
        elif "fourth" in lowered:
            reference = "previous_candidate_4"
        return _proposal([_op("resolve_reference", text, reference=reference, reason="cross_turn_reference")])
    if "forget the budget" in lowered or "drop the price cap" in lowered or "remove my maximum price" in lowered or "budget no longer" in lowered:
        return _proposal([_op("retract_constraint", text, attribute="price_max")])
    if "forget" in lowered and "preference" in lowered:
        return _proposal([_op("retract_preference", text, attribute="weight")])
    if re.search(r"under \$?\d+|limit to \$?\d+|cap at \$?\d+|budget:? \$?\d+|ceiling to \$?\d+", text, re.I):
        match = re.search(r"\$?(\d+(?:\.\d+)?)", text)
        amount = float(match.group(1)) if match else 50
        if amount.is_integer():
            amount = int(amount)
        evidence = re.search(r"under \$?\d+|limit to \$?\d+|cap at \$?\d+|budget:? \$?\d+|ceiling to \$?\d+", text, re.I).group(0)
        return _proposal([_op("set_constraint" if "under" in evidence.casefold() and "budget:" not in evidence.casefold() else "update_constraint", evidence, attribute="price_max", value=amount)])
    if "titanium" in lowered:
        return _proposal([_op("add_preference", text, attribute="material", value="titanium", direction="exact")])
    if "lighter" in lowered or "least heavy" in lowered:
        return _proposal([_op("add_preference", text, attribute="weight", direction="minimize")])
    if "replacement mouthpiece" in lowered or "spare mouthpiece" in lowered or "mouthpiece alone" in lowered:
        match = re.search(r"replacement mouthpiece|spare mouthpiece|mouthpiece", text, re.I)
        return _proposal([_op("update_goal", match.group(0) if match else text, attribute="role", value="accessory")])
    if "not a bottle" in lowered or "not the bottle" in lowered or "complete purifier" in lowered or "full filter device" in lowered:
        match = re.search(r"complete water filter|complete purifier|full filter device", text, re.I)
        return _proposal([_op("update_goal", match.group(0) if match else text, attribute="role", value="main")])
    for phrase, target in _TARGETS:
        if phrase in lowered:
            return _proposal([_op("set_goal", phrase, attribute="product_type", value=target)])
    return _proposal([_op("noop", text, reason="no_change")])


def rule_text(message: str) -> str:
    payload = rule_proposal(message)
    errors = validate_proposal(payload, message)
    if errors:
        raise ValueError(f"rule baseline produced invalid proposal: {errors}")
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))

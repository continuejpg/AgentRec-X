"""Current-purchase constraints recovered from an explicit clarification reply.

These constraints are **task-scoped**.  They are not written to M9 and never delete or
rewrite a historical preference.  A current avoidance ("for this purchase, avoid wired")
overrides a historical positive preference only for the current task's effective
preference list; the M9 audit trail remains intact.  An explicit retraction ("forget my
wired preference") is handled by M9's existing writer and is not turned into a task
constraint here.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

from experiments.p14b1.diagnose import canonical
from recommendation.memory.schemas import PreferenceKind

CURRENT_TASK_CONSTRAINT_SCHEMA = "agentrecx.current_task_constraints.v1"
FORM_ATTRS = ("over_ear", "on_ear", "in_ear")


def _norm(value: Any) -> str:
    if hasattr(value, "value"):
        value = value.value
    return " ".join(str(value or "").lower().split())


def _kind_value(kind: Any) -> str:
    return str(getattr(kind, "value", kind) or "").lower()


def constraint_attribute(kind: Any, value: Any) -> str:
    """Map a stated preference to the catalogue attribute it constrains.

    Known connection/form/preference vocabulary is canonicalised (``wired``,
    ``wireless``, ``over_ear``...).  Brand is a named slot, not a substring match.
    Unknown values stay as explicit context (``constraint:<value>``) rather than being
    guessed into an unrelated catalogue attribute.
    """
    known = canonical(value)
    if known:
        return known
    kind_value = _kind_value(kind)
    if kind_value == PreferenceKind.BRAND.value:
        return "brand"
    if kind_value == PreferenceKind.CATEGORY.value:
        return "product_type"
    if kind_value in {PreferenceKind.COLOR.value, PreferenceKind.MATERIAL.value}:
        return kind_value
    return "constraint:" + _norm(value)


def constraint_value(kind: Any, value: Any) -> str:
    known = canonical(value)
    return known or _norm(value)


@dataclass(frozen=True)
class CurrentTaskConstraint:
    """One explicit constraint from the current clarification reply."""

    attribute: str
    value: str
    kind: str
    polarity: str  # "exclude" | "prefer"
    source_text: str
    source_turn_id: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "attribute": self.attribute,
            "value": self.value,
            "kind": self.kind,
            "polarity": self.polarity,
            "source_text": self.source_text,
            "source_turn_id": self.source_turn_id,
        }


@dataclass
class CurrentTaskConstraints:
    """Task-scoped hard exclusions and soft preferences.

    ``hard_exclusions`` are explicit current requirements (AVOID).  They override
    historical positive preferences for the same attribute **in this task only**.
    ``soft_preferences`` are current positive preferences; they add ranking signal but
    do not prove an unknown catalogue fact.
    """

    hard_exclusions: list[CurrentTaskConstraint] = field(default_factory=list)
    soft_preferences: list[CurrentTaskConstraint] = field(default_factory=list)
    source_turns: list[str] = field(default_factory=list)

    @property
    def is_empty(self) -> bool:
        return not self.hard_exclusions and not self.soft_preferences

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": CURRENT_TASK_CONSTRAINT_SCHEMA,
            "hard_exclusions": [item.as_dict() for item in self.hard_exclusions],
            "soft_preferences": [item.as_dict() for item in self.soft_preferences],
            "source_turns": list(self.source_turns),
        }

    @classmethod
    def from_payload(
        cls,
        payload: Mapping[str, Any] | None,
        *, source_turn_id: str = "",
    ) -> "CurrentTaskConstraints":
        """Build from the M9 extraction payload (``preferences`` only).

        ``removals`` are deliberately ignored: they belong to M9's permanent retraction
        semantics, not to the current task constraint layer.
        """
        state = cls()
        if not isinstance(payload, Mapping):
            return state
        preferences = payload.get("preferences") or ()
        for item in preferences:
            if not isinstance(item, Mapping):
                continue
            mode = _norm(item.get("mode") or "add")
            if mode == "remove":
                continue
            raw_value = str(item.get("value") or "").strip()
            if not raw_value:
                continue
            kind = _kind_value(item.get("kind"))
            attribute = constraint_attribute(kind, raw_value)
            value = constraint_value(kind, raw_value)
            polarity = _norm(item.get("polarity") or "prefer")
            source_text = str(item.get("source_text") or raw_value)
            constraint = CurrentTaskConstraint(
                attribute=attribute,
                value=value,
                kind=kind or "unknown",
                polarity="exclude" if polarity in {"avoid", "negative", "dislike"} else "prefer",
                source_text=source_text,
                source_turn_id=str(source_turn_id or ""),
            )
            state = state.merged(cls(
                hard_exclusions=[constraint] if constraint.polarity == "exclude" else [],
                soft_preferences=[constraint] if constraint.polarity == "prefer" else [],
                source_turns=[str(source_turn_id)] if source_turn_id else [],
            ))
        return state

    def merged(self, other: "CurrentTaskConstraints") -> "CurrentTaskConstraints":
        """Merge a later current-turn statement over this task state.

        A current explicit preference for an attribute wins over an earlier current
        exclusion of that same attribute.  A later avoidance wins over a current soft
        preference.  Explicit M9 retractions do not clear current task constraints.
        """
        hard = [dict(item.as_dict()) for item in self.hard_exclusions]
        soft = [dict(item.as_dict()) for item in self.soft_preferences]
        for item in other.hard_exclusions:
            item_dict = item.as_dict()
            soft = [row for row in soft if row["attribute"] != item.attribute]
            hard = [row for row in hard if row["attribute"] != item.attribute]
            hard.append(item_dict)
        for item in other.soft_preferences:
            item_dict = item.as_dict()
            hard = [row for row in hard if row["attribute"] != item.attribute]
            if not any(row["attribute"] == item.attribute for row in soft):
                soft.append(item_dict)
        turns: list[str] = []
        for turn in list(self.source_turns) + list(other.source_turns):
            if turn and turn not in turns:
                turns.append(turn)
        return CurrentTaskConstraints(
            hard_exclusions=[CurrentTaskConstraint(**row) for row in hard],
            soft_preferences=[CurrentTaskConstraint(**row) for row in soft],
            source_turns=turns,
        )

    def to_payload(self) -> dict[str, Any]:
        return self.as_dict()


def effective_preferences(
    history_preferences: Sequence[Mapping[str, Any]],
    constraints: CurrentTaskConstraints | None,
) -> list[dict[str, Any]]:
    """Return the preference list used for the **current task**.

    Historical entries are copied, never mutated.  A current hard exclusion removes a
    historical positive entry with the same canonical attribute from this task view and
    adds an explicit negative preference; the original M9 entry remains active and
    auditable.
    """
    out = [dict(item) for item in history_preferences]
    if constraints is None or constraints.is_empty:
        return out
    for exclusion in constraints.hard_exclusions:
        attribute = exclusion.attribute
        out = [
            item for item in out
            if not (
                str(item.get("polarity") or "positive") == "positive"
                and _norm(item.get("value")) == attribute
            )
        ]
        # Keep exactly one current negative entry for this attribute.
        out = [
            item for item in out
            if not (
                str(item.get("polarity") or "positive") == "negative"
                and _norm(item.get("value")) == attribute
            )
        ]
        out.append({
            "value": attribute if not attribute.startswith("constraint:") else exclusion.value,
            "polarity": "negative",
            "kind": "attribute",
            "source_turn_id": exclusion.source_turn_id,
            "source_text": exclusion.source_text,
        })
    for soft in constraints.soft_preferences:
        attribute = soft.attribute
        if any(
            str(item.get("polarity") or "positive") == "positive"
            and _norm(item.get("value")) == attribute
            for item in out
        ):
            continue
        # Do not turn a canonical exclusion into a positive at the same time.
        out = [
            item for item in out
            if not (
                str(item.get("polarity") or "positive") == "negative"
                and _norm(item.get("value")) == attribute
            )
        ]
        out.append({
            "value": attribute if not attribute.startswith("constraint:") else soft.value,
            "polarity": "positive",
            "kind": "attribute",
            "source_turn_id": soft.source_turn_id,
            "source_text": soft.source_text,
        })
    return out


def _attribute_present(
    attributes: Mapping[str, Any],
    form_factor: str | None,
    constraint: CurrentTaskConstraint,
    product_type: str | None = None,
) -> bool:
    attribute = constraint.attribute
    if attribute in FORM_ATTRS:
        return _norm(form_factor) == attribute
    if attribute == "product_type":
        return _norm(product_type) == constraint.value
    if attribute == "brand":
        if "brand" not in {_norm(name) for name in attributes}:
            return False
        fact = attributes.get("brand")
        evidence = _norm(fact.get("evidence") if isinstance(fact, Mapping) else fact)
        wanted = _norm(constraint.value)
        return bool(wanted and wanted in evidence.split())
    if attribute in {_norm(name) for name in attributes}:
        # A catalogue fact can be present lexically under negation ("not wired",
        # "no wireless").  Do not turn that into a positive constraint match.
        fact = next(
            (value for name, value in attributes.items() if _norm(name) == attribute), None
        )
        evidence = _norm(fact.get("evidence") if isinstance(fact, Mapping) else fact)
        if re.search(rf"\b(?:not|no|without|non)[ -]+{re.escape(attribute)}\b", evidence):
            return False
        return True
    if attribute.startswith("constraint:"):
        wanted = attribute.split(":", 1)[1]
        for name, fact in attributes.items():
            evidence = _norm(fact.get("evidence") if isinstance(fact, Mapping) else fact)
            if _norm(name) == wanted or wanted in evidence.split():
                return True
    return False


def candidate_constraint_status(
    *,
    attributes: Mapping[str, Any],
    form_factor: str | None,
    constraints: CurrentTaskConstraints | None,
    product_type: str | None = None,
) -> dict[str, Any]:
    """Check one candidate's trusted catalogue attributes against current constraints."""
    if constraints is None or constraints.is_empty:
        return {"violations": [], "unknown": [], "matched": [], "status": "none"}
    violations: list[dict[str, Any]] = []
    unknown: list[dict[str, Any]] = []
    matched: list[dict[str, Any]] = []
    for item in constraints.hard_exclusions:
        if _attribute_present(attributes, form_factor, item, product_type=product_type):
            violations.append(item.as_dict())
        else:
            unknown.append(item.as_dict())
    for item in constraints.soft_preferences:
        if _attribute_present(attributes, form_factor, item, product_type=product_type):
            matched.append(item.as_dict())
        else:
            unknown.append(item.as_dict())
    status = "violated" if violations else ("unknown" if unknown else "verified")
    return {
        "violations": violations,
        "unknown": unknown,
        "matched": matched,
        "status": status,
    }

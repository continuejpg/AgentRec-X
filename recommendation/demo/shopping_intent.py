"""Current shopping intent, derived deterministically from the existing preference memory.

Why this module exists
---------------------
Preference memory already stores explicit user statements with provenance and a
supersede/remove lifecycle.  What it does not express is the **current shopping intent**:
which of those statements are hard constraints now, which are ranking preferences, whether
two hard constraints contradict each other, and what a query should look for right now.

This module is a read-only projection, not a second store:

* it starts from a :class:`PreferenceMemorySnapshot` (the single memory store);
* it applies this turn's extraction *preview* on top, so the current message influences the
  current retrieval without writing the turn twice;
* it never mutates memory and never invents a preference the extractor did not produce.

Strength is a deliberately small, deterministic rule over the user's own wording.  A price
with "under / budget / at most" is a hard ceiling; "I prefer" is soft.  Anything ambiguous
stays soft, which cannot exclude a product.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Iterable, Mapping, Sequence

from recommendation.control.grounded_reasoning import (
    attribute_value,
    first_attribute,
    parse_price,
)

__all__ = [
    "ActiveEntry",
    "Conflict",
    "IntentEntry",
    "ShoppingIntent",
    "apply_extraction",
    "build_reason",
    "build_unresolved_question",
    "derive_intent",
    "query_terms",
]

BRAND_KEYS = ("Brand Name", "Brand", "Manufacturer")

#: Wording that makes a price a hard ceiling/floor rather than a soft preference.
_PRICE_HARD = re.compile(
    r"\b(under|below|less than|no more than|at most|up to|max(?:imum)?|budget|"
    r"at least|over|more than|min(?:imum)?|between)\b",
    re.IGNORECASE,
)
#: Wording that makes a non-price statement a hard requirement.
_HARD_MARKER = re.compile(
    r"\b(must|only|exactly|required|require|have to|has to|need it to be)\b",
    re.IGNORECASE,
)
#: Kinds that can be turned into a deterministic constraint.
_CONSTRAINT_KINDS = frozenset(
    {
        "category",
        "brand",
        "price_max",
        "price_min",
        "color",
        "material",
        "feature",
        "free_form_constraint",
    }
)
#: Kinds a lexical catalogue query can use, in priority order.
_QUERY_KINDS = ("category", "brand", "feature")


class Strength(str, Enum):
    HARD = "hard"
    SOFT = "soft"


@dataclass(frozen=True)
class ActiveEntry:
    """One current preference, with the turn that stated it."""

    kind: str
    value: str
    polarity: str = "prefer"
    source_turn_id: str = ""
    source_text: str = ""
    #: True for an application-supplied hard constraint: its strength is explicit, not
    #: inferred from wording, so an unclassifiable kind (e.g. free-form) stays hard.
    forced_hard: bool = False

    def key(self) -> tuple[str, str, str]:
        return (self.kind, self.value.casefold(), self.polarity)


@dataclass(frozen=True)
class IntentEntry:
    """One entry of the current intent, with its strength and provenance."""

    kind: str
    value: str
    polarity: str
    strength: Strength
    source_turn_id: str
    source_text: str

    def as_constraint(self) -> dict[str, str] | None:
        if self.strength is not Strength.HARD or self.kind not in _CONSTRAINT_KINDS:
            return None
        return {"kind": self.kind, "value": self.value}


@dataclass(frozen=True)
class Conflict:
    """An explicit contradiction the intent cannot resolve on its own."""

    code: str
    detail: str
    question: str
    options: tuple[str, ...] = ()


@dataclass(frozen=True)
class ShoppingIntent:
    """The current, deterministic projection of what the user is shopping for."""

    hard: tuple[IntentEntry, ...] = ()
    soft: tuple[IntentEntry, ...] = ()
    conflicts: tuple[Conflict, ...] = ()
    turn_ids: tuple[str, ...] = ()
    active_preference_count: int = 0

    def hard_constraints(self) -> tuple[dict[str, str], ...]:
        seen: set[tuple[str, str]] = set()
        out: list[dict[str, str]] = []
        for entry in self.hard:
            constraint = entry.as_constraint()
            if constraint is None:
                continue
            key = (constraint["kind"], constraint["value"].casefold())
            if key in seen:
                continue
            seen.add(key)
            out.append(constraint)
        return tuple(out)

    def constraints_by_kind(self) -> dict[str, list[IntentEntry]]:
        grouped: dict[str, list[IntentEntry]] = {}
        for entry in self.hard:
            grouped.setdefault(entry.kind, []).append(entry)
        return grouped

    def query_values(self) -> tuple[str, ...]:
        """Current query values, hard first, then soft, in kind priority order."""
        values: list[str] = []
        seen: set[str] = set()

        def add(value: str) -> None:
            text = " ".join(str(value).split())
            key = text.casefold()
            if text and key not in seen:
                seen.add(key)
                values.append(text)

        for strength in (self.hard, self.soft):
            for kind in _QUERY_KINDS:
                for entry in strength:
                    if entry.kind == kind and entry.polarity == "prefer":
                        add(entry.value)
        return tuple(values)


def _strength(kind: str, source_text: str, polarity: str) -> Strength:
    """Decide hard vs soft from the user's own wording, conservatively."""
    text = str(source_text or "")
    if kind in ("price_max", "price_min"):
        return Strength.HARD if _PRICE_HARD.search(text) else Strength.SOFT
    if kind in ("category", "brand", "color", "material", "feature"):
        if polarity == "avoid":
            return Strength.HARD if _HARD_MARKER.search(text) else Strength.SOFT
        return Strength.HARD if _HARD_MARKER.search(text) else Strength.SOFT
    return Strength.SOFT


def apply_extraction(
    active: Iterable[ActiveEntry], extraction: Any, *, turn_id: str
) -> tuple[ActiveEntry, ...]:
    """Apply one turn's extraction preview to the active entries, without persisting.

    Mirrors the accepted memory semantics: removals tombstone matching entries first, then
    ``ADD`` appends nothing it already has, and ``REPLACE`` retracts the same kind/polarity
    (or the explicitly named value) before appending.  The memory service remains the only
    store; this is a turn-local projection.
    """
    entries = list(active)
    for removal in getattr(extraction, "removals", ()) or ():
        kind = getattr(removal, "kind", None)
        kind = getattr(kind, "value", kind)
        value = getattr(removal, "value", None)
        entries = [
            entry
            for entry in entries
            if not (
                (kind is None or entry.kind == kind)
                and (value is None or entry.value.casefold() == str(value).casefold())
            )
        ]
    for candidate in getattr(extraction, "preferences", ()) or ():
        kind = getattr(candidate.kind, "value", candidate.kind)
        value = str(candidate.value)
        polarity = getattr(candidate.polarity, "value", candidate.polarity)
        mode = getattr(candidate.mode, "value", candidate.mode)
        if mode == "remove":
            entries = [
                entry
                for entry in entries
                if not (entry.kind == kind and entry.value.casefold() == value.casefold())
            ]
            continue
        if mode == "replace":
            replaces = getattr(candidate, "replaces", None)
            if replaces is not None:
                entries = [
                    entry
                    for entry in entries
                    if not (
                        entry.kind == kind
                        and entry.value.casefold() == str(replaces).casefold()
                    )
                ]
            else:
                entries = [
                    entry
                    for entry in entries
                    if not (entry.kind == kind and entry.polarity == polarity)
                ]
        if any(entry.key() == ActiveEntry(kind, value, polarity).key() for entry in entries):
            continue
        entries.append(
            ActiveEntry(
                kind=kind,
                value=value,
                polarity=polarity,
                source_turn_id=turn_id,
                source_text=str(getattr(candidate, "source_text", "") or ""),
            )
        )
    return tuple(entries)


def _tightest(entries: Sequence[IntentEntry], kind: str) -> list[IntentEntry]:
    """Keep the strictest hard bound per kind: min ceiling, max floor."""
    values = [entry for entry in entries if entry.kind == kind]
    if not values:
        return []
    if kind == "price_max":
        priced = [(parse_price(entry.value), entry) for entry in values]
        priced = [(number, entry) for number, entry in priced if number is not None]
        return [min(priced, key=lambda pair: pair[0])[1]] if priced else values[:1]
    if kind == "price_min":
        priced = [(parse_price(entry.value), entry) for entry in values]
        priced = [(number, entry) for number, entry in priced if number is not None]
        return [max(priced, key=lambda pair: pair[0])[1]] if priced else values[:1]
    return values


def derive_intent(
    snapshot: Any,
    extraction: Any,
    *,
    turn_id: str,
    explicit_constraints: Sequence[Mapping[str, str]] = (),
) -> ShoppingIntent:
    """Project the active snapshot + this turn's preview into the current intent."""
    active = [
        ActiveEntry(
            kind=getattr(entry.kind, "value", entry.kind),
            value=str(entry.value),
            polarity=getattr(entry.polarity, "value", entry.polarity),
            source_turn_id=str(getattr(entry, "source_turn_id", "") or ""),
            source_text=str(getattr(entry, "source_text", "") or ""),
        )
        for entry in getattr(snapshot, "active_entries", ()) or ()
    ]
    entries = apply_extraction(active, extraction, turn_id=turn_id)
    for constraint in explicit_constraints:
        kind = str(constraint.get("kind", "")).strip().lower()
        value = str(constraint.get("value", "")).strip()
        if kind and value:
            entries = tuple(entries) + (
                ActiveEntry(
                    kind=kind,
                    value=value,
                    source_turn_id=turn_id,
                    source_text="must satisfy the explicit request constraint",
                    forced_hard=True,
                ),
            )

    hard_entries: list[IntentEntry] = []
    soft_entries: list[IntentEntry] = []
    for entry in entries:
        strength = (
            Strength.HARD
            if entry.forced_hard
            else _strength(entry.kind, entry.source_text, entry.polarity)
        )
        projected = IntentEntry(
            kind=entry.kind,
            value=entry.value,
            polarity=entry.polarity,
            strength=strength,
            source_turn_id=entry.source_turn_id,
            source_text=entry.source_text,
        )
        (hard_entries if strength is Strength.HARD else soft_entries).append(projected)

    grouped = {
        kind: _tightest(hard_entries, kind)
        for kind in ("price_max", "price_min")
    }
    effective_hard: list[IntentEntry] = []
    for kind in ("price_max", "price_min"):
        effective_hard.extend(grouped[kind])
    for entry in hard_entries:
        if entry.kind not in ("price_max", "price_min"):
            effective_hard.append(entry)

    conflicts: list[Conflict] = []
    ceilings = [entry for entry in hard_entries if entry.kind == "price_max"]
    floors = [entry for entry in hard_entries if entry.kind == "price_min"]
    ceiling_values = [
        value for value in (parse_price(entry.value) for entry in ceilings) if value is not None
    ]
    floor_values = [
        value for value in (parse_price(entry.value) for entry in floors) if value is not None
    ]
    if ceiling_values and floor_values:
        ceiling = min(ceiling_values)
        floor = max(floor_values)
        if floor > ceiling:
            conflicts.append(Conflict(
                code="price_range",
                detail=f"minimum {floor:.2f} is above maximum {ceiling:.2f}",
                question=(
                    f"Your minimum (${floor:.2f}) is above your maximum (${ceiling:.2f}). "
                    "Should I raise the maximum or lower the minimum?"
                ),
                options=(f"raise the maximum above ${floor:.2f}", f"lower the minimum below ${ceiling:.2f}"),
            ))
    for entry in hard_entries:
        if entry.kind != "brand":
            continue
        opposite = "avoid" if entry.polarity == "prefer" else "prefer"
        for other in hard_entries:
            if (
                other.kind == "brand"
                and other.polarity == opposite
                and other.value.casefold() == entry.value.casefold()
            ):
                conflicts.append(Conflict(
                    code="brand_contradiction",
                    detail=f"brand {entry.value} is both required and avoided",
                    question=(
                        f"You want {entry.value} and also want to avoid it. "
                        "Which one should I keep?"
                    ),
                    options=(f"keep {entry.value}", f"avoid {entry.value}"),
                ))
                break

    turn_ids: list[str] = []
    for entry in entries:
        if entry.source_turn_id and entry.source_turn_id not in turn_ids:
            turn_ids.append(entry.source_turn_id)
    return ShoppingIntent(
        hard=tuple(effective_hard),
        soft=tuple(soft_entries),
        conflicts=tuple(conflicts),
        turn_ids=tuple(turn_ids),
        active_preference_count=len(entries),
    )


def query_terms(intent: ShoppingIntent, message: str, *, limit: int = 8) -> tuple[str, ...]:
    """Terms a catalogue query should use now: current intent first, message last.

    Removed/superseded values are absent by construction: the intent only contains active
    entries.  The limit is applied after de-duplication, so a repeated term cannot crowd
    out a field term.
    """
    from recommendation.control import SearchCatalogArguments

    values: list[str] = list(intent.query_values())
    base = SearchCatalogArguments.from_text(message, limit=limit)
    values.extend(str(term) for term in getattr(base, "terms", ()) or ())
    seen: set[str] = set()
    terms: list[str] = []
    for value in values:
        text = " ".join(str(value).split())
        key = text.casefold()
        if text and key not in seen:
            seen.add(key)
            terms.append(text)
        if len(terms) >= limit:
            break
    return tuple(terms)


def _categories(record: Any) -> tuple[str, ...]:
    values = list(getattr(record, "categories", ()) or ())
    main = getattr(record, "main_category", None)
    if main:
        values.append(main)
    return tuple(str(value) for value in values)


def _match_category(record: Any, value: str) -> bool:
    needle = str(value).casefold()
    return any(needle in category.casefold() for category in _categories(record))


def _brand(record: Any) -> str | None:
    return first_attribute(record, BRAND_KEYS)


def _match_brand(record: Any, value: str) -> bool:
    brand = _brand(record)
    return bool(brand) and str(value).casefold() in brand.casefold()


def _price(record: Any) -> float | None:
    return parse_price(getattr(record, "price_text", None))


def build_reason(record: Any, intent: ShoppingIntent) -> str:
    """A deterministic reason for one final product, from catalogue facts and current intent.

    Every clause is checked against this product's own record before it is written: a fact
    the catalogue does not state is never asserted, and a preference that does not match is
    not claimed to match.  The clause set is intentionally small so the reason is auditable.
    """
    if record is None:
        return "Catalogue facts are unavailable for this product."
    parts: list[str] = []
    price = _price(record)
    brand = _brand(record)
    for entry in intent.hard:
        if entry.kind == "price_max" and price is not None:
            limit = parse_price(entry.value)
            if limit is not None and price <= limit:
                parts.append(f"within your ${limit:.2f} budget (catalogue price ${price:.2f})")
        elif entry.kind == "price_min" and price is not None:
            limit = parse_price(entry.value)
            if limit is not None and price >= limit:
                parts.append(f"above your ${limit:.2f} minimum (catalogue price ${price:.2f})")
        elif entry.kind == "category" and _match_category(record, entry.value):
            parts.append(f"matches required category '{entry.value}'")
        elif entry.kind == "brand" and _match_brand(record, entry.value):
            parts.append(f"matches required brand '{entry.value}'")
        elif entry.kind == "color":
            colour = attribute_value(record, "Color")
            if colour and str(entry.value).casefold() in colour.casefold():
                parts.append(f"colour matches '{entry.value}' (catalogue: {colour})")
    for entry in intent.soft:
        if entry.polarity != "prefer":
            continue
        if entry.kind == "category" and _match_category(record, entry.value):
            parts.append(f"matches your interest in '{entry.value}'")
        elif entry.kind == "brand" and _match_brand(record, entry.value):
            parts.append(f"matches your preferred brand '{entry.value}'")
    if not parts:
        facts: list[str] = []
        if brand:
            facts.append(f"brand {brand}")
        categories = _categories(record)
        if categories:
            facts.append(f"category {categories[-1]}")
        if price is not None:
            facts.append(f"price ${price:.2f}")
        if facts:
            parts.append("catalogue facts: " + ", ".join(facts))
        else:
            parts.append("no catalogue fact was available to support a specific reason")
    return "; ".join(parts) + "."


def build_unresolved_question(context: Any) -> str:
    """Ask about the undecided candidates without claiming they are non-compliant."""
    constraints = tuple(getattr(context, "task_constraints", ()) or ())
    labels = ", ".join(str(item) for item in constraints[:3]) or "your requirements"
    count = int(getattr(context, "unresolved_count", 0) or 0)
    return (
        f"I couldn't verify {labels} for {count} candidate(s). "
        "Could you confirm a specific value, or relax that requirement?"
    )

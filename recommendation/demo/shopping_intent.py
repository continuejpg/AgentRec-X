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
from collections import Counter
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Iterable, Mapping, Sequence

from recommendation.control.grounded_reasoning import (
    attribute_value,
    first_attribute,
    parse_price,
    parse_weight_kg,
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
    "promote_explicit_type_matches",
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
    #: Where this entry came from: ``user_stated`` (this turn's validated statement),
    #: ``validated_existing`` (accepted memory from an earlier turn) or ``decision_state``
    #: (an authorized M5 constraint merged by M5.2).  It is provenance for the effective
    #: intent view, never a change to hardness.
    origin: str = "validated_existing"

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
    #: See :attr:`ActiveEntry.origin`; carried into the user-visible intent view so a
    #: remembered or state-authorized constraint is never presented as a current-turn
    #: user statement.
    origin: str = "user_stated"

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
                and (
                    value is None
                    or entry.value.casefold() == str(value).casefold()
                    or _removal_value_prefix(
                        str(value).casefold(), entry.value.casefold()
                    )
                )
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
                origin="user_stated",
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
            origin="validated_existing",
        )
        for entry in getattr(snapshot, "active_entries", ()) or ()
    ]
    # An explicit budget correction in this turn replaces the active budget for the same
    # task-scoped kind.  Event history remains untouched; the current projection must not
    # keep evaluating an older ceiling/floor once the user has stated a newer one.
    current_price_kinds = {
        str(getattr(getattr(preference, "kind", ""), "value", getattr(preference, "kind", "")))
        for preference in (getattr(extraction, "preferences", ()) or ())
        if str(getattr(getattr(preference, "kind", ""), "value", getattr(preference, "kind", "")))
        in {"price_max", "price_min"}
    }
    current_price_kinds.update(
        str(constraint.get("kind", "")).strip().lower()
        for constraint in explicit_constraints
        if str(constraint.get("kind", "")).strip().lower() in {"price_max", "price_min"}
    )
    if current_price_kinds:
        active = [entry for entry in active if entry.kind not in current_price_kinds]
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
                    source_text=str(
                        constraint.get("source_text")
                        or "must satisfy the explicit request constraint"
                    ),
                    forced_hard=True,
                    origin="user_stated",
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
            origin=entry.origin,
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


def promote_explicit_type_matches(
    message: str,
    recommendations: Sequence[Any],
    records: Mapping[str, Any],
) -> tuple[Any, ...]:
    """Scope explicit product-type requests to trusted catalogue types.

    The target type is grounded in the user's own type words plus trusted candidate
    titles/categories; when those leaves are split, a catalogue-source leaf consensus is
    still the fallback.  Broad requests (for example "hiking gear") have no product-type
    consensus and are left untouched.  When at least two grounded same-type candidates
    exist, trusted different-type candidates are removed from the requested scope;
    candidates without trustworthy category evidence remain after the grounded matches.
    Every retained identity keeps its real score, score kind and source provenance.
    """
    from recommendation.control import SearchCatalogArguments

    def tokens(text: str) -> set[str]:
        base = SearchCatalogArguments.from_text(text, limit=16)
        return {
            str(term).casefold().rstrip("s")
            for term in getattr(base, "terms", ()) or ()
            if str(term).strip()
        }

    generic = {
        "gear", "equipment", "camping", "hiking", "outdoor", "sports", "trip",
        "i", "need", "a", "an", "the", "for", "under", "with", "and", "or",
        "show", "me", "want", "complete", "budget", "options", "color", "colors",
        "please", "looking", "find",
    }
    modifiers = {
        "lightweight", "portable", "best", "good", "small", "large", "cheap",
        "compact", "durable", "premium", "ultralight", "new",
    }
    accessory_terms = {"replacement", "mouthpiece", "part", "parts", "spare", "accessory", "case"}

    query_tokens = tokens(message)
    product_tokens = (query_tokens - generic - modifiers)
    query_accessory_tokens = query_tokens & accessory_terms
    target_ids: set[str] = set()

    # Role gating before type consensus: a mouthpiece query for a water filter must
    # not fall back to every mouthpiece in the catalogue when no leaf consensus exists.
    # Only trusted title evidence of a filter device qualifies; bottle bite valves and
    # unrelated scuba/hydration mouthpieces stay UNKNOWN and are not presented.
    if _asks_for_accessory(message) and _water_filter_query(message) and "bottle" not in query_tokens:
        recommendations = tuple(
            item
            for item in recommendations
            if _is_water_filter_accessory_candidate(
                records.get(str(getattr(item, "parent_asin", "")))
            )
        )
        if not recommendations:
            return ()

    def trusted_text(record: Any) -> str:
        return " ".join(
            part
            for part in (
                str(getattr(record, "title", "") or ""),
                " ".join(str(value) for value in getattr(record, "categories", ()) or ()),
                str(_leaf_category(record) or ""),
            )
            if part
        )

    if len(product_tokens) >= 2:
        wants_accessory = _asks_for_accessory(message)
        for item in recommendations:
            identity = str(getattr(item, "parent_asin", ""))
            record = records.get(identity)
            if record is None:
                continue
            text_tokens = tokens(trusted_text(record))
            if not product_tokens <= text_tokens:
                continue
            if wants_accessory:
                # The user's accessory words must all be grounded in the title;
                # "Extra Filters Replacement Set" is not a replacement mouthpiece.
                required_accessory = query_accessory_tokens or accessory_terms
                if not required_accessory <= text_tokens:
                    continue
            target_ids.add(identity)
        if len(target_ids) < 2:
            target_ids = set()

    # Fallback: the accepted catalogue-source leaf consensus.  This covers a clear
    # product type that the user names but whose title/category wording is split
    # across leaves (for example camping cookware).
    if not target_ids:
        enriched: list[tuple[Any, str]] = []
        for item in recommendations:
            if "catalog_search" not in tuple(getattr(item, "sources", ()) or ()):
                continue
            record = records.get(str(getattr(item, "parent_asin", "")))
            leaf = _leaf_category(record)
            if leaf:
                enriched.append((item, leaf))
        if len(enriched) < 2:
            return tuple(recommendations)
        counts = Counter(leaf for _, leaf in enriched)
        leaf, top_count = counts.most_common(1)[0]
        if top_count < 2:
            return tuple(recommendations)
        leaf_tokens = tokens(leaf)
        if not (query_tokens & leaf_tokens) - (generic | modifiers):
            return tuple(recommendations)
        target = leaf.casefold()
        target_ids = {
            str(getattr(item, "parent_asin", ""))
            for item, item_leaf in enriched
            if item_leaf.casefold() == target
        }

    def relevance(item: Any) -> int:
        identity = str(getattr(item, "parent_asin", ""))
        if identity in target_ids:
            return 0
        record = records.get(identity)
        if _leaf_category(record) is None:
            return 1  # UNKNOWN: neutral, kept only after grounded type matches
        return 2

    ordered = sorted(recommendations, key=relevance)
    # A clear product-type request is a scope, not a ranking hint: once at least two
    # grounded type matches exist, keep those and genuinely unknown candidates; do not
    # present trusted products of a different type as if they satisfied the request.
    grounded = [item for item in ordered if relevance(item) == 0]
    if len(grounded) >= 2:
        ordered = [item for item in ordered if relevance(item) in (0, 1)]
    if _asks_for_accessory(message) and _water_filter_query(message):
        # A generic bottle bite valve is not evidence of compatibility with a water
        # purifier.  Keep it only when the title also names a filter device or the
        # user asked for a bottle explicitly; otherwise it stays UNKNOWN and is not
        # presented as a matching replacement part.
        if "bottle" not in query_tokens:
            ordered = [
                item
                for item in ordered
                if not _is_bottle_accessory_without_filter_device(
                    records.get(str(getattr(item, "parent_asin", "")))
                )
            ]
    elif _water_filter_query(message) and "bottle" not in query_tokens:
        # A "complete water filter" request is about a filtration device, not a
        # filtered drinking bottle.  Keep bottles only when the user names a bottle.
        ordered = [
            item
            for item in ordered
            if not _is_bottle_product_without_filter_device(
                records.get(str(getattr(item, "parent_asin", "")))
            )
        ]
    # A replacement part is a different product role from the main appliance.  Only
    # remove it when the user asked for the main product and the title gives trusted
    # evidence that it is a part; an explicit parts query keeps the candidate.
    if not _asks_for_accessory(message):
        filtered = [
            item
            for item in ordered
            if not _is_accessory_record(
                records.get(str(getattr(item, "parent_asin", "")))
            )
        ]
        if filtered:
            ordered = filtered
    result: list[Any] = []
    for position, item in enumerate(ordered, start=1):
        copy = getattr(item, "model_copy", None)
        result.append(
            copy(update={"rank": position}) if callable(copy) else item
        )
    return tuple(result)


def _categories(record: Any) -> tuple[str, ...]:
    values = list(getattr(record, "categories", ()) or ())
    main = getattr(record, "main_category", None)
    if main:
        values.append(main)
    return tuple(str(value) for value in values)


def _removal_value_prefix(selector: str, candidate: str) -> bool:
    """Whole leading-phrase match for a value-only withdrawal selector."""
    selector = " ".join(str(selector).split())
    candidate = " ".join(str(candidate).split())
    if not selector or not candidate.startswith(selector):
        return False
    if len(candidate) == len(selector):
        return True
    return candidate[len(selector)] in " \t,;:-"


def _leaf_category(record: Any) -> str | None:
    """The trusted source leaf, without appending the broad main category.

    ``main_category`` is typically "Sports & Outdoors" for every candidate and cannot
    distinguish a water filter from a sleeping pad, so it is only a fallback when the
    source did not provide any ordered categories.
    """
    categories = tuple(
        str(value).strip() for value in getattr(record, "categories", ()) or ()
        if str(value).strip()
    )
    if categories:
        return categories[-1]
    main = getattr(record, "main_category", None)
    return str(main).strip() if main and str(main).strip() else None


_ACCESSORY_TITLE = re.compile(
    r"\b(replacement\s+mouthpiece|mouthpiece|carrying\s+case|case\s+only|"
    r"spare\s+part|accessory)\b",
    re.IGNORECASE,
)
_ACCESSORY_REQUEST = re.compile(
    r"\b(replacement|replace|mouthpiece|part|spare|accessory|accessories|case)\b",
    re.IGNORECASE,
)


def _asks_for_accessory(message: str) -> bool:
    return bool(_ACCESSORY_REQUEST.search(str(message or "")))


def _is_accessory_record(record: Any) -> bool:
    """True only when the trusted title explicitly describes a part/accessory."""
    title = getattr(record, "title", None)
    return bool(title and _ACCESSORY_TITLE.search(str(title)))


def _water_filter_query(message: str) -> bool:
    lowered = " ".join(str(message or "").casefold().split())
    return ("water filter" in lowered or "water purifier" in lowered) and "water bottle" not in lowered


def _is_bottle_accessory_without_filter_device(record: Any) -> bool:
    """A bottle bite valve is not evidence of compatibility with a water purifier.

    Only trusted title text is used.  A part whose title also names a dedicated filter
    device (straw/purifier/pump/system) may legitimately be a filter replacement; a
    generic water-bottle mouthpiece stays UNKNOWN and is not presented as compatible.
    """
    title = " ".join(str(getattr(record, "title", "") or "").casefold().split())
    if not title:
        return False
    bottle_role = any(
        term in title
        for term in ("water bottle", "bite valve", "bottle mouthpiece", "sports bottle")
    )
    if not bottle_role:
        return False
    filter_device = any(
        term in title
        for term in (
            "filter straw", "water purifier", "filtration system", "filter pump",
            "water filter straw", "survival filter", "filter housing",
        )
    )
    return not filter_device


def _is_bottle_product_without_filter_device(record: Any) -> bool:
    """True for a filtered drinking bottle offered against a main-filter request."""
    title = " ".join(str(getattr(record, "title", "") or "").casefold().split())
    if not title:
        return False
    if not any(term in title for term in ("water bottle", "bottle with filter", "filtered bottle")):
        return False
    return not any(
        term in title
        for term in (
            "filter straw", "water purifier", "filtration system", "filter pump",
            "water filter straw", "survival filter", "filter housing",
        )
    )


def _is_water_filter_accessory_candidate(record: Any) -> bool:
    """Trusted-title gate for an accessory explicitly requested for a water filter."""
    if record is None:
        return False
    title = " ".join(str(getattr(record, "title", "") or "").casefold().split())
    if not title or _is_bottle_accessory_without_filter_device(record):
        return False
    if not _ACCESSORY_REQUEST.search(title):
        # A complete filter is not a replacement part merely because it is a water
        # filter; the accessory role itself must be grounded in the title.
        return False
    return any(
        term in title
        for term in (
            "water filter", "filter straw", "water purifier", "filtration system",
            "filter pump", "filter housing",
        )
    )


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
        elif entry.kind == "price" and price is not None:
            # This is a soft direction, never a hard price bound.  It states only that the
            # preference was applied to this product's real catalogue price.
            parts.append(
                f"price preference applied (catalogue price ${price:.2f})"
            )
        elif entry.kind == "weight":
            raw = first_attribute(record, ("Item Weight", "Package Weight"))
            weight = parse_weight_kg(raw) if raw else None
            if weight is not None:
                parts.append(f"weight preference applied (catalogue {weight:g} kg)")
        elif entry.kind == "color":
            colour = attribute_value(record, "Color")
            if colour and str(entry.value).casefold() in colour.casefold():
                parts.append(f"matches your preferred colour '{entry.value}'")
        elif entry.kind == "material":
            material = attribute_value(record, "Material")
            if material and str(entry.value).casefold() in material.casefold():
                parts.append(f"matches your preferred material '{entry.value}'")
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

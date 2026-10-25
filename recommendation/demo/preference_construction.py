"""Candidate-aware preference construction for the shopping path.

This module is deliberately small and deterministic.  It reuses the accepted
:class:`~recommendation.control.product_research.ProductResearchTool` for catalogue facts,
never infers a fact from a title, and produces only options that are backed by values present
on the current candidates.

A preference is a **soft** task preference: it may reorder candidates that already passed
hard-constraint filtering, and it may never add, remove, or override a candidate because of a
hard constraint.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence
from urllib.parse import quote

from recommendation.control.product_research import ProductResearchTool
from recommendation.memory import PreferenceKind

from .shopping_intent import IntentEntry, ShoppingIntent, Strength

__all__ = [
    "PREFERENCE_ATTRIBUTES",
    "CandidatePreference",
    "PreferenceOption",
    "PreferencePrompt",
    "TaskPreference",
    "PreferenceRanking",
    "PreferenceRankedCandidate",
    "augment_intent",
    "build_preference_prompt",
    "preference_from_option",
    "rank_with_task_preference",
    "resolve_preference_option",
]

#: Attributes that can be phrased as a useful preference question.  The order is the
#: deterministic priority used when several attributes differ.
PREFERENCE_ATTRIBUTES: tuple[str, ...] = ("price", "weight", "color", "material", "brand")

_ATTRIBUTE_LABELS = {
    "price": "price",
    "weight": "weight",
    "color": "color",
    "material": "material",
    "brand": "brand",
}

_NUMERIC_DIRECTIONS = {
    "price": ("lower", "higher", "Lower price", "Higher price"),
    "weight": ("lower", "higher", "Lighter weight", "Heavier weight"),
}


@dataclass(frozen=True)
class PreferenceOption:
    """One machine-readable, executable preference option."""

    option_id: str
    label: str
    action: str  # lower | higher | value | none | skip
    attribute: str
    value: str | float | None = None


@dataclass(frozen=True)
class PreferencePrompt:
    """Evidence and options for one candidate-aware preference question."""

    attribute: str
    label: str
    reason: str
    products: tuple[str, ...]
    values: Mapping[str, Any]
    comparison_kind: str  # numeric | categorical
    options: tuple[PreferenceOption, ...]


@dataclass(frozen=True)
class TaskPreference:
    """The current task's soft preference (never a long-term memory record)."""

    attribute: str
    action: str  # lower | higher | value
    value: str | float | None
    option_id: str
    label: str


@dataclass(frozen=True)
class PreferenceRankedCandidate:
    """A candidate's position under the task preference."""

    parent_asin: str
    item_id: int
    rank: int
    reranked_rank: int
    match_count: int
    violation_count: int
    unknown_count: int
    evidence: tuple[Any, ...] = ()
    reason: str | None = None


@dataclass(frozen=True)
class PreferenceRanking:
    """A ranking object compatible with ``build_recommendations_view``."""

    candidates: tuple[PreferenceRankedCandidate, ...]
    baseline_order: tuple[str, ...]
    ranked_order: tuple[str, ...]
    changed_order: bool

    def by_identity(self) -> dict[str, PreferenceRankedCandidate]:
        return {candidate.parent_asin: candidate for candidate in self.candidates}


def _value_text(value: Any) -> str:
    if isinstance(value, (list, tuple)):
        return ", ".join(str(item) for item in value)
    return str(value)


def _option_id(attribute: str, action: str, value: Any = None) -> str:
    base = f"pref:{attribute}:{action}"
    if value is None:
        return base
    return f"{base}:{quote(_value_text(value), safe='')}"


def _numeric_label(attribute: str, direction: str, values: Sequence[float]) -> str:
    low = min(values)
    high = max(values)
    if attribute == "price":
        return f"{'Lower' if direction == 'lower' else 'Higher'} price (${low:.2f}–${high:.2f})"
    if attribute == "weight":
        return (
            f"{'Lighter' if direction == 'lower' else 'Heavier'} weight "
            f"({low:.2f}–{high:.2f} kg)"
        )
    return f"{direction} {attribute} ({low:g}–{high:g})"


def _reason_for(attribute: str, kind: str, values: Mapping[str, Any]) -> str:
    rendered = ", ".join(_value_text(value) for value in values.values())
    if kind == "numeric":
        return f"Candidates differ on {attribute}: {rendered}."
    return f"Candidates offer different {attribute} values: {rendered}."


def build_preference_prompt(
    candidate_ids: Sequence[str],
    records: Mapping[str, Any],
    *,
    blocked_attributes: Sequence[str] = (),
) -> PreferencePrompt | None:
    """Find one useful, evidence-backed preference dimension, or return ``None``.

    The comparison is over the accepted tool's trusted records.  Only a ``comparable`` row
    (every candidate carries one value of the same type/unit) is eligible; missing or
    unreadable values are never filled in or treated as equal.
    """
    identities = tuple(dict.fromkeys(str(item) for item in candidate_ids if str(item)))
    if len(identities) < 2:
        return None
    identities = identities[:5]
    tool = ProductResearchTool(records)
    try:
        comparison = tool.compare_products(
            identities, attributes=PREFERENCE_ATTRIBUTES
        )
    except Exception:
        return None

    blocked = {str(item).strip().lower() for item in blocked_attributes}
    rows = {str(row.attribute): row for row in comparison.rows}
    for attribute in PREFERENCE_ATTRIBUTES:
        if attribute in blocked:
            continue
        row = rows.get(attribute)
        if row is None or row.status != "comparable" or row.comparison is None:
            continue
        values = dict(row.comparison.get("values") or {})
        if len(values) < 2 or any(identity not in values for identity in identities):
            continue
        kind = str(row.comparison.get("kind") or "")
        distinct: list[Any] = []
        for value in values.values():
            text = _value_text(value)
            if not any(text == _value_text(existing) for existing in distinct):
                distinct.append(value)
        if len(distinct) < 2:
            continue

        options: list[PreferenceOption] = []
        if kind == "numeric":
            numeric = [float(value) for value in values.values()]
            directions = _NUMERIC_DIRECTIONS.get(attribute, ("lower", "higher", "Lower", "Higher"))
            low_action, high_action, low_label, high_label = directions
            options.extend(
                (
                    PreferenceOption(
                        option_id=_option_id(attribute, low_action),
                        label=_numeric_label(attribute, "lower", numeric),
                        action=low_action,
                        attribute=attribute,
                    ),
                    PreferenceOption(
                        option_id=_option_id(attribute, high_action),
                        label=_numeric_label(attribute, "higher", numeric),
                        action=high_action,
                        attribute=attribute,
                    ),
                )
            )
        else:
            for value in distinct:
                text = _value_text(value)
                options.append(
                    PreferenceOption(
                        option_id=_option_id(attribute, "value", text),
                        label=f"Prefer {text}",
                        action="value",
                        attribute=attribute,
                        value=text,
                    )
                )
        options.extend(
            (
                PreferenceOption(
                    option_id="pref:none",
                    label="No additional preference",
                    action="none",
                    attribute=attribute,
                ),
                PreferenceOption(
                    option_id="pref:skip",
                    label="Skip preference and continue",
                    action="skip",
                    attribute=attribute,
                ),
            )
        )
        return PreferencePrompt(
            attribute=attribute,
            label=_ATTRIBUTE_LABELS[attribute],
            reason=_reason_for(attribute, kind, values),
            products=identities,
            values=values,
            comparison_kind=kind,
            options=tuple(options),
        )
    return None


def resolve_preference_option(
    prompt: PreferencePrompt,
    *,
    option_id: str | None,
    message: str = "",
) -> tuple[PreferenceOption | None, str | None]:
    """Resolve an explicit option id first, then a visible label; never guess."""
    if option_id:
        for option in prompt.options:
            if option.option_id == option_id:
                return option, None
        return None, "the selected preference option is not available"
    text = " ".join(str(message or "").split()).casefold()
    if not text:
        return None, "a preference option is required"
    for option in prompt.options:
        if option.label.casefold() == text:
            return option, None
    # A visible option label may be sent as a substring by a client; this is still an exact
    # catalogued option, not a free-text preference.
    matches = [option for option in prompt.options if option.label.casefold() in text]
    if len(matches) == 1:
        return matches[0], None
    return None, "the selected preference option is not available"


def preference_from_option(
    prompt: PreferencePrompt, option: PreferenceOption, *, turn_id: str = ""
) -> TaskPreference:
    """Turn an executable option into the current task's soft preference."""
    if option.action not in {"lower", "higher", "value"}:
        raise ValueError(f"{option.action!r} does not create a preference")
    return TaskPreference(
        attribute=option.attribute,
        action=option.action,
        value=option.value,
        option_id=option.option_id,
        label=option.label,
    )


def preference_kind_for(attribute: str) -> str:
    """The intent kind used for a task preference.

    Numeric directions are represented as ``price``/``weight`` with a direction value; this is
    deliberately a *soft* intent entry, not a hard price bound.
    """
    return str(attribute)


def augment_intent(intent: ShoppingIntent, preference: TaskPreference) -> ShoppingIntent:
    """Add the selected task preference as a soft intent entry.

    The task choice is never written to the preference store.  It is a current-intent signal
    and therefore lives only in the response intent projection and the shopping task state.
    """
    value = (
        str(preference.action)
        if preference.action in {"lower", "higher"}
        else str(preference.value)
    )
    entry = IntentEntry(
        kind=preference_kind_for(preference.attribute),
        value=value,
        polarity="prefer",
        strength=Strength.SOFT,
        source_turn_id="task-preference",
        source_text=preference.label,
    )
    soft = tuple(intent.soft)
    if not any(
        item.kind == entry.kind and item.value.casefold() == entry.value.casefold()
        for item in soft
    ):
        soft = (*soft, entry)
    return ShoppingIntent(
        hard=intent.hard,
        soft=soft,
        conflicts=intent.conflicts,
        turn_ids=intent.turn_ids,
        active_preference_count=intent.active_preference_count + 1,
    )


def _fact_value(records: Mapping[str, Any], identity: str, attribute: str) -> tuple[Any, bool]:
    try:
        inspection = ProductResearchTool(records).inspect_product(
            identity, attributes=(attribute,)
        )
    except Exception:
        return None, False
    fact = inspection.facts.get(attribute)
    if fact is None or fact.status != "value":
        return None, False
    return fact.value, True


def _matches(attribute: str, value: Any, preference: TaskPreference) -> bool:
    if attribute in {"price", "weight"}:
        return True
    if preference.value is None:
        return False
    return str(preference.value).casefold() in _value_text(value).casefold()


def rank_with_task_preference(
    candidates: Sequence[Any],
    records: Mapping[str, Any],
    preference: TaskPreference | None,
) -> PreferenceRanking | None:
    """Reorder already-hard-filtered candidates by a task soft preference.

    The candidate set is never changed.  Unknown values are placed after known values; equal
    evidence preserves the baseline order.  A categorical preference puts matches first and
    leaves non-matches in their original order rather than hiding them.
    """
    if preference is None or len(candidates) < 2:
        return None
    baseline = tuple(str(item.parent_asin) for item in candidates)
    rows: list[tuple[int, float, int, str, Any, bool]] = []
    for position, item in enumerate(candidates):
        value, known = _fact_value(records, str(item.parent_asin), preference.attribute)
        if not known:
            group = 2
            numeric_key = 0.0
        elif preference.attribute in {"price", "weight"}:
            try:
                numeric_key = float(value)
            except (TypeError, ValueError):
                group = 2
                numeric_key = 0.0
            else:
                group = 0
                if preference.action == "higher":
                    numeric_key = -numeric_key
        else:
            group = 0 if _matches(preference.attribute, value, preference) else 1
            numeric_key = 0.0
        rows.append((group, numeric_key, position, str(item.parent_asin), value, known))
    rows.sort(key=lambda row: (row[0], row[1], row[2]))
    ranked: list[PreferenceRankedCandidate] = []
    for new_rank, row in enumerate(rows, start=1):
        _, _, position, identity, value, known = row
        item = candidates[position]
        matched = known and (row[0] == 0)
        ranked.append(
            PreferenceRankedCandidate(
                parent_asin=identity,
                item_id=int(getattr(item, "item_id", position + 1)),
                rank=new_rank,
                reranked_rank=new_rank,
                match_count=1 if matched else 0,
                violation_count=0,
                unknown_count=0 if known else 1,
                evidence=(),
                reason=preference.label,
            )
        )
    ranked_order = tuple(candidate.parent_asin for candidate in ranked)
    return PreferenceRanking(
        candidates=tuple(ranked),
        baseline_order=baseline,
        ranked_order=ranked_order,
        changed_order=ranked_order != baseline,
    )


def blocked_attributes_from_intent(intent: ShoppingIntent) -> set[str]:
    """Attributes already specified (hard or soft) in the current task intent."""
    mapping = {
        "price_max": "price",
        "price_min": "price",
        "price": "price",
        "weight": "weight",
        "color": "color",
        "material": "material",
        "brand": "brand",
    }
    result: set[str] = set()
    for entry in (*intent.hard, *intent.soft):
        canonical = mapping.get(str(entry.kind).strip().lower())
        if canonical:
            result.add(canonical)
    return result


def blocked_attributes_from_history(entries: Sequence[Any]) -> set[str]:
    """Attributes with a revoked/superseded preference in the existing memory history."""
    mapping = {
        PreferenceKind.PRICE_MAX.value: "price",
        PreferenceKind.PRICE_MIN.value: "price",
        PreferenceKind.COLOR.value: "color",
        PreferenceKind.MATERIAL.value: "material",
        PreferenceKind.BRAND.value: "brand",
    }
    result: set[str] = set()
    for entry in entries:
        if getattr(entry, "is_active", False):
            continue
        kind = getattr(getattr(entry, "kind", None), "value", None)
        canonical = mapping.get(str(kind))
        if canonical:
            result.add(canonical)
    return result

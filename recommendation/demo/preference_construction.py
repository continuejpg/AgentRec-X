"""Candidate-aware, context-sensitive preference construction.

The module is deliberately pure and deterministic.  It turns **trusted candidate evidence**
into one bounded interaction decision:

1. build comparable attribute evidence from the accepted product-research tool;
2. mark already-known / confirmed / revoked attributes as unavailable;
3. simulate the top-K consequences of each executable answer;
4. choose an interaction only when the evidence and the counterfactual effect justify it.

It is a heuristic decision policy, not a learned optimal policy and not an information-gain
estimate.  The values it exposes are copied from the catalogue or are explicitly absent.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence
from urllib.parse import quote

from recommendation.control.product_research import ProductResearchTool
from recommendation.memory import PreferenceKind

from .shopping_intent import IntentEntry, ShoppingIntent, Strength

__all__ = [
    "PREFERENCE_ATTRIBUTES",
    "PreferenceDecision",
    "PreferenceOption",
    "PreferencePrompt",
    "PreferenceRanking",
    "PreferenceRankedCandidate",
    "PreferenceState",
    "TaskPreference",
    "augment_intent",
    "blocked_attributes_from_history",
    "blocked_attributes_from_intent",
    "build_preference_prompt",
    "choose_preference_action",
    "preference_from_option",
    "rank_with_task_preference",
    "rank_with_task_preferences",
    "resolve_preference_option",
]

#: Attributes the tool can turn into a preference question.  This is the evidence surface, not
#: a priority order: final selection is by contextual decision value.
PREFERENCE_ATTRIBUTES: tuple[str, ...] = ("price", "weight", "color", "material", "brand")

_ATTRIBUTE_LABELS = {
    "price": "price",
    "weight": "weight",
    "color": "color",
    "material": "material",
    "brand": "brand",
}

#: Context terms that make an attribute relevant to the current request.  This is deliberately
#: lexical and inspectable; it is not a semantic model.
_CONTEXT_TERMS: dict[str, tuple[str, ...]] = {
    "price": ("price", "budget", "cheap", "expensive", "cost", "afford", "dollar", "$"),
    "weight": (
        "weight", "light", "lighter", "lightweight", "heavy", "portable", "hiking",
        "camping", "backpacking", "carry",
    ),
    "color": ("color", "colour", "red", "blue", "black", "green", "style"),
    "material": ("material", "leather", "cotton", "fabric", "waterproof", "durable"),
    "brand": ("brand", "manufacturer", "make"),
}

_NUMERIC_DIRECTIONS = {
    "price": ("lower", "higher"),
    "weight": ("lower", "higher"),
}


class _PreferenceStatus:
    """Machine-readable statuses used by the decision inputs (string values on the wire)."""

    KNOWN = "known"
    CONFIRMED = "confirmed"
    UNCONFIRMED = "unconfirmed"
    REVOKED = "revoked"


@dataclass(frozen=True)
class PreferenceState:
    """The current task's preference knowledge.

    The distinction matters: an explicit/confirmed preference must not be asked again, while
    an unconfirmed dimension is exactly what candidate-aware discovery may explore.
    """

    explicit: frozenset[str] = frozenset()
    confirmed: frozenset[str] = frozenset()
    revoked: frozenset[str] = frozenset()
    pending: frozenset[str] = frozenset()


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
    """Evidence and projected consequences for one preference question."""

    attribute: str
    label: str
    reason: str
    products: tuple[str, ...]
    values: Mapping[str, Any]
    comparison_kind: str  # numeric | categorical
    options: tuple[PreferenceOption, ...]
    #: Decision-policy diagnostics.  They are deterministic inputs to the choice, not claims
    #: of statistical confidence or learned utility.
    interaction: str = "SHOW_CONTRAST"  # ASK_DIRECT | SHOW_CONTRAST | EXPLAIN
    status: str = _PreferenceStatus.UNCONFIRMED
    coverage: float = 1.0
    difference: float = 0.0
    relevance: float = 1.0
    impact: float = 0.0
    utility: float = 0.0
    baseline_top_k: tuple[str, ...] = ()
    counterfactual: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
    explanation: str | None = None


@dataclass(frozen=True)
class PreferenceDecision:
    """The result of the four-step decision: ask or skip."""

    action: str  # ASK_DIRECT | SHOW_CONTRAST | EXPLAIN | SKIP
    dimension: PreferencePrompt | None
    reason: str
    second_round: bool = False


@dataclass(frozen=True)
class TaskPreference:
    """A user-confirmed current-task soft preference (never a memory entry)."""

    attribute: str
    action: str  # lower | higher | value
    value: str | float | None
    option_id: str
    label: str
    source: str = "user_selected"
    scope: str = "task"
    confirmed: bool = True


@dataclass(frozen=True)
class PreferenceRankedCandidate:
    """A candidate's position under the current task preferences."""

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


def _normalise_ids(candidate_ids: Sequence[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(str(item) for item in candidate_ids if str(item)))[:5]


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


def _pref_key(
    identity: str,
    records: Mapping[str, Any],
    preference: TaskPreference,
    baseline_position: int,
) -> tuple[int, float, int]:
    """A stable sort key for one task preference.

    Unknown values sort after known values.  Equal evidence falls back to the baseline
    position, so a preference never invents a new candidate or a new tie-break rule.
    """
    value, known = _fact_value(records, identity, preference.attribute)
    if not known:
        return 2, 0.0, baseline_position
    if preference.attribute in {"price", "weight"}:
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return 2, 0.0, baseline_position
        if preference.action == "higher":
            numeric = -numeric
        return 0, numeric, baseline_position
    group = 0 if _matches(preference.attribute, value, preference) else 1
    return group, 0.0, baseline_position


def _ordered_ids(
    candidate_ids: Sequence[str],
    records: Mapping[str, Any],
    preferences: Sequence[TaskPreference],
) -> tuple[str, ...]:
    order = list(_normalise_ids(candidate_ids))
    baseline = {identity: position for position, identity in enumerate(order)}
    # Stable sorts from earlier to newest; the newest confirmed preference is the most
    # specific refinement, while every earlier choice remains a preserved tie-break.
    for preference in tuple(preferences):
        order.sort(key=lambda identity: _pref_key(identity, records, preference, baseline[identity]))
    return tuple(order)


def _difference_score(attribute: str, values: Sequence[Any]) -> float:
    if len(values) < 2:
        return 0.0
    rendered = {_value_text(value) for value in values}
    if attribute in {"price", "weight"}:
        try:
            numeric = [float(value) for value in values]
        except (TypeError, ValueError):
            return 0.0
        if not numeric:
            return 0.0
        spread = max(numeric) - min(numeric)
        scale = max(abs(max(numeric)), abs(min(numeric)), 1e-9)
        return max(0.0, min(1.0, spread / scale))
    # Category spread: the fraction of candidates that are not in the largest group.
    from collections import Counter

    counts = Counter(rendered)
    if len(counts) < 2:
        return 0.0
    return max(0.0, min(1.0, 1.0 - max(counts.values()) / len(values)))


def _top_k_change(before: Sequence[str], after: Sequence[str]) -> float:
    if not before and not after:
        return 0.0
    count = max(len(before), len(after), 1)
    position_changes = sum(
        1
        for index in range(min(len(before), len(after)))
        if before[index] != after[index]
    )
    set_changes = len(set(before).symmetric_difference(set(after)))
    return min(1.0, (position_changes + set_changes) / (2 * count))


def _relevance(message: str, attribute: str) -> float:
    text = " ".join(str(message or "").casefold().split())
    hits = sum(1 for term in _CONTEXT_TERMS.get(attribute, ()) if term in text)
    return 1.0 + 2.0 * min(hits, 2)


def _reason_for(attribute: str, kind: str, values: Mapping[str, Any]) -> str:
    if kind == "numeric":
        numbers = [float(value) for value in values.values()]
        if attribute == "price":
            rendered = ", ".join(f"${number:.2f}" for number in numbers)
        elif attribute == "weight":
            rendered = ", ".join(f"{number:.3g} kg" for number in numbers)
        else:
            rendered = ", ".join(f"{number:g}" for number in numbers)
        return f"Candidates differ on {attribute}: {rendered}."
    rendered = ", ".join(_value_text(value) for value in values.values())
    return f"Candidates offer different {attribute} values: {rendered}."


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


def _task_preference(attribute: str, action: str, value: Any, option_id: str, label: str) -> TaskPreference:
    return TaskPreference(
        attribute=attribute,
        action=action,
        value=value,
        option_id=option_id,
        label=label,
    )


def _options_for_row(attribute: str, row: Any, values: Mapping[str, Any]) -> tuple[PreferenceOption, ...]:
    kind = str(row.comparison.get("kind") or "") if row.comparison else ""
    options: list[PreferenceOption] = []
    if kind == "numeric":
        directions = _NUMERIC_DIRECTIONS.get(attribute, ("lower", "higher"))
        numeric = [float(value) for value in values.values()]
        for action in directions:
            options.append(
                PreferenceOption(
                    option_id=_option_id(attribute, action),
                    label=_numeric_label(attribute, action, numeric),
                    action=action,
                    attribute=attribute,
                )
            )
    else:
        distinct: list[Any] = []
        for value in values.values():
            text = _value_text(value)
            if not any(text == _value_text(existing) for existing in distinct):
                distinct.append(value)
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
    return tuple(options)


def _decide_dimension(
    *,
    attribute: str,
    row: Any,
    identities: tuple[str, ...],
    records: Mapping[str, Any],
    message: str,
    current_preferences: Sequence[TaskPreference],
    baseline_top_k: tuple[str, ...],
    top_k: int,
    round_no: int,
) -> PreferencePrompt | None:
    if row.status != "comparable" or row.comparison is None:
        return None
    values = dict(row.comparison.get("values") or {})
    if len(values) < 2 or any(identity not in values for identity in identities):
        return None
    kind = str(row.comparison.get("kind") or "")
    comparison_kind = "numeric" if kind == "numeric" else "categorical"
    difference = _difference_score(attribute, list(values.values()))
    if difference <= 0:
        return None
    options = _options_for_row(attribute, row, values)
    preference_options = [option for option in options if option.action in {"lower", "higher", "value"}]
    counterfactual: dict[str, tuple[str, ...]] = {}
    impact = 0.0
    for option in preference_options:
        preference = _task_preference(
            attribute, option.action, option.value, option.option_id, option.label
        )
        order = _ordered_ids(
            identities, records, (*current_preferences, preference)
        )
        projected = order[: max(int(top_k), 0)]
        counterfactual[option.option_id] = projected
        impact = max(impact, _top_k_change(baseline_top_k, projected))
    if not preference_options or impact <= 0:
        return None

    relevance = _relevance(message, attribute)
    coverage = min(1.0, len(values) / len(identities))
    # Interaction cost is one bounded question; it is not hidden in a learned score.
    interaction = "SHOW_CONTRAST"
    explanation = None
    if current_preferences and round_no > 1 and relevance >= 2:
        # A prior confirmed task choice is a reliable basis for explaining why another
        # dimension is still worth asking about.
        interaction = "EXPLAIN"
        explanation = (
            "Your previous task choice is already applied; this remaining difference "
            "can still change the current top results."
        )
    elif len(identities) == 2 and relevance >= 2:
        interaction = "ASK_DIRECT"
    utility = (relevance * coverage * difference * impact) / 1.0
    return PreferencePrompt(
        attribute=attribute,
        label=_ATTRIBUTE_LABELS[attribute],
        reason=_reason_for(attribute, kind, values),
        products=identities,
        values=values,
        comparison_kind=comparison_kind,
        options=options,
        interaction=interaction,
        status=_PreferenceStatus.UNCONFIRMED,
        coverage=coverage,
        difference=difference,
        relevance=relevance,
        impact=impact,
        utility=utility,
        baseline_top_k=baseline_top_k,
        counterfactual=counterfactual,
        explanation=explanation,
    )


def choose_preference_action(
    candidate_ids: Sequence[str],
    records: Mapping[str, Any],
    *,
    intent: ShoppingIntent | None = None,
    current_preferences: Sequence[TaskPreference] = (),
    blocked_attributes: Sequence[str] = (),
    message: str = "",
    top_k: int = 5,
    round_no: int = 1,
    state: PreferenceState | None = None,
) -> PreferenceDecision:
    """Choose ASK_DIRECT / SHOW_CONTRAST / EXPLAIN, or SKIP, deterministically.

    A dimension is eligible only when every candidate has a trustworthy comparable value,
    it is not explicit/confirmed/revoked, it varies, and at least one executable answer
    changes the current top-K.  The chosen dimension maximises contextual relevance, coverage,
    normalized difference and counterfactual impact; this is a heuristic, not a learned
    optimum.
    """
    identities = _normalise_ids(candidate_ids)
    if len(identities) < 2:
        return PreferenceDecision("SKIP", None, "fewer than two grounded candidates")
    blocked = {
        str(item).strip().casefold()
        for item in (*blocked_attributes,)
        if str(item).strip()
    }
    if intent is not None:
        blocked |= blocked_attributes_from_intent(intent)
    if state is not None:
        blocked |= state.explicit | state.confirmed | state.revoked
    blocked |= {preference.attribute for preference in current_preferences}
    try:
        comparison = ProductResearchTool(records).compare_products(
            identities, attributes=PREFERENCE_ATTRIBUTES
        )
    except Exception:
        return PreferenceDecision("SKIP", None, "trusted candidate comparison is unavailable")
    baseline_order = _ordered_ids(identities, records, current_preferences)
    baseline_top_k = baseline_order[: max(int(top_k), 0)]
    dimensions: list[PreferencePrompt] = []
    for row in comparison.rows:
        attribute = str(row.attribute)
        if attribute in blocked:
            continue
        prompt = _decide_dimension(
            attribute=attribute,
            row=row,
            identities=identities,
            records=records,
            message=message,
            current_preferences=current_preferences,
            baseline_top_k=baseline_top_k,
            top_k=top_k,
            round_no=round_no,
        )
        if prompt is not None:
            dimensions.append(prompt)
    if not dimensions:
        return PreferenceDecision(
            "SKIP",
            None,
            "no reliable attribute difference changes the current top-K",
        )
    chosen = max(
        dimensions,
        key=lambda prompt: (
            prompt.utility,
            prompt.impact,
            prompt.difference,
            prompt.attribute,
        ),
    )
    return PreferenceDecision(chosen.interaction, chosen, chosen.reason, second_round=round_no > 1)


def build_preference_prompt(
    candidate_ids: Sequence[str],
    records: Mapping[str, Any],
    *,
    blocked_attributes: Sequence[str] = (),
) -> PreferencePrompt | None:
    """Backward-compatible wrapper for the first decision's chosen evidence.

    The decision is still made by :func:`choose_preference_action`; this helper only returns
    the dimension when the policy chooses to ask.  It exits with ``None`` for SKIP.
    """
    decision = choose_preference_action(
        candidate_ids,
        records,
        blocked_attributes=blocked_attributes,
        top_k=len(_normalise_ids(candidate_ids)),
    )
    return decision.dimension


def resolve_preference_option(
    prompt: PreferencePrompt,
    *,
    option_id: str | None,
    message: str = "",
) -> tuple[PreferenceOption | None, str | None]:
    """Resolve an explicit option id first, then an exact visible label; never guess."""
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
    matches = [option for option in prompt.options if option.label.casefold() in text]
    if len(matches) == 1:
        return matches[0], None
    return None, "the selected preference option is not available"


def preference_from_option(
    prompt: PreferencePrompt, option: PreferenceOption, *, turn_id: str = ""
) -> TaskPreference:
    """Turn an executable option into the current task's confirmed soft preference."""
    if option.action not in {"lower", "higher", "value"}:
        raise ValueError(f"{option.action!r} does not create a preference")
    return TaskPreference(
        attribute=option.attribute,
        action=option.action,
        value=option.value,
        option_id=option.option_id,
        label=option.label,
    )


def augment_intent(intent: ShoppingIntent, preference: TaskPreference) -> ShoppingIntent:
    """Add a user-confirmed task preference as a soft intent entry."""
    value = (
        str(preference.action)
        if preference.action in {"lower", "higher"}
        else str(preference.value)
    )
    entry = IntentEntry(
        kind=preference.attribute,
        value=value,
        polarity="prefer",
        strength=Strength.SOFT,
        source_turn_id=preference.source or "task-preference",
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


def rank_with_task_preferences(
    candidates: Sequence[Any],
    records: Mapping[str, Any],
    preferences: Sequence[TaskPreference],
) -> PreferenceRanking | None:
    """Reorder already-hard-filtered candidates by one or more task soft preferences.

    Preferences are applied from oldest to newest, so the newest confirmed choice is the
    primary refinement and earlier choices remain deterministic tie-breaks.  This is what lets
    a second-round preference change the current Top-K when the first round already narrowed it.
    """
    prefs = tuple(preferences)
    if not prefs or len(candidates) < 2:
        return None
    baseline = tuple(str(item.parent_asin) for item in candidates)
    positions = {identity: index for index, identity in enumerate(baseline)}
    ordered = list(range(len(candidates)))
    for preference in prefs:
        ordered.sort(
            key=lambda index: _pref_key(
                baseline[index], records, preference, positions[baseline[index]]
            )
        )
    ranked: list[PreferenceRankedCandidate] = []
    for new_rank, index in enumerate(ordered, start=1):
        item = candidates[index]
        identity = baseline[index]
        matched = False
        unknown = 0
        for preference in prefs:
            value, known = _fact_value(records, identity, preference.attribute)
            if not known:
                unknown += 1
            elif preference.attribute in {"price", "weight"} or _matches(
                preference.attribute, value, preference
            ):
                matched = True
        ranked.append(
            PreferenceRankedCandidate(
                parent_asin=identity,
                item_id=int(getattr(item, "item_id", index + 1)),
                rank=new_rank,
                reranked_rank=new_rank,
                match_count=1 if matched else 0,
                violation_count=0,
                unknown_count=unknown,
                evidence=(),
                reason="; ".join(preference.label for preference in prefs),
            )
        )
    ranked_order = tuple(candidate.parent_asin for candidate in ranked)
    return PreferenceRanking(
        candidates=tuple(ranked),
        baseline_order=baseline,
        ranked_order=ranked_order,
        changed_order=ranked_order != baseline,
    )


def rank_with_task_preference(
    candidates: Sequence[Any],
    records: Mapping[str, Any],
    preference: TaskPreference | None,
) -> PreferenceRanking | None:
    """Backward-compatible single-preference wrapper."""
    return (
        None
        if preference is None
        else rank_with_task_preferences(candidates, records, (preference,))
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

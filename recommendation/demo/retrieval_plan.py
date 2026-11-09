"""Minimal memory-guided retrieval planning for the shopping experiment (R1).

This is intentionally *not* another retrieval engine.  It decides between the two
actions the existing DemoAgentService already supports:

* ``RERANK`` - the current trusted candidate window is sufficient, so keep it;
* ``RETRIEVE`` - one bounded recall pass with an updated lexical query/source plan.

The plan is derived only from the user's current request, confirmed task preferences,
and trusted catalogue attributes.  Numeric preferences (price/weight) never rewrite the
query and never trigger recall: they are ordering signals.  A categorical value that is
missing from the current candidate window may trigger a recall only when the value is
trusted and a catalogue source is available.  Missing attributes stay UNKNOWN.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

__all__ = [
    "ENV_R1_DYNAMIC_RETRIEVAL",
    "RetrievalPlan",
    "dynamic_retrieval_enabled",
    "plan_dynamic_retrieval",
]

ENV_R1_DYNAMIC_RETRIEVAL = "AGENTRECX_R1_DYNAMIC_RETRIEVAL"
_TRUE_VALUES = frozenset({"1", "true", "on", "yes", "enabled"})
_NUMERIC_ATTRIBUTES = frozenset({"price", "weight"})


def dynamic_retrieval_enabled(value: str | None = None) -> bool:
    """Whether R1 dynamic retrieval is explicitly enabled (default off)."""
    import os

    raw = value if value is not None else os.environ.get(ENV_R1_DYNAMIC_RETRIEVAL, "")
    return str(raw).strip().casefold() in _TRUE_VALUES


def _attribute_text(record: Any, attribute: str) -> str | None:
    """Read one trusted catalogue attribute, or return None (UNKNOWN stays unknown)."""
    if record is None:
        return None
    name = str(attribute).casefold()
    if name in {"color", "colour"}:
        keys = ("Color", "Colour")
    elif name == "material":
        keys = ("Material",)
    elif name == "brand":
        keys = ("Brand Name", "Brand", "Manufacturer")
    elif name == "store":
        keys = ("Store",)
    else:
        keys = ()
    if keys:
        for key, value in getattr(record, "details", ()) or ():
            if str(key).casefold() in {item.casefold() for item in keys}:
                text = str(value).strip()
                return text or None
        if name in {"brand", "store"}:
            value = getattr(record, "store", None)
            return str(value).strip() if value and str(value).strip() else None
        return None
    if name == "category":
        categories = tuple(getattr(record, "categories", ()) or ())
        main = getattr(record, "main_category", None)
        values = [str(item) for item in categories]
        if main:
            values.append(str(main))
        return " / ".join(values) if values else None
    return None


def _matches_preference(record: Any, preference: Any) -> bool:
    attribute = str(getattr(preference, "attribute", "") or "").casefold()
    value = getattr(preference, "value", None)
    if attribute in _NUMERIC_ATTRIBUTES or value is None:
        return False
    trusted = _attribute_text(record, attribute)
    if trusted is None:
        return False
    needle = " ".join(str(value).split()).casefold()
    return bool(needle) and needle in " ".join(trusted.split()).casefold()


def _query_terms(intent: Any, message: str, active_preferences: Sequence[Any]) -> tuple[str, ...]:
    """Query terms from the existing intent projection plus confirmed categorical values."""
    terms: list[str] = []
    if intent is not None:
        try:
            from .shopping_intent import query_terms

            terms.extend(str(item) for item in query_terms(intent, message))
        except Exception:  # noqa: BLE001 - fall back to the message if no intent is available
            pass
    if not terms:
        from recommendation.control import SearchCatalogArguments

        terms.extend(
            str(item)
            for item in getattr(SearchCatalogArguments.from_text(message, limit=8), "terms", ())
            or ()
        )
    seen = {term.casefold() for term in terms}
    for preference in active_preferences:
        if str(getattr(preference, "attribute", "")).casefold() in _NUMERIC_ATTRIBUTES:
            continue
        value = getattr(preference, "value", None)
        if value is None:
            continue
        text = " ".join(str(value).split())
        if text and text.casefold() not in seen:
            terms.append(text)
            seen.add(text.casefold())
    return tuple(terms)


def _unique_sources(
    sources: Sequence[str], available: Sequence[str]
) -> tuple[str, ...]:
    allowed = {str(item).casefold(): str(item) for item in available}
    result: list[str] = []
    for source in sources:
        key = str(source).casefold()
        if key in allowed and allowed[key] not in result:
            result.append(allowed[key])
    if "catalog_search" in allowed and "catalog_search" not in result:
        result.append("catalog_search")
    return tuple(result)


@dataclass(frozen=True)
class RetrievalPlan:
    """The two-action plan consumed by the service for one answered preference."""

    action: str  # RERANK | RETRIEVE
    reason: str
    query_terms: tuple[str, ...] = ()
    sources: tuple[str, ...] = ()
    matched_candidates: int = 0
    changed_attributes: tuple[str, ...] = ()

    @property
    def should_retrieve(self) -> bool:
        return self.action == "RETRIEVE"


def plan_dynamic_retrieval(
    message: str,
    candidate_ids: Sequence[str],
    records: Mapping[str, Any],
    *,
    active_preferences: Sequence[Any] = (),
    changed_preferences: Sequence[Any] = (),
    intent: Any = None,
    sources: Sequence[str] = ("history", "catalog_search"),
    available_sources: Sequence[str] = ("history", "catalog_search"),
    already_retrieved: bool = False,
) -> RetrievalPlan:
    """Decide whether an answered preference needs one bounded recall pass.

    ``changed_preferences`` are the options the user just confirmed.  Numeric-only
    changes are always RERANK.  A trusted categorical value is compared against the
    current candidate window; if fewer than two candidates carry it, and the query can be
    extended without inventing anything, the plan is RETRIEVE.  The function never
    mutates candidates, state, the ledger, or any model.
    """
    if already_retrieved:
        return RetrievalPlan("RERANK", "retrieval_budget_used")
    changed = tuple(
        preference
        for preference in changed_preferences
        if str(getattr(preference, "attribute", "")).casefold() not in _NUMERIC_ATTRIBUTES
    )
    if not changed:
        return RetrievalPlan("RERANK", "numeric_or_no_changed_preference")

    identities = tuple(str(item) for item in candidate_ids if str(item))
    matched = sum(
        1
        for identity in identities
        for preference in changed
        if _matches_preference(records.get(identity), preference)
    )
    # One trusted current candidate already satisfies the confirmed value.  A recall
    # pass can find more options, but the existing window is not insufficient for the
    # stated need; do not spend recall budget just because the set is small.
    if matched >= 1:
        return RetrievalPlan(
            "RERANK",
            "changed_preference_already_covered",
            matched_candidates=matched,
            changed_attributes=tuple(
                str(getattr(preference, "attribute", "")) for preference in changed
            ),
        )

    terms = _query_terms(intent, message, active_preferences)
    sources_plan = _unique_sources(sources, available_sources)
    if "catalog_search" not in sources_plan:
        return RetrievalPlan(
            "RERANK",
            "no_lexical_source_for_query_update",
            matched_candidates=matched,
            changed_attributes=tuple(
                str(getattr(preference, "attribute", "")) for preference in changed
            ),
        )
    return RetrievalPlan(
        "RETRIEVE",
        "changed_categorical_preference_not_covered",
        query_terms=terms,
        sources=sources_plan,
        matched_candidates=matched,
        changed_attributes=tuple(
            str(getattr(preference, "attribute", "")) for preference in changed
        ),
    )

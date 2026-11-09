"""Bounded M5 soft-preference → M6.1 reranking adapter.

The M5 state projector already emits a typed ``RankerStateProjection`` with active,
provenance-bearing features.  This module is the narrow consumer used by the demo
service: it only maps fields that the accepted M6.1 foundation can actually evaluate
against trusted catalogue evidence, then delegates the ordering itself to M6.1
(``preference_construction.rank_with_task_preferences``).

It never creates/filters candidates, never changes a hard constraint, and never reads
Gold/expected identities.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

from .preference_construction import (
    PreferenceRanking,
    TaskPreference,
    rank_with_task_preferences,
)

__all__ = [
    "SoftRerankPreference",
    "merge_soft_preferences",
    "rank_with_m5_soft_preferences",
    "soft_preferences_from_ranker_projection",
]


@dataclass(frozen=True)
class SoftRerankPreference:
    """One confirmed M5 preference that M6.1 is allowed to consume."""

    preference_id: str
    attribute: str
    action: str
    value: str | float | None
    label: str
    source: str = "m5_soft"
    scope: str = "task"
    polarity: str = "prefer"

    def as_task_preference(self) -> TaskPreference:
        return TaskPreference(
            attribute=self.attribute,
            action=self.action,
            value=self.value,
            option_id=f"m5:{self.preference_id}",
            label=self.label,
            source=self.source,
            scope=self.scope,
            confirmed=True,
        )


def _text(value: Any) -> str:
    return str(getattr(value, "value", value) or "").strip().casefold()


def _feature_to_soft(feature: Any) -> SoftRerankPreference | None:
    """Map only fields whose semantics M6.1 can ground in the catalogue.

    * ``feature`` + ``weight`` + ``minimize/maximize`` -> numeric weight ordering;
    * ``color``/``material`` + prefer -> categorical equality ordering.

    Everything else (category/feature free text, avoid polarity, inferred hypotheses)
    is deliberately left out instead of being guessed into an attribute.
    """
    if bool(getattr(feature, "is_soft", True)):
        return None
    lifecycle = _text(getattr(feature, "lifecycle", ""))
    if lifecycle not in {"explicit", "confirmed"}:
        return None
    kind = _text(getattr(feature, "kind", ""))
    direction = _text(getattr(feature, "direction", ""))
    value = str(getattr(feature, "value", "") or "").strip()
    preference_id = str(getattr(feature, "preference_id", "") or "")
    scope = _text(getattr(feature, "scope", "task")) or "task"
    if not preference_id or not value:
        return None

    if kind == "feature":
        normalized = value.casefold()
        # The rule extractor can carry the user's surface phrase ("lighter option")
        # instead of the canonical attribute name.  Map only unambiguous weight
        # vocabulary; do not guess unrelated feature text.
        if normalized == "weight" or normalized in {
            "lighter",
            "lighter option",
            "lightest",
            "lightweight",
            "lower weight",
            "minimize weight",
        }:
            if direction in {"minimize", "lower", ""}:
                return SoftRerankPreference(
                    preference_id,
                    "weight",
                    "lower",
                    None,
                    "M5 confirmed preference: lower weight",
                    scope=scope,
                )
            if direction in {"maximize", "higher"}:
                return SoftRerankPreference(
                    preference_id,
                    "weight",
                    "higher",
                    None,
                    "M5 confirmed preference: higher weight",
                    scope=scope,
                )
            return None

    if kind in {"color", "material"} and _text(getattr(feature, "polarity", "prefer")) == "prefer":
        return SoftRerankPreference(
            preference_id,
            kind,
            "value",
            value,
            f"M5 confirmed preference: {kind}={value}",
            scope=scope,
        )
    return None


def soft_preferences_from_ranker_projection(
    projection: Any | None,
) -> tuple[SoftRerankPreference, ...]:
    """Return only confirmed, supported preferences from ``for_ranker``."""
    if projection is None:
        return ()
    result: list[SoftRerankPreference] = []
    seen: set[tuple[str, str, str]] = set()
    for feature in getattr(projection, "features", ()) or ():
        preference = _feature_to_soft(feature)
        if preference is None:
            continue
        key = (preference.attribute, preference.action, str(preference.value))
        if key in seen:
            continue
        seen.add(key)
        result.append(preference)
    return tuple(result)


def merge_soft_preferences(
    existing: Sequence[TaskPreference],
    soft: Iterable[SoftRerankPreference],
) -> tuple[TaskPreference, ...]:
    """Merge without double-weighting.  Legacy/task-confirmed inputs keep precedence."""
    merged: list[TaskPreference] = list(existing)
    seen = {
        (preference.attribute, preference.action, str(preference.value))
        for preference in merged
    }
    for item in soft:
        key = (item.attribute, item.action, str(item.value))
        if key in seen:
            continue
        seen.add(key)
        merged.append(item.as_task_preference())
    return tuple(merged)


def rank_with_m5_soft_preferences(
    candidates: Sequence[Any],
    records: Mapping[str, Any],
    projection: Any | None,
    *,
    existing: Sequence[TaskPreference] = (),
) -> tuple[PreferenceRanking | None, tuple[TaskPreference, ...], tuple[SoftRerankPreference, ...]]:
    """M6.1 rerank over the unchanged candidate set; no candidate is added/removed."""
    soft = soft_preferences_from_ranker_projection(projection)
    if not soft:
        return None, tuple(existing), ()
    combined = merge_soft_preferences(existing, soft)
    ranking = rank_with_task_preferences(candidates, records, combined)
    return ranking, combined, soft

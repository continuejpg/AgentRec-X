"""Unified Event-Sourced Memory feature adapter.

Active-only semantics:
  * positive preferences -> positive match/unknown features
  * negative preferences -> conflict/unknown features
  * removed/superseded events are never active inputs
  * UNKNOWN is never converted to absence / negative evidence
  * multiple active preferences are supported and counted separately
  * query soft preferences are kept separate from historical memory
"""

from __future__ import annotations

from typing import Any

ATTRS = (
    "wireless", "wired", "bluetooth", "microphone", "rechargeable", "noise_cancelling",
    "waterproof", "over_ear", "on_ear", "in_ear", "gaming", "mechanical", "portable",
    "foldable", "usb", "hdmi", "4k", "remote", "led", "rgb", "touch", "ergonomic", "silent",
    "lightweight", "compact", "durable", "adjustable", "fast_charging", "long_battery",
    "professional", "travel", "running", "workout", "office", "home", "kids", "women", "men",
    "black", "white", "blue", "red", "pink", "silver", "gold", "small", "large", "mini",
    "premium", "cheap", "affordable",
)


def item_attrs(item: dict[str, Any]) -> set[str]:
    attributes = item.get("attributes")
    if isinstance(attributes, dict):
        return set(attributes.keys())
    if isinstance(attributes, (list, tuple, set)):
        return set(attributes)
    return set()


def active_preferences(memory: dict[str, Any] | list[dict[str, Any]] | None) -> tuple[list[str], list[str]]:
    """Return (positive values, negative values) from an active snapshot only."""
    positive: list[str] = []
    negative: list[str] = []
    if isinstance(memory, dict):
        for entry in memory.get("positive") or []:
            value = str(entry.get("value") or "").strip().lower()
            if value:
                positive.append(value)
        for entry in memory.get("negative") or []:
            value = str(entry.get("value") or "").strip().lower()
            if value:
                negative.append(value)
    elif isinstance(memory, list):
        for entry in memory:
            value = str(entry.get("value") or "").strip().lower()
            if not value:
                continue
            polarity = str(entry.get("polarity") or "positive").lower()
            if polarity in {"negative", "avoid", "dislike"}:
                negative.append(value)
            else:
                positive.append(value)
    return positive, negative


def structured_memory_text(memory: dict[str, Any] | list[dict[str, Any]] | None) -> str:
    positive, negative = active_preferences(memory)
    lines = ["Active Long-Term Preferences:"]
    lines.append("Positive:")
    if positive:
        lines.extend(f"- {value}: preferred" for value in positive)
    else:
        lines.append("- none")
    lines.append("Negative:")
    if negative:
        lines.extend(f"- {value}: avoided" for value in negative)
    else:
        lines.append("- none")
    return "\n".join(lines)


def natural_memory_text(memory: dict[str, Any] | list[dict[str, Any]] | None) -> str:
    positive, negative = active_preferences(memory)
    parts = []
    if positive:
        parts.append("prefer " + ", ".join(positive))
    if negative:
        parts.append("avoid " + ", ".join(negative))
    return "; ".join(parts)


def candidate_features(
    item: dict[str, Any],
    memory: dict[str, Any] | list[dict[str, Any]] | None,
    *,
    query_soft: list[dict[str, Any]] | None = None,
) -> dict[str, float]:
    attrs = item_attrs(item)
    positive, negative = active_preferences(memory)
    positive_matches = [value for value in positive if value in attrs]
    negative_conflicts = [value for value in negative if value in attrs]
    positive_unknown = [value for value in positive if value not in attrs]
    negative_unknown = [value for value in negative if value not in attrs]
    active_count = len(positive) + len(negative)
    known = len(positive_matches) + len(negative_conflicts)
    soft = list(query_soft or [])
    soft_positive = [str(entry.get("value") or "").lower() for entry in soft if str(entry.get("polarity") or "positive").lower() not in {"negative", "avoid", "dislike"}]
    soft_negative = [str(entry.get("value") or "").lower() for entry in soft if str(entry.get("polarity") or "").lower() in {"negative", "avoid", "dislike"}]
    verified_match = len([value for value in attrs if value in set(positive)])
    verified_conflict = len([value for value in attrs if value in set(negative)])
    attribute_unknown = len([attr for attr in ATTRS if attr not in attrs])
    return {
        "positive_match_count": float(len(positive_matches)),
        "negative_conflict_count": float(len(negative_conflicts)),
        "positive_unknown_count": float(len(positive_unknown)),
        "negative_unknown_count": float(len(negative_unknown)),
        "active_preference_count": float(active_count),
        "preference_evidence_coverage": float(known / active_count) if active_count else 0.0,
        "positive_match_rate": float(len(positive_matches) / max(1, len(positive))),
        "negative_conflict_rate": float(len(negative_conflicts) / max(1, len(negative))),
        "query_soft_preference_match": float(
            sum(1 for value in soft_positive if value in attrs) / max(1, len(soft_positive)) if soft_positive else 0.0
        ),
        "query_soft_preference_conflict": float(
            sum(1 for value in soft_negative if value in attrs) / max(1, len(soft_negative)) if soft_negative else 0.0
        ),
        "verified_attribute_match": float(verified_match),
        "verified_attribute_conflict": float(verified_conflict),
        "attribute_unknown_count": float(attribute_unknown),
        "requires_ask": 0.0,
        "positive_preference_count": float(len(positive)),
        "negative_preference_count": float(len(negative)),
    }

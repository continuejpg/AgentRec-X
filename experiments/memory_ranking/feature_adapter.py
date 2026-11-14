"""Reusable feature adapter for a future Memory-Aware Ranker integration.

This module is deliberately an adapter, not a production replacement.  It shows what
can be fed to the trained Amazon-Reviews model and what cannot:

* item metadata features are available from the existing Electronics metadata sidecar;
* training-split popularity features require a supplied popularity table;
* query-relevance features (rank/score) are accepted as a *future* model input, but the
  v1 Amazon-Reviews model was not trained with them;
* Event-Sourced active preferences can provide an **attribute-level proxy** for a
  subset of memory features, but they cannot reconstruct rating-history statistics such
  as the user's historical mean rating, trend, decay, price profile or category
  distribution.

The adapter therefore returns ``model_ready=False`` unless every trained feature is
available.  It never silently pretends the production Event-Sourced Memory is the same
as the ``inferred_from_ratings`` memory used by this dataset.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

from experiments.memory_ranking import features

PRODUCTION_STATUS = "partial_adapter_not_wired_into_production_ranker"

#: Features the trained model expects which the current Event-Sourced memory cannot
#: reconstruct by itself without a rating-history projection.
EVENT_MEMORY_GAPS: tuple[str, ...] = (
    "mem_hist_mean_rating", "mem_hist_std_rating", "mem_hist_min_rating",
    "mem_hist_max_rating", "mem_hist_high_share", "mem_hist_low_share",
    "mem_hist_last_rating", "mem_hist_last2_mean", "mem_hist_last3_mean",
    "mem_hist_last5_mean", "mem_hist_last_minus_prev_mean", "mem_hist_trend_slope",
    "mem_hist_days_since_last", "mem_hist_tenure_days", "mem_hist_avg_gap_days",
    "mem_hist_last_gap_days", "mem_hist_decay_mean_rating", "mem_hist_decay_high_share",
    "mem_hist_decay_low_share", "mem_hist_attr_coverage_mean",
    "mem_hist_high_attr_coverage_mean", "mem_hist_low_attr_coverage_mean",
    "mem_hist_price_mean_log", "mem_hist_high_price_mean_log",
    "mem_hist_low_price_mean_log", "mem_hist_rating_1_share", "mem_hist_rating_2_share",
    "mem_hist_rating_3_share", "mem_hist_rating_4_share", "mem_hist_rating_5_share",
)


@dataclass
class AdapterOutput:
    """Feature vector plus an explicit compatibility report."""

    non_memory: dict[str, float] = field(default_factory=dict)
    memory: dict[str, float] = field(default_factory=dict)
    query_relevance: dict[str, float] = field(default_factory=dict)
    missing_non_memory: list[str] = field(default_factory=list)
    missing_memory: list[str] = field(default_factory=list)
    unknown_memory_values: list[str] = field(default_factory=list)
    model_ready: bool = False
    status: str = PRODUCTION_STATUS
    notes: list[str] = field(default_factory=list)

    def vector(self, names: Sequence[str]) -> list[float]:
        source = {**self.non_memory, **self.memory}
        return [source.get(name, float("nan")) for name in names]

    def memory_off_vector(self) -> list[float]:
        return self.vector(features.non_memory_feature_names())

    def memory_on_vector(self) -> list[float]:
        return self.vector(features.non_memory_feature_names() + features.memory_feature_names())

    def compatibility(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "model_ready": self.model_ready,
            "non_memory_available": len(self.non_memory),
            "non_memory_missing": self.missing_non_memory,
            "memory_available": len(self.memory),
            "memory_missing_count": len(self.missing_memory),
            "memory_missing": self.missing_memory,
            "unknown_memory_values": self.unknown_memory_values,
            "query_relevance_features_supplied": sorted(self.query_relevance),
            "notes": self.notes,
        }


def _attr_for_value(value: str) -> str | None:
    text = str(value or "").strip().lower()
    if not text:
        return None
    for attr in features.ATTRS:
        if text == attr:
            return attr
    for attr, pattern in features.ATTR_PATTERNS.items():
        if pattern.search(text):
            return attr
    return None


def active_memory_features(
    active_preferences: Iterable[Mapping[str, Any]],
    candidate_attributes: Iterable[str],
) -> tuple[dict[str, float], list[str], list[str], list[str]]:
    """Map Event-Sourced active preferences to a subset of the trained memory features.

    Returns ``(features, available_names, missing_names, unknown_values)``.  The
    values are event counts (positive/negative), **not** rating-history counts, so the
    mapping is an explicit proxy and is not directly equivalent to inference-time
    training features.
    """
    positive: dict[str, int] = {attr: 0 for attr in features.ATTRS}
    negative: dict[str, int] = {attr: 0 for attr in features.ATTRS}
    unknown_values: list[str] = []
    count = 0
    for preference in active_preferences:
        value = str(preference.get("value") or "")
        attr = _attr_for_value(value)
        polarity = str(preference.get("polarity") or "positive").lower()
        if attr is None:
            if value:
                unknown_values.append(value)
            continue
        count += 1
        if polarity in {"avoid", "negative", "dislike"}:
            negative[attr] += 1
        else:
            positive[attr] += 1

    candidate = [str(attr) for attr in candidate_attributes if attr in features.ATTRS]
    available: dict[str, float] = {
        "mem_hist_count": float(count),
        "mem_hist_count_log": float(math.log1p(count)),
        "mem_hist_high_count": float(sum(positive.values())),
        "mem_hist_low_count": float(sum(negative.values())),
        "mem_hist_neutral_count": 0.0,
    }
    available["mem_hist_high_share"] = (
        available["mem_hist_high_count"] / count if count else 0.0
    )
    available["mem_hist_low_share"] = (
        available["mem_hist_low_count"] / count if count else 0.0
    )
    available["mem_hist_neutral_share"] = 0.0
    for attr in features.ATTRS:
        available[f"mem_all_attr_count_{attr}"] = float(positive[attr] + negative[attr])
        available[f"mem_high_attr_count_{attr}"] = float(positive[attr])
        available[f"mem_low_attr_count_{attr}"] = float(negative[attr])

    high_match = [attr for attr in candidate if positive[attr] > 0]
    low_conflict = [attr for attr in candidate if negative[attr] > 0]
    seen = [attr for attr in candidate if positive[attr] or negative[attr]]
    high_count = sum(positive.values())
    low_count = sum(negative.values())
    available["mem_candidate_high_match_count"] = float(len(high_match))
    available["mem_candidate_low_conflict_count"] = float(len(low_conflict))
    available["mem_candidate_high_match_rate"] = (
        len(high_match) / len(candidate) if candidate else 0.0
    )
    available["mem_candidate_low_conflict_rate"] = (
        len(low_conflict) / len(candidate) if candidate else 0.0
    )
    available["mem_candidate_high_attr_weight"] = (
        sum(positive[attr] for attr in candidate) / high_count if high_count else 0.0
    )
    available["mem_candidate_low_attr_weight"] = (
        sum(negative[attr] for attr in candidate) / low_count if low_count else 0.0
    )
    available["mem_candidate_attr_seen_ratio"] = (
        len(seen) / len(candidate) if candidate else 0.0
    )
    available["mem_candidate_attr_unknown_count"] = float(
        len(candidate) - len(seen)
    )
    # Category/price/rating-history features are not derivable from event-sourced
    # preferences; they are left missing and reported.
    missing = [
        name for name in features.memory_feature_names() if name not in available
    ]
    return available, sorted(available), missing, unknown_values


def non_memory_features(
    item_id: str,
    raw_metadata: Mapping[str, Any] | None,
    *,
    popularity_context: Mapping[str, Any] | None = None,
) -> tuple[dict[str, float], list[str]]:
    """Build item-metadata + optional popularity features for one candidate.

    ``popularity_context`` must contain ``table``, ``rating``, ``split``,
    ``timestamp``, ``main_category`` and ``leaf_category`` if the caller has an
    authorised historical popularity table.  Without it, all ``pop_*`` features are
    missing; the adapter does not invent them.
    """
    metadata = (
        features.normalise_metadata_item(raw_metadata)
        if raw_metadata is not None
        else features.empty_item_metadata()
    )
    row = features.item_feature_row(item_id, metadata)
    available = {
        name: float(row[name])
        for name in features.item_feature_names()
        if name in row
    }
    missing: list[str] = []
    if popularity_context is not None:
        available.update(
            features.popularity_feature_row(
                item_id=item_id,
                rating=float(popularity_context["rating"]),
                split=str(popularity_context["split"]),
                target_timestamp=int(popularity_context["timestamp"]),
                item_main_category=popularity_context.get("main_category"),
                item_leaf_category=popularity_context.get("leaf_category"),
                table=popularity_context["table"],
                category_counts=popularity_context["table"]["category_counts"],
            )
        )
    else:
        missing = list(features.POP_FEATURES)
    return available, missing


def build_adapter_output(
    *,
    item_id: str,
    raw_metadata: Mapping[str, Any] | None,
    active_preferences: Iterable[Mapping[str, Any]] = (),
    query_relevance: Mapping[str, float] | None = None,
    popularity_context: Mapping[str, Any] | None = None,
) -> AdapterOutput:
    """Compose the two feature groups and report exactly what is missing."""
    non_memory, missing_pop = non_memory_features(
        item_id, raw_metadata, popularity_context=popularity_context
    )
    metadata = (
        features.normalise_metadata_item(raw_metadata)
        if raw_metadata is not None
        else features.empty_item_metadata()
    )
    memory, _available, missing_memory, unknown_values = active_memory_features(
        active_preferences, metadata.get("attributes") or ()
    )
    missing_memory = list(missing_memory)
    model_ready = not missing_pop and not missing_memory
    return AdapterOutput(
        non_memory=non_memory,
        memory=memory,
        query_relevance=dict(query_relevance or {}),
        missing_non_memory=missing_pop,
        missing_memory=missing_memory,
        unknown_memory_values=unknown_values,
        model_ready=model_ready,
        notes=[
            "Event-Sourced active memory is mapped to attribute-level positive/negative counts, not rating-history statistics.",
            "Query relevance is accepted for a future retrained model; the v1 Amazon-Reviews model does not contain query features.",
            "Popularity features require an authorised historical popularity table; they are not inferred from the current turn.",
            "This adapter is not wired into the production Scorer/Gate and changes no production ranking behavior.",
        ],
    )

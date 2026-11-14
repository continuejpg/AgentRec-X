"""Tests for the non-production Memory-Aware Ranker feature adapter."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

pytest.importorskip("pandas")

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.memory_ranking import feature_adapter, features  # noqa: E402

RAW_METADATA = {
    "item_id": "B0TESTITEM",
    "title": "Wireless Over-Ear Headphones with Microphone",
    "features": ["Rechargeable battery", "Noise cancelling"],
    "description": ["A test product description."],
    "details": {"Form Factor": "Over Ear", "Brand": "Acme"},
    "categories": ["Electronics", "Headphones", "Over-Ear Headphones"],
    "main_category": "All Electronics",
    "price": 99.0,
    "store": None,
}


def test_adapter_maps_event_memory_to_attribute_proxy_and_reports_gaps():
    output = feature_adapter.build_adapter_output(
        item_id="B0TESTITEM",
        raw_metadata=RAW_METADATA,
        active_preferences=[
            {"value": "microphone", "polarity": "positive"},
            {"value": "over-ear", "polarity": "avoid"},
            {"value": "leather", "polarity": "positive"},
        ],
        query_relevance={"query_rank": 3.0, "query_score": 0.72},
    )
    assert output.non_memory["meta_attr_microphone"] == 1.0
    assert output.non_memory["meta_attr_over_ear"] == 1.0
    assert output.memory["mem_hist_count"] == 2.0
    assert output.memory["mem_hist_high_count"] == 1.0
    assert output.memory["mem_hist_low_count"] == 1.0
    assert output.memory["mem_candidate_high_match_count"] == 1.0
    assert output.memory["mem_candidate_low_conflict_count"] == 1.0
    # wireless, rechargeable and noise_cancelling are unknown in active memory.
    assert output.memory["mem_candidate_attr_unknown_count"] == 3.0
    assert output.unknown_memory_values == ["leather"]
    assert output.query_relevance == {"query_rank": 3.0, "query_score": 0.72}
    compatibility = output.compatibility()
    assert compatibility["model_ready"] is False
    assert "mem_hist_mean_rating" in compatibility["memory_missing"]
    assert "pop_count" in compatibility["non_memory_missing"]
    assert compatibility["status"] == feature_adapter.PRODUCTION_STATUS


def test_adapter_vector_follows_training_feature_orders_and_has_nan_for_gaps():
    output = feature_adapter.build_adapter_output(
        item_id="B0TESTITEM",
        raw_metadata=RAW_METADATA,
        active_preferences=[{"value": "microphone", "polarity": "positive"}],
    )
    off = output.memory_off_vector()
    on = output.memory_on_vector()
    assert len(off) == len(features.non_memory_feature_names())
    assert len(on) == len(features.non_memory_feature_names()) + len(
        features.memory_feature_names()
    )
    assert all(math.isnan(value) for value in off[-len(features.POP_FEATURES):])
    assert math.isnan(on[features.non_memory_feature_names().index("pop_count")])
    memory_start = len(features.non_memory_feature_names())
    mean_index = memory_start + features.memory_feature_names().index("mem_hist_mean_rating")
    assert math.isnan(on[mean_index])


def test_adapter_does_not_use_identity_fields_as_features():
    output = feature_adapter.build_adapter_output(
        item_id="B0TESTITEM",
        raw_metadata=RAW_METADATA,
        active_preferences=[],
    )
    forbidden = {"user_id", "target_item_id", "target_rating", "target_high"}
    assert not (set(output.non_memory) & forbidden)
    assert not (set(output.memory) & forbidden)
    assert "B0TESTITEM" not in output.non_memory

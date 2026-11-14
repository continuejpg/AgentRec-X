from __future__ import annotations

import json
from pathlib import Path

from experiments.agent_native_ranking.memory_adapter import candidate_features, structured_memory_text

REPO = Path(__file__).resolve().parents[1]
RUNS = REPO / "runs/agent_native_ranking_phase42"


def test_adapter_multiple_active_preferences_and_unknown():
    item = {"attributes": {"wireless": {}, "microphone": {}}}
    memory = {"positive": [{"value": "wireless"}, {"value": "rechargeable"}],
              "negative": [{"value": "over_ear"}]}
    features = candidate_features(item, memory)
    assert features["positive_match_count"] == 1
    assert features["positive_unknown_count"] == 1
    assert features["negative_conflict_count"] == 0
    assert features["negative_unknown_count"] == 1
    assert features["active_preference_count"] == 3
    assert features["preference_evidence_coverage"] == 1 / 3


def test_structured_memory_contains_active_only():
    memory = {"positive": [{"value": "microphone"}], "negative": [{"value": "over_ear"}]}
    text = structured_memory_text(memory)
    assert "microphone: preferred" in text
    assert "over_ear: avoided" in text
    assert "removed" not in text.lower()
    assert "superseded" not in text.lower()


def test_phase42_outputs_exist_and_counts():
    controlled = json.loads((RUNS / "shopranker_controlled_gpu.json").read_text(encoding="utf-8"))
    assert controlled["benchmark_v2"]["scenarios"] == 300
    assert controlled["p20"]["tasks"] == 19
    assert controlled["counterfactual"]["cases"] == 16
    assert controlled["benchmark_v2"]["pairwise"]["hybrid"]["n"] == 145
    rows = 0
    with (RUNS / "hybrid_gpu_predictions.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows += 1
    assert rows == 6000


def test_dcn_diagnostics_schema_and_no_bug_claim():
    diagnostics = json.loads((RUNS / "dcn_alignment_diagnostics.json").read_text(encoding="utf-8"))
    assert diagnostics["schema_check"]["B_feature_order_matches_schema"] is True
    assert diagnostics["schema_check"]["normalization_mean_finite"] is True
    assert diagnostics["pairwise_variants"]["B_one_pref"]["accuracy"] < 0.5
    assert diagnostics["pairwise_variants"]["A_base_query"]["accuracy"] > 0.5

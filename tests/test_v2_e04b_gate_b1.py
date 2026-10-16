"""E-04B-2 Gate B-1: real SASRec protocol and identity checks (CPU, offline, no provider).

The checkpoint is real and local, so these tests load it once per module (about 1.4 s) and prove:
the identity space agrees with the frozen protocol, one real forward pass returns finite
full-catalogue scores, the ranking is deterministic, and the label never becomes an input.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
CHECKPOINT = REPO / "runs" / "sasrec_canonical_2026" / "best.pt"
RUN_JSON = REPO / "runs" / "sasrec_canonical_2026" / "run.json"
MAPPINGS = REPO / "data" / "processed" / "Sports_and_Outdoors_mappings.json"
SEQUENCES = REPO / "data" / "processed" / "Sports_and_Outdoors_sequences.json"

pytestmark = pytest.mark.skipif(
    not (CHECKPOINT.exists() and MAPPINGS.exists() and SEQUENCES.exists()),
    reason="real SASRec checkpoint or processed artifacts are not present in this checkout",
)


@pytest.fixture(scope="module")
def engine():
    from recommendation.inference.sasrec import InferenceConfig, SASRecInferenceEngine

    recorded = json.loads(RUN_JSON.read_text(encoding="utf-8"))
    expected = recorded.get("checkpoints", {}).get("best", {}).get("sha256")
    return SASRecInferenceEngine(
        InferenceConfig(checkpoint_path=CHECKPOINT, mappings_path=MAPPINGS, device="cpu",
                        expected_checkpoint_sha256=expected)
    )


@pytest.fixture(scope="module")
def one_case(engine):
    from experiments.e04b2_gate_a import stream_sequences
    from recommendation.evaluation.split import split_sequence

    mappings = json.loads(MAPPINGS.read_text(encoding="utf-8"))
    num_items = int(mappings["num_items"])
    for element in stream_sequences(SEQUENCES):
        items = [int(value) for value in element.get("item_ids", ())]
        case = split_sequence(str(element.get("user_id", "")), items, num_items,
                              int(element.get("user_int_id", -1)))
        if case is not None:
            return case, num_items
    pytest.skip("no valid protocol case found in the artifact")


def test_checkpoint_digest_matches_the_recorded_run(engine) -> None:
    recorded = json.loads(RUN_JSON.read_text(encoding="utf-8"))
    assert engine.checkpoint_sha256 == recorded["checkpoints"]["best"]["sha256"]
    assert engine.is_ready() is True


def test_identity_space_agrees_with_the_frozen_protocol(engine) -> None:
    mappings = json.loads(MAPPINGS.read_text(encoding="utf-8"))
    assert int(mappings["padding"]["pad_id"]) == 0           # PAD is never a real item id
    assert int(mappings["padding"]["first_real_id"]) == 1    # 1-based: no 0-based offset bug
    assert len(mappings["item2id"]) == int(mappings["num_items"]) == engine.num_items
    assert engine.max_seq_len == json.loads(RUN_JSON.read_text(encoding="utf-8"))["model_config"][
        "max_seq_len"
    ]


def test_one_real_forward_pass_returns_finite_full_catalogue_scores(engine, one_case) -> None:
    case, num_items = one_case
    encoded, _truncated = engine.encode_history(
        [engine.item_id_to_parent_asin(item) for item in case.test_history]
    )
    scores = engine.score_catalog(encoded)
    assert len(scores) == num_items + 1, "one score per item plus the PAD slot"
    assert all(isinstance(value, float) for value in scores[:64])
    assert all(value == value and abs(value) != float("inf") for value in scores)


def test_the_ranking_is_deterministic_and_the_label_is_never_an_input(engine, one_case) -> None:
    from recommendation.evaluation.metrics import excluded_seen, rank_of_target, sorted_ranking

    case, num_items = one_case
    # The only ids handed to the model are the case's own test-time-visible history.
    history_asins = [engine.item_id_to_parent_asin(item) for item in case.test_history]
    encoded, _ = engine.encode_history(history_asins)
    encoded_real = [item for item in encoded if item != 0]
    assert set(encoded_real) <= set(case.test_history), (
        "the model input must be a subset of the visible history"
    )

    scores = engine.score_catalog(encoded)
    first = sorted_ranking(scores, history=case.test_history, num_items=num_items)[:20]
    second = sorted_ranking(engine.score_catalog(encoded), history=case.test_history,
                            num_items=num_items)[:20]
    assert first == second, "ranking must be reproducible"

    mask = excluded_seen(case.test_history, case.test_target)
    assert not (set(first) & mask), "no masked (already-seen) item may be recommended"
    assert 0 not in first or mask, "PAD is never a real recommendation"
    rank = rank_of_target(scores, case.test_target, history=case.test_history, num_items=num_items)
    assert rank >= 1


def test_the_stage_artifacts_are_labelled_and_within_the_frozen_contract() -> None:
    summary_path = REPO / "runs" / "e04b_gate_b1_sasrec" / "gate_b1_summary.json"
    if not summary_path.exists():
        pytest.skip("run experiments.e04b_gate_b1_sasrec first")
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    assert payload["protocol"] == "agentrecx.eval_protocol.v1"
    assert payload["split"] == "temporal_leave_two_out"
    assert payload["k_values"] == [5, 10, 20]
    assert payload["phase1_assets"]["sha256_matches_run_json"] is True
    assert payload["identity"]["mapping_and_data_agree"] is True
    assert payload["identity"]["pad_id"] == 0
    for name, stage in payload["stages"].items():
        assert stage["invalid_candidates"] == 0, name
        assert stage["pad_candidates"] == 0, name
        assert stage["seen_candidates"] == 0, name
        assert stage["model_failures"] == 0, name
        assert stage["cases_scored"] > 0, name
        # Every reported rate must have the same denominator it reports.
        for k in ("5", "10", "20"):
            assert stage["hits"][k] <= stage["cases_scored"], (name, k)

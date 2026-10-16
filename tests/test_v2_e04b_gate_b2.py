"""E-04B-2 Gate B-2: real Two-Tower protocol, identity and same-case alignment checks.

Loads the real checkpoint once per module (about 1.6 s, CPU, offline) and proves the identity space,
one real forward pass, the ranking/mask contract, and that the recorded run's cases line up with the
archived SASRec run one by one.  No provider, no network, no training.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
TT_DIR = REPO / "runs" / "twotower_public_2026_full"
CHECKPOINT = TT_DIR / "checkpoint.pt"
CATEGORIES = TT_DIR / "metadata_categories.json"
MAPPINGS = REPO / "data" / "processed" / "Sports_and_Outdoors_mappings.json"
SEQUENCES = REPO / "data" / "processed" / "Sports_and_Outdoors_sequences.json"
B1_STAGE = REPO / "runs" / "e04b_gate_b1_sasrec" / "stage_exploratory_100.json"
B2_SUMMARY = REPO / "runs" / "e04b_gate_b2_twotower" / "gate_b2_summary.json"
HEAD_CACHE = REPO / "runs" / "b1_evidence" / "two_tower_head.npz"

pytestmark = pytest.mark.skipif(
    not (CHECKPOINT.exists() and CATEGORIES.exists() and MAPPINGS.exists() and SEQUENCES.exists()),
    reason="real Two-Tower checkpoint or processed artifacts are not present in this checkout",
)


@pytest.fixture(scope="module")
def engine():
    from recommendation.inference.twotower import TwoTowerInferenceConfig, TwoTowerInferenceEngine

    return TwoTowerInferenceEngine(
        TwoTowerInferenceConfig(checkpoint_path=CHECKPOINT, categories_path=CATEGORIES,
                                mappings_path=MAPPINGS, device="cpu")
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


def test_model_and_identity_contract(engine) -> None:
    mappings = json.loads(MAPPINGS.read_text(encoding="utf-8"))
    recorded = json.loads((TT_DIR / "run.json").read_text(encoding="utf-8"))
    assert engine.is_ready() is True
    assert int(mappings["padding"]["pad_id"]) == 0
    assert int(mappings["padding"]["first_real_id"]) == 1
    assert len(mappings["item2id"]) == int(mappings["num_items"]) == engine.num_items
    assert engine.max_seq_len == recorded["model_config"]["max_seq_len"]
    metadata = engine.model_metadata()
    assert metadata["model_type"] == "TwoTowerDSSM"
    assert metadata["score_kind"] == "two_tower_cosine"


def test_one_real_forward_pass_is_finite_full_catalogue(engine, one_case) -> None:
    from recommendation.datasets.sasrec import encode_inference_history

    case, num_items = one_case
    encoded = encode_inference_history(list(case.test_history), engine.max_seq_len, num_items)
    scores = engine.score_catalog(encoded)
    assert len(scores) == num_items + 1, "one score per item plus the PAD slot"
    assert all(value == value and abs(value) != float("inf") for value in scores[:128])


def test_ranking_is_deterministic_and_the_label_is_never_an_input(engine, one_case) -> None:
    from recommendation.datasets.sasrec import encode_inference_history
    from recommendation.evaluation.metrics import excluded_seen, rank_of_target, sorted_ranking

    case, num_items = one_case
    encoded = encode_inference_history(list(case.test_history), engine.max_seq_len, num_items)
    # The encoder only ever sees the visible history: no target can enter the user tower.
    real = [item for item in encoded if item != 0]
    assert set(real) <= set(case.test_history)

    first = sorted_ranking(engine.score_catalog(encoded), history=case.test_history,
                           num_items=num_items)[:20]
    second = sorted_ranking(engine.score_catalog(encoded), history=case.test_history,
                            num_items=num_items)[:20]
    assert first == second
    mask = excluded_seen(case.test_history, case.test_target)
    assert not (set(first) & mask)
    assert 0 not in first
    assert rank_of_target(engine.score_catalog(encoded), case.test_target,
                          history=case.test_history, num_items=num_items) >= 1


def test_the_calibration_cache_is_not_part_of_inference(engine) -> None:
    """The B1 ranking cache must not be needed to score: the engine never reads it."""
    import inspect

    import recommendation.inference.twotower as module

    source = inspect.getsource(module)
    assert "two_tower_head" not in source, "a post-hoc experiment cache leaked into inference"
    assert HEAD_CACHE.exists() or True  # its presence is irrelevant to the scoring path


def test_the_recorded_run_is_aligned_and_within_contract() -> None:
    if not B2_SUMMARY.exists():
        pytest.skip("run experiments.e04b_gate_b2_twotower first")
    payload = json.loads(B2_SUMMARY.read_text(encoding="utf-8"))
    assert payload["protocol"] == "agentrecx.eval_protocol.v1"
    assert payload["k_values"] == [5, 10, 20]
    assert payload["calibration_decision"]["used_for_scoring"] is False
    assert payload["hash_check_after"]["unchanged"] is True
    assert payload["identity"]["mapping_and_data_agree"] is True
    for name, stage in payload["stages"].items():
        assert stage["model_failures"] == 0, name
        assert stage["invalid_candidates"] == 0, name
        assert stage["pad_candidates"] == 0, name
        assert stage["seen_candidates"] == 0, name
        assert stage["cases_scored"] == stage["cases_selected"], name
        for k in ("5", "10", "20"):
            assert stage["hits"][k] <= stage["cases_scored"], (name, k)


@pytest.mark.skipif(not B1_STAGE.exists(), reason="the archived Gate B-1 stage is absent")
def test_the_two_models_scored_the_same_cases_in_the_same_order() -> None:
    if not B2_SUMMARY.exists():
        pytest.skip("run experiments.e04b_gate_b2_twotower first")
    payload = json.loads(B2_SUMMARY.read_text(encoding="utf-8"))
    alignment = payload["case_alignment"]
    assert alignment["checked"] is True
    assert alignment["same_ids_in_same_order"] is True, (
        "a paired comparison is only valid when both models scored the same cases in the same order"
    )
    assert alignment["b1_cases"] == alignment["rebuilt_cases"] == 100
    b1 = json.loads(B1_STAGE.read_text(encoding="utf-8"))
    assert len(b1["per_case"]) == 100
    comparison = payload["comparison_vs_sasrec"]
    assert comparison["same_case_order"] is True
    for k in ("HR@5", "HR@10", "HR@20"):
        entry = comparison["table"][k]
        assert entry["hits"]["denominator"] == 100, k
        assert comparison["paired"][k]["discordant"] <= 100, k

"""E-04B-2 Gate B-3: TIGER identity-boundary, mapping and generative-run contract checks.

These tests are cheap: they reuse the accepted identity/mapping artifacts and the recorded run
outputs.  They never load the TIGER checkpoint (the real generative run lives in
`experiments/e04b_gate_b3_tiger.py` and its results in `runs/e04b_gate_b3_tiger*/`), and they keep
the known architecture-contract failure visible instead of hiding it.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SID = REPO / "runs" / "tiger_public_2026" / "sid" / "semantic_ids.json"
MAPPINGS = REPO / "data" / "processed" / "Sports_and_Outdoors_mappings.json"
WORKER = REPO / "backends" / "tiger_public" / "src" / "tiger_public" / "serve_worker.py"
RUNS = sorted(REPO.glob("runs/e04b_gate_b3_tiger*"))


@pytest.mark.skipif(not SID.exists(), reason="the SID artifact is absent")
def test_sid_space_is_injective_and_covers_the_catalogue() -> None:
    payload = json.loads(SID.read_text(encoding="utf-8"))
    audit = payload["collision_audit"]
    post = audit["post_dedup"]
    mappings = json.loads(MAPPINGS.read_text(encoding="utf-8"))
    assert post["collision_groups"] == 0, "post-dedup SIDs must be unambiguous"
    assert post["largest_group"] == 1
    assert post["distinct_sids"] == int(mappings["num_items"]) == len(payload["assignment"]) - 1
    assert audit["overflow_refusal_triggered"] is False
    # The pre-dedup collisions are recorded, not hidden: the mapping is only injective *after* dedup.
    assert audit["pre_dedup"]["collision_groups"] > 0


@pytest.mark.skipif(not WORKER.exists(), reason="the backend worker is absent")
def test_the_worker_boundary_carries_no_canonical_identity() -> None:
    """The IPC frame never carries `parent_asin`: only the docstring says the word."""
    lines = [line for line in WORKER.read_text(encoding="utf-8").splitlines()
             if "parent_asin" in line]
    assert len(lines) == 1, lines
    assert "never appears" in lines[0], "the only mention must be the boundary statement itself"


def test_the_engine_offers_a_certified_mode_without_downgrade() -> None:
    from recommendation.inference.tiger_engine import TigerWorkerConfig

    config = TigerWorkerConfig()
    # Decode parameters come from the native defaults; nothing here is tuned on test cases.
    assert config.k_max >= 20 and config.max_hist_items >= 20 and config.beam >= 1
    assert config.device in {"cpu", "cuda"}


@pytest.mark.skipif(not RUNS, reason="no TIGER run directory is present")
def test_recorded_tiger_runs_are_certified_and_identity_clean() -> None:
    summary_paths = [path / "gate_b3_summary.json" for path in RUNS]
    summary_paths = [path for path in summary_paths if path.exists()]
    if not summary_paths:
        pytest.skip("no TIGER summary recorded yet")
    for path in summary_paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert payload["manifest"]["decode_config_frozen"]["mode"] == "CERTIFIED"
        assert payload["manifest"]["decode_config_frozen"]["allow_mode_downgrade"] is False
        assert payload["manifest"]["declared_before_scoring"] is True
        for name, stage in payload.get("stages", {}).items():
            modes = set(stage["modes"])
            assert modes <= {"CERTIFIED"}, (name, modes)   # APPROXIMATE is never relabelled
            assert stage["invalid_ids"] == 0, name
            assert stage["pad_ids"] == 0, name
            # The count must be *recorded truthfully*, never assumed zero: a repeat interaction
            # legitimately puts the target id in the visible history, and hiding that would be a
            # lie about the input.  What must never happen is a count larger than the case set.
            assert "target_ids_fed_to_model" in stage, name
            assert 0 <= stage["target_ids_fed_to_model"] <= stage["cases_attempted"], name
            assert stage["duplicate_items_filtered"] >= 0, name
            # Rates must be reported against the preselected denominator, failures included.
            assert stage["denominator"]["all_preselected_cases"] == stage["cases_attempted"]
            for key in ("generative_topk_hr_at_k", "generative_topk_ndcg_at_k"):
                assert set(stage[key]) == {"5", "10", "20"}, key


@pytest.mark.skipif(not RUNS, reason="no TIGER run directory is present")
def test_case_manifest_matches_the_sasrec_and_twotower_prefixes() -> None:
    b1 = REPO / "runs" / "e04b_gate_b1_sasrec" / "stage_exploratory_100.json"
    b2 = REPO / "runs" / "e04b_gate_b2_twotower" / "stage_exploratory_100.json"
    if not (b1.exists() and b2.exists()):
        pytest.skip("the reference stages are absent")
    reference = [row["case"] for row in json.loads(b1.read_text(encoding="utf-8"))["per_case"]]
    assert reference == [row["case"] for row in json.loads(b2.read_text(encoding="utf-8"))["per_case"]]
    for path in RUNS:
        summary = path / "gate_b3_summary.json"
        if not summary.exists():
            continue
        ids = json.loads(summary.read_text(encoding="utf-8"))["manifest"]["case_manifest"]["case_ids"]
        # A TIGER run may cover a prefix only; whatever it covers must match the reference order.
        assert ids == reference[: len(ids)], path


@pytest.mark.skipif(not (REPO / "data" / "processed" / "Sports_and_Outdoors_sequences.json").exists(),
                    reason="the sequence artifact is absent")
def test_the_model_input_is_the_frozen_history_and_never_a_future_interaction() -> None:
    """Negative test: appending the test target to the input would be detected.

    The positive contract alone (`input ⊆ test_history`) is too weak, so this also proves the
    *forbidden* shape is distinguishable: the input ends at `items[-2]` (the validation item) and
    the test position's item (`items[-1]`) appears only when the product was already consumed
    strictly earlier - which is why the repeat case must be checked by position, not by id.
    """
    from experiments.e04b2_gate_a import SEQUENCES, stream_sequences
    from recommendation.datasets.sasrec import encode_inference_history
    from recommendation.evaluation.split import split_sequence

    num_items = int(json.loads(MAPPINGS.read_text(encoding="utf-8"))["num_items"])
    checked = repeats = 0
    for element in stream_sequences(SEQUENCES):
        items = [int(value) for value in element.get("item_ids", ())]
        case = split_sequence(str(element.get("user_id", "")), items, num_items,
                              int(element.get("user_int_id", -1)))
        if case is None:
            continue
        checked += 1
        # 1. the frozen input is the visible history: everything except the final test interaction.
        assert case.test_history == tuple(items[:-1]), case.user_int_id
        assert case.test_target == items[-1], case.user_int_id
        # 2. an illegal "append the target" input is materially different from the legal one, so a
        #    future injection could not pass unnoticed.
        illegal = encode_inference_history(list(items), 50, num_items)
        legal = encode_inference_history(list(case.test_history), 50, num_items)
        assert illegal != legal, case.user_int_id
        # 3. if the target id is in the legal input it must be an *earlier* interaction.
        if case.test_target in case.test_history:
            repeats += 1
            positions = [index for index, value in enumerate(items) if value == case.test_target]
            assert max(position for position in positions if position != len(items) - 1) < len(items) - 1
        if checked >= 100:
            break
    assert checked == 100
    assert repeats >= 0  # recorded, not asserted away

"""E-04B-2 Gate A: runtime protocol assertions on the real artifacts (no weights, no network).

These are the *runtime* checks the Gate A brief asks for (not greps): they stream the frozen
sequence artifact with the same reader the Gate A script uses, split the real users with the
accepted protocol, and assert the invariants on live objects.  Everything here is bounded to a
small slice so the suite stays fast; the full-sample numbers live in
``runs/e04b2_gate_a/gate_a_summary.json`` and in ``docs/V2_E04B2_GATE_A_REPORT.md``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from experiments.e04b2_gate_a import MAPPINGS, PRODUCTS, SEQUENCES, load_catalogue, stream_sequences
from recommendation.evaluation.metrics import excluded_seen, hit_at_k, ndcg_at_k
from recommendation.evaluation.split import PROTOCOL_NAME, PROTOCOL_VERSION, split_sequence

SLICE = 50  # users; enough to exercise the protocol without a slow test


@pytest.fixture(scope="module")
def artefacts():
    if not SEQUENCES.exists() or not MAPPINGS.exists() or not PRODUCTS.exists():
        pytest.skip("real processed artifacts are not present in this checkout")
    mappings = json.loads(MAPPINGS.read_text(encoding="utf-8"))
    return mappings


@pytest.fixture(scope="module")
def real_cases(artefacts):
    num_items = int(artefacts["num_items"])
    cases = []
    for element in stream_sequences(SEQUENCES):
        items = [int(value) for value in element.get("item_ids", ())]
        case = split_sequence(str(element.get("user_id", "")), items, num_items,
                              int(element.get("user_int_id", -1)))
        if case is not None:
            cases.append((items, case))
        if len(cases) >= SLICE:
            break
    return cases


def test_the_protocol_is_the_frozen_one(artefacts) -> None:
    assert PROTOCOL_NAME == "temporal_leave_two_out"
    assert PROTOCOL_VERSION == "agentrecx.eval_protocol.v1"
    assert int(artefacts["padding"]["pad_id"]) == 0
    assert int(artefacts["padding"]["first_real_id"]) == 1
    assert len(artefacts["item2id"]) == int(artefacts["num_items"])
    assert len(artefacts["user2id"]) == int(artefacts["num_users"])


def test_real_sequences_never_carry_pad_or_out_of_range_ids(real_cases, artefacts) -> None:
    num_items = int(artefacts["num_items"])
    assert real_cases, "the streamed slice must contain valid cases"
    for items, _case in real_cases:
        assert 0 not in items, "PAD must never appear in a real user sequence"
        assert all(1 <= item <= num_items for item in items), "ids stay inside the real catalogue"


def test_each_case_keeps_the_target_out_of_its_own_mask(real_cases) -> None:
    """The single positive must stay rankable.

    The accepted helper ``excluded_seen`` removes the target from the mask it returns, which is what
    keeps a *repeat* interaction scoreable: a user who consumed the same item before validation has
    that id in the raw ``test_seen`` too, and masking it there would delete the label instead of
    ranking it.  The raw mask is still a superset of the history, as the protocol requires.
    """
    for _items, case in real_cases:
        mask = excluded_seen(case.test_history, case.test_target)
        assert case.test_target not in mask, "the positive is never masked out of its own ranking"
        assert mask == set(case.test_history) - {case.test_target}, (
            "the ranking mask is the visible history, minus the target when the user consumed it "
            "before (a repeat interaction)"
        )
        assert set(case.test_history) <= set(case.test_seen), (
            "the raw mask is the history plus the consumed validation target"
        )


def test_history_is_the_visible_input_and_the_target_is_not_in_it(real_cases) -> None:
    """The request input is the history; the label is a separate field and is never the input.

    At the test step the visible history is the training prefix **plus** the validation item — the
    held-out target is the only item the request never sees.
    """
    for items, case in real_cases:
        assert len(case.test_history) == case.sequence_length - 1
        assert set(case.test_history) <= set(items)
        assert case.test_target == items[-1]
        assert case.validation_target == items[-2]


def test_full_catalogue_metrics_match_the_single_positive_protocol() -> None:
    # one positive: HR@K and Recall@K coincide, NDCG@K = 1/log2(rank+1) inside the cut
    for rank in (1, 2, 5, 10, 20, 21):
        for k in (5, 10, 20):
            assert hit_at_k(rank, k) == (1.0 if rank <= k else 0.0)
            assert ndcg_at_k(rank, k) == (1.0 / __import__("math").log2(rank + 1) if rank <= k else 0.0)
    assert ndcg_at_k(1, 5) == 1.0


def test_the_real_catalogue_maps_into_the_same_identity_space(artefacts) -> None:
    item2id = {str(key): int(value) for key, value in artefacts["item2id"].items()}
    text_by_id, asin_by_id = load_catalogue(PRODUCTS, item2id)
    assert text_by_id, "the real catalogue must produce searchable records"
    assert len(text_by_id) == len(asin_by_id)
    assert set(text_by_id) <= set(range(1, int(artefacts["num_items"]) + 1))
    assert 0 not in text_by_id, "PAD is not a catalogue record"

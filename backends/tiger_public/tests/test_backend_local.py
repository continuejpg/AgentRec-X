"""Self-contained tests for the public-TIGER backend (Step 2.3 skeleton).

These run in the **backend's own virtual environment** and are deliberately kept out of
AgentRec-X's suite: ``pytest.ini`` sets ``norecursedirs = backends``, so `pytest -q` at the
repository root never collects them and never needs a backend dependency installed.

Run them with the backend interpreter::

    cd backends/tiger_public
    PYTHONPATH=src python -m pytest tests -q

Step 2.3 has almost nothing to test that AgentRec-X's ``tests/test_backend_boundaries.py``
does not already cover from the other side of the boundary; what lives here are the
*backend-local* invariants - token layout arithmetic, the PAD sentinel, the frozen score rule,
and the collision-audit math.
"""

from __future__ import annotations

import numpy as np
import pytest

from tiger_public.contracts import (
    CONTRACT_VERSION,
    PAD_SENTINEL,
    ContractError,
    build_token_layout,
    collision_audit,
    compute_scores,
    required_frontier_for,
)
from tiger_public.scoring import (
    SCORE_RULE,
    assert_monotone_prefix_bound,
    sequence_log_probability,
    validate_score_rule,
)


def test_token_layout_puts_specials_above_the_code_space() -> None:
    layout = build_token_layout(levels=3, codebook_size=256, dedup_levels=1)
    assert layout.vocab_size == 4 * 256 + 3
    assert (layout.pad_token, layout.bos_token, layout.eos_token) == (1024, 1025, 1026)
    assert layout.code_space == 1024
    # Token 0 is the legal code (level 0, code 0), not PAD.
    assert layout.tokenise((0, 0, 0, 0)) == (0, 256, 512, 768)
    assert layout.pad_token != 0


def test_pad_sentinel_is_not_tokenisable() -> None:
    layout = build_token_layout(levels=3, codebook_size=64, dedup_levels=1)
    assert not layout.is_tokenisable((PAD_SENTINEL,) * 4)
    with pytest.raises(ContractError):
        layout.tokenise((PAD_SENTINEL,) * 4)


def test_tokenise_refuses_an_out_of_range_digit() -> None:
    layout = build_token_layout(levels=2, codebook_size=8, dedup_levels=1)
    with pytest.raises(ContractError):
        layout.tokenise((0, 8, 0))
    with pytest.raises(ContractError):
        layout.tokenise((0, 0, 8))


def test_vocab_size_is_derived_not_hardcoded() -> None:
    small = build_token_layout(levels=2, codebook_size=32, dedup_levels=1)
    assert (small.code_space, small.pad_token, small.vocab_size) == (96, 96, 99)
    with pytest.raises(ContractError):
        build_token_layout(levels=0, codebook_size=8)


def test_required_frontier_is_k_max_plus_unique_seen() -> None:
    frontier = required_frontier_for([5, 5, 9, 1], case_id=3, k_max=20)
    assert frontier.seen_count == 3
    assert frontier.required == 23
    assert frontier.history_length == 4


def test_required_frontier_refuses_pad_in_history() -> None:
    with pytest.raises(ContractError):
        required_frontier_for([1, 0, 2], case_id=0, k_max=20)


def test_compute_scores_is_bounded_and_deterministic() -> None:
    rows = np.arange(16, dtype=np.int64)
    cases = np.asarray([7], dtype=np.int64)
    first = compute_scores(rows, cases)
    second = compute_scores(rows, cases)
    assert first.shape == (16, 1)
    assert np.array_equal(first, second)
    assert (first <= 0).all() and (first > -1.0).all()


def test_collision_audit_reports_both_blocks() -> None:
    # Three items collapse to one code, so the pre-dedup group is 3 and the dedup digit makes
    # the post-dedup assignment unique.
    codes = [(1, 2), (1, 2), (1, 2), (3, 4)]
    audit = collision_audit(codes, dedup_levels=1, dedup_vocab_size=256)
    assert audit["pre_dedup"]["collision_groups"] == 1
    assert audit["pre_dedup"]["largest_group"] == 3
    assert audit["post_dedup"]["collision_groups"] == 0
    assert audit["post_dedup"]["distinct_sids"] == 4
    assert audit["largest_group_le_dedup_vocab"] is True
    assert audit["overflow_refusal_triggered"] is False


def test_collision_audit_flags_overflow() -> None:
    codes = [(1, 2)] * 9
    audit = collision_audit(codes, dedup_levels=1, dedup_vocab_size=4)
    assert audit["pre_dedup"]["largest_group"] == 9
    assert audit["largest_group_le_dedup_vocab"] is False
    assert audit["overflow_refusal_triggered"] is True


def test_score_rule_is_frozen() -> None:
    validate_score_rule(SCORE_RULE)
    for broken in (
        {**SCORE_RULE, "eos_in_score": True},
        {**SCORE_RULE, "softmax": "valid_children"},
        {**SCORE_RULE, "child_renormalisation": True},
        {**SCORE_RULE, "version": 2},
    ):
        with pytest.raises(ContractError):
            validate_score_rule(broken)


def test_sequence_log_probability_excludes_eos() -> None:
    # Two steps, three tokens; the path picks token 1 then token 0.
    table = np.log(np.asarray([[[0.2, 0.5, 0.3], [0.1, 0.1, 0.8]]], dtype=np.float64))
    total = sequence_log_probability(table, (1, 0))
    expected = float(np.log(0.5) + np.log(0.1))
    assert total.shape == (1,)
    assert total[0] == pytest.approx(expected, rel=1e-6)


def test_prefix_bound_is_admissible_for_a_log_softmax() -> None:
    logits = np.asarray([[[2.0, 1.0, 0.5], [0.1, 3.0, 0.2]]], dtype=np.float64)
    shifted = logits - logits.max(axis=-1, keepdims=True)
    table = shifted - np.log(np.exp(shifted).sum(axis=-1, keepdims=True))
    assert_monotone_prefix_bound(table)
    with pytest.raises(ContractError):
        assert_monotone_prefix_bound(np.abs(table) + 0.1)


def test_contract_version_is_pinned() -> None:
    assert CONTRACT_VERSION == "agentrecx.tiger_backend.v3"

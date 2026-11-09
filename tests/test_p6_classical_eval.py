"""CPU-only tests for the P6 classical full-ranking evaluation path.

These tests do not load a model checkpoint: they exercise the history variants, the
scorer contract, the full-ranking -> P6-metric plumbing and the frozen split audit with a
fake full-catalogue scorer.  The real SASRec / Two-Tower smoke run lives in the
experiment report, not in the unit test suite.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import pytest

from experiments.p6_benchmark.full_ranking import (
    CANONICAL_VARIANT,
    P6_VARIANT,
    VARIANTS,
    history_for_row,
    scalar_identity_checksum,
)
from experiments.p6_benchmark.run_classical import determinism_check, evaluate_rows
from experiments.p6_benchmark.uncertainty import bootstrap_ci

ROOT = Path("data/p6_benchmark_v0")


def _id2asin(num_items: int) -> list[str]:
    return ["PAD"] + [f"A{item_id}" for item_id in range(1, num_items + 1)]


class FakeScorer:
    """Deterministic full-catalogue scorer with the adapter's public surface."""

    name = "fake"
    score_kind = "fake_score"

    def __init__(self, scores: Sequence[float], num_items: int) -> None:
        self._scores = [float(value) for value in scores]
        self.num_items = num_items
        self._id2asin = _id2asin(num_items)
        self.model_histories: list[tuple[int, ...]] = []

    def score_catalog(self, history_item_ids: Sequence[int]) -> list[float]:
        self.model_histories.append(tuple(int(value) for value in history_item_ids))
        return list(self._scores)

    def item_id_to_parent_asin(self, item_id: int) -> str:
        return self._id2asin[int(item_id)]

    def model_metadata(self) -> dict[str, Any]:
        return {"name": self.name, "num_items": self.num_items}


def _row(
    *,
    num_items: int,
    history: Sequence[int],
    validation: int,
    test: int,
    case_id: str = "case-0",
    user_id: str = "user-0",
) -> dict[str, Any]:
    id2asin = _id2asin(num_items)
    return {
        "case_id": case_id,
        "user_id": user_id,
        "split": "dev",
        "history_item_ids": list(history),
        "validation_item_id": validation,
        "test_item_id": test,
        "validation_parent_asin": id2asin[validation],
        "test_parent_asin": id2asin[test],
    }


def _load_split(split: str) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in (ROOT / f"classical_{split}.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def test_history_variants_differ_only_by_the_validation_interaction() -> None:
    row = _row(num_items=10, history=(2, 3), validation=4, test=7)
    assert history_for_row(row, P6_VARIANT) == (2, 3)
    assert history_for_row(row, CANONICAL_VARIANT) == (2, 3, 4)
    with pytest.raises(ValueError):
        history_for_row(row, "not-a-variant")
    assert set(VARIANTS) == {P6_VARIANT, CANONICAL_VARIANT}


def test_known_scores_produce_exact_p6_metrics() -> None:
    num_items = 10
    scores = [0.0] * (num_items + 1)
    scores[7] = 100.0  # target, rank 1
    scores[2] = 50.0  # masked history item must not appear
    scores[3] = 40.0  # masked history item must not appear
    scorer = FakeScorer(scores, num_items)
    row = _row(num_items=num_items, history=(2, 3), validation=4, test=7)

    result = evaluate_rows(
        scorer=scorer, rows=[row], variant=P6_VARIANT, num_items=num_items
    )

    assert result["variant"] == P6_VARIANT
    assert result["cases"] == 1
    assert result["metrics"] == {
        "cases": 1,
        "Recall@10": 1.0,
        "Recall@50": 1.0,
        "NDCG@10": 1.0,
        "MRR@10": 1.0,
        "CatalogueCoverage@10": 0.8,  # 8 eligible candidates / 10
    }
    assert result["diagnostics"]["exact_rank_mean"] == 1
    assert result["diagnostics"]["target_in_top10"] == 1
    # The model saw history only; the target was never a scorer input in this variant.
    assert scorer.model_histories == [(2, 3)]


def test_target_beyond_top50_contributes_zero_to_all_p6_metrics() -> None:
    num_items = 100
    scores = [0.0] * (num_items + 1)
    scores[1] = -1.0  # target ranks last among the 98 eligible candidates
    scorer = FakeScorer(scores, num_items)
    row = _row(num_items=num_items, history=(2, 3), validation=4, test=1)

    result = evaluate_rows(
        scorer=scorer, rows=[row], variant=P6_VARIANT, num_items=num_items
    )

    assert result["diagnostics"]["exact_rank_mean"] == 98
    assert result["diagnostics"]["target_in_top10"] == 0
    assert result["diagnostics"]["target_in_top50"] == 0
    assert result["metrics"] == {
        "cases": 1,
        "Recall@10": 0.0,
        "Recall@50": 0.0,
        "NDCG@10": 0.0,
        "MRR@10": 0.0,
        "CatalogueCoverage@10": 0.1,  # 10 distinct ids in top-10 / 100
    }


def test_scorer_must_return_full_catalogue_score_vector() -> None:
    num_items = 10
    scorer = FakeScorer([0.0] * num_items, num_items)  # missing the PAD slot
    row = _row(num_items=num_items, history=(2, 3), validation=4, test=7)
    with pytest.raises(ValueError, match="num_items \\+ 1"):
        evaluate_rows(scorer=scorer, rows=[row], variant=P6_VARIANT, num_items=num_items)


def test_frozen_p6_split_keeps_test_target_out_of_the_shared_history() -> None:
    rows = _load_split("dev") + _load_split("holdout")
    assert len(rows) == 200

    for row in rows:
        shared = set(history_for_row(row, P6_VARIANT))
        assert int(row["test_item_id"]) not in shared, row["case_id"]

    # The canonical variant legitimately includes the validation interaction.  The only way
    # the test target can then appear in the model history is the repeated-interaction case
    # (validation target == test target), which the protocol records and re-admits.
    canonical_hits = [
        row
        for row in rows
        if int(row["test_item_id"]) in set(history_for_row(row, CANONICAL_VARIANT))
    ]
    for row in canonical_hits:
        assert int(row["test_item_id"]) == int(row["validation_item_id"]), row["case_id"]


def test_case_identity_checksum_is_deterministic_and_frozen() -> None:
    dev = _load_split("dev")
    holdout = _load_split("holdout")
    first = scalar_identity_checksum(dev)
    second = scalar_identity_checksum(dev)
    assert first == second
    assert len(first) == 64
    assert first != scalar_identity_checksum(holdout)


def test_determinism_check_scores_the_same_inputs_twice() -> None:
    num_items = 10
    scores = [0.0] * (num_items + 1)
    scores[7] = 1.0
    scorer = FakeScorer(scores, num_items)
    rows = [
        _row(num_items=num_items, history=(2, 3), validation=4, test=7, case_id="c0"),
        _row(num_items=num_items, history=(4, 5), validation=6, test=7, case_id="c1"),
    ]
    result = determinism_check(
        scorer=scorer, rows=rows, variant=P6_VARIANT, num_items=num_items, count=2
    )
    assert result["cases_checked"] == 2
    assert result["deterministic"] is True
    # Each case is scored twice, so four model calls happened.
    assert len(scorer.model_histories) == 4


def test_bootstrap_ci_collapses_for_constant_values() -> None:
    interval = bootstrap_ci([0.5, 0.5, 0.5, 0.5], resamples=200, seed=1)
    assert interval == {"mean": 0.5, "ci_low": 0.5, "ci_high": 0.5}


def test_bootstrap_ci_is_seeded_and_bounded() -> None:
    values = [0.0] * 156 + [1.0] * 4
    first = bootstrap_ci(values, resamples=500, seed=20261107)
    second = bootstrap_ci(values, resamples=500, seed=20261107)
    assert first == second
    assert 0.0 <= first["ci_low"] <= first["mean"] <= first["ci_high"] <= 1.0
    assert abs(first["mean"] - 0.025) < 1e-12

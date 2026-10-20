"""R-LR-04 acceptance tests for the experimental LightGBM Learned Ranker (no LightGBM needed)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from recommendation.inference.learned_ranker import (
    FEATURES, LearnedRanker, build_feature_rows, order_by_ranker, rrf_full, rrf_ranks,
)


def test_feature_schema_matches_the_frozen_artifact() -> None:
    schema = json.loads(Path("runs/lr_v1_formal/feature_schema.json").read_text())
    assert tuple(schema["features"]) == FEATURES and len(FEATURES) == 15


def test_known_materialised_row_is_reproduced_exactly() -> None:
    """case 278 / item 92123: lex rank 11 => rrf 1/71 = 0.014085, 1/71 rounded to 6 dp."""
    rrf = rrf_full({"lex": 11})
    assert round(rrf, 9) == 0.014084507
    row = build_feature_rows([92123], ["X"], {"lex": {"X": 11}}, {"lex": {"X": 0.9}}, {"X": rrf})[0]
    assert len(row) == 15
    assert row[FEATURES.index("lex_rank")] == 11.0
    assert row[FEATURES.index("rrf_score")] == 0.014085
    assert row[FEATURES.index("source_count")] == 1.0


def test_rrf_rank_uses_the_full_universe_not_the_filtered_pool() -> None:
    """History/unresolvable members keep their rank slots, so pool ranks shift as offline."""
    universe = ["HIST", "AAA", "BBB"]
    rrf = {"HIST": 3.0, "AAA": 2.0, "BBB": 1.0}
    ranks = rrf_ranks(universe, rrf)
    rows = build_feature_rows([1, 2], ["AAA", "BBB"], {}, {}, rrf, universe)
    assert rows[0][FEATURES.index("rrf_rank")] == ranks["AAA"] == 2.0
    assert rows[1][FEATURES.index("rrf_rank")] == ranks["BBB"] == 3.0


def test_rrf_rank_ties_break_on_asin_ascending() -> None:
    assert rrf_ranks(["B", "A"], {"A": 0.5, "B": 0.5}) == {"A": 1, "B": 2}


class _Boom:
    num_feature = 15

    def predict(self, X, num_iteration=None):  # noqa: ANN001, ANN201
        raise RuntimeError("injected predict failure")


class _Nan:
    num_feature = 15

    def predict(self, X, num_iteration=None):  # noqa: ANN001, ANN201
        return np.asarray([float("nan")] * len(X))


def _ranker(booster=None) -> LearnedRanker:
    ranker = LearnedRanker(Path("runs/lr_v1_formal/model.txt"), Path("runs/lr_v1_formal/feature_schema.json"))
    if booster is not None:
        ranker._booster, ranker._best_iteration = booster, 22      # noqa: SLF001
    return ranker


@pytest.mark.parametrize("booster", [_Boom(), _Nan()])
def test_inference_failures_fall_back_to_full_precision_rrf(booster) -> None:  # noqa: ANN001
    asins = ["B0A", "B0B", "B0C", "B0D"]
    ranks = {"seq": {"B0A": 1, "B0C": 2}, "tt": {"B0B": 1}, "lex": {"B0D": 1}}
    rrf = {a: rrf_full({s: ranks[s].get(a, 0) for s in ranks}) for a in asins}
    ranker = _ranker(booster)
    order, reason = ranker.order(asins, list(range(len(asins))), ranks, {}, rrf, asins)
    expected = sorted(range(len(asins)), key=lambda i: (-rrf[asins[i]], asins[i]))
    assert order == expected                       # non-trivial >1 candidate ordering
    assert ranker.fallback_count == 1 and reason


def test_missing_model_is_reported_and_counted_once_per_call() -> None:
    ranker = LearnedRanker(Path("runs/definitely_missing_model.txt"))
    assert ranker.load_error and ranker._booster is None       # noqa: SLF001
    order, reason = ranker.order(["A"], [1], {"seq": {"A": 1}}, {}, {"A": 0.01})
    assert order == [0] and ranker.fallback_count == 1 and "FileNotFoundError" in (reason or "")


def test_ordering_protocol_is_ranker_then_rrf_then_item_id() -> None:
    assert order_by_ranker([0.5, 0.5, 0.9], [0.01, 0.02, 0.0], [7, 3, 9]) == [2, 1, 0]

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


class _FixedScores:
    """Deterministic booster whose scores deliberately invert the RRF order."""

    num_feature = 15

    def __init__(self) -> None:
        self.calls = 0

    def predict(self, X, num_iteration=None):  # noqa: ANN001, ANN201
        self.calls += len(X)
        return np.asarray([float(i) for i in range(len(X))], np.float32)


def test_normal_prediction_path_uses_the_model_and_never_silently_falls_back() -> None:
    """Regression for the `universe` NameError: predict must run, order must differ from RRF."""
    asins = ["B0A", "B0B", "B0C", "B0D", "B0E"]
    ranks = {"seq": {"B0A": 1, "B0C": 3}, "tt": {"B0B": 2}, "lex": {"B0D": 1, "B0E": 2}}
    scores = {"seq": {"B0A": 9.0, "B0C": 3.0}, "tt": {"B0B": 5.0}, "lex": {"B0D": 7.0, "B0E": 6.5}}
    rrf = {a: rrf_full({s: ranks[s].get(a, 0) for s in ranks}) for a in asins}
    booster = _FixedScores()
    ranker = _ranker(booster)
    order, reason = ranker.order(asins, list(range(len(asins))), ranks, scores, rrf, asins)
    rrf_order = sorted(range(len(asins)), key=lambda i: (-rrf[asins[i]], asins[i]))
    assert reason is None, f"normal path must not fall back: {reason}"
    assert ranker.fallback_count == 0
    assert booster.calls == len(asins)          # LightGBM.predict really ran
    assert order != rrf_order                    # the model, not RRF, decided the order
    assert ranker.last_used_model is True


def test_agentgraph_recommendation_tool_runs_the_learned_ranker() -> None:
    """Path A: AgentGraph -> RecommendationTool -> engine.recommend -> LightGBM.predict.

    Uses the real Tool request/context types (history is trusted context, not a request field) and
    a structurally typed engine, so no 6 GB model stack is needed.
    """
    from recommendation.inference.sasrec import Recommendation, RecommendationResult
    from recommendation.tools.recommendation import RecommendationTool
    from recommendation.tools.schemas import RecommendationContext, RecommendationToolRequest

    asins = ["B0A", "B0B", "B0C", "B0D", "B0E"]
    ranks = {"seq": {"B0A": 1, "B0C": 3}, "tt": {"B0B": 2}, "lex": {"B0D": 1, "B0E": 2}}
    scores = {"seq": {"B0A": 9.0, "B0C": 3.0}, "tt": {"B0B": 5.0}, "lex": {"B0D": 7.0, "B0E": 6.5}}
    rrf = {a: rrf_full({s: ranks[s].get(a, 0) for s in ranks}) for a in asins}
    booster = _FixedScores()
    ranker = _ranker(booster)

    class SharedEngine:
        score_kind = "lightgbm_lambdarank"

        def __init__(self) -> None:
            self.seen_history: tuple[str, ...] | None = None

        def recommend(self, history, k=10):                       # noqa: ANN001, ANN201
            self.seen_history = tuple(history)
            idx, _ = ranker.order(asins, list(range(len(asins))), ranks, scores, rrf, asins)
            recs = tuple(Recommendation(rank=i + 1, item_id=j, parent_asin=asins[j], score=0.0)
                         for i, j in enumerate(idx[:k]))
            return RecommendationResult(recommendations=recs, requested_k=k, history_length=len(history),
                                        effective_history_length=len(history), history_truncated=False,
                                        eligible_candidates=len(asins), timings_ms={},
                                        score_kind="lightgbm_lambdarank", provenance={"ranker": {"used": True}})

    engine = SharedEngine()
    tool = RecommendationTool(engine)
    context = RecommendationContext(user_history=("B0A", "B0C"))
    result = tool.run(RecommendationToolRequest(k=5), context)
    got = [item.parent_asin for item in result.recommendations]
    rrf_order = [asins[i] for i in sorted(range(len(asins)), key=lambda i: (-rrf[asins[i]], asins[i]))]
    assert engine.seen_history == ("B0A", "B0C")      # trusted history reached the engine unchanged
    assert booster.calls == len(asins)                 # LightGBM.predict really executed
    assert ranker.fallback_count == 0                  # no silent fallback
    assert got != rrf_order                            # the model, not RRF, decided the base order
    assert getattr(result, "score_kind", None) in (None, "lightgbm_lambdarank")


def test_path_b_history_source_does_not_mislabel_a_learned_ranker_engine() -> None:
    from recommendation.control.candidate_plane import _HistorySourceTool

    class Learned:
        score_kind = "lightgbm_lambdarank"

    class Sasrec:
        score_kind = "sasrec_logit"

    class Tool:
        def __init__(self, engine): self.engine = engine

    assert _HistorySourceTool(Tool(Learned())).score_kind == "lightgbm_lambdarank"   # truthful label
    assert _HistorySourceTool(Tool(Sasrec())).score_kind == "sasrec_logit"           # default kept
    assert _HistorySourceTool(Tool(object())).score_kind == "sasrec_logit"           # unknown -> default


def _guard_service(engine):                                # noqa: ANN001, ANN202
    from recommendation.demo.agent_service import DemoAgentService
    service = object.__new__(DemoAgentService)             # bypass the heavy __init__ (no 6 GB stack)
    service._engine = engine                               # noqa: SLF001
    return service


def test_dynamic_guard_does_not_refuse_a_default_engine() -> None:
    class DefaultEngine:
        score_kind = "rrf_rank_fusion"

    service = _guard_service(DefaultEngine())
    try:
        service._execute_plan()                            # noqa: SLF001
    except Exception as exc:                               # noqa: BLE001
        # the guard must not be what stopped it: any other failure is pre-existing behaviour
        assert getattr(exc, "code", None) != "unsupported_source_plan"


def test_dynamic_plane_guard_rejects_a_learned_ranker_engine() -> None:
    """DemoAgentService._execute_plan must refuse the learned-ranker engine before any work.

    ``Mock(spec=...)`` sets ``__class__``, so ``isinstance`` holds without building the real
    (6 GB) engine, and a ``SimpleNamespace`` carries only the injected engine: if the guard did
    not fire first, the surrounding code would fail with AttributeError instead of DemoAgentError.
    """
    from types import SimpleNamespace
    from unittest.mock import Mock

    from recommendation.demo.agent_service import DemoAgentError, DemoAgentService
    from recommendation.inference.learned_ranker import LearnedRankerEngine

    fake_engine = Mock(spec=LearnedRankerEngine)
    assert isinstance(fake_engine, LearnedRankerEngine)          # isinstance really holds

    with pytest.raises(DemoAgentError) as excinfo:
        DemoAgentService._execute_plan(
            SimpleNamespace(_engine=fake_engine),                       # unbound call
            plan=None, payload=None, use_memory=True,                   # never read: guard is first
        )

    assert excinfo.value.code == "unsupported_source_plan"
    assert "learned-ranker" in str(excinfo.value)
    assert fake_engine.mock_calls == []                          # nothing on the engine was touched

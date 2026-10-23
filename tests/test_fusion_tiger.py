"""TIGER as a peer source in the Serving fusion (T-02.5 sections A, B, D, E).

Everything here runs on injected fakes: the four-source wiring, the profile/depth contract, the
unified candidate conversion, provenance, failure semantics and concurrency isolation are all
engine-independent.  One real-model serving request is reported separately in the T-02.5 report,
because a real CPU cold start costs minutes and does not belong in a unit run.

The two properties that must never regress:

* **Serving V1 is untouched.**  ``FusionInferenceConfig()`` still describes three sources at the
  accepted depth, and a deployment without TIGER takes the identical code path (section E).
* **No silent degradation.**  A profile that names ``tiger`` refuses to run without a TIGER
  engine, and a TIGER failure aborts the request instead of returning a three-source answer
  labelled as four (section A).
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Sequence

import pytest

from recommendation.inference.tiger_engine import TigerEngineNotReady
from recommendation.control.tiger_source import TIGER_SCORE_KIND
from recommendation.inference.fusion import (
    FUSION_PROFILES,
    FUSION_SOURCES,
    FUSION_SOURCES_V2_TIGER,
    TIGER_SERVING_HEAD,
    TIGER_SOURCE,
    TIGER_SOURCE_SCORE_KIND,
    FusionError,
    FusionInferenceConfig,
    FusionInferenceEngine,
)
from recommendation.inference.sasrec import Recommendation, RecommendationResult

IDENTITY = {"A1": 1, "A2": 2, "A3": 3, "A4": 4, "A5": 5, "A6": 6}
KNOWN = frozenset(IDENTITY)


def item_id_for(parent_asin: str) -> int:
    return IDENTITY[parent_asin]


def result_for(parent_asins: Sequence[str], *, score_kind: str = "test") -> RecommendationResult:
    return RecommendationResult(
        recommendations=[
            Recommendation(rank=position, item_id=IDENTITY[asin], parent_asin=asin, score=-1.0 * position)
            for position, asin in enumerate(parent_asins, start=1)
        ],
        requested_k=len(parent_asins),
        history_length=0,
        effective_history_length=0,
        history_truncated=False,
        eligible_candidates=len(parent_asins),
        timings_ms={},
        score_kind=score_kind,
    )


class Sequential:
    device = "cpu"

    def __init__(self, ranking: Sequence[str]) -> None:
        self._ranking = tuple(ranking)

    def is_ready(self) -> bool:
        return True

    def has_parent_asin(self, value: str) -> bool:
        return value in KNOWN

    def parent_asin_to_item_id(self, value: str) -> int:
        return IDENTITY[value]

    def item_id_to_parent_asin(self, item_id: int) -> str:
        return {value: key for key, value in IDENTITY.items()}[int(item_id)]

    def model_metadata(self) -> dict[str, Any]:
        return {"active_engine": "sasrec", "checkpoint": "test"}

    def recommend(self, history: Sequence[str], k: int = 10) -> RecommendationResult:
        return result_for(self._ranking[:k])


class TwoTower:
    def __init__(self, ranking: Sequence[str]) -> None:
        self._ranking = tuple(ranking)

    def is_ready(self) -> bool:
        return True

    def parent_asin_to_item_id(self, value: str) -> int:
        return IDENTITY[value]

    def recommend(self, history_item_ids: Sequence[int], k: int = 10) -> RecommendationResult:
        return result_for(self._ranking[:k])


class Lexical:
    indexed_documents = 6
    catalogue_size = 6

    def __init__(self, ranking: Sequence[str]) -> None:
        self._ranking = tuple(ranking)

    def ranking(self, query: str, limit: int = 100) -> list[str]:
        return list(self._ranking[:limit])


@dataclass(frozen=True)
class TigerItem:
    parent_asin: str
    rank: int
    score: float
    semantic_id: tuple[int, ...]
    item_id: int


@dataclass(frozen=True)
class TigerResult:
    retrieval_mode: str
    recommendations: tuple[TigerItem, ...]


class Tiger:
    """A TIGER-shaped engine: per-history heads, so interleaving is observable."""

    def __init__(self, heads: dict[int, Sequence[str]], *, mode: str = "CERTIFIED",
                 started: bool = True, fail: Exception | None = None) -> None:
        self._heads = {int(key): tuple(value) for key, value in heads.items()}
        self._mode = mode
        self._started = started
        self._fail = fail
        self.calls: list[tuple[int, ...]] = []
        self._lock = threading.Lock()

    @property
    def started(self) -> bool:
        return self._started

    def health(self) -> dict[str, Any]:
        return {
            "state": "ready" if self._started else "loading",
            "started": self._started,
            "device": "cpu",
            "retrieval_mode": self._mode,
            "checkpoint_sha256": "c" * 64,
            "sid_artifact_sha256": "d" * 64,
            "error": None,
        }

    def parent_asin_to_item_id(self, value: str) -> int:
        return IDENTITY[value]

    def recommend(self, history_item_ids: Sequence[int], *, k: int = 20, **_: Any) -> TigerResult:
        if self._fail is not None:
            raise self._fail
        key = int(tuple(history_item_ids)[0]) if history_item_ids else 0
        with self._lock:
            self.calls.append(tuple(int(v) for v in history_item_ids))
        head = self._heads.get(key, ())
        return TigerResult(
            retrieval_mode=self._mode,
            recommendations=tuple(
                TigerItem(parent_asin=asin, rank=position, score=-0.5 * position,
                          semantic_id=(10, 20, position), item_id=IDENTITY.get(asin, 0))
                for position, asin in enumerate(head[:k], start=1)
            ),
        )


def build_engine(*, profile: str = "serving_v1", tiger: Any = None, **overrides: Any) -> FusionInferenceEngine:
    config = overrides.pop("config", None) or FusionInferenceConfig.for_profile(profile)
    return FusionInferenceEngine(
        sequential=Sequential(["A2", "A3"]),
        two_tower=TwoTower(["A3", "A4"]),
        lexical=Lexical(["A4"]),
        catalogue={},
        item_id_for=item_id_for,
        is_known_identity=lambda value: value in KNOWN,
        config=config,
        tiger=tiger,
    )


# --------------------------------------------------------------------------- #
# A. profiles and head depth
# --------------------------------------------------------------------------- #


def test_serving_v1_is_unchanged_and_remains_the_default() -> None:
    config = FusionInferenceConfig()
    assert config.sources == FUSION_SOURCES == ("sequential", "two_tower", "catalog_lexical")
    assert config.profile == "serving_v1"
    assert config.head_depths == {"sequential": 100, "two_tower": 100, "catalog_lexical": 100}
    engine = build_engine()  # no TIGER injected: the accepted path
    assert engine.tiger is None
    assert engine.tiger_health() is None
    assert engine.is_ready() is True
    fused = engine.recommend(["A1"], k=3)
    assert [item.parent_asin for item in fused.recommendations]
    assert all("tiger" not in entry["sources"] for entry in fused.provenance.values())


def test_the_v2_profile_is_separate_and_uniform_at_tigers_ceiling() -> None:
    assert "serving_v2_tiger" in FUSION_PROFILES
    config = FusionInferenceConfig.for_profile("serving_v2_tiger")
    assert config.sources == FUSION_SOURCES_V2_TIGER
    assert config.sources[:3] == FUSION_SOURCES, "V1's sources are a prefix, not a replacement"
    assert config.head_depths == {name: TIGER_SERVING_HEAD for name in FUSION_SOURCES_V2_TIGER}
    assert config.depth_for(TIGER_SOURCE) == TIGER_SERVING_HEAD == 20


def test_the_mixed_head_profile_records_the_depth_difference() -> None:
    config = FusionInferenceConfig.for_profile("serving_v2_tiger_mixed_head")
    assert config.depth_for("sequential") == 100
    assert config.depth_for("two_tower") == 100
    assert config.depth_for("catalog_lexical") == 100
    assert config.depth_for(TIGER_SOURCE) == 20


def test_a_tiger_profile_without_a_tiger_engine_is_refused() -> None:
    with pytest.raises(FusionError) as excinfo:
        build_engine(profile="serving_v2_tiger")
    assert "tiger" in str(excinfo.value)


def test_a_tiger_depth_above_its_ceiling_is_refused() -> None:
    config = FusionInferenceConfig(
        sources=FUSION_SOURCES_V2_TIGER,
        profile="serving_v2_tiger",
        source_head_depths={"sequential": 100, "two_tower": 100, "catalog_lexical": 100, TIGER_SOURCE: 50},
    )
    with pytest.raises(FusionError):
        build_engine(tiger=Tiger({1: ["A6"]}), config=config)


def test_an_unknown_profile_is_refused() -> None:
    with pytest.raises(FusionError):
        FusionInferenceConfig.for_profile("serving_v9")


# --------------------------------------------------------------------------- #
# D1. TIGER only
# --------------------------------------------------------------------------- #


def test_tiger_only_serving_request() -> None:
    config = FusionInferenceConfig(
        sources=(TIGER_SOURCE,), profile="serving_v2_tiger",
        source_head_depths={TIGER_SOURCE: 20},
    )
    engine = build_engine(tiger=Tiger({1: ["A5", "A6"]}), config=config)
    fused = engine.recommend(["A1"], k=5)
    assert [item.parent_asin for item in fused.recommendations] == ["A5", "A6"]
    assert fused.score_kind == "rrf_rank_fusion"
    assert "source_tiger_ms" in fused.timings_ms
    entry = fused.provenance["A5"]
    assert entry["sources"] == [TIGER_SOURCE]
    assert entry["tiger"]["score"] == -0.5
    assert entry["tiger"]["semantic_id"] == [10, 20, 1]
    assert entry["tiger"]["retrieval_mode"] == "CERTIFIED"


# --------------------------------------------------------------------------- #
# D2/D4/D5. four sources, provenance, source trace
# --------------------------------------------------------------------------- #


def test_four_source_fusion_keeps_provenance_and_a_tiger_trace() -> None:
    tiger = Tiger({1: ["A5", "A6"]})
    engine = build_engine(profile="serving_v2_tiger", tiger=tiger)
    fused = engine.recommend(["A1"], k=5)

    assert engine.config.profile == "serving_v2_tiger"
    assert tiger.calls == [(1,)], "the trusted history reached TIGER as item ids"
    # D5: the source trace names every source that ran, tiger included.
    for source in FUSION_SOURCES_V2_TIGER:
        assert f"source_{source}_ms" in fused.timings_ms
    # D4: a TIGER-only candidate carries TIGER's own numbers beside its fused rank.
    tiger_only = [asin for asin, entry in fused.provenance.items() if entry["sources"] == [TIGER_SOURCE]]
    assert tiger_only, "the TIGER head contributed nothing to the fused result"
    entry = fused.provenance[tiger_only[0]]
    assert entry["tiger"]["score_kind"] == TIGER_SOURCE_SCORE_KIND
    assert entry["source_ranks"][TIGER_SOURCE] >= 1
    assert entry["score_kind"] == "rrf_rank_fusion"


def test_already_seen_history_is_never_re_presented_by_tiger() -> None:
    engine = build_engine(profile="serving_v2_tiger", tiger=Tiger({1: ["A1", "A5"]}))
    fused = engine.recommend(["A1"], k=5)
    assert "A1" not in [item.parent_asin for item in fused.recommendations]
    assert fused.provenance["A5"]["sources"] == [TIGER_SOURCE]


def test_an_unknown_identity_from_tiger_is_dropped_not_presented() -> None:
    engine = build_engine(profile="serving_v2_tiger", tiger=Tiger({1: ["A5", "NOT-IN-CATALOGUE"]}))
    fused = engine.recommend(["A1"], k=5)
    assert "NOT-IN-CATALOGUE" not in [item.parent_asin for item in fused.recommendations]


def test_the_declared_tiger_score_kind_matches_the_source_module() -> None:
    assert TIGER_SOURCE_SCORE_KIND == TIGER_SCORE_KIND


# --------------------------------------------------------------------------- #
# D3/B. readiness and unavailability
# --------------------------------------------------------------------------- #


def test_a_cold_tiger_worker_keeps_the_engine_not_ready() -> None:
    engine = build_engine(profile="serving_v2_tiger", tiger=Tiger({1: ["A5"]}, started=False))
    assert engine.is_ready() is False, "a configured but cold worker must not report ready"
    assert engine.tiger_health()["state"] == "loading"

    warm = build_engine(profile="serving_v2_tiger", tiger=Tiger({1: ["A5"]}, started=True))
    assert warm.is_ready() is True
    assert warm.tiger_health()["state"] == "ready"


def test_a_tiger_failure_aborts_the_request_instead_of_serving_three_sources() -> None:
    engine = build_engine(
        profile="serving_v2_tiger",
        tiger=Tiger({1: ["A5"]}, fail=TigerEngineNotReady("the worker is still loading")),
    )
    with pytest.raises(TigerEngineNotReady):
        engine.recommend(["A1"], k=4)


def test_the_model_metadata_reports_the_profile_depths_and_tiger_health() -> None:
    engine = build_engine(profile="serving_v2_tiger", tiger=Tiger({1: ["A5"]}))
    described = engine.model_metadata()
    assert described["serving_fusion_profile"] == "serving_v2_tiger"
    assert described["source_head_depths"][TIGER_SOURCE] == 20
    assert described["depth_policy"] == "uniform_tiger_head"
    assert described["source_score_kinds"][TIGER_SOURCE] == TIGER_SOURCE_SCORE_KIND
    assert described["tiger_health"]["state"] == "ready"
    assert described["fusion_sources"] == list(FUSION_SOURCES_V2_TIGER)


# --------------------------------------------------------------------------- #
# D6. concurrency isolation
# --------------------------------------------------------------------------- #


def test_interleaved_requests_do_not_share_tiger_metadata() -> None:
    """Two users' requests run concurrently; neither may see the other's TIGER view."""
    tiger = Tiger({1: ["A5"], 2: ["A6"]})
    engine = build_engine(profile="serving_v2_tiger", tiger=tiger)
    barrier = threading.Barrier(2)
    seen: dict[int, dict[str, Any]] = {}

    def run(user: int, history: str) -> None:
        barrier.wait()
        for _ in range(5):
            fused = engine.recommend([history], k=5)
            # The isolation property: only the identities *this* request's TIGER head returned
            # may carry a tiger provenance block.  Other sources' candidates come from the shared
            # fakes and are irrelevant here.
            seen[user] = {
                "identities": {asin for asin, entry in fused.provenance.items() if "tiger" in entry},
                "modes": {entry["tiger"]["retrieval_mode"] for entry in fused.provenance.values()
                          if "tiger" in entry},
            }

    threads = [
        threading.Thread(target=run, args=(1, "A1")),
        threading.Thread(target=run, args=(2, "A2")),
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert seen[1]["identities"] == {"A5"}, seen[1]
    assert seen[2]["identities"] == {"A6"}, seen[2]
    assert seen[1]["modes"] == {"CERTIFIED"}
    assert seen[2]["modes"] == {"CERTIFIED"}


def test_the_fusion_engine_exposes_the_identity_surface_the_agent_grounds_with() -> None:
    """The T-02.7D 502: ``GroundingVerifier(fusion_engine, …)`` raised TypeError.

    The Agent layer grounds candidates with the trusted identity map; the fusion engine holds
    that map already and must expose it, or the agent endpoint fails before any source runs.
    """
    engine = build_engine(profile="serving_v2_tiger", tiger=Tiger({1: ["A5"]}))
    assert engine.has_parent_asin("A1") is True
    assert engine.has_parent_asin("NOT-IN-CATALOGUE") is False
    assert engine.parent_asin_to_item_id("A1") == 1
    assert engine.item_id_to_parent_asin(1) == "A1"

    # These three methods are exactly what ``GroundingVerifier.__init__`` requires
    # (grounding.py:163 raises when any is missing), so their presence is the condition the
    # Agent layer depends on.
    for required in ("has_parent_asin", "parent_asin_to_item_id", "item_id_to_parent_asin"):
        assert callable(getattr(engine, required, None)), required


def test_a_sequential_engine_without_an_identity_map_fails_loudly() -> None:
    from recommendation.inference.fusion import FusionError, FusionInferenceConfig

    engine = FusionInferenceEngine(
        sequential=SimpleNamespace(),           # no identity methods at all
        two_tower=TwoTower(["A4"]),
        lexical=Lexical(["A4"]),
        catalogue={},
        item_id_for=item_id_for,
        config=FusionInferenceConfig(sources=("sequential",)),
    )
    with pytest.raises(FusionError):
        engine.has_parent_asin("A1")

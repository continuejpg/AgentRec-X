"""Serving V1 acceptance: fixed multi-source retrieval fused into the **existing** serving chain.

What this proves, end to end and offline:

```text
history -> sequential + two_tower + catalogue lexical
        -> deterministic RRF
        -> fused candidates
        -> existing catalogue evidence (ProductEnricher)
        -> existing preference matcher (PreferenceCandidateMatcher)
        -> existing preference reranker (PreferenceReranker)
        -> final order + complete fusion provenance
```

The catalogue is synthetic and small, but every component after the retrieval step is the real one:
the lexical index, the fusion engine, the Tool, the enricher, the matcher and the reranker are the
shipped implementations.  The two model sources are injected stubs so the suite never loads a
640 MB checkpoint — the same convention the rest of the suite uses.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Sequence

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from recommendation.catalog.metadata import normalize_product_record  # noqa: E402
from recommendation.config import PAD_ID  # noqa: E402
from recommendation.control.candidate_ledger import RRF_K  # noqa: E402
from recommendation.inference.fusion import (  # noqa: E402
    DEFAULT_SOURCE_HEAD,
    FUSION_SCORE_KIND,
    FUSION_SOURCES,
    FusionError,
    FusionInferenceConfig,
    FusionInferenceEngine,
)
from recommendation.inference.lexical import CatalogueLexicalIndex  # noqa: E402
from recommendation.inference.sasrec import (  # noqa: E402
    Recommendation,
    RecommendationResult,
)
from recommendation.preference_matching import PreferenceCandidateMatcher  # noqa: E402
from recommendation.rag import ProductEnricher  # noqa: E402
from recommendation.reranking import PreferenceReranker  # noqa: E402
from recommendation.tools.recommendation import RecommendationTool  # noqa: E402
from recommendation.tools.schemas import (  # noqa: E402
    RecommendationContext,
    RecommendationToolRequest,
)
from tests.preference_matching_fixture import make_entry, make_snapshot  # noqa: E402

# --------------------------------------------------------------------------- #
# synthetic catalogue
# --------------------------------------------------------------------------- #

#: ``(parent_asin, item_id, title, features, colour)``.
#:
#: The vocabulary deliberately overlaps: ``camping`` appears in three titles and ``waterproof`` in
#: three feature lists, so the lexical source can contribute candidates the model sources never
#: proposed.  Without overlap a lexical retriever would be untestable here.
ROWS: tuple[tuple[str, int, str, tuple[str, ...], str], ...] = (
    ("cand-tent", 11, "Camping Tent 2 Person", ("waterproof", "lightweight"), "green"),
    ("cand-stove", 12, "Camping Stove Mini", ("lightweight",), "black"),
    ("cand-lamp", 13, "LED Camping Lamp", ("waterproof",), "white"),
    ("cand-pack", 14, "Hiking Backpack 40L", ("waterproof", "durable"), "blue"),
    ("cand-rope", 15, "Climbing Rope 30m", ("durable",), "red"),
)

HISTORY: tuple[str, ...] = ("cand-tent",)
ITEM_IDS: dict[str, int] = {row[0]: row[1] for row in ROWS}


def _records() -> dict[str, Any]:
    records = {}
    for parent_asin, _item_id, title, features, colour in ROWS:
        records[parent_asin] = normalize_product_record(
            {
                "parent_asin": parent_asin,
                "title": title,
                "features": list(features),
                "details": {"Color": colour},
            }
        )
    return records


def _index() -> Any:
    from recommendation.catalog import MetadataIndex

    return MetadataIndex.from_records(list(_records().values()))


# --------------------------------------------------------------------------- #
# injected sources
# --------------------------------------------------------------------------- #


class _StubSequential:
    """A deterministic single-model source that proposes its ranking verbatim."""

    def __init__(self, ranking: Sequence[str]) -> None:
        self.ranking = tuple(ranking)
        self.calls = 0

    def is_ready(self) -> bool:
        return True

    def recommend(self, history_parent_asins: Sequence[str], k: int = 10) -> RecommendationResult:
        self.calls += 1
        items = [
            Recommendation(
                rank=i,
                # An unknown identity gets a synthetic id so this stub can express what a
                # misbehaving source would return; the fusion engine must drop it.
                item_id=ITEM_IDS.get(asin, 900 + i),
                parent_asin=asin,
                score=1.0 / i,
            )
            for i, asin in enumerate(self.ranking, start=1)
            if asin not in set(history_parent_asins)
        ]
        return RecommendationResult(
            recommendations=items[:k],
            requested_k=k,
            history_length=len(history_parent_asins),
            effective_history_length=len(history_parent_asins),
            history_truncated=False,
            eligible_candidates=len(items),
        )

    def parent_asin_to_item_id(self, parent_asin: str) -> int:
        return ITEM_IDS[parent_asin]

    def has_parent_asin(self, parent_asin: str) -> bool:
        return parent_asin in ITEM_IDS


class _StubTwoTower(_StubSequential):
    """The Two-Tower source: integer-id history in, identities out."""

    def recommend(self, history_item_ids: Sequence[int], k: int = 10) -> RecommendationResult:
        self.calls += 1
        seen = set(int(value) for value in history_item_ids)
        items = [
            Recommendation(
                rank=i,
                item_id=ITEM_IDS.get(asin, 900 + i),
                parent_asin=asin,
                score=0.5,
            )
            for i, asin in enumerate(self.ranking, start=1)
            if ITEM_IDS.get(asin, -1) not in seen
        ]
        return RecommendationResult(
            recommendations=items[:k],
            requested_k=k,
            history_length=len(history_item_ids),
            effective_history_length=len(history_item_ids),
            history_truncated=False,
            eligible_candidates=len(items),
        )


def _engine(
    sequential_ranking: Sequence[str] = ("cand-rope", "cand-pack", "cand-tent"),
    two_tower_ranking: Sequence[str] = ("cand-rope", "cand-lamp"),
    *,
    known: bool = True,
) -> tuple[FusionInferenceEngine, _StubSequential, _StubTwoTower]:
    index = _index()
    records = index.records
    sequential = _StubSequential(sequential_ranking)
    two_tower = _StubTwoTower(two_tower_ranking)
    engine = FusionInferenceEngine(
        sequential=sequential,
        two_tower=two_tower,
        lexical=CatalogueLexicalIndex.build(records),
        catalogue=records,
        item_id_for=sequential.parent_asin_to_item_id,
        is_known_identity=sequential.has_parent_asin if known else None,
    )
    return engine, sequential, two_tower


# --------------------------------------------------------------------------- #
# the acceptance test
# --------------------------------------------------------------------------- #


def test_serving_v1_end_to_end_through_the_existing_chain() -> None:
    """All three sources -> RRF -> existing evidence, matcher and reranker."""
    engine, sequential, two_tower = _engine()

    result = engine.recommend(HISTORY, k=4)

    # -- the three sources really ran -------------------------------------- #
    assert sequential.calls == 1 and two_tower.calls == 1
    provenance = result.provenance
    assert provenance, "the fusion must report per-item provenance"
    participating = {name for entry in provenance.values() for name in entry["sources"]}
    assert participating == set(FUSION_SOURCES), participating
    # Fusion really combined: something is corroborated by two sources, and the lexical
    # source contributed something no model source proposed.
    assert any(len(entry["sources"]) >= 2 for entry in provenance.values())
    lexical_only = [a for a, e in provenance.items() if e["sources"] == ["catalog_lexical"]]
    assert lexical_only, provenance

    # -- identity, PAD and seen-item invariants ----------------------------- #
    returned = [item.parent_asin for item in result.recommendations]
    assert HISTORY[0] not in returned, "seen history must never be re-presented"
    assert all(item.item_id != PAD_ID for item in result.recommendations)
    assert all(item.item_id > 0 for item in result.recommendations)
    for item in result.recommendations:
        assert ITEM_IDS[item.parent_asin] == item.item_id, "identity must round-trip"
        assert provenance[item.parent_asin]["item_id"] == item.item_id
    assert [item.rank for item in result.recommendations] == list(range(1, len(returned) + 1))

    original_fused_first = returned[0]

    # -- the fused order is RRF(v1) ----------------------------------------- #
    assert FUSION_SOURCES == ("sequential", "two_tower", "catalog_lexical")
    assert FusionInferenceConfig().rrf_k == RRF_K == 60
    assert result.score_kind == FUSION_SCORE_KIND == "rrf_rank_fusion"
    for entry in provenance.values():
        assert entry["score_kind"] == FUSION_SCORE_KIND
        assert set(entry["source_score_kinds"].values()) <= {
            "sasrec_logit",
            "two_tower_cosine",
            "bm25",
        }
        for name, rank in entry["source_ranks"].items():
            assert and_rank_is_positive(rank), (name, rank)
    for first, second in zip(result.recommendations, result.recommendations[1:]):
        assert first.score >= second.score, "fused scores must be non-increasing"

    # -- determinism --------------------------------------------------------- #
    again = engine.recommend(HISTORY, k=4)
    assert [i.parent_asin for i in again.recommendations] == returned
    assert [i.score for i in again.recommendations] == [i.score for i in result.recommendations]
    assert again.provenance == provenance

    # -- the fused candidates reach the EXISTING evidence/matcher/reranker --- #
    tool = RecommendationTool(engine)
    tool_result = tool.run(
        RecommendationToolRequest(k=4),
        RecommendationContext(user_history=list(HISTORY)),
    )
    assert tool_result.score_kind == FUSION_SCORE_KIND
    assert set(tool_result.provenance) == set(returned)
    assert all(rec.score_kind == FUSION_SCORE_KIND for rec in tool_result.recommendations)
    assert tuple(rec.parent_asin for rec in tool_result.recommendations) == tuple(returned)

    enricher = ProductEnricher(_index())
    query = " ".join(row[2] for row in ROWS)
    enrichment = enricher.enrich(tool_result, query)
    assert {item.parent_asin for item in enrichment.items} == set(returned), (
        "enrichment must be candidate-scoped: it can annotate the fused set, never widen it"
    )

    assert original_fused_first == "cand-rope", original_fused_first
    snapshot = make_snapshot(make_entry(kind="color", value="red", polarity="avoid"))
    report = PreferenceCandidateMatcher().match(
        candidates=enrichment.items, preferences=snapshot
    )
    reranking = PreferenceReranker().rerank(report)

    final = list(reranking.parent_asins)
    assert sorted(final) == sorted(returned), "the reranker emits a permutation of its input"
    assert final[0] != "cand-rope"
    assert reranking.candidate_count == len(returned)
    # The avoided colour is supported on cand-rope, so it must end up last.
    assert final[-1] == "cand-rope", final
    assert reranking.moved_count >= 1, "the reorder must be observable"
    original = {c.parent_asin: c.original_rank for c in reranking.candidates}
    assert original == {asin: rank for rank, asin in enumerate(returned, start=1)}, (
        "the fused rank must survive as the reranker's original_rank"
    )
    # Provenance still describes every presented item after reranking.
    for candidate in reranking.candidates:
        assert candidate.parent_asin in result.provenance
        assert result.provenance[candidate.parent_asin]["fused_rank"] == candidate.original_rank


def and_rank_is_positive(rank: Any) -> bool:
    """A source rank is a 1-based position."""
    return isinstance(rank, int) and rank >= 1


# --------------------------------------------------------------------------- #
# guards and failure modes
# --------------------------------------------------------------------------- #


def test_the_lexical_source_matches_the_research_catalogue_source() -> None:
    """Behavioural drift guard: serving and research lexical retrieval must agree.

    The serving side reuses the shared tokenizer/IDF/BM25 constants; this asserts the *ranking* is
    identical too, so the two implementations cannot silently diverge.
    """
    from recommendation.control.catalog_search import CatalogSearchSource

    index = _index()
    research = CatalogSearchSource(index)
    serving = CatalogueLexicalIndex.build(index.records)
    for query in (
        "camping",
        "waterproof camping",
        "hiking backpack",
        "durable rope",
        "nothing-matches-this-token",
        "",
    ):
        assert serving.ranking(query, limit=10) == list(
            research.search(query, limit=10).parent_asins
        ), query


def test_the_lexical_source_contributes_an_item_no_model_proposed() -> None:
    """The third source is not decorative: it must be able to widen the candidate set."""
    engine, _sequential, two_tower = _engine()
    model_only = {"cand-pack", "cand-rope", "cand-lamp"}
    result = engine.recommend(HISTORY, k=5)
    lexical_only = [
        asin
        for asin, entry in result.provenance.items()
        if entry["sources"] == ["catalog_lexical"]
    ]
    assert lexical_only, [two_tower.ranking, result.provenance]
    assert set(lexical_only) - model_only


def test_history_is_never_partially_consumed() -> None:
    engine, _sequential, _two_tower = _engine()
    with pytest.raises(FusionError):
        engine.recommend((), k=3)
    with pytest.raises(FusionError):
        engine.recommend(HISTORY, k=0)


def test_an_unconfirmed_identity_is_rejected_in_the_history() -> None:
    engine, _sequential, _two_tower = _engine()
    with pytest.raises(FusionError) as excinfo:
        engine.recommend(("cand-unknown",), k=3)
    assert "outside the served catalogue" in str(excinfo.value)


def test_an_unconfirmed_returned_identity_is_dropped_not_presented() -> None:
    """Fail closed: a source that returns an unknown identity cannot get it presented."""
    engine, _sequential, _two_tower = _engine(two_tower_ranking=("cand-ghost", "cand-lamp"))
    result = engine.recommend(HISTORY, k=5)
    assert "cand-ghost" not in {item.parent_asin for item in result.recommendations}
    assert "cand-ghost" not in result.provenance


def test_fusion_configuration_is_fixed_not_tunable_at_request_time() -> None:
    """No weights, no per-request source selection: the source set is a construction argument."""
    engine, _sequential, _two_tower = _engine()
    assert engine.config.sources == FUSION_SOURCES
    assert engine.config.head_depth == DEFAULT_SOURCE_HEAD
    assert not hasattr(engine, "set_sources")
    assert not hasattr(engine, "weights")


def test_engine_metadata_declares_what_is_serving() -> None:
    engine, _sequential, _two_tower = _engine()
    metadata = engine.model_metadata()
    assert metadata["score_kind"] == FUSION_SCORE_KIND
    assert metadata["serving_fusion"] == "rrf"
    assert metadata["fusion_sources"] == list(FUSION_SOURCES)
    assert metadata["rrf_k"] == RRF_K


def test_build_engine_never_degrades_silently() -> None:
    """An explicitly requested fusion reports unready; it does not fall back to SASRec."""
    from recommendation.api.app import ServiceSettings, build_engine

    engine, error = build_engine(
        ServiceSettings(serving_fusion="rrf3", two_tower_dir=None)
    )
    assert engine is None
    assert error is not None and "rrf3" in error and "could not be built" in error

    engine, error = build_engine(ServiceSettings(serving_fusion="wishful"))
    assert engine is None
    assert error is not None and "unknown AGENTRECX_SERVING_FUSION" in error

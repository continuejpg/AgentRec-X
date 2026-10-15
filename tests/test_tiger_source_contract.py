"""Contract tests for the corrected TIGER candidate-source seam (section A of T-02).

The engine is a duck-typed fake here: these tests pin the *source layer's* contract - validate
the whole returned frontier, filter already-interacted items deterministically, cap at the
requested ``k``, re-assign contiguous source ranks, never touch the model's scores, never
relabel the retrieval mode.  Real-model behaviour is covered by ``tests/test_tiger_engine_real.py``.

Written because the pre-T-02 seam forwarded the engine's raw list: in CERTIFIED mode that is the
certified frontier (``k + |unique history|`` rows), which could exceed ``limit`` and could contain
items the user had already interacted with, with ranks that stopped being contiguous once
anything was dropped.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import pytest

from recommendation.control.candidate_plane import CandidateSourceUnavailable
from recommendation.control.tiger_source import TIGER_SCORE_KIND, TigerSourceTool

CERTIFIED = "CERTIFIED"
APPROXIMATE = "APPROXIMATE"


@dataclass(frozen=True)
class FakeItem:
    item_id: int
    parent_asin: str
    rank: int
    score: float
    semantic_id: tuple[int, ...] = (1, 2, 3, 0)


@dataclass(frozen=True)
class FakeResult:
    retrieval_mode: str
    recommendations: tuple[FakeItem, ...]
    checkpoint_sha256: str = "a" * 64
    sid_artifact_sha256: str = "b" * 64
    resolver: Mapping[str, Any] = field(
        default_factory=lambda: {"candidate_item_for_path": "smallest_item_id_wins"}
    )


class FakeEngine:
    """Identity maps 101.. to ``ASIN-<n>``; history is given as integer item ids."""

    def __init__(self, result: FakeResult) -> None:
        self._result = result
        self.calls: list[dict[str, Any]] = []

    def parent_asin_to_item_id(self, parent_asin: str) -> int:
        return int(str(parent_asin).split("-")[1])

    def item_id_to_parent_asin(self, item_id: int) -> str:
        return f"ASIN-{int(item_id)}"

    def recommend(self, history_item_ids: Sequence[int], *, k: int = 20, **_: Any) -> FakeResult:
        self.calls.append({"history": list(history_item_ids), "k": int(k)})
        return self._result


def frontier(*item_ids: int, mode: str = CERTIFIED, scores: dict[int, float] | None = None) -> FakeResult:
    """A model-ordered frontier with contiguous 1..N ranks, exactly as the engine emits it."""
    scores = scores or {}
    items = tuple(
        FakeItem(item_id=int(item_id), parent_asin=f"ASIN-{item_id}", rank=position,
                 score=float(scores.get(int(item_id), -1.0 * position)))
        for position, item_id in enumerate(item_ids, start=1)
    )
    return FakeResult(retrieval_mode=mode, recommendations=items)


def tool_for(result: FakeResult, **kwargs: Any) -> tuple[TigerSourceTool, FakeEngine]:
    engine = FakeEngine(result)
    return TigerSourceTool(engine, max_k=kwargs.pop("max_k", 20), **kwargs), engine


# --------------------------------------------------------------------------- #
# requested_k / returned_k
# --------------------------------------------------------------------------- #


def test_frontier_longer_than_limit_is_capped_at_the_requested_k() -> None:
    """A CERTIFIED frontier of 35 rows must come back as at most ``limit`` triples."""
    tool, _ = tool_for(frontier(*range(101, 136)))
    triples = tool.propose(arguments=None, limit=5, history=("ASIN-900",))
    assert len(triples) == 5
    outcome = tool.last_outcome
    assert outcome["requested_k"] == 5
    assert outcome["returned_k"] == 5
    assert outcome["frontier_size"] == 35
    assert outcome["seen_filtered"] == 0
    assert outcome["frontier_short"] is False
    assert [rank for _, rank, _ in triples] == [1, 2, 3, 4, 5]
    assert [asin for asin, _, _ in triples] == [f"ASIN-{i}" for i in range(101, 106)]


def test_limit_is_itself_capped_by_the_tools_max_k() -> None:
    tool, _ = tool_for(frontier(*range(101, 121)), max_k=8)
    triples = tool.propose(arguments=None, limit=100, history=("ASIN-900",))
    assert len(triples) == 8
    assert tool.last_outcome["requested_k"] == 8


# --------------------------------------------------------------------------- #
# the full frontier is validated before anything is dropped
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "broken",
    [
        FakeItem(item_id=0, parent_asin="ASIN-0", rank=3, score=-3.0),          # PAD is not an item
        FakeItem(item_id=104, parent_asin="", rank=4, score=-4.0),             # identity unknown
        FakeItem(item_id=105, parent_asin="ASIN-105", rank=9, score=-5.0),     # rank gap
        FakeItem(item_id=106, parent_asin="ASIN-106", rank=6, score=-6.0,
                 semantic_id=()),                                              # no provenance
    ],
)
def test_a_malformed_row_is_refused_even_when_filtering_would_have_dropped_it(broken: FakeItem) -> None:
    items = [FakeItem(item_id=i, parent_asin=f"ASIN-{i}", rank=i - 100, score=-1.0 * i)
             for i in range(101, 104)]
    items.append(broken)
    # The broken row is also a seen item, i.e. one the filter would have removed.  It must still
    # fail the call: validation happens before filtering, so nothing is hidden.
    seen_item = broken.item_id
    result = FakeResult(retrieval_mode=CERTIFIED, recommendations=tuple(items))
    tool, _ = tool_for(result)
    history = (f"ASIN-{seen_item}", f"ASIN-{seen_item}", "ASIN-900")
    with pytest.raises(CandidateSourceUnavailable):
        tool.propose(arguments=None, limit=5, history=history)


def test_an_unknown_engine_mode_is_refused() -> None:
    tool, _ = tool_for(frontier(101, mode="BEST_EFFORT"))
    with pytest.raises(CandidateSourceUnavailable):
        tool.propose(arguments=None, limit=5, history=("ASIN-900",))


def test_approximate_is_never_reported_as_certified() -> None:
    tool, _ = tool_for(frontier(101, 102, mode=APPROXIMATE), require_certified=True)
    with pytest.raises(CandidateSourceUnavailable):
        tool.propose(arguments=None, limit=5, history=("ASIN-900",))
    assert tool.last_outcome["reason"] == "not_certified"
    assert tool.last_outcome["retrieval_mode"] == APPROXIMATE

    relaxed, _ = tool_for(frontier(101, 102, mode=APPROXIMATE), require_certified=False)
    relaxed.propose(arguments=None, limit=5, history=("ASIN-900",))
    assert relaxed.last_outcome["retrieval_mode"] == APPROXIMATE
    assert relaxed.last_proposals[0].retrieval_mode == APPROXIMATE


# --------------------------------------------------------------------------- #
# seen-item filtering
# --------------------------------------------------------------------------- #


def test_already_interacted_items_are_filtered_and_order_is_preserved() -> None:
    tool, _ = tool_for(frontier(101, 102, 103, 104, 105, 106))
    triples = tool.propose(arguments=None, limit=4, history=("ASIN-103", "ASIN-104", "ASIN-900"))
    assert [asin for asin, _, _ in triples] == ["ASIN-101", "ASIN-102", "ASIN-105", "ASIN-106"]
    outcome = tool.last_outcome
    assert outcome["seen_filtered"] == 2
    assert outcome["frontier_size"] == 6
    assert outcome["returned_k"] == 4


def test_ranks_are_reassigned_contiguously_while_scores_are_untouched() -> None:
    scores = {101: -1.5, 102: -2.25, 103: -3.75, 104: -4.5}
    tool, _ = tool_for(frontier(101, 102, 103, 104, scores=scores))
    triples = tool.propose(arguments=None, limit=4, history=("ASIN-102", "ASIN-900"))
    # Rank 1 is now the first *surviving* model row; its score is still the model's own.
    assert triples == [("ASIN-101", 1, -1.5), ("ASIN-103", 2, -3.75), ("ASIN-104", 3, -4.5)]
    assert [p.rank for p in tool.last_proposals] == [1, 2, 3]
    assert [p.item_id for p in tool.last_proposals] == [101, 103, 104]


def test_every_seen_item_yields_an_explicit_empty_outcome() -> None:
    tool, _ = tool_for(frontier(101, 102, 103))
    triples = tool.propose(arguments=None, limit=5, history=("ASIN-101", "ASIN-102", "ASIN-103"))
    assert triples == []
    assert tool.last_proposals == ()
    assert tool.last_outcome["state"] == "EMPTY"
    assert tool.last_outcome["reason"] == "legal_catalogue_exhausted"
    assert tool.last_outcome["seen_filtered"] == 3


def test_a_short_frontier_is_reported_rather_than_padded() -> None:
    tool, _ = tool_for(frontier(101, 102, 103))
    triples = tool.propose(arguments=None, limit=10, history=("ASIN-900",))
    assert len(triples) == 3, "the source never invents candidates to reach k"
    assert tool.last_outcome["frontier_short"] is True
    assert tool.last_outcome["reason"] == "frontier_short"


def test_a_frontier_shortened_by_filtering_says_so() -> None:
    tool, _ = tool_for(frontier(101, 102, 103, 104))
    triples = tool.propose(arguments=None, limit=4, history=("ASIN-102", "ASIN-104", "ASIN-900"))
    assert len(triples) == 2
    assert tool.last_outcome["reason"] == "frontier_short_after_seen_filter"


def test_an_empty_retrieval_is_not_an_error_but_is_labelled() -> None:
    tool, _ = tool_for(FakeResult(retrieval_mode=CERTIFIED, recommendations=()))
    assert tool.propose(arguments=None, limit=5, history=("ASIN-900",)) == []
    assert tool.last_outcome["reason"] == "retrieval_returned_no_candidates"
    assert tool.last_outcome["frontier_size"] == 0


# --------------------------------------------------------------------------- #
# provenance
# --------------------------------------------------------------------------- #


def test_provenance_rides_on_every_surviving_proposal() -> None:
    tool, _ = tool_for(frontier(101, 102, 103))
    triples = tool.propose(arguments=None, limit=2, history=("ASIN-900",))
    assert len(triples) == 2
    for proposal in tool.last_proposals:
        assert proposal.retrieval_mode == CERTIFIED
        assert proposal.checkpoint_sha256 == "a" * 64
        assert proposal.sid_artifact_sha256 == "b" * 64
        assert proposal.resolver["candidate_item_for_path"] == "smallest_item_id_wins"
        assert proposal.provenance_token().startswith("mode=CERTIFIED;ckpt=aaaaaaaaaaaa")
    assert tool.score_kind == TIGER_SCORE_KIND
    assert tool.provenance_token() == tool.last_proposals[0].provenance_token()


def test_no_history_is_a_source_local_failure() -> None:
    tool, engine = tool_for(frontier(101))
    with pytest.raises(CandidateSourceUnavailable):
        tool.propose(arguments=None, limit=5, history=())
    assert tool.last_outcome["reason"] == "no_trusted_history"
    assert engine.calls == []


def test_unmappable_history_never_reaches_the_engine() -> None:
    tool, engine = tool_for(frontier(101))
    with pytest.raises(CandidateSourceUnavailable):
        tool.propose(arguments=None, limit=5, history=("not-an-asin",))
    assert tool.last_outcome["reason"] == "history_unmappable"
    assert engine.calls == []

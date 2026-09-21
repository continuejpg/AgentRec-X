"""Candidate-source integration tests for Two-Tower.

What these protect
------------------
Exposing a new model to the agent runtime must not change the trust boundary.  Three properties
are asserted here:

* Two-Tower introduces candidates **only** through the accepted candidate-source seam, so every
  identity it proposes is grounded by the plane's verifier and recorded in the ledger with its
  own ``score_kind``;
* an identity the trusted mapping does not know is recorded as an UNGROUNDED audit entry rather
  than silently dropped, exactly as for every other source;
* its scores are labelled as cosines, so nothing can compare them against a SASRec logit.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from recommendation.control import (  # noqa: E402
    CandidateLedger,
    CandidatePlane,
    CandidateSource,
    GroundingVerifier,
)
from recommendation.control.two_tower_source import (  # noqa: E402
    TWO_TOWER_SCORE_KIND,
    TwoTowerSourceTool,
)
from recommendation.inference import Recommendation, RecommendationResult  # noqa: E402


class _IdentityMap:
    """A trusted mapping over a tiny fixed catalogue."""

    def __init__(self, identities: dict[str, int]) -> None:
        self._by_asin = dict(identities)
        self._by_id = {value: key for key, value in identities.items()}

    def has_parent_asin(self, parent_asin: str) -> bool:
        return str(parent_asin) in self._by_asin

    def parent_asin_to_item_id(self, parent_asin: str) -> int:
        if str(parent_asin) not in self._by_asin:
            raise KeyError(parent_asin)
        return self._by_asin[str(parent_asin)]

    def item_id_to_parent_asin(self, item_id: int) -> str:
        if int(item_id) not in self._by_id:
            raise KeyError(item_id)
        return self._by_id[int(item_id)]


class _Catalog:
    """A catalogue that knows a subset of identities."""

    def __init__(self, identities: set[str]) -> None:
        self._identities = set(identities)

    def __contains__(self, parent_asin: object) -> bool:
        return str(parent_asin) in self._identities


class _Engine:
    """A minimal two-tower engine: returns a fixed ranking over the trusted history's ids."""

    def __init__(self, mapping: dict[str, int], ordering: list[str]) -> None:
        self.mapping = dict(mapping)
        self.ordering = list(ordering)
        self.calls: list[list[int]] = []

    def has_parent_asin(self, parent_asin: str) -> bool:
        return str(parent_asin) in self.mapping

    def parent_asin_to_item_id(self, parent_asin: str) -> int:
        if str(parent_asin) not in self.mapping:
            raise KeyError(parent_asin)
        return self.mapping[str(parent_asin)]

    def recommend(self, history_item_ids, k: int = 10) -> RecommendationResult:
        history = [int(value) for value in history_item_ids]
        self.calls.append(history)
        return RecommendationResult(
            recommendations=[
                Recommendation(rank=rank, item_id=1, parent_asin=asin, score=0.5 - rank * 0.01)
                for rank, asin in enumerate(self.ordering[:k], start=1)
            ],
            requested_k=k,
            history_length=len(history),
            effective_history_length=len(history),
            history_truncated=False,
            eligible_candidates=10,
        )


MAPPING = {"B1": 1, "B2": 2, "B3": 3, "B4": 4, "B5": 5}


def build_plane(*, ordering: list[str], catalog: set[str] | None = None) -> tuple[CandidatePlane, CandidateLedger]:
    ledger = CandidateLedger()
    plane = CandidatePlane(
        ledger=ledger,
        grounding=GroundingVerifier(_IdentityMap(MAPPING), _Catalog(catalog or set(MAPPING))),
        two_tower_tool=TwoTowerSourceTool(_Engine(MAPPING, ordering)),
    )
    return plane, ledger


# --------------------------------------------------------------------------- #
# The seam
# --------------------------------------------------------------------------- #


def test_two_tower_is_a_named_candidate_source():
    """A distinct member, not a reuse of HISTORY: the score kinds differ."""
    assert CandidateSource.TWO_TOWER.value == "two_tower"
    assert CandidateSource.TWO_TOWER is not CandidateSource.HISTORY


def test_the_tool_declares_its_source_and_score_kind():
    tool = TwoTowerSourceTool(_Engine(MAPPING, ["B1", "B2"]))
    assert tool.source is CandidateSource.TWO_TOWER
    assert tool.score_kind == TWO_TOWER_SCORE_KIND == "two_tower_cosine"


def test_a_tool_that_is_not_a_two_tower_source_is_refused():
    class _Wrong:
        @property
        def source(self) -> CandidateSource:
            return CandidateSource.HISTORY

        @property
        def score_kind(self) -> str:
            return "sasrec_logit"

        def propose(self, *, arguments, limit):  # pragma: no cover - never reached
            return []

    with pytest.raises(Exception, match="TWO_TOWER"):
        CandidatePlane(
            ledger=CandidateLedger(),
            grounding=GroundingVerifier(_IdentityMap(MAPPING), _Catalog(set(MAPPING))),
            two_tower_tool=_Wrong(),
        )


def test_plane_reports_the_source_it_actually_has():
    plane, _ledger = build_plane(ordering=["B1", "B2"])
    assert plane.has_source(CandidateSource.TWO_TOWER)
    assert not plane.has_source(CandidateSource.CATALOG_SEARCH)


# --------------------------------------------------------------------------- #
# Grounding and provenance
# --------------------------------------------------------------------------- #


def test_two_tower_candidates_are_grounded_and_recorded_with_provenance():
    """The core guarantee: identities arrive grounded and attributed, with a labelled score."""
    plane, ledger = build_plane(ordering=["B3", "B1"])
    result = plane.execute(
        _validated(),
        read_trusted_history=lambda: ("B1", "B2"),
        step_index=0,
    )
    assert result.returned_k == 2
    assert tuple(result.grounded_parent_asins) == ("B3", "B1")
    assert result.ungrounded_count == 0

    entries = {entry.parent_asin: entry for entry in ledger.grounded_entries}
    assert set(entries) == {"B3", "B1"}
    for entry in entries.values():
        assert len(entry.provenance) == 1
        record = entry.provenance[0]
        assert record.source is CandidateSource.TWO_TOWER
        assert record.score_kind == "two_tower_cosine"
        assert record.source_rank >= 1
    # The ledger reports the source as present, which is what a fused arm reads.
    assert ledger.sources_present() == (CandidateSource.TWO_TOWER,)


def test_an_identity_the_mapping_does_not_know_is_recorded_as_ungrounded():
    """A source that invents an identity must be visible, not silently dropped."""
    plane, ledger = build_plane(ordering=["B3", "NOT-IN-CATALOGUE"])
    result = plane.execute(
        _validated(), read_trusted_history=lambda: ("B1",), step_index=0
    )
    assert tuple(result.grounded_parent_asins) == ("B3",)
    assert result.ungrounded_count == 1
    assert len(ledger.ungrounded_entries) == 1
    # Grounded-only views stay clean, so a downstream renderer cannot present the bad identity.
    assert ledger.grounded_parent_asins() == ("B3",)


def test_a_source_with_no_trusted_history_fails_rather_than_answering_from_nothing():
    plane, _ledger = build_plane(ordering=["B1"])
    with pytest.raises(Exception):
        plane.execute(_validated(), read_trusted_history=lambda: (), step_index=0)


def test_an_unresolvable_history_identity_is_an_unavailable_source():
    """The engine is keyed by trusted identities; an unknown one cannot be silently skipped."""
    plane, _ledger = build_plane(ordering=["B1"])
    with pytest.raises(Exception):
        plane.execute(
            _validated(), read_trusted_history=lambda: ("B1", "NOT-KNOWN"), step_index=0
        )


def _validated():
    """A validated ``SELECT_SOURCE {two_tower}`` action, built through the accepted types."""
    from recommendation.control import (
        ActionKind,
        CandidateSource as _Source,
        SelectSourceArguments,
        ValidatedAction,
    )

    return ValidatedAction(
        action=ActionKind.SELECT_SOURCE,
        action_id="a1",
        step_index=0,
        run_id="r1",
        # ``k`` is carried alongside the typed arguments for a candidate-producing action;
        # the source's own budget lives in ``SelectSourceArguments.limit``.
        k=3,
        arguments=SelectSourceArguments(source=_Source.TWO_TOWER, limit=3),
    )

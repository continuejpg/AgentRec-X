"""Step 2.7 qualification tests for the TIGER-FP32 candidate source.

Covers the qualification gates Q1-Q7.  Tests that need the frozen certified evidence are
skipped when the artifact is absent, so the suite runs anywhere while the real-data gates run
where the artifact lives.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from recommendation.control import (
    ActionKind,
    ActionProposal,
    CandidateLedger,
    CandidatePlane,
    CandidateSource,
    CandidateVerificationStatus,
    GroundingVerifier,
    SelectSourceArguments,
    ValidatedAction,
)
from recommendation.control.candidate_plane import CandidateSourceTool, CandidateSourceUnavailable
from recommendation.control.tiger_source import (
    TIGER_SCORE_KIND,
    TIGER_SOURCE_NAME,
    TigerProposal,
    TigerSourceTool,
)
from recommendation.control.two_tower_source import TWO_TOWER_SCORE_KIND

ARTIFACT_DIR = Path("/root/autodl-tmp/step26_h7_canonical")
MAPPINGS = Path("data/processed/Sports_and_Outdoors_mappings.json")
def _probe(path: Path) -> bool:
    try:
        return path.exists()
    except OSError:  # an unreadable path is simply "not present" for skip purposes
        return False


EVIDENCE = _probe(ARTIFACT_DIR) and _probe(MAPPINGS)
needs_evidence = pytest.mark.skipif(not EVIDENCE, reason="frozen certified evidence not present")


def engine_module():
    import sys

    root = Path(__file__).resolve().parents[1]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from experiments import step27_tiger_engines as mod

    return mod


# --------------------------------------------------------------------------- #
# Grounding stand-ins (the same shape the other control tests use)
# --------------------------------------------------------------------------- #
class _Map:
    def __init__(self, identities) -> None:
        self._ids = sorted(str(value) for value in identities)
        self._to_id = {value: index + 1 for index, value in enumerate(self._ids)}

    def has_parent_asin(self, parent_asin: str) -> bool:
        return parent_asin in self._to_id

    def parent_asin_to_item_id(self, parent_asin: str) -> int:
        return self._to_id[parent_asin]

    def item_id_to_parent_asin(self, item_id: int) -> str:
        return self._ids[int(item_id) - 1]


class _Catalog:
    def __init__(self, identities) -> None:
        self._ids = {str(value) for value in identities}

    def __contains__(self, parent_asin: object) -> bool:
        return str(parent_asin) in self._ids


def make_plane(tiger_tool=None, *, identities=("A1", "A2", "A3"), extra_tools=()):
    mapping = _Map(identities)
    verifier = GroundingVerifier(mapping, _Catalog(identities))
    kwargs = {"tiger_tool": tiger_tool} if tiger_tool is not None else {}
    for name, tool in extra_tools:
        kwargs[name] = tool
    plane = CandidatePlane(ledger=CandidateLedger(), grounding=verifier, **kwargs)
    return plane, mapping


def select(source, limit=10):
    """A controller-stamped SELECT_SOURCE action, exactly as the loop would hand the plane."""
    return ValidatedAction(
        action=ActionKind.SELECT_SOURCE,
        action_id=f"act:{source.value}:{limit}",
        step_index=0,
        run_id="r",
        k=limit,
        arguments=SelectSourceArguments(source=source, limit=limit),
    )


def rec(asin, rank, score=None, sid=(0, 1, 2, 3), item_id=None):
    mod = engine_module()
    value = float(-2.0 - rank) if score is None else float(score)
    return mod.EngineRecommendation(parent_asin=asin, rank=rank, score=value,
                                    semantic_id=tuple(sid),
                                    item_id=int(item_id if item_id is not None else rank))


# --------------------------------------------------------------------------- #
# Q1 -- source contract compliance
# --------------------------------------------------------------------------- #
def test_tool_satisfies_the_candidate_source_protocol():
    mod = engine_module()
    tool = TigerSourceTool(mod.FakeTigerEngine(recommendations=(rec("A1", 1),)))
    assert isinstance(tool, CandidateSourceTool)
    assert tool.source is CandidateSource.TIGER
    assert tool.score_kind == TIGER_SCORE_KIND
    assert isinstance(tool.propose, object)


def test_score_kind_is_distinct_from_the_other_learned_sources():
    assert TIGER_SCORE_KIND != TWO_TOWER_SCORE_KIND
    assert TIGER_SCORE_KIND not in ("sasrec_logit", "bm25", "item_item_similarity", "unspecified")


def test_enum_member_is_candidate_producing_and_registrable():
    from recommendation.control.arguments import CANDIDATE_PRODUCING_SOURCES

    assert CandidateSource.TIGER.value == "tiger"
    assert CandidateSource.TIGER in CANDIDATE_PRODUCING_SOURCES


def test_plane_registers_the_tiger_source_and_rejects_a_mislabelled_tool():
    mod = engine_module()
    tool = TigerSourceTool(mod.FakeTigerEngine(recommendations=(rec("A1", 1),)))
    plane, _ = make_plane(tool)
    assert plane.has_source(CandidateSource.TIGER)
    assert CandidateSource.TIGER in plane.available_sources()
    assert plane.status()["sources"]

    class _Wrong:
        source = CandidateSource.HISTORY
        score_kind = "nope"

        def propose(self, **kwargs):
            return []

    with pytest.raises(Exception):
        make_plane(_Wrong())


def test_engine_without_the_required_seam_is_refused():
    with pytest.raises(CandidateSourceUnavailable):
        TigerSourceTool(object())
    with pytest.raises(CandidateSourceUnavailable):
        TigerSourceTool(type("E", (), {"recommend": lambda self, ids, k: None})())


# --------------------------------------------------------------------------- #
# Q2 / Q3 -- identity, grounding and ledger provenance
# --------------------------------------------------------------------------- #
def test_every_recorded_candidate_is_grounded_and_carries_tiger_provenance():
    mod = engine_module()
    recs = (rec("A1", 1, -2.5), rec("A2", 2, -3.5))
    tool = TigerSourceTool(mod.FakeTigerEngine(recommendations=recs))
    plane, _ = make_plane(tool, identities=("A1", "A2", "A3"))
    result = plane.execute(select(CandidateSource.TIGER, 5),
                           read_trusted_history=lambda: ("A3",), step_index=0)
    ledger = plane.ledger
    assert len(ledger) == 2
    assert len(ledger.ungrounded_entries) == 0
    for entry in ledger.entries:
        assert entry.provenance_for(CandidateSource.TIGER)
        provenance = entry.provenance[0]
        assert provenance.source is CandidateSource.TIGER
        assert provenance.score_kind == TIGER_SCORE_KIND
        assert provenance.source_rank >= 1
        assert provenance.source_score is not None
        assert provenance.source_query and "mode=CERTIFIED" in provenance.source_query
        assert "ckpt=" in provenance.source_query and "sid=" in provenance.source_query


def test_unknown_identity_is_recorded_as_ungrounded_not_fabricated():
    mod = engine_module()
    tool = TigerSourceTool(mod.FakeTigerEngine(recommendations=(rec("A1", 1), rec("ZZZ", 2))))
    plane, _ = make_plane(tool, identities=("A1", "A2"))
    plane.execute(select(CandidateSource.TIGER, 5), read_trusted_history=lambda: ("A2",))
    ledger = plane.ledger
    # The ledger records an unknown identity as an UNGROUNDED audit entry rather than dropping
    # it silently, and it never enters the grounded set a renderer may use.
    assert len(ledger.grounded_entries) == 1
    assert len(ledger.ungrounded_entries) == 1
    assert "ZZZ" not in ledger.grounded_parent_asins()
    entry = ledger.get("ZZZ")
    assert entry is not None
    assert entry.verification_status is CandidateVerificationStatus.UNGROUNDED
    assert {e.parent_asin for e in ledger.grounded_entries} == {"A1"}


def test_full_proposal_record_retains_every_required_field():
    mod = engine_module()
    tool = TigerSourceTool(mod.FakeTigerEngine(recommendations=(rec("A2", 1, -1.25),)))
    plane, _ = make_plane(tool, identities=("A1", "A2"))
    plane.execute(select(CandidateSource.TIGER, 3), read_trusted_history=lambda: ("A1",))
    proposal = tool.last_proposals[0]
    assert isinstance(proposal, TigerProposal)
    payload = proposal.as_dict()
    for field in ("parent_asin", "semantic_id", "score", "rank", "retrieval_mode",
                  "checkpoint_sha256", "sid_artifact_sha256", "resolver", "source",
                  "source_version", "item_id"):
        assert field in payload, field
    assert payload["parent_asin"] == "A2"
    assert payload["retrieval_mode"] == "CERTIFIED"
    assert payload["source"] == TIGER_SOURCE_NAME
    assert probe_provenance_token(tool)


def probe_provenance_token(tool) -> bool:
    token = tool.last_proposals[0].provenance_token()
    return 0 < len(token) <= 200 and "mode=CERTIFIED" in token


# --------------------------------------------------------------------------- #
# Q4 -- deterministic deduplication and multi-source provenance
# --------------------------------------------------------------------------- #
def test_same_identity_from_two_sources_keeps_both_provenances():
    mod = engine_module()

    class _Other:
        source = CandidateSource.SIMILAR_ITEM
        score_kind = "item_item_similarity"

        def propose(self, *, arguments, limit):
            return [("A1", 1, 0.9), ("A9", 2, 0.5)]

    tool = TigerSourceTool(mod.FakeTigerEngine(recommendations=(rec("A1", 1, -2.0),)))
    plane, _ = make_plane(tool, identities=("A1", "A2", "A9"),
                          extra_tools=(("similar_item_tool", _Other()),))
    plane.execute(select(CandidateSource.TIGER, 3), read_trusted_history=lambda: ("A2",))
    plane.execute(select(CandidateSource.SIMILAR_ITEM, 3), read_trusted_history=lambda: ("A2",))

    ledger = plane.ledger
    assert len(ledger.grounded_entries) == 2  # A1 deduplicated, A9 distinct
    entry = ledger.get("A1")
    assert entry is not None
    assert entry.is_multi_source
    assert {p.source for p in entry.provenance} == {CandidateSource.TIGER,
                                                    CandidateSource.SIMILAR_ITEM}
    tiger = entry.provenance_for(CandidateSource.TIGER)[0]
    other = entry.provenance_for(CandidateSource.SIMILAR_ITEM)[0]
    assert tiger.source_score == -2.0 and tiger.score_kind == TIGER_SCORE_KIND
    assert other.source_score == 0.9 and other.score_kind == "item_item_similarity"


def test_repeated_calls_produce_identical_ledgers_and_ranks():
    mod = engine_module()
    recs = (rec("A1", 1), rec("A2", 2), rec("A3", 3))
    snapshots = []
    for _ in range(3):
        tool = TigerSourceTool(mod.FakeTigerEngine(recommendations=recs))
        plane, _ = make_plane(tool, identities=("A1", "A2", "A3"))
        plane.execute(select(CandidateSource.TIGER, 5), read_trusted_history=lambda: ("A3",))
        snapshots.append((plane.ledger.ranked_parent_asins(),
                          tuple((e.parent_asin, e.provenance[0].source_rank,
                                 e.provenance[0].source_score) for e in plane.ledger.entries)))
    assert snapshots[0] == snapshots[1] == snapshots[2]


# --------------------------------------------------------------------------- #
# Q5 -- failure isolation
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("mode", ["backend_unavailable", "checkpoint_unavailable",
                                  "sid_artifact_unavailable", "unresolved_identity", "timeout"])
def test_unavailable_engine_fails_the_source_without_touching_the_ledger(mode):
    mod = engine_module()
    tool = TigerSourceTool(mod.FakeTigerEngine(recommendations=(rec("A1", 1),), fail=mode))
    plane, _ = make_plane(tool, identities=("A1", "A2"))
    before = len(plane.ledger)
    with pytest.raises(CandidateSourceUnavailable):
        plane.execute(select(CandidateSource.TIGER, 5), read_trusted_history=lambda: ("A2",))
    assert len(plane.ledger) == before
    assert tool.last_outcome["state"] == "UNAVAILABLE"


@pytest.mark.parametrize("mode", ["invalid_sid", "non_finite_score"])
def test_malformed_proposals_are_refused_not_repaired(mode):
    mod = engine_module()
    tool = TigerSourceTool(mod.FakeTigerEngine(recommendations=(), fail=mode))
    plane, _ = make_plane(tool, identities=("A1", "A2"))
    with pytest.raises(CandidateSourceUnavailable):
        plane.execute(select(CandidateSource.TIGER, 5), read_trusted_history=lambda: ("A2",))
    assert len(plane.ledger) == 0


def test_a_genuine_empty_result_is_explicit_and_not_a_failure():
    mod = engine_module()
    tool = TigerSourceTool(mod.FakeTigerEngine(recommendations=(), fail="empty_result"))
    plane, _ = make_plane(tool, identities=("A1", "A2"))
    result = plane.execute(select(CandidateSource.TIGER, 5),
                           read_trusted_history=lambda: ("A2",))
    assert len(plane.ledger) == 0
    assert tool.last_outcome["state"] == "EMPTY"
    assert tool.last_outcome["reason"] == "retrieval_returned_no_candidates"
    assert result is not None


def test_no_trusted_history_is_unavailable_not_an_empty_answer():
    mod = engine_module()
    tool = TigerSourceTool(mod.FakeTigerEngine(recommendations=(rec("A1", 1),)))
    plane, _ = make_plane(tool, identities=("A1",))
    with pytest.raises(CandidateSourceUnavailable):
        plane.execute(select(CandidateSource.TIGER, 5), read_trusted_history=lambda: ())
    assert tool.last_outcome["reason"] == "no_trusted_history"


def test_failure_of_tiger_leaves_other_sources_usable():
    mod = engine_module()

    class _Other:
        source = CandidateSource.SIMILAR_ITEM
        score_kind = "item_item_similarity"

        def propose(self, *, arguments, limit):
            return [("A1", 1, 0.5)]

    broken = TigerSourceTool(mod.FakeTigerEngine(recommendations=(rec("A1", 1),),
                                                 fail="backend_unavailable"))
    plane, _ = make_plane(broken, identities=("A1", "A2"),
                          extra_tools=(("similar_item_tool", _Other()),))
    with pytest.raises(CandidateSourceUnavailable):
        plane.execute(select(CandidateSource.TIGER, 5), read_trusted_history=lambda: ("A2",))
    plane.execute(select(CandidateSource.SIMILAR_ITEM, 5), read_trusted_history=lambda: ("A2",))
    assert plane.ledger.get("A1") is not None
    assert len(plane.ledger.grounded_entries) == 1


# --------------------------------------------------------------------------- #
# Q6 -- retrieval-status preservation
# --------------------------------------------------------------------------- #
def test_certified_status_is_accepted_and_recorded():
    mod = engine_module()
    tool = TigerSourceTool(mod.FakeTigerEngine(recommendations=(rec("A1", 1),),
                                               mode=mod.CERTIFIED))
    plane, _ = make_plane(tool, identities=("A1",))
    plane.execute(select(CandidateSource.TIGER, 3), read_trusted_history=lambda: ("A1",))
    assert tool.last_outcome["retrieval_mode"] == "CERTIFIED"
    assert tool.last_proposals[0].retrieval_mode == "CERTIFIED"
    assert "mode=CERTIFIED" in plane.ledger.entries[0].provenance[0].source_query


def test_approximate_is_refused_under_the_canonical_posture():
    mod = engine_module()
    tool = TigerSourceTool(mod.FakeTigerEngine(recommendations=(rec("A1", 1),),
                                               mode=mod.APPROXIMATE))
    assert tool.require_certified is True
    plane, _ = make_plane(tool, identities=("A1",))
    with pytest.raises(CandidateSourceUnavailable):
        plane.execute(select(CandidateSource.TIGER, 3), read_trusted_history=lambda: ("A1",))
    assert tool.last_outcome["reason"] == "not_certified"
    assert len(plane.ledger) == 0


def test_approximate_is_never_upgraded_when_explicitly_allowed():
    mod = engine_module()
    tool = TigerSourceTool(mod.FakeTigerEngine(recommendations=(rec("A1", 1),),
                                               mode=mod.APPROXIMATE),
                           require_certified=False)
    plane, _ = make_plane(tool, identities=("A1",))
    plane.execute(select(CandidateSource.TIGER, 3), read_trusted_history=lambda: ("A1",))
    assert tool.last_proposals[0].retrieval_mode == "APPROXIMATE"
    assert "mode=APPROXIMATE" in plane.ledger.entries[0].provenance[0].source_query
    assert "mode=CERTIFIED" not in plane.ledger.entries[0].provenance[0].source_query


def test_budget_exhaustion_downgrade_is_refused_not_relabelled():
    mod = engine_module()
    tool = TigerSourceTool(mod.FakeTigerEngine(recommendations=(rec("A1", 1),),
                                               fail="budget_exhausted"))
    plane, _ = make_plane(tool, identities=("A1",))
    with pytest.raises(CandidateSourceUnavailable):
        plane.execute(select(CandidateSource.TIGER, 3), read_trusted_history=lambda: ("A1",))
    assert len(plane.ledger) == 0


def test_an_engine_reporting_an_unknown_mode_is_refused():
    mod = engine_module()
    tool = TigerSourceTool(mod.FakeTigerEngine(recommendations=(rec("A1", 1),), mode="SORT_OF"))
    plane, _ = make_plane(tool, identities=("A1",))
    with pytest.raises(CandidateSourceUnavailable):
        plane.execute(select(CandidateSource.TIGER, 3), read_trusted_history=lambda: ("A1",))


# --------------------------------------------------------------------------- #
# Q2 (real evidence) -- the frozen certified artifact through the real plane
# --------------------------------------------------------------------------- #
@needs_evidence
def test_real_certified_evidence_grounds_cleanly_through_the_real_plane():
    mod = engine_module()
    base = mod.CertifiedEvidenceTigerEngine("/root/autodl-tmp/step26_h7_canonical",
                                            mappings_path=str(MAPPINGS))
    checked = 0
    for case_index in range(5):
        engine = base.for_case(case_index)
        record = base._cache[case_index]
        # The certified artifact deliberately carries NO parent_asin (boundary rule 19.2(1)),
        # so the trusted mapping is what re-attaches identity here - exactly as at runtime.
        identities = tuple(str(base._id2item[int(c["item_id"])])
                           for c in record["candidates"][:20])
        tool = TigerSourceTool(engine)
        mapping = _Map(identities)
        plane = CandidatePlane(ledger=CandidateLedger(),
                               grounding=GroundingVerifier(mapping, _Catalog(identities)),
                               tiger_tool=tool)
        plane.execute(select(CandidateSource.TIGER, 20),
                      read_trusted_history=lambda: identities[:1], step_index=0)
        assert len(plane.ledger.ungrounded_entries) == 0, case_index
        assert len(plane.ledger.grounded_entries) == min(20, len(record["candidates"]))
        for entry in plane.ledger.entries:
            provenance = entry.provenance_for(CandidateSource.TIGER)[0]
            assert provenance.score_kind == TIGER_SCORE_KIND
            assert provenance.source_query.startswith("mode=CERTIFIED")
            assert provenance.item_id is not None
            assert mapping.item_id_to_parent_asin(provenance.item_id) == entry.parent_asin
        assert tool.last_proposals[0].retrieval_mode == "CERTIFIED"
        checked += 1
    assert checked == 5

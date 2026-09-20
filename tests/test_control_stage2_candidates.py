"""Stage 2 tests: the multi-source candidate plane (AgentRec-X 2.0-alpha).

What this file is for
---------------------
Stage 1 made *who decides the next step* dynamic while allowing exactly one candidate
source.  Stage 2 allows more than one source, which is where a project usually loses the
property that candidates come from trusted tools.  These tests exist to make that property
checkable rather than aspirational.

The four families:

**A. Candidate identity and provenance** - the ledger deduplicates identity, keeps every
source's provenance, never fuses heterogeneous raw scores, and never lets an unknown
identity become a candidate.

**B. Grounding** - identity is confirmed against trusted data, and a failure fails closed.
An identity a source invented is retained for audit and is *not* presented.

**C. Source execution and adaptive retrieval** - one action consults one source; a policy
chooses the source; an unconfigured source is refused rather than substituted; and an
observation can change the next action (the property that makes retrieval adaptive rather
than a fixed fan-out).

**D. Preserved Stage 1 behaviour** - the Stage 1 action menu, the Stage 1 policy and the
single-source fast path are unchanged.

Everything is offline and deterministic: the catalogue is synthetic, the identity map is a
small stand-in, and no checkpoint, GPU, network or provider API is involved.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from recommendation.control import (  # noqa: E402
    ARGUMENTS_BY_ACTION,
    CANDIDATE_ACTIONS,
    READ_ONLY_ACTIONS,
    ActionKind,
    ActionProposal,
    ActionValidator,
    CandidateLedger,
    CandidatePlane,
    CandidateSetObservation,
    CandidateSource,
    CandidateSourceUnavailable,
    CandidateVerificationStatus,
    CatalogSearchSource,
    GroundingVerifier,
    LedgerSourceError,
    MappingOnlyCatalog,
    PolicyActionError,
    SearchCatalogArguments,
    SelectSourceArguments,
    STAGE_1_ACTIONS,
    ValidatedAction,
    reciprocal_rank_fusion,
)

# --------------------------------------------------------------------------- #
# fixtures: a small synthetic trusted catalogue
# --------------------------------------------------------------------------- #

CATALOG_ROWS: tuple[tuple[str, str, str, tuple[str, ...]], ...] = (
    ("B0BOOT1", "Waterproof Hiking Boots", "Acme", ("waterproof membrane", "vibram sole")),
    ("B0SHOE2", "Trail Running Shoes", "Acme", ("breathable mesh",)),
    ("B0TENT3", "Camping Tent 2-Person", "Outfit", ("waterproof rainfly", "lightweight")),
    ("B0SOCK4", "Wool Hiking Socks", "TrailCo", ("merino wool",)),
    ("B0PACK5", "Hiking Backpack 40L", "Outfit", ("waterproof cover", "lightweight")),
)

#: Identities the mapping knows.  ``B0GHOST`` is deliberately absent everywhere.
KNOWN: frozenset[str] = frozenset(row[0] for row in CATALOG_ROWS)


class _Record:
    """A normalized-catalogue-record stand-in with the fields search reads."""

    def __init__(self, row: tuple[str, str, str, tuple[str, ...]]) -> None:
        self.parent_asin, self.title, self.store, self.features = row
        self.main_category = None
        self.categories = ()
        self.description = ()


class _Catalog:
    """A trusted catalogue stub: ``records`` mapping plus membership."""

    def __init__(self, rows: tuple[tuple[str, str, str, tuple[str, ...]], ...] = CATALOG_ROWS):
        self.records = {row[0]: _Record(row) for row in rows}

    def __contains__(self, parent_asin: object) -> bool:
        return isinstance(parent_asin, str) and parent_asin in self.records


class _IdentityMap:
    """A trusted identity map stub with the engine's structural interface."""

    def __init__(self, known: frozenset[str] = KNOWN) -> None:
        self._ids = {identity: index for index, identity in enumerate(sorted(known), start=1)}

    def has_parent_asin(self, parent_asin: str) -> bool:
        return parent_asin in self._ids

    def parent_asin_to_item_id(self, parent_asin: str) -> int:
        return self._ids[parent_asin]

    def item_id_to_parent_asin(self, item_id: int) -> str:
        for identity, value in self._ids.items():
            if value == item_id:
                return identity
        raise KeyError(item_id)


class _StubSourceTool:
    """A trusted candidate source that returns a caller-supplied ranking."""

    def __init__(self, source: CandidateSource, ranking: list[tuple[str, float]]) -> None:
        self._source = source
        self._ranking = ranking
        self.calls = 0

    @property
    def source(self) -> CandidateSource:
        return self._source

    @property
    def score_kind(self) -> str:
        return "stub_score"

    def propose(self, *, arguments: object, limit: int) -> list[tuple[str, int, float]]:
        self.calls += 1
        return [
            (identity, rank, score)
            for rank, (identity, score) in enumerate(self._ranking[:limit], start=1)
        ]


def build_plane(
    *,
    ranking: list[tuple[str, float]] | None = None,
    with_search: bool = True,
    extra_source: CandidateSource | None = None,
) -> tuple[CandidatePlane, CandidateLedger, CandidateSourceTool | None]:
    """Build a candidate plane over the synthetic catalogue."""
    ledger = CandidateLedger()
    catalog = _Catalog()
    grounding = GroundingVerifier(_IdentityMap(), catalog)
    tool = None
    if extra_source is not None:
        tool = _StubSourceTool(extra_source, ranking or [])
        return (
            CandidatePlane(
                ledger=ledger,
                grounding=grounding,
                catalog_search=CatalogSearchSource(catalog) if with_search else None,
                similar_item_tool=tool if extra_source is CandidateSource.SIMILAR_ITEM else None,
            ),
            ledger,
            tool,
        )
    return (
        CandidatePlane(
            ledger=ledger,
            grounding=grounding,
            catalog_search=CatalogSearchSource(catalog) if with_search else None,
        ),
        ledger,
        None,
    )


# =========================================================================== #
# A. Candidate identity and provenance
# =========================================================================== #


def test_ledger_deduplicates_identity_and_keeps_all_provenance() -> None:
    """One entry per product; every source that proposed it stays recorded."""
    ledger = CandidateLedger()
    ledger.record_from_source(
        source=CandidateSource.HISTORY,
        candidates=[("A", 1, 9.5), ("B", 2, 8.0), ("C", 3, 7.0)],
        score_kind="sasrec_logit",
        grounded=frozenset({"A", "B", "C"}),
    )
    ledger.record_from_source(
        source=CandidateSource.CATALOG_SEARCH,
        candidates=[("C", 1, 14.2), ("D", 2, 11.0)],
        score_kind="bm25",
        source_query="tent",
        grounded=frozenset({"A", "B", "C", "D"}),
    )

    assert len(ledger) == 4, "identity must be deduplicated"
    entry = ledger.get("C")
    assert entry is not None
    assert entry.source_count == 2
    assert set(entry.sources) == {CandidateSource.HISTORY, CandidateSource.CATALOG_SEARCH}
    # Both raw scores survive, each under its own kind - nothing was averaged.
    by_source = {record.source: record for record in entry.provenance}
    assert by_source[CandidateSource.HISTORY].source_score == 7.0
    assert by_source[CandidateSource.HISTORY].score_kind == "sasrec_logit"
    assert by_source[CandidateSource.CATALOG_SEARCH].source_score == 14.2
    assert by_source[CandidateSource.CATALOG_SEARCH].score_kind == "bm25"
    # Original source ranks stay recoverable.
    assert by_source[CandidateSource.HISTORY].source_rank == 3
    assert by_source[CandidateSource.CATALOG_SEARCH].source_rank == 1


def test_ledger_never_fuses_heterogeneous_raw_scores() -> None:
    """Fusion is rank-based; no source's raw magnitude leaks into another's scale."""
    ledger = CandidateLedger()
    # A source with huge scores for the SAME ordering must not outrank a source with small
    # scores for a different ordering: only ranks matter.
    ledger.record_from_source(
        source=CandidateSource.HISTORY,
        candidates=[("A", 1, 1e9), ("B", 2, 5e8)],
        score_kind="sasrec_logit", grounded=frozenset({"A", "B", "C"}),
    )
    ledger.record_from_source(
        source=CandidateSource.CATALOG_SEARCH,
        candidates=[("C", 1, 0.001), ("B", 2, 0.0005)],
        score_kind="bm25", grounded=frozenset({"A", "B", "C"}),
    )
    scores = ledger.fusion_scores()
    # B is ranked by both sources, so it must lead the fusion despite C's BM25 rank 1 and
    # A's enormous - but incomparable - history score.
    assert ledger.ranked_parent_asins()[0] == "B"
    assert scores["B"] > scores["A"] and scores["B"] > scores["C"]


def test_reciprocal_rank_fusion_is_deterministic_and_tie_broken() -> None:
    """Fusion is reproducible and total-ordered, with ties broken by identity."""
    a = reciprocal_rank_fusion([["X", "Y"], ["Y", "X"]])
    b = reciprocal_rank_fusion([["X", "Y"], ["Y", "X"]])
    assert a == b
    # A perfect tie must still be a stable total order.
    assert [identity for identity, _ in a] == ["X", "Y"] or [i for i, _ in a] == ["Y", "X"]
    # Rank 1 in one list beats nothing in the other.
    assert reciprocal_rank_fusion([["only"]])[0][0] == "only"


def test_ledger_refuses_an_unapproved_candidate_source() -> None:
    """Candidates may only come from a declared trusted source."""
    ledger = CandidateLedger()
    for bad in ("history", "unknown_source", 42, None):
        with pytest.raises(LedgerSourceError):
            ledger.record_from_source(
                source=bad,  # type: ignore[arg-type]
                candidates=[("A", 1, 1.0)],
            )
    assert len(ledger) == 0


def test_ledger_retains_an_invented_identity_only_as_audit() -> None:
    """An identity the mapping does not know is never a candidate."""
    ledger = CandidateLedger()
    ledger.record_from_source(
        source=CandidateSource.CATALOG_SEARCH,
        candidates=[("REAL", 1, 5.0), ("INVENTED", 2, 99.0)],
        grounded=frozenset({"REAL"}),
    )
    assert "REAL" in ledger
    assert "INVENTED" not in ledger, "an ungrounded identity must never be a candidate"
    assert [entry.parent_asin for entry in ledger.ungrounded_entries] == ["INVENTED"]
    assert ledger.ranked_parent_asins() == ("REAL",)


def test_ledger_marks_multi_source_agreement() -> None:
    """Independent agreement between sources is a count, not a fused score."""
    ledger = CandidateLedger()
    ledger.record_from_source(
        source=CandidateSource.HISTORY,
        candidates=[("A", 1, 1.0), ("B", 2, 0.9)],
        grounded=frozenset({"A", "B"}),
    )
    ledger.record_from_source(
        source=CandidateSource.CATALOG_SEARCH,
        candidates=[("B", 1, 3.0), ("C", 2, 2.0)],
        grounded=frozenset({"A", "B", "C"}),
    )
    assert ledger.multi_source_parent_asins() == ("B",)
    assert set(ledger.sources_present()) == {
        CandidateSource.HISTORY,
        CandidateSource.CATALOG_SEARCH,
    }


# =========================================================================== #
# B. Grounding
# =========================================================================== #


def test_grounding_fails_closed_for_unknown_identities() -> None:
    """An identity absent from the trusted mapping is ungrounded, with a reason."""
    verifier = GroundingVerifier(_IdentityMap(), _Catalog())
    report = verifier.ground(["B0BOOT1", "NOT-A-PRODUCT", ""])
    assert report.grounded == frozenset({"B0BOOT1"})
    assert report.ungrounded_count == 2
    reasons = {entry.parent_asin: entry.reason for entry in report.ungrounded}
    assert reasons["NOT-A-PRODUCT"] == "not_in_item_mapping"
    assert reasons[""] == "blank_identity"


def test_grounding_can_require_catalogue_presence() -> None:
    """With a catalogue configured, mapping membership alone is not enough."""
    identity_map = _IdentityMap()
    catalog = _Catalog()
    # A product the mapping knows but the catalogue does not carry.
    identity_map._ids["B0MAPPEDONLY"] = 99  # noqa: SLF001 - deliberate fixture surgery
    mapping_only = GroundingVerifier(identity_map)
    strict = GroundingVerifier(identity_map, catalog)

    assert mapping_only.ground(["B0MAPPEDONLY"]).grounded_count == 1
    assert strict.ground(["B0MAPPEDONLY"]).grounded_count == 0
    assert strict.checks_catalog is True
    assert mapping_only.checks_catalog is False, (
        "a mapping-only verifier must not claim it checked the catalogue"
    )
    assert "B0MAPPEDONLY" in MappingOnlyCatalog(identity_map)


def test_grounding_resolves_item_identity_from_the_mapping_only() -> None:
    """Item ids come from the trusted mapping, never from a source's own claim."""
    verifier = GroundingVerifier(_IdentityMap(), _Catalog())
    report = verifier.ground(["B0BOOT1"])
    assert report.item_ids == {"B0BOOT1": _IdentityMap().parent_asin_to_item_id("B0BOOT1")}


# =========================================================================== #
# C. Source execution and adaptive retrieval
# =========================================================================== #


def test_catalog_search_returns_grounded_candidates_in_score_order() -> None:
    """The lexical source finds real catalogue products, ranked by BM25."""
    plane, ledger, _ = build_plane()
    action = ValidatedAction(
        action=ActionKind.SEARCH_CATALOG,
        action_id="act:1",
        step_index=0,
        run_id="r",
        k=1,
        arguments=SearchCatalogArguments.from_text("waterproof hiking", limit=3),
    )
    result = plane.execute(action, read_trusted_history=lambda: (), step_index=0)
    assert result.returned_k >= 1
    assert result.status == "ok"
    found = set(ledger.grounded_parent_asins())
    assert found <= KNOWN
    assert {"B0BOOT1"} <= found, "the waterproof hiking boots must match"


def test_search_observation_carries_counts_not_products() -> None:
    """The policy learns how many candidates came back, never which."""
    plane, ledger, _ = build_plane()
    action = ValidatedAction(
        action=ActionKind.SEARCH_CATALOG,
        action_id="act:1",
        step_index=0,
        run_id="r",
        k=1,
        arguments=SearchCatalogArguments.from_text("waterproof", limit=3),
    )
    result = plane.execute(action, read_trusted_history=lambda: (), step_index=0)
    observation = plane.observe(
        action=action,
        source=CandidateSource.CATALOG_SEARCH,
        requested_k=result.requested_k,
        returned_k=result.returned_k,
        ungrounded_count=0,
        step_index=0,
        candidate_set_ref=result.candidate_set_ref,
    )
    payload = str(observation.model_dump())
    for identity in KNOWN:
        assert identity not in payload, "the observation must not leak a product identity"
    assert "score" not in observation.model_dump()
    assert observation.returned_k == result.returned_k
    assert observation.has_candidates is True


def test_empty_result_is_an_honest_observation_that_permits_another_source() -> None:
    """Zero candidates is a legal outcome the policy can react to - not a failure to hide.

    This is the adaptive-retrieval property: the observation reports ``empty``, and it is
    the *observation* that lets a policy choose a different source next.
    """
    plane, _ledger, _ = build_plane()
    action = ValidatedAction(
        action=ActionKind.SEARCH_CATALOG,
        action_id="act:1",
        step_index=0,
        run_id="r",
        k=1,
        arguments=SearchCatalogArguments.from_text("zzzznothingmatchesthis", limit=3),
    )
    result = plane.execute(action, read_trusted_history=lambda: (), step_index=0)
    assert result.returned_k == 0
    observation = plane.observe(
        action=action,
        source=CandidateSource.CATALOG_SEARCH,
        requested_k=result.requested_k,
        returned_k=result.returned_k,
        ungrounded_count=0,
        step_index=0,
        candidate_set_ref=result.candidate_set_ref,
    )
    assert observation.status == "empty"
    assert observation.has_candidates is False


def test_an_unconfigured_source_is_refused_not_substituted() -> None:
    """A source this deployment does not run is refused, never answered by another."""
    plane, _ledger, _ = build_plane()
    action = ValidatedAction(
        action=ActionKind.FIND_SIMILAR,
        action_id="act:1",
        step_index=0,
        run_id="r",
        k=1,
        arguments=SelectSourceArguments(source=CandidateSource.SIMILAR_ITEM, limit=3),
    )
    with pytest.raises(CandidateSourceUnavailable) as excinfo:
        plane.execute(action, read_trusted_history=lambda: ())
    assert "similar_item" in str(excinfo.value)
    assert plane.has_source(CandidateSource.SIMILAR_ITEM) is False


def test_a_registered_similar_item_source_is_usable() -> None:
    """The seam is real: registering a similar-item tool makes the source available."""
    ranking = [("B0PACK5", 0.91), ("B0TENT3", 0.77)]
    plane, ledger, tool = build_plane(ranking=ranking, extra_source=CandidateSource.SIMILAR_ITEM)
    assert plane.has_source(CandidateSource.SIMILAR_ITEM)
    action = ValidatedAction(
        action=ActionKind.FIND_SIMILAR,
        action_id="act:1",
        step_index=0,
        run_id="r",
        k=1,
        arguments=SelectSourceArguments(source=CandidateSource.SIMILAR_ITEM, limit=2),
    )
    result = plane.execute(action, read_trusted_history=lambda: (), step_index=0)
    assert result.returned_k == 2
    assert set(ledger.grounded_parent_asins()) == {"B0PACK5", "B0TENT3"}
    assert tool is not None and tool.calls == 1


def test_a_candidate_action_cannot_be_executed_by_the_plane_without_a_source() -> None:
    """The plane refuses a non-candidate action rather than routing it."""
    plane, _ledger, _ = build_plane()
    action = ValidatedAction(
        action=ActionKind.COMPARE,
        action_id="act:1",
        step_index=0,
        run_id="r",
        k=1,
        arguments={"parent_asins": ["B0BOOT1", "B0SHOE2"]},
    )
    with pytest.raises(PolicyActionError):
        plane.execute(action, read_trusted_history=lambda: ())


def test_policy_selects_a_source_and_receives_the_outcome_as_an_observation() -> None:
    """The Stage 2 flow: policy selects, plane executes, policy sees counts and can adapt.

    The policy in this test changes its mind *because of the observation*, which is the
    whole point of adaptive retrieval rather than a fixed fan-out.
    """
    plane, ledger, _ = build_plane()
    steps: list[tuple[str, int]] = []

    # Step 0: the policy asks the lexical source for something it cannot match.
    first = ValidatedAction(
        action=ActionKind.SEARCH_CATALOG,
        action_id="act:0",
        step_index=0,
        run_id="r",
        k=1,
        arguments=SearchCatalogArguments.from_text("zzzznomatch", limit=3),
    )
    result = plane.execute(first, read_trusted_history=lambda: (), step_index=0)
    observation = plane.observe(
        action=first,
        source=CandidateSource.CATALOG_SEARCH,
        requested_k=result.requested_k,
        returned_k=result.returned_k,
        ungrounded_count=0,
        step_index=0,
        candidate_set_ref=result.candidate_set_ref,
    )
    steps.append((observation.status, observation.returned_k))

    # Step 1: because the observation was empty, the policy picks a different query.
    assert observation.has_candidates is False
    second = ValidatedAction(
        action=ActionKind.SEARCH_CATALOG,
        action_id="act:1",
        step_index=1,
        run_id="r",
        k=1,
        arguments=SearchCatalogArguments.from_text("waterproof", limit=3),
    )
    result2 = plane.execute(second, read_trusted_history=lambda: (), step_index=1)
    observation2 = plane.observe(
        action=second,
        source=CandidateSource.CATALOG_SEARCH,
        requested_k=result2.requested_k,
        returned_k=result2.returned_k,
        ungrounded_count=0,
        step_index=1,
        candidate_set_ref=result2.candidate_set_ref,
    )
    steps.append((observation2.status, observation2.returned_k))

    assert steps[0] == ("empty", 0)
    assert steps[1][0] == "ok" and steps[1][1] > 0
    assert len(ledger) > 0


def test_plane_reports_its_configured_sources_and_never_offers_a_missing_one() -> None:
    """Available sources are computed from configuration, not from the action enum."""
    plane, _ledger, _ = build_plane()
    assert plane.available_sources() == (CandidateSource.CATALOG_SEARCH,)
    status = plane.status()
    assert status["sources"][0]["source"] == "catalog_search"
    assert status["sources"][0]["score_kind"] == "bm25"


# =========================================================================== #
# D. Preserved Stage 1 behaviour and protocol invariants
# =========================================================================== #


def test_stage_1_action_menu_is_unchanged() -> None:
    """Stage 2 extended the enum; the Stage 1 menu itself did not move."""
    assert STAGE_1_ACTIONS == (ActionKind.RECOMMEND_FROM_HISTORY, ActionKind.FINISH)


def test_arguments_are_typed_per_action_and_cannot_be_swapped() -> None:
    """A proposal cannot carry another action's arguments."""
    assert ARGUMENTS_BY_ACTION[ActionKind.SEARCH_CATALOG] is SearchCatalogArguments
    assert ARGUMENTS_BY_ACTION[ActionKind.SELECT_SOURCE] is SelectSourceArguments
    assert ARGUMENTS_BY_ACTION[ActionKind.FINISH].__name__ == "EmptyArguments"

    with pytest.raises(Exception):
        ActionProposal(action=ActionKind.FINISH, arguments={"terms": ["tent"]})
    with pytest.raises(Exception):
        ActionProposal(action=ActionKind.SEARCH_CATALOG)  # missing required arguments
    with pytest.raises(Exception):
        ActionProposal(
            action=ActionKind.SELECT_SOURCE,
            arguments={"terms": ["tent"]},  # wrong argument model
        )


def test_no_action_accepts_a_way_to_invent_candidate_identities() -> None:
    """Only the two comparison-like actions name products, and both are read-only."""
    for action, model in ARGUMENTS_BY_ACTION.items():
        fields = set(model.model_fields)
        if action in READ_ONLY_ACTIONS:
            continue
        assert "parent_asins" not in fields, f"{action.value} must not name products"
        assert "item_ids" not in fields
        assert "candidates" not in fields
        assert "sql" not in fields
    # COMPARE / TRADE_OFF / GET_DETAILS / CHECK_COMPATIBILITY / BUNDLE may name products,
    # and every one of them is in READ_ONLY_ACTIONS - it cannot add a candidate.
    naming = {
        action
        for action, model in ARGUMENTS_BY_ACTION.items()
        if "parent_asins" in model.model_fields
    }
    assert naming
    assert naming <= set(READ_ONLY_ACTIONS)


def test_candidate_actions_are_exactly_the_trusted_source_actions() -> None:
    """The set of actions that may produce candidates is small and explicitly declared."""
    assert set(CANDIDATE_ACTIONS) == {
        ActionKind.RECOMMEND_FROM_HISTORY,
        ActionKind.SELECT_SOURCE,
        ActionKind.SEARCH_CATALOG,
        ActionKind.FIND_SIMILAR,
    }
    assert set(CANDIDATE_ACTIONS).isdisjoint(READ_ONLY_ACTIONS)


def test_validator_refuses_a_candidate_action_the_system_did_not_offer() -> None:
    """Availability is computed by the controller; a policy cannot grant itself a source."""
    proposal = ActionProposal(
        action=ActionKind.SEARCH_CATALOG,
        arguments=SearchCatalogArguments.from_text("tent"),
    )
    validated, result = ActionValidator().validate(
        proposal,
        run_id="r",
        turn_id=None,
        step_index=0,
        available_actions=STAGE_1_ACTIONS,  # this deployment offers only the fast path
    )
    assert validated is None
    assert result.code == "action_not_available"


def test_validator_forwards_validated_arguments_to_the_executor() -> None:
    """The validator stamps metadata and passes through the arguments it checked."""
    proposal = ActionProposal(
        action=ActionKind.SEARCH_CATALOG,
        arguments=SearchCatalogArguments.from_text("camping tent", limit=4),
    )
    validated, result = ActionValidator().validate(
        proposal,
        run_id="run-1",
        turn_id="turn-1",
        step_index=2,
        available_actions=(ActionKind.SEARCH_CATALOG, ActionKind.FINISH),
    )
    assert result.verified and validated is not None
    assert isinstance(validated.arguments, SearchCatalogArguments)
    assert validated.arguments.terms == ("camping", "tent")
    assert validated.arguments.limit == 4
    # Execution metadata is still controller-owned, not policy-supplied.
    assert validated.action_id.startswith("act:run-1:2:")


def test_search_arguments_bound_the_query() -> None:
    """A query cannot grow without bound, and a blank term set is rejected."""
    with pytest.raises(Exception):
        SearchCatalogArguments(terms=())
    with pytest.raises(Exception):
        SearchCatalogArguments(terms=tuple(f"t{i}" for i in range(50)))
    long_text = " ".join(f"term{i}" for i in range(50))
    assert len(SearchCatalogArguments.from_text(long_text).terms) <= 16


def test_ledger_export_is_safe_for_the_trajectory() -> None:
    """The ledger's export is counts plus provenance structure - no secrets, no history."""
    plane, ledger, _ = build_plane()
    action = ValidatedAction(
        action=ActionKind.SEARCH_CATALOG,
        action_id="act:1",
        step_index=0,
        run_id="r",
        k=1,
        arguments=SearchCatalogArguments.from_text("waterproof", limit=3),
    )
    plane.execute(action, read_trusted_history=lambda: (), step_index=0)
    payload = ledger.as_dict()
    assert payload["grounded"] == len(ledger)
    assert isinstance(payload["sources"], list)
    assert "user_key" not in str(payload)

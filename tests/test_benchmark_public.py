"""Phase-5 public-benchmark tests.

What these tests protect, and why each one exists
-------------------------------------------------
The Phase-5 benchmark publishes numbers that are compared against each other, so a bug in an *arm*
is more dangerous than a crash: it produces a plausible, wrong table.  Two bugs of exactly that kind
were found while building this phase, and both now have a named regression test:

* ``ranking_batches`` scored ``scores[:, 1:] = base``, which scores ascending *item id* and ignores
  its ``ranking`` argument completely.  The popularity arm therefore measured "prefer the highest
  item id" while reporting itself as a popularity baseline, with a mean target rank *below chance*.
* ``arm_metadata_retrieval`` materialized a complete 156 746-item permutation per user, which at
  cohort scale is ~30 GB of CPython list overhead and was OOM-killed.

The remaining tests pin the properties the phase's honesty claims rest on: that the retrieval index
agrees with the accepted catalogue search, that fusion is fuse-not-concatenate, that the agent arm
really varies its source per user, and that leakage checks decide on timestamps rather than on the
naive "target not in history" rule that a legal repurchase would fail.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments import benchmark_public as B  # noqa: E402
from recommendation.catalog.metadata import MetadataIndex  # noqa: E402
from recommendation.catalog.schemas import ProductMetadata  # noqa: E402
from recommendation.control.candidate_ledger import reciprocal_rank_fusion  # noqa: E402
from recommendation.control.catalog_search import CatalogSearchSource  # noqa: E402
from recommendation.evaluation.split import EvaluationCase  # noqa: E402


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #


def _catalogue(size: int = 200) -> MetadataIndex:
    """A synthetic catalogue large enough that the query-term cap is meaningful.

    ``ItemTextIndex.MAX_QUERY_DOCUMENT_FRACTION`` is 5% of the catalogue, so at least ~100 records
    are needed before a term can fall below the cap.  Each record carries one unique term
    (``unique000``...) plus a term shared by everything (``universal``), which is what the cap must
    drop.
    """
    records = []
    for position in range(size):
        records.append(
            ProductMetadata(
                parent_asin=f"asin{position:04d}",
                title=f"Trail Widget unique{position:04d}",
                store="Acme Outdoors",
                main_category="Sports & Outdoors",
                categories=("Sports & Outdoors", "Outdoor Recreation"),
                source="phase5-test",
            )
        )
    return MetadataIndex.from_records(records)


@pytest.fixture()
def catalogue() -> MetadataIndex:
    return _catalogue()


@pytest.fixture()
def text_index(catalogue: MetadataIndex) -> B.ItemTextIndex:
    return B.ItemTextIndex(catalogue.records)


def _case(
    user_id: str = "u1",
    user_int_id: int = 1,
    history: tuple[int, ...] = (5, 6, 7, 8, 9),
    validation: int = 10,
    target: int = 11,
) -> EvaluationCase:
    return EvaluationCase(
        user_id=user_id,
        user_int_id=user_int_id,
        train_history=history[:-1] if history else (),
        validation_target=validation,
        test_target=target,
        sequence_length=len(history) + 2,
    )


# --------------------------------------------------------------------------- #
# The two regressions
# --------------------------------------------------------------------------- #


def test_ranking_batches_scatters_ranking_into_item_positions():
    """REGRESSION: the best item must get the highest score, not the lowest item id.

    ``scores[:, 1:] = base`` passed every shape check and produced a full, finite score matrix, so
    only a test that inspects *which item wins* can catch it.
    """
    import torch

    # Deliberately not ascending: item 3 is best, item 1 is worst.
    ranking = [3, 5, 2, 4, 1]
    cases = [_case()]
    (histories, targets, scores), = list(B.ranking_batches(ranking, cases, batch_size=1))

    assert scores.shape == (1, 6)
    # Best item has the largest score...
    assert int(scores[0].argmax()) == 3
    assert float(scores[0, 3]) > float(scores[0, 1])
    # ...and the ordering is strict and consistent down the ranking.
    values = [float(scores[0, item]) for item in ranking]
    assert values == sorted(values, reverse=True)
    # PAD never receives a real score.
    assert float(scores[0, 0]) == 0.0
    # Histories and targets are handed to the evaluator untouched, so it owns masking.
    assert histories == [tuple(cases[0].test_history)]
    assert targets == [cases[0].test_target]
    assert isinstance(scores, torch.Tensor)


def test_ranking_batches_rejects_a_non_permutation():
    """A duplicate or missing id would be silently overwritten, ranking fewer items than claimed."""
    with pytest.raises(ValueError, match="permutation"):
        list(B.ranking_batches([1, 1, 2], [_case()], batch_size=1))


def test_metadata_arm_streams_per_batch_and_yields_full_width_matrices():
    """REGRESSION: the metadata arm must not materialize a full permutation per user.

    The arm is given 3 users and a 5-item catalogue and must yield matrices of the evaluator's
    expected width (``num_items + 1``) one batch at a time.  Peak memory is a property of the batch,
    so asserting the width and the batch split is what a test can hold on to.
    """
    cases = [
        _case(user_id="a", user_int_id=1, target=2),
        _case(user_id="b", user_int_id=2, target=3),
        _case(user_id="c", user_int_id=3, target=4),
    ]
    heads = [[4, 5], [5, 3], []]
    arm = B.arm_metadata_retrieval(num_items=5, heads=heads)
    batches = list(arm(cases, 2))

    assert [batch[1] for batch in batches] == [[2, 3], [4]]
    for histories, targets, scores in batches:
        assert scores.shape == (len(targets), 6)
    first, second = batches[0][2], batches[1][2]
    # The fused head occupies the top ranks...
    assert int(first[0].argmax()) == 4
    # ...and the unretrieved tail shares one explicit score below every retrieved item.
    assert float(first[0, 1]) == float(first[0, 2]) == -1.0
    # A user with no retrieved head at all gets every *real* item at the tail score, which is a
    # stated convention rather than a fabricated ranking.  Column 0 is PAD and keeps its zero score;
    # the evaluator masks PAD positionally, so that value can never be chosen.
    assert float(second[0, 1:].max()) == -1.0
    assert float(second[0, 0]) == 0.0


def test_metadata_arm_requires_aligned_heads():
    arm = B.arm_metadata_retrieval(num_items=5, heads=[[1]])
    with pytest.raises(ValueError, match="aligned"):
        list(arm([_case(), _case(user_int_id=2)], 2))


# --------------------------------------------------------------------------- #
# Sequential arm input construction
# --------------------------------------------------------------------------- #


def test_sasrec_arm_left_pads_histories_into_the_model_window():
    """REGRESSION: the sequential arm must left-pad, matching training and accepted inference.

    ``full_catalog_scores`` reads the hidden state at the **last valid position**, so a
    right-padded window makes the model summarise a run of PAD rows instead of the user's
    history.  The failure is silent - shapes and finiteness are all correct - and it cost ~20% of
    Recall@10 while this phase was being built, which is why the layout is asserted directly rather
    than inferred from a metric.
    """
    from recommendation.datasets.sasrec import encode_inference_history

    captured: list = []
    max_seq_len = 4

    class _Model:
        def load_state_dict(self, state_dict):
            return None

        def eval(self):
            return self

        def full_catalog_scores(self, input_ids):
            captured.append(input_ids.clone())
            return torch.zeros((input_ids.shape[0], 6), dtype=torch.float32)

    # Stand in at the module boundary: ``arm_sasrec`` imports ``SASRec``/``SASRecConfig`` from
    # ``recommendation.models.sasrec`` and filters the checkpoint config through the config's own
    # dataclass fields, so the fake has to be a real dataclass.
    import dataclasses
    import recommendation.models.sasrec as sasrec_module

    @dataclasses.dataclass
    class _Config:
        num_items: int = 20
        max_seq_len: int = 4

    monkeypatch = pytest.MonkeyPatch()
    try:
        monkeypatch.setattr(sasrec_module, "SASRecConfig", _Config, raising=False)
        monkeypatch.setattr(sasrec_module, "SASRec", lambda config: _Model(), raising=False)
        monkeypatch.setattr(torch, "load", lambda *a, **k: _checkpoint(max_seq_len))
        scorer = B.arm_sasrec(checkpoint=Path("unused.pt"), batch_size=2)
        # Explicit cases, so ``test_history`` is exactly what the assertion below expects:
        # test_history == train_history + (validation_target,), i.e. (9, 11, 12).
        cases = [
            EvaluationCase(
                user_id="u1",
                user_int_id=1,
                train_history=(9, 11),
                validation_target=12,
                test_target=13,
                sequence_length=4,
            ),
            EvaluationCase(
                user_id="u2",
                user_int_id=2,
                train_history=(4, 5),
                validation_target=6,
                test_target=7,
                sequence_length=4,
            ),
        ]
        list(scorer(cases))
    finally:
        monkeypatch.undo()

    padded = captured[0]
    assert padded.shape == (2, max_seq_len)
    # Row 0's history (9, 11, 12) must occupy the *last* three columns and keep its order.
    assert padded[0].tolist() == [0, 9, 11, 12]
    # PAD leads, it does not trail.
    assert padded[0, -1].item() == 12
    # And the layout is exactly what the accepted inference encoder produces.
    assert padded[0].tolist() == list(
        encode_inference_history((9, 11, 12), max_seq_len, num_items=20)
    )


def _checkpoint(max_seq_len: int) -> dict:
    return {
        "model_config": {"num_items": 20, "max_seq_len": max_seq_len},
        "model_state_dict": {},
    }


# --------------------------------------------------------------------------- #
# Retrieval index
# --------------------------------------------------------------------------- #


def test_item_text_index_agrees_with_catalog_search_source(catalogue):
    """The fast index must not change what BM25 says.

    ``CatalogSearchSource`` is the accepted implementation; ``ItemTextIndex`` only changes the data
    structure and drops non-discriminative terms.  For a query whose terms are all below the cap, the
    two must agree on the ranking exactly.
    """
    index = B.ItemTextIndex(catalogue.records)
    accepted = CatalogSearchSource(catalogue)
    query = "unique0007"

    mine = [identity for _score, identity in index.score(query)]
    theirs = list(accepted.search(query, limit=len(mine)).parent_asins)
    assert mine
    assert mine == theirs
    # And the term really is below the cap, or the comparison above would be vacuous.
    assert len(mine) < index.max_document_frequency


def test_item_text_index_drops_non_discriminative_terms(text_index: B.ItemTextIndex):
    """A term in every document matches everything, which is what the cap exists to prevent."""
    assert "universal" not in text_index.query_terms("universal")
    # A unique term survives, so retrieval is still possible at all.
    assert text_index.query_terms("unique0007") == ("unique0007",)
    assert text_index.max_document_frequency == int(
        text_index.size * B.ItemTextIndex.MAX_QUERY_DOCUMENT_FRACTION
    )


def test_item_text_index_returns_nothing_for_an_unknown_query(text_index: B.ItemTextIndex):
    """An honest miss, never a fabricated fallback slice of the catalogue."""
    assert text_index.score("zzzznotacatalogueword") == []
    assert text_index.score("") == []


def test_seed_query_flattens_only_query_fields(catalogue):
    record = catalogue.records["asin0003"]
    text = B.seed_query(record)
    assert "unique0003" in text
    assert "Acme Outdoors" in text
    assert "Sports & Outdoors" in text
    # A missing record contributes no text rather than a placeholder.
    assert B.seed_query(None) == ""


# --------------------------------------------------------------------------- #
# Fusion
# --------------------------------------------------------------------------- #


def test_fused_head_is_fusion_not_concatenation():
    """An item both sources rank *lower* must beat an item only one source ranks first.

    This is the property that separates rank fusion from concatenation, and it is not a
    restatement of the implementation: under concatenation (or under "take the first source"),
    item 1 would win outright because it is rank 1 somewhere.
    """
    first = [1, 2]
    second = [3, 2]
    fused = B.fused_head([first, second], top_n=3)
    assert fused == [2, 1, 3]

    # Reciprocal-rank fusion is additive in rank: item 2's score is the sum of its two
    # contributions, which is exactly why agreement across sources wins.
    expected = 1.0 / (60 + 2) + 1.0 / (60 + 2)
    from recommendation.control.candidate_ledger import reciprocal_rank_fusion

    scores = dict(reciprocal_rank_fusion([["1", "2"], ["3", "2"]]))
    assert scores["2"] == pytest.approx(expected)
    assert scores["2"] > scores["1"] and scores["2"] > scores["3"]


def test_fused_head_uses_the_accepted_fusion():
    """The arm must call the repository's accepted RRF rather than reimplementing it."""
    rankings = [[1, 2, 3], [3, 1, 2]]
    expected = [
        int(identity)
        for identity, _score in reciprocal_rank_fusion(
            [[str(item) for item in ranking] for ranking in rankings]
        )
    ]
    assert B.fused_head(rankings, top_n=3) == expected[:3]


def test_fixed_fusion_requires_aligned_sources():
    cases = [_case(), _case(user_int_id=2)]
    with pytest.raises(ValueError, match="aligned"):
        B.arm_fixed_fusion(
            num_items=5,
            source_rankings={"popularity": [[1], [2]], "sequential": [[1]]},
        )


def test_fixed_fusion_consults_every_source_for_every_user():
    """A fixed fusion is defined by not varying; its heads must be identical across users."""
    cases = [_case(user_int_id=1), _case(user_int_id=2)]
    rankings = {
        "popularity": [[3, 1], [3, 1]],
        "sequential": [[2, 3], [2, 3]],
        "metadata": [[1, 2], [1, 2]],
    }
    arm = B.arm_fixed_fusion(num_items=5, source_rankings=rankings)
    batches = list(arm(cases, 2))
    scores = batches[0][2]
    assert scores.shape == (2, 6)
    # Same plan, same scores: nothing about the user changes the fixed arm's ordering.
    assert float(scores[0, 2]) == float(scores[1, 2])


# --------------------------------------------------------------------------- #
# Agent selection
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("distinct_items", "expected"),
    [
        (1, "popularity"),
        (2, "popularity"),
        (3, "metadata"),
        (7, "metadata"),
        (8, "sequential"),
        (30, "sequential"),
    ],
)
def test_agent_select_sources_thresholds(distinct_items: int, expected: str):
    """The documented rule boundaries, pinned so a threshold change is a visible test change."""
    history = tuple(range(1, distinct_items + 1))
    case = _case(history=history)
    selected, rationale = B.agent_select_sources(case)
    assert selected == [expected]
    assert rationale.split(":")[0] in {"cold_start", "short_history", "long_history"}


def test_agent_select_sources_uses_distinct_items_not_history_length():
    """Repurchase makes a long history with few distinct items, and that is a cold start."""
    # Seven interactions, but only one distinct product.
    case = _case(history=(4, 4, 4, 4, 4, 4, 4))
    selected, rationale = B.agent_select_sources(case)
    assert selected == ["popularity"]
    assert rationale.startswith("cold_start")


def test_agent_select_sources_never_names_an_unavailable_source():
    """The policy is constrained by what the run actually has, like the accepted action validator."""
    case = _case(history=tuple(range(1, 30)))
    selected, rationale = B.agent_select_sources(case, available=("metadata",))
    assert selected == []
    assert "unavailable" in rationale


def test_agent_selection_plan_reports_a_distribution():
    cases = [
        _case(user_id="cold", user_int_id=1, history=(1, 1)),
        _case(user_id="short", user_int_id=2, history=(1, 2, 3)),
        _case(user_id="long", user_int_id=3, history=tuple(range(1, 20))),
    ]
    plan = B.agent_selection_plan(cases)
    assert plan["distribution"] == {"popularity": 1, "metadata": 1, "sequential": 1}
    assert len(plan["choices"]) == len(cases) == len(plan["reasons"])


def test_agent_selected_arm_can_serve_every_rule_branch():
    """Every rule must produce a served, non-empty head - a branch with no data would be invisible."""
    cases = [
        _case(user_id="cold", user_int_id=1, history=(1, 1)),
        _case(user_id="short", user_int_id=2, history=(1, 2, 3)),
        _case(user_id="long", user_int_id=3, history=tuple(range(1, 20))),
    ]
    rankings = {
        "popularity": [[5, 4] for _ in cases],
        "sequential": [[4, 3] for _ in cases],
        "metadata": [[3, 2] for _ in cases],
    }
    plan = B.agent_selection_plan(cases)
    arm = B.arm_agent_selected(num_items=5, source_rankings=rankings, plan=plan)
    (histories, targets, scores), = list(arm(cases, 3))

    # Each rule served a different source, so each row's winner differs accordingly.
    assert int(scores[0].argmax()) == 5  # popularity rule
    assert int(scores[1].argmax()) == 3  # metadata rule
    assert int(scores[2].argmax()) == 4  # sequential rule


def test_agent_selected_falls_back_to_the_tail_for_an_unservable_plan():
    """An unservable selection is visible in the metrics rather than papered over with a default."""
    cases = [_case(history=tuple(range(1, 20)))]
    plan = B.agent_selection_plan(cases)
    # The plan asks for 'sequential', but the arm is only given 'metadata'.
    arm = B.arm_agent_selected(num_items=5, source_rankings={"metadata": [[4, 3]]}, plan=plan)
    (histories, targets, scores), = list(arm(cases, 1))
    # Every real item is at the tail; PAD (column 0) is not a candidate the evaluator can pick.
    assert float(scores[0, 1:].max()) == -1.0


# --------------------------------------------------------------------------- #
# Protocol / leakage
# --------------------------------------------------------------------------- #


def test_ranking_batches_for_shares_the_direction_rule():
    """The helper used by retrieval arms must rank the best item first, like ``ranking_batches``."""
    cases = [_case(), _case(user_int_id=2)]
    rankings = [[3, 5, 2, 4, 1], [1, 2, 3, 4, 5]]
    (histories, targets, scores), = list(B.ranking_batches_for(rankings, cases, batch_size=2))
    assert scores.shape == (2, 6)
    assert int(scores[0].argmax()) == 3
    assert int(scores[1].argmax()) == 1


def test_leakage_checks_pass_when_every_history_precedes_its_target():
    """The no-future-leakage claim, decided on timestamps."""
    cases = [_case(user_id="u1", history=(1, 2, 3))]
    checks = B.leakage_checks(cases, cases, timestamps={"u1": [10, 20, 30, 40, 50]})
    assert checks["passed"] is True
    evidence = checks["timestamp_evidence"]
    assert evidence["cases_checked"] == 1
    assert evidence["history_strictly_before_target"] == 1
    assert evidence["non_monotonic_sequences"] == 0
    assert evidence["all_checked_cases_strictly_ordered"] is True
    assert checks["unexplained_target_in_history"] == 0


def test_leakage_checks_allow_a_legal_repurchase():
    """A repurchase puts the target in the history with an *earlier* timestamp.

    The naive rule "the target must not appear in its own history" flags this as leakage.  Applying
    that rule to the real cohort flagged 2.74% of users, all of them legitimate repurchase, which is
    how the false positive was found and why the check is decided on timestamps.
    """
    case = EvaluationCase(
        user_id="u1",
        user_int_id=1,
        train_history=(7, 1, 2),
        validation_target=3,
        test_target=7,
        sequence_length=5,
    )
    checks = B.leakage_checks([case], [case], timestamps={"u1": [10, 20, 30, 40, 50]})
    assert checks["passed"] is True
    assert checks["of_which_repurchase_of_an_earlier_item"] == 1
    assert checks["repeat_share"] == 1.0
    # The repurchase is legal *because* the earlier occurrence precedes the target.
    assert checks["timestamp_evidence"]["history_strictly_before_target"] == 1


def test_leakage_checks_fail_on_a_non_monotonic_sequence():
    """Out-of-order timestamps mean the history is not a past-only prefix, so the check must fail."""
    cases = [_case(user_id="u1", history=(1, 2, 3))]
    checks = B.leakage_checks(cases, cases, timestamps={"u1": [50, 20, 30, 40, 10]})
    assert checks["passed"] is False
    assert checks["timestamp_evidence"]["non_monotonic_sequences"] >= 1
    # The target's stamp (40) is still after its own history, but the sequence is not monotonic, so
    # the check cannot certify the split.
    assert checks["timestamp_evidence"]["all_checked_cases_strictly_ordered"] is False


def test_leakage_checks_report_none_without_timestamps():
    """A structural pass must not be mistaken for timestamp evidence."""
    cases = [_case(user_id="u1", history=(1, 2, 3))]
    checks = B.leakage_checks(cases, cases, timestamps=None)
    assert checks["timestamp_evidence"] is None
    assert checks["passed"] is True  # structural checks only


def test_leakage_checks_fail_when_a_sample_target_is_not_in_the_full_cohort():
    """Subsetting must not be able to introduce a target the protocol would not have produced."""
    full = [_case(user_id="u1", history=(1, 2, 3))]
    sample = [_case(user_id="u1", history=(1, 2, 3), target=999)]
    checks = B.leakage_checks(sample, full, timestamps={"u1": [10, 20, 30, 40, 50]})
    assert checks["passed"] is False
    assert checks["sample_targets_subset_of_full"] is False


def test_assert_protocol_raises_on_a_failed_check():
    with pytest.raises(RuntimeError, match="protocol/leakage checks failed"):
        B.assert_protocol({"passed": False, "why": "synthetic"})

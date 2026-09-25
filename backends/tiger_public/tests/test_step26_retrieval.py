"""Step 2.6 tests: retrieval, resolver, provenance, and the CERTIFIED brute-force oracle.

Split deliberately into two halves:

* **model-free** tests cover the search, the resolver, ranking, collisions, budget handling and
  the *exactness* of CERTIFIED against an independent exhaustive oracle.  They need neither torch
  nor a checkpoint, so the correctness argument is checkable in any environment.
* **model-backed** tests use a tiny randomly initialised generator over the same synthetic
  layout, so the batched scorer, the encoder-input construction and batch/single parity are
  exercised hermetically and fast.  The accepted-checkpoint gate (G13.7) is a separate test that
  skips when the frozen artifact is absent.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from tiger_public.retrieve import (
    APPROXIMATE,
    CERTIFIED,
    Candidate,
    RetrievalConfig,
    RetrievalError,
    SidResolver,
    TigerCaseScorer,
    aggregate_certificate,
    assert_next_log_probs_admissible,
    build_history_input,
    path_to_semantic_id,
    retrieve,
    retrieve_approximate,
    retrieve_certified,
    tail_ranking,
)
from tiger_public.trie import CatalogueTrie, item_token_path

# --------------------------------------------------------------------------- #
# A synthetic layout and catalogue, small enough to enumerate by hand
# --------------------------------------------------------------------------- #
# level 0 -> tokens 0..2 ; level 1 -> tokens 3..5 ; dedup level -> tokens 6..8
# specials: sep 9, eos 10, bos 11, pad 12 ; vocab_size 13
LAYOUT = {
    "format": "agentrecx.tiger.generator_layout.v1",
    "levels": 2,
    "dedup_levels": 1,
    "codebook_size": 3,
    "dedup_vocab_size": 3,
    "level_offsets": [0, 3, 6],
    "per_item_tokens": 3,
    "vocab_size": 13,
    "sentinel_tokenisable": False,
    "special": {"pad": 12, "bos": 11, "eos": 10, "sep": 9},
}

#: row 0 is the reserved PAD sentinel row and is skipped by the trie.
ASSIGNMENT = [
    (-1, -1, -1),
    (0, 0, 0),
    (0, 0, 1),
    (0, 1, 0),
    (1, 0, 0),
    (2, 2, 2),
]


def build_trie(assignment=ASSIGNMENT, layout=LAYOUT) -> CatalogueTrie:
    return CatalogueTrie(assignment, layout=layout)


def complete_paths(trie: CatalogueTrie) -> list[tuple[int, ...]]:
    """Every complete catalogue token path, by depth-first enumeration."""
    out: list[tuple[int, ...]] = []

    def walk(prefix: tuple[int, ...]) -> None:
        if trie.is_complete(prefix):
            out.append(prefix)
        for token in trie.children(prefix):
            walk(prefix + (token,))

    walk(())
    return out


# --------------------------------------------------------------------------- #
# A deterministic, hash-seeded synthetic "model"
# --------------------------------------------------------------------------- #
class OracleScorer:
    """A pure function of the prefix: a hash-seeded full-vocabulary log-softmax.

    Deliberately *not* a real model.  Its only job is to be an independently reproducible score
    field so that the exhaustive oracle and the branch-and-bound search can be compared without
    either one being derived from the other.
    """

    def __init__(self, vocab_size: int = LAYOUT["vocab_size"], scale: float = 2.0) -> None:
        self.vocab_size = int(vocab_size)
        self.scale = float(scale)
        self.calls = 0

    def table(self, prefix: tuple[int, ...]) -> np.ndarray:
        digest = hashlib.sha256(repr(tuple(prefix)).encode("utf-8")).digest()
        seed = int.from_bytes(digest[:8], "big") % (2**32)
        rng = np.random.default_rng(seed)
        logits = rng.normal(size=self.vocab_size) * self.scale
        logits -= logits.max()
        return logits - np.log(np.exp(logits).sum())

    def __call__(self, prefixes):
        self.calls += 1
        rows = [self.table(tuple(int(t) for t in p)) for p in prefixes]
        if not rows:
            return np.zeros((0, self.vocab_size), dtype=np.float64)
        return np.vstack(rows)


def oracle_top_k(trie: CatalogueTrie, scorer, k: int):
    """Exhaustive enumeration + resolution.  The independent ground truth for CERTIFIED.

    Works against ANY :class:`NextTokenScorer`, including the model-backed one, so the oracle
    and the certified search share nothing but the published scorer contract.
    """
    scored = []
    for path in complete_paths(trie):
        total = 0.0
        for level, token in enumerate(path):
            total += float(scorer([path[:level]])[0][token])
        scored.append((total, path))
    # descending score, ascending path
    scored.sort(key=lambda item: (-item[0], item[1]))
    resolver = SidResolver(trie)
    out = []
    seen = set()
    for score, path in scored:
        resolved = resolver.resolve(path)
        if resolved is None:
            continue
        item_id, _ = resolved
        if item_id in seen:
            continue
        seen.add(item_id)
        out.append((item_id, score, path))
    return out[:k]


# --------------------------------------------------------------------------- #
# G15 - CERTIFIED exactness against the oracle
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("k", [1, 2, 3, 5, 10, 20])
def test_certified_equals_bruteforce_oracle(k: int) -> None:
    """CERTIFIED top-k must equal the exhaustive oracle top-k, for every k."""
    trie = build_trie()
    scorer = OracleScorer()
    result = retrieve_certified(trie, scorer, k_prime=k, batch_prefixes=1)
    expected = oracle_top_k(trie, scorer, k)
    assert [c.item_id for c in result.candidates] == [item for item, _, _ in expected]
    assert result.status == CERTIFIED
    assert result.budget_exhausted is False
    for got, (_, score, path) in zip(result.candidates, expected):
        assert got.score == pytest.approx(score, abs=1e-9)
        assert got.token_path == path


def non_leaf_node_count(trie: CatalogueTrie) -> int:
    """Nodes that have at least one child; the ones an exhaustive search must expand."""
    count = 0

    def walk(prefix: tuple[int, ...]) -> None:
        nonlocal count
        children = trie.children(prefix)
        if children:
            count += 1
            for token in children:
                walk(prefix + (token,))

    walk(())
    return count


def test_certified_enumerates_everything_when_frontier_is_capped() -> None:
    """G15.4: with k_prime >= the number of items the search is exhaustive, not truncating.

    Complete leaves are *recorded* as candidates rather than expanded (they have no children),
    so an exhaustive run expands every non-leaf node and returns every catalogue item.
    """
    trie = build_trie()
    result = retrieve_certified(trie, OracleScorer(), k_prime=len(ASSIGNMENT) + 5, batch_prefixes=1)
    assert result.prefixes_expanded == non_leaf_node_count(trie)
    assert result.prefixes_expanded == trie.stats.nodes - trie.stats.leaves
    assert result.nodes_pruned_by_bound == 0
    assert len(result.candidates) == len(ASSIGNMENT) - 1 == trie.stats.catalogue_items
    assert sorted(c.item_id for c in result.candidates) == [1, 2, 3, 4, 5]


def test_certified_single_item_catalogue() -> None:
    trie = build_trie([(-1, -1, -1), (1, 1, 1)])
    scorer = OracleScorer()
    result = retrieve_certified(trie, scorer, k_prime=5, batch_prefixes=1)
    assert len(result.candidates) == 1
    assert result.candidates[0].item_id == 1
    assert result.status == CERTIFIED


def test_certified_budget_exhaustion_is_reported_not_hidden() -> None:
    """A starved budget must set budget_exhausted and must NOT claim certification."""
    trie = build_trie()
    result = retrieve_certified(trie, OracleScorer(), k_prime=1, prefix_budget=1, batch_prefixes=1)
    assert result.budget_exhausted is True
    assert result.exhaustion_reason == "prefix_budget"
    block = aggregate_certificate([result], requested_frontier=1, method="branch_and_bound",
                                 status=CERTIFIED)
    assert block.certificate_holds_for_all_cases is False
    assert block.budget_exhausted_cases == 1


def test_certified_determinism_repeat_runs_identical() -> None:
    trie = build_trie()
    a = retrieve_certified(trie, OracleScorer(), k_prime=3, batch_prefixes=1)
    b = retrieve_certified(trie, OracleScorer(), k_prime=3, batch_prefixes=1)
    assert a.as_dict() == b.as_dict()


def test_certified_batch_size_does_not_change_the_result() -> None:
    """Batching is an efficiency device; it must not alter ranking or the certified set."""
    trie = build_trie()
    small = retrieve_certified(trie, OracleScorer(), k_prime=4, batch_prefixes=1)
    large = retrieve_certified(trie, OracleScorer(), k_prime=4, batch_prefixes=64)
    assert [c.item_id for c in small.candidates] == [c.item_id for c in large.candidates]
    assert [c.score for c in small.candidates] == pytest.approx([c.score for c in large.candidates])


def test_approximate_is_labelled_and_never_certified() -> None:
    trie = build_trie()
    result = retrieve_approximate(trie, OracleScorer(), k=3, beam=2, batch_prefixes=1)
    assert result.status == APPROXIMATE
    assert all(c.retrieval_mode == APPROXIMATE for c in result.candidates)
    assert result.threshold_final != result.threshold_final  # NaN: APPROXIMATE issues no threshold


def test_approximate_with_full_beam_matches_the_oracle() -> None:
    """A beam wide enough to keep everything is exhaustive, so it should agree with the oracle."""
    trie = build_trie()
    scorer = OracleScorer()
    result = retrieve_approximate(trie, scorer, k=5, beam=10_000, batch_prefixes=1)
    assert [c.item_id for c in result.candidates] == [i for i, _, _ in oracle_top_k(trie, scorer, 5)]


# --------------------------------------------------------------------------- #
# Resolver, collisions, identity discipline
# --------------------------------------------------------------------------- #


def test_resolver_maps_known_paths_and_refuses_unknown_ones() -> None:
    trie = build_trie()
    resolver = SidResolver(trie)
    assert resolver.resolve((0, 3, 6))[0] == 1
    assert resolver.resolve((0, 3, 7))[0] == 2
    assert resolver.resolve((2, 5, 8))[0] == 5
    # not a catalogue path
    assert resolver.resolve((2, 5, 6)) is None
    # a legal prefix is not a complete item
    assert resolver.resolve((0, 3)) is None
    assert resolver.is_prefix((0, 3)) is True
    assert resolver.is_complete((0, 3)) is False


def test_resolver_refuses_pad_and_out_of_range_digits() -> None:
    with pytest.raises(Exception):
        item_token_path((-1, 0, 0), LAYOUT)  # the PAD sentinel is not tokenisable
    with pytest.raises(Exception):
        item_token_path((0, 0, 99), LAYOUT)  # digit outside its level block


def test_sid_collision_is_deterministic_and_counted() -> None:
    """Two items on one path: smallest id wins, and the collision is counted."""
    colliding = [(-1, -1, -1), (0, 0, 0), (0, 0, 0), (1, 0, 0)]
    trie = build_trie(colliding)
    assert trie.collision_free() is False
    resolver = SidResolver(trie)
    item_id, record = resolver.resolve((0, 3, 6))
    assert item_id == 1  # smallest registered id, deterministically
    assert record.collisions_observed == 1
    assert record.collision_policy == "smallest_item_id_wins"
    assert resolver.collisions_observed == 1
    # repeat resolution is identical and does not drift
    assert SidResolver(trie).resolve((0, 3, 6))[0] == 1


def test_frozen_style_assignment_is_collision_free() -> None:
    assert build_trie().collision_free() is True


def test_duplicate_candidates_are_deduplicated() -> None:
    """A colliding assignment must not emit the same item id twice."""
    colliding = [(-1, -1, -1), (0, 0, 0), (0, 0, 0), (1, 0, 0)]
    trie = build_trie(colliding)
    result = retrieve_certified(trie, OracleScorer(), k_prime=10, batch_prefixes=1)
    ids = [c.item_id for c in result.candidates]
    assert len(ids) == len(set(ids))
    assert ids[0] == 1


def test_item_id_is_never_canonical_identity() -> None:
    """The boundary rule: no parent_asin may appear anywhere in a retrieval record."""
    trie = build_trie()
    result = retrieve_certified(trie, OracleScorer(), k_prime=2, batch_prefixes=1)
    blob = json.dumps(result.as_dict()).lower()
    assert "parent_asin" not in blob
    assert "asin" not in blob


def test_path_to_semantic_id_inverts_the_offsets() -> None:
    assert path_to_semantic_id((0, 3, 6), LAYOUT) == (0, 0, 0)
    assert path_to_semantic_id((2, 5, 8), LAYOUT) == (2, 2, 2)
    with pytest.raises(RetrievalError):
        path_to_semantic_id((0, 3), LAYOUT)  # too short to be complete


# --------------------------------------------------------------------------- #
# Ranking, tails, empty results, K > available
# --------------------------------------------------------------------------- #


def test_ranking_is_descending_score_then_ascending_id() -> None:
    trie = build_trie()
    result = retrieve_certified(trie, OracleScorer(), k_prime=5, batch_prefixes=1)
    scores = [c.score for c in result.candidates]
    assert scores == sorted(scores, reverse=True)
    assert [c.rank for c in result.candidates] == list(range(1, len(result.candidates) + 1))


def test_k_greater_than_available_is_not_an_error() -> None:
    trie = build_trie()
    result = retrieve_certified(trie, OracleScorer(), k_prime=999, batch_prefixes=1)
    assert len(result.candidates) == len(ASSIGNMENT) - 1
    assert result.status == CERTIFIED


def test_tail_ranking_is_complete_and_starts_with_the_frontier() -> None:
    trie = build_trie()
    result = retrieve_certified(trie, OracleScorer(), k_prime=2, batch_prefixes=1)
    ranking = tail_ranking(result.candidates, num_items=5)
    assert ranking[: len(result.candidates)] == tuple(c.item_id for c in result.candidates)
    assert sorted(ranking) == [1, 2, 3, 4, 5]
    assert len(ranking) == len(set(ranking))


def test_empty_catalogue_yields_no_candidates_and_no_crash() -> None:
    """A catalogue whose only reachable path is trimmed away must return nothing, quietly."""
    trie = CatalogueTrie([(-1, -1, -1), (0, 0, 0), (1, 1, 1)], layout=LAYOUT)
    result = retrieve_certified(trie, OracleScorer(), k_prime=3, batch_prefixes=1)
    assert result.status == CERTIFIED
    assert len(result.candidates) == 2


# --------------------------------------------------------------------------- #
# Admissibility guard
# --------------------------------------------------------------------------- #


def test_admissibility_guard_rejects_non_log_softmax() -> None:
    good = np.log(np.full((2, 5), 0.2))
    assert_next_log_probs_admissible(good)
    with pytest.raises(RetrievalError):
        assert_next_log_probs_admissible(np.full((2, 5), 0.5))  # positive => not a log-softmax
    with pytest.raises(RetrievalError):
        assert_next_log_probs_admissible(np.array([[np.nan] * 5]))  # non-finite
    with pytest.raises(RetrievalError):
        assert_next_log_probs_admissible(np.zeros((2, 5, 1)))  # wrong rank


def test_search_refuses_a_non_admissible_scorer() -> None:
    trie = build_trie()

    def bad(prefixes):
        return np.full((len(prefixes), LAYOUT["vocab_size"]), 0.25)  # positive

    with pytest.raises(RetrievalError):
        retrieve_certified(trie, bad, k_prime=1, batch_prefixes=1)


# --------------------------------------------------------------------------- #
# Config validation
# --------------------------------------------------------------------------- #


def test_config_rejects_unknown_mode_and_bad_budgets() -> None:
    with pytest.raises(RetrievalError):
        RetrievalConfig(mode="CERTIFIEDISH")
    with pytest.raises(RetrievalError):
        RetrievalConfig(k=0)
    with pytest.raises(RetrievalError):
        RetrievalConfig(beam=0)
    with pytest.raises(RetrievalError):
        RetrievalConfig(prefix_budget=0)
    with pytest.raises(RetrievalError):
        RetrievalConfig(batch_prefixes=0)


def test_retrieve_dispatch_uses_the_requested_mode() -> None:
    trie = build_trie()
    scorer = OracleScorer()
    certified = retrieve(trie, scorer, config=RetrievalConfig(mode=CERTIFIED, k=3),
                         required_frontier=3)
    approximate = retrieve(trie, scorer, config=RetrievalConfig(mode=APPROXIMATE, k=3, beam=4),
                           required_frontier=3)
    assert certified.status == CERTIFIED
    assert approximate.status == APPROXIMATE


# --------------------------------------------------------------------------- #
# Encoder-input construction (model-free half; parity with build_examples)
# --------------------------------------------------------------------------- #


def test_build_history_input_matches_the_training_construction() -> None:
    history = [1, 2, 4]
    tokens = build_history_input(history, assignment=ASSIGNMENT, layout=LAYOUT, max_hist_items=20)
    expected = (11, 0, 3, 6, 9, 0, 3, 7, 9, 1, 3, 6, 9)
    assert tokens == expected


def test_build_history_input_windows_and_marks_every_item() -> None:
    history = [1, 2, 3, 4, 5]
    tokens = build_history_input(history, assignment=ASSIGNMENT, layout=LAYOUT, max_hist_items=2)
    # only the last two items survive the window, each followed by SEP
    assert tokens == (11, 1, 3, 6, 9, 2, 5, 8, 9)
    assert tokens.count(9) == 2
    with pytest.raises(RetrievalError):
        build_history_input([999], assignment=ASSIGNMENT, layout=LAYOUT)


def test_build_history_input_refuses_a_layout_without_sep() -> None:
    stripped = dict(LAYOUT)
    stripped["special"] = {"pad": 12, "bos": 11, "eos": 10}
    with pytest.raises(RetrievalError):
        build_history_input([1], assignment=ASSIGNMENT, layout=stripped)


# --------------------------------------------------------------------------- #
# Model-backed half: tiny hermetic generator
# --------------------------------------------------------------------------- #
torch = pytest.importorskip("torch", reason="the model-backed scorer needs torch")


def tiny_generator():
    from tiger_public.tiger import TigerConfig, TigerGenerator

    config = TigerConfig(
        max_hist_items=4, d_model=32, num_layers=1, num_heads=2, d_ff=64, dropout=0.0,
        batch_size=8, epochs=1, learning_rate=1e-3, weight_decay=0.0, max_grad_norm=1.0,
        warmup_fraction=0.1, bf16=False, seed=2026,
    )
    torch.manual_seed(config.seed)
    return TigerGenerator(config=config, layout=LAYOUT, vocab_size=LAYOUT["vocab_size"])


def test_model_backed_scorer_is_a_full_vocabulary_log_softmax() -> None:
    generator = tiny_generator()
    tokens = build_history_input([1, 2], assignment=ASSIGNMENT, layout=LAYOUT)
    scorer = TigerCaseScorer(generator, encoder_tokens=tokens, batch_prefixes=8)
    table = scorer([()])
    assert table.shape == (1, LAYOUT["vocab_size"])
    assert np.all(table <= 0)
    assert float(np.exp(table).sum()) == pytest.approx(1.0, abs=1e-5)
    assert_next_log_probs_admissible(table)


def test_model_backed_scorer_batch_equals_single() -> None:
    """Batching must not leak padding into the readout.

    Scores are compared with a float tolerance rather than bitwise: a different batch shape
    means different GEMM reduction shapes, so the last few ULPs legitimately differ.  The
    property that actually matters - identical ranking - is asserted separately below.  A
    canonical run therefore fixes ``batch_prefixes`` and reports it.
    """
    generator = tiny_generator()
    tokens = build_history_input([1, 3], assignment=ASSIGNMENT, layout=LAYOUT)
    batched = TigerCaseScorer(generator, encoder_tokens=tokens, batch_prefixes=64)
    single = TigerCaseScorer(generator, encoder_tokens=tokens, batch_prefixes=1)
    prefixes = [(0,), (1,), (0, 3), (0, 3, 6), (2, 5), (2, 5, 8)]
    a = batched(prefixes)
    b = single(prefixes)
    assert a.shape == b.shape
    np.testing.assert_allclose(a, b, rtol=0, atol=1e-4)
    # padding must not change which token wins
    assert list(np.argmax(a, axis=1)) == list(np.argmax(b, axis=1))


def test_model_backed_certified_ranking_is_batch_size_invariant() -> None:
    """The certified item ORDER must not depend on the batching used to compute it."""
    generator = tiny_generator()
    trie = build_trie()
    tokens = build_history_input([1, 2, 4], assignment=ASSIGNMENT, layout=LAYOUT)

    def run(batch_prefixes: int):
        scorer = TigerCaseScorer(generator, encoder_tokens=tokens, batch_prefixes=batch_prefixes)
        return retrieve_certified(trie, lambda p: scorer(p), k_prime=5,
                                  batch_prefixes=batch_prefixes)

    one = run(1)
    many = run(64)
    assert [c.item_id for c in one.candidates] == [c.item_id for c in many.candidates]


def test_model_backed_scores_are_bit_identical_for_a_fixed_batch_size() -> None:
    """Reproducibility contract: a FIXED configuration is bit-identical across instances (G15.3)."""
    tokens = build_history_input([1, 2], assignment=ASSIGNMENT, layout=LAYOUT)
    prefixes = [(0,), (0, 3), (2, 5, 8)]
    a = TigerCaseScorer(tiny_generator(), encoder_tokens=tokens, batch_prefixes=8)(prefixes)
    b = TigerCaseScorer(tiny_generator(), encoder_tokens=tokens, batch_prefixes=8)(prefixes)
    assert np.array_equal(a, b)


def test_model_backed_scorer_is_deterministic_across_instances() -> None:
    tokens = build_history_input([1, 2], assignment=ASSIGNMENT, layout=LAYOUT)
    first = TigerCaseScorer(tiny_generator(), encoder_tokens=tokens, batch_prefixes=4)([(0, 3)])
    second = TigerCaseScorer(tiny_generator(), encoder_tokens=tokens, batch_prefixes=4)([(0, 3)])
    np.testing.assert_allclose(first, second, rtol=0, atol=1e-7)


def test_model_backed_certified_matches_oracle_on_the_tiny_model() -> None:
    """End-to-end exactness: the real search over a real (tiny) model equals brute force."""
    generator = tiny_generator()
    trie = build_trie()
    tokens = build_history_input([1, 2, 4], assignment=ASSIGNMENT, layout=LAYOUT)
    scorer = TigerCaseScorer(generator, encoder_tokens=tokens, batch_prefixes=8)

    class ScorerAdapter:
        def __call__(self, prefixes):
            return scorer(prefixes)

    adapter = ScorerAdapter()
    result = retrieve_certified(trie, adapter, k_prime=4, batch_prefixes=4)
    expected = oracle_top_k(trie, adapter, 4)
    assert [c.item_id for c in result.candidates] == [i for i, _, _ in expected]
    assert [c.score for c in result.candidates] == pytest.approx(
        [s for _, s, _ in expected], abs=1e-6
    )


# --------------------------------------------------------------------------- #
# G13.7 - the accepted checkpoint loads through the production loader
# --------------------------------------------------------------------------- #
ACCEPTED_CHECKPOINT = Path("/root/autodl-tmp/gate_d_fp32_canonical_remediation/tiger.pt")
ACCEPTED_SHA256 = "95e5cb6f816bb271b8551a6dd9b62ee33c8e0f6b5908ce426622113a072c6c0a"


@pytest.mark.skipif(not ACCEPTED_CHECKPOINT.exists(), reason="accepted checkpoint not present")
def test_accepted_checkpoint_hash_and_load() -> None:
    from tiger_public.tiger import load_generator

    digest = hashlib.sha256(ACCEPTED_CHECKPOINT.read_bytes()).hexdigest()
    assert digest == ACCEPTED_SHA256
    run_dir = ACCEPTED_CHECKPOINT.parent
    generator, metadata, _resume = load_generator(
        out_dir=run_dir,
        sid_dir=Path("/root/autodl-tmp/step24f/runs/tiger_public_2026/sid"),
        exposure_path=Path(
            "/root/autodl-tmp/gate_d_handoff_restore/runs/tiger_backend_handoff_prod"
            "/train_exposure.jsonl"
        ),
        catalogue_dir=Path(
            "/root/autodl-tmp/gate_d_handoff_restore/runs/tiger_backend_handoff_prod"
        ),
        generator_layout_path=run_dir / "generator_layout.json",
        device="cpu",
    )
    assert metadata["score_rule"]["version"] == 1
    assert metadata["score_rule"]["eos_in_score"] is False
    assert metadata["score_rule"]["child_renormalisation"] is False
    assert metadata["training"]["precision"] == "fp32"
    assert metadata["training"]["global_step"] == 88420
    assert metadata["training"]["completed_epoch"] == 19
    assert generator.layout["special"]["pad"] == 1024
    assert generator.vocab_size == 1028
    assert generator.per_item_tokens == 4

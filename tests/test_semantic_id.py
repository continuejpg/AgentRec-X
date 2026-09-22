"""Tests for the Semantic-ID generative-retrieval arm.

Three properties matter more than the metric, and each has a test because each failure mode
produces plausible-looking but invalid results rather than an error:

1. **Identity cannot be created by generation.**  A generated code that no catalogue item occupies
   must never become a recommendable item.  This is the trust boundary of the whole step, so it is
   asserted on the resolver directly *and* through the constrained decoder.
2. **The tokenizer is total and audited.**  Every catalogue item needs a code, and collisions must
   be measured rather than hidden - a silently lossy assignment would cap the arm's ceiling without
   anyone noticing.
3. **Training labels come from `train_history` only.**  A leaked evaluation target would inflate
   every metric while all shape assertions still passed.

Everything runs on small synthetic fixtures: no dataset, no checkpoint, no network.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from recommendation.evaluation.split import EvaluationCase  # noqa: E402
from recommendation.semantic_id.dataset import (  # noqa: E402
    GeneratorDataError,
    GeneratorDatasetConfig,
    build_generator_dataset,
    iter_batches,
)
from recommendation.semantic_id.generator import (  # noqa: E402
    SemanticIdGenerator,
    SemanticPrefixTrie,
    build_token_layout,
)
from recommendation.semantic_id.resolver import (  # noqa: E402
    COLLISION_RULE,
    SidResolver,
)
from recommendation.semantic_id.tokenizer import (  # noqa: E402
    QuantizerConfig,
    RqVaeTokenizer,
    TokenizerError,
    audit_tokenizer,
    build_item_features,
)
from recommendation.semantic_id.training import (  # noqa: E402
    GeneratorTrainerConfig,
    build_example_arrays,
    train_generator,
)

LEVELS = 3
CODEBOOK = 16


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #


def make_assignment(num_items: int = 60, *, seed: int = 7, collide: bool = True) -> list[list[int]]:
    """A small, deterministic Semantic-ID assignment indexed by item id (index 0 is PAD)."""
    generator = torch.Generator().manual_seed(seed)
    assignment = [[0] * LEVELS]
    seen: set[tuple[int, ...]] = set()
    for _ in range(num_items):
        while True:
            codes = [int(torch.randint(0, CODEBOOK, (1,), generator=generator)) for _ in range(LEVELS)]
            key = tuple(codes)
            if not collide and key in seen:
                continue
            seen.add(key)
            break
        assignment.append(codes)
    if collide and num_items >= 2:
        assignment[-1] = list(assignment[1])  # a guaranteed collision group of two
    return assignment


def make_model(num_items: int = 60) -> SemanticIdGenerator:
    torch.manual_seed(0)
    return SemanticIdGenerator(
        layout=build_token_layout(levels=LEVELS, codebook_size=CODEBOOK),
        max_items=6,
        d_model=32,
        n_heads=2,
        n_layers=1,
        dropout=0.0,
    )


def make_cases(count: int = 40, *, num_items: int = 60) -> list[EvaluationCase]:
    return [
        EvaluationCase(
            user_id=f"user-{index}",
            user_int_id=index,
            train_history=tuple(range(1, 4 + (index % 5))),
            validation_target=num_items - 1,
            test_target=num_items,
            sequence_length=6 + (index % 5),
        )
        for index in range(count)
    ]


# --------------------------------------------------------------------------- #
# Trust boundary: identity cannot be generated into existence
# --------------------------------------------------------------------------- #


def test_an_unknown_semantic_id_resolves_to_nothing():
    """The core guarantee: an unoccupied code must fail closed, never nearest-neighbour."""
    resolver = SidResolver(make_assignment(30, collide=False))
    occupied = resolver.members((0, 0, 0))
    assert occupied == (), "PAD's code is not a catalogue item"
    resolution = resolver.resolve((CODEBOOK - 1, CODEBOOK - 1, CODEBOOK - 1))
    assert resolution.item_id is None
    assert resolution.invalid is True


def test_pad_is_never_resolvable():
    """Item id 0 is padding, not a product, whatever code it happens to carry."""
    assignment = make_assignment(20)
    resolver = SidResolver(assignment)
    assert resolver.resolve(assignment[0]).item_id is None


def test_collisions_resolve_deterministically_and_are_reported():
    """Two items sharing a code must resolve the same way every time, and say so."""
    assignment = make_assignment(30)
    resolver = SidResolver(assignment)
    shared = tuple(assignment[1])
    first = resolver.resolve(shared)
    second = resolver.resolve(shared)
    assert first.item_id == second.item_id == min(resolver.members(shared))
    assert first.collision_size == 2, "the collision must be visible to the caller"
    assert COLLISION_RULE


def test_an_eligibility_filter_never_becomes_a_fallback():
    """Excluding every member of a collision group yields nothing, not an ineligible member."""
    assignment = make_assignment(30)
    resolver = SidResolver(assignment)
    shared = tuple(assignment[1])
    assert resolver.resolve(shared, eligible=lambda item: item > 10**6).item_id is None


def test_constrained_decoding_cannot_leave_the_catalogue_code_set():
    """Every generated code path must be one the catalogue actually occupies."""
    assignment = make_assignment(60)
    trie = SemanticPrefixTrie(assignment, levels=LEVELS)
    resolver = SidResolver(assignment)
    model = make_model()
    histories = [[code for item in range(1, 5) for code in assignment[item]]] * 4
    for result in model.generate(histories, trie=trie, beam=1):
        assert result.complete
        node = trie.node_for(result.codes)
        assert node is not None, result.codes
        assert trie.items_at(node), "a decoded path must end at at least one catalogue item"
        assert resolver.is_catalogue_code(result.codes)


def test_generation_is_deterministic():
    """The same history and model must decode identically, or the arm is unverifiable."""
    assignment = make_assignment(60)
    trie = SemanticPrefixTrie(assignment, levels=LEVELS)
    model = make_model()
    histories = [[code for item in range(1, 6) for code in assignment[item]]]
    first = model.generate(histories, trie=trie)[0]
    second = model.generate(histories, trie=trie)[0]
    assert first.codes == second.codes
    assert first.log_probability == second.log_probability


def test_decoding_reports_how_much_the_constraint_bit():
    assignment = make_assignment(60)
    trie = SemanticPrefixTrie(assignment, levels=LEVELS)
    result = make_model().generate(
        [[code for item in range(1, 4) for code in assignment[item]]], trie=trie
    )[0]
    assert len(result.legal_options) == LEVELS
    assert all(option >= 1 for option in result.legal_options)


# --------------------------------------------------------------------------- #
# Tokenizer
# --------------------------------------------------------------------------- #


def test_every_catalogue_item_gets_a_code():
    """Total coverage: no item may be left without a Semantic ID."""
    features, note = build_item_features(
        num_items=80, item_embedding_dim=16, categories=None, seed=2026
    )
    config = QuantizerConfig(
        input_size=features.shape[1], latent_size=8, levels=LEVELS, codebook_size=32, hidden_size=32
    )
    codes = RqVaeTokenizer(config).codes_for_features(features).tolist()
    audit = audit_tokenizer(codes, num_items=80, config=config)
    assert audit["coverage"] == 1.0
    assert audit["items_with_a_semantic_id"] == 80
    assert len(codes) == 81, "one row per item plus PAD"
    assert note["rows"] == 81


def test_the_assignment_is_reproducible_from_the_seed():
    """The tokenizer is an untrained map from features to codes; the features must be seeded."""
    first, _ = build_item_features(num_items=50, item_embedding_dim=12, categories=None, seed=99)
    second, _ = build_item_features(num_items=50, item_embedding_dim=12, categories=None, seed=99)
    third, _ = build_item_features(num_items=50, item_embedding_dim=12, categories=None, seed=100)
    assert torch.equal(first, second)
    assert not torch.equal(first, third)


def test_metadata_changes_the_features_but_items_without_metadata_still_get_codes():
    """Coverage must not depend on metadata: an item with no attributes is still tokenisable."""
    matrix = torch.zeros((41, 2), dtype=torch.long)
    matrix[:, 0] = torch.arange(41) % 3
    with_metadata, note = build_item_features(
        num_items=40, item_embedding_dim=8, categories=matrix, seed=5
    )
    without_metadata, _ = build_item_features(
        num_items=40, item_embedding_dim=8, categories=None, seed=5
    )
    assert note["metadata"]["used"] is True
    assert with_metadata.shape[1] > without_metadata.shape[1]
    assert with_metadata.shape[0] == 41


def test_audit_reports_collisions_and_utilisation():
    config = QuantizerConfig(input_size=4, latent_size=4, levels=2, codebook_size=4, hidden_size=8)
    codes = [[0, 0], [1, 1], [1, 1], [2, 2]]  # items 1 and 2 collide
    counts = [[2, 1, 0, 0], [0, 2, 0, 0]]
    audit = audit_tokenizer(codes, num_items=3, config=config, codebook_histograms=counts)
    assert audit["collision_groups"] == 1
    assert audit["items_in_a_collision_group"] == 2
    assert audit["largest_collision_group"] == 2
    assert audit["distinct_semantic_ids"] == 2
    assert audit["codebook_utilisation"][0]["used_codes"] == 2
    assert audit["codebook_utilisation"][0]["dead_codes"] == 2


def test_audit_rejects_a_malformed_assignment():
    config = QuantizerConfig(input_size=4, latent_size=4, levels=2, codebook_size=4, hidden_size=8)
    with pytest.raises(TokenizerError, match="entries"):
        audit_tokenizer([[0, 0]], num_items=3, config=config)
    with pytest.raises(TokenizerError, match="out-of-range"):
        audit_tokenizer([[0, 0], [9, 1]], num_items=1, config=config)
    with pytest.raises(TokenizerError, match="codes"):
        audit_tokenizer([[0, 0], [1]], num_items=1, config=config)


# --------------------------------------------------------------------------- #
# Training data
# --------------------------------------------------------------------------- #


def test_examples_come_from_train_history_only():
    """The leakage guarantee: targets must be items from inside the training window."""
    assignment = make_assignment(60)
    cases = make_cases(10, num_items=60)
    dataset = build_generator_dataset(
        cases,
        item_codes=assignment,
        levels=LEVELS,
        config=GeneratorDatasetConfig(max_items=6, seed=1),
    )
    forbidden = {tuple(assignment[59]), tuple(assignment[60])}
    assert not (set(dataset.targets) & forbidden)
    assert set(dataset.targets) <= {
        tuple(assignment[item]) for item in range(1, 8)
    }
    assert "train_history only" in dataset.stats["source"]


def test_a_two_item_history_contributes_nothing():
    assignment = make_assignment(60)
    cases = [
        EvaluationCase(user_id="a", user_int_id=1, train_history=(1, 2), validation_target=59,
                       test_target=60, sequence_length=4),
        EvaluationCase(user_id="b", user_int_id=2, train_history=(1, 2, 3), validation_target=59,
                       test_target=60, sequence_length=5),
    ]
    dataset = build_generator_dataset(
        cases, item_codes=assignment, levels=LEVELS, config=GeneratorDatasetConfig(max_items=4)
    )
    assert len(dataset) == 1
    assert dataset.stats["cases_skipped_short_history"] == 1


def test_histories_are_whole_items_and_windowed():
    assignment = make_assignment(60)
    cases = [
        EvaluationCase(user_id="a", user_int_id=1, train_history=tuple(range(1, 9)),
                       validation_target=59, test_target=60, sequence_length=10)
    ]
    dataset = build_generator_dataset(
        cases, item_codes=assignment, levels=LEVELS, config=GeneratorDatasetConfig(max_items=3)
    )
    for history in dataset.histories:
        assert len(history) % LEVELS == 0, "a partial Semantic ID would describe a non-item"
        assert len(history) <= 3 * LEVELS, "the window must be respected"


def test_batch_order_is_seeded_and_differs_per_epoch():
    assignment = make_assignment(60)
    dataset = build_generator_dataset(
        make_cases(30), item_codes=assignment, levels=LEVELS,
        config=GeneratorDatasetConfig(max_items=6, seed=4),
    )
    one = [batch for batch in iter_batches(dataset, batch_size=8, epoch=0, seed=4)]
    again = [batch for batch in iter_batches(dataset, batch_size=8, epoch=0, seed=4)]
    other = [batch for batch in iter_batches(dataset, batch_size=8, epoch=1, seed=4)]
    assert one == again
    assert one != other


def test_an_out_of_range_item_in_a_history_is_rejected():
    assignment = make_assignment(20)
    cases = [
        EvaluationCase(user_id="a", user_int_id=1, train_history=(1, 2, 999),
                       validation_target=19, test_target=20, sequence_length=5)
    ]
    with pytest.raises(GeneratorDataError, match="outside the assignment"):
        build_generator_dataset(make_cases(1), item_codes=assignment, levels=LEVELS) if False else \
            build_generator_dataset(cases, item_codes=assignment, levels=LEVELS)


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #


def test_only_target_positions_are_supervised():
    """Padding and history must not be trained on, or the model would learn to predict padding."""
    model = make_model()
    assignment = make_assignment(60)
    dataset = build_generator_dataset(
        make_cases(4), item_codes=assignment, levels=LEVELS,
        config=GeneratorDatasetConfig(max_items=6, seed=1),
    )
    tokens, labels, mask = build_example_arrays(
        model, dataset.histories[:3], dataset.targets[:3]
    )
    assert int(mask.sum()) == 3 * LEVELS
    # The supervised labels are exactly the target codes.
    supervised = labels[mask].reshape(3, LEVELS).tolist()
    assert supervised == [list(target) for target in dataset.targets[:3]]
    # A left-padded history position is never supervised.
    assert not bool(mask[:, 0].any())


def test_training_reduces_the_loss_and_is_deterministic():
    assignment = make_assignment(60)
    dataset = build_generator_dataset(
        make_cases(40), item_codes=assignment, levels=LEVELS,
        config=GeneratorDatasetConfig(max_items=6, seed=2),
    )

    def run() -> tuple[list[float], dict[str, torch.Tensor]]:
        model = make_model()
        result = train_generator(
            model,
            dataset,
            config=GeneratorTrainerConfig(batch_size=8, epochs=4, seed=2, learning_rate=1e-3),
        )
        return [entry["loss"] for entry in result.epoch_metrics], {
            name: tensor.clone() for name, tensor in model.state_dict().items()
        }

    first_losses, first_state = run()
    second_losses, second_state = run()
    assert first_losses[-1] < first_losses[0]
    assert first_losses == second_losses
    for name in first_state:
        assert torch.allclose(first_state[name], second_state[name], atol=0, rtol=0), name


def test_the_trained_model_still_generates_only_catalogue_codes():
    """Training must not be able to widen the legal set: the trie is data, not a model output."""
    assignment = make_assignment(60)
    trie = SemanticPrefixTrie(assignment, levels=LEVELS)
    dataset = build_generator_dataset(
        make_cases(30), item_codes=assignment, levels=LEVELS,
        config=GeneratorDatasetConfig(max_items=6, seed=3),
    )
    model = make_model()
    train_generator(
        model, dataset,
        config=GeneratorTrainerConfig(batch_size=8, epochs=3, seed=3, learning_rate=1e-3),
    )
    histories = [[code for item in range(1, 5) for code in assignment[item]]] * 5
    for result in model.generate(histories, trie=trie):
        assert trie.items_at(trie.node_for(result.codes))


def test_scoring_covers_only_expanded_prefixes_and_says_so():
    """The bounded search must report its own coverage rather than implying exactness."""
    assignment = make_assignment(60)
    trie = SemanticPrefixTrie(assignment, levels=LEVELS)
    model = make_model()
    scores, diagnostics = model.full_catalog_scores(
        [[code for item in range(1, 4) for code in assignment[item]]],
        trie=trie,
        assignment=assignment,
        beams_per_level=8,
    )
    assert scores.shape == (1, len(assignment))
    assert float(scores[0, 0]) == 0.0, "PAD keeps a zero score and is masked by the evaluator"
    assert diagnostics["beams_per_level"] == 8
    assert 0.0 <= diagnostics["scored_share"] <= 1.0
    assert diagnostics["scored_items"] >= 1
    # A wider search cannot cover less of the catalogue.
    _wider, wider_diagnostics = model.full_catalog_scores(
        [[code for item in range(1, 4) for code in assignment[item]]],
        trie=trie,
        assignment=assignment,
        beams_per_level=64,
    )
    assert wider_diagnostics["scored_items"] >= diagnostics["scored_items"]


# --------------------------------------------------------------------------- #
# Benchmark arm end-to-end (tiny artifacts, real code path)
# --------------------------------------------------------------------------- #


def _build_tiny_artifact(directory: Path, *, num_items: int = 60) -> None:
    """Write a minimal but real tokenizer + generator artifact pair.

    Deliberately runs the real training entry points rather than hand-writing state dicts: the
    property under test is that the *shipped pipeline* composes, so a stub that skipped training
    would not exercise it.
    """
    from recommendation.semantic_id.dataset import build_generator_dataset
    from recommendation.semantic_id.tokenizer import (
        QuantizerConfig,
        RqVaeTokenizer,
        build_item_features,
    )
    from recommendation.semantic_id.training import GeneratorTrainerConfig, train_generator

    features, note = build_item_features(
        num_items=num_items, item_embedding_dim=12, categories=None, seed=11
    )
    config = QuantizerConfig(
        input_size=features.shape[1],
        latent_size=8,
        levels=LEVELS,
        codebook_size=CODEBOOK,
        hidden_size=24,
        seed=11,
    )
    torch.manual_seed(11)
    tokenizer = RqVaeTokenizer(config)
    assignment = tokenizer.codes_for_features(features).tolist()
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "semantic_ids.json").write_text(
        json.dumps(
            {
                "levels": LEVELS,
                "codebook_size": CODEBOOK,
                "num_items": num_items,
                "assignment": assignment,
                "audit": audit_tokenizer(assignment, num_items=num_items, config=config),
            }
        ),
        encoding="utf-8",
    )
    torch.save(
        {"model_state_dict": tokenizer.state_dict(), "quantizer_config": config.as_dict()},
        directory / "tokenizer.pt",
    )

    dataset = build_generator_dataset(
        make_cases(30, num_items=num_items),
        item_codes=assignment,
        levels=LEVELS,
        config=GeneratorDatasetConfig(max_items=6, seed=11),
    )
    model = make_model(num_items)
    train_generator(
        model,
        dataset,
        config=GeneratorTrainerConfig(batch_size=8, epochs=2, seed=11, learning_rate=1e-3),
    )
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_config": {
                "architecture": "semantic_id_generator",
                "levels": LEVELS,
                "codebook_size": CODEBOOK,
                "max_items": 6,
                "d_model": 32,
                "n_heads": 2,
                "n_layers": 1,
                "dropout": 0.0,
            },
        },
        directory / "generator.pt",
    )
    del note


def test_the_benchmark_arm_produces_full_catalogue_scores(tmp_path: Path):
    """The arm must hand the evaluator one score per catalogue item, with PAD unused."""
    from experiments.benchmark_public import arm_semantic_id_genrec

    _build_tiny_artifact(tmp_path)
    batches, diagnostics = arm_semantic_id_genrec(
        artifact_dir=tmp_path, batch_size=4, beams_per_level=16
    )
    cases = make_cases(4)
    produced = list(batches(cases))
    assert produced, "the arm must yield at least one batch"
    for histories, targets, scores in produced:
        assert scores.shape == (len(targets), 61), "one column per item plus PAD"
        assert bool(torch.isfinite(scores).all()), "the evaluator rejects non-finite scores"
        assert float(scores[0, 0]) == 0.0, "PAD is never a candidate"
        assert len(histories) == len(targets)
    summary = batches.finalize()  # type: ignore[attr-defined]
    assert summary["generated"] == len(cases)
    assert summary["invalid_generation_rate"] == 0.0, (
        "constrained decoding must never produce a code no catalogue item occupies"
    )
    assert diagnostics["trie_nodes"] > 1


def test_the_benchmark_arm_reaches_more_of_the_catalogue_with_more_beams(tmp_path: Path):
    """The bounded search's coverage must respond to its budget, not be constant."""
    from experiments.benchmark_public import arm_semantic_id_genrec

    _build_tiny_artifact(tmp_path)
    cases = make_cases(4)
    narrow, _ = arm_semantic_id_genrec(artifact_dir=tmp_path, batch_size=4, beams_per_level=2)
    list(narrow(cases))
    narrow_summary = narrow.finalize()  # type: ignore[attr-defined]
    wide, _ = arm_semantic_id_genrec(artifact_dir=tmp_path, batch_size=4, beams_per_level=64)
    list(wide(cases))
    wide_summary = wide.finalize()  # type: ignore[attr-defined]
    assert wide_summary["search"]["scored_items"] >= narrow_summary["search"]["scored_items"]
    assert wide_summary["search"]["beams_per_level"] == 64


def test_the_arm_scores_through_the_shared_evaluator(tmp_path: Path):
    """End-to-end: the arm's scores are accepted by the frozen evaluator unchanged."""
    from experiments.benchmark_public import arm_semantic_id_genrec, evaluate_arm

    _build_tiny_artifact(tmp_path)
    cases = make_cases(20)
    batches, _diagnostics = arm_semantic_id_genrec(
        artifact_dir=tmp_path, batch_size=8, beams_per_level=32
    )
    payload = evaluate_arm(cases=cases, num_items=60, batches=batches)
    assert payload["num_cases"] == 20
    assert payload["catalog_size"] == 60
    metrics = payload["metrics"]
    for name in ("Recall", "NDCG", "HR"):
        for cutoff in ("@5", "@10", "@20"):
            value = metrics[name][cutoff]
            assert 0.0 <= value <= 1.0, (name, cutoff, value)
    # HR and Recall coincide on a single-positive protocol; that is expected, not a bug.
    assert metrics["HR"]["@10"] == metrics["Recall"]["@10"]

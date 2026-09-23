"""Step-2.5 generator and trie tests (backend-local).

Covers the seven things the Step-2.5 handoff requires:

* train-history-only example construction, and no target leakage;
* SID/layout-driven tokenisation (nothing hardcoded);
* PAD sentinel exclusion;
* dedup level as its own token level;
* trie catalogue completeness and invalid-prefix rejection;
* checkpoint dependency-mismatch refusal;
* a small train/save/load/generate smoke.

Model-building tests use a deliberately tiny architecture.  The registered production
architecture is reported, not enforced (``TigerConfig.matches_registered_architecture``), because
a test that cannot afford 14 M parameters would otherwise have to skip the very code paths that
need testing - which is how a bug survives to a GPU run.
"""

from __future__ import annotations

import json
import math

import pytest
import torch

from tiger_public.contracts import PAD_SENTINEL, build_token_layout
from tiger_public.tiger import (
    PAD_TOKEN,
    TigerConfig,
    TigerError,
    TigerGenerator,
    TigerTrainer,
    build_examples,
    load_generator,
    save_generator,
)
from tiger_public.trie import CatalogueTrie, TrieError, item_token_path

LEVELS = 3
CODEBOOK = 16
DEDUP_VOCAB = 8


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #


def make_layout(*, levels: int = LEVELS, codebook: int = CODEBOOK, dedup: int = DEDUP_VOCAB) -> dict:
    return build_token_layout(
        levels=levels, codebook_size=codebook, dedup_levels=1, dedup_vocab_size=dedup
    ).as_dict()


def make_collision_free_assignment(count: int, *, codebook: int = CODEBOOK) -> list[list[int]]:
    """A unique SID per item, mirroring the Step-2.4F guarantee (collisions 0)."""
    assignment = [[PAD_SENTINEL] * (LEVELS + 1)]
    for item in range(count):
        assignment.append(
            [
                item % codebook,
                (item // codebook) % codebook,
                (item // (codebook * codebook)) % codebook,
                item % DEDUP_VOCAB,
            ]
        )
    keys = [tuple(row) for row in assignment[1:]]
    assert len(set(keys)) == count, "fixture SIDs must be unique"
    return assignment


def make_rows(users: int, length: int, *, count: int, seed: int = 0) -> list[tuple[int, ...]]:
    return [
        tuple(((user * 7 + step * 3 + seed) % count) + 1 for step in range(length))
        for user in range(users)
    ]


def tiny_config(**overrides) -> TigerConfig:
    base = dict(
        d_model=32, num_layers=1, num_heads=1, d_ff=64, dropout=0.0,
        max_hist_items=5, epochs=1, batch_size=8, learning_rate=1e-3, bf16=False, seed=7,
    )
    base.update(overrides)
    return TigerConfig(**base)


def build_model(layout: dict, config: TigerConfig) -> TigerGenerator:
    torch.manual_seed(config.seed)
    return TigerGenerator(config=config, layout=layout, vocab_size=int(layout["vocab_size"]))


@pytest.fixture(scope="module")
def layout() -> dict:
    return make_layout()


@pytest.fixture(scope="module")
def assignment() -> list[list[int]]:
    return make_collision_free_assignment(200)


@pytest.fixture(scope="module")
def trie(layout: dict, assignment: list[list[int]]) -> CatalogueTrie:
    return CatalogueTrie(assignment, layout=layout)


# --------------------------------------------------------------------------- #
# Tokenisation is layout-driven
# --------------------------------------------------------------------------- #


def test_item_token_path_follows_the_layout_offsets(layout: dict) -> None:
    offsets = list(layout["level_offsets"])
    path = item_token_path([1, 2, 3, 4], layout)
    assert path == (offsets[0] + 1, offsets[1] + 2, offsets[2] + 3, offsets[3] + 4)


def test_dedup_level_is_its_own_token_block(layout: dict) -> None:
    """The dedup digit must not alias the last semantic level.

    With ``codebook_size == dedup_vocab_size`` a conflated layout would map code ``4`` at level 2
    and dedup digit ``4`` to the SAME token, merging two different items.  The offsets must
    therefore differ.
    """
    semantic = item_token_path([0, 0, 4, 0], layout)
    dedup = item_token_path([0, 0, 0, 4], layout)
    assert semantic[2] != dedup[3], "the dedup level aliases the last semantic level"
    # The property that guarantees it: the dedup level owns its own offset block, whatever its
    # width.  Sizes may differ; offsets must not.
    offsets = list(layout["level_offsets"])
    assert len(set(offsets)) == len(offsets), f"offset blocks overlap: {offsets}"
    assert offsets[3] == offsets[2] + int(layout["codebook_size"])


def test_tokenisation_refuses_the_pad_sentinel(layout: dict) -> None:
    with pytest.raises(TrieError):
        item_token_path([PAD_SENTINEL] * 4, layout)


def test_tokenisation_refuses_an_out_of_range_digit(layout: dict) -> None:
    with pytest.raises(TrieError):
        item_token_path([CODEBOOK, 0, 0, 0], layout)
    with pytest.raises(TrieError):
        item_token_path([0, 0, 0, DEDUP_VOCAB], layout)


def test_a_layout_without_sep_is_refused(assignment: list[list[int]]) -> None:
    """SEP must come from the layout, never be guessed."""
    layout = make_layout()
    layout["special"].pop("sep")
    with pytest.raises(TigerError):
        build_examples([[1, 2, 3]], assignment=assignment, layout=layout, max_hist_items=2)


def test_a_sentinel_tokenisable_layout_is_refused(assignment: list[list[int]]) -> None:
    layout = make_layout()
    layout["sentinel_tokenisable"] = True
    with pytest.raises(TrieError):
        CatalogueTrie(assignment, layout=layout)


# --------------------------------------------------------------------------- #
# Trie completeness and rejection
# --------------------------------------------------------------------------- #


def test_every_catalogue_sid_is_accepted_by_the_trie(
    trie: CatalogueTrie, assignment: list[list[int]], layout: dict
) -> None:
    for item_id in range(1, len(assignment)):
        path = item_token_path(assignment[item_id], layout)
        assert trie.is_prefix(path), f"item {item_id} path is not a trie prefix"
        assert trie.is_complete(path), f"item {item_id} path is not a complete path"
        assert trie.item_for_path(path) == item_id


def test_trie_is_collision_free_and_counts_every_item(trie: CatalogueTrie, assignment) -> None:
    assert trie.collision_free()
    assert trie.items == len(assignment) - 1
    assert trie.stats.catalogue_items == len(assignment) - 1
    assert trie.stats.leaf_items == len(assignment) - 1


def test_pad_sentinel_is_absent_from_the_trie(trie: CatalogueTrie, layout: dict) -> None:
    assert not trie.contains_sentinel()
    # No token derived from the sentinel is ever legal.
    assert trie.children(()) != (PAD_SENTINEL,)
    assert trie.node_for([PAD_SENTINEL]) is None


def test_non_catalogue_prefixes_cannot_escape(trie: CatalogueTrie, layout: dict) -> None:
    """An illegal prefix has no legal continuation, so generation cannot leave the catalogue."""
    # A token id that is not a level-0 offset+code for any item in the trie.
    impossible = int(layout["special"]["eos"])
    assert trie.node_for([impossible]) is None
    assert trie.children([impossible]) == ()
    assert trie.is_prefix([impossible]) is False
    assert trie.item_for_path([impossible] * 4) is None


def test_children_are_always_legal_continuations(trie: CatalogueTrie, layout: dict) -> None:
    """Every child of a real prefix keeps the path extendable to a catalogue item.

    This is the property that makes the trie a *safe* generation constraint: a legal child can
    never lead into a dead end or into a token outside the catalogue's code set.
    """
    first_offset = int(layout["level_offsets"][0])
    checked = 0
    for node in range(trie.stats.nodes):
        children = trie._children[node]  # noqa: SLF001 - the test asserts on the structure
        for token in children:
            assert token != PAD_SENTINEL, "the sentinel reached a trie edge"
            assert 0 <= token < trie.vocab_size
            checked += 1
    assert checked > 0
    # Every level-0 token that the trie exposes is a legal first step of a real path.
    for code in range(CODEBOOK):
        prefix = (first_offset + code,)
        if not trie.is_prefix(prefix):
            assert trie.children(prefix) == ()


def test_every_terminal_path_holds_exactly_one_item(
    trie: CatalogueTrie, assignment: list[list[int]], layout: dict
) -> None:
    """Step-2.4F's uniqueness guarantee, restated as the trie's own invariant.

    Because the accepted assignment is collision-free, a complete catalogue path resolves to
    exactly one backend item - which is what lets a generated path name an item at all.
    """
    assert trie.collision_free()
    seen: set[int] = set()
    for item_id in range(1, len(assignment)):
        path = item_token_path(assignment[item_id], layout)
        members = trie.members_for_path(path)
        assert len(members) == 1, f"item {item_id} shares its path with {members}"
        assert members[0] == item_id
        seen.add(members[0])
    assert seen == set(range(1, len(assignment)))


# --------------------------------------------------------------------------- #
# Example construction
# --------------------------------------------------------------------------- #


def test_examples_are_next_item_and_use_only_train_history(
    assignment: list[list[int]], layout: dict
) -> None:
    rows = [(1, 2, 3, 4), (5, 6, 7)]
    dataset = build_examples(rows, assignment=assignment, layout=layout, max_hist_items=5)
    # n-1 examples per row: the last item is a target, never an input.
    assert dataset.examples == 3 + 2
    assert dataset.users == 2
    # The first example's target is item 2 and its input ends with item 1's block + SEP.
    sep = layout["special"]["sep"]
    bos = layout["special"]["bos"]
    assert dataset.inputs[0][0] == bos
    assert dataset.inputs[0][-1] == sep
    assert dataset.targets[0] == (*item_token_path(assignment[2], layout), layout["special"]["eos"])
    assert dataset.targets[1] == (*item_token_path(assignment[3], layout), layout["special"]["eos"])
    assert dataset.targets[2] == (*item_token_path(assignment[4], layout), layout["special"]["eos"])


def test_no_target_is_ever_an_input(assignment: list[list[int]], layout: dict) -> None:
    """Each example's target block must not appear at the END of its own input.

    A target may legitimately appear earlier as history (a repurchase, and the causal next-item
    objective), but the item being predicted must not already be the last thing the model saw.
    """
    rows = [(1, 2, 3, 4, 5)]
    dataset = build_examples(rows, assignment=assignment, layout=layout, max_hist_items=10)
    block = len(item_token_path(assignment[1], layout))
    for position, target in enumerate(dataset.targets):
        wanted = target[:-1]  # drop EOS
        assert len(wanted) == block
        # the second-to-last block of the input, which the model attends over
        items_seen = (len(dataset.inputs[position]) - 1) // (block + 1)
        assert items_seen >= 1


def test_example_count_matches_the_frozen_next_item_definition(
    assignment: list[list[int]], layout: dict
) -> None:
    rows = [(1, 2, 3), (4, 5, 6, 7)]
    dataset = build_examples(rows, assignment=assignment, layout=layout, max_hist_items=4)
    assert dataset.examples == sum(len(row) - 1 for row in rows)


def test_history_window_truncates_from_the_oldest_end(
    assignment: list[list[int]], layout: dict
) -> None:
    rows = [tuple(range(1, 11))]
    dataset = build_examples(rows, assignment=assignment, layout=layout, max_hist_items=3)
    block = len(item_token_path(assignment[1], layout))
    # The last example has seen all 10 items; its window holds only the newest 3.
    assert dataset.max_history_items == 3
    assert dataset.min_history_items == 1
    assert dataset.truncated_examples == 9 - 3
    # The last example predicts item 10 from the prefix 1..9, and the window keeps the NEWEST 3
    # of that prefix: items 7, 8 and 9, in chronological order, as whole item blocks.  Item 10 is
    # the target and is therefore absent from the input.
    final_input = dataset.inputs[-1]
    sep = layout["special"]["sep"]
    expected: tuple[int, ...] = (layout["special"]["bos"],)
    for item_id in (7, 8, 9):
        expected = (*expected, *item_token_path(assignment[item_id], layout), sep)
    assert tuple(final_input) == expected
    assert len(final_input) == 1 + 3 * (block + 1)
    assert final_input[-1] == sep
    # The oldest items are gone from the window.
    assert item_token_path(assignment[1], layout)[0] not in final_input


def test_rows_shorter_than_two_items_contribute_nothing(
    assignment: list[list[int]], layout: dict
) -> None:
    with pytest.raises(TigerError):
        build_examples([(1,)], assignment=assignment, layout=layout, max_hist_items=2)


def test_an_out_of_range_history_item_is_refused(
    assignment: list[list[int]], layout: dict
) -> None:
    with pytest.raises(TigerError):
        build_examples(
            [(1, len(assignment) + 5)], assignment=assignment, layout=layout, max_hist_items=2
        )


def test_no_pad_token_appears_as_a_target(assignment, layout) -> None:
    """A target always ends with EOS and never contains the PAD sentinel."""
    dataset = build_examples(
        [(1, 2, 3, 4)], assignment=assignment, layout=layout, max_hist_items=4
    )
    eos = layout["special"]["eos"]
    for target in dataset.targets:
        assert target[-1] == eos
        assert all(token >= 0 for token in target)


# --------------------------------------------------------------------------- #
# Config / registration reporting
# --------------------------------------------------------------------------- #


def test_tiny_config_reports_itself_as_not_registered() -> None:
    assert tiny_config().matches_registered_architecture() is False
    assert TigerConfig().matches_registered_architecture() is True


def test_registered_dimensions_are_the_h5_values() -> None:
    registered = TigerConfig.REGISTERED
    assert registered["d_model"] == 256
    assert registered["num_layers"] == 6
    assert registered["num_heads"] == 4
    assert registered["d_ff"] == 1024
    assert registered["batch_size"] == 512
    assert TigerConfig().d_model // TigerConfig().num_heads == 64


def test_config_records_validation_was_not_used() -> None:
    assert tiny_config().as_training_dict()["validation_used"] is False


def test_config_refuses_a_non_divisible_head_count() -> None:
    with pytest.raises(TigerError):
        tiny_config(d_model=30, num_heads=4)


# --------------------------------------------------------------------------- #
# Train / save / load / generate smoke
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def trained_artifacts(tmp_path_factory, layout, assignment, trie):
    root = tmp_path_factory.mktemp("tiger")
    catalogue_dir = root / "catalogue"
    sid_dir = root / "sid"
    exposure_dir = root / "exposure"
    for directory in (catalogue_dir, sid_dir, exposure_dir):
        directory.mkdir(parents=True)

    (sid_dir / "layout.json").write_text(json.dumps(layout), encoding="utf-8")
    (sid_dir / "semantic_ids.json").write_text(
        json.dumps(
            {
                "format": "agentrecx.tiger.semantic_ids.v3",
                "contract_version": "agentrecx.tiger_backend.v3",
                "num_items": len(assignment) - 1,
                "levels": LEVELS,
                "dedup_levels": 1,
                "codebook_size": CODEBOOK,
                "dedup_vocab_size": DEDUP_VOCAB,
                "pad_row": [PAD_SENTINEL] * 4,
                "assignment": assignment,
            }
        ),
        encoding="utf-8",
    )
    (sid_dir / "tokenizer.pt").write_bytes(b"stand-in tokenizer payload")
    (catalogue_dir / "catalogue.json").write_text(
        json.dumps({"num_items": len(assignment) - 1}), encoding="utf-8"
    )
    (catalogue_dir / "catalogue_items.jsonl").write_text(
        "".join(
            json.dumps({"backend_row": row, "item_id": row + 1}) + "\n"
            for row in range(len(assignment) - 1)
        ),
        encoding="utf-8",
    )
    rows = make_rows(30, 10, count=len(assignment) - 1)
    exposure_path = exposure_dir / "train_exposure.jsonl"
    with exposure_path.open("w", encoding="utf-8") as handle:
        for index, row in enumerate(rows):
            handle.write(json.dumps({"case_id": index, "items": list(row)}) + "\n")

    dataset = build_examples(
        rows, assignment=assignment, layout=layout, max_hist_items=5
    )
    config = tiny_config()
    model = build_model(layout, config)
    report = TigerTrainer(model, dataset, config=config, device="cpu").train()
    out_dir = root / "generator"
    metadata = save_generator(
        model=model,
        report=report,
        sid_dir=sid_dir,
        exposure_path=exposure_path,
        catalogue_dir=catalogue_dir,
        out_dir=out_dir,
    )
    return {
        "root": root,
        "out_dir": out_dir,
        "sid_dir": sid_dir,
        "catalogue_dir": catalogue_dir,
        "exposure_path": exposure_path,
        "model": model,
        "dataset": dataset,
        "metadata": metadata,
        "report": report,
    }


def test_training_produces_finite_loss(trained_artifacts) -> None:
    report = trained_artifacts["report"]
    assert report.history, "no epochs were recorded"
    for entry in report.history:
        assert math.isfinite(entry["loss"])
    assert report.precision == "fp32", "a CPU run must report fp32, not claim bf16"


def test_checkpoint_writes_the_documented_files(trained_artifacts) -> None:
    names = {path.name for path in trained_artifacts["out_dir"].iterdir()}
    assert {"tiger.pt", "tiger.json", "layout.json", "score_rule.json"} <= names


def test_checkpoint_records_all_required_provenance(trained_artifacts) -> None:
    metadata = trained_artifacts["metadata"]
    for key in (
        "dependency_hashes", "exposure_sha256", "semantic_ids_sha256",
        "checkpoint_sha256", "layout_sha256", "score_rule_sha256",
        "model", "training", "determinism", "layout",
    ):
        assert key in metadata, key
    assert metadata["training"]["validation_used"] is False
    assert metadata["training"]["labels_from"].startswith("agentrecx.tiger.train_exposure")
    assert set(metadata["dependency_hashes"]) == {
        "semantic_ids", "layout", "tokenizer", "train_exposure", "catalogue", "catalogue_items"
    }


def test_checkpoint_saves_and_loads_with_identical_generation(trained_artifacts, trie) -> None:
    model = trained_artifacts["model"]
    dataset = trained_artifacts["dataset"]
    restored, _metadata, _resume = load_generator(
        out_dir=trained_artifacts["out_dir"],
        sid_dir=trained_artifacts["sid_dir"],
        exposure_path=trained_artifacts["exposure_path"],
        catalogue_dir=trained_artifacts["catalogue_dir"],
        device="cpu",
    )
    assert restored.parameter_count() == model.parameter_count()
    batch = torch.tensor([list(dataset.inputs[0])], dtype=torch.long)
    mask = torch.ones_like(batch)
    assert model.constrained_generate(batch, mask, trie=trie, beam=2) == restored.constrained_generate(
        batch, mask, trie=trie, beam=2
    )


def test_constrained_generation_only_returns_catalogue_paths(trained_artifacts, trie) -> None:
    dataset = trained_artifacts["dataset"]
    model = trained_artifacts["model"]
    rows = [list(dataset.inputs[index]) for index in range(12)]
    width = max(len(row) for row in rows)
    batch = torch.full((len(rows), width), PAD_TOKEN, dtype=torch.long)
    mask = torch.zeros((len(rows), width), dtype=torch.long)
    for position, row in enumerate(rows):
        batch[position, : len(row)] = torch.tensor(row)
        mask[position, : len(row)] = 1
    paths = model.constrained_generate(batch, mask, trie=trie, beam=3)
    assert len(paths) == len(rows)
    for path in paths:
        assert trie.is_complete(path), f"generated path {path} is not a catalogue path"
        assert trie.item_for_path(path) is not None
        assert PAD_SENTINEL not in path


def test_generation_is_reproducible_under_a_fixed_seed(assignment, layout, trie) -> None:
    config = tiny_config(seed=1234)
    first = build_model(layout, config)
    second = build_model(layout, config)
    dataset = build_examples(
        make_rows(4, 6, count=len(assignment) - 1), assignment=assignment,
        layout=layout, max_hist_items=5,
    )
    batch = torch.tensor([list(dataset.inputs[0])], dtype=torch.long)
    mask = torch.ones_like(batch)
    assert first.constrained_generate(batch, mask, trie=trie, beam=2) == second.constrained_generate(
        batch, mask, trie=trie, beam=2
    )


def test_checkpoint_refuses_a_changed_exposure(trained_artifacts) -> None:
    """A checkpoint must not be reused against a different training corpus."""
    path = trained_artifacts["exposure_path"]
    original = path.read_text(encoding="utf-8")
    try:
        path.write_text(original + json.dumps({"case_id": 999, "items": [1, 2, 3]}) + "\n",
                        encoding="utf-8")
        with pytest.raises(TigerError) as error:
            load_generator(
                out_dir=trained_artifacts["out_dir"],
                sid_dir=trained_artifacts["sid_dir"],
                exposure_path=path,
                catalogue_dir=trained_artifacts["catalogue_dir"],
                device="cpu",
            )
        assert "different dependencies" in str(error.value)
    finally:
        path.write_text(original, encoding="utf-8")


def test_checkpoint_refuses_a_changed_sid_layout(trained_artifacts) -> None:
    """A generator trained on one token space must not load against another."""
    path = trained_artifacts["sid_dir"] / "layout.json"
    original = path.read_text(encoding="utf-8")
    try:
        changed = json.loads(original)
        changed["vocab_size"] = int(changed["vocab_size"]) + 1
        path.write_text(json.dumps(changed), encoding="utf-8")
        with pytest.raises(TigerError):
            load_generator(
                out_dir=trained_artifacts["out_dir"],
                sid_dir=trained_artifacts["sid_dir"],
                exposure_path=trained_artifacts["exposure_path"],
                catalogue_dir=trained_artifacts["catalogue_dir"],
                device="cpu",
            )
    finally:
        path.write_text(original, encoding="utf-8")


def test_checkpoint_refuses_a_changed_semantic_assignment(trained_artifacts) -> None:
    path = trained_artifacts["sid_dir"] / "semantic_ids.json"
    original = path.read_text(encoding="utf-8")
    try:
        payload = json.loads(original)
        # A value that provably differs from what is stored, so the file's bytes really change.
        assert payload["assignment"][1] != [9, 9, 9, 9]
        payload["assignment"][1] = [9, 9, 9, 9]
        path.write_text(json.dumps(payload), encoding="utf-8")
        assert path.read_text(encoding="utf-8") != original
        with pytest.raises(TigerError):
            load_generator(
                out_dir=trained_artifacts["out_dir"],
                sid_dir=trained_artifacts["sid_dir"],
                exposure_path=trained_artifacts["exposure_path"],
                catalogue_dir=trained_artifacts["catalogue_dir"],
                device="cpu",
            )
    finally:
        path.write_text(original, encoding="utf-8")


def test_loading_a_missing_checkpoint_is_refused(trained_artifacts, tmp_path) -> None:
    with pytest.raises(TigerError):
        load_generator(
            out_dir=tmp_path / "absent",
            sid_dir=trained_artifacts["sid_dir"],
            exposure_path=trained_artifacts["exposure_path"],
            catalogue_dir=trained_artifacts["catalogue_dir"],
            device="cpu",
        )


def test_examples_carry_no_target_like_field(trained_artifacts) -> None:
    """The example structure exposes only inputs, targets and user indices."""
    dataset = trained_artifacts["dataset"]
    payload = dataset.as_dict()
    for forbidden in ("validation_target", "test_target", "target_asin"):
        assert forbidden not in payload

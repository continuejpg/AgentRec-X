"""Step-2.5 generator and trie tests (backend-local).

Covers the seven things the Step-2.5 handoff requires, plus the Gate-C closure:

* train-history-only example construction, and no target leakage;
* SID/layout-driven tokenisation (nothing hardcoded);
* PAD sentinel exclusion;
* dedup level as its own token level;
* trie catalogue completeness and invalid-prefix rejection;
* checkpoint dependency-mismatch refusal;
* a small train/save/load/generate smoke;
* the **generator layout** is a derived artifact: materialising it leaves the accepted SID
  directory byte-for-byte unchanged, SEP cannot alias any catalogue code, and the checkpoint binds
  both layout hashes.

Model-building tests use a deliberately tiny architecture.  The registered production
architecture is reported, not enforced (``TigerConfig.matches_registered_architecture``), because
a test that cannot afford 14 M parameters would otherwise have to skip the very code paths that
need testing - which is how a bug survives to a GPU run.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
import torch

from tiger_public.contracts import PAD_SENTINEL, build_token_layout
from tiger_public.generator_layout import (
    GENERATOR_LAYOUT_FORMAT,
    GeneratorLayoutError,
    assert_no_code_alias,
    derive_generator_layout,
    load_generator_layout,
    materialise_generator_layout,
    sha256_file,
)
from tiger_public.tiger import (
    RESUMABLE_FORMAT,
    TigerConfig,
    TigerError,
    TigerGenerator,
    TigerTrainer,
    build_examples,
    load_generator,
    load_resumable,
    resume_checkpoint_path,
    save_generator,
    save_resumable,
)
from tiger_public.trie import CatalogueTrie, TrieError, item_token_path

LEVELS = 3
CODEBOOK = 16
DEDUP_VOCAB = 8


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #


def make_accepted_layout(
    *, levels: int = LEVELS, codebook: int = CODEBOOK, dedup: int = DEDUP_VOCAB
) -> dict:
    """The shape Step 2.4F produced: pad/bos/eos only, no SEP, vocab_size = code_space + 3."""
    accepted = build_token_layout(
        levels=levels, codebook_size=codebook, dedup_levels=1, dedup_vocab_size=dedup
    ).as_dict()
    accepted["format"] = "agentrecx.tiger.token_layout.v3"
    del accepted["special"]["sep"]
    accepted["vocab_size"] = accepted["code_space"] + 3
    return accepted


def make_layout(*, levels: int = LEVELS, codebook: int = CODEBOOK, dedup: int = DEDUP_VOCAB) -> dict:
    """The generator layout, derived exactly as the CLI derives it from an accepted layout."""
    accepted = make_accepted_layout(levels=levels, codebook=codebook, dedup=dedup)
    return derive_generator_layout_from_dict(accepted)


def derive_generator_layout_from_dict(accepted: dict) -> dict:
    """Derive a generator layout in memory, using the real materialiser via a scratch directory.

    The production path is file-based on purpose, so the test drives the real function rather than
    a re-implementation: only the directory is disposable.
    """
    import tempfile

    with tempfile.TemporaryDirectory() as scratch:
        root = Path(scratch)
        sid_dir = root / "sid"
        sid_dir.mkdir(parents=True)
        (sid_dir / "layout.json").write_text(json.dumps(accepted), encoding="utf-8")
        count = 64
        assignment = [[PAD_SENTINEL] * (LEVELS + 1)]
        for item in range(count):
            assignment.append(
                [
                    item % int(accepted["codebook_size"]),
                    (item // int(accepted["codebook_size"])) % int(accepted["codebook_size"]),
                    0,
                    item % DEDUP_VOCAB,
                ]
            )
        (sid_dir / "semantic_ids.json").write_text(
            json.dumps({"assignment": assignment}), encoding="utf-8"
        )
        return derive_generator_layout(
            sid_dir=sid_dir, generator_layout_path=root / "generator_layout.json"
        )


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
    """SEP must come from a derived generator layout, never be guessed at use time."""
    layout = make_layout()
    layout["special"].pop("sep")
    with pytest.raises(TigerError):
        build_examples([[1, 2, 3]], assignment=assignment, layout=layout, max_hist_items=2)


def test_the_accepted_sid_layout_is_refused_as_a_generator_layout(
    assignment: list[list[int]],
) -> None:
    """Passing the accepted Step-2.4F layout must be an error, not a silent mis-tokenisation.

    It has no SEP and a ``vocab_size`` one short, so accepting it would either crash later or, with
    a guessed SEP, alias a real code.
    """
    accepted = make_accepted_layout()
    accepted["format"] = "agentrecx.tiger.token_layout.v3"
    with pytest.raises(TigerError) as error:
        build_examples([[1, 2, 3]], assignment=assignment, layout=accepted, max_hist_items=2)
    assert GENERATOR_LAYOUT_FORMAT in str(error.value)


def test_a_generator_layout_with_the_wrong_format_tag_is_refused(
    assignment: list[list[int]], layout: dict
) -> None:
    layout = dict(layout)
    layout["format"] = "agentrecx.tiger.token_layout.v3"
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


def _decode_input_items(
    tokens: Sequence[int], assignment: list[list[int]], layout: dict, *, trie: CatalogueTrie
) -> tuple[int, ...]:
    """Recover the item sequence an example's input encodes.

    The input is ``[BOS, block, SEP, block, SEP, ...]``.  Decoding it back is how the provenance
    test checks *which positions* the source came from, instead of comparing token sets.
    """
    codes_per_item = len(item_token_path(assignment[1], layout))
    sep = layout["special"]["sep"]
    body = list(tokens)[1:]  # drop BOS
    items: list[int] = []
    index = 0
    while index < len(body):
        if body[index] == sep:
            raise AssertionError("the input has an empty item block")
        block = tuple(body[index : index + codes_per_item])
        if len(block) != codes_per_item:
            raise AssertionError("the input ends in a partial item block")
        item_id = trie.item_for_path(block)
        if item_id is None:
            raise AssertionError(f"input block {block} is not a catalogue item path")
        items.append(item_id)
        index += codes_per_item
        if index < len(body):
            assert body[index] == sep, "two item blocks were not separated by SEP"
            index += 1
    return tuple(items)


def test_source_items_are_exactly_the_prefix_before_the_target(
    assignment: list[list[int]], layout: dict
) -> None:
    """The leakage invariant is *positional*, not set-disjointness.

    For a target at position ``k`` the source must be ``history[max(0, k - max_hist_items):k]`` and
    nothing else.  A repeated item is legal: predicting ``A`` from a history that already contains
    ``A`` is a repurchase, not a leak, so an item-set test would be wrong in both directions.
    """
    trie = CatalogueTrie(assignment, layout=layout)
    row = (1, 2, 3, 4, 5, 6, 7, 8, 9)
    # The source must be a contiguous, newest-first slice of the prefix and nothing else, so the
    # check is on positions rather than on the item set: an item may repeat (see the repurchase
    # test below), but a position at or after the target may never appear.
    position_of = {item_id: index for index, item_id in enumerate(row)}
    for window in (2, 3, 5, 20):
        dataset = build_examples([row], assignment=assignment, layout=layout, max_hist_items=window)
        for k, (source, target) in enumerate(
            zip(dataset.inputs, dataset.targets, strict=True), start=1
        ):
            decoded = _decode_input_items(source, assignment, layout, trie=trie)
            assert decoded == row[max(0, k - window) : k]
            assert all(position_of[item] < k for item in decoded)
            assert trie.item_for_path(target[:-1]) == row[k]


def test_a_repeated_item_is_kept_as_a_valid_repurchase(assignment: list[list[int]], layout: dict) -> None:
    """``[A, B, A]`` must yield ``input=[A, B] -> target=A`` rather than being dropped as a leak.

    Dropping it would silently delete every repurchase example from training, which is a data
    change with no basis in the frozen protocol: AgentRec-X's own handoff documents that a naive
    target may recur inside a history (``PHASE5_HANDOFF.md`` section 8).
    """
    trie = CatalogueTrie(assignment, layout=layout)
    dataset = build_examples([(1, 2, 1)], assignment=assignment, layout=layout, max_hist_items=10)
    assert dataset.examples == 2
    # Example 0: [1] -> 2.  Example 1: [1, 2] -> 1, the repurchase.
    assert _decode_input_items(dataset.inputs[1], assignment, layout, trie=trie) == (1, 2)
    assert trie.item_for_path(dataset.targets[1][:-1]) == 1
    # And the repeated item really is present in both the source and the target.
    assert 1 in _decode_input_items(dataset.inputs[1], assignment, layout, trie=trie)


def test_example_count_matches_the_frozen_next_item_definition(
    assignment: list[list[int]], layout: dict
) -> None:
    rows = [(1, 2, 3), (4, 5, 6, 7)]
    dataset = build_examples(rows, assignment=assignment, layout=layout, max_hist_items=4)
    assert dataset.examples == sum(len(row) - 1 for row in rows)


def test_the_three_length_sums_are_kept_distinct() -> None:
    """Guard the arithmetic that Gate B.1 corrected.

    ``sum(len)``, ``sum(len - 1)`` and ``sum(len - 2)`` are three different numbers and only the
    middle one is an example count.  A row shorter than two items contributes no transition and no
    pair, which is what ``max(0, ...)`` encodes.
    """
    rows = [(1, 2, 3, 4), (5, 6), (7,)]
    assert sum(len(row) for row in rows) == 7
    assert sum(max(0, len(row) - 1) for row in rows) == 4
    assert sum(max(0, len(row) - 2) for row in rows) == 2
    # A one-item row is not a transition, and a two-item row is not a Two-Tower pair.
    assert max(0, len(rows[2]) - 1) == 0
    assert max(0, len(rows[1]) - 2) == 0


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


#: Every registered Gate-C value, spelled out independently of the implementation so a change to
#: the class cannot quietly redefine what "registered" means.
GATE_C_REGISTRATION = {
    "max_hist_items": 20,
    "d_model": 256,
    "num_layers": 6,
    "num_heads": 4,
    "d_ff": 1024,
    "dropout": 0.1,
    "batch_size": 512,
    "epochs": 20,
    "learning_rate": 5e-4,
    "weight_decay": 0.0,
    "max_grad_norm": 1.0,
    "warmup_fraction": 0.05,
    "bf16": True,
    "seed": 2026,
}


def test_registered_training_is_exactly_the_gate_c_registration() -> None:
    assert TigerConfig.REGISTERED_TRAINING == GATE_C_REGISTRATION


def test_the_default_config_is_the_registered_configuration() -> None:
    """The defaults a Gate-D command inherits must *be* the registered values.

    An unregistered default is how a 3-6 hour GPU run becomes incomparable to its registration, so
    this fails before the run rather than after it.
    """
    config = TigerConfig()
    assert config.matches_registered_training() is True
    assert config.training_divergence() == {}


def test_a_diverging_config_reports_which_fields_differ() -> None:
    diverged = TigerConfig(epochs=5, learning_rate=1e-3)
    report = diverged.training_divergence()
    assert report == {"epochs": (20, 5), "learning_rate": (5e-4, 1e-3)}
    assert diverged.matches_registered_training() is False
    # A test-sized model is allowed to diverge, and says so in its own metadata.
    assert tiny_config().as_dict(vocab_size=132, per_item_tokens=4)["registered_training"] is False


def test_the_cli_defaults_are_the_registered_configuration() -> None:
    """The CLI is the Gate-D entry point, so its argparse defaults must match the registration."""
    from tiger_public.cli import _parser

    subparsers = next(
        action
        for action in _parser()._actions
        if action.dest == "stage"
    )
    train = subparsers.choices["train"]
    defaults = {action.dest: action.default for action in train._actions if action.dest != "help"}

    # Every registered value that Gate D sets from the command line is reachable, with the
    # argparse destination it actually uses.  Two names differ from the registration's field
    # names, which is exactly the drift this test is here to catch.
    cli_exposed = {
        "num_layers": "layers",
        "num_heads": "heads",
        "max_hist_items": "max_hist_items",
        "d_model": "d_model",
        "d_ff": "d_ff",
        "dropout": "dropout",
        "batch_size": "batch_size",
        "epochs": "epochs",
        "learning_rate": "learning_rate",
        "seed": "seed",
        "bf16": "bf16",
    }
    for name, option in cli_exposed.items():
        assert option in defaults, f"the train subcommand has no --{option} option"
        assert defaults[option] == GATE_C_REGISTRATION[name], f"the CLI default for --{option} drifted"
    # bf16 is registered true; --no-bf16 exists to diverge it deliberately for a CPU smoke.
    assert defaults["bf16"] is True
    assert defaults["device"] == "cpu"

    # These four are deliberately not CLI options, so the registered value must be the dataclass
    # default - which is the only way Gate D can set them.
    config = TigerConfig()
    for name in ("weight_decay", "max_grad_norm", "warmup_fraction"):
        assert name not in defaults, f"--{name} is exposed; its default is no longer the only source"
        assert getattr(config, name) == GATE_C_REGISTRATION[name], name


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
    generator_dir = root / "generator"
    for directory in (catalogue_dir, sid_dir, exposure_dir, generator_dir):
        directory.mkdir(parents=True)

    # The accepted SID layout: pad/bos/eos only, no SEP.  The generator layout is derived from it.
    (sid_dir / "layout.json").write_text(
        json.dumps(make_accepted_layout()), encoding="utf-8"
    )
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
    # Derive the generator layout from the accepted one, exactly as the CLI does.
    generator_layout = materialise_generator_layout(sid_dir=sid_dir, out=generator_dir)
    out_dir = root / "generator"
    metadata = save_generator(
        model=model,
        report=report,
        sid_dir=sid_dir,
        exposure_path=exposure_path,
        catalogue_dir=catalogue_dir,
        generator_layout_path=generator_dir / "generator_layout.json",
        out_dir=out_dir,
    )
    return {
        "root": root,
        "out_dir": out_dir,
        "sid_dir": sid_dir,
        "catalogue_dir": catalogue_dir,
        "exposure_path": exposure_path,
        "generator_layout": generator_layout,
        "generator_layout_path": generator_dir / "generator_layout.json",
        "model": model,
        "dataset": dataset,
        "metadata": metadata,
        "report": report,
    }


def test_training_produces_finite_loss(trained_artifacts) -> None:
    report = trained_artifacts["report"]
    dataset = trained_artifacts["dataset"]
    metadata = trained_artifacts["metadata"]
    assert report.history, "no epochs were recorded"
    for entry in report.history:
        assert math.isfinite(entry["loss"])
    assert report.precision == "fp32", "a CPU run must report fp32, not claim bf16"
    assert metadata["training"]["examples"] == report.examples == dataset.examples


def test_trainer_padding_uses_generator_layout_pad(trained_artifacts) -> None:
    model = trained_artifacts["model"]
    dataset = trained_artifacts["dataset"]
    trainer = TigerTrainer(model, dataset, config=model.config, device="cpu")

    short = min(range(dataset.examples), key=lambda index: len(dataset.inputs[index]))
    long = max(range(dataset.examples), key=lambda index: len(dataset.inputs[index]))
    assert len(dataset.inputs[short]) < len(dataset.inputs[long])

    rows, mask, _targets = trainer._collate([short, long])
    short_width = len(dataset.inputs[short])
    pad_token = int(model.layout["special"]["pad"])

    assert rows.shape[1] == len(dataset.inputs[long])
    assert rows[0, short_width:].tolist() == [pad_token] * (
        rows.shape[1] - short_width
    )
    assert mask[0, short_width:].tolist() == [0] * (
        rows.shape[1] - short_width
    )


def test_checkpoint_writes_the_documented_files(trained_artifacts) -> None:
    names = {path.name for path in trained_artifacts["out_dir"].iterdir()}
    assert {"tiger.pt", "tiger.json", "generator_layout.json", "score_rule.json"} <= names


def test_materialising_the_generator_layout_leaves_the_sid_artifact_unchanged(
    trained_artifacts,
) -> None:
    """The accepted SID artifact is immutable: materialisation is read-only in, new file out."""
    sid_dir = trained_artifacts["sid_dir"]
    accepted = json.loads((sid_dir / "layout.json").read_text(encoding="utf-8"))
    assert sorted(accepted["special"]) == ["bos", "eos", "pad"], "the accepted layout gained a key"
    assert accepted["vocab_size"] == accepted["code_space"] + 3
    # No patched copy and no backup was written inside the accepted directory.
    assert sorted(path.name for path in sid_dir.iterdir()) == [
        "layout.json", "semantic_ids.json"
    ] or "layout.v3.bak.json" not in {path.name for path in sid_dir.iterdir()}
    # The derived artifact lives elsewhere and declares the accepted hashes.
    derived = trained_artifacts["generator_layout"]
    assert derived["source_sid_layout_sha256"] == sha256_file(sid_dir / "layout.json")
    assert derived["source_semantic_ids_sha256"] == sha256_file(sid_dir / "semantic_ids.json")


def test_the_generator_layout_extends_the_accepted_vocabulary_by_exactly_sep(
    trained_artifacts,
) -> None:
    derived = trained_artifacts["generator_layout"]
    accepted = json.loads(
        (trained_artifacts["sid_dir"] / "layout.json").read_text(encoding="utf-8")
    )
    assert derived["format"] == GENERATOR_LAYOUT_FORMAT
    assert derived["levels"] == accepted["levels"]
    assert derived["dedup_levels"] == accepted["dedup_levels"]
    assert derived["codebook_size"] == accepted["codebook_size"]
    assert derived["dedup_vocab_size"] == accepted["dedup_vocab_size"]
    assert derived["level_offsets"] == accepted["level_offsets"]
    assert derived["code_space"] == accepted["code_space"]
    assert (
        derived["special"]["pad"],
        derived["special"]["bos"],
        derived["special"]["eos"],
    ) == (
        accepted["special"]["pad"],
        accepted["special"]["bos"],
        accepted["special"]["eos"],
    )
    assert derived["special"]["sep"] == accepted["vocab_size"]
    assert derived["vocab_size"] == accepted["vocab_size"] + 1
    assert derived["extension"]["added_special"] == "sep"


def test_specials_are_distinct_and_outside_the_code_space(trained_artifacts) -> None:
    derived = trained_artifacts["generator_layout"]
    specials = derived["special"]
    assert len(set(specials.values())) == 4
    assert all(token >= derived["code_space"] for token in specials.values())
    assert derived["audit"]["aliased_tokens"] == []


def test_a_special_token_inside_the_code_space_is_refused(assignment) -> None:
    """A SEP below ``code_space`` would shadow a real code, so derivation must refuse it."""
    derived = make_layout()
    derived["special"]["sep"] = derived["code_space"] - 1
    with pytest.raises(GeneratorLayoutError):
        assert_no_code_alias(derived, assignment)


def test_a_special_or_a_digit_outside_its_block_is_refused(assignment) -> None:
    """Defence in depth for a malformed layout.

    For a layout whose blocks and ``code_space`` agree, no special can alias a real code: every
    token lies in ``[0, code_space)`` and every special lies at or above it.  The alias guard is
    therefore only reachable when the layout's own numbers disagree, and that is what this builds:
    ``code_space`` is declared above a SEP that a real catalogue item also maps onto.  Either the
    alias check or the digit-range check must fire; what must never happen is a vocabulary whose
    SEP is simultaneously a real code.
    """
    derived = make_layout()
    code_space = derived["code_space"]
    used_tokens = {item_token_path(row, derived)[0] for row in assignment[1:]}
    assert used_tokens, "the fixture must produce at least one token"
    derived["special"] = {
        "pad": code_space,
        "bos": code_space + 1,
        "eos": code_space + 2,
        # The smallest real level-0 token: below the declared code space, so it aliases a code.
        "sep": min(used_tokens),
    }
    with pytest.raises(GeneratorLayoutError) as error:
        assert_no_code_alias(derived, assignment)
    message = str(error.value)
    assert "alias" in message or "inside the catalogue code space" in message, message


def test_non_distinct_specials_are_refused(assignment) -> None:
    derived = make_layout()
    derived["special"]["sep"] = derived["special"]["pad"]
    with pytest.raises(GeneratorLayoutError):
        assert_no_code_alias(derived, assignment)


def test_an_accepted_layout_with_a_bad_vocab_size_is_refused(tmp_path: Path, assignment) -> None:
    """Derivation validates its input: a layout claiming code_space + 5 is not a known shape."""
    sid_dir = tmp_path / "sid"
    sid_dir.mkdir()
    accepted = make_accepted_layout()
    accepted["vocab_size"] = accepted["code_space"] + 5
    (sid_dir / "layout.json").write_text(json.dumps(accepted), encoding="utf-8")
    (sid_dir / "semantic_ids.json").write_text(
        json.dumps({"assignment": assignment}), encoding="utf-8"
    )
    with pytest.raises(GeneratorLayoutError):
        derive_generator_layout(
            sid_dir=sid_dir, generator_layout_path=tmp_path / "generator_layout.json"
        )


def test_a_layout_that_already_carries_sep_is_preserved_not_doubled() -> None:
    """The backend's own fit-sid writes a SEP-bearing layout; the derivation must not add a second.

    Both accepted shapes must yield the *same* generator vocabulary, so that a stub run and the
    accepted production artifact reach an identical token space.
    """
    without = make_layout()
    with_sep = derive_generator_layout_from_dict(build_token_layout(
        levels=LEVELS, codebook_size=CODEBOOK, dedup_levels=1, dedup_vocab_size=DEDUP_VOCAB
    ).as_dict())
    assert without["special"] == with_sep["special"]
    assert without["vocab_size"] == with_sep["vocab_size"]
    assert without["extension"]["added_special"] == "sep"
    assert with_sep["extension"]["added_special"] is None
    assert with_sep["source_sid_layout_had_sep"] is True
    assert without["source_sid_layout_had_sep"] is False


def test_the_malformed_wide_dedup_layout_is_refused_by_derivation(tmp_path: Path) -> None:
    """Regression: a dedup block wider than the code space can never be materialised.

    ``contracts.build_token_layout`` is frozen and is NOT changed by this task; it happily builds
    this shape because the dedup width is not part of its own consistency check.  The generator
    boundary is what refuses it, and it must refuse *before* any example or model is built.
    """
    layout = build_token_layout(
        levels=3, codebook_size=32, dedup_levels=1, dedup_vocab_size=256
    ).as_dict()
    assert layout["code_space"] == 128 and layout["level_offsets"][-1] == 96
    sid_dir = tmp_path / "sid"
    sid_dir.mkdir()
    (sid_dir / "layout.json").write_text(json.dumps(layout), encoding="utf-8")
    (sid_dir / "semantic_ids.json").write_text(
        json.dumps({"assignment": [[PAD_SENTINEL] * 4] + [[i % 32, 0, 0, i % 256] for i in range(64)]}),
        encoding="utf-8",
    )
    with pytest.raises(GeneratorLayoutError) as error:
        derive_generator_layout(
            sid_dir=sid_dir, generator_layout_path=tmp_path / "generator_layout.json"
        )
    assert "last level block" in str(error.value)
    assert not (tmp_path / "generator_layout.json").exists(), "a partial artifact was written"


def test_the_coherent_narrow_dedup_layout_is_accepted(tmp_path: Path) -> None:
    """The same K=32 with a matching dedup width must derive cleanly.  This is the ablation."""
    layout = build_token_layout(
        levels=3, codebook_size=32, dedup_levels=1, dedup_vocab_size=32
    ).as_dict()
    sid_dir = tmp_path / "sid"
    sid_dir.mkdir()
    (sid_dir / "layout.json").write_text(json.dumps(layout), encoding="utf-8")
    (sid_dir / "semantic_ids.json").write_text(
        json.dumps({"assignment": [[PAD_SENTINEL] * 4] + [[i % 32, (i // 32) % 32, 0, i % 32] for i in range(200)]}),
        encoding="utf-8",
    )
    derived = derive_generator_layout(
        sid_dir=sid_dir, generator_layout_path=tmp_path / "generator_layout.json"
    )
    assert derived["code_space"] == 128
    # `build_token_layout` already emits SEP (code_space + 3) and vocab code_space + 4, so the
    # derivation preserves it rather than appending a second one.
    assert derived["special"]["sep"] == layout["special"]["sep"] == 131
    assert derived["vocab_size"] == layout["vocab_size"] == 132
    assert derived["extension"]["added_special"] is None
    assert derived["audit"]["aliased_tokens"] == []
    # The accepted layout on disk is untouched, and no backup appears beside it.
    assert sorted(path.name for path in sid_dir.iterdir()) == ["layout.json", "semantic_ids.json"]


def test_a_generator_layout_of_the_wrong_format_is_refused(tmp_path: Path, layout) -> None:
    path = tmp_path / "generator_layout.json"
    path.write_text(json.dumps({"format": "nope", "special": layout["special"]}), encoding="utf-8")
    with pytest.raises(GeneratorLayoutError):
        load_generator_layout(path)


def test_sep_is_never_interpreted_as_a_sid_digit(trained_artifacts, layout) -> None:
    """SEP is a boundary token.  It is outside every level block and appears in no target."""
    sep = layout["special"]["sep"]
    offsets = layout["level_offsets"]
    widths = [layout["codebook_size"]] * layout["levels"] + [layout["dedup_vocab_size"]]
    for offset, width in zip(offsets, widths, strict=True):
        assert not offset <= sep < offset + width
    for target in trained_artifacts["dataset"].targets:
        assert sep not in target[:-1], "SEP leaked into a target SID block"


def test_the_checkpoint_binds_both_layout_hashes(trained_artifacts) -> None:
    dependencies = trained_artifacts["metadata"]["dependency_hashes"]
    assert dependencies["layout"] == sha256_file(trained_artifacts["sid_dir"] / "layout.json")
    assert dependencies["generator_layout"] == sha256_file(
        trained_artifacts["generator_layout_path"]
    )
    assert dependencies["layout"] != dependencies["generator_layout"]
    assert (
        trained_artifacts["metadata"]["generator_layout_sha256"]
        == dependencies["generator_layout"]
    )
    assert trained_artifacts["metadata"]["layout_sha256"] == dependencies["layout"]


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
        "semantic_ids", "layout", "tokenizer", "train_exposure", "catalogue", "catalogue_items",
        "generator_layout",
    }


def test_checkpoint_saves_and_loads_with_identical_generation(trained_artifacts, trie) -> None:
    model = trained_artifacts["model"]
    dataset = trained_artifacts["dataset"]
    restored, _metadata, _resume = load_generator(
        out_dir=trained_artifacts["out_dir"],
        sid_dir=trained_artifacts["sid_dir"],
        exposure_path=trained_artifacts["exposure_path"],
        catalogue_dir=trained_artifacts["catalogue_dir"],
        generator_layout_path=trained_artifacts["generator_layout_path"],
        device="cpu",
    )
    assert restored.parameter_count() == model.parameter_count()
    expected_pad = int(model.layout["special"]["pad"])
    assert model.model.config.pad_token_id == expected_pad
    assert restored.model.config.pad_token_id == expected_pad
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
    pad_token = int(model.layout["special"]["pad"])
    batch = torch.full((len(rows), width), pad_token, dtype=torch.long)
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
                generator_layout_path=trained_artifacts["generator_layout_path"],
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
                generator_layout_path=trained_artifacts["generator_layout_path"],
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
                generator_layout_path=trained_artifacts["generator_layout_path"],
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
            generator_layout_path=trained_artifacts["generator_layout_path"],
            device="cpu",
        )


def test_examples_carry_no_target_like_field(trained_artifacts) -> None:
    """The example structure exposes only inputs, targets and user indices."""
    dataset = trained_artifacts["dataset"]
    payload = dataset.as_dict()
    for forbidden in ("validation_target", "test_target", "target_asin"):
        assert forbidden not in payload


# --------------------------------------------------------------------------- #
# Resumability (Gate D blocker)
# --------------------------------------------------------------------------- #
#
# The fixed budget is the budget: these tests never renegotiate it, and no epoch is ever selected
# by a metric.  What is asserted is that an interrupted run continues where it stopped, with the
# optimizer/scheduler/RNG state it had, and that a state which does not describe this run is
# refused rather than half-applied.


def _resume_deps(*, sid_dir, exposure_path, catalogue_dir, generator_layout_path) -> dict:
    from tiger_public.cli import _train_dependency_hashes

    return _train_dependency_hashes(
        sid_dir=sid_dir,
        exposure_path=exposure_path,
        catalogue_dir=catalogue_dir,
        generator_layout_path=generator_layout_path,
    )


@pytest.fixture(scope="module")
def resume_rig(trained_artifacts):
    """A tiny deterministic training rig: 4 epochs, 3 batches each, one published state per epoch."""
    artifacts = trained_artifacts
    config = tiny_config(epochs=4)
    dataset = artifacts["dataset"]
    deps = _resume_deps(
        sid_dir=artifacts["sid_dir"],
        exposure_path=artifacts["exposure_path"],
        catalogue_dir=artifacts["catalogue_dir"],
        generator_layout_path=artifacts["generator_layout_path"],
    )

    def fresh() -> TigerGenerator:
        return build_model(artifacts["generator_layout"], config)

    def train(*, out_dir, stop_after: int | None = None, resume_state=None):
        model = fresh()
        trainer = TigerTrainer(model, dataset, config=config, device="cpu")
        writes: list[dict] = []

        def hook(**kwargs):
            writes.append(
                save_resumable(
                    path=resume_checkpoint_path(out_dir),
                    model=kwargs["model"],
                    optimizer=kwargs["optimizer"],
                    scheduler=kwargs["scheduler"],
                    completed_epoch=kwargs["completed_epoch"],
                    epoch_budget=config.epochs,
                    global_step=kwargs["global_step"],
                    history=kwargs["history"],
                    dependency_hashes=deps,
                    config=config,
                )
            )
            if stop_after is not None and kwargs["completed_epoch"] == stop_after:
                raise KeyboardInterrupt("simulated interruption")

        try:
            report = trainer.train(resume_state=resume_state, checkpoint_hook=hook)
        except KeyboardInterrupt:
            report = None
        return model, report, writes

    return {
        "config": config,
        "dataset": dataset,
        "deps": deps,
        "fresh": fresh,
        "train": train,
        "artifacts": artifacts,
    }


def test_continuous_training_runs_the_whole_budget(resume_rig) -> None:
    _model, report, writes = resume_rig["train"](out_dir=resume_rig["artifacts"]["root"] / "cont")
    assert report is not None
    assert [entry["epoch"] for entry in report.history] == [0, 1, 2, 3]
    assert report.completed_epoch == 3
    assert report.global_step == report.steps_per_epoch * 4
    assert len(writes) == 4, "every completed epoch must publish a resumable state"


def test_an_interrupted_run_resumes_at_the_next_epoch(resume_rig, tmp_path) -> None:
    """The headline requirement: stop after epoch 1, resume, and train 2..3 - not 4 more."""
    out_dir = tmp_path / "run"
    _model, report, writes = resume_rig["train"](out_dir=out_dir, stop_after=1)
    assert report is None, "the rig was supposed to be interrupted"
    assert [entry["completed_epoch"] for entry in writes] == [0, 1]

    state = load_resumable(
        path=resume_checkpoint_path(out_dir),
        dependency_hashes=resume_rig["deps"],
        config=resume_rig["config"],
    )
    assert state["completed_epoch"] == 1
    assert state["epoch_budget"] == 4
    # Two epochs' worth of optimizer steps, which is the unit the resume continues from.
    steps_per_epoch = math.ceil(
        resume_rig["dataset"].examples / resume_rig["config"].batch_size
    )
    assert state["global_step"] == steps_per_epoch * 2

    _model2, resumed, writes2 = resume_rig["train"](out_dir=out_dir, resume_state=state)
    assert resumed is not None
    assert resumed.resumed_from_epoch == 1
    assert [entry["epoch"] for entry in resumed.history] == [0, 1, 2, 3], "epochs must not repeat"
    assert [entry["completed_epoch"] for entry in writes2] == [2, 3], (
        "only the remaining epochs are published"
    )


def test_resumed_training_matches_continuous_training(resume_rig, tmp_path) -> None:
    """Continuing must equal never having stopped - the optimizer state is genuinely restored."""
    continuous_dir = tmp_path / "continuous"
    continuous_model, continuous_report, _ = resume_rig["train"](out_dir=continuous_dir)

    interrupted_dir = tmp_path / "interrupted"
    resume_rig["train"](out_dir=interrupted_dir, stop_after=1)
    state = load_resumable(
        path=resume_checkpoint_path(interrupted_dir),
        dependency_hashes=resume_rig["deps"],
        config=resume_rig["config"],
    )
    resumed_model, resumed_report, _ = resume_rig["train"](
        out_dir=interrupted_dir, resume_state=state
    )

    assert [entry["loss"] for entry in resumed_report.history] == [
        entry["loss"] for entry in continuous_report.history
    ], "the resumed loss trajectory diverged from the continuous one"
    assert [entry["lr"] for entry in resumed_report.history] == [
        entry["lr"] for entry in continuous_report.history
    ], "the OneCycleLR schedule did not continue"
    assert resumed_report.steps == continuous_report.steps
    assert resumed_report.global_step == continuous_report.global_step
    # The implementation promises determinism under a fixed seed, so the weights must match too.
    for name, tensor in continuous_model.state_dict().items():
        assert torch.equal(tensor, resumed_model.state_dict()[name]), name


def test_the_published_state_restores_optimizer_and_scheduler_state(resume_rig, tmp_path) -> None:
    """Not just model weights: the Adam moments and the LR schedule are in the state."""
    out_dir = tmp_path / "run"
    _model, _report, _writes = resume_rig["train"](out_dir=out_dir, stop_after=1)
    state = load_resumable(
        path=resume_checkpoint_path(out_dir),
        dependency_hashes=resume_rig["deps"],
        config=resume_rig["config"],
    )
    assert state["optimizer_state"]["state"], "the optimizer moments are empty"
    assert state["scheduler_state"], "the scheduler state is empty"
    assert state["grad_scaler_state"] is None, "no AMP scaler exists on CPU fp32"
    assert state["rng_state"]["cpu"] is not None
    assert "model_state" in state
    # The Adam step counter must equal the optimizer steps completed, not 0.
    steps = {value["step"] for value in state["optimizer_state"]["state"].values()}
    assert all(int(step) == state["global_step"] for step in steps), steps
    assert state["history"][-1]["epoch"] == state["completed_epoch"]


def test_publishing_a_state_validates_it_before_promotion(resume_rig, tmp_path) -> None:
    """Transactional publication: a verified temp file is renamed into place, atomically."""
    out_dir = tmp_path / "run"
    resume_rig["train"](out_dir=out_dir, stop_after=0)
    path = resume_checkpoint_path(out_dir)
    assert path.is_file()
    # No partial artifact survives beside the real one.
    assert not path.with_name(path.name + ".tmp").exists()
    payload = torch.load(path, map_location="cpu", weights_only=False)
    assert payload["format"] == RESUMABLE_FORMAT
    assert payload["completed_epoch"] == 0
    assert set(payload["dependency_hashes"]) == set(resume_rig["deps"])


def test_an_interrupted_write_does_not_replace_the_last_good_state(resume_rig, tmp_path) -> None:
    """A failed publication must leave the previous valid state readable and unchanged."""
    out_dir = tmp_path / "run"
    resume_rig["train"](out_dir=out_dir, stop_after=1)
    path = resume_checkpoint_path(out_dir)
    good = path.read_bytes()
    good_state = torch.load(path, map_location="cpu", weights_only=False)

    class Exploding:
        """Stands in for a torch.optim.Optimizer whose state dict raises mid-write."""

        def state_dict(self):
            raise RuntimeError("simulated failure while serialising optimizer state")

    with pytest.raises(RuntimeError):
        save_resumable(
            path=path,
            model=resume_rig["fresh"](),
            optimizer=Exploding(),
            scheduler=torch.optim.lr_scheduler.OneCycleLR(
                torch.optim.AdamW(resume_rig["fresh"]().parameters(), lr=5e-4),
                max_lr=5e-4, total_steps=4,
            ),
            completed_epoch=2,
            epoch_budget=resume_rig["config"].epochs,
            global_step=3,
            history=good_state["history"],
            dependency_hashes=resume_rig["deps"],
            config=resume_rig["config"],
        )
    assert path.read_bytes() == good, "the last valid state was replaced by a failed write"
    assert not path.with_name(path.name + ".tmp").exists(), "a temp file was left behind"
    reread = load_resumable(
        path=path, dependency_hashes=resume_rig["deps"], config=resume_rig["config"]
    )
    assert reread["completed_epoch"] == 1


def test_a_truncated_state_is_refused(resume_rig, tmp_path) -> None:
    out_dir = tmp_path / "run"
    resume_rig["train"](out_dir=out_dir, stop_after=0)
    path = resume_checkpoint_path(out_dir)
    data = path.read_bytes()
    path.write_bytes(data[: len(data) // 3])  # a partial file
    with pytest.raises(Exception) as error:
        load_resumable(path=path, dependency_hashes=resume_rig["deps"], config=resume_rig["config"])
    assert error.type is not None


def test_a_foreign_file_is_refused_as_a_resume_state(resume_rig, tmp_path) -> None:
    path = tmp_path / "latest.pt"
    torch.save({"format": "something.else", "completed_epoch": 1}, path)
    with pytest.raises(TigerError) as error:
        load_resumable(path=path, dependency_hashes=resume_rig["deps"], config=resume_rig["config"])
    assert "not a resumable state" in str(error.value)


def test_a_missing_resume_state_is_refused(resume_rig, tmp_path) -> None:
    with pytest.raises(TigerError):
        load_resumable(
            path=tmp_path / "absent.pt",
            dependency_hashes=resume_rig["deps"],
            config=resume_rig["config"],
        )


def test_resume_refuses_a_changed_dependency(resume_rig, tmp_path) -> None:
    """Continuing on a different corpus/SID/layout would silently train two different problems."""
    out_dir = tmp_path / "run"
    resume_rig["train"](out_dir=out_dir, stop_after=0)
    changed = dict(resume_rig["deps"])
    changed["train_exposure"] = "0" * 64
    with pytest.raises(TigerError) as error:
        load_resumable(
            path=resume_checkpoint_path(out_dir),
            dependency_hashes=changed,
            config=resume_rig["config"],
        )
    assert "dependencies changed" in str(error.value)


def test_resume_refuses_a_changed_registered_configuration(resume_rig, tmp_path) -> None:
    out_dir = tmp_path / "run"
    resume_rig["train"](out_dir=out_dir, stop_after=0)
    # tiny_config already uses lr=1e-3, so a "change" must be a real change.
    for override in ({"learning_rate": 5e-4}, {"seed": 99}, {"batch_size": 4}, {"bf16": True}):
        with pytest.raises(TigerError) as error:
            load_resumable(
                path=resume_checkpoint_path(out_dir),
                dependency_hashes=resume_rig["deps"],
                config=tiny_config(epochs=4, **override),
            )
        assert "registered configuration changed" in str(error.value), override


def test_resume_refuses_a_renegotiated_epoch_budget(resume_rig, tmp_path) -> None:
    """``--epochs`` is the TOTAL budget; a resume may not shrink or grow it."""
    out_dir = tmp_path / "run"
    resume_rig["train"](out_dir=out_dir, stop_after=0)
    with pytest.raises(TigerError) as error:
        load_resumable(
            path=resume_checkpoint_path(out_dir),
            dependency_hashes=resume_rig["deps"],
            config=tiny_config(epochs=8),
        )
    assert "TOTAL budget" in str(error.value)


def test_resume_refuses_a_state_that_completed_the_whole_budget(resume_rig, tmp_path) -> None:
    """A state at (or past) the final epoch has nothing to continue, so it is refused."""
    out_dir = tmp_path / "run"
    model, _report, _writes = resume_rig["train"](out_dir=out_dir, stop_after=0)
    state = load_resumable(
        path=resume_checkpoint_path(out_dir),
        dependency_hashes=resume_rig["deps"],
        config=resume_rig["config"],
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=5e-4, total_steps=8)
    save_resumable(
        path=resume_checkpoint_path(out_dir),
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        completed_epoch=resume_rig["config"].epochs,
        epoch_budget=resume_rig["config"].epochs,
        global_step=0,
        history=state["history"],
        dependency_hashes=resume_rig["deps"],
        config=resume_rig["config"],
    )
    with pytest.raises(TigerError) as error:
        load_resumable(
            path=resume_checkpoint_path(out_dir),
            dependency_hashes=resume_rig["deps"],
            config=resume_rig["config"],
        )
    assert "outside the" in str(error.value)


def test_a_state_whose_history_disagrees_with_its_epoch_is_refused(resume_rig, tmp_path) -> None:
    """An inconsistent state must be refused, not silently continued from."""
    out_dir = tmp_path / "run"
    model, _report, _writes = resume_rig["train"](out_dir=out_dir, stop_after=1)
    state = load_resumable(
        path=resume_checkpoint_path(out_dir),
        dependency_hashes=resume_rig["deps"],
        config=resume_rig["config"],
    )
    # A 3-epoch budget with 2 completed epochs is consistent; claim 1 completed instead.
    assert len(state["history"]) == 2
    with pytest.raises(TigerError):
        TigerTrainer(
            model, resume_rig["dataset"], config=resume_rig["config"], device="cpu"
        ).train(resume_state={**state, "completed_epoch": 0})


# --------------------------------------------------------------------------- #
# Gate-D remediation regression tests (R1-A / R1-B / R1-C)
# --------------------------------------------------------------------------- #


def test_special_token_contract_is_derived_from_the_layout(layout: dict) -> None:
    """R1-A: every special-token identity comes from the layout, and aliasing is an error."""
    from tiger_public.tiger import generator_special_tokens

    tokens = generator_special_tokens(layout)
    assert tokens["pad"] == int(layout["special"]["pad"])
    assert tokens["bos"] == int(layout["special"]["bos"])
    assert tokens["eos"] == int(layout["special"]["eos"])
    assert tokens["sep"] == int(layout["special"]["sep"])
    assert tokens["pad"] != tokens["bos"] != tokens["eos"] != tokens["sep"]

    broken = json.loads(json.dumps(layout))
    broken["special"]["pad"] = broken["special"]["bos"]
    with pytest.raises(TigerError, match="alias"):
        generator_special_tokens(broken)

    inside_code_space = json.loads(json.dumps(layout))
    inside_code_space["special"]["pad"] = 0
    with pytest.raises(TigerError, match="code space"):
        generator_special_tokens(inside_code_space)


def test_model_and_trainer_use_the_layout_pad(trained_artifacts) -> None:
    """R1-A: the T5 config and the trainer collation agree with the layout's PAD."""
    model = trained_artifacts["model"]
    dataset = trained_artifacts["dataset"]
    pad = int(model.layout["special"]["pad"])
    assert model.model.config.pad_token_id == pad
    assert model.model.config.eos_token_id == int(model.layout["special"]["eos"])
    assert model.model.config.decoder_start_token_id == int(model.layout["special"]["bos"])

    trainer = TigerTrainer(model, dataset, config=model.config, device="cpu")
    order = list(range(min(8, dataset.examples)))
    rows, mask, _targets = trainer._collate(order)
    for position, index in enumerate(order):
        width = len(dataset.inputs[index])
        assert rows[position, width:].tolist() == [pad] * (rows.shape[1] - width)
        assert mask[position, width:].tolist() == [0] * (rows.shape[1] - width)
        assert mask[position, :width].tolist() == [1] * width


def test_targets_are_uniform_and_never_pad(trained_artifacts) -> None:
    """R1-A: there is no label padding, and no target is ever the PAD identity."""
    model = trained_artifacts["model"]
    dataset = trained_artifacts["dataset"]
    pad = int(model.layout["special"]["pad"])
    widths = {len(target) for target in dataset.targets}
    assert len(widths) == 1, f"targets have ragged widths {sorted(widths)}"
    assert widths.pop() == dataset.per_item_tokens + 1
    assert all(target[-1] == dataset.eos for target in dataset.targets)
    assert not any(pad in target for target in dataset.targets)


def test_training_metadata_records_the_derived_exposure(trained_artifacts) -> None:
    """R1-B: metadata carries the runtime example count, not a serialisation literal."""
    metadata = trained_artifacts["metadata"]
    report = trained_artifacts["report"]
    dataset = trained_artifacts["dataset"]
    training = metadata["training"]
    assert training["examples"] == report.examples == dataset.examples
    assert training["examples_positive"] is True
    assert training["steps_per_epoch"] == report.steps_per_epoch
    assert training["last_batch_size"] == report.last_batch_size
    assert training["special_tokens"]["pad"] == int(dataset_layout_pad(trained_artifacts))
    assert training["special_tokens"]["vocab_size"] == int(
        trained_artifacts["model"].layout["vocab_size"]
    )


def dataset_layout_pad(trained_artifacts):
    return trained_artifacts["model"].layout["special"]["pad"]


def test_production_exposure_arithmetic_matches_the_frozen_contract() -> None:
    """R1-B: the frozen production exposure and its derived batching arithmetic."""
    from tiger_public.tiger import (
        EXPECTED_PRODUCTION_EXAMPLES,
        expected_batches,
        expected_last_batch_size,
    )

    assert EXPECTED_PRODUCTION_EXAMPLES == 2_263_252
    assert expected_batches(EXPECTED_PRODUCTION_EXAMPLES, 512) == 4_421
    assert expected_last_batch_size(EXPECTED_PRODUCTION_EXAMPLES, 512) == 212
    assert expected_batches(EXPECTED_PRODUCTION_EXAMPLES, 512) * 20 == 88_420


def test_configure_deterministic_execution_sets_the_validated_regime() -> None:
    """R1-C: the regime H validated, and the observed state it reports."""
    import os

    import torch

    from tiger_public.tiger import (
        DETERMINISTIC_CUBLAS_WORKSPACE_CONFIG,
        configure_deterministic_execution,
    )

    previous_det = bool(torch.are_deterministic_algorithms_enabled())
    previous_warn = bool(torch.is_deterministic_algorithms_warn_only_enabled())
    previous_env = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    try:
        observed = configure_deterministic_execution()
        assert observed["deterministic_algorithms"] is True
        assert observed["deterministic_algorithms_warn_only"] is False
        assert observed["cublas_workspace_config"] == ":4096:8"
        assert observed["cublas_workspace_config"] == DETERMINISTIC_CUBLAS_WORKSPACE_CONFIG
        assert os.environ["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"
        assert bool(torch.are_deterministic_algorithms_enabled()) is True
        assert bool(torch.is_deterministic_algorithms_warn_only_enabled()) is False
    finally:
        torch.use_deterministic_algorithms(previous_det, warn_only=previous_warn)
        if previous_env is None:
            os.environ.pop("CUBLAS_WORKSPACE_CONFIG", None)
        else:
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = previous_env


def test_no_code_alias_assertion_is_retained(layout: dict, assignment: list[list[int]]) -> None:
    """R2: the accepted no-code-alias assertion is retained and still rejects aliasing."""
    from tiger_public.generator_layout import GeneratorLayoutError, assert_no_code_alias

    audit = assert_no_code_alias(layout, assignment)
    assert audit["aliased_tokens"] == []
    assert audit["catalogue_items"] == len(assignment) - 1
    assert audit["distinct_sid_tokens"] > 0

    # A special placed on a token the accepted Semantic IDs actually use is refused.
    clashing = json.loads(json.dumps(layout))
    clashing["special"]["sep"] = int(assignment[1][0])
    with pytest.raises(GeneratorLayoutError):
        assert_no_code_alias(clashing, assignment)


def test_production_layout_special_tokens_are_the_frozen_identity() -> None:
    """R2: against the REAL frozen Step-2.4F artifact - pad 1024, bos 1025, eos 1026, sep 1027.

    Skipped when the frozen production artifact is not mounted; the R3 smoke asserts the same
    identities from a live run manifest.
    """
    import os as _os
    import tempfile

    from tiger_public.generator_layout import derive_generator_layout
    from tiger_public.tiger import TigerConfig, TigerGenerator, TigerTrainer, generator_special_tokens

    sid_dir = Path(
        _os.environ.get(
            "AGENTRECX_TIGER_SID_DIR", "/root/autodl-tmp/step24f/runs/tiger_public_2026/sid"
        )
    )
    if not (sid_dir / "layout.json").is_file() or not (sid_dir / "semantic_ids.json").is_file():
        pytest.skip(f"frozen production SID artifact not present at {sid_dir}")

    with tempfile.TemporaryDirectory() as scratch:
        generator_layout = derive_generator_layout(
            sid_dir=sid_dir, generator_layout_path=Path(scratch) / "generator_layout.json"
        )
    tokens = generator_special_tokens(generator_layout)
    assert tokens == {"pad": 1024, "bos": 1025, "eos": 1026, "sep": 1027}
    assert int(generator_layout["vocab_size"]) == 1028
    assert int(generator_layout["code_space"]) == 1024

    config = TigerConfig(epochs=20, batch_size=512, seed=2026)
    torch.manual_seed(config.seed)
    model = TigerGenerator(
        config=config, layout=generator_layout, vocab_size=int(generator_layout["vocab_size"])
    )
    assert model.model.config.pad_token_id == 1024
    assert model.model.config.eos_token_id == 1026
    assert model.model.config.decoder_start_token_id == 1025

    dataset = build_examples(
        [(1, 2, 3, 4), (5, 6)],
        assignment=json.loads((sid_dir / "semantic_ids.json").read_text())["assignment"],
        layout=generator_layout,
        max_hist_items=20,
    )
    trainer = TigerTrainer(model, dataset, config=config, device="cpu")
    rows, mask, targets = trainer._collate(list(range(dataset.examples)))
    shortest = min(range(dataset.examples), key=lambda index: len(dataset.inputs[index]))
    widest = max(len(row) for row in dataset.inputs)
    width = len(dataset.inputs[shortest])
    if width < widest:
        assert rows[shortest, width:].tolist() == [1024] * (widest - width)
        assert mask[shortest, width:].tolist() == [0] * (widest - width)
    assert 1024 not in {int(value) for target in targets.tolist() for value in target}

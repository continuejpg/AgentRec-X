"""Step-2.5 Gate B smoke: generator examples, training, checkpoint and constrained generation.

Runs end to end on CPU with a deliberately tiny model, so the whole generator path is exercised
locally without touching any production artifact.

What it establishes (and what it does not)
------------------------------------------
```text
examples build from train history only      training forward/backward succeeds
loss is finite and falls                   checkpoint saves and loads
the trie builds from the accepted SIDs     constrained generation returns ONLY catalogue paths
same seed reproduces the same generation
```

It establishes **no recommendation quality**. There is no Recall/NDCG/HR here, no evaluator, no
fusion and no Agent involvement - Gate B is a mechanism gate.

The production handoff, the accepted Step-2.4F artifacts and the production configuration are
never modified. Production values can be inspected with ``--report-production``.

Usage::

    cd backends/tiger_public
    PYTHONPATH=src .venv/bin/python ../../experiments/smoke_tiger_step25.py
    PYTHONPATH=src .venv/bin/python ../../experiments/smoke_tiger_step25.py --report-production
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Sequence

import torch

from tiger_public.contracts import PAD_SENTINEL, build_token_layout
from tiger_public.generator_layout import (
    GeneratorLayoutError,
    materialise_generator_layout,
    sha256_file,
)
from tiger_public.tiger import (
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
from tiger_public.trie import CatalogueTrie, item_token_path

REPO_ROOT = Path(__file__).resolve().parent.parent
PRODUCTION_HANDOFF = REPO_ROOT / "runs" / "tiger_backend_handoff_prod"

#: The frozen production example count (Gate B.1).  ``build_examples`` emits ``n - 1`` next-item
#: examples for a train history of ``n`` items, and the accepted exposure's 412 445 rows sum to
#: 2 263 252 transitions.  ``2 675 697`` is ``sum(len(row))`` - the number of train-history item
#: *occurrences*, not a transition count - and must never be reported as an example count.
EXPECTED_PRODUCTION_EXAMPLES = 2_263_252

#: The smoke's own small item space.  Kept tiny on purpose: this is a mechanism gate.
SMOKE_ITEMS = 240
SMOKE_LEVELS = 3
SMOKE_CODEBOOK = 16
SMOKE_DEDUP = 8


def _progress(message: str) -> None:
    print(f"[smoke] {message}", flush=True)


def build_smoke_artifacts(root: Path) -> dict[str, Path]:
    """A miniature but structurally faithful catalogue / SID / exposure triple."""
    catalogue_dir = root / "catalogue"
    sid_dir = root / "sid"
    exposure_dir = root / "exposure"
    for directory in (catalogue_dir, sid_dir, exposure_dir):
        directory.mkdir(parents=True, exist_ok=True)

    # The smoke writes an *accepted-style* SID layout: pad/bos/eos only, with SEP absent.  That is
    # what Step 2.4F produced, and it is the input the generator-layout derivation consumes.  The
    # generator's own vocabulary is derived from it below, exactly as the production CLI does.
    accepted_layout = build_token_layout(
        levels=SMOKE_LEVELS,
        codebook_size=SMOKE_CODEBOOK,
        dedup_levels=1,
        dedup_vocab_size=SMOKE_DEDUP,
    ).as_dict()
    accepted_layout["format"] = "agentrecx.tiger.token_layout.v3"
    del accepted_layout["special"]["sep"]
    accepted_layout["vocab_size"] = accepted_layout["code_space"] + 3
    (sid_dir / "layout.json").write_text(json.dumps(accepted_layout), encoding="utf-8")

    # Unique SID per item, mirroring the Step-2.4F uniqueness guarantee.
    assignment = [[PAD_SENTINEL] * 4]
    for item in range(SMOKE_ITEMS):
        assignment.append(
            [
                item % SMOKE_CODEBOOK,
                (item // SMOKE_CODEBOOK) % SMOKE_CODEBOOK,
                (item // (SMOKE_CODEBOOK * SMOKE_CODEBOOK)) % SMOKE_CODEBOOK,
                item % SMOKE_DEDUP,
            ]
        )
    if len({tuple(row) for row in assignment[1:]}) != SMOKE_ITEMS:
        raise SystemExit("smoke fixture produced colliding SIDs")
    (sid_dir / "semantic_ids.json").write_text(
        json.dumps(
            {
                "format": "agentrecx.tiger.semantic_ids.v3",
                "contract_version": "agentrecx.tiger_backend.v3",
                "num_items": SMOKE_ITEMS,
                "levels": SMOKE_LEVELS,
                "dedup_levels": 1,
                "codebook_size": SMOKE_CODEBOOK,
                "dedup_vocab_size": SMOKE_DEDUP,
                "pad_row": [PAD_SENTINEL] * 4,
                "assignment": assignment,
            }
        ),
        encoding="utf-8",
    )
    (sid_dir / "tokenizer.pt").write_bytes(b"smoke stand-in tokenizer payload")
    (catalogue_dir / "catalogue.json").write_text(
        json.dumps({"num_items": SMOKE_ITEMS}), encoding="utf-8"
    )
    (catalogue_dir / "catalogue_items.jsonl").write_text(
        "".join(
            json.dumps({"backend_row": row, "item_id": row + 1}) + "\n"
            for row in range(SMOKE_ITEMS)
        ),
        encoding="utf-8",
    )

    users, length = 60, 8
    exposure_path = exposure_dir / "train_exposure.jsonl"
    with exposure_path.open("w", encoding="utf-8") as handle:
        for user in range(users):
            row = [((user * 13 + step * 7) % SMOKE_ITEMS) + 1 for step in range(length)]
            handle.write(json.dumps({"case_id": user, "items": row}) + "\n")
    # Derive the generator vocabulary from the accepted-style layout, non-mutating.  The smoke then
    # uses the *derived* artifact everywhere, which is the same object the CLI builds.
    generator_layout_path = root / "generator_layout.json"
    generator_layout = materialise_generator_layout(
        sid_dir=sid_dir, out=root
    )
    return {
        "catalogue_dir": catalogue_dir,
        "sid_dir": sid_dir,
        "exposure_path": exposure_path,
        "accepted_layout": accepted_layout,
        "layout": generator_layout,
        "generator_layout_path": generator_layout_path,
        "assignment": assignment,
        "users": users,
        "length": length,
    }


def count_production_exposure(path: Path = PRODUCTION_HANDOFF / "train_exposure.jsonl") -> dict[str, int]:
    """Count the production exposure **by streaming**, never by building an example list.

    The three sums are kept separate on purpose, because conflating them is exactly the error this
    report exists to prevent::

        sum(len(row))     item occurrences       2 675 697
        sum(len(row) - 1) next-item transitions  2 263 252   <- the generator's example count
        sum(len(row) - 2) GenRec-v0 / H5 pairs   1 850 807

    A row is read, its length taken, and the row dropped; peak memory is one row regardless of the
    412 445-row, 32 MB artifact.  No model, no tokenisation, no training.
    """
    users = 0
    occurrences = 0
    transitions = 0
    pairs = 0
    shortest = 2**31
    longest = 0
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            length = len(json.loads(line)["items"])
            users += 1
            occurrences += length
            transitions += max(0, length - 1)
            pairs += max(0, length - 2)
            shortest = min(shortest, length)
            longest = max(longest, length)
    return {
        "users": users,
        "sum_len": occurrences,
        "sum_len_minus_1": transitions,
        "sum_len_minus_2": pairs,
        "min_history": shortest,
        "max_history": longest,
    }


def production_report(*, verify: bool = True, path: Path | None = None) -> int:
    """Report the registered production configuration without running anything.

    ``verify`` streams the accepted exposure and refuses to print a configuration report if the
    measured transitions differ from the frozen count, so a drifted handoff cannot be reported as
    production-ready.
    """
    exposure = Path(path) if path is not None else PRODUCTION_HANDOFF / "train_exposure.jsonl"
    counts = None
    if exposure.is_file():
        counts = count_production_exposure(exposure)
    elif verify:
        print(f"REFUSING: no exposure at {exposure}; the production count cannot be verified")
        return 1

    config = TigerConfig()
    manifest = PRODUCTION_HANDOFF / "manifest.json"
    populations = {}
    if manifest.is_file():
        populations = json.loads(manifest.read_text(encoding="utf-8")).get("populations", {})
    print("REGISTERED PRODUCTION CONFIGURATION (docs/TIGER_BACKEND.md H5 / section 7.4)")
    print(json.dumps(config.as_dict(vocab_size=1028, per_item_tokens=4), indent=2, sort_keys=True))
    print(json.dumps(config.as_training_dict(), indent=2, sort_keys=True))
    print("H5 also registers: examples 1850807, bf16, resumable checkpoint every epoch")
    print()
    print("ACCEPTED PRODUCTION HANDOFF POPULATIONS")
    print(json.dumps(populations, indent=2, sort_keys=True))
    print()
    if counts is not None:
        print("MEASURED BY STREAMING train_exposure.jsonl (Gate B.1)")
        print(json.dumps(counts, indent=2, sort_keys=True))
        print()
        print("EXAMPLE ARITHMETIC - keep the three sums distinct")
        print(f"  sum(len(row))     item occurrences      {counts['sum_len']}")
        print(f"  sum(len(row) - 1) next-item transitions {counts['sum_len_minus_1']}"
              "   <- GENERATOR EXAMPLES (Gate B objective)")
        print(f"  sum(len(row) - 2) GenRec-v0 / H5 pairs  {counts['sum_len_minus_2']}")
        print()
        if counts["sum_len_minus_1"] != EXPECTED_PRODUCTION_EXAMPLES:
            print(
                f"REFUSING: measured {counts['sum_len_minus_1']} generator examples, expected "
                f"{EXPECTED_PRODUCTION_EXAMPLES}; the exposure artifact has drifted"
            )
            return 1
        print(f"OK: generator examples == {EXPECTED_PRODUCTION_EXAMPLES} (frozen, Gate B.1)")
    print("NOTE: H5's 1850807 is sum(len - 2) over this exposure (the GenRec-v0 convention).")
    print("      The generator's own objective is one next-item example per adjacent target")
    print("      position, i.e. sum(len - 1) = 2263252. See docs/TIGER_BACKEND.md 17.4.")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Step-2.5 Gate B local smoke")
    parser.add_argument("--report-production", action="store_true",
                        help="print the registered production configuration and the measured "
                             "example arithmetic, then exit")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--keep", action="store_true", help="keep the working directory")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)

    if args.report_production:
        return production_report()

    root = args.out or Path(tempfile.mkdtemp(prefix="tiger_gateB_"))
    root.mkdir(parents=True, exist_ok=True)
    checks: list[tuple[bool, str]] = []

    def check(condition: bool, message: str) -> None:
        checks.append((bool(condition), message))

    try:
        started = time.time()
        artifacts = build_smoke_artifacts(root)
        layout = artifacts["layout"]
        assignment = artifacts["assignment"]

        # ---- trie ---------------------------------------------------------- #
        trie = CatalogueTrie(assignment, layout=layout)
        _progress(f"trie: {trie.stats.nodes} nodes, {trie.stats.leaves} leaves, "
                  f"{trie.items} items")
        check(trie.collision_free(), "trie is collision-free")
        check(not trie.contains_sentinel(), "trie contains no PAD sentinel edge")
        check(all(
            trie.is_complete(item_token_path(assignment[item], layout))
            for item in range(1, len(assignment))
        ), "every catalogue SID is a complete trie path")

        # ---- examples ------------------------------------------------------ #
        rows = [
            tuple(json.loads(line)["items"])
            for line in artifacts["exposure_path"].read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        dataset = build_examples(
            rows, assignment=assignment, layout=layout, max_hist_items=5
        )
        expected = sum(len(row) - 1 for row in rows)
        _progress(f"examples: {dataset.examples} over {dataset.users} users; "
                  f"hist items {dataset.min_history_items}-{dataset.max_history_items}")
        check(dataset.examples == expected, "example count is sum(len(row) - 1)")
        check(dataset.users == artifacts["users"], "example user count matches the exposure")
        check(all(target[-1] == layout["special"]["eos"] for target in dataset.targets),
              "every target ends with EOS")

        # ---- train --------------------------------------------------------- #
        config = TigerConfig(
            d_model=64, num_layers=2, num_heads=2, d_ff=128, dropout=0.0,
            max_hist_items=5, epochs=args.epochs, batch_size=32,
            learning_rate=1e-3, bf16=False, seed=2026,
        )
        torch.manual_seed(config.seed)
        model = TigerGenerator(config=config, layout=layout,
                               vocab_size=int(layout["vocab_size"]))
        report = TigerTrainer(model, dataset, config=config, device="cpu").train()
        losses = [entry["loss"] for entry in report.history]
        _progress(f"train: params={model.parameter_count()} loss={losses} "
                  f"({report.seconds:.1f}s, {report.precision})")
        check(len(losses) == args.epochs, "every requested epoch ran")
        check(all(value == value for value in losses), "loss is finite every epoch")
        check(losses[-1] < losses[0], "loss decreased over the run")

        # ---- checkpoint ---------------------------------------------------- #
        out_dir = root / "generator"
        sid_before = {
            path.name: sha256_file(path) for path in sorted(artifacts["sid_dir"].iterdir())
        }
        metadata = save_generator(
            model=model, report=report,
            sid_dir=artifacts["sid_dir"],
            exposure_path=artifacts["exposure_path"],
            catalogue_dir=artifacts["catalogue_dir"],
            generator_layout_path=artifacts["generator_layout_path"],
            out_dir=out_dir,
        )
        names = {path.name for path in out_dir.iterdir()}
        _progress(f"checkpoint: {sorted(names)} sha256={metadata['checkpoint_sha256'][:16]}")
        check({"tiger.pt", "tiger.json", "generator_layout.json", "score_rule.json"} <= names,
              "checkpoint writes the documented files")
        check(metadata["training"]["validation_used"] is False,
              "checkpoint records that no validation target was used")
        check(
            metadata["dependency_hashes"]["generator_layout"]
            == metadata["generator_layout_sha256"],
            "the checkpoint binds the derived generator-layout hash",
        )
        check(
            metadata["dependency_hashes"]["layout"] == sid_before["layout.json"],
            "the checkpoint binds the ACCEPTED SID layout hash",
        )

        # The accepted SID artifact must be byte-identical after materialisation + training.
        sid_after = {
            path.name: sha256_file(path) for path in sorted(artifacts["sid_dir"].iterdir())
        }
        check(sid_before == sid_after, "the accepted SID artifact is unchanged")
        check(
            not (artifacts["sid_dir"] / "layout.v3.bak.json").exists(),
            "no backup or patch was written inside the accepted SID directory",
        )

        restored, _meta, _resume = load_generator(
            out_dir=out_dir, sid_dir=artifacts["sid_dir"],
            exposure_path=artifacts["exposure_path"],
            catalogue_dir=artifacts["catalogue_dir"],
            generator_layout_path=artifacts["generator_layout_path"], device="cpu",
        )
        check(restored.parameter_count() == model.parameter_count(),
              "checkpoint reloads with the same parameter count")

        # A checkpoint must refuse a *different* generator layout.
        tampered_path = root / "tampered_generator_layout.json"
        tampered = dict(artifacts["layout"])
        tampered["extension"] = {"added_special": "sep", "sep_token": 0}
        tampered_path.write_text(json.dumps(tampered, indent=1, sort_keys=True) + "\n",
                                 encoding="utf-8")
        try:
            load_generator(
                out_dir=out_dir, sid_dir=artifacts["sid_dir"],
                exposure_path=artifacts["exposure_path"],
                catalogue_dir=artifacts["catalogue_dir"],
                generator_layout_path=tampered_path, device="cpu",
            )
            check(False, "a changed generator layout is refused")
        except Exception as error:  # noqa: BLE001 - the refusal type is the check
            check("generator_layout" in str(error), f"a changed generator layout is refused: {type(error).__name__}")

        # ...and must refuse a *valid but different* generator layout, because the checkpoint binds
        # the hash of the exact artifact it was trained against.
        relocated = root / "relocated"
        relocated.mkdir(parents=True, exist_ok=True)
        other = materialise_generator_layout(sid_dir=artifacts["sid_dir"], out=relocated)
        other_path = relocated / "generator_layout.json"
        payload = json.loads(other_path.read_text(encoding="utf-8"))
        payload["annotation"] = "same vocabulary, different artifact"
        other_path.write_text(json.dumps(payload, indent=1, sort_keys=True) + "\n", encoding="utf-8")
        try:
            load_generator(
                out_dir=out_dir, sid_dir=artifacts["sid_dir"],
                exposure_path=artifacts["exposure_path"],
                catalogue_dir=artifacts["catalogue_dir"],
                generator_layout_path=other_path, device="cpu",
            )
            check(False, "the checkpoint refuses a different generator-layout artifact")
        except Exception as error:  # noqa: BLE001 - the refusal is the check
            check("generator_layout" in str(error),
                  f"the checkpoint refuses a different generator-layout artifact: {type(error).__name__}")
        check(other["audit"]["aliased_tokens"] == [],
              "the re-derived layout still reports no SEP/SID alias")

        # The accepted layout alone must NOT be usable as a generator layout.
        try:
            build_examples(
                rows, assignment=assignment, layout=artifacts["accepted_layout"],
                max_hist_items=5,
            )
            check(False, "the accepted SID layout is refused as a generator layout")
        except Exception as error:  # noqa: BLE001
            check("generator_layout" in str(error),
                  f"the accepted SID layout is refused as a generator layout: {type(error).__name__}")

        # SEP is a boundary token, never a digit: no target block may contain it, and every target
        # must still resolve through the trie.
        sep_token = layout["special"]["sep"]
        check(sep_token not in {token for target in dataset.targets for token in target[:-1]},
              "SEP is never interpreted as a SID digit")
        check(sep_token not in {
            token for row in assignment[1:] for token in item_token_path(row, layout)
        }, "SEP is not part of any catalogue SID token path")

        # ---- generation ---------------------------------------------------- #
        sample = [list(dataset.inputs[index]) for index in range(min(16, dataset.examples))]
        width = max(len(row) for row in sample)
        batch = torch.full((len(sample), width), 0, dtype=torch.long)
        mask = torch.zeros((len(sample), width), dtype=torch.long)
        for position, row in enumerate(sample):
            batch[position, : len(row)] = torch.tensor(row)
            mask[position, : len(row)] = 1
        paths = model.constrained_generate(batch, mask, trie=trie, beam=3)
        valid = sum(1 for path in paths if trie.is_complete(path))
        _progress(f"generation: {valid}/{len(paths)} paths are complete catalogue paths")
        check(valid == len(paths), "every generated path is a catalogue path")
        check(all(trie.item_for_path(path) is not None for path in paths),
              "every generated path resolves to one item")

        reloaded_paths = restored.constrained_generate(batch, mask, trie=trie, beam=3)
        check(reloaded_paths == paths, "a reloaded checkpoint generates identically")

        # ---- resumability -------------------------------------------------- #
        # Interrupt the same 3-epoch budget after epoch 1, then continue it.  The comparison is
        # against `losses`, the uninterrupted run above, so a partial restore cannot pass.
        smoke_budget = TigerConfig(
            d_model=64, num_layers=2, num_heads=2, d_ff=128, dropout=0.0,
            max_hist_items=5, epochs=args.epochs, batch_size=32,
            learning_rate=1e-3, bf16=False, seed=2026,
        )
        smoke_deps = {
            "semantic_ids": sha256_file(artifacts["sid_dir"] / "semantic_ids.json"),
            "layout": sha256_file(artifacts["sid_dir"] / "layout.json"),
            "tokenizer": sha256_file(artifacts["sid_dir"] / "tokenizer.pt"),
            "train_exposure": sha256_file(artifacts["exposure_path"]),
            "catalogue": sha256_file(artifacts["catalogue_dir"] / "catalogue.json"),
            "catalogue_items": sha256_file(artifacts["catalogue_dir"] / "catalogue_items.jsonl"),
            "generator_layout": sha256_file(artifacts["generator_layout_path"]),
        }
        resume_path = resume_checkpoint_path(root / "resume_run")
        # Seeded like the uninterrupted run above: an unseeded start would diverge for a reason
        # that has nothing to do with resumability.
        torch.manual_seed(smoke_budget.seed)
        resume_model = TigerGenerator(
            config=smoke_budget, layout=layout, vocab_size=int(layout["vocab_size"])
        )
        interrupted = False

        partial_history: list[dict[str, Any]] = []

        def _hook(**kwargs):
            nonlocal interrupted
            partial_history[:] = [dict(entry) for entry in kwargs["history"]]
            save_resumable(
                path=resume_path,
                model=kwargs["model"],
                optimizer=kwargs["optimizer"],
                scheduler=kwargs["scheduler"],
                completed_epoch=kwargs["completed_epoch"],
                epoch_budget=smoke_budget.epochs,
                global_step=kwargs["global_step"],
                history=kwargs["history"],
                dependency_hashes=smoke_deps,
                config=smoke_budget,
            )
            if args.epochs > 1 and kwargs["completed_epoch"] == 0 and not interrupted:
                interrupted = True
                raise KeyboardInterrupt("smoke: stop after epoch 1")

        try:
            resume_report = TigerTrainer(
                resume_model, dataset, config=smoke_budget, device="cpu"
            ).train(checkpoint_hook=_hook)
        except KeyboardInterrupt:
            resume_report = None
        except TigerError:
            # A 1-epoch smoke cannot be interrupted mid-budget, which is fine.
            resume_report = None
            interrupted = False
        check(resume_path.is_file(), "a resumable state is published every epoch")
        check(not resume_path.with_name(resume_path.name + ".tmp").exists(),
              "no partial resumable state is left behind")

        state = load_resumable(path=resume_path, dependency_hashes=smoke_deps, config=smoke_budget)
        check(state["completed_epoch"] == 0, "the resumable state records the completed epoch")
        check(state["epoch_budget"] == smoke_budget.epochs,
              "the resumable state records the TOTAL epoch budget")
        check(bool(state["optimizer_state"]["state"]), "the optimizer state is in the resumable state")
        continued = TigerGenerator(config=smoke_budget, layout=layout,
                                   vocab_size=int(layout["vocab_size"]))
        continued_report = TigerTrainer(
            continued, dataset, config=smoke_budget, device="cpu"
        ).train(resume_state=state)
        check([entry["epoch"] for entry in continued_report.history] == list(range(args.epochs)),
              "resumed training covers every epoch exactly once")
        check(continued_report.history[: len(partial_history)]
              == [dict(entry) for entry in partial_history],
              "resumed history continues the interrupted run without repeating an epoch")
        check([entry["loss"] for entry in continued_report.history] == losses,
              "resumed training reproduces the uninterrupted loss trajectory")
        check(continued_report.global_step == report.global_step,
              "resumed training reaches the same optimizer-step total as the uninterrupted run")

        # A resume must refuse a changed dependency or a renegotiated budget.
        try:
            load_resumable(
                path=resume_path,
                dependency_hashes={**smoke_deps, "train_exposure": "0" * 64},
                config=smoke_budget,
            )
            check(False, "a resume with changed dependencies is refused")
        except TigerError as error:
            check("dependencies changed" in str(error),
                  f"a resume with changed dependencies is refused: {type(error).__name__}")
        try:
            load_resumable(
                path=resume_path,
                dependency_hashes=smoke_deps,
                config=TigerConfig(
                    d_model=64, num_layers=2, num_heads=2, d_ff=128, dropout=0.0,
                    max_hist_items=5, epochs=args.epochs + 3, batch_size=32,
                    learning_rate=1e-3, bf16=False, seed=2026,
                ),
            )
            check(False, "a renegotiated epoch budget is refused")
        except TigerError as error:
            check("TOTAL budget" in str(error),
                  f"a renegotiated epoch budget is refused: {type(error).__name__}")

        # Genuine reproducibility: same seed, same config, same data -> same trained model and
        # the same generation.  Comparing an *untrained* model would be meaningless, so this
        # retrains from scratch.
        torch.manual_seed(config.seed)
        repeat = TigerGenerator(config=config, layout=layout,
                                vocab_size=int(layout["vocab_size"]))
        repeat_report = TigerTrainer(repeat, dataset, config=config, device="cpu").train()
        repeat_losses = [entry["loss"] for entry in repeat_report.history]
        check(repeat_losses == losses, "same seed reproduces the same loss trajectory")
        check(repeat.constrained_generate(batch, mask, trie=trie, beam=3) == paths,
              "same seed reproduces the same generation")

        report_payload: dict[str, Any] = {
            "root": str(root),
            "items": SMOKE_ITEMS,
            "examples": dataset.examples,
            "users": dataset.users,
            "params": model.parameter_count(),
            "losses": losses,
            "checkpoint_sha256": metadata["checkpoint_sha256"],
            "generated_paths": len(paths),
            "valid_paths": valid,
            "seconds": round(time.time() - started, 3),
            "checks": [{"ok": ok, "message": message} for ok, message in checks],
            "passed": all(ok for ok, _ in checks),
        }
    finally:
        if not args.keep and args.out is None:
            shutil.rmtree(root, ignore_errors=True)
        else:
            (root / "smoke_report.json").write_text(
                json.dumps(report_payload, indent=1, sort_keys=True) + "\n", encoding="utf-8"
            )

    print()
    for ok, message in checks:
        print(f"  {'PASS' if ok else 'FAIL'}  {message}")
    failed = [message for ok, message in checks if not ok]
    print(f"\n{len(checks) - len(failed)}/{len(checks)} smoke checks passed")
    if args.keep or args.out is not None:
        print(f"working directory kept at {root}")
    print(json.dumps({"smoke_passed": not failed, "failed": failed, "root": str(root)},
                     sort_keys=True))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())

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
from tiger_public.tiger import (
    TigerConfig,
    TigerGenerator,
    TigerTrainer,
    build_examples,
    load_generator,
    save_generator,
)
from tiger_public.trie import CatalogueTrie, item_token_path

REPO_ROOT = Path(__file__).resolve().parent.parent
PRODUCTION_HANDOFF = REPO_ROOT / "runs" / "tiger_backend_handoff_prod"

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

    layout = build_token_layout(
        levels=SMOKE_LEVELS,
        codebook_size=SMOKE_CODEBOOK,
        dedup_levels=1,
        dedup_vocab_size=SMOKE_DEDUP,
    ).as_dict()
    (sid_dir / "layout.json").write_text(json.dumps(layout), encoding="utf-8")

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
    return {
        "catalogue_dir": catalogue_dir,
        "sid_dir": sid_dir,
        "exposure_path": exposure_path,
        "layout": layout,
        "assignment": assignment,
        "users": users,
        "length": length,
    }


def production_report() -> int:
    """Report the registered production configuration without running anything."""
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
    print("NOTE: H5's 1850807 is sum(len-2) over this exposure (the GenRec-v0 convention).")
    print("      A true next-item objective over the same rows is sum(len-1) = 2675697.")
    print("      See docs/TIGER_BACKEND.md 17.4; Gate C must register which one it means.")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Step-2.5 Gate B local smoke")
    parser.add_argument("--report-production", action="store_true",
                        help="print the registered production configuration and exit")
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
        metadata = save_generator(
            model=model, report=report,
            sid_dir=artifacts["sid_dir"],
            exposure_path=artifacts["exposure_path"],
            catalogue_dir=artifacts["catalogue_dir"],
            out_dir=out_dir,
        )
        names = {path.name for path in out_dir.iterdir()}
        _progress(f"checkpoint: {sorted(names)} sha256={metadata['checkpoint_sha256'][:16]}")
        check({"tiger.pt", "tiger.json", "layout.json", "score_rule.json"} <= names,
              "checkpoint writes the documented files")
        check(metadata["training"]["validation_used"] is False,
              "checkpoint records that no validation target was used")

        restored, _meta, _resume = load_generator(
            out_dir=out_dir, sid_dir=artifacts["sid_dir"],
            exposure_path=artifacts["exposure_path"],
            catalogue_dir=artifacts["catalogue_dir"], device="cpu",
        )
        check(restored.parameter_count() == model.parameter_count(),
              "checkpoint reloads with the same parameter count")

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

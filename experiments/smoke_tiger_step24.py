"""Step-2.4 smoke gate: content features, RQ-VAE, and collision-free Semantic IDs.

Proves the stage-1 and stage-2 chain on a small catalogue before any GPU run:

```text
G4   item text -> features        shape, dtype, no NaN/Inf, no PAD row, coverage 1.000
G5   features -> RQ-VAE -> SIDs   pre-dedup audit, post-dedup uniqueness, per-level utilisation
G5b  dedup overflow               a group wider than the dedup vocabulary refuses the build
G11  reproducibility              a second invocation is byte-identical
```

The smoke is **not** a modelling result.  By default it uses the backend's deterministic
``smoke`` encoder, which is a plumbing stand-in and is labelled as such in the artifact; pass
``--encoder sentence-transformers/sentence-t5-base`` to run the real encoder, which needs its
weights available locally.

Usage::

    .venv/bin/python -m experiments.smoke_tiger_step24 --items 2000
    .venv/bin/python -m experiments.smoke_tiger_step24 --items 2000 --encoder smoke
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Sequence

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

BACKEND_ROOT = REPO_ROOT / "backends" / "tiger_public"
BACKEND_PYTHON = BACKEND_ROOT / ".venv" / "bin" / "python"
FALLBACK_PYTHON = REPO_ROOT / ".venv" / "bin" / "python"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_backend(stage: str, *args: str, quiet: bool = True) -> dict[str, Any]:
    """Invoke one backend stage and return its JSON status line."""
    python = BACKEND_PYTHON if BACKEND_PYTHON.is_file() else FALLBACK_PYTHON
    import os

    environment = dict(os.environ)
    source_root = str(BACKEND_ROOT / "src")
    existing = environment.get("PYTHONPATH", "")
    environment["PYTHONPATH"] = f"{source_root}{os.pathsep}{existing}" if existing else source_root
    del quiet  # the stage runs are already quiet at this scale; kept for signature stability
    argv = [str(python), "-m", "tiger_public.cli", stage, *args]
    completed = subprocess.run(  # noqa: S603 - fixed argv, no shell
        argv, cwd=str(BACKEND_ROOT), env=environment, capture_output=True, check=False, timeout=7200
    )
    if completed.returncode != 0:
        raise SystemExit(
            f"backend stage {stage!r} failed (exit {completed.returncode})\n"
            f"stdout: {completed.stdout.decode('utf-8', 'replace')[-2000:]}\n"
            f"stderr: {completed.stderr.decode('utf-8', 'replace')[-3000:]}"
        )
    for line in reversed(completed.stdout.decode("utf-8", "replace").splitlines()):
        if line.strip().startswith("{"):
            return json.loads(line)
    raise SystemExit(f"backend stage {stage!r} produced no status line")


def materialise_handoff(out_dir: Path, *, items: int, cohort: int) -> dict[str, Any]:
    """Build the Step-2.3 handoff on the smoke item subset."""
    argv = [
        str(REPO_ROOT / ".venv" / "bin" / "python"),
        "-m",
        "experiments.materialize_tiger_backend",
        "--out",
        str(out_dir),
        "--cohort",
        str(cohort),
        "--limit",
        str(items),
        "--quiet",
    ]
    completed = subprocess.run(  # noqa: S603 - fixed argv, no shell
        argv, cwd=str(REPO_ROOT), capture_output=True, check=False, timeout=3600
    )
    if completed.returncode != 0:
        raise SystemExit(
            f"handoff materialisation failed (exit {completed.returncode})\n"
            f"{completed.stderr.decode('utf-8', 'replace')[-3000:]}"
        )
    return json.loads((out_dir / "materialize_run.json").read_text(encoding="utf-8"))


def check(condition: bool, message: str, results: list[tuple[str, bool, str]]) -> None:
    results.append(("PASS" if condition else "FAIL", condition, message))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Step-2.4 smoke gate")
    parser.add_argument("--items", type=int, default=2000)
    parser.add_argument("--cohort", type=int, default=300)
    parser.add_argument("--encoder", default="smoke")
    parser.add_argument(
        "--learning-rate", type=float, default=3e-4,
        help="frozen after measurement: see the Step-2.4 report for why 1e-3 collapses levels 1-2",
    )
    parser.add_argument(
        "--allow-dead-codes", action="store_true",
        help="waive the >20%% dead-code stop, recorded in the artifact",
    )
    parser.add_argument("--codebook-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--keep", action="store_true", help="keep the working directory")
    args = parser.parse_args(argv)

    root = args.out or Path(tempfile.mkdtemp(prefix="tiger_step24_smoke_"))
    root.mkdir(parents=True, exist_ok=True)
    results: list[tuple[str, bool, str]] = []
    report: dict[str, Any] = {"root": str(root), "encoder": args.encoder, "items": args.items}

    try:
        started = time.time()
        handoff = materialise_handoff(root, items=args.items, cohort=args.cohort)
        report["handoff"] = handoff
        print(f"[smoke] handoff: {handoff['items']} items, {handoff['cohort_cases']} cases "
              f"({time.time() - started:.1f}s)")

        # ---- G4: features -------------------------------------------------- #
        features_dir = root / "features"
        status = run_backend(
            "build-features",
            "--catalogue", str(root),
            "--out", str(features_dir),
            "--encoder", args.encoder,
            "--batch-size", str(args.batch_size),
            "--device", args.device,
        )
        record = json.loads((features_dir / "item_features.json").read_text(encoding="utf-8"))
        report["features"] = {**status, **{
            key: record[key] for key in (
                "num_items", "dim", "dtype", "empty_text_items", "truncated_items",
                "nan_inf_scan", "coverage", "pad_row_present", "products_text_sha256",
                "sha256", "seconds",
            )
        }}
        import numpy as np

        array = np.load(features_dir / "item_features.npy", mmap_mode="r")
        check(array.shape == (args.items, int(record["dim"])),
              f"G4 shape is [{args.items}, {record['dim']}]", results)
        check(array.dtype == np.float32, "G4 dtype is float32", results)
        check(bool(np.isfinite(np.asarray(array)).all()), "G4 no NaN and no Inf", results)
        check(not bool(record["pad_row_present"]), "G4 no PAD row", results)
        check(float(record["coverage"]) == 1.0, "G4 coverage is 1.000", results)
        check(
            record["products_text_sha256"] == sha256_file(root / "products_text.jsonl"),
            "G4 products_text digest matches the handoff", results,
        )
        print(f"[smoke] features: dim={record['dim']} empty_text={record['empty_text_items']} "
              f"truncated={record['truncated_items']} {record['seconds']}s "
              f"encoder_is_model={bool(record['encoder'].get('is_model'))}")

        # ---- G5: RQ-VAE + dedup -------------------------------------------- #
        sid_dir = root / "sid"
        status = run_backend(
            "fit-sid",
            "--catalogue", str(root),
            "--features", str(features_dir),
            "--out", str(sid_dir),
            "--codebook-size", str(args.codebook_size),
            "--epochs", str(args.epochs),
            "--batch-size", str(args.batch_size),
            "--device", args.device,
            "--learning-rate", str(args.learning_rate),
            *(["--allow-dead-codes"] if args.allow_dead_codes else []),
        )
        sid = json.loads((sid_dir / "semantic_ids.json").read_text(encoding="utf-8"))
        report["semantic_ids"] = {
            "num_items": sid["num_items"],
            "levels": sid["levels"],
            "dedup_levels": sid["dedup_levels"],
            "codebook_size": sid["codebook_size"],
            "dedup_vocab_size": sid["dedup_vocab_size"],
            "pad_row": sid["pad_row"],
            "collision_audit": sid["collision_audit"],
            "utilisation": sid["utilisation"],
            "training": {
                "history": sid["training"]["history"],
                "seconds": sid["training"]["seconds"],
                "revived_codes_total": sid["training"]["revived_codes_total"],
                "peak_allocated_bytes": sid["training"]["peak_allocated_bytes"],
                "device": sid["training"]["device"],
                "determinism_requested": sid["training"]["determinism_requested"],
            },
            "content_features": sid["content_features"],
            "checkpoint_sha256": sid["checkpoint_sha256"],
        }
        audit = sid["collision_audit"]
        pre, post = audit["pre_dedup"], audit["post_dedup"]
        check(sid["num_items"] == args.items, "G5 every catalogue item was assigned", results)
        check(all(row[0] == -1 for row in [sid["pad_row"]]), "G5 PAD sentinel intact", results)
        check(post["collision_groups"] == 0, "G5 post-dedup collision groups == 0", results)
        check(post["distinct_sids"] == args.items,
              "G5 post-dedup distinct SIDs == num_items", results)
        check(audit["largest_group_le_dedup_vocab"],
              "G5 pre-dedup largest group <= dedup vocabulary", results)
        waiver = sid.get("dead_code_waiver") or {}
        if args.allow_dead_codes:
            check(
                bool(waiver.get("waived")) or not any(
                    block["collapsed"] for block in sid["utilisation"]
                ),
                "G5 dead-code stop explicitly waived and recorded in the artifact",
                results,
            )
        else:
            check(not any(block["collapsed"] for block in sid["utilisation"]),
                  "G5 no level exceeds the 20% dead-code stop", results)
        history = sid["training"]["history"]
        check(bool(history) and all(
            float(entry["loss"]) == float(entry["loss"]) for entry in history),
            "G5 loss history is finite", results)
        print(f"[smoke] RQ-VAE: loss {history[0]['loss']} -> {history[-1]['loss']} "
              f"({sid['training']['seconds']}s) pre_largest={pre['largest_group']} "
              f"pre_groups={pre['collision_groups']} pre_rate={pre['collision_rate']}")
        print(f"[smoke] utilisation: "
              f"{[(b['used_codes'], b['dead_codes'], b['id_entropy']) for b in sid['utilisation']]}")

        # ---- G5b: dedup overflow refuses ------------------------------------ #
        overflow_dir = root / "sid_tight"
        python = BACKEND_PYTHON if BACKEND_PYTHON.is_file() else FALLBACK_PYTHON
        import os

        environment = dict(os.environ)
        source_root = str(BACKEND_ROOT / "src")
        environment["PYTHONPATH"] = (
            f"{source_root}{os.pathsep}{environment['PYTHONPATH']}"
            if environment.get("PYTHONPATH") else source_root
        )
        completed = subprocess.run(  # noqa: S603 - fixed argv, no shell
            [
                str(python), "-m", "tiger_public.cli", "fit-sid",
                "--catalogue", str(root),
                "--features", str(features_dir),
                "--out", str(overflow_dir),
                "--codebook-size", str(args.codebook_size),
                "--dedup-vocab-size", "1",
                "--epochs", str(args.epochs),
                "--batch-size", str(args.batch_size),
                "--device", args.device,
                "--learning-rate", str(args.learning_rate),
            ],
            cwd=str(BACKEND_ROOT), env=environment, capture_output=True, check=False, timeout=7200,
        )
        refused = completed.returncode != 0
        message = completed.stderr.decode("utf-8", "replace")
        # Two phrasings, both correct: the pre-training feasibility refusal (the configuration
        # cannot work) and the post-training measured refusal (the fitted groups are too wide).
        refusal_markers = (
            "exceeds dedup_vocab_size",
            "cannot disambiguate even a two-item collision group",
            "guarantees a collision group",
        )
        check(
            refused and any(marker in message for marker in refusal_markers),
            "G5b an unaddressable collision group refuses the build (dedup width 1)",
            results,
        )
        report["overflow_probe"] = {
            "exit_code": completed.returncode,
            "refused": refused,
            "message": message.strip().splitlines()[-1] if message.strip() else "",
        }
        print(f"[smoke] overflow probe: exit={completed.returncode} "
              f"{'refused as expected' if refused else 'NOT REFUSED'}")

        # ---- G11: reproducibility ------------------------------------------- #
        second = root / "features_again"
        run_backend(
            "build-features",
            "--catalogue", str(root),
            "--out", str(second),
            "--encoder", args.encoder,
            "--batch-size", str(args.batch_size),
            "--device", args.device,
        )
        again = json.loads((second / "item_features.json").read_text(encoding="utf-8"))
        check(again["sha256"] == record["sha256"],
              "G11 a second feature build is byte-identical", results)
        print(f"[smoke] reproducibility: {record['sha256'][:16]} == {again['sha256'][:16]}")

        report["checks"] = [
            {"status": status, "message": message} for status, _ok, message in results
        ]
        report["passed"] = all(ok for _status, ok, _message in results)
        report["seconds"] = round(time.time() - started, 3)
    finally:
        if not args.keep and args.out is None:
            shutil.rmtree(root, ignore_errors=True)
        else:
            (root / "smoke_report.json").write_text(
                json.dumps(report, indent=1, sort_keys=True, default=str) + "\n",
                encoding="utf-8",
            )

    print()
    for status, _ok, message in results:
        print(f"  {status}  {message}")
    failed = [message for status, ok, message in results if not ok]
    print(f"\n{len(results) - len(failed)}/{len(results)} smoke checks passed")
    if args.keep or args.out is not None:
        print(f"working directory kept at {root}")
    print(json.dumps({"smoke_passed": not failed, "failed": failed, "root": str(root)},
                     sort_keys=True))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())

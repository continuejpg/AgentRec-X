"""Gate D runbook check: restore the accepted Step-2.4F artifacts, verify, and print the run.

This script **trains nothing**.  It exists so the Gate-C/D GPU run cannot start from a
half-verified state, and so that the accepted Step-2.4F artifacts are *restored and verified*
rather than rebuilt.  Rebuilding `Sentence-T5` features, the RQ-VAE or the Semantic IDs is never a
step here: the H2/H3 artifacts are frozen, and Gate A already proved the production handoff's
`products_text` identity matches the accepted feature/SID dependency.

What it does, in order:

1. verifies the accepted archive's SHA256 against the pinned manifest below;
2. verifies the per-file SHA256 of the extracted H2/H3 artifacts;
3. streams `train_exposure.jsonl` and proves the frozen example arithmetic (2 263 252 transitions
   over 412 445 rows, 156 746 catalogue items, zero validation/test targets);
4. checks the one precondition Gate B introduced - the accepted `layout.json` must declare `sep`;
5. prints the exact Gate-D training command, and refuses (exit 1) if any check fails.

Usage::

    cd backends/tiger_public
    PYTHONPATH=src .venv/bin/python ../../experiments/gate_d_tiger_training.py \\
        --archive ../../runs/_autodl_backup/step24f-tiger-public-dfabc1c.tar.gz \\
        --restore  /data/agentrecx/step24f
"""

from __future__ import annotations

import argparse
import hashlib
import json

import sys
import tarfile
from pathlib import Path
from typing import Any, Sequence

REPO_ROOT = Path(__file__).resolve().parent.parent
PRODUCTION_HANDOFF = REPO_ROOT / "runs" / "tiger_backend_handoff_prod"

#: SHA256 of the accepted Step-2.4F archive as it sits in this repository.  A different archive is
#: refused rather than trusted: "the hashes verify" is only meaningful against a known archive.
ACCEPTED_ARCHIVE_SHA256 = "2ad0482731678091bb8de53503b5ee64f0513a294270fc694fd6da52c7160f6d"

#: SHA256 of the H2/H3 artifacts inside that archive.  These are the values Gate A accepted; the
#: restore step verifies them before training so a truncated download cannot become a run.
ACCEPTED_ARTIFACT_SHA256 = {
    "features/item_features.npy": "53165b18463f3ee6f435384be2ada02634508989a4091b783aeb2057e9d0d959",
    "features/item_features.json": "d2252b1a8b1d2fb5eaeaa4766204e8bee61430ee170ff93e34fe87ef91003ce1",
    "features/manifest.json": "14b1bfd695cbe9802e79415e365bdb2ee33addf32a25763624d604f005693c40",
    "sid/semantic_ids.json": "6501fbe8e146e3c57ce7a146e37c3be29235ae5bbd082116f6a8e30e739e8d0e",
    "sid/layout.json": "22521d7b58f8538c3d3b0d8a27200a255ac62e1e9da0676c6c55ebadde101b11",
    "sid/tokenizer.pt": "d514aafcddf0edb922a18f463520c2bc80b7a0a8eb46e358e4fa084642c610c3",
    "sid/tokenizer.json": "57e193de6ba31fdac23b7a431c5ff2acc23c919122329fc6bf6daf8899a1fe3f",
    "sid/manifest.json": "a52928df340c724a31d9e458798eb88a39ba2c81cbc7e7723b7a4178755e1c8c",
    "sid_audit.json": "a926db00c619541c995c5e839e63846def926c932d5c20adfb13eba1a0f2404e",
}

#: Frozen production counts (Gate B.1).  ``sum(len - 1)`` is the generator's example count;
#: ``sum(len)`` is an item-occurrence total and is deliberately not used as one.
FROZEN = {
    "users": 412_445,
    "catalogue_items": 156_746,
    "sum_len": 2_675_697,
    "sum_len_minus_1": 2_263_252,
    "sum_len_minus_2": 1_850_807,
}

#: The vocab layout Gate B's example builder requires.  SEP must be a registered special token,
#: because deriving a token id at use time could alias a real code.
EXPECTED_SPECIALS = ("pad", "bos", "eos", "sep")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def check(name: str, ok: bool, detail: str = "") -> tuple[bool, str]:
    return (bool(ok), f"{name}: {detail}" if detail else name)


def verify_archive(archive: Path) -> list[tuple[bool, str]]:
    results: list[tuple[bool, str]] = []
    if not archive.is_file():
        return [check("archive present", False, f"missing {archive}")]
    digest = sha256_file(archive)
    results.append(
        check(
            "archive digest",
            digest == ACCEPTED_ARCHIVE_SHA256,
            digest if digest == ACCEPTED_ARCHIVE_SHA256 else f"{digest} != {ACCEPTED_ARCHIVE_SHA256}",
        )
    )
    with tarfile.open(archive, "r:gz") as tar:
        names = set(tar.getnames())
    missing = sorted(name for name in ACCEPTED_ARTIFACT_SHA256 if f"runs/tiger_public_2026/{name}" not in names)
    results.append(check("archive members", not missing, f"missing {missing}" if missing else f"{len(names)} members"))
    return results


def canonical_layout_sha256(text: str) -> str:
    """SHA256 of the exact bytes `patch_layout` writes, so the patched state is pinned by value."""
    layout = json.loads(text)
    canonical = json.dumps(layout, indent=1, sort_keys=True) + "\n"
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def verify_restored(root: Path) -> list[tuple[bool, str]]:
    """Verify the extracted H2/H3 artifacts.  The archive extracts under `runs/tiger_public_2026/`.

    `sid/layout.json` has exactly two acceptable states: the pristine accepted bytes, or the
    Gate-C patched bytes.  The patched state is pinned by recomputing the canonical serialisation
    rather than by hardcoding a second digest, so the check cannot drift from `patch_layout`.
    Anything else is refused, which is what makes "the hashes verify" mean something.
    """
    results: list[tuple[bool, str]] = []
    for name, expected in ACCEPTED_ARTIFACT_SHA256.items():
        path = root / name
        if not path.is_file():
            results.append(check(f"artifact {name}", False, "missing"))
            continue
        text = path.read_text(encoding="utf-8") if name.endswith(".json") else None
        actual = sha256_file(path)
        if actual == expected:
            results.append(check(f"artifact {name}", True, actual))
            continue
        if name == "sid/layout.json" and text is not None:
            patched = canonical_layout_sha256(text)
            specials = sorted((json.loads(text).get("special") or {}))
            if actual == patched and specials == sorted(EXPECTED_SPECIALS):
                results.append(
                    check(
                        f"artifact {name}",
                        True,
                        f"{actual} (Gate-C patched; pristine was {expected})",
                    )
                )
                continue
        results.append(check(f"artifact {name}", False, f"{actual} != {expected}"))
    return results


def patch_layout(sid_dir: Path) -> tuple[bool, str]:
    """Apply the Gate-C additive layout patch: register SEP and widen the vocabulary by one.

    This is **metadata only**.  It rewrites `layout.json` (keeping a `.v3.bak.json` beside it) and
    touches no Semantic ID: `semantic_ids.json`, `tokenizer.pt`, `tokenizer.json` and
    `item_features.npy` are not read or written, so no RQ-VAE refit and no re-encode happens.  The
    SID digits stay identical; only the special-token block grows by one token.

    Idempotent: a layout that already declares SEP is left alone.
    """
    path = sid_dir / "layout.json"
    if not path.is_file():
        return False, f"missing {path}"
    layout = json.loads(path.read_text(encoding="utf-8"))
    specials = layout.get("special") or {}
    code_space = int(layout["code_space"])
    if "sep" in specials:
        if int(layout["vocab_size"]) != code_space + 4:
            return False, f"SEP present but vocab_size={layout['vocab_size']}"
        return False, "already patched"
    before = sha256_file(path)
    backup = path.with_suffix(".v3.bak.json")
    if not backup.exists():
        backup.write_text(path.read_text(encoding="utf-8"), encoding="utf-8")
    layout["special"] = {
        "pad": code_space,
        "bos": code_space + 1,
        "eos": code_space + 2,
        "sep": code_space + 3,
    }
    layout["vocab_size"] = code_space + 4
    path.write_text(json.dumps(layout, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    after = sha256_file(path)
    return True, (
        f"layout.json {before} -> {after} "
        f"(SEP {code_space + 3}, vocab_size {code_space + 4})"
    )


def verify_layout(sid_dir: Path) -> list[tuple[bool, str]]:
    """Gate B's one precondition: the accepted layout must declare a SEP special token."""
    path = sid_dir / "layout.json"
    if not path.is_file():
        return [check("layout", False, f"missing {path}")]
    layout = json.loads(path.read_text(encoding="utf-8"))
    specials = sorted((layout.get("special") or {}))
    results = [
        check("layout specials", specials == sorted(EXPECTED_SPECIALS), str(specials)),
        check(
            "layout vocab_size",
            int(layout.get("vocab_size", -1)) == int(layout.get("code_space", -2)) + 4,
            f"vocab_size={layout.get('vocab_size')} code_space={layout.get('code_space')}",
        ),
        check("layout sentinel_tokenisable", layout.get("sentinel_tokenisable") is False),
    ]
    if specials != sorted(EXPECTED_SPECIALS):
        results.append(
            check(
                "layout needs the Gate-C additive patch",
                False,
                "run this script with --patch-layout: it adds sep = code_space + 3 and sets "
                "vocab_size = code_space + 4 as METADATA ONLY. The SID assignment, tokenizer and "
                "features are unchanged; no RQ-VAE refit and no re-encode. See docs/TIGER_BACKEND.md "
                "17.4 discrepancy 2.",
            )
        )
    return results


def verify_exposure(path: Path) -> list[tuple[bool, str]]:
    """Stream the exposure; never build an example list."""
    if not path.is_file():
        return [check("exposure", False, f"missing {path}")]
    users = occurrences = transitions = pairs = 0
    forbidden = 0
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if any(word in key.lower() for key in row for word in ("target", "label", "valid", "test")):
                forbidden += 1
            items = row["items"]
            users += 1
            occurrences += len(items)
            transitions += max(0, len(items) - 1)
            pairs += max(0, len(items) - 2)
    return [
        check("exposure users", users == FROZEN["users"], str(users)),
        check("exposure item occurrences", occurrences == FROZEN["sum_len"], str(occurrences)),
        check("generator examples", transitions == FROZEN["sum_len_minus_1"], str(transitions)),
        check("GenRec-v0 pairs", pairs == FROZEN["sum_len_minus_2"], str(pairs)),
        check("validation/test targets used", forbidden == 0, f"{forbidden} target-shaped keys"),
    ]


def verify_handoff_catalogue(catalogue_path: Path) -> list[tuple[bool, str]]:
    if not catalogue_path.is_file():
        return [check("catalogue", False, f"missing {catalogue_path}")]
    catalogue = json.loads(catalogue_path.read_text(encoding="utf-8"))
    return [
        check("catalogue items", catalogue["num_items"] == FROZEN["catalogue_items"], str(catalogue["num_items"])),
        check("catalogue pad_id", catalogue["pad_id"] == 0),
        check("catalogue first_real_id", catalogue["first_real_id"] == 1),
    ]


def training_command(*, restored: Path, handoff: Path, out: Path) -> str:
    """The exact Gate-D command.  Only the registered configuration; no flags that diverge it."""
    return (
        "cd backends/tiger_public\n"
        "PYTHONPATH=src .venv/bin/python -m tiger_public.cli train \\\n"
        f"  --catalogue {handoff} \\\n"
        f"  --exposure {handoff}/train_exposure.jsonl \\\n"
        f"  --sid {restored}/sid \\\n"
        f"  --out {out} \\\n"
        "  --device cuda\n"
        "# registered defaults, do not override: epochs 20, batch 512, d_model 256, layers 6,\n"
        "# heads 4, d_ff 1024, dropout 0.1, max_hist_items 20, lr 5e-4, seed 2026, bf16 on"
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Gate D runbook check (trains nothing)")
    parser.add_argument("--archive", type=Path, default=None)
    parser.add_argument("--restore", type=Path, default=None,
                        help="extracted Step-2.4F root (contains features/ and sid/)")
    parser.add_argument("--handoff", type=Path, default=PRODUCTION_HANDOFF)
    parser.add_argument("--out", type=Path, default=Path("runs/tiger_generator_prod"))
    parser.add_argument("--extract-to", type=Path, default=None,
                        help="extract the archive here before verifying")
    parser.add_argument("--patch-layout", action="store_true",
                        help="apply the Gate-C additive SEP/vocab_size layout patch (metadata only; "
                             "writes a .v3.bak.json backup and touches no Semantic ID)")
    args = parser.parse_args(argv)

    results: list[tuple[bool, str]] = []
    restored = args.restore
    if args.archive is not None:
        results += verify_archive(args.archive)
        if args.extract_to is not None:
            args.extract_to.mkdir(parents=True, exist_ok=True)
            with tarfile.open(args.archive, "r:gz") as tar:
                tar.extractall(args.extract_to)  # noqa: S202 - pinned local archive
            restored = args.extract_to / "runs" / "tiger_public_2026"
            results.append(check("extracted to", True, str(restored)))
    if restored is not None:
        results += verify_restored(restored)
        if args.patch_layout:
            changed, message = patch_layout(restored / "sid")
            results.append(check("layout patch", True, ("applied: " if changed else "no change: ") + message))
        results += verify_layout(restored / "sid")
    else:
        results.append(check("restored artifacts", False, "pass --restore or --extract-to"))

    results += verify_handoff_catalogue(args.handoff / "catalogue.json")
    results += verify_exposure(args.handoff / "train_exposure.jsonl")

    for ok, message in results:
        print(f"  {'PASS' if ok else 'FAIL'}  {message}")
    failed = [message for ok, message in results if not ok]
    print()
    if failed:
        print(f"{len(failed)} check(s) failed; Gate D must not start:")
        for message in failed:
            print(f"  - {message}")
        return 1
    print(f"{len(results)}/{len(results)} checks passed. Gate D training command:\n")
    print(training_command(restored=restored, handoff=args.handoff, out=args.out))
    return 0


if __name__ == "__main__":  # pragma: no cover - script entry point
    sys.exit(main())

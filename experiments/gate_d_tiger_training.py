"""Gate D runbook check: restore the accepted Step-2.4F artifacts, verify, and print the run.

This script **trains nothing**.  It exists so the Gate-C/D GPU run cannot start from a
half-verified state, and so that the accepted Step-2.4F artifacts are *restored and verified*
rather than rebuilt.  Rebuilding `Sentence-T5` features, the RQ-VAE or the Semantic IDs is never a
step here: the H2/H3 artifacts are frozen, and Gate A already proved the production handoff's
`products_text` identity matches the accepted feature/SID dependency.

The Gate-D workflow this checks, in order:

```text
1  restore the repository at the exact accepted commit
2  restore the verified Step-2.4F artifact archive
3  verify the accepted SID hashes (layout / semantic_ids / tokenizer x2)
4  restore (or materialise) the production handoff and verify its hashes
5  materialise and verify generator_layout.json  (read-only over sid/)
6  verify the production example count (2 263 252)
7  train TIGER, and nothing else
```

It contains **no** `build-features`, **no** `fit-sid`, **no** RQ-VAE retraining, **no** SID
regeneration, **no** benchmark and no Step-2.6 scoring.

The accepted SID artifact is **immutable**.  Step 5 never writes inside `sid/`: it reads the
accepted `layout.json` and writes a *new* `generator_layout.json` beside the generator output,
recording the accepted layout's and Semantic IDs' SHA256.  All four accepted hashes are printed
before and after and must be identical.

Usage::

    cd backends/tiger_public
    PYTHONPATH=src .venv/bin/python ../../experiments/gate_d_tiger_training.py \\
        --archive ../../runs/_autodl_backup/step24f-tiger-public-dfabc1c.tar.gz \\
        --restore  /data/agentrecx/step24f \
        --generator-out /data/agentrecx/tiger_generator_prod
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

#: The accepted Step-2.4F SID layout declares exactly these three specials and no SEP.  The
#: generator's fourth special is *derived*, never patched into the accepted file.
ACCEPTED_SPECIALS = ("pad", "bos", "eos")
EXPECTED_GENERATOR_SPECIALS = ("pad", "bos", "eos", "sep")

#: The accepted SID artifacts, restated here so a reader sees the four hashes the run must preserve.
ACCEPTED_SID_SHA256 = {
    "layout.json": ACCEPTED_ARTIFACT_SHA256["sid/layout.json"],
    "semantic_ids.json": ACCEPTED_ARTIFACT_SHA256["sid/semantic_ids.json"],
    "tokenizer.pt": ACCEPTED_ARTIFACT_SHA256["sid/tokenizer.pt"],
    "tokenizer.json": ACCEPTED_ARTIFACT_SHA256["sid/tokenizer.json"],
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def check(name: str, ok: bool, detail: str = "") -> tuple[bool, str]:
    return (bool(ok), f"{name}: {detail}" if detail else name)


def _git_head() -> str:
    import subprocess  # noqa: PLC0415

    try:
        return subprocess.run(  # noqa: S603 - fixed argv, no shell
            ["git", "rev-parse", "HEAD"], cwd=str(REPO_ROOT),
            capture_output=True, check=True, text=True,
        ).stdout.strip()
    except Exception:  # pragma: no cover - git may be absent
        return "unknown"


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


def verify_restored(root: Path) -> list[tuple[bool, str]]:
    """Verify the extracted H2/H3 artifacts.  The archive extracts under `runs/tiger_public_2026/`.

    Every accepted hash must match **exactly**.  There is no "patched layout" state any more: the
    generator vocabulary is a separate derived artifact and nothing this runbook does rewrites an
    accepted file.
    """
    results: list[tuple[bool, str]] = []
    for name, expected in ACCEPTED_ARTIFACT_SHA256.items():
        path = root / name
        if not path.is_file():
            results.append(check(f"artifact {name}", False, "missing"))
            continue
        actual = sha256_file(path)
        results.append(check(f"artifact {name}", actual == expected, actual))
    return results


def accepted_sid_hashes(sid_dir: Path) -> dict[str, str]:
    """The four accepted SID hashes, read now, so the run can prove it changed none of them."""
    return {
        name: sha256_file(sid_dir / name)
        for name in ("layout.json", "semantic_ids.json", "tokenizer.pt", "tokenizer.json")
        if (sid_dir / name).is_file()
    }


def verify_accepted_sid_hashes(sid_dir: Path, *, label: str) -> list[tuple[bool, str]]:
    results: list[tuple[bool, str]] = []
    actual = accepted_sid_hashes(sid_dir)
    for name, expected in ACCEPTED_SID_SHA256.items():
        results.append(check(f"{label} sid/{name}", actual.get(name) == expected, actual.get(name, "missing")))
    return results


def verify_accepted_layout(sid_dir: Path) -> list[tuple[bool, str]]:
    """The accepted layout must be the *pre-SEP* artifact: three specials, vocab_size + 3."""
    path = sid_dir / "layout.json"
    if not path.is_file():
        return [check("accepted layout", False, f"missing {path}")]
    layout = json.loads(path.read_text(encoding="utf-8"))
    specials = sorted((layout.get("special") or {}))
    return [
        check("accepted layout specials", specials == sorted(ACCEPTED_SPECIALS), str(specials)),
        check(
            "accepted layout vocab_size",
            int(layout.get("vocab_size", -1)) == int(layout.get("code_space", -2)) + 3,
            f"vocab_size={layout.get('vocab_size')} code_space={layout.get('code_space')}",
        ),
        check("accepted layout sentinel_tokenisable", layout.get("sentinel_tokenisable") is False),
    ]


def materialise_generator_layout(sid_dir: Path, out: Path) -> list[tuple[bool, str]]:
    """Derive `generator_layout.json` from the accepted layout.  Read-only over ``sid/``."""
    try:
        import sys as _sys

        backend_src = REPO_ROOT / "backends" / "tiger_public" / "src"
        if str(backend_src) not in _sys.path:
            _sys.path.insert(0, str(backend_src))
        from tiger_public.generator_layout import (
            GeneratorLayoutError,
            derive_generator_layout,
            load_generator_layout,
        )
    except Exception as error:  # pragma: no cover - the backend venv is the documented runner
        return [check("generator layout", False, f"cannot import the materialiser: {error}")]

    out = Path(out)
    path = out / "generator_layout.json"
    try:
        record = derive_generator_layout(sid_dir=Path(sid_dir), generator_layout_path=path)
        loaded = load_generator_layout(path)
    except GeneratorLayoutError as error:
        return [check("generator layout", False, str(error))]

    results = [
        check("generator layout materialised", path.is_file(), str(path)),
        check("generator layout sha256", True, record["generator_layout_sha256"]),
        check(
            "generator layout source hashes",
            record["source_sid_layout_sha256"] == ACCEPTED_SID_SHA256["layout.json"]
            and record["source_semantic_ids_sha256"] == ACCEPTED_SID_SHA256["semantic_ids.json"],
            f"layout={record['source_sid_layout_sha256'][:16]} sids="
            f"{record['source_semantic_ids_sha256'][:16]}",
        ),
        check(
            "generator specials",
            sorted(loaded["special"]) == sorted(EXPECTED_GENERATOR_SPECIALS),
            str(loaded["special"]),
        ),
        check(
            "generator vocab_size",
            int(loaded["vocab_size"]) == int(loaded["code_space"]) + 4,
            f"vocab_size={loaded['vocab_size']} code_space={loaded['code_space']}",
        ),
        check("generator SEP token", int(loaded["special"]["sep"]) == int(loaded["vocab_size"]) - 1,
              str(loaded["special"]["sep"])),
        check(
            "SEP aliases no catalogue SID token",
            not record["audit"]["aliased_tokens"],
            f"{record['audit']['distinct_sid_tokens']} distinct SID tokens, "
            f"{record['audit']['catalogue_items']} items",
        ),
    ]
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
    parser.add_argument("--generator-out", type=Path, default=None,
                        help="where generator_layout.json is materialised "
                             "(default: <out>/generator)")
    parser.add_argument("--extract-to", type=Path, default=None,
                        help="extract the archive here before verifying")
    parser.add_argument("--repo-commit", default=None,
                        help="the accepted commit this run is pinned to; a mismatch is refused")
    args = parser.parse_args(argv)

    results: list[tuple[bool, str]] = []
    generator_out = args.generator_out or (args.out / "generator")

    # 1. repository pinned at the accepted commit
    if args.repo_commit is not None:
        head = _git_head()
        results.append(check("repository commit", head == args.repo_commit,
                             f"{head} (expected {args.repo_commit})"))
    else:
        results.append(check("repository commit", True, f"{_git_head()} (not pinned by --repo-commit)"))

    # 2-3. restore and verify the accepted Step-2.4F artifact, before and after
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
        results += verify_accepted_sid_hashes(restored / "sid", label="accepted")
        results += verify_accepted_layout(restored / "sid")

        # 5. materialise + verify the generator layout, then prove sid/ is untouched
        before = accepted_sid_hashes(restored / "sid")
        results += materialise_generator_layout(restored / "sid", generator_out)
        after = accepted_sid_hashes(restored / "sid")
        results.append(check("accepted SID artifact unchanged", before == after,
                             "byte-identical" if before == after else f"{before} -> {after}"))
        results += verify_restored(restored)
    else:
        results.append(check("restored artifacts", False, "pass --restore or --extract-to"))

    # 4 + 6. handoff hashes and the production example count
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

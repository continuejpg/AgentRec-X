#!/usr/bin/env python3
"""Audit adapted-upstream files in the AgentRec-X web demo.

What it checks
--------------
1. Every file under the given root that carries the upstream Apache-2.0 marker
   ("Copyright 2026 Anthropic PBC") must also carry:
     - "SPDX-License-Identifier: Apache-2.0" in the same comment header;
     - a modification note ("Modified for AgentRec-X Web Demo ...").
2. If a notices file is given, every adapted file path must also appear in it
   (the path is looked up relative to the repository root).
3. The verbatim Apache-2.0 license copy exists in third_party/licenses/ and its
   SHA-256 matches the hash of the upstream LICENSE at the recorded commit.
4. No copied upstream product imagery: any image under the audited tree is
   reported for manual review, and if `--refs` points at the read-only upstream
   clone, byte-identical images are flagged as violations.

5. New, non-upstream frontend modules that carry the project provenance header
   ("new frontend module for the local shopping-assistant demo") must reference
   THIRD_PARTY_NOTICES.md, so the attribution pointer stays live.

The script is read-only. Exit codes: 0 = pass, 1 = compliance issue, 2 = target
tree not found (e.g. frontend not built yet).

Usage
-----
    python third_party/tools/check_copied_headers.py --root web
    python third_party/tools/check_copied_headers.py --root web \\
        --notices THIRD_PARTY_NOTICES.md \\
        --refs D:/IT/CODE/PROJECTION/agentrec/_refs/commerce-agents \\
        --json-out third_party/header-audit.json

Only the Python standard library is used.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

UPSTREAM_COPYRIGHT = "Copyright 2026 Anthropic PBC"
# A file counts as adapted only when the copyright line is the *header statement* itself,
# not when the phrase is merely mentioned in a provenance note such as
# "keep their Copyright 2026 Anthropic PBC + SPDX headers". Accepted forms:
#   // Copyright 2026 Anthropic PBC
#    * Copyright 2026 Anthropic PBC
#   /* Copyright 2026 Anthropic PBC */
UPSTREAM_COPYRIGHT_RE = re.compile(
    r"^\s*(?://|/\*+|\*)\s*Copyright 2026 Anthropic PBC\s*(?:\*/)?\s*$", re.MULTILINE
)
UPSTREAM_SPDX = "SPDX-License-Identifier: Apache-2.0"
# Header used by files that were written for this demo and contain no copied upstream code.
NEW_MODULE_MARKER = "new frontend module for the local shopping-assistant demo"
MODIFICATION_RE = re.compile(r"Modified for AgentRec-X", re.IGNORECASE)
UPSTREAM_LICENSE_SHA256 = "3DDF9BE5C28FE27DAD143A5DC76EEA25222AD1DD68934A047064E56ED2FA40C5"
LICENSE_REL = Path("third_party/licenses/commerce-agents-Apache-2.0.txt")

AUDITED_SUFFIXES = {".ts", ".tsx", ".js", ".jsx", ".mjs", ".cjs", ".css", ".scss", ".md", ".mdx", ".py"}
IMAGE_SUFFIXES = {".webp", ".png", ".jpg", ".jpeg", ".gif", ".avif", ".svg"}
HEADER_WINDOW = 40  # lines searched for the header when the marker appears later in a file


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 256), b""):
            digest.update(chunk)
    return digest.hexdigest()


def header_block(text: str) -> str:
    """Return the leading comment/license header of a source file.

    Covers both cases: a header at the very top, or a header that follows an
    optional shebang / 'use client' / imports (search the first HEADER_WINDOW
    lines for the copyright marker and include the surrounding comments).
    """
    lines = text.splitlines()
    marker = None
    for index, line in enumerate(lines[:HEADER_WINDOW]):
        if UPSTREAM_COPYRIGHT_RE.match(line):
            marker = index
            break
    if marker is None:
        return ""
    start = max(0, marker - 2)
    end = min(len(lines), marker + 8)
    return "\n".join(lines[start:end])


def iter_files(root: Path):
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        if any(part in {".git", "node_modules", ".next", ".turbo"} for part in path.parts):
            continue
        yield path


def check_adapted_headers(root: Path, repo_root: Path, notices: Path | None) -> list[dict]:
    problems: list[dict] = []
    adapted: list[str] = []
    for path in iter_files(root):
        try:
            text = path.read_text(encoding="utf-8", errors="strict")
        except (UnicodeDecodeError, OSError):
            continue
        if not UPSTREAM_COPYRIGHT_RE.search(text):
            continue
        rel = path.relative_to(repo_root).as_posix() if path.is_relative_to(repo_root) else path.as_posix()
        adapted.append(rel)
        block = header_block(text)
        missing = []
        if not UPSTREAM_COPYRIGHT_RE.search(block):
            missing.append("copyright header")
        if UPSTREAM_SPDX not in block:
            missing.append("SPDX-License-Identifier: Apache-2.0")
        if not MODIFICATION_RE.search(block):
            missing.append("modification note (Modified for AgentRec-X Web Demo ...)")
        if missing:
            problems.append({"file": rel, "missing": missing})
    if notices is not None and notices.is_file():
        notices_text = notices.read_text(encoding="utf-8", errors="replace")
        for rel in adapted:
            if rel not in notices_text:
                problems.append({"file": rel, "missing": ["entry in THIRD_PARTY_NOTICES.md"]})
    return problems


def check_new_module_headers(root: Path, repo_root: Path) -> tuple[list[str], list[dict]]:
    """Files with the project (non-upstream) provenance header must point at the notices."""
    modules: list[str] = []
    problems: list[dict] = []
    for path in iter_files(root):
        try:
            text = path.read_text(encoding="utf-8", errors="strict")
        except (UnicodeDecodeError, OSError):
            continue
        head = "\n".join(text.splitlines()[:8])
        if NEW_MODULE_MARKER not in head:
            continue
        rel = path.relative_to(repo_root).as_posix() if path.is_relative_to(repo_root) else path.as_posix()
        modules.append(rel)
        if "THIRD_PARTY_NOTICES.md" not in head:
            problems.append({"file": rel, "missing": ["THIRD_PARTY_NOTICES.md pointer in project header"]})
    return modules, problems


def check_imagery(root: Path, refs: Path | None) -> tuple[list[str], list[dict]]:
    images: list[str] = []
    violations: list[dict] = []
    ref_hashes: dict[str, Path] = {}
    if refs is not None and refs.is_dir():
        for path in refs.rglob("*"):
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
                try:
                    ref_hashes.setdefault(sha256_file(path), path)
                except OSError:
                    pass
    for path in iter_files(root):
        if path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        images.append(path.as_posix())
        if ref_hashes:
            try:
                digest = sha256_file(path)
            except OSError:
                continue
            if digest in ref_hashes:
                violations.append({"file": path.as_posix(), "identical_to": ref_hashes[digest].as_posix()})
    return images, violations


def check_license_copy(repo_root: Path) -> list[str]:
    license_path = repo_root / LICENSE_REL
    problems: list[str] = []
    if not license_path.is_file():
        problems.append(f"missing license copy: {LICENSE_REL.as_posix()}")
        return problems
    digest = sha256_file(license_path).upper()
    if digest != UPSTREAM_LICENSE_SHA256:
        problems.append(
            f"{LICENSE_REL.as_posix()} SHA-256 {digest} != upstream {UPSTREAM_LICENSE_SHA256}"
        )
    return problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", default="web", help="tree to audit (default: web)")
    parser.add_argument("--repo-root", default=".", help="repository root (default: cwd)")
    parser.add_argument("--notices", default="THIRD_PARTY_NOTICES.md", help="notices file to cross-check ('' disables)")
    parser.add_argument("--refs", default="", help="read-only upstream clone used to detect copied imagery")
    parser.add_argument("--json-out", default="", help="optional JSON report path")
    args = parser.parse_args(argv)

    repo_root = Path(args.repo_root).resolve()
    root = (repo_root / args.root).resolve() if not Path(args.root).is_absolute() else Path(args.root)
    notices = Path(args.notices) if args.notices else None
    refs = Path(args.refs).resolve() if args.refs else None

    report: dict = {
        "root": str(root),
        "repo_root": str(repo_root),
        "status": "unknown",
        "adapted_files": [],
        "new_modules": [],
        "header_problems": [],
        "new_module_problems": [],
        "notices_problems": [],
        "license_problems": [],
        "images_found": [],
        "image_violations": [],
    }

    if not root.is_dir():
        report["status"] = "target-not-found"
        print(f"[check_copied_headers] target tree not found: {root}")
        print("[check_copied_headers] (expected before the frontend creates web/)")
        return 2

    adapted_problems = check_adapted_headers(root, repo_root, notices if notices and notices.is_file() else None)
    adapted_files: list[str] = []
    for path in iter_files(root):
        try:
            if UPSTREAM_COPYRIGHT_RE.search(path.read_text(encoding="utf-8", errors="strict")):
                adapted_files.append(path.relative_to(repo_root).as_posix() if path.is_relative_to(repo_root) else str(path))
        except (UnicodeDecodeError, OSError):
            continue

    report["adapted_files"] = sorted(adapted_files)
    report["header_problems"] = adapted_problems
    new_modules, new_module_problems = check_new_module_headers(root, repo_root)
    report["new_modules"] = sorted(new_modules)
    report["new_module_problems"] = new_module_problems
    report["license_problems"] = check_license_copy(repo_root)
    images, violations = check_imagery(root, refs)
    report["images_found"] = images
    report["image_violations"] = violations

    ok = (
        not adapted_problems
        and not new_module_problems
        and not report["license_problems"]
        and not violations
    )
    report["status"] = "pass" if ok else "fail"

    print(f"[check_copied_headers] root={root}")
    print(f"[check_copied_headers] adapted files with upstream copyright: {len(adapted_files)}")
    for rel in adapted_files:
        print(f"  - {rel}")
    print(f"[check_copied_headers] new modules with project provenance header: {len(new_modules)}")
    for rel in new_modules:
        print(f"  - {rel}")
    if adapted_problems:
        print("[check_copied_headers] FAIL header/notice problems:")
        for problem in adapted_problems:
            print(f"  ! {problem['file']}: missing {', '.join(problem['missing'])}")
    if new_module_problems:
        print("[check_copied_headers] FAIL project-header problems:")
        for problem in new_module_problems:
            print(f"  ! {problem['file']}: missing {', '.join(problem['missing'])}")
    if report["license_problems"]:
        print("[check_copied_headers] FAIL license copy problems:")
        for problem in report["license_problems"]:
            print(f"  ! {problem}")
    if violations:
        print("[check_copied_headers] FAIL copied upstream imagery:")
        for violation in violations:
            print(f"  ! {violation['file']} == {violation['identical_to']}")
    if images:
        print(f"[check_copied_headers] images present under {args.root} ({len(images)}), manual review:")
        for image in images:
            print(f"  - {image}")
    print(f"[check_copied_headers] status: {report['status'].upper()}")

    if args.json_out:
        out = Path(args.json_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"[check_copied_headers] json report: {out}")

    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

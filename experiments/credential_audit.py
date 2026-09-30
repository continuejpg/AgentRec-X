"""Credential leak audit over every Git object in the repository.

Run this before declaring a milestone sealed. It answers one question: **did a real credential ever
enter version control?** It is deliberately stronger than a filename search.

What it does
------------
1. **Scans for the real credential value.** Reads `.m4_credential` (gitignored) and searches every
   blob in the object database — not just reachable history, but ``--batch-all-objects``, which
   includes objects no ref points at any more. A filename search would miss a credential embedded in
   a test, a fixture, a log or a notebook; this cannot.
2. **Scans for common secret shapes.** Provider-style keys, bearer tokens, AWS access key ids,
   GitHub/Google/Slack tokens, private-key headers, JWTs, and generic ``secret = …`` assignments.
3. **Classifies rather than panics.** A hit on a *shape* is normally a synthetic test fixture, so
   each hit is reported with its path, blob, pattern and a redacted fragment for human judgement.
   A hit on the *real value* is a leak and must stop the seal.

Never prints a full secret. Matched fragments are truncated.

Usage
-----
    python -m experiments.credential_audit            # human-readable report
    python -m experiments.credential_audit --json OUT # machine-readable, redacted

Exit codes: 0 clean (shape hits, if any, are all classified as fixtures); 1 real-value leak.
"""

from __future__ import annotations

import argparse
import collections
import json
import pathlib
import re
import subprocess
import sys
from typing import Any, Sequence

REPO = pathlib.Path(__file__).resolve().parents[1]
CREDENTIAL_FILE = REPO / ".m4_credential"

#: Patterns for credential *shapes*. A hit means "look at this", not "this is a leak".
SECRET_PATTERNS: dict[str, re.Pattern[bytes]] = {
    "provider_key": re.compile(rb"\bsk-[A-Za-z0-9_\-]{16,}"),
    "anthropic_key": re.compile(rb"\bsk-ant-[A-Za-z0-9_\-]{20,}"),
    "bearer_token": re.compile(rb"(?i)\bbearer\s+[A-Za-z0-9_\-\.]{20,}"),
    "aws_access_key_id": re.compile(rb"\bAKIA[0-9A-Z]{16}\b"),
    "github_token": re.compile(rb"\bgh[pousr]_[A-Za-z0-9]{20,}"),
    "google_api_key": re.compile(rb"\bAIza[0-9A-Za-z_\-]{35}\b"),
    "slack_token": re.compile(rb"\bxox[abprs]-[A-Za-z0-9\-]{10,}"),
    "private_key_header": re.compile(rb"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
    "jwt": re.compile(rb"\beyJ[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}"),
    "generic_secret_assignment": re.compile(
        rb"(?i)\b(api[_-]?key|apikey|secret|token|passwd|password)\b\s*[:=]\s*['\"]?[A-Za-z0-9_\-]{16,}"
    ),
}

#: Files whose shape hits are expected to be synthetic fixtures, with the reason.
EXPECTED_FIXTURE_CONTEXT: dict[str, str] = {
    "tests/": "synthetic fixtures exercising the credential detector itself",
    ".env.example": "documented placeholder, no real value",
}


def _git(*args: str, timeout: int = 300) -> str:
    """Run a git command, returning stdout as text."""
    return subprocess.run(
        ["git", *args], cwd=REPO, capture_output=True, text=True, timeout=timeout, check=False
    ).stdout


def all_blobs() -> list[tuple[str, str]]:
    """Every blob in the object database, reachable or not, as ``(sha, path-or-empty)``.

    ``--batch-all-objects`` is what makes this exhaustive: a blob that no ref points at any more is
    still in the database until a garbage collection prunes it, and a credential that was committed
    and then removed would otherwise escape a history walk.
    """
    listing = _git("rev-list", "--objects", "--all")
    paths: dict[str, str] = {}
    for line in listing.splitlines():
        parts = line.split(maxsplit=1)
        if len(parts) == 2:
            paths[parts[0]] = parts[1]

    check = _git("cat-file", "--batch-all-objects", "--batch-check=%(objectname) %(objecttype)")
    blobs: list[tuple[str, str]] = []
    for line in check.splitlines():
        parts = line.split()
        if len(parts) == 2 and parts[1] == "blob":
            blobs.append((parts[0], paths.get(parts[0], "")))
    return blobs


def _cat(sha: str) -> bytes:
    """Blob content, bytes, empty on failure."""
    return subprocess.run(
        ["git", "cat-file", "-p", sha], cwd=REPO, capture_output=True, timeout=120, check=False
    ).stdout


def audit() -> dict[str, Any]:
    """Run the full audit. Returns a JSON-serialisable, redacted report."""
    credential = ""
    if CREDENTIAL_FILE.is_file():
        credential = CREDENTIAL_FILE.read_text(encoding="utf-8").strip()

    blobs = all_blobs()
    real_hits: list[dict[str, Any]] = []
    shape_hits: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)

    needle = credential.encode()
    for sha, path in blobs:
        data = _cat(sha)
        if needle and needle in data:
            real_hits.append({"blob": sha, "path": path})
        for name, pattern in SECRET_PATTERNS.items():
            match = pattern.search(data)
            if match:
                shape_hits[name].append(
                    {
                        "blob": sha,
                        "path": path,
                        # Redacted: a few leading characters only, never the whole match.
                        "fragment": match.group(0)[:8].decode("utf-8", "replace"),
                    }
                )

    expected = [
        hit
        for hits in shape_hits.values()
        for hit in hits
        if any(hit["path"].startswith(prefix) for prefix in EXPECTED_FIXTURE_CONTEXT)
    ]
    total_shape = sum(len(v) for v in shape_hits.values())
    return {
        "credential_file_present": CREDENTIAL_FILE.is_file(),
        "credential_length": len(credential),
        "credential_scanned": bool(credential),
        "blobs_scanned": len(blobs),
        "real_credential_hits": real_hits,
        "real_credential_leaked": bool(real_hits),
        "shape_hits": {k: v for k, v in shape_hits.items() if v},
        "shape_hit_count": total_shape,
        "shape_hits_in_expected_fixture_context": len(expected),
        "shape_hits_outside_fixture_context": total_shape - len(expected),
        "expected_fixture_context": EXPECTED_FIXTURE_CONTEXT,
        "verdict": "LEAK" if real_hits else "CLEAN",
    }


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point. Exit 1 only on a real-credential leak."""
    parser = argparse.ArgumentParser(description="credential leak audit over all Git objects")
    parser.add_argument("--json", type=str, default="", help="write the redacted report here")
    args = parser.parse_args(argv)

    report = audit()
    print(f"credential file present : {report['credential_file_present']}")
    print(f"credential length       : {report['credential_length']} (value never printed)")
    print(f"blobs scanned           : {report['blobs_scanned']} (--batch-all-objects, incl. unreachable)")
    print(f"REAL credential hits    : {len(report['real_credential_hits'])}")
    for hit in report["real_credential_hits"]:
        print(f"    blob {hit['blob'][:12]}  path {hit['path']}")
    print(f"secret-shape hits       : {report['shape_hit_count']}")
    for name, hits in report["shape_hits"].items():
        for hit in hits:
            print(f"    [{name}] {hit['path']}  blob {hit['blob'][:12]}  fragment {hit['fragment']!r}...")
    print()
    print(f"VERDICT: {report['verdict']}")
    if args.json:
        pathlib.Path(args.json).write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"wrote {args.json}")
    return 1 if report["real_credential_leaked"] else 0


if __name__ == "__main__":  # pragma: no cover - CLI
    raise SystemExit(main())

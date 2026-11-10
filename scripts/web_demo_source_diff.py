"""Generate WEB_DEMO_SOURCE_DIFF.txt for the AgentRec-X Web Demo milestone.

The repository worktree already contained a large number of uncommitted changes from earlier
milestones (recorded before this milestone started in runs/web_demo/baseline/).  This script
does not try to attribute those.  It reports, with byte counts and SHA-256 digests:

  * files ADDED by this milestone (bounded to the declared scopes);
  * the one existing file MODIFIED by this milestone, with a real unified diff against the
    baseline copy saved before any edit;
  * generated demo artifacts under runs/web_demo/;
  * the pre-existing worktree baseline (head/status counts) for honesty.

Usage (WSL or any Python >= 3.10, from the repository root):
    python scripts/web_demo_source_diff.py
"""

from __future__ import annotations

import difflib
import hashlib
import json
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
BASELINE = REPO / "runs/web_demo/baseline"
OUT = REPO / "docs/web-demo/WEB_DEMO_SOURCE_DIFF.txt"

# Directories/files added by this milestone.  Keep this list explicit so unrelated untracked
# files from earlier milestones are never silently attributed to the Web Demo.
ADDED_SCOPES = (
    "web",
    "web_demo",
    "third_party",
    "e2e",
)
ADDED_FILES = (
    "THIRD_PARTY_NOTICES.md",
    "docs/web-demo/WEB_DEMO_REPORT.md",
    "docs/web-demo/WEB_DEMO_GUIDE.md",
    "docs/WEB_DEMO_CONTRACT.md",
    "scripts/run_web_demo.ps1",
    "scripts/stop_web_demo.ps1",
    "scripts/run_web_demo_e2e.ps1",
    "scripts/run_web_demo_bff.sh",
    "scripts/web_demo_source_diff.py",
    "tests/test_web_demo_unit.py",
    "tests/test_web_demo_integration.py",
)
ADDED_GLOBS = (
    "tests/test_web_demo_*.py",
)
EXCLUDE_PARTS = {
    "node_modules", ".next", "__pycache__", ".pytest_cache", ".turbo",
    ".venv", "venv", ".mypy_cache", ".ruff_cache",
}
EXCLUDE_SUFFIXES = {".pyc", ".pyo", ".log", ".tmp"}

GENERATED_SCOPES = (
    "runs/web_demo/screenshots",
    "runs/web_demo/e2e",
    "runs/web_demo/bff_smoke_20261109T103643Z.json",
    "runs/web_demo/mcp_integration_transcript.json",
    "runs/web_demo/traces/.last-run.json",
)

MODIFIED = {
    # current path -> baseline copy saved before any edit
    "recommendation/mcp/server.py": BASELINE / "recommendation_mcp_server.py.baseline",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def iter_files(root: Path):
    if root.is_file():
        yield root
        return
    if not root.exists():
        return
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        if any(part in EXCLUDE_PARTS for part in path.parts):
            continue
        if path.suffix.lower() in EXCLUDE_SUFFIXES:
            continue
        yield path


def collect_added() -> list[Path]:
    found: set[Path] = set()
    for scope in ADDED_SCOPES:
        found.update(iter_files(REPO / scope))
    for name in ADDED_FILES:
        candidate = REPO / name
        if candidate.is_file():
            found.add(candidate)
    for pattern in ADDED_GLOBS:
        found.update(path for path in REPO.glob(pattern) if path.is_file())
    return sorted(found, key=lambda p: p.relative_to(REPO).as_posix())


def collect_generated() -> list[Path]:
    found: set[Path] = set()
    for scope in GENERATED_SCOPES:
        found.update(iter_files(REPO / scope))
    # transcripts written by the BFF smoke tests (their names are timestamped)
    runs = REPO / "runs/web_demo"
    if runs.exists():
        for pattern in ("bff_smoke_*.json", "mcp_integration_transcript.json", "launcher_pids.json"):
            found.update(path for path in runs.glob(pattern) if path.is_file())
    return sorted(found, key=lambda p: p.relative_to(REPO).as_posix())


def git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=REPO, check=False, text=True, capture_output=True
    ).stdout.strip()


def _read_text(path: Path) -> str:
    """Read a text artifact written by PowerShell (UTF-8 with a possible BOM)."""
    if not path.exists():
        return "?"
    return path.read_text(encoding="utf-8-sig").strip()


def basename_sections() -> list[str]:
    lines: list[str] = []
    head = BASELINE / "head.txt"
    branch = BASELINE / "branch.txt"
    status = BASELINE / "git_status_before.txt"
    captured = BASELINE / "captured_at.txt"
    lines.append("BASELINE (recorded before this milestone made any change)")
    lines.append("  captured_at: " + _read_text(captured))
    lines.append("  branch:      " + _read_text(branch))
    lines.append("  HEAD:        " + _read_text(head))
    if status.exists():
        rows = [line for line in _read_text(status).splitlines() if line.strip()]
        lines.append(f"  pre-existing status entries: {len(rows)} (tracked M/D + untracked)")
    lines.append("  This milestone did not revert, stage or commit any pre-existing change.")
    return lines


def main() -> int:
    added = collect_added()
    generated = collect_generated()
    lines: list[str] = []
    lines.append("AGENTREC-X WEB DEMO — SOURCE DIFF")
    lines.append("=" * 80)
    lines.append(
        "Milestone: user-facing Web Shopping Assistant (Next.js 16 frontend + FastAPI BFF over\n"
        "the existing P17 MCP stdio server) with real search/recommend, evidence, comparison and\n"
        "clarification + M9 memory update. No model training, no BGE rebuild, no new Agent runtime,\n"
        "no public deployment, no commit/push."
    )
    lines.append("")
    lines.extend(basename_sections())
    lines.append("")

    lines.append("ADDED FILES / SOURCE (this milestone)")
    for path in added:
        rel = path.relative_to(REPO).as_posix()
        lines.append(f"{rel}")
        lines.append(f"  bytes={path.stat().st_size} sha256={sha256(path)}")
    lines.append("")

    lines.append("MODIFIED EXISTING FILES (this milestone)")
    for rel, baseline in MODIFIED.items():
        current = REPO / rel
        lines.append(f"{rel}")
        if not current.is_file():
            lines.append("  MISSING (was it moved?)")
            continue
        lines.append(f"  bytes={current.stat().st_size} sha256={sha256(current)}")
        if baseline.is_file():
            before = baseline.read_text(encoding="utf-8", errors="replace").splitlines(keepends=True)
            after = current.read_text(encoding="utf-8", errors="replace").splitlines(keepends=True)
            diff = list(
                difflib.unified_diff(
                    before,
                    after,
                    fromfile=rel + " (baseline)",
                    tofile=rel + " (web demo)",
                    n=3,
                )
            )
            lines.append("  unified diff:")
            lines.extend("    " + line.rstrip("\n") for line in diff)
        else:
            lines.append("  baseline copy not found; cannot produce a diff")
    lines.append("")

    lines.append("GENERATED DEMO ARTIFACTS (git-ignored / evidence)")
    if generated:
        for path in generated:
            rel = path.relative_to(REPO).as_posix()
            lines.append(f"{rel}")
            lines.append(f"  bytes={path.stat().st_size} sha256={sha256(path)}")
    else:
        lines.append("  (none found — run the demo/E2E first)")
    lines.append("")

    lines.append("SCOPE NOTES")
    lines.append("- New Python package: web_demo/ (FastAPI BFF, MCP manager, sessions, view models, demo flow).")
    lines.append("- New frontend: web/ (Next.js 16 / React 19 / TypeScript / Tailwind 4).")
    lines.append("- Adapted upstream UI: third_party/ records licenses/reuse guide; THIRD_PARTY_NOTICES.md is authoritative.")
    lines.append("- Only existing file touched: recommendation/mcp/server.py, adding the additive")
    lines.append('  confirmation_source value "web_ui_explicit_user_reply" while keeping old values.')
    lines.append("- The pre-existing untracked files and the pre-existing modified files are NOT part of")
    lines.append("  this milestone and are not attributed here.")
    lines.append("")
    lines.append("END OF SOURCE DIFF")

    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {OUT} ({len(added)} added + {len(generated)} artifacts + {len(MODIFIED)} modified)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

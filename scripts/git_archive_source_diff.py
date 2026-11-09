"""Generate GIT_ARCHIVE_SOURCE_DIFF.txt for the archive branch.

The archive branch contains three local commits on top of the pre-existing
`4dec6d2` worktree snapshot:

  A  P14-P18 implementation/report snapshot
  B  P19 Web Demo + README
  C  archive metadata (this report/source diff, resource manifest, refreshed screenshots)

This script deliberately reads the Git index (staged commit C files) as well as
the committed history, so the generated file can describe the complete archive
content while still being committed in commit C itself.  It excludes itself to
avoid a self-referential digest.

Usage (after commit C files are staged, before committing C):
    python scripts/git_archive_source_diff.py
"""

from __future__ import annotations

import hashlib
import subprocess
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
BASE = "4dec6d2"
OUT = REPO / "GIT_ARCHIVE_SOURCE_DIFF.txt"
BRANCH = "archive/agentrecx-web-demo-v1"


def git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=REPO, text=True, capture_output=True, check=False
    ).stdout


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def current_bytes(rel: str) -> int | None:
    path = REPO / rel
    return path.stat().st_size if path.is_file() else None


def category(rel: str) -> str:
    if rel.startswith(("web/", "web_demo/", "e2e/", "third_party/")):
        return "P19 Web Demo (frontend/BFF/E2E/licences)"
    if rel.startswith("docs/assets/web_demo/"):
        return "P19 Web Demo (screenshots)"
    if rel.startswith("docs/") or rel.startswith("WEB_DEMO_") or rel in {
        "README.md", "THIRD_PARTY_NOTICES.md", "GIT_ARCHIVE_REPORT.md",
        "GIT_ARCHIVE_SOURCE_DIFF.txt", "docs/ARCHIVE_MANIFEST.md",
        "docs/archive_python_freeze.txt",
    }:
        return "Archive metadata / documentation"
    if rel.startswith("tests/"):
        return "Tests and test fixtures"
    if rel.startswith("experiments/") or rel.startswith("benchmarks/"):
        return "P14-P18 experiments / benchmarks"
    if rel.startswith("training/"):
        return "Training code/configs/reports"
    if rel.startswith("archive/"):
        return "Legacy experiment archives"
    if rel.startswith("recommendation/"):
        return "Serving / Agent / control-plane source"
    if rel.startswith("scripts/"):
        return "Operational scripts"
    if rel.startswith("docs/") or rel.endswith((".md", ".txt", ".csv", ".json")):
        return "Reports / documents"
    return "Other source/assets"


def main() -> int:
    status = [line for line in git("diff", "--cached", "--name-status", BASE).splitlines() if line.strip()]
    rows: list[tuple[str, str]] = []
    for line in status:
        parts = line.split("\t")
        if len(parts) < 2:
            continue
        kind, rel = parts[0], parts[-1]
        if rel == OUT.name:
            continue
        rows.append((kind, rel))
    rows.sort(key=lambda item: item[1])

    baseline_status = REPO / "runs/web_demo/baseline/git_status_before.txt"
    baseline_rows = [
        line for line in baseline_status.read_text(encoding="utf-8-sig").splitlines() if line.strip()
    ] if baseline_status.exists() else []
    baseline_head = (REPO / "runs/web_demo/baseline/head.txt").read_text(encoding="utf-8-sig").strip()
    baseline_branch = (REPO / "runs/web_demo/baseline/branch.txt").read_text(encoding="utf-8-sig").strip()

    commits = git("log", "--oneline", f"{BASE}..HEAD").strip().splitlines()
    counts = Counter(kind for kind, _ in rows)
    groups: dict[str, list[tuple[str, str]]] = {}
    for kind, rel in rows:
        groups.setdefault(category(rel), []).append((kind, rel))

    lines: list[str] = []
    lines.append("AGENTREC-X WEB DEMO — GIT ARCHIVE SOURCE DIFF")
    lines.append("=" * 80)
    lines.append(
        "Local archive of the completed AgentRec-X worktree: P14-P18 research/agent/serving\n"
        "implementation and reports + the P19 Web Shopping Assistant. No history rewrite,\n"
        "no push, no release. Large data/models/runs/adapters/secrets are excluded by design."
    )
    lines.append("")
    lines.append(f"BRANCH: {BRANCH}")
    lines.append(f"BASELINE HEAD: {baseline_head} (branch {baseline_branch})")
    lines.append(f"ARCHIVE COMMITS (before this metadata commit):")
    for commit in commits:
        lines.append(f"  {commit}")
    lines.append(
        "COMMIT C (this file, GIT_ARCHIVE_REPORT.md, resource manifest and refreshed screenshots)\n"
        "  is the branch tip that contains this source diff; run `git rev-parse HEAD` after checkout."
    )
    lines.append(f"PRE-EXISTING WORKTREE STATUS AT BASELINE: {len(baseline_rows)} entries")
    lines.append("  The archive did not revert, delete or rewrite those pre-existing changes.")
    lines.append("")
    lines.append("SUMMARY (staged archive content relative to baseline, excluding this file)")
    lines.append(f"  files: {len(rows)}  (A={counts.get('A', 0)}, M={counts.get('M', 0)}, D={counts.get('D', 0)})")
    for group, items in sorted(groups.items()):
        existing = [(k, r) for k, r in items if current_bytes(r) is not None]
        size = sum(current_bytes(r) or 0 for _, r in existing)
        lines.append(f"  {group}: {len(items)} files ({size/1e6:.2f} MB present)")
    lines.append("")

    for group, items in sorted(groups.items()):
        lines.append(f"== {group} ==")
        for kind, rel in items:
            size = current_bytes(rel)
            if kind == "D":
                lines.append(f"{kind}\t{rel}\tdeleted")
            else:
                lines.append(f"{kind}\t{rel}\tbytes={size}\tsha256={sha256(REPO / rel)}")
        lines.append("")

    readme_diff = git("diff", f"{BASE}..HEAD", "--", "README.md")
    if readme_diff.strip():
        lines.append("== README WEB ENTRY DIFF (committed in commit B) ==")
        lines.extend(readme_diff.rstrip().splitlines())
        lines.append("")

    lines.append("SCOPE NOTES")
    lines.append("- The archive includes full source, tests, reports, docs, dependency locks,")
    lines.append("  licence/notice files, the P19 Web Demo and committed screenshots.")
    lines.append("- `data/`, `models/`, `runs/` (except ten already-tracked small files),")
    lines.append("  training adapters, virtualenvs, node_modules, caches and secrets are not committed.")
    lines.append("- External assets and their SHA-256/obtain/rebuild notes are in")
    lines.append("  docs/ARCHIVE_MANIFEST.md; the honest reproducibility answer is in")
    lines.append("  GIT_ARCHIVE_REPORT.md.")
    lines.append("")
    lines.append("END OF GIT ARCHIVE SOURCE DIFF")

    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {OUT} ({len(rows)} entries, {counts})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

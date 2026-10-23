"""Rebuild the project's milestone timeline from the harness session transcripts.

Why this exists
---------------
Git says *what* changed; `docs/` says *why* it was accepted. Neither records the sequence of
human instructions that drove the work (the "milestone briefs"). Those live in the harness
session store (`~/.dsh/sessions/<workspace>/<session>/session.v{3,4}.jsonl.zstd`). This script
extracts them into `docs/SESSION_MILESTONES.md` so the timeline can be regenerated instead of
hand-maintained, and so a new session can see how the project actually progressed.

Usage
-----
    .venv/bin/python scripts/mine_session_milestones.py            # write docs/SESSION_MILESTONES.md
    .venv/bin/python scripts/mine_session_milestones.py --check    # exit 1 if the file is stale

Read-only with respect to the session store: transcripts are streamed, never modified.
"""
from __future__ import annotations

import argparse
import datetime as dt
import glob
import io
import os
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "docs" / "SESSION_MILESTONES.md"
STORE = Path.home() / ".dsh" / "sessions"
WORKSPACES = (
    "--mnt-d-IT-CODE-PROJECTION-agentrec-X--",
    "--mnt-d-IT-CODE-PROJECTION-agentrec--",
    "--home-zxt-AgentRec-X--",
)

#: Prompts that carry a milestone brief, as opposed to follow-up questions or subagent traffic.
MILESTONE_MARKERS = (
    "# AgentRec-X", "# 新任务", "# V2", "E2 / ", "E-04", "Gate B", "Gate C", "P3",
    "R-LR", "AG-LR", "# R-", "# Harness 任务",
)
NOISE = re.compile(r"^(Background subagent|Agent [0-9a-f]{8}|background job bash-|\\resumeProject)",
                   re.IGNORECASE)
INJECTED = re.compile(r"\n(?=Current runtime context|You are a |<system-reminder>)")


def _text(data: dict) -> str:
    payload = data.get("text") or data.get("content")
    if isinstance(payload, str):
        return payload
    if isinstance(payload, list):
        return "\n".join(block.get("text", "") for block in payload if isinstance(block, dict))
    return ""


def _milestone_of(raw: str) -> str | None:
    """The human-authored line of a turn, or None when the turn is harness/subagent traffic."""
    for block in INJECTED.split(raw):
        block = block.strip()
        if len(block) < 40 or NOISE.match(block):
            continue
        head = block.splitlines()[0].strip()
        if any(marker in head for marker in MILESTONE_MARKERS):
            return head[:120]
    return None


def harvest() -> list[dict[str, object]]:
    try:
        import zstandard as zstd
    except ModuleNotFoundError:  # pragma: no cover - dependency guidance
        raise SystemExit("zstandard is required: .venv/bin/pip install zstandard") from None
    found: list[dict[str, object]] = []
    for workspace in WORKSPACES:
        for transcript in sorted(glob.glob(str(STORE / workspace / "*" / "session.v*.jsonl.zstd"))):
            session = os.path.basename(os.path.dirname(transcript)).removeprefix("session-")[:8]
            with open(transcript, "rb") as handle, zstd.ZstdDecompressor().stream_reader(handle) as stream:
                for line in io.TextIOWrapper(stream, encoding="utf-8"):
                    try:
                        event = __import__("json").loads(line)
                    except Exception:
                        continue
                    if event.get("type") != "user/message":
                        continue
                    head = _milestone_of(_text(event.get("data") or {}))
                    if head:
                        found.append({"time": event.get("time") or 0, "session": session, "head": head})
    found.sort(key=lambda row: row["time"])
    unique, seen = [], set()
    for row in found:
        key = re.sub(r"\s+", "", str(row["head"]))[:80]
        if key not in seen:
            seen.add(key)
            unique.append(row)
    return unique


def render(rows: list[dict[str, object]]) -> str:
    lines = [
        "# Session milestones",
        "",
        "Regenerate with `.venv/bin/python scripts/mine_session_milestones.py`.",
        "Each row is the human-authored opening line of one work request, in the order it was made.",
        "",
        "| date | session | milestone |",
        "| --- | --- | --- |",
    ]
    for row in rows:
        when = dt.datetime.fromtimestamp(row["time"] / 1000).strftime("%Y-%m-%d %H:%M")
        head = str(row["head"]).replace("|", "\\|")
        lines.append(f"| {when} | `{row['session']}` | {head} |")
    lines.append("")
    lines.append(f"_{len(rows)} milestones._")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="fail if the committed file is stale")
    args = parser.parse_args(argv)
    rows = harvest()
    rendered = render(rows)
    if args.check:
        current = OUT.read_text(encoding="utf-8") if OUT.exists() else ""
        if current != rendered:
            print("docs/SESSION_MILESTONES.md is stale; re-run without --check", file=sys.stderr)
            return 1
        print(f"up to date ({len(rows)} milestones)")
        return 0
    OUT.write_text(rendered, encoding="utf-8")
    print(f"wrote {OUT.relative_to(REPO)} ({len(rows)} milestones)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

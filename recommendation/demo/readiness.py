"""Demo readiness report: can this checkout serve the demo right now?

This is the report ``./scripts/setup_demo.sh`` prints once the environment is prepared.  It answers
one question for a user who has just cloned the repository: *what is missing before the demo can
start?*

It deliberately owns **no artifact truth of its own**:

* the list of runtime artifacts, their paths, sizes and digests come from
  ``config/demo_runtime_artifacts.json`` through :mod:`scripts.fetch_artifacts`;
* the verification is that module's ``verify_file``, so there is exactly one implementation of
  "size and SHA256 match";
* artifact-recovery credential presence is that module's ``load_environment``, which already reads
  ``.env`` / ``.env.local`` with the documented precedence;
* LLM provider state is :func:`recommendation.demo.llm_policy.provider_availability`, the same probe
  ``GET /v1/demo/decision-modes`` uses, so the report cannot disagree with the browser.

What is *required* versus *optional*:

```text
required   the environment (python + the server entry point's imports)
required   every runtime artifact the product serves from
optional   an LLM provider configuration      (the deterministic product needs none)
optional   an artifact-recovery credential    (only needed to download artifacts)
```

A missing optional item never blocks: it is reported as ``OPTIONAL`` and the verdict stays READY.
No secret is ever printed: only presence is reported, never a value.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

#: Check levels.  MISSING and FAIL are the two ways a required item is absent; the distinction
#: matters because the remedies differ (download it vs. replace a corrupted copy).
PASS = "PASS"
MISSING = "MISSING"
FAIL = "FAIL"
OPTIONAL = "OPTIONAL"

#: Files a runnable checkout must contain.  These are project files under version control, not
#: artifacts, so they are named here rather than read from the artifact manifest.
REQUIRED_PROJECT_FILES: tuple[str, ...] = (
    "requirements.txt",
    "requirements-cpu.txt",
    "config/demo_runtime_artifacts.json",
    "scripts/fetch_artifacts.py",
    "scripts/start_demo.sh",
    "recommendation/api/app.py",
    "runs/sasrec_canonical_2026/run.json",
)

#: The module the server is started with; importing it proves the whole runtime closure is present.
ENTRY_POINT_MODULE = "recommendation.api.app"


@dataclass(frozen=True)
class Check:
    """One readiness line."""

    level: str
    name: str
    detail: str


def _fetch_artifacts() -> Any:
    """Import the shared artifact tooling.

    ``scripts/`` is a namespace package, so the module is imported by name rather than loaded by
    path: a second, divergent copy of the verification rules is exactly what this report must not
    introduce.
    """
    from scripts import fetch_artifacts

    return fetch_artifacts


# --------------------------------------------------------------------------- #
# required: environment
# --------------------------------------------------------------------------- #


def environment_checks(repo_root: Path, *, check_imports: bool = True) -> list[Check]:
    """Repository shape, interpreter, and the server entry point's import closure."""
    checks: list[Check] = []

    if (repo_root / "config" / "demo_runtime_artifacts.json").is_file():
        checks.append(Check(PASS, "repository", str(repo_root)))
    else:
        checks.append(
            Check(
                FAIL,
                "repository",
                f"{repo_root} does not look like the repository root "
                "(config/demo_runtime_artifacts.json is absent)",
            )
        )

    missing = [name for name in REQUIRED_PROJECT_FILES if not (repo_root / name).is_file()]
    if missing:
        checks.append(Check(MISSING, "project files", ", ".join(missing)))
    else:
        checks.append(Check(PASS, "project files", f"{len(REQUIRED_PROJECT_FILES)} present"))

    version = "%d.%d.%d" % sys.version_info[:3]
    checks.append(
        Check(PASS if sys.version_info[:2] == (3, 10) else FAIL, "python", version)
    )

    if check_imports:
        try:
            __import__(ENTRY_POINT_MODULE)
        except Exception as exc:  # noqa: BLE001 - the report must survive any import failure
            checks.append(
                Check(FAIL, "core imports", f"{ENTRY_POINT_MODULE} failed to import: {type(exc).__name__}: {exc}")
            )
        else:
            checks.append(Check(PASS, "core imports", f"{ENTRY_POINT_MODULE} imports cleanly"))

    return checks


# --------------------------------------------------------------------------- #
# required: runtime artifacts
# --------------------------------------------------------------------------- #


def artifact_checks(repo_root: Path, manifest_path: Path | None = None) -> list[Check]:
    """Verify every runtime artifact through the shared verifier.

    A file that is absent is ``MISSING`` (download it); a file that is present but does not match
    the recorded size or digest is ``FAIL`` (replace it).  Both block the verdict.
    """
    artifacts = _fetch_artifacts()
    manifest_file = manifest_path or (repo_root / "config" / "demo_runtime_artifacts.json")
    try:
        manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
        entries = artifacts.load_recovery_entries(manifest)
    except Exception as exc:  # noqa: BLE001 - a broken manifest is a readiness failure, not a crash
        return [Check(FAIL, "artifact manifest", f"{type(exc).__name__}: {exc}")]

    checks: list[Check] = []
    for entry in entries:
        target = entry.absolute(repo_root)
        if not target.is_file():
            checks.append(Check(MISSING, entry.role, str(entry.path)))
            continue
        try:
            artifacts.verify_file(target, entry)
        except artifacts.VerificationError as exc:
            checks.append(Check(FAIL, entry.role, str(exc)))
        else:
            checks.append(Check(PASS, entry.role, str(entry.path)))
    return checks


# --------------------------------------------------------------------------- #
# optional: provider and recovery credentials
# --------------------------------------------------------------------------- #


def llm_check() -> Check:
    """Provider configuration state, read through the probe the HTTP endpoint uses.

    Never prints a credential: the probe reports the provider profile and model name only.
    """
    from recommendation.demo.llm_policy import provider_availability

    state = provider_availability()
    if state.get("available"):
        return Check(
            OPTIONAL,
            "LLM provider",
            f"CONFIGURED ({state.get('provider')} / {state.get('model')}) - the LLM Agent decision "
            "mode is available",
        )
    reason = state.get("reason")
    detail = {
        "not_configured": "NOT CONFIGURED (set AGENTRECX_LLM_BASE_URL and AGENTRECX_LLM_MODEL in .env)",
        "missing_api_key": "NOT CONFIGURED (a provider is set but AGENTRECX_LLM_API_KEY is empty)",
        "control_plane_unsupported": "NOT AVAILABLE on this control plane",
    }.get(reason, f"NOT CONFIGURED ({reason})")
    return Check(OPTIONAL, "LLM provider", f"{detail}; the deterministic demo needs none")


def hf_recovery_check(repo_root: Path) -> Check:
    """Whether an artifact-recovery credential is configured.  Presence only, never a value."""
    artifacts = _fetch_artifacts()
    try:
        environment = artifacts.load_environment(repo_root)
    except Exception as exc:  # noqa: BLE001
        return Check(OPTIONAL, "HF recovery", f"NOT CONFIGURED (could not read .env: {type(exc).__name__})")
    repo_id = bool(environment.get("HF_REPO_ID"))
    token = bool(environment.get("HF_TOKEN"))
    if repo_id:
        return Check(
            OPTIONAL,
            "HF recovery",
            "CONFIGURED (a repository and token are set; only needed to download artifacts)"
            if token
            else "CONFIGURED (repository set; a token is needed only if the repository is private)",
        )
    return Check(
        OPTIONAL,
        "HF recovery",
        "NOT CONFIGURED (only needed to download artifacts; a complete local set needs no credential)",
    )


# --------------------------------------------------------------------------- #
# report
# --------------------------------------------------------------------------- #


def blocked_by(checks: Sequence[Check]) -> list[Check]:
    """The required checks that are not satisfied."""
    return [check for check in checks if check.level in (MISSING, FAIL)]


def readiness(repo_root: Path, *, check_imports: bool = True) -> tuple[list[Check], list[Check]]:
    """Return ``(all_checks, blockers)``."""
    checks = [
        *environment_checks(repo_root, check_imports=check_imports),
        *artifact_checks(repo_root),
        llm_check(),
        hf_recovery_check(repo_root),
    ]
    return checks, blocked_by(checks)


def render(checks: Sequence[Check], repo_root: Path) -> str:
    """A short report: what was checked, then the verdict and the next step."""
    lines: list[str] = []
    lines.append("=" * 70)
    lines.append(" AgentRec-X demo readiness")
    lines.append("=" * 70)
    lines.append(f" repository : {repo_root}")
    lines.append("")
    for check in checks:
        lines.append(f"  {check.level:<8} {check.name:<28} {check.detail}")

    blockers = blocked_by(checks)
    lines.append("=" * 70)
    if blockers:
        lines.append("DEMO READINESS: BLOCKED")
        lines.append("")
        lines.append("Required items are not satisfied:")
        for check in blockers:
            lines.append(f"  {check.level}: {check.name} - {check.detail}")
        lines.append("")
        lines.append("Next step (artifact recovery uses the committed manifest and digests):")
        lines.append("  python3 scripts/fetch_artifacts.py verify            # offline: what is missing")
        lines.append("  python3 scripts/fetch_artifacts.py fetch --all       # download and verify them")
        lines.append("  ./scripts/setup_demo.sh                              # re-run this check")
        lines.append("")
        lines.append("Then start the demo:  ./scripts/start_demo.sh")
    else:
        lines.append("DEMO READINESS: READY")
        lines.append("")
        lines.append("Start the demo with:  ./scripts/start_demo.sh")
    lines.append("=" * 70)
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    """Print the readiness report.  Exit 0 when READY, 1 when BLOCKED."""
    parser = argparse.ArgumentParser(description="AgentRec-X demo readiness report")
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--skip-imports",
        action="store_true",
        help="skip importing the server entry point (faster; used by the tests)",
    )
    args = parser.parse_args(argv)

    repo_root = args.repo_root.resolve()
    checks, blockers = readiness(repo_root, check_imports=not args.skip_imports)
    print(render(checks, repo_root))
    return 1 if blockers else 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

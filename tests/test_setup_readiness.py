"""Readiness-report tests: the acceptance cases for ``./scripts/setup_demo.sh``.

The report is exercised against a synthetic repository root whose manifest is the **real** one with
each entry's path/size/digest rewritten to a tiny local file.  That keeps the artifact verification
genuine (it is the shared verifier) while letting a case such as "one artifact is missing" run in
milliseconds instead of hashing a gigabyte.

Cases, in the order the task states them:

``A`` all artifacts present -> READY
``B`` one required artifact missing -> BLOCKED, named, with the recovery command
``C`` no LLM configuration -> OPTIONAL, still READY
``D`` LLM configured -> CONFIGURED, credential never printed
``E`` no recovery credential but artifacts complete -> OPTIONAL, still READY
``F`` no secret appears in stdout or stderr
"""

from __future__ import annotations

import copy
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from recommendation.demo import readiness as R  # noqa: E402

REAL_MANIFEST = json.loads(
    (REPO_ROOT / "config" / "demo_runtime_artifacts.json").read_text(encoding="utf-8")
)
ROLES = [entry["role"] for entry in REAL_MANIFEST["recoverability"]["artifacts"]]


def build_repo(tmp_path: Path, *, omit: str | None = None, tamper: str | None = None) -> Path:
    """A synthetic repository root: required project files stubbed, artifacts tiny but real."""
    root = tmp_path / "repo"
    for name in R.REQUIRED_PROJECT_FILES:
        target = root / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("{}\n", encoding="utf-8")

    manifest = copy.deepcopy(REAL_MANIFEST)
    for entry in manifest["recoverability"]["artifacts"]:
        role = entry["role"]
        # The manifest records the *expected* content; a tampered role gets different bytes on
        # disk, so the recorded size and digest no longer match what is there.
        expected = b"x" * (512 + len(role))
        if role != omit:
            target = root / entry["path"]
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(b"tampered" if role == tamper else expected)
        entry["size_bytes"] = len(expected)
        entry["sha256"] = hashlib.sha256(expected).hexdigest()

    target = root / "config" / "demo_runtime_artifacts.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return root


def levels(checks: list[R.Check]) -> dict[str, str]:
    return {check.name: check.level for check in checks}


def clear_optional_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (
        "AGENTRECX_LLM_BASE_URL",
        "AGENTRECX_LLM_MODEL",
        "AGENTRECX_LLM_PROFILE",
        "AGENTRECX_LLM_API_KEY",
        "HF_REPO_ID",
        "HF_REVISION",
        "HF_TOKEN",
    ):
        monkeypatch.delenv(name, raising=False)


# --------------------------------------------------------------------------- #
# A
# --------------------------------------------------------------------------- #


def test_a_complete_environment_is_ready(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    clear_optional_env(monkeypatch)
    root = build_repo(tmp_path)

    checks, blockers = R.readiness(root, check_imports=False)

    assert blockers == []
    assert set(levels(checks)) >= set(ROLES), "every runtime artifact must be reported"
    assert all(levels(checks)[role] == R.PASS for role in ROLES)
    assert "DEMO READINESS: READY" in R.render(checks, root)
    assert R.main(["--repo-root", str(root), "--skip-imports"]) == 0


# --------------------------------------------------------------------------- #
# B
# --------------------------------------------------------------------------- #


def test_b_missing_artifact_blocks_with_guidance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    clear_optional_env(monkeypatch)
    root = build_repo(tmp_path, omit="catalogue")

    checks, blockers = R.readiness(root, check_imports=False)
    report = R.render(checks, root)

    assert levels(checks)["catalogue"] == R.MISSING
    assert [check.name for check in blockers] == ["catalogue"]
    assert "DEMO READINESS: BLOCKED" in report
    assert "catalogue" in report
    # The recovery command must be the real CLI, not an invented one.
    assert "scripts/fetch_artifacts.py verify" in report
    assert "scripts/fetch_artifacts.py fetch --all" in report

    assert R.main(["--repo-root", str(root), "--skip-imports"]) == 1
    assert "DEMO READINESS: BLOCKED" in capsys.readouterr().out


def test_b_tampered_artifact_fails_rather_than_reads_as_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A wrong copy and an absent copy need different remedies, so they differ in the report."""
    clear_optional_env(monkeypatch)
    root = build_repo(tmp_path, tamper="mappings")

    checks, blockers = R.readiness(root, check_imports=False)

    assert levels(checks)["mappings"] == R.FAIL
    assert [check.name for check in blockers] == ["mappings"]
    assert "size mismatch" in dict((c.name, c.detail) for c in checks)["mappings"]


# --------------------------------------------------------------------------- #
# C / D
# --------------------------------------------------------------------------- #


def test_c_unconfigured_llm_is_optional_and_does_not_block(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    clear_optional_env(monkeypatch)
    root = build_repo(tmp_path)

    checks, blockers = R.readiness(root, check_imports=False)
    report = R.render(checks, root)

    assert levels(checks)["LLM provider"] == R.OPTIONAL
    assert "NOT CONFIGURED" in report
    assert blockers == []
    assert "DEMO READINESS: READY" in report


def test_d_configured_llm_is_reported_without_the_credential(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    clear_optional_env(monkeypatch)
    secret = "sk-thisMustNeverBePrinted"
    monkeypatch.setenv("AGENTRECX_LLM_BASE_URL", "https://api.deepseek.com/v1")
    monkeypatch.setenv("AGENTRECX_LLM_MODEL", "deepseek-flash")
    monkeypatch.setenv("AGENTRECX_LLM_PROFILE", "deepseek")
    monkeypatch.setenv("AGENTRECX_LLM_API_KEY", secret)
    root = build_repo(tmp_path)

    checks, blockers = R.readiness(root, check_imports=False)
    report = R.render(checks, root)

    assert "CONFIGURED" in dict((c.name, c.detail) for c in checks)["LLM provider"]
    assert "deepseek-flash" in report
    assert secret not in report
    assert blockers == []


# --------------------------------------------------------------------------- #
# E / F
# --------------------------------------------------------------------------- #


def test_e_absent_recovery_credential_is_optional_and_does_not_block(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    clear_optional_env(monkeypatch)
    root = build_repo(tmp_path)

    checks, blockers = R.readiness(root, check_imports=False)

    assert levels(checks)["HF recovery"] == R.OPTIONAL
    assert "NOT CONFIGURED" in dict((c.name, c.detail) for c in checks)["HF recovery"]
    assert blockers == []


def test_e_empty_dotenv_placeholders_do_not_hide_a_real_dotenv_local(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    """The onboarding case: `.env` has copied empty placeholders, `.env.local` has the real values.

    The report must agree with what ``scripts/fetch_artifacts.py fetch`` actually resolves, so the
    readiness line is compared against that module's own resolution rather than against a guess.
    """
    clear_optional_env(monkeypatch)
    token = "hf_thisMustNeverBePrinted"
    root = build_repo(tmp_path)
    (root / ".env").write_text("HF_REPO_ID=\nHF_REVISION=\nHF_TOKEN=\n", encoding="utf-8")
    (root / ".env.local").write_text(
        f"HF_REPO_ID=owner/name\nHF_REVISION=main\nHF_TOKEN={token}\n", encoding="utf-8"
    )

    checks, blockers = R.readiness(root, check_imports=False)
    detail = dict((c.name, c.detail) for c in checks)["HF recovery"]

    # The invariant the request asks for: readiness == what the artifact tool resolves.
    resolved = R._fetch_artifacts().load_environment(root)
    assert resolved["HF_REPO_ID"] == "owner/name"
    assert "CONFIGURED" in detail
    assert "NOT CONFIGURED" not in detail
    assert blockers == []

    status = R.main(["--repo-root", str(root), "--skip-imports"])
    captured = capsys.readouterr()
    assert status == 0
    assert token not in captured.out
    assert token not in captured.err


def test_f_no_secret_reaches_stdout_or_stderr(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    clear_optional_env(monkeypatch)
    api_key = "sk-thisMustNeverBePrinted"
    hf_token = "hf_thisMustNeverBePrinted"
    root = build_repo(tmp_path)
    # Both files the loader reads, so the report has every chance to leak them.
    (root / ".env").write_text(
        f"AGENTRECX_LLM_BASE_URL=https://api.deepseek.com/v1\n"
        f"AGENTRECX_LLM_MODEL=deepseek-flash\n"
        f"AGENTRECX_LLM_API_KEY={api_key}\n",
        encoding="utf-8",
    )
    (root / ".env.local").write_text(f"HF_REPO_ID=owner/name\nHF_TOKEN={hf_token}\n", encoding="utf-8")

    status = R.main(["--repo-root", str(root), "--skip-imports"])
    captured = capsys.readouterr()

    assert status == 0
    for stream in (captured.out, captured.err):
        assert api_key not in stream
        assert hf_token not in stream
        assert "Authorization" not in stream
        assert "Bearer" not in stream
    # The optional capabilities are still reported truthfully.
    assert "CONFIGURED" in captured.out


def test_f_cli_never_prints_a_secret_from_the_environment(tmp_path: Path) -> None:
    """The real entry point, run as a subprocess, with credentials in its environment."""
    api_key = "sk-subprocessMustNotLeak"
    hf_token = "hf_subprocessMustNotLeak"
    root = build_repo(tmp_path)
    environment = {
        "PATH": "/usr/bin:/bin",
        "HOME": str(Path.home()),
        "PYTHONPATH": str(REPO_ROOT),
        "AGENTRECX_LLM_BASE_URL": "https://api.deepseek.com/v1",
        "AGENTRECX_LLM_MODEL": "deepseek-flash",
        "AGENTRECX_LLM_API_KEY": api_key,
        "HF_REPO_ID": "owner/name",
        "HF_TOKEN": hf_token,
    }

    result = subprocess.run(
        [sys.executable, "-m", "recommendation.demo.readiness", "--repo-root", str(root), "--skip-imports"],
        capture_output=True,
        text=True,
        env=environment,
        cwd=str(REPO_ROOT),
    )

    assert result.returncode == 0, result.stderr
    assert "DEMO READINESS: READY" in result.stdout
    for stream in (result.stdout, result.stderr):
        assert api_key not in stream
        assert hf_token not in stream


def test_setup_script_delegates_artifact_truth_to_the_shared_verifier() -> None:
    """``setup_demo.sh`` must not grow a second artifact list, digest table or verifier."""
    script = (REPO_ROOT / "scripts" / "setup_demo.sh").read_text(encoding="utf-8")

    assert "recommendation.demo.readiness" in script
    # No digest arithmetic and no artifact path/role list of its own.
    assert "sha256sum" not in script
    assert "hashlib" not in script
    for role in ROLES:
        assert role not in script, f"{role} must come from the manifest, not from the script"
    # The readiness report is the last step, and its verdict is the script's exit status.
    assert "READINESS_STATUS" in script
    assert 'exit "${READINESS_STATUS}"' in script

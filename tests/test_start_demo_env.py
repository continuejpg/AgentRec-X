"""Behavioural tests for the ``.env`` autoload in ``scripts/start_demo.sh``.

The launcher parses ``<repo>/.env`` into the process environment with the precedence

    existing process environment  >  .env  >  application defaults

These tests exercise that loader directly by sourcing the launcher (it returns early when
sourced, so no server starts) and by asserting on booleans and counts only.  No harness here
prints a variable's value, so a credential copied out of the environment cannot leak into the
test output, and one test asserts exactly that.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
LAUNCHER = REPO_ROOT / "scripts" / "start_demo.sh"

pytestmark = pytest.mark.skipif(
    shutil.which("bash") is None, reason="the launcher is a bash script"
)

#: Reports counts and SET/UNSET, plus two value *comparisons*; never a value itself.
HARNESS = r"""
set -Eeuo pipefail
source "$1"
load_dotenv_into_environment "$2"
printf 'loaded=%s\n' "${DOTENV_LOADED}"
printf 'skipped=%s\n' "${DOTENV_SKIPPED}"
for name in AGENTRECX_LLM_BASE_URL AGENTRECX_LLM_MODEL AGENTRECX_LLM_PROFILE \
            AGENTRECX_LLM_API_KEY HF_TOKEN; do
    if printenv "${name}" >/dev/null 2>&1; then
        printf '%s=SET\n' "${name}"
    else
        printf '%s=UNSET\n' "${name}"
    fi
done
printf 'base_url_is_file=%s\n' \
    "$([ "$(printenv AGENTRECX_LLM_BASE_URL || true)" = 'https://file.invalid/v1' ] && echo yes || echo no)"
printf 'model_is_shell=%s\n' \
    "$([ "$(printenv AGENTRECX_LLM_MODEL || true)" = 'shell-model' ] && echo yes || echo no)"
printf 'model_is_file=%s\n' \
    "$([ "$(printenv AGENTRECX_LLM_MODEL || true)" = 'file-model' ] && echo yes || echo no)"
printf 'model_is_quoted=%s\n' \
    "$([ "$(printenv AGENTRECX_LLM_MODEL || true)" = 'quoted model' ] && echo yes || echo no)"
"""


def run_loader(env_file: Path, shell_env: dict[str, str] | None = None) -> subprocess.CompletedProcess:
    """Source the launcher, run the loader, and report only non-secret facts."""
    env = {
        "PATH": "/usr/bin:/bin",
        "HOME": str(Path.home()),
        "AGENTRECX_LLM_BASE_URL": "",
        "AGENTRECX_LLM_MODEL": "",
        "AGENTRECX_LLM_PROFILE": "",
        "AGENTRECX_LLM_API_KEY": "",
        **(shell_env or {}),
    }
    for name in [n for n, v in env.items() if v == ""]:
        # An empty string would count as "already set" and mask what we are testing;
        # the caller re-adds the ones it wants through ``shell_env``.
        if name not in (shell_env or {}):
            env.pop(name)
    return subprocess.run(
        ["bash", "-c", HARNESS, "_", str(LAUNCHER), str(env_file)],
        capture_output=True,
        text=True,
        env=env,
    )


def facts(result: subprocess.CompletedProcess) -> dict[str, str]:
    assert result.returncode == 0, result.stderr
    parsed: dict[str, str] = {}
    for line in result.stdout.splitlines():
        if "=" in line:
            key, _, value = line.partition("=")
            parsed[key] = value
    return parsed


COMPLETE_ENV = """\
# provider configuration
AGENTRECX_LLM_BASE_URL=https://file.invalid/v1
AGENTRECX_LLM_MODEL="quoted model"
AGENTRECX_LLM_PROFILE=deepseek
AGENTRECX_LLM_API_KEY=sk-not-a-real-credential
"""


def test_absent_env_file_is_not_an_error(tmp_path: Path) -> None:
    """A. No `.env` at all: the launcher proceeds with the environment it was given."""
    result = run_loader(tmp_path / ".env")
    values = facts(result)
    assert values["loaded"] == "0"
    assert values["AGENTRECX_LLM_MODEL"] == "UNSET"
    assert values["AGENTRECX_LLM_API_KEY"] == "UNSET"


def test_env_file_is_loaded(tmp_path: Path) -> None:
    """B. A complete `.env` reaches the environment the server will inherit."""
    env_file = tmp_path / ".env"
    env_file.write_text(COMPLETE_ENV, encoding="utf-8")

    values = facts(run_loader(env_file))

    assert values["loaded"] == "4"
    assert values["AGENTRECX_LLM_BASE_URL"] == "SET"
    assert values["AGENTRECX_LLM_MODEL"] == "SET"
    assert values["AGENTRECX_LLM_PROFILE"] == "SET"
    assert values["AGENTRECX_LLM_API_KEY"] == "SET"
    # Quotes are stripped, so the value the provider client sees is the intended one.
    assert values["base_url_is_file"] == "yes"
    assert values["model_is_quoted"] == "yes"


def test_existing_environment_wins_over_env_file(tmp_path: Path) -> None:
    """C. An explicit export is never overwritten by the file."""
    env_file = tmp_path / ".env"
    env_file.write_text(COMPLETE_ENV, encoding="utf-8")

    values = facts(run_loader(env_file, {"AGENTRECX_LLM_MODEL": "shell-model"}))

    assert values["model_is_shell"] == "yes"
    assert values["model_is_file"] == "no"
    assert values["skipped"] == "1"
    assert values["loaded"] == "3"
    # The other three entries still load; precedence is per variable, not all-or-nothing.
    assert values["AGENTRECX_LLM_BASE_URL"] == "SET"


def test_partial_configuration_still_loads_and_is_not_an_error(tmp_path: Path) -> None:
    """D. An incomplete provider configuration must not stop the deterministic product."""
    env_file = tmp_path / ".env"
    env_file.write_text(
        "AGENTRECX_LLM_BASE_URL=https://file.invalid/v1\n", encoding="utf-8"
    )

    values = facts(run_loader(env_file))

    assert values["loaded"] == "1"
    assert values["base_url_is_file"] == "yes"
    assert values["AGENTRECX_LLM_MODEL"] == "UNSET"
    assert values["AGENTRECX_LLM_API_KEY"] == "UNSET"


def test_dotenv_local_is_never_read(tmp_path: Path) -> None:
    """E. The artifact-recovery credential is neither loaded nor echoed."""
    secret = "hf_thisMustNeverBeLoaded"
    (tmp_path / ".env.local").write_text(f"HF_TOKEN={secret}\n", encoding="utf-8")

    result = run_loader(tmp_path / ".env")

    values = facts(result)
    assert values["HF_TOKEN"] == "UNSET"
    assert values["loaded"] == "0"
    assert secret not in result.stdout
    assert secret not in result.stderr


def test_loader_never_echoes_a_value(tmp_path: Path) -> None:
    """The loader itself prints nothing, so a credential cannot reach a log."""
    secret = "sk-thisMustNeverBePrinted"
    env_file = tmp_path / ".env"
    env_file.write_text(
        f"AGENTRECX_LLM_API_KEY={secret}\nAGENTRECX_LLM_MODEL=file-model\n", encoding="utf-8"
    )

    result = run_loader(env_file)

    assert facts(result)["AGENTRECX_LLM_API_KEY"] == "SET"
    assert secret not in result.stdout
    assert secret not in result.stderr


def test_env_file_is_parsed_not_sourced(tmp_path: Path) -> None:
    """A shell construct in `.env` is data, not code: it must not execute."""
    marker = tmp_path / "executed"
    env_file = tmp_path / ".env"
    env_file.write_text(
        f"AGENTRECX_LLM_MODEL=$(touch {marker})\n", encoding="utf-8"
    )

    values = facts(run_loader(env_file))

    assert not marker.exists(), "the .env file must be parsed, never sourced"
    assert values["AGENTRECX_LLM_MODEL"] == "SET"


def test_comments_blanks_and_malformed_lines_are_ignored(tmp_path: Path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text(
        "# a comment\n"
        "\n"
        "   \n"
        "not-an-assignment\n"
        "=novalue\n"
        "1INVALID=x\n"
        "  AGENTRECX_LLM_PROFILE=deepseek  \n"
        "export AGENTRECX_LLM_MODEL=file-model\n",
        encoding="utf-8",
    )

    values = facts(run_loader(env_file))

    assert values["loaded"] == "2"
    assert values["AGENTRECX_LLM_PROFILE"] == "SET"
    assert values["model_is_file"] == "yes"


def test_launcher_help_still_runs_when_executed() -> None:
    """Sourcing returns early; executing the script still behaves as before."""
    syntax = subprocess.run(["bash", "-n", str(LAUNCHER)], capture_output=True, text=True)
    assert syntax.returncode == 0, syntax.stderr

    result = subprocess.run(
        ["bash", str(LAUNCHER), "--help"], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
    assert "start_demo.sh [options]" in result.stdout

"""Packaging-phase tests: the one-command path, the demo artifacts, and leak guards.

What these protect
------------------
The packaging phase claims three things that are easy to break silently and cheap to check:

1. **the offline path actually is offline** - no provider credential is read, no API key is
   committed, and the default container configuration neuters `AGENTRECX_LLM_*`;
2. **the one-command path works on a fresh clone** - the demo catalogue and the checkpoint
   that `scripts/run_demo.sh` and `docker compose up` depend on can be generated from files in
   this repository, with no download and no accepted artifact present;
3. **the demo is not a benchmark claim** - the generated checkpoint is untrained, the demo
   catalogue says what it is, and no documentation presents the demo's ranking as measured
   quality.

Nothing here needs a network, a credential or a container: the Docker image itself is built
and start-smoked by `.github/workflows/verify.yml`, which is the only place a daemon is
available.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


# --------------------------------------------------------------------------- #
# Demo catalogue and generated artifacts
# --------------------------------------------------------------------------- #


def test_demo_catalog_is_committed_and_describes_itself():
    """The catalogue is reviewable in the repository and cannot be mistaken for Amazon data."""
    from experiments.build_demo_catalog import DEMO_CATALOG_PATH, catalog_records

    assert DEMO_CATALOG_PATH.is_file(), "the demo catalogue must be committed"
    first_line = DEMO_CATALOG_PATH.read_text(encoding="utf-8").splitlines()[0]
    envelope = json.loads(first_line)
    assert envelope["format"] == "agentrecx.catalog_products.v1"
    assert "NOT Amazon" in envelope["category"] or "NOT Amazon" in envelope.get("description", "")
    # Every record is a real catalogue record: the accepted loader reads them.
    from recommendation.catalog import MetadataIndex

    index = MetadataIndex.load(DEMO_CATALOG_PATH)
    assert len(index.records) == len(catalog_records())
    for record in index.records.values():
        assert record.source, "provenance is required on every record"
        assert record.title


def test_demo_catalog_builds_a_loadable_engine_without_accepted_artifacts(tmp_path: Path):
    """The one-command path must work on a fresh clone: no checkpoint, no dataset, no network."""
    from experiments.build_demo_catalog import build_artifacts, catalog_records
    from recommendation.inference import InferenceConfig, SASRecInferenceEngine

    built = build_artifacts(out_dir=tmp_path, catalog_path=tmp_path / "catalog.jsonl")
    assert built["trained"] is False, "the demo checkpoint must declare itself untrained"

    engine = SASRecInferenceEngine(
        InferenceConfig(
            checkpoint_path=Path(built["checkpoint"]),
            mappings_path=Path(built["mappings"]),
            device="cpu",
        ),
        verify_checkpoint_sha256=False,
    )
    assert engine.is_ready()
    assert engine.num_items == len(catalog_records())
    history = [record["parent_asin"] for record in catalog_records()][:3]
    result = engine.recommend(history, k=3)
    assert len(result.recommendations) == 3
    # The engine is real; the *scores* are meaningless, which the built output states.
    for item in result.recommendations:
        assert item.parent_asin not in history
        assert isinstance(item.score, float)


def test_demo_catalog_ids_are_offline_and_opaque():
    """Identities must not look like real Amazon ids, or a reader could mistake the fixture."""
    from experiments.build_demo_catalog import catalog_records

    asins = [record["parent_asin"] for record in catalog_records()]
    assert len(asins) == len(set(asins)), "identities must be unique"
    for asin in asins:
        assert not re.match(r"^B0[0-9A-Z]{8}$", asin), f"{asin} looks like a real ASIN"


# --------------------------------------------------------------------------- #
# The canonical scenario
# --------------------------------------------------------------------------- #


def test_canonical_scenario_runs_and_filters(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """The scenario reaches a successful terminal state, with a real, visible exclusion.

    Runs the shipped scenario (not a copy of it) so the README's example and this test cannot
    drift.  A synthetic engine and catalogue keep it offline and instant.
    """
    from fastapi.testclient import TestClient

    from experiments.demo_scenario import SCENARIO, ensure_demo_artifacts, format_trajectory
    from recommendation.api.app import create_app

    built = ensure_demo_artifacts(quiet=True)
    assert Path(built["catalog"]).is_file()
    for name, value in _demo_env().items():
        monkeypatch.setenv(name, value)

    with TestClient(create_app(enable_demo=True)) as client:
        response = client.post("/v1/demo/agent/recommend", json=SCENARIO)
        assert response.status_code == 200, response.text
        body = response.json()

    assert body["terminal"]["succeeded"] is True, body["terminal"]
    assert body["route"] == "recommend"
    assert body["control_plane"] == "loop"
    assert body["grounded"]["grounded_count"] >= 1
    assert body["trajectory"]["steps"], "the trajectory must be inspectable"

    eligibility = body["eligibility"]
    assert eligibility["evaluated"] is True
    assert eligibility["requirements"], "the scenario states a constraint"
    excluded = (eligibility.get("projection") or {}).get("excluded") or []
    assert excluded, "the scenario is chosen so a proved violation is excluded"

    presented = {item["parent_asin"] for item in body["recommendations"]}
    assert body["recommendations"], "the scenario must present something"
    assert not (presented & {item["parent_asin"] for item in excluded}), (
        "an ineligible candidate must never be presented as a recommendation"
    )
    # Every presented identity is attributed to a source.
    for item in body["recommendations"]:
        assert item["sources"], "a presented candidate must carry provenance"
        assert item["score_kind"], "a score must say what it measures"

    # The summary the documentation shows is renderable from the same response.
    rendered = format_trajectory(body)
    assert "recommendations:" in rendered
    assert "trajectory:" in rendered


def _demo_env() -> dict[str, str]:
    from experiments.demo_scenario import DEMO_ENV

    return dict(DEMO_ENV)


def test_scenario_summary_marks_absent_constraints_honestly():
    """When no hard constraint is active the summary must not imply everything passed."""
    from experiments.demo_scenario import format_trajectory

    body = {
        "route": "direct",
        "control_plane": "loop",
        "terminal": {
            "status": "finished",
            "termination_reason": "completed",
            "succeeded": True,
        },
        "grounded": {
            "grounded_count": 0,
            "sources_present": [],
            "multi_source_count": 0,
            "ungrounded_count": 0,
        },
        "eligibility": {
            "evaluated": False,
            "requirements": [],
            "verified_eligible_count": 0,
            "ineligible_count": 0,
            "unresolved_count": 0,
            "projection": None,
        },
        "recommendations": [],
        "trajectory": {"steps": []},
        "timing": {"total_ms": 1.0, "agent_ms": 1.0, "serialize_ms": 0.0},
    }
    rendered = format_trajectory(body)
    assert "nothing was evaluated" in rendered
    assert "no verified candidate" in rendered


# --------------------------------------------------------------------------- #
# One-command entry points
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "relative",
    [
        "Dockerfile",
        "docker-compose.yml",
        ".dockerignore",
        ".env.example",
        "scripts/run_demo.sh",
        "scripts/entrypoint.sh",
        ".github/workflows/verify.yml",
        "DOCKER.md",
    ],
)
def test_packaging_artifacts_exist(relative: str):
    path = REPO_ROOT / relative
    assert path.is_file(), f"missing packaging file {relative}"
    assert path.stat().st_size > 200, f"{relative} looks like a stub"


def test_documented_scripts_are_syntactically_valid():
    """A one-command path that cannot parse is worse than none: check both shell entry points."""
    for relative in ("scripts/run_demo.sh", "scripts/entrypoint.sh"):
        result = subprocess.run(
            ["bash", "-n", str(REPO_ROOT / relative)],
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0, f"{relative}: {result.stderr}"


def test_compose_defaults_are_offline_and_opt_in_for_the_provider():
    """The offline default must be explicit, and the provider must be opt-in, not inherited."""
    text = (REPO_ROOT / "docker-compose.yml").read_text(encoding="utf-8")
    assert "AGENTRECX_AGENT_POLICY: deterministic" in text
    assert "AGENTRECX_LLM_API_KEY: ${AGENTRECX_LLM_API_KEY:-}" in text, (
        "the provider key must default to empty rather than to a value"
    )
    # No literal credential can appear in the compose file.
    assert not re.search(r"(?i)api[_-]?key\s*[:=]\s*[\"']?[A-Za-z0-9]{16,}", text)


def test_entrypoint_refuses_llm_mode_without_a_provider():
    """`AGENTRECX_AGENT_POLICY=llm` with no base URL must fail loudly, not fall back silently."""
    script = (REPO_ROOT / "scripts" / "entrypoint.sh").read_text(encoding="utf-8")
    assert "AGENTRECX_LLM_BASE_URL" in script and "AGENTRECX_LLM_MODEL" in script
    assert "exit 2" in script
    # And the offline branch must clear every provider variable it can find.
    assert "unset AGENTRECX_LLM_BASE_URL" in script


def test_entrypoint_refuses_llm_mode_without_a_provider_script(tmp_path: Path):
    """Execute the entrypoint: in `llm` mode with no base URL it must exit non-zero.

    Runs the real script rather than grepping it, because the guarantee is a *behaviour*: the
    container must fail loudly rather than fall back to a policy the operator did not ask for.
    """
    entrypoint = REPO_ROOT / "scripts" / "entrypoint.sh"
    env = {
        **os.environ,
        "AGENTRECX_AGENT_POLICY": "llm",
        "AGENTRECX_DEMO_ARTIFACTS_DIR": str(tmp_path),
    }
    env.pop("AGENTRECX_LLM_BASE_URL", None)
    env.pop("AGENTRECX_LLM_MODEL", None)
    result = subprocess.run(
        ["bash", str(entrypoint)], capture_output=True, text=True, check=False, env=env
    )
    assert result.returncode == 2, result.stdout + result.stderr
    assert "AGENTRECX_LLM_BASE_URL" in result.stderr
    # It must not have started the server.
    assert "starting AgentRec-X" not in result.stdout


def test_entrypoint_offline_mode_clears_provider_variables_without_printing_them(tmp_path: Path):
    """In the default mode the entrypoint unsets provider variables and never echoes a value."""
    entrypoint = REPO_ROOT / "scripts" / "entrypoint.sh"
    secret = "sk-THIS-MUST-NOT-BE-LOGGED-0123456789"
    # Point the generated artifacts at a directory this test owns, and give the script an
    # explicit interpreter (``AGENTRECX_PYTHON``) so it runs the same code this suite runs.  An
    # invalid device then makes the server exit immediately instead of binding a port.
    env = {
        **os.environ,
        "AGENTRECX_AGENT_POLICY": "deterministic",
        "AGENTRECX_DEMO_ARTIFACTS_DIR": str(tmp_path),
        "AGENTRECX_LLM_API_KEY": secret,
        "AGENTRECX_LLM_BASE_URL": "https://example.invalid/v1",
        "AGENTRECX_LLM_MODEL": "not-a-real-model",
        # Fail fast instead of binding a port: an invalid device makes the server exit.
        "AGENTRECX_DEVICE": "not-a-device",
        "AGENTRECX_PYTHON": sys.executable,
        "OMP_NUM_THREADS": "1",
    }
    result = subprocess.run(
        ["bash", str(entrypoint)],
        capture_output=True,
        text=True,
        check=False,
        env=env,
        timeout=300,
    )
    combined = result.stdout + result.stderr
    assert "agent policy : deterministic (offline" in combined
    assert secret not in combined, "the entrypoint must never print a credential value"
    assert "starting AgentRec-X" in combined


def test_dockerfile_pins_cpu_only_torch_and_is_not_root():
    text = (REPO_ROOT / "Dockerfile").read_text(encoding="utf-8")
    assert "requirements-cpu.txt" in text, "the CPU wheel index is what keeps the image CPU-only"
    assert "pip check" in text
    assert re.search(r"^USER (?!root)", text, re.MULTILINE), "the image must not run as root"
    assert "HEALTHCHECK" in text


def test_dockerignore_excludes_secrets_and_large_data():
    text = (REPO_ROOT / ".dockerignore").read_text(encoding="utf-8")
    for pattern in (".git", "data/", "runs/", ".env", "*.key"):
        assert pattern in text, f".dockerignore must exclude {pattern}"


# --------------------------------------------------------------------------- #
# Leak guards
# --------------------------------------------------------------------------- #


#: Words that mark a literal as a deliberate placeholder rather than a real credential.
#: The suite legitimately contains one such canary (a redaction test asserts the recorder
#: never writes a key to an artifact), and a real key would not spell anything.
PLACEHOLDER_MARKERS: tuple[str, ...] = (
    "CANARY",
    "FAKE",
    "EXAMPLE",
    "PLACEHOLDER",
    "DUMMY",
    "REDACT",
    "MUST-NOT-BE",
    "MUST_NOT_BE",
    "NOTAREALKEY",
)

#: Shapes a committed credential would have.  Deliberately specific: ``AGENTRECX_LLM_API_KEY``
#: appears legitimately as a variable *name*, so only an assignment to a long literal value is
#: flagged, and a marked placeholder is excluded.
CREDENTIAL_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"sk-[A-Za-z0-9]{20,}"),
    re.compile(r"(?i)authorization\s*:\s*[\"']?bearer\s+[A-Za-z0-9._-]{20,}"),
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
    re.compile(r"""(?i)(?:api[_-]?key|secret|token)\s*[:=]\s*["']([A-Za-z0-9/+_-]{24,})["']"""),
)


def _looks_like_a_placeholder(value: str) -> bool:
    upper = value.upper()
    return any(marker in upper for marker in PLACEHOLDER_MARKERS)


def test_no_credential_is_committed_anywhere():
    """No API key, token or private key in the tracked source tree.

    Checked by shape rather than by a known value, so a *new* credential is caught too.  A
    literal that spells a placeholder marker (the suite's redaction canary) is not a
    credential: a real key does not contain the word ``CANARY``.
    """
    offenders: list[str] = []
    result = subprocess.run(
        ["git", "ls-files"], cwd=REPO_ROOT, capture_output=True, text=True, check=True
    )
    for relative in result.stdout.split():
        path = REPO_ROOT / relative
        if not path.is_file() or path.stat().st_size > 2_000_000:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        for pattern in CREDENTIAL_PATTERNS:
            for match in pattern.finditer(text):
                captured = match.group(1) if match.groups() else match.group(0)
                if _looks_like_a_placeholder(captured):
                    continue
                offenders.append(f"{relative}: {captured[:12]}...")
    assert not offenders, "credential-shaped strings in tracked files: " + "; ".join(offenders)


def test_env_example_holds_no_secret_and_is_tracked():
    """The example must be committable (it is referenced from the README) and empty of values."""
    path = REPO_ROOT / ".env.example"
    text = path.read_text(encoding="utf-8")
    match = re.search(r"^AGENTRECX_LLM_API_KEY=(.*)$", text, re.MULTILINE)
    assert match is not None, "the example must document the key variable"
    assert match.group(1).strip() == "", "the example must not carry a key value"
    tracked = subprocess.run(
        ["git", "check-ignore", "-q", ".env.example"], cwd=REPO_ROOT, check=False
    )
    assert tracked.returncode != 0, ".env.example must not be git-ignored"


def test_runtime_does_not_read_a_key_unless_the_llm_policy_is_selected(
    monkeypatch: pytest.MonkeyPatch,
):
    """Only the exact switch value selects the model policy; anything else stays offline.

    Asserted on the switch itself rather than by composing a service: composing one needs a
    real engine and catalogue, and the property under test is precisely that no provider
    client is built (which would require credentials) unless the deployment opted in.
    """
    from recommendation.demo.agent_service import (
        AGENT_POLICY_LLM,
        ENV_AGENT_POLICY,
        llm_policy_selected,
    )

    assert AGENT_POLICY_LLM == "llm"
    monkeypatch.delenv(ENV_AGENT_POLICY, raising=False)
    assert llm_policy_selected() is False
    for value in ("deterministic", "LLM-ish", "false", "0", "off", " true", "llm2"):
        monkeypatch.setenv(ENV_AGENT_POLICY, value)
        assert llm_policy_selected() is False, f"{value!r} must not enable the model policy"
    monkeypatch.setenv(ENV_AGENT_POLICY, " LLM ")
    assert llm_policy_selected() is True

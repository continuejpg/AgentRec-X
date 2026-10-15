"""Guard: the LLM Agent must never be offered as unavailable while its provider is configured.

The browser greys out the LLM decision mode purely from ``/v1/demo/decision-modes`` ->
``modes[llm].available``, which is ``_llm_availability`` -> ``provider_availability()``.  That
probe reads the accepted provider configuration, so the whole choice hinges on whether the
service process inherited it.  These tests pin that chain at runtime (and pin the configuration
*names*), so a renamed variable - or a launch path that forgets to load them - fails here instead
of silently disabling the feature in the browser.
"""

from __future__ import annotations

import pathlib
import re

import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]
PROVIDER_NAMES = ("AGENTRECX_LLM_BASE_URL", "AGENTRECX_LLM_MODEL", "AGENTRECX_LLM_API_KEY",
                  "AGENTRECX_LLM_PROFILE")


def test_availability_follows_the_provider_configuration_at_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from recommendation.demo.llm_policy import provider_availability

    for name in PROVIDER_NAMES:
        monkeypatch.delenv(name, raising=False)
    unset = provider_availability()
    assert unset["available"] is False, "no configuration must not report the mode as available"
    assert unset.get("reason"), "an unavailable provider must carry a machine-readable reason"

    monkeypatch.setenv("AGENTRECX_LLM_BASE_URL", "http://127.0.0.1:9/v1")
    monkeypatch.setenv("AGENTRECX_LLM_MODEL", "deepseek-chat")
    monkeypatch.setenv("AGENTRECX_LLM_API_KEY", "unit-test-placeholder-not-a-credential")
    monkeypatch.setenv("AGENTRECX_LLM_PROFILE", "deepseek")
    configured = provider_availability()
    assert configured["available"] is True, (
        "with the accepted provider configuration present the LLM mode must be advertised as "
        "available; a False here is exactly what disables the browser option"
    )
    assert configured.get("provider") and configured.get("model")


def test_the_launcher_loads_the_same_configuration_names() -> None:
    """The names the code reads must be the names the launcher's .env supplies."""
    example = REPO / ".env.example"
    assert example.is_file(), "the tracked provider template must exist"
    declared = set(re.findall(r"^\s*(?:export\s+)?(AGENTRECX_LLM_[A-Z0-9_]+)",
                              example.read_text(encoding="utf-8"), re.M))
    for name in PROVIDER_NAMES:
        assert name in declared, f"{name} is read by the adapter but not documented in .env.example"

    launcher = (REPO / "scripts" / "start_demo.sh").read_text(encoding="utf-8")
    assert 'ENV_FILE="${REPO_ROOT}/.env"' in launcher, "the launcher must autoload <repo>/.env"
    assert ".env" in launcher


def test_the_browser_option_is_gated_only_on_the_served_availability() -> None:
    js = (REPO / "recommendation" / "web" / "app.js").read_text(encoding="utf-8")
    assert "els.modeLlm.disabled = !llmAvailable" in js
    # The view must not carry its own allow-list that could diverge from the server's answer.
    assert "llmAvailable = " in js
    assert 'mode === "llm"' in js

from __future__ import annotations

import pytest

from recommendation.demo.qwen_proposal import (
    ENV_ENABLED,
    ENV_URL,
    QwenProposalClient,
    QwenProposalError,
    qwen_proposal_enabled,
)


def test_qwen_proposal_disabled_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(ENV_ENABLED, raising=False)
    monkeypatch.delenv(ENV_URL, raising=False)
    assert qwen_proposal_enabled() is False
    with pytest.raises(QwenProposalError) as exc:
        QwenProposalClient.from_env()
    assert exc.value.code == "qwen_proposal_disabled"


def test_qwen_proposal_requires_explicit_url(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(ENV_ENABLED, "on")
    monkeypatch.delenv(ENV_URL, raising=False)
    with pytest.raises(QwenProposalError) as exc:
        QwenProposalClient.from_env()
    assert exc.value.code == "qwen_proposal_misconfigured"


def test_qwen_proposal_unavailable_raises_not_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(ENV_ENABLED, "on")
    monkeypatch.setenv(ENV_URL, "http://127.0.0.1:9")
    client = QwenProposalClient.from_env()
    with pytest.raises(QwenProposalError) as exc:
        client.propose(messages=[], current="I need a water filter.")
    assert exc.value.code == "qwen_proposal_unavailable"

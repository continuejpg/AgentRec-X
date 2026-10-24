"""Phase 2B: the Agent source budget is independent of the final ``k``."""

from __future__ import annotations

import pytest

from recommendation.demo.agent_policy import DemoSourcePlanPolicy
from recommendation.demo.source_limit import ENV_SOURCE_LIMIT, resolve_source_limit


def test_unconfigured_keeps_the_historical_behaviour(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(ENV_SOURCE_LIMIT, raising=False)
    assert resolve_source_limit(5) == 5
    assert resolve_source_limit(20) == 20


def test_blank_configuration_is_treated_as_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(ENV_SOURCE_LIMIT, "   ")
    assert resolve_source_limit(7) == 7


def test_configured_limit_is_independent_of_requested_k(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(ENV_SOURCE_LIMIT, "50")
    assert resolve_source_limit(5) == 50


@pytest.mark.parametrize("bad", ["abc", "0", "101", "-3", "5.0"])
def test_an_invalid_limit_is_refused_not_clamped(bad: str, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(ENV_SOURCE_LIMIT, bad)
    with pytest.raises(ValueError):
        resolve_source_limit(5)


def test_the_policy_asks_every_source_for_the_configured_budget() -> None:
    """Once the controller is built with the budget, all three action shapes use it."""
    policy = DemoSourcePlanPolicy(("history", "catalog_search"), default_k=37, use_memory=False)
    assert policy.default_k == 37
    assert policy._select_arguments("history").limit == 37
    assert policy._select_arguments("catalog_search").limit == 37


def test_the_policy_proposal_carries_the_budget_not_the_requested_k() -> None:
    policy = DemoSourcePlanPolicy(("history",), default_k=37, use_memory=False)

    class _Plan:  # the minimum the proposal needs
        available_actions = ("recommend_from_history",)
        step_index = 0

        class candidate_state:  # noqa: N801 - shaped like the real context
            grounded = False

        @staticmethod
        def action_available(_action: object) -> bool:
            return True

    proposal = policy.choose(_Plan())
    assert proposal.k == 37

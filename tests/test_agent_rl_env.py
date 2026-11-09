from __future__ import annotations

from training.agent_sft.agent_rl_env import AgentAction, Item, MinimalAgentEnv


def _env() -> MinimalAgentEnv:
    return MinimalAgentEnv(
        items=[Item("main", "main", 20.0), Item("accessory", "accessory", 5.0)],
        required_role="main",
        max_price=30.0,
        top_k=1,
    )


def test_illegal_finalize_is_rejected_and_penalized() -> None:
    env = _env()
    first = env.step("RETRIEVE")
    assert first.legal
    result = env.step("FINALIZE")
    assert result.legal
    assert result.done
    assert result.reward > 0


def test_unknown_action_and_duplicate_retrieve() -> None:
    env = _env()
    assert not env.step("FLY").legal
    assert env.step("RETRIEVE").legal
    assert not env.step("RETRIEVE").legal


def test_multi_action_rollout_records_each_action() -> None:
    env = _env()
    results = env.rollout([AgentAction.ASK, AgentAction.RETRIEVE, AgentAction.RERANK, AgentAction.FINALIZE])
    assert [item.legal for item in results] == [True, True, True, True]
    assert results[-1].done
    assert len(env.actions) == 4
    assert env.total_reward == sum(item["reward"] for item in env.actions)


def test_reward_does_not_depend_on_target_asin_but_on_role_and_price() -> None:
    env = MinimalAgentEnv(items=[Item("main", "main", 10.0)], required_role="main", max_price=50.0, top_k=1)
    env.rollout(["RETRIEVE", "FINALIZE"])
    assert env.total_reward > 0
    env2 = MinimalAgentEnv(items=[Item("accessory", "accessory", 10.0)], required_role="main", max_price=50.0, top_k=1)
    env2.rollout(["RETRIEVE", "FINALIZE"])
    assert env2.total_reward < 0

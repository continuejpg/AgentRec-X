"""Minimal Agent action/environment contract for future RL.

This is deliberately a contract + deterministic unit-test environment, not a claim that
AgentRec-X RL is ready.  It exposes ASK/RETRIEVE/RERANK/FINALIZE, legal-action checks,
state transitions and a reward that does not use a target ASIN or a model's own role
judgement as ground truth.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping, Sequence


class AgentAction(str, Enum):
    ASK = "ASK"
    RETRIEVE = "RETRIEVE"
    RERANK = "RERANK"
    FINALIZE = "FINALIZE"


@dataclass(frozen=True)
class Item:
    item_id: str
    role: str
    price: float | None = None


@dataclass
class StepResult:
    observation: dict[str, Any]
    reward: float
    done: bool
    legal: bool
    reason: str = ""


@dataclass
class MinimalAgentEnv:
    """Deterministic toy environment used to validate the RL interface contract."""

    items: Sequence[Item]
    required_role: str
    max_price: float | None = None
    max_steps: int = 6
    top_k: int = 3
    candidates: list[str] = field(default_factory=list)
    asked: bool = False
    retrieved: bool = False
    reranked: bool = False
    t: int = 0
    done: bool = False
    total_reward: float = 0.0
    actions: list[dict[str, Any]] = field(default_factory=list)

    def observe(self) -> dict[str, Any]:
        return {
            "t": self.t,
            "candidates": list(self.candidates),
            "asked": self.asked,
            "retrieved": self.retrieved,
            "reranked": self.reranked,
            "max_steps": self.max_steps,
        }

    def legal_actions(self) -> tuple[AgentAction, ...]:
        if self.done:
            return ()
        actions = [AgentAction.FINALIZE]
        if not self.asked:
            actions.append(AgentAction.ASK)
        if not self.retrieved:
            actions.append(AgentAction.RETRIEVE)
        if self.retrieved and not self.reranked:
            actions.append(AgentAction.RERANK)
        return tuple(actions)

    def _ranked(self) -> list[Item]:
        by_id = {item.item_id: item for item in self.items}
        selected = [by_id[item_id] for item_id in self.candidates if item_id in by_id]
        return sorted(selected, key=lambda item: (item.role != self.required_role, item.price if item.price is not None else 1e9))

    def step(self, action: str | AgentAction) -> StepResult:
        if self.done:
            return StepResult(self.observe(), -0.1, True, False, "episode already done")
        try:
            action = AgentAction(action)
        except ValueError:
            self.total_reward -= 0.2
            return StepResult(self.observe(), -0.2, False, False, "unknown action")
        if action not in self.legal_actions():
            self.total_reward -= 0.1
            return StepResult(self.observe(), -0.1, False, False, "illegal action")
        self.t += 1
        reward = 0.0
        done = False
        if action is AgentAction.ASK:
            self.asked = True
            reward = 0.1
        elif action is AgentAction.RETRIEVE:
            self.retrieved = True
            self.candidates = [item.item_id for item in self.items]
            reward = -0.02  # tool cost
        elif action is AgentAction.RERANK:
            self.reranked = True
            self.candidates = [item.item_id for item in self._ranked()]
            reward = -0.01
        elif action is AgentAction.FINALIZE:
            done = True
            final = self._ranked()[: self.top_k]
            eligible = [item for item in final if item.role == self.required_role]
            hard_ok = all(self.max_price is None or item.price is None or item.price <= self.max_price for item in final)
            reward = (1.0 if eligible else -1.0) + (0.2 if hard_ok else -0.5)
        self.total_reward += reward
        self.done = done
        self.actions.append({"action": action.value, "reward": reward, "done": done})
        return StepResult(self.observe(), reward, done, True)

    def rollout(self, actions: Sequence[str]) -> list[StepResult]:
        return [self.step(action) for action in actions]

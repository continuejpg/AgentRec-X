"""Minimal explicit Skill registry for AgentRec-X P16.

Not a generic framework: two task-specific callable skills, registered by name.  Each skill is a
thin orchestration over existing catalogue/search, Product Info, P15 Evidence Bundle, M5/M9 memory,
and the Client-side existing Gate/Planner contracts.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping


@dataclass(frozen=True)
class SkillSpec:
    name: str
    description: str
    handler: Callable[..., Mapping[str, Any]]


class SkillRegistry:
    """Explicit name -> callable registry; no dynamic loading or routing model."""

    def __init__(self) -> None:
        self._skills: dict[str, SkillSpec] = {}

    def register(self, spec: SkillSpec) -> None:
        if spec.name in self._skills:
            raise ValueError(f"duplicate skill {spec.name!r}")
        self._skills[spec.name] = spec

    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._skills))

    def describe(self, name: str) -> dict[str, Any]:
        spec = self._skills.get(str(name))
        if spec is None:
            raise KeyError(name)
        return {"name": spec.name, "description": spec.description}

    def invoke(self, name: str, **arguments: Any) -> Mapping[str, Any]:
        spec = self._skills.get(str(name))
        if spec is None:
            raise KeyError(f"unknown skill {name!r}; available: {', '.join(self.names())}")
        return spec.handler(**arguments)

"""Optional **external** benchmark adapters (AgentRec-X Offline Lab extension).

Everything under this package is an optional, isolated evaluation tool.  It is
never imported by production code, the AgentRec-X serving path or the core
``recommendation/`` lab components, and it is never vendored: an external
benchmark keeps its own source, licence, environment and data.

Deletion safety: removing this directory must leave AgentRec-X fully working and
must not change any test in ``tests/`` outside the external-benchmark test file.
"""

from __future__ import annotations

__all__: list[str] = []

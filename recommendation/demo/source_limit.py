"""Per-source candidate budget for the Agent control plane (Phase 2B).

Two different numbers used to be one here:

``requested_k``
    how many products the user finally sees;
``agent_source_limit``
    how many candidates each candidate source may return into the plane.

Until this module existed the second was hard-wired to the first, so asking for five
products also limited every source to five candidates - a deliberately shallow pool.

The default is unchanged.  With ``AGENTRECX_AGENT_SOURCE_LIMIT`` unset (or empty) the
source budget is exactly ``requested_k``, which is the behaviour every accepted deployment
already had.

The final truncation is *not* affected: :meth:`recommendation.demo.agent_service.DemoAgentService._execute_plan`
still cuts to ``requested_k`` after fusion, which is what keeps the visible answer the same
size no matter how deep the pool is.
"""

from __future__ import annotations

import os

from ..tools.schemas import MAX_K, MIN_K

__all__ = ["ENV_SOURCE_LIMIT", "resolve_source_limit"]

#: Runtime variable that sets the per-source candidate budget.
ENV_SOURCE_LIMIT = "AGENTRECX_AGENT_SOURCE_LIMIT"


def resolve_source_limit(requested_k: int, configured: object = None) -> int:
    """Return the per-source candidate budget for one run.

    ``configured`` exists so a caller (or a test) can supply the value directly; ``None``
    means "read the environment", and an unset/blank value means "keep the historical
    behaviour" (the budget equals ``requested_k``).

    Raises
    ------
    ValueError
        The configured value is not an integer in ``MIN_K..MAX_K``.  It is refused rather
        than clamped: silently rewriting a configured depth would make the experiment's
        own record disagree with what the sources were actually asked for.
    """
    raw = os.environ.get(ENV_SOURCE_LIMIT) if configured is None else configured
    text = "" if raw is None else str(raw).strip()
    if not text:
        return int(requested_k)
    try:
        value = int(text)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{ENV_SOURCE_LIMIT} must be an integer in {MIN_K}..{MAX_K}, got {text!r}"
        ) from exc
    if not MIN_K <= value <= MAX_K:
        raise ValueError(f"{ENV_SOURCE_LIMIT} must be in {MIN_K}..{MAX_K}, got {value}")
    return value

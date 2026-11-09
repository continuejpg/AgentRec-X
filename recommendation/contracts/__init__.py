"""Shared contracts for AgentRec-X backends and datasets (P14-0 seed).

This package exists because P14 introduces a new Shopping Companion dataset while
the historical recommendation system (Amazon Reviews 2023 *Sports_and_Outdoors*
5-core, integer ``item_id``) and the P13-R research catalogue (Amazon-C4 sampled
*Electronics*, ``parent_asin`` identity) must keep working unchanged.

The one thing that must never be allowed to drift silently is **product identity**:
one integer or string that means "item 12" in one catalogue does not mean the same
product in another.  :mod:`recommendation.contracts.namespaces` therefore states that
every product reference carries an explicit catalogue namespace.

This package deliberately contains *only* identity/structure contracts.  It holds no
I/O, no model, no Agent logic and no backend adapter -- P14 adds adapters on top.
"""

from __future__ import annotations

__all__: list[str] = []

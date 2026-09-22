"""AgentRec-X backend bridge package.

Owns exactly one thing: the adapter that speaks to an external generative-recommendation
backend over a filesystem + subprocess boundary.

* :mod:`recommendation.backends.tiger_backend` - ``TigerBackendAdapter``, the mirrored
  contract records, the deterministic SID resolver, and the conversion into the frozen
  evaluator's input shape.

Delivery rules this package obeys (see ``docs/TIGER_BACKEND.md`` and ``AGENTS.md`` section 19):

* **standard library + NumPy only** - no ``torch``, no ``transformers``, no backend import;
* **no ``parent_asin``** - canonical identity belongs to the layers outside this package, and
  an AST guard asserts its absence here;
* **no target, no split, no masking, no ranking, no metric** - the adapter prepares inputs and
  converts outputs; ``recommendation.evaluation.batched`` keeps every evaluation decision.
"""

from __future__ import annotations

__all__ = ["tiger_backend"]

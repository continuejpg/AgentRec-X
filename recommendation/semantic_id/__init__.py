"""Semantic-ID generative retrieval (model expansion step 2).

Two separately testable pieces:

* :mod:`recommendation.semantic_id.tokenizer` - the RQ-VAE-style semantic tokenizer that maps
  every catalogue item to a discrete Semantic ID, with a full audit of coverage, collisions and
  codebook utilisation;
* :mod:`recommendation.semantic_id.generator` - the autoregressive model that generates the next
  item's Semantic ID from a history of Semantic IDs.

Identity never crosses the boundary in the other direction: a generated Semantic ID is only
*evidence*, and :mod:`recommendation.semantic_id.resolver` is the deterministic component that
turns it back into a trusted catalogue item.  Nothing in this package is wired into the Agent
runtime in this step.
"""

from __future__ import annotations

__all__ = ["__version__"]

__version__ = "0.1.0"

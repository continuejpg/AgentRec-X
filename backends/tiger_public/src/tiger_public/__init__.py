"""Public-TIGER backend (Step 2.3 skeleton).

A generative-retrieval backend that AgentRec-X consumes through an explicit adapter
boundary.  It owns *learning and scoring* only:

* ``item_id <-> backend_row`` - opaque integer tags, never ``parent_asin``;
* item content encoding, RQ-VAE / Semantic IDs, the generative model (Steps 2.4-2.6);
* catalogue-constrained and certified retrieval, and raw item scores.

It owns none of the evaluation: no target, no split, no seen-item masking, no ranking, no
metric.  Those are refused by *shape* rather than by convention - no stage accepts a target
and no artifact declares one.

Step 2.3 status
---------------

Every stage in :mod:`tiger_public.cli` is a deterministic **placeholder**, marked
``step-2.3-placeholder-no-ml`` in the artifact it writes.  No model, no GPU, no training and
no measurement is produced here; the point of this milestone is the process, schema, digest
and score-batch boundary.

See ``docs/TIGER_BACKEND.md`` for the full specification.
"""

from __future__ import annotations

__all__ = ["CONTRACT_VERSION", "STUB_MARKER"]

from tiger_public.contracts import CONTRACT_VERSION

#: Written into every Step-2.3 artifact so a placeholder can never be mistaken for a result.
STUB_MARKER = "step-2.3-placeholder-no-ml"

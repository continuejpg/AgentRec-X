"""Shopping Companion integration package (P14-A).

This package is the **adapter layer** between the official Shopping Companion
benchmark and the accepted AgentRec-X Agent core.  It deliberately contains no
second Agent runtime, no second memory store, no second candidate ledger and no
second recommender: it translates real Shopping Companion data and tools into the
contracts that already exist in ``recommendation/`` (Planner Action contract,
CandidateLedger, GroundedReasoner / Eligibility, M5/M9 memory adapters, catalogue
BM25 search) and records a traceable trajectory.

Modules:

* :mod:`recommendation.shopping_companion.namespace` -- ``shopping_companion``
  identity helpers built on :mod:`recommendation.contracts.namespaces`.
* :mod:`recommendation.shopping_companion.catalogue` -- real limited catalogue
  loaded from the projected ``products.jsonl`` sample; records are normalised to the
  accepted :class:`recommendation.catalog.schemas.ProductMetadata` contract and the
  BM25 implementation is the existing
  :class:`recommendation.control.catalog_search.CatalogSearchSource`.
* :mod:`recommendation.shopping_companion.search` -- the P14-A ``product_search``
  tool (namespace, filters, provenance, latency).
* :mod:`recommendation.shopping_companion.product_info` -- the P14-A
  ``product_view``/Product-Info tool with explicit UNKNOWN semantics.
* :mod:`recommendation.shopping_companion.dataset` -- official train-task adapter
  (stage pairing, oracle isolation); written by the dataset-adapter workstream.
* :mod:`recommendation.shopping_companion.memory` -- M5/M9 memory adapter with an
  honest ``history_unavailable`` state; written by the memory workstream.
* :mod:`recommendation.shopping_companion.planner_bridge` -- Planner Action
  contract -> real tool execution bridge.
* :mod:`recommendation.shopping_companion.runtime` -- one end-to-end task run and
  the structured trajectory.
* :mod:`recommendation.shopping_companion.evaluator` -- post-run evaluator that is
  the only component allowed to read the evaluator-only ground truth.
"""

from __future__ import annotations

__all__: list[str] = []

"""CoShop external-benchmark adapter (optional, strictly isolated).

CoShop is an **external** benchmark; AgentRec-X is the subject under evaluation.
Nothing in this package is imported by production code.  The official CoShop
checkout is read-only reference material and is **never vendored or copied**
(the upstream repository carries no licence), so the adapter imports it lazily
from an external path (``coshop_root`` argument or ``AGENTRECX_COSHOP_ROOT``)
only when an evaluation is actually requested.

Import isolation properties:
    * ``import experiments.external_benchmarks.coshop`` needs only the Python
      standard library: no CoShop, pandas, torch or AgentRec-X package is
      imported at module import time;
    * the AgentRec-X offline snapshot contract
      (:mod:`recommendation.ranking.schemas`) is imported lazily inside the
      conversion function, only when a candidate snapshot is built;
    * hidden benchmark targets never enter a public task view, a candidate
      snapshot, a ranking input or the report payload.

See ``README.md`` for the boundary, the metric table and the reproducibility
conditions.
"""

from __future__ import annotations

from .adapter import (
    ADAPTER_VERSION,
    BASELINE_NAME,
    CANDIDATE_SOURCE,
    DEFAULT_DOMAIN,
    DEFAULT_K,
    DEFAULT_LIMIT,
    DEFAULT_SEED,
    DEFAULT_SPLIT,
    DEFAULT_VERSION,
    ENV_COSHOP_ROOT,
    PROTOCOL_NAME,
    SNAPSHOT_SOURCE,
    SUPPORTED_DOMAINS,
    SUPPORTED_SPLITS,
    SUPPORTED_VERSIONS,
    CoshopAdapterError,
    CoshopContractError,
    CoshopEvaluationReport,
    CoshopTaskView,
    CoshopUnavailable,
    HiddenTaskTargets,
    TaskEvaluationRecord,
    build_task_view,
    extract_hidden_targets,
    evaluate_coshop,
    rank_baseline_candidates,
    resolve_coshop_root,
    to_candidate_snapshot,
)

__all__ = [
    "ADAPTER_VERSION",
    "BASELINE_NAME",
    "CANDIDATE_SOURCE",
    "DEFAULT_DOMAIN",
    "DEFAULT_K",
    "DEFAULT_LIMIT",
    "DEFAULT_SEED",
    "DEFAULT_SPLIT",
    "DEFAULT_VERSION",
    "ENV_COSHOP_ROOT",
    "PROTOCOL_NAME",
    "SNAPSHOT_SOURCE",
    "SUPPORTED_DOMAINS",
    "SUPPORTED_SPLITS",
    "SUPPORTED_VERSIONS",
    "CoshopAdapterError",
    "CoshopContractError",
    "CoshopEvaluationReport",
    "CoshopTaskView",
    "CoshopUnavailable",
    "HiddenTaskTargets",
    "TaskEvaluationRecord",
    "build_task_view",
    "extract_hidden_targets",
    "evaluate_coshop",
    "rank_baseline_candidates",
    "resolve_coshop_root",
    "to_candidate_snapshot",
]

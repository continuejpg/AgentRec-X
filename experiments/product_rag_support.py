"""Real-stack support for the Milestone 8 smokes.

Builds the full M8 chain over the accepted real artifacts::

    accepted M5 checkpoint
        -> SASRecInferenceEngine (M6)
        -> RecommendationTool (M7A)
        -> AgentGraph (M7B/M7C)
        -> ProductEnricher over the M8-A catalogue metadata artifact
        -> grounded final response

Two smokes use this module:

* ``experiments/product_rag_smoke.py``    -- metadata + candidate-scoped retrieval;
* ``experiments/agent_product_rag_smoke.py`` -- the integrated Agent route.

Artifact discovery reuses the repository's existing surface
(:class:`~recommendation.api.app.ServiceSettings` for the checkpoint/manifest/mappings
and :mod:`recommendation.config` for processed paths), and the deterministic history
selection is the accepted M7C rule from :mod:`tests.agent_tool_e2e_runtime`, so the
same real user is used across M7C and M8.

No network access happens here, and no recommendation metric is computed: these
smokes validate that the real stack is correctly wired, not that its
recommendations are any good.
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from recommendation import config as project_config  # noqa: E402
from recommendation.agent import AgentGraph  # noqa: E402
from recommendation.catalog import MetadataIndex  # noqa: E402
from recommendation.rag import ProductEnricher  # noqa: E402
from tests.agent_tool_e2e_runtime import (  # noqa: E402
    ACCEPTED_CHECKPOINT_SHA256,
    ACCEPTED_MANIFEST_SHA256,
    AgentToolRuntime,
    HistorySelection,
    build_runtime,
    select_history,
    sha256_file,
)

__all__ = [
    "ACCEPTED_CHECKPOINT_SHA256",
    "ACCEPTED_MANIFEST_SHA256",
    "M8Runtime",
    "SMOKE_QUERY",
    "build_m8_runtime",
    "metadata_artifact_path",
    "metadata_manifest_path",
]

#: Fixed smoke query.  Chosen once, before looking at any result, and never tuned
#: against candidate output.
SMOKE_QUERY = "waterproof hiking boots for wet trails"


def metadata_artifact_path() -> Path:
    """Conventional normalized catalogue-metadata artifact path."""
    return project_config.default_catalog_metadata_path()


def metadata_manifest_path() -> Path:
    """Conventional catalogue-metadata processing manifest path."""
    return (
        project_config.PROCESSED_DIR
        / f"{project_config.metadata_category_slug()}_products_manifest.json"
    )


@dataclass
class M8Runtime:
    """The real M8 chain plus its metadata identities and timings."""

    tool_runtime: AgentToolRuntime
    metadata: MetadataIndex
    enricher: ProductEnricher
    graph: AgentGraph
    metadata_artifact: Path
    metadata_sha256: str
    metadata_bytes: int
    metadata_load_seconds: float
    raw_metadata_sha256: str
    raw_metadata_bytes: int

    # -- convenience passthroughs ----------------------------------------- #

    @property
    def engine(self) -> Any:
        """The real inference engine."""
        return self.tool_runtime.engine

    @property
    def tool(self) -> Any:
        """The accepted Recommendation Tool."""
        return self.tool_runtime.tool

    @property
    def decision_model(self) -> Any:
        """The injected deterministic decision model."""
        return self.tool_runtime.decision_model

    def metadata_summary(self) -> dict[str, Any]:
        """JSON-serialisable identity of the metadata layer."""
        envelope = self.metadata.envelope
        return {
            "artifact_path": str(self.metadata_artifact),
            "artifact_sha256": self.metadata_sha256,
            "artifact_bytes": self.metadata_bytes,
            "records": self.metadata.size,
            "normalization_version": envelope.get("normalization_version"),
            "duplicate_policy": envelope.get("duplicate_policy"),
            "source_url": envelope.get("source", {}).get("url"),
            "raw_sha256": self.raw_metadata_sha256,
            "raw_bytes": self.raw_metadata_bytes,
            "load_seconds": round(self.metadata_load_seconds, 3),
            "counts": envelope.get("counts", {}),
            "coverage": envelope.get("coverage", {}),
        }

    def chain_summary(self) -> dict[str, Any]:
        """JSON-serialisable identity of the recommendation chain."""
        return self.tool_runtime.metadata()


def build_m8_runtime(
    *,
    device: str = "cpu",
    k: int = 5,
    artifact: Path | None = None,
) -> M8Runtime:
    """Construct the real M8 chain exactly once.

    Raises
    ------
    FileNotFoundError
        If the accepted checkpoint, mappings or the M8-A metadata artifact is absent.
    """
    artifact_path = Path(artifact) if artifact is not None else metadata_artifact_path()
    if not artifact_path.exists():
        raise FileNotFoundError(
            f"catalogue metadata artifact not found: {artifact_path}\n"
            "Build it first with: .venv/bin/python -m experiments.prepare_product_metadata"
        )

    tool_runtime = build_runtime(device=device, k=k)

    started = time.perf_counter()
    metadata = MetadataIndex.load(artifact_path)
    load_seconds = time.perf_counter() - started

    enricher = ProductEnricher(metadata)
    graph = AgentGraph(
        tool_runtime.decision_model, tool_runtime.tool, product_enricher=enricher
    )

    envelope_source = metadata.envelope.get("source", {})
    return M8Runtime(
        tool_runtime=tool_runtime,
        metadata=metadata,
        enricher=enricher,
        graph=graph,
        metadata_artifact=artifact_path,
        metadata_sha256=sha256_file(artifact_path),
        metadata_bytes=artifact_path.stat().st_size,
        metadata_load_seconds=load_seconds,
        raw_metadata_sha256=str(envelope_source.get("raw_sha256", "")),
        raw_metadata_bytes=int(envelope_source.get("raw_size_bytes") or 0),
    )


def select_smoke_history() -> HistorySelection:
    """The same deterministic real-history rule the M7C smoke uses."""
    return select_history()

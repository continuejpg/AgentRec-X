"""Real-chain integration support for Milestone 7C.

Milestone 7C proves that the already accepted M7B graph can execute against the
already accepted M7A Recommendation Tool, backed by the real M6
:class:`~recommendation.inference.sasrec.SASRecInferenceEngine` and the accepted
Milestone 5 ``best.pt``::

    trusted application history
        -> AgentGraph
        -> decision = RECOMMEND(k)
        -> RecommendationTool
        -> SASRecInferenceEngine
        -> accepted SASRec best.pt
        -> full-catalog ranking
        -> structured recommendation result
        -> AgentGraph finalize
        -> honest final response

This module builds that chain **once** and exposes it to the Milestone 7C test and
smoke.  It adds no production behaviour: it constructs the accepted classes with
the accepted configuration and observes them.

Artifact discovery
------------------
Paths come from the repository's existing configuration surface, not from a second
loader and not from machine-specific constants here:

* :class:`~recommendation.api.app.ServiceSettings` supplies the repository-relative
  defaults used by the accepted Milestone 6 service
  (``runs/sasrec_canonical_2026/best.pt``, its ``run.json`` manifest, and
  ``data/processed/Sports_and_Outdoors_mappings.json``) and honours the existing
  ``AGENTRECX_*`` environment overrides;
* :meth:`ServiceSettings.to_inference_config` builds the engine configuration,
  including accepted-checkpoint digest verification.  Only that configuration
  object is reused -- **no HTTP request is made and the service is never started**;
* :mod:`recommendation.config` supplies ``DATA_DIR`` for the processed sequences
  artifact.

The point of reusing this surface is that the E2E test verifies the *same* artifact
identity and engine configuration the accepted service uses.

No internal HTTP
----------------
The chain is constructed entirely in process.  ``RecommendationTool`` receives the
engine object directly; nothing resolves a host, port or URL, and ``requests`` /
``httpx`` are never imported here.

Invocation counting
-------------------
The accepted engine and Tool are not instrumented (production code is unchanged).
Counting is done with a thin subclass, so the object under test is still the real
``SASRecInferenceEngine`` running its real code -- only the entry point increments a
counter.
"""

from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from recommendation import config as project_config  # noqa: E402
from recommendation.agent import (  # noqa: E402
    AgentDecision,
    AgentGraph,
    DecisionMessage,
)
from recommendation.api.app import (  # noqa: E402
    ACCEPTED_CHECKPOINT_SHA256,
    ServiceSettings,
)
from recommendation.inference import (  # noqa: E402
    InferenceConfig,
    SASRecInferenceEngine,
)
from recommendation.tools import RecommendationTool  # noqa: E402

#: Digest of the accepted Milestone 5 run manifest, re-exported here so the M7C
#: test and smoke can assert manifest identity without restating the value.
ACCEPTED_MANIFEST_SHA256 = "e3549049f955bb0540444c221523444b2a333e8df5eac6b97984342e40c6e2c7"

#: ``k`` used by the formal M7C integration path unless overridden.
FORMAL_K = 5

#: Minimum number of model-visible transitions a selected history must provide.
MIN_HISTORY_LENGTH = 3


def sha256_file(path: Path) -> str:
    """Return the SHA-256 of a file, streamed so large files are never fully read."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class HistorySelection:
    """One deterministically selected trusted history, with its provenance."""

    parent_asins: tuple[str, ...]
    user_id: str
    user_int_id: int
    source_length: int
    source_path: Path

    @property
    def length(self) -> int:
        """Number of supplied history items."""
        return len(self.parent_asins)

    @property
    def digest(self) -> str:
        """Short one-way digest, safe to print in logs and reports."""
        joined = "\x1f".join(self.parent_asins).encode("utf-8")
        return hashlib.sha256(joined).hexdigest()[:16]

    @property
    def distinct_length(self) -> int:
        """Distinct items in the supplied history (duplicates are preserved)."""
        return len(set(self.parent_asins))

    def as_dict(self) -> dict[str, Any]:
        """JSON-serialisable provenance record (never the full history)."""
        return {
            "user_id": self.user_id,
            "user_int_id": self.user_int_id,
            "history_length": self.length,
            "history_distinct": self.distinct_length,
            "history_digest": self.digest,
            "source_sequence_length": self.source_length,
            "source_path": str(self.source_path),
        }


def sequences_path() -> Path:
    """Path to the accepted processed sequences artifact."""
    return project_config.PROCESSED_DIR / f"{project_config.DEFAULT_CATEGORY}_sequences.json"


def select_history(
    *,
    min_length: int = MIN_HISTORY_LENGTH,
    exclude_most_recent: bool = True,
    require_partial_availability: bool = False,
    max_k: int = 100,
    sequences_file: Path | None = None,
) -> HistorySelection:
    """Select one real trusted history deterministically from accepted artifacts.

    Selection rule (fully reproducible, independent of recommendation output):

    1. read the accepted processed sequences artifact;
    2. walk user records in stored order, which is ascending ``user_int_id`` and is
       therefore stable and independent of dictionary or hash ordering;
    3. take the **first** record that satisfies every filter below.

    Filters:

    * ``len(parent_asins) > min_length`` so the history has at least
      ``min_length`` supplied items;
    * the supplied history is ``parent_asins[:-2] + [parent_asins[-2]]`` when
      ``exclude_most_recent`` is true -- the training prefix plus the validation
      target, with the final leave-one-out **test target excluded**, matching the
      accepted Milestone 7A integration convention so no test-target item is ever
      fed to the model;
    * at least one distinct item remains in the supplied history;
    * when ``require_partial_availability`` is set, the supplied history must already
      contain more than ``max_k`` items, so a request for ``k = max_k`` cannot return
      ``k`` candidates.

    The selection never inspects scores or candidates, so it cannot be tuned toward a
    preferred output.
    """
    path = sequences_file or sequences_path()
    if not path.exists():
        raise FileNotFoundError(f"accepted sequences artifact not found: {path}")

    payload = json.loads(path.read_text(encoding="utf-8"))
    for record in payload["sequences"]:
        asins = list(record["parent_asins"])
        history = asins[:-2] + [asins[-2]] if exclude_most_recent else asins
        if len(history) <= min_length:
            continue
        if not any(history):
            continue
        if len(set(history)) < 1:
            continue
        if require_partial_availability and len(history) <= max_k:
            continue
        return HistorySelection(
            parent_asins=tuple(history),
            user_id=str(record["user_id"]),
            user_int_id=int(record["user_int_id"]),
            source_length=int(len(asins)),
            source_path=path,
        )

    raise LookupError(
        "no user in the accepted sequences artifact satisfies the requested filters "
        f"(min_length={min_length}, require_partial_availability={require_partial_availability})"
    )


class CountingEngine(SASRecInferenceEngine):
    """The real engine plus an invocation counter.

    This is a subclass, not a mock: ``recommend`` runs the real accepted code path
    (history encoding, mapping, masking, SASRec scoring, ranking) and only counts
    entry.  Used to prove dependency reuse and "no inference on the direct route"
    without touching production code.
    """

    def __init__(self, config: InferenceConfig) -> None:
        super().__init__(config)
        #: Number of ``recommend`` invocations (each may score the catalog once).
        self.recommend_calls: int = 0
        #: Histor*ies* observed, recorded as digests only.
        self.history_digests: list[str] = []

    def recommend(self, history_parent_asins: Any, k: int = 10):  # noqa: ANN201
        """Count the call, record a digest, then delegate to the real engine."""
        self.recommend_calls += 1
        self.history_digests.append(
            hashlib.sha256(
                "\x1f".join(str(a) for a in history_parent_asins).encode("utf-8")
            ).hexdigest()[:16]
        )
        return super().recommend(history_parent_asins, k=k)

    def reset_counters(self) -> None:
        """Zero the counters so a later assertion measures only new invocations."""
        self.recommend_calls = 0
        self.history_digests = []


class SwitchableDecisionModel:
    """A deterministic, injected decision model for the formal M7C path.

    Milestone 7C is not an LLM-provider milestone: the route is chosen by an
    injected object, never by an external service.  This model returns exactly the
    decision it is told to return, and defaults to::

        AgentDecision(action=RECOMMEND, k=FORMAL_K)

    ``set_decision`` exists only so one runtime can also exercise the direct route;
    it is still fully deterministic and records every prompt it was shown so tests
    can prove the decision model never receives trusted history.
    """

    def __init__(self, decision: AgentDecision | None = None) -> None:
        self.decision = decision or AgentDecision(action="recommend", k=FORMAL_K)
        #: Every message tuple handed to this model, for leak assertions.
        self.calls: list[tuple[DecisionMessage, ...]] = []

    @property
    def call_count(self) -> int:
        """How many times ``decide`` was invoked."""
        return len(self.calls)

    @property
    def last_prompt_text(self) -> str:
        """All prompt content concatenated, for boundary assertions."""
        messages = self.calls[-1] if self.calls else ()
        return "\n".join(getattr(m, "content", "") for m in messages)

    def set_decision(self, decision: AgentDecision) -> None:
        """Switch the returned decision (used to exercise the direct route)."""
        self.decision = decision

    def decide(self, messages: Sequence[DecisionMessage]) -> AgentDecision:
        """Record the prompt and return the configured decision."""
        self.calls.append(tuple(messages))
        return self.decision


@dataclass
class AgentToolRuntime:
    """The real M7C chain, constructed exactly once.

    ``engine -> tool -> graph`` are built together and reused, which is what makes
    "do not reload the model per call" observable: object identity is stable and the
    engine's ``loaded_at`` timestamp never changes across graph invocations.
    """

    engine: CountingEngine
    tool: RecommendationTool
    graph: AgentGraph
    decision_model: SwitchableDecisionModel
    settings: ServiceSettings
    checkpoint_sha256: str
    manifest_sha256: str
    timings: dict[str, float] = field(default_factory=dict)

    @property
    def loaded_at(self) -> float:
        """When the accepted checkpoint finished loading."""
        return self.engine.loaded_at

    def metadata(self) -> dict[str, Any]:
        """JSON-serialisable identity of the constructed chain."""
        return {
            "checkpoint_sha256": self.checkpoint_sha256,
            "manifest_sha256": self.manifest_sha256,
            "checkpoint_path": str(self.settings.checkpoint_path),
            "mappings_path": str(self.settings.mappings_path),
            "manifest_path": str(self.settings.manifest_path),
            "device": str(self.engine.device),
            "num_items": self.engine.num_items,
            "max_seq_len": self.engine.max_seq_len,
            "model_parameters": int(sum(p.numel() for p in self.engine.model.parameters())),
            "graph_version": self.graph.version,
            "tool": self.tool.metadata(),
            "load_seconds": self.timings.get("load_seconds"),
            "build_seconds": self.timings.get("build_seconds"),
        }


def build_runtime(
    *,
    settings: ServiceSettings | None = None,
    device: str = "cpu",
    k: int = FORMAL_K,
) -> AgentToolRuntime:
    """Construct the real chain once: engine -> Tool -> AgentGraph.

    ``settings`` defaults to :meth:`ServiceSettings.from_env`, which resolves the
    accepted repository-relative artifacts and enables checkpoint digest
    verification.  No HTTP client is created and the FastAPI application is never
    instantiated.
    """
    import time

    settings = settings or ServiceSettings.from_env()
    settings = ServiceSettings(
        checkpoint_path=settings.checkpoint_path,
        manifest_path=settings.manifest_path,
        mappings_path=settings.mappings_path,
        device=device,
        # Identity verification is mandatory for the formal M7C path.
        verify_checkpoint_sha256=True,
    )

    started = time.perf_counter()
    # Reuse the accepted service's configuration translation, including the
    # accepted-checkpoint digest check.  Only the config object is used in process.
    engine = CountingEngine(settings.to_inference_config())
    load_seconds = time.perf_counter() - started

    started = time.perf_counter()
    tool = RecommendationTool(engine)
    decision_model = SwitchableDecisionModel(AgentDecision(action="recommend", k=k))
    graph = AgentGraph(decision_model, tool)
    build_seconds = time.perf_counter() - started

    manifest_sha256 = (
        sha256_file(settings.manifest_path)
        if settings.manifest_path is not None and Path(settings.manifest_path).exists()
        else ""
    )
    return AgentToolRuntime(
        engine=engine,
        tool=tool,
        graph=graph,
        decision_model=decision_model,
        settings=settings,
        checkpoint_sha256=engine.checkpoint_sha256,
        manifest_sha256=manifest_sha256,
        timings={"load_seconds": load_seconds, "build_seconds": build_seconds},
    )


__all__ = [
    "ACCEPTED_CHECKPOINT_SHA256",
    "ACCEPTED_MANIFEST_SHA256",
    "FORMAL_K",
    "MIN_HISTORY_LENGTH",
    "AgentToolRuntime",
    "CountingEngine",
    "HistorySelection",
    "SwitchableDecisionModel",
    "build_runtime",
    "select_history",
    "sequences_path",
    "sha256_file",
]

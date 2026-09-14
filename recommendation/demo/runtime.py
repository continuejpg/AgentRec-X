"""Milestone 11 demo runtime: build the accepted backend once, serve many turns.

Composition path::

    browser
      -> FastAPI demo endpoints            recommendation/api/demo_routes.py
      -> DemoSessionManager                session registry, turn ids, isolation
      -> AgentGraph                        accepted M7B/M8/M9/M10A/M10B/M10D route
      -> RecommendationTool -> SASRecInferenceEngine
      -> ProductEnricher    -> MetadataIndex
      -> PreferenceCandidateMatcher -> PreferenceReranker
      -> PreferenceMemoryService -> SQLitePreferenceStore

Everything expensive is constructed exactly once per server process:

==============================  ==================================================
``SASRecInferenceEngine``       ~349 MB checkpoint, loaded once
``RecommendationTool``          thin wrapper over that one engine
``MetadataIndex``               ~300 MB catalogue artifact, loaded once
``ProductEnricher``             thin wrapper over that one index
``PreferenceMemoryService``     one SQLite store for the whole demo
``PreferenceCandidateMatcher``  stateless, one instance
``PreferenceReranker``          stateless, one instance
``DemoSessionManager``          bounded live-session registry
==============================  ==================================================

Why the compiled graph is cached per ``(k, user_key)``
------------------------------------------------------
The accepted Milestone 9 graph contract binds a memory namespace at *construction*:
``AgentGraph(memory_service=..., user_key=...)`` loads preferences for that
``user_key``.  Two demo sessions must never share preference memory, so one graph
instance cannot serve two sessions.  Rather than change the frozen Milestone 7-10
components, the runtime caches one compiled :class:`AgentGraph` per
``(k, session user_key)``.

That cache is cheap and bounded: compiling a graph is pure Python over the **same**
process-scoped collaborators (engine, Tool, enricher, memory service, matcher,
reranker), and no model, index or service is ever rebuilt.  Entries are dropped when a
session is reset, and the cache has a hard size cap as a backstop.

The other per-request value is ``k``.  The accepted trust boundary routes ``k``
exclusively through ``AgentDecision``, so it cannot be passed to the graph any other
way; a per-``k`` decision model is why ``k`` is part of the cache key.
"""

from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping

from recommendation import config as project_config
from recommendation.agent import AgentGraph, DecisionModel
from recommendation.catalog import MetadataIndex
from recommendation.memory import (
    MEMORY_DB_ENV_VAR,
    PreferenceMemoryService,
    RuleBasedPreferenceExtractor,
    SQLitePreferenceStore,
)
from recommendation.preference_matching import PreferenceCandidateMatcher
from recommendation.rag import ProductEnricher
from recommendation.reranking import PreferenceReranker
from recommendation.tools import RecommendationTool

from .decision import DEFAULT_DEMO_K, DemoDecisionModel
from .profiles import DEFAULT_DEMO_PROFILE_COUNT, DemoProfile, build_demo_profiles
from .sessions import DEFAULT_MAX_SESSIONS, DemoSessionManager

__all__ = [
    "DEFAULT_DEMO_MEMORY_DB",
    "DemoRuntime",
    "DemoRuntimeError",
    "build_demo_runtime",
    "catalog_metadata_path",
    "memory_database_path",
]

#: Default preference-memory database for the demo.  One database file for the whole
#: runtime (never one per session); each session owns a distinct ``user_key`` namespace
#: inside it.  Git-ignored: it is runtime data, not an artifact.
DEFAULT_DEMO_MEMORY_DB = project_config.ARTIFACTS_DIR / "demo" / "preference_memory.sqlite3"

#: Hard cap on cached compiled graphs, independent of the session cap, so a long-lived
#: process with many short sessions cannot grow the cache without limit.
MAX_CACHED_GRAPHS = 512


class DemoRuntimeError(RuntimeError):
    """The demo runtime could not be composed from the accepted artifacts."""

    code = "demo_unavailable"


def catalog_metadata_path() -> Path:
    """Conventional normalized catalogue-metadata artifact path."""
    return project_config.default_catalog_metadata_path()


def memory_database_path() -> Path:
    """Resolve the demo preference-memory database path.

    Precedence: ``AGENTRECX_MEMORY_DB`` (the accepted Milestone 9 environment variable),
    then the demo default.
    """
    configured = os.environ.get(MEMORY_DB_ENV_VAR)
    if configured:
        return Path(configured)
    return DEFAULT_DEMO_MEMORY_DB


@dataclass
class DemoRuntime:
    """The composed demo stack plus its bounded compiled-graph cache."""

    settings: Any
    engine: Any
    tool: RecommendationTool
    metadata: MetadataIndex
    enricher: ProductEnricher
    memory_service: PreferenceMemoryService
    matcher: PreferenceCandidateMatcher
    reranker: PreferenceReranker
    profiles: Mapping[str, DemoProfile]
    sessions: DemoSessionManager
    decision_model_factory: Callable[[int], DecisionModel] = DemoDecisionModel
    metadata_artifact: Path | None = None
    memory_artifact: Path | None = None
    max_cached_graphs: int = MAX_CACHED_GRAPHS
    _graphs: dict[tuple[int, str], AgentGraph] = field(default_factory=dict, repr=False)
    _graph_lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    # -- readiness --------------------------------------------------------- #

    @property
    def ready(self) -> bool:
        """True when the full accepted chain is present and usable."""
        return bool(
            self.engine is not None
            and self.metadata is not None
            and self.memory_service is not None
            and self.matcher is not None
            and self.reranker is not None
            and self.profiles
        )

    @property
    def model_loaded(self) -> bool:
        """True when the injected engine reports itself ready."""
        checker = getattr(self.engine, "is_ready", None)
        return bool(checker()) if callable(checker) else self.engine is not None

    @property
    def metadata_loaded(self) -> bool:
        """True when a catalogue index is present."""
        return self.metadata is not None

    # -- graph cache ------------------------------------------------------- #

    def graph_for(self, k: int = DEFAULT_DEMO_K, *, user_key: str) -> AgentGraph:
        """Return the compiled graph for one ``(k, user_key)`` pair.

        The graph is reused across turns of the same session and across sessions that
        share a ``k``; only the memory namespace and the decision model differ, and both
        are immutable per cached instance.
        """
        cache_key = (k, user_key)
        with self._graph_lock:
            graph = self._graphs.get(cache_key)
            if graph is not None:
                return graph
        graph = AgentGraph(
            self.decision_model_factory(k),
            self.tool,
            product_enricher=self.enricher,
            memory_service=self.memory_service,
            user_key=user_key,
            preference_matcher=self.matcher,
            reranker=self.reranker,
        )
        with self._graph_lock:
            existing = self._graphs.get(cache_key)
            if existing is not None:
                return existing
            if len(self._graphs) >= self.max_cached_graphs:
                # FIFO eviction: dict preserves insertion order, so the oldest key goes.
                oldest = next(iter(self._graphs))
                self._graphs.pop(oldest, None)
            self._graphs[cache_key] = graph
            return graph

    def release_user_key(self, user_key: str) -> int:
        """Drop every cached graph bound to a memory namespace; returns how many."""
        with self._graph_lock:
            stale = [key for key in self._graphs if key[1] == user_key]
            for key in stale:
                self._graphs.pop(key, None)
            return len(stale)

    @property
    def compiled_graph_count(self) -> int:
        """How many ``(k, user_key)`` pairs currently have a compiled graph."""
        with self._graph_lock:
            return len(self._graphs)

    # -- diagnostics ------------------------------------------------------- #

    def build_report(self) -> dict[str, Any]:
        """JSON-serialisable identity of the composed runtime.

        Reports *counts* (each heavy object exists once) and never a filesystem path that
        a client could use to locate the host's data.
        """
        return {
            "engine_builds": 1,
            "tool_builds": 1,
            "metadata_loads": 1,
            "enricher_builds": 1,
            "memory_service_builds": 1,
            "matcher_builds": 1,
            "reranker_builds": 1,
            "profile_count": len(self.profiles),
            "max_sessions": self.sessions.max_sessions,
            "profile_ids": tuple(self.profiles),
            "metadata_records": getattr(self.metadata, "size", None),
            "compiled_graph_count": self.compiled_graph_count,
        }

    def close(self) -> None:
        """Close the preference store, if it owns one."""
        closer = getattr(getattr(self.memory_service, "store", None), "close", None)
        if callable(closer):
            closer()

    # -- composition ------------------------------------------------------- #

    @classmethod
    def from_env(
        cls,
        *,
        engine: Any = None,
        metadata: MetadataIndex | None = None,
        memory_service: PreferenceMemoryService | None = None,
        store: Any = None,
        profiles: Mapping[str, DemoProfile] | None = None,
        profile_count: int = DEFAULT_DEMO_PROFILE_COUNT,
        decision_model_factory: Callable[[int], DecisionModel] = DemoDecisionModel,
        max_sessions: int = DEFAULT_MAX_SESSIONS,
        ttl_seconds: float | None = None,
        clock: Callable[[], float] = time.time,
        memory_db: Path | None = None,
    ) -> DemoRuntime:
        """Compose the demo runtime from ``AGENTRECX_*`` configuration.

        See :func:`build_demo_runtime`, which this delegates to.
        """
        return build_demo_runtime(
            engine=engine,
            metadata=metadata,
            memory_service=memory_service,
            store=store,
            profiles=profiles,
            profile_count=profile_count,
            decision_model_factory=decision_model_factory,
            max_sessions=max_sessions,
            ttl_seconds=ttl_seconds,
            clock=clock,
            memory_db=memory_db,
        )


def build_demo_runtime(
    *,
    settings: Any = None,
    engine: Any = None,
    metadata: MetadataIndex | None = None,
    memory_service: PreferenceMemoryService | None = None,
    store: Any = None,
    profiles: Mapping[str, DemoProfile] | None = None,
    profile_count: int = DEFAULT_DEMO_PROFILE_COUNT,
    decision_model_factory: Callable[[int], DecisionModel] = DemoDecisionModel,
    max_sessions: int = DEFAULT_MAX_SESSIONS,
    ttl_seconds: float | None = None,
    clock: Callable[[], float] = time.time,
    memory_db: Path | None = None,
) -> DemoRuntime:
    """Compose the demo runtime, constructing heavy objects once.

    Every collaborator may be injected, which is how the offline tests and the formal
    smoke run against a small synthetic checkpoint and a synthetic catalogue instead of
    the accepted 349 MB / 300 MB artifacts.  Injection is explicit: there is no hidden
    default engine and no global.

    Raises
    ------
    DemoRuntimeError
        A required accepted artifact is missing or unusable.  Failing here is deliberate:
        :mod:`recommendation.api.app` calls this during startup, so the server refuses to
        start rather than failing on the first browser request.
    """
    from recommendation.api.app import ServiceSettings  # local import: avoids a cycle

    settings = settings if settings is not None else ServiceSettings.from_env()

    resolved_engine = engine
    if resolved_engine is None:
        try:
            # Reuse the accepted Milestone 6 construction path (including its
            # checkpoint-digest verification switch) instead of duplicating model loading.
            from recommendation.api.app import build_engine

            resolved_engine, error = build_engine(settings)
        except Exception as exc:  # noqa: BLE001 - normalize to a demo-level failure
            raise DemoRuntimeError(
                f"the accepted checkpoint could not be loaded: {type(exc).__name__}"
            ) from exc
        if resolved_engine is None:
            raise DemoRuntimeError(f"the accepted checkpoint could not be loaded: {error}")

    artifact: Path | None = None
    resolved_metadata = metadata
    if resolved_metadata is None:
        artifact = catalog_metadata_path()
        if not artifact.exists():
            raise DemoRuntimeError(
                "the catalogue metadata artifact is missing; build it with "
                "`.venv/bin/python -m experiments.prepare_product_metadata`"
            )
        resolved_metadata = MetadataIndex.load(artifact)

    resolved_memory = memory_service
    resolved_db: Path | None = memory_db
    if resolved_memory is None:
        if store is None:
            resolved_db = Path(memory_db) if memory_db is not None else memory_database_path()
            resolved_db.parent.mkdir(parents=True, exist_ok=True)
            store = SQLitePreferenceStore(resolved_db)
        resolved_memory = PreferenceMemoryService(store, RuleBasedPreferenceExtractor())

    resolved_profiles = profiles if profiles is not None else build_demo_profiles(count=profile_count)

    return DemoRuntime(
        settings=settings,
        engine=resolved_engine,
        tool=RecommendationTool(resolved_engine),
        metadata=resolved_metadata,
        enricher=ProductEnricher(resolved_metadata),
        memory_service=resolved_memory,
        matcher=PreferenceCandidateMatcher(),
        reranker=PreferenceReranker(),
        profiles=resolved_profiles,
        sessions=DemoSessionManager(
            resolved_profiles, max_sessions=max_sessions, ttl_seconds=ttl_seconds, clock=clock
        ),
        decision_model_factory=decision_model_factory,
        metadata_artifact=artifact,
        memory_artifact=resolved_db,
    )

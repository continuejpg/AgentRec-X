"""FastAPI recommendation service (Milestone 6) plus the Milestone 11 demo API.

The HTTP layer is a thin adapter: it validates the wire contract, delegates to
:class:`~recommendation.inference.sasrec.SASRecInferenceEngine`, and maps domain
errors to status codes.  It contains **no** model, encoding, masking or ranking logic.

Endpoints::

    GET  /health              readiness (process up vs. model loaded)   [M6, frozen]
    GET  /v1/model            non-sensitive model metadata               [M6, frozen]
    POST /v1/recommend        top-k recommendations for a parent_asin history

    GET  /v1/demo/health      demo readiness                            [M11]
    GET  /v1/demo/profiles    server-owned demo profiles                [M11]
    POST /v1/demo/sessions    create an isolated demo session           [M11]
    GET  /v1/demo/sessions/{id}          session state + preferences    [M11]
    POST /v1/demo/sessions/{id}/chat     one conversational turn        [M11]
    DELETE /v1/demo/sessions/{id}        reset the session              [M11]
    GET  /demo/               the browser demo (static assets)          [M11]
    GET  /                    redirect to /demo/                        [M11]

The three Milestone 6 endpoints keep their exact paths, schemas and semantics.  The
demo API is **opt-in**: ``create_app(..., enable_demo=True)``.  With it disabled (the
default, and what the Milestone 6 tests use) the application is byte-for-byte the
Milestone 6 service.

The model is loaded once during application startup, never per request.  The demo
runtime is likewise composed once (see :mod:`recommendation.demo.runtime`) and shares
that same engine -- the demo never loads a second copy of the checkpoint.

Run the development server (local binding by default)::

    .venv/bin/python -m recommendation.api.app
    .venv/bin/uvicorn recommendation.api.app:app --host 127.0.0.1 --port 8000
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, AsyncIterator, Callable

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError as FastAPIRequestValidationError
from fastapi.responses import JSONResponse, RedirectResponse

from recommendation import config as project_config
from recommendation.inference import (
    InferenceConfig,
    InferenceError,
    RequestValidationError as EngineRequestValidationError,
    SASRecInferenceEngine,
    UnknownItemError,
)

from .schemas import (
    ErrorResponse,
    HealthResponse,
    ModelInfoResponse,
    RecommendRequest,
    RecommendResponse,
)

#: Module logger.  Demo turn diagnostics are logged at INFO with a deliberately narrow
#: field set (session id, turn id, route, counts, latency) and never the raw user text,
#: any secret, any environment variable or any store content.
LOGGER = logging.getLogger("agentrecx.api")

#: Repository layout defaults, matching the accepted Milestone 5 artifacts.
DEFAULT_CHECKPOINT = project_config.REPO_ROOT / "runs" / "sasrec_canonical_2026" / "best.pt"
DEFAULT_MANIFEST = project_config.REPO_ROOT / "runs" / "sasrec_canonical_2026" / "run.json"
DEFAULT_MAPPINGS = project_config.DATA_DIR / "processed" / "Sports_and_Outdoors_mappings.json"

#: Serving V1 collaborators.  The Two-Tower directory holds ``checkpoint.pt`` and the
#: ``metadata_categories.json`` the engine resolves as its sibling; the catalogue is the same
#: normalized artifact the enricher reads.
DEFAULT_TWOTOWER_DIR = project_config.REPO_ROOT / "runs" / "twotower_public_2026_full"
DEFAULT_CATALOG = project_config.DATA_DIR / "processed" / "Sports_and_Outdoors_products.jsonl"

#: ``auto`` uses Serving V1 when its artifacts are present and falls back to SASRec-only **with a
#: logged reason**; ``rrf3`` demands fusion and reports the process unready rather than degrading;
#: ``sasrec`` keeps the accepted single-source engine.
SERVING_FUSION_MODES: tuple[str, ...] = ("auto", "rrf3", "sasrec")

#: Digest of the accepted formal checkpoint, used for startup identity verification.
ACCEPTED_CHECKPOINT_SHA256 = (
    "352bd3ae7ebc5e20adbaac0388ac20fa4b9cc547a579500191d940f77a105912"
)

#: Directory holding the browser demo's static assets.
WEB_ROOT = project_config.REPO_ROOT / "recommendation" / "web"


@dataclass
class ServiceSettings:
    """Service configuration.

    Values may be supplied explicitly or through environment variables so that no
    machine-specific absolute path is hard-coded in the request path.
    """

    checkpoint_path: Path = DEFAULT_CHECKPOINT
    manifest_path: Path | None = DEFAULT_MANIFEST
    mappings_path: Path = DEFAULT_MAPPINGS
    device: str = "cpu"
    host: str = "127.0.0.1"
    port: int = 8000
    verify_checkpoint_sha256: bool = True
    #: Serving V1: ``auto`` | ``rrf3`` | ``sasrec`` (see :data:`SERVING_FUSION_MODES`).
    serving_fusion: str = "auto"
    two_tower_dir: Path | None = DEFAULT_TWOTOWER_DIR
    catalog_path: Path | None = DEFAULT_CATALOG

    @classmethod
    def from_env(cls) -> ServiceSettings:
        """Build settings from ``AGENTRECX_*`` environment variables.

        ``AGENTRECX_MANIFEST_PATH`` unset keeps the accepted run manifest (which cross-checks
        the served checkpoint against the accepted training run).  Setting it to an explicit
        "none" value - ``0``, ``none``, ``off`` or ``false`` - disables the cross-check, which
        is what a deployment serving a **different** checkpoint needs: with the accepted
        manifest still in force, a demo checkpoint would be rejected for disagreeing with a
        training run it was never part of.  Same shape as the existing
        ``AGENTRECX_VERIFY_CHECKPOINT`` switch.
        """
        manifest = os.environ.get("AGENTRECX_MANIFEST_PATH")
        manifest_disabled = (manifest or "").strip().lower() in ("0", "none", "off", "false")
        return cls(
            checkpoint_path=Path(os.environ.get("AGENTRECX_CHECKPOINT_PATH", DEFAULT_CHECKPOINT)),
            manifest_path=(
                DEFAULT_MANIFEST
                if manifest is None
                else (None if manifest_disabled else Path(manifest))
            ),
            mappings_path=Path(os.environ.get("AGENTRECX_MAPPINGS_PATH", DEFAULT_MAPPINGS)),
            device=os.environ.get("AGENTRECX_DEVICE", "cpu"),
            host=os.environ.get("AGENTRECX_HOST", "127.0.0.1"),
            port=int(os.environ.get("AGENTRECX_PORT", "8000")),
            verify_checkpoint_sha256=os.environ.get(
                "AGENTRECX_VERIFY_CHECKPOINT", "1"
            ).lower()
            not in ("0", "false", "no"),
            serving_fusion=os.environ.get("AGENTRECX_SERVING_FUSION", "auto").strip().lower(),
            two_tower_dir=Path(
                os.environ.get("AGENTRECX_TWOTOWER_DIR", DEFAULT_TWOTOWER_DIR)
            ),
            catalog_path=Path(os.environ.get("AGENTRECX_CATALOG_PATH", DEFAULT_CATALOG)),
        )

    def to_inference_config(self) -> InferenceConfig:
        """Translate to the engine's configuration object."""
        return InferenceConfig(
            checkpoint_path=self.checkpoint_path,
            mappings_path=self.mappings_path,
            manifest_path=self.manifest_path,
            device=self.device,
            expected_checkpoint_sha256=(
                ACCEPTED_CHECKPOINT_SHA256 if self.verify_checkpoint_sha256 else None
            ),
        )


@dataclass
class ServiceState:
    """Mutable runtime state: whether the model actually loaded."""

    engine: Any | None = None
    error: str | None = None
    #: Which engine is actually serving, and whether a fallback occurred.  Surfaced by /health so a
    #: degraded deployment is visible over HTTP rather than only in a log line.
    serving: dict[str, Any] | None = None
    loaded_at: float | None = None
    settings: ServiceSettings | None = None

    @property
    def model_loaded(self) -> bool:
        """True when an engine exists and reports itself ready."""
        return self.engine is not None and self.engine.is_ready()


def _build_sequential_engine(settings: ServiceSettings) -> Any:
    """The accepted single-source engine (SASRec), unchanged."""
    return SASRecInferenceEngine(
        settings.to_inference_config(),
        verify_checkpoint_sha256=settings.verify_checkpoint_sha256,
    )


def _build_fusion_engine(settings: ServiceSettings) -> Any:
    """Serving V1: SASRec + Two-Tower + catalogue lexical, fused with the accepted RRF.

    Raises on any missing collaborator.  Nothing here degrades quietly; the caller decides what a
    failure means for the configured mode.
    """
    from recommendation.catalog import MetadataIndex
    from recommendation.backends.tiger_engine import set_shared_engine  # noqa: F401
    from recommendation.inference.fusion import (
        TIGER_SERVING_HEAD,
        FusionInferenceConfig,
        FusionInferenceEngine,
    )
    from recommendation.inference.lexical import CatalogueLexicalIndex
    from recommendation.inference.twotower import (
        TwoTowerInferenceConfig,
        TwoTowerInferenceEngine,
    )

    sequential = _build_sequential_engine(settings)
    if settings.two_tower_dir is None:
        raise RuntimeError("no Two-Tower artifact directory is configured")
    two_tower = TwoTowerInferenceEngine(
        TwoTowerInferenceConfig(
            checkpoint_path=Path(settings.two_tower_dir) / "checkpoint.pt",
            mappings_path=settings.mappings_path,
            device=settings.device,
        )
    )
    if settings.catalog_path is None:
        raise RuntimeError("no catalogue artifact is configured")
    records = MetadataIndex.load(settings.catalog_path).records
    lexical = CatalogueLexicalIndex.build(records)
    # Serving V1 stays the default and is byte-identical: the TIGER profile is only realised when
    # the deployment opts in, and it is a *separate* profile rather than a change to V1.
    # Serving profile selection: the deployment names it through the existing startup configuration
    # (``AGENTRECX_FUSION_PROFILE``).  Absent => ``serving_v1``, byte-identical to before.
    profile_name = (os.environ.get("AGENTRECX_FUSION_PROFILE") or "serving_v1").strip().lower()
    from recommendation.inference.fusion import FUSION_PROFILES as _FUSION_PROFILES

    if profile_name not in _FUSION_PROFILES:
        raise RuntimeError(
            f"unknown AGENTRECX_FUSION_PROFILE {profile_name!r}; known: {sorted(_FUSION_PROFILES)}"
        )
    config = FusionInferenceConfig.for_profile(profile_name)
    tiger_engine = None
    if _tiger_serving_enabled():
        from recommendation.backends.tiger_engine import TigerServingEngine, TigerWorkerConfig

        config = FusionInferenceConfig.for_profile("serving_v2_tiger")
        tiger_engine = TigerServingEngine(
            sequential,  # the deployment's trusted identity map, reused unchanged
            config=TigerWorkerConfig(device=settings.device, k_max=TIGER_SERVING_HEAD),
            blocking_start=False,  # a request must never wait out a minutes-long cold start
        )
        # Load once, in the background: /health reports "loading" until the handshake completes.
        # Registered before the prewarm so the Agent service adopts this very engine rather than
        # building a second worker with its own cold start.
        from recommendation.backends.tiger_engine import set_shared_engine

        set_shared_engine(tiger_engine)
        tiger_engine.prewarm(background=True)
    return FusionInferenceEngine(
        sequential=sequential,
        two_tower=two_tower,
        lexical=lexical,
        catalogue=records,
        item_id_for=sequential.parent_asin_to_item_id,
        is_known_identity=sequential.has_parent_asin,
        config=config,
        tiger=tiger_engine,
    )


def build_engine_report(
    settings: ServiceSettings,
) -> tuple[Any | None, str | None, dict[str, Any]]:
    """Like :func:`build_engine`, but also return what actually happened.

    The third element is the serving report exposed over HTTP, so an operator never has to read a
    log to discover which engine is answering or that a fallback occurred.
    """
    mode = (settings.serving_fusion or "auto").strip().lower()
    report: dict[str, Any] = {
        "requested_engine": mode,
        "active_engine": None,
        "fallback_used": False,
        "fallback_reason": None,
        "fusion_sources": None,
        "rrf_k": None,
        "source_head": None,
    }
    engine, error = build_engine(settings)
    if engine is not None:
        metadata = getattr(engine, "model_metadata", None)
        described = metadata() if callable(metadata) else {}
        report["active_engine"] = described.get("active_engine", "sasrec")
        if report["active_engine"] == "fusion_rrf":
            report["fusion_sources"] = list(described.get("fusion_sources") or [])
            report["rrf_k"] = described.get("rrf_k")
            report["source_head"] = described.get("source_head_depth")
            report["serving_fusion_profile"] = described.get("serving_fusion_profile")
            report["source_head_depths"] = described.get("source_head_depths")
            report["depth_policy"] = described.get("depth_policy")
            # Four states an operator must be able to tell apart: disabled (profile without
            # TIGER), loading, ready, failed - plus the artifact hashes and the retrieval mode.
            tiger_health = described.get("tiger_health")
            if tiger_health is None:
                report["tiger"] = {"state": "disabled", "ready": False}
            else:
                state = str(tiger_health.get("state") or "not_started")
                report["tiger"] = {
                    "state": state,
                    # Only the completed handshake makes the worker ready; a loading engine is
                    # reported as loading, never as ready.
                    "ready": state == "ready" and bool(tiger_health.get("started")),
                    "started": bool(tiger_health.get("started")),
                    "device": tiger_health.get("device"),
                    "checkpoint_sha256": tiger_health.get("checkpoint_sha256"),
                    "sid_artifact_sha256": tiger_health.get("sid_artifact_sha256"),
                    "retrieval_mode": tiger_health.get("retrieval_mode"),
                    "error": tiger_health.get("error"),
                    "assets_missing": list(tiger_health.get("assets_missing") or []),
                }
            report["tiger_in_fusion"] = bool(tiger_health is not None)
        if mode != "sasrec" and report["active_engine"] != "fusion_rrf":
            report["fallback_used"] = True
            report["fallback_reason"] = (
                "Serving V1 was requested but could not be built; SASRec-only is serving"
            )
    else:
        report["fallback_reason"] = error
    return engine, error, report


def release_serving_engine(state: Any) -> dict[str, Any]:
    """Release the serving TIGER worker and clear the shared registry.  Never raises.

    Called from the lifespan's ``finally`` and directly testable, which is why it is a function
    rather than three inline lines: the shutdown path must be *executed* by a test, not read.
    """
    from recommendation.backends.tiger_engine import set_shared_engine

    engine = getattr(state, "engine", None)
    tiger = getattr(engine, "tiger", None) if engine is not None else None
    released = False
    if tiger is not None:
        try:
            tiger.close()
            released = True
        except Exception as exc:  # noqa: BLE001 - shutdown must not raise
            LOGGER.error("releasing the TIGER worker failed: %s", type(exc).__name__)
    set_shared_engine(None)
    return {"tiger_released": released, "registry_cleared": True}


def live_tiger_block(engine: Any, serving: Mapping[str, Any]) -> dict[str, Any] | None:
    """TIGER's lifecycle as of **this request**, for ``/health``.

    Reads ``engine.tiger_health()`` when the serving engine offers it (the V2 fusion engine does)
    and otherwise falls back to the startup report, which is the correct answer for a deployment
    that does not include TIGER at all.  ``ready`` is derived here so it can never disagree with
    the state: only a completed handshake makes the worker ready.
    """
    probe = getattr(engine, "tiger_health", None) if engine is not None else None
    health: Mapping[str, Any] | None = None
    if callable(probe):
        try:
            health = probe()
        except Exception:  # noqa: BLE001 - a broken probe must not break /health
            health = {"state": "failed", "error": "the TIGER health probe raised"}
    if health is None:
        cached = serving.get("tiger")
        return dict(cached) if isinstance(cached, dict) else None
    state = str(health.get("state") or "not_started")
    return {
        "state": state,
        "ready": state == "ready" and bool(health.get("started")),
        "started": bool(health.get("started")),
        "device": health.get("device"),
        "checkpoint_sha256": health.get("checkpoint_sha256"),
        "sid_artifact_sha256": health.get("sid_artifact_sha256"),
        "retrieval_mode": health.get("retrieval_mode"),
        "error": health.get("error"),
        "assets_missing": list(health.get("assets_missing") or []),
    }


def _tiger_serving_enabled() -> bool:
    """Whether this deployment opted TIGER into the serving fusion (off by default)."""
    value = (os.environ.get("AGENTRECX_TIGER") or "").strip().lower()
    return value in {"1", "true", "yes", "on"}


def build_engine(settings: ServiceSettings) -> tuple[Any | None, str | None]:
    """Construct the serving engine, returning ``(engine, error_message)``.

    Never raises: a startup failure is recorded so ``/health`` can report
    ``model_loaded: false`` instead of the process claiming readiness.

    Mode ``sasrec`` builds the accepted single-source engine.  Modes ``auto`` and ``rrf3`` build
    Serving V1; ``rrf3`` refuses to degrade, while ``auto`` falls back to SASRec-only and **logs
    why**, so an operator can always see which engine is actually serving.
    """
    mode = (settings.serving_fusion or "auto").strip().lower()
    if mode not in SERVING_FUSION_MODES:
        return None, (
            f"unknown AGENTRECX_SERVING_FUSION {settings.serving_fusion!r}; "
            f"expected one of {SERVING_FUSION_MODES}"
        )
    if mode == "sasrec":
        try:
            return _build_sequential_engine(settings), None
        except Exception as exc:  # noqa: BLE001 - surface any startup failure as unready
            return None, f"{type(exc).__name__}: {exc}"
    try:
        return _build_fusion_engine(settings), None
    except Exception as exc:  # noqa: BLE001 - classified below, never swallowed
        detail = f"{type(exc).__name__}: {exc}"
        if mode == "rrf3":
            return None, f"AGENTRECX_SERVING_FUSION=rrf3 but Serving V1 could not be built: {detail}"
        LOGGER.warning(
            "Serving V1 unavailable; serving SASRec only. reason=%s", detail
        )
        try:
            return _build_sequential_engine(settings), None
        except Exception as inner:  # noqa: BLE001
            return None, f"{type(inner).__name__}: {inner}"


def title_projection(asins: Sequence[str], catalogue: Any, *, limit: int = 50) -> dict[str, Any]:
    """Read-only product titles for identities a caller already holds.

    Reuses the catalogue the serving engine loaded (no reload, no second identity map).  It never
    resolves identity, never reorders or rescores a recommendation, and never exposes a filesystem
    path: an unknown identity - or a record without a title - maps to ``None`` so the caller can
    fall back to the ASIN.
    """
    ordered: list[str] = []
    for raw in asins:
        value = str(raw).strip()
        if value and value not in ordered:
            ordered.append(value)
    if len(ordered) > limit:
        raise ValueError(f"at most {limit} identities may be looked up at once")
    titles: dict[str, str | None] = {}
    for value in ordered:
        record = None
        getter = getattr(catalogue, "get", None)
        if callable(getter):
            try:
                record = getter(value)
            except Exception:  # noqa: BLE001 - a broken lookup degrades to "no title"
                record = None
        title = getattr(record, "title", None) if record is not None else None
        titles[value] = str(title) if isinstance(title, str) and title.strip() else None
    return {"titles": titles, "count": len(titles),
            "with_title": sum(1 for value in titles.values() if value)}


def create_app(
    settings: ServiceSettings | None = None,
    *,
    engine: SASRecInferenceEngine | None = None,
    load_on_startup: bool = True,
    warmup: bool = True,
    demo: Any = None,
    enable_demo: bool = False,
    demo_factory: Callable[..., Any] | None = None,
) -> FastAPI:
    """Build the FastAPI application.

    ``engine`` may be injected (used by tests with a tiny synthetic checkpoint) so the
    suite never has to load the 349 MB formal run.

    Milestone 11 additions, both opt-in so the Milestone 6 contract is untouched:

    * ``demo`` -- a ready :class:`~recommendation.demo.DemoRuntime` to serve (tests inject
      a lightweight one built over a synthetic checkpoint);
    * ``enable_demo`` -- compose and serve a demo runtime from configuration at startup.
      The composition happens inside the lifespan, **after** the Milestone 6 engine is
      built, so the demo reuses that one engine instead of loading a second checkpoint.
      A composition failure propagates, so the server refuses to start with a clear error
      rather than failing on the first browser request.

    ``enable_demo`` implies nothing about the Milestone 6 endpoints: they behave
    identically whether or not the demo is enabled.
    """
    settings = settings or ServiceSettings.from_env()
    state = ServiceState(settings=settings, engine=engine)
    if engine is not None:
        state.loaded_at = time.time()

    wants_demo = bool(enable_demo or demo is not None)
    #: True only when this application composed the demo runtime itself.  An injected
    #: runtime belongs to the injector (a test or a supervisor), so the lifespan must not
    #: close it -- doing so would close a SQLite store the injector may still own, and
    #: would do it from the lifespan's thread rather than the thread that created it.
    owns_demo = False

    @asynccontextmanager
    async def lifespan(application: FastAPI) -> AsyncIterator[None]:
        nonlocal owns_demo
        if state.engine is None and load_on_startup:
            built, error, serving = build_engine_report(settings)
            state.engine = built
            state.error = error
            state.serving = serving
            state.loaded_at = time.time() if built is not None else None
            if built is not None and warmup:
                # Warm one bounded inference path so the first real request is not
                # paying lazy allocation costs.
                try:
                    built.recommend([built.item_id_to_parent_asin(1)], k=1)
                except Exception:  # noqa: BLE001 - warmup must never block startup
                    pass
        if wants_demo and getattr(application.state, "demo_runtime", None) is None:
            if demo is not None:
                application.state.demo_runtime = demo
            else:
                # Imported lazily so importing this module never pulls in the demo stack.
                from recommendation.demo import DemoRuntimeError, build_demo_runtime

                factory = demo_factory or build_demo_runtime
                try:
                    application.state.demo_runtime = factory(settings=settings, engine=state.engine)
                except DemoRuntimeError:
                    # Fail fast and loudly: a demo server that starts without its
                    # artifacts would fail on the first browser request instead.
                    LOGGER.error("demo runtime could not be composed; refusing to start")
                    raise
                owns_demo = True
            LOGGER.info(
                "demo runtime ready: profiles=%d max_sessions=%d",
                len(application.state.demo_runtime.profiles),
                application.state.demo_runtime.sessions.max_sessions,
            )
        try:
            yield
        finally:
            # The TIGER worker is a child of this process; closing it here (once) is what makes a
            # shutdown explicit rather than a side effect of the parent exiting.
            release_serving_engine(state)
            if owns_demo:
                closer = getattr(application.state.demo_runtime, "close", None)
                if callable(closer):
                    closer()

    app = FastAPI(
        title="AgentRec-X Recommendation Service",
        version="1.0.0",
        summary="SASRec sequential recommendation inference over parent_asin histories.",
        lifespan=lifespan,
    )
    app.state.service = state
    app.state.demo_runtime = None

    # ---- error handlers -------------------------------------------------- #

    @app.exception_handler(EngineRequestValidationError)
    async def _engine_validation_handler(_request: Request, exc: EngineRequestValidationError):
        return JSONResponse(
            status_code=422,
            content=ErrorResponse(error="invalid_request", detail=str(exc)).model_dump(),
        )

    @app.exception_handler(UnknownCatalogueIdentity)
    async def _unknown_catalogue_identity(_request: Request, exc: Exception) -> JSONResponse:
        """A history identity outside the served catalogue is a client error, not a server fault."""
        return JSONResponse(
            status_code=422,
            content=ErrorResponse(error="unknown_item", detail=str(exc)).model_dump(),
        )

    @app.exception_handler(UnknownItemError)
    async def _unknown_item_handler(_request: Request, exc: UnknownItemError):
        return JSONResponse(
            status_code=422,
            content=ErrorResponse(error="unknown_item", detail=str(exc)).model_dump(),
        )

    @app.exception_handler(FastAPIRequestValidationError)
    async def _pydantic_handler(_request: Request, exc: FastAPIRequestValidationError):
        # Collapse pydantic's verbose structure into a stable client-facing shape.
        first = exc.errors()[0] if exc.errors() else {}
        location = ".".join(str(part) for part in first.get("loc", ()) if part != "body")
        detail = first.get("msg", "request validation failed")
        return JSONResponse(
            status_code=422,
            content=ErrorResponse(
                error="invalid_request",
                detail=f"{location}: {detail}" if location else detail,
            ).model_dump(),
        )

    @app.exception_handler(InferenceError)
    async def _inference_error_handler(_request: Request, exc: InferenceError):
        # Internal/model failures: no stack trace, paths or secrets leak to clients.
        return JSONResponse(
            status_code=500,
            content=ErrorResponse(
                error="inference_failed",
                detail="the recommendation engine could not complete this request",
            ).model_dump(),
        )

    # ---- endpoints ------------------------------------------------------- #

    @app.get("/health", response_model=HealthResponse, tags=["service"])
    async def health() -> HealthResponse:
        """Readiness: distinguishes "process running" from "model loaded"."""
        loaded = state.model_loaded
        serving = state.serving or {}
        # TIGER is read from the **live engine** on every request: the startup report is a
        # snapshot taken while the worker was still loading, and a cached block would report
        # "loading" for the rest of the process's life.
        tiger_block = live_tiger_block(state.engine, serving)
        return HealthResponse(
            status="ok" if loaded else "unavailable",
            model_loaded=loaded,
            device=str(state.engine.device) if state.engine is not None else None,
            detail=None if loaded else (state.error or "model not loaded"),
            requested_engine=serving.get("requested_engine"),
            active_engine=serving.get("active_engine"),
            fallback_used=bool(serving.get("fallback_used")),
            fallback_reason=serving.get("fallback_reason") if serving.get("fallback_used") else None,
            fusion_sources=serving.get("fusion_sources"),
            rrf_k=serving.get("rrf_k"),
            tiger=tiger_block,
            source_head=serving.get("source_head"),
        )

    @app.get(
        "/v1/model",
        response_model=ModelInfoResponse,
        # Serving V1 fields are null for a single-source engine; omitting them keeps the accepted
        # response identical and adds the fusion description only when fusion actually serves.
        response_model_exclude_none=True,
        tags=["service"],
    )
    async def model_info() -> ModelInfoResponse:
        """Non-sensitive model metadata, including device and checkpoint identity."""
        if not state.model_loaded:
            raise InferenceError(state.error or "model not loaded")
        assert state.engine is not None
        return ModelInfoResponse(**state.engine.model_metadata())

    @app.get("/v1/catalogue/titles", tags=["catalogue"])
    async def catalogue_titles(asins: str = "") -> Any:
        """Titles for parent_asin values the caller already has (comma separated, max 50).

        A projection, not a search: it adds no candidates and changes no score.  The Fusion view
        uses it to show a product name beside an identity it received from /v1/recommend.
        """
        engine = state.engine
        catalogue = getattr(engine, "catalogue", None) if engine is not None else None
        values = [part.strip() for part in str(asins).split(",") if part.strip()]
        try:
            return title_projection(values, catalogue)
        except ValueError as exc:
            return JSONResponse(
                status_code=422,
                content=ErrorResponse(error="invalid_request", detail=str(exc)).model_dump(),
            )

    @app.post("/v1/recommend", response_model=RecommendResponse, tags=["recommendation"])
    async def recommend(payload: RecommendRequest) -> RecommendResponse:
        """Return deterministic top-k recommendations for a ``parent_asin`` history.

        The returned ``score`` is a raw SASRec model score, not a probability.
        """
        if not state.model_loaded:
            raise InferenceError(state.error or "model not loaded")
        assert state.engine is not None
        result = state.engine.recommend(payload.history, k=payload.k)
        return RecommendResponse(**result.as_dict())

    # ---- Milestone 11 demo API (opt-in) ---------------------------------- #

    if wants_demo:
        from .demo_routes import DemoHTTPError, build_demo_router

        @app.exception_handler(DemoHTTPError)
        async def _demo_error_handler(_request: Request, exc: DemoHTTPError):
            """Render a mapped demo failure with the accepted ``{"error", "detail"}`` body."""
            return JSONResponse(
                status_code=exc.status_code,
                content=ErrorResponse(error=exc.code, detail=exc.detail).model_dump(),
            )

        app.include_router(build_demo_router())

        if WEB_ROOT.is_dir():
            from fastapi.staticfiles import StaticFiles

            # Same-origin static assets: the page calls this same server, so there is no
            # cross-origin surface and no wildcard remote mutation API is exposed.
            app.mount("/demo", StaticFiles(directory=str(WEB_ROOT), html=True), name="demo")

            @app.get("/", include_in_schema=False)
            async def _demo_index() -> RedirectResponse:
                """Convenience entry point: send a browser to the demo page."""
                return RedirectResponse(url="/demo/")

    return app


#: Module-level application for ``uvicorn recommendation.api.app:app``.
#:
#: The demo API is enabled here (and only here) so the served process is the full demo,
#: while ``create_app()`` used directly -- including by the Milestone 6 tests -- stays
#: exactly the Milestone 6 service.
from ..inference.fusion import UnknownCatalogueIdentity  # noqa: E402
app = create_app(enable_demo=True)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point: ``python -m recommendation.api.app``."""
    parser = argparse.ArgumentParser(description="AgentRec-X recommendation service")
    parser.add_argument("--host", default=None, help="bind host (default 127.0.0.1)")
    parser.add_argument("--port", type=int, default=None, help="bind port (default 8000)")
    parser.add_argument("--device", default=None, help="cpu | cuda | cuda:0 (default cpu)")
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--mappings", type=Path, default=None)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--reload", action="store_true", help="uvicorn auto-reload (development)")
    args = parser.parse_args(argv)

    settings = ServiceSettings.from_env()
    if args.host:
        settings.host = args.host
    if args.port:
        settings.port = args.port
    if args.device:
        settings.device = args.device
    if args.checkpoint:
        settings.checkpoint_path = args.checkpoint
    if args.mappings:
        settings.mappings_path = args.mappings
    if args.manifest is not None:
        settings.manifest_path = args.manifest

    # Fail fast on an unusable device rather than starting and serving errors.
    import uvicorn

    from recommendation.inference import resolve_device

    resolve_device(settings.device)
    os.environ["AGENTRECX_DEVICE"] = settings.device

    uvicorn.run(
        "recommendation.api.app:app",
        host=settings.host,
        port=settings.port,
        reload=args.reload,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - manual invocation
    sys.exit(main())

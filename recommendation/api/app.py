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

    @classmethod
    def from_env(cls) -> ServiceSettings:
        """Build settings from ``AGENTRECX_*`` environment variables."""
        manifest = os.environ.get("AGENTRECX_MANIFEST_PATH")
        return cls(
            checkpoint_path=Path(os.environ.get("AGENTRECX_CHECKPOINT_PATH", DEFAULT_CHECKPOINT)),
            manifest_path=Path(manifest) if manifest else DEFAULT_MANIFEST,
            mappings_path=Path(os.environ.get("AGENTRECX_MAPPINGS_PATH", DEFAULT_MAPPINGS)),
            device=os.environ.get("AGENTRECX_DEVICE", "cpu"),
            host=os.environ.get("AGENTRECX_HOST", "127.0.0.1"),
            port=int(os.environ.get("AGENTRECX_PORT", "8000")),
            verify_checkpoint_sha256=os.environ.get(
                "AGENTRECX_VERIFY_CHECKPOINT", "1"
            ).lower()
            not in ("0", "false", "no"),
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

    engine: SASRecInferenceEngine | None = None
    error: str | None = None
    loaded_at: float | None = None
    settings: ServiceSettings | None = None

    @property
    def model_loaded(self) -> bool:
        """True when an engine exists and reports itself ready."""
        return self.engine is not None and self.engine.is_ready()


def build_engine(settings: ServiceSettings) -> tuple[SASRecInferenceEngine | None, str | None]:
    """Construct the inference engine, returning ``(engine, error_message)``.

    Never raises: a startup failure is recorded so ``/health`` can report
    ``model_loaded: false`` instead of the process claiming readiness.
    """
    try:
        engine = SASRecInferenceEngine(
            settings.to_inference_config(),
            verify_checkpoint_sha256=settings.verify_checkpoint_sha256,
        )
    except Exception as exc:  # noqa: BLE001 - surface any startup failure as unready
        return None, f"{type(exc).__name__}: {exc}"
    return engine, None


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
            built, error = build_engine(settings)
            state.engine = built
            state.error = error
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
        return HealthResponse(
            status="ok" if loaded else "unavailable",
            model_loaded=loaded,
            device=str(state.engine.device) if state.engine is not None else None,
            detail=None if loaded else (state.error or "model not loaded"),
        )

    @app.get("/v1/model", response_model=ModelInfoResponse, tags=["service"])
    async def model_info() -> ModelInfoResponse:
        """Non-sensitive model metadata, including device and checkpoint identity."""
        if not state.model_loaded:
            raise InferenceError(state.error or "model not loaded")
        assert state.engine is not None
        return ModelInfoResponse(**state.engine.model_metadata())

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

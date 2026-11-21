"""Optional OpenTelemetry tracing for the AgentRec-X Web Demo.

Design boundaries
-----------------
* The SQLite request ledger remains the authority for business state.  Tracing is an
  observational side channel: no span, exporter, or collector failure may change a business
  result, retry decision, or MCP call.
* The official OpenTelemetry SDK owns trace IDs, span lifecycle, W3C propagation, batching,
  OTLP export and shutdown.  This module only configures those primitives and records a small
  set of allow-listed business attributes.
* MCP's official Python SDK ships ``OpenTelemetryMiddleware`` on the server side of a tool
  call.  The BFF injects standard W3C context into the MCP request ``_meta``; the child
  extracts it and emits a real server span.  No synthetic parent/child relationship is
  fabricated.
* Tracing is disabled unless ``AGENTRECX_OTEL_ENABLED`` is truthy.  If the SDK or exporter is
  missing, the app logs a setup warning and continues with tracing disabled.

Privacy
-------
Only allow-listed scalar attributes are recorded.  User messages, query text, preference
values, product titles/descriptions, tool arguments/responses and exception messages are never
recorded by this module.  Tool spans may record argument *keys* (not values) and count fields.
"""

from __future__ import annotations

import logging
import os
import sys
import threading
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Iterator, Mapping

LOGGER = logging.getLogger("agentrecx.observability")

# These are the stable OpenInference span-kind values.  They are duplicated as literals so the
# disabled path and unit tests do not need the optional openinference package installed.
OPENINFERENCE_AGENT = "AGENT"
OPENINFERENCE_CHAIN = "CHAIN"
OPENINFERENCE_RETRIEVER = "RETRIEVER"
OPENINFERENCE_TOOL = "TOOL"

_SPAN_KIND = "openinference.span.kind"

_ATTR_MAX_STRING = 512
_ATTR_MAX_SEQUENCE = 32

_ACTIVE_LOCK = threading.Lock()
_ACTIVE: "Tracing | None" = None


def _truthy(value: str | None) -> bool:
    return str(value or "").strip().casefold() in {"1", "true", "yes", "on"}


def _number(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _safe_value(value: Any) -> Any:
    """Return a scalar/sequence OTel accepts, or ``None`` to omit it.

    This is intentionally an allow-list rather than a redaction pass: callers cannot
    accidentally export a message, dictionary, exception or arbitrary object.
    """
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value
    if isinstance(value, str):
        return value[:_ATTR_MAX_STRING]
    if isinstance(value, (tuple, list)):
        items = [item for item in (_safe_value(v) for v in value[:_ATTR_MAX_SEQUENCE]) if item is not None]
        if all(isinstance(item, (bool, int, float, str)) for item in items):
            return items
        return None
    # OTel spans accept scalars and homogeneous scalar sequences, not arbitrary mappings.
    # Structured tool payloads are deliberately not eligible for export.
    return None


def clean_attributes(attributes: Mapping[str, Any] | None) -> dict[str, Any]:
    """Filter attributes down to the exporter-safe, privacy-safe allow-list."""
    cleaned: dict[str, Any] = {}
    for key, value in (attributes or {}).items():
        if not isinstance(key, str) or not key:
            continue
        safe = _safe_value(value)
        if safe is not None:
            cleaned[key] = safe
    return cleaned


class _NoopSpan:
    """Small no-op span used when tracing is disabled or not installed."""

    def set_attribute(self, key: str, value: Any) -> None:  # noqa: ARG002
        return None

    def set_attributes(self, attributes: Mapping[str, Any] | None) -> None:  # noqa: ARG002
        return None

    def add_event(self, name: str, attributes: Mapping[str, Any] | None = None) -> None:  # noqa: ARG002
        return None

    def set_status(self, status: Any) -> None:  # noqa: ARG002
        return None

    def record_exception(self, exception: BaseException) -> None:  # noqa: ARG002
        return None

    def is_recording(self) -> bool:
        return False


class Span:
    """Thin, exception-isolated facade over an OpenTelemetry span."""

    def __init__(self, span: Any) -> None:
        self._span = span

    def set_attribute(self, key: str, value: Any) -> None:
        try:
            safe = _safe_value(value)
            if safe is not None:
                self._span.set_attribute(key, safe)
        except Exception:  # noqa: BLE001 - telemetry never affects business flow
            pass

    def set_attributes(self, attributes: Mapping[str, Any] | None) -> None:
        try:
            self._span.set_attributes(clean_attributes(attributes))
        except Exception:  # noqa: BLE001
            pass

    def add_event(self, name: str, attributes: Mapping[str, Any] | None = None) -> None:
        try:
            self._span.add_event(name, clean_attributes(attributes))
        except Exception:  # noqa: BLE001
            pass

    def set_status(self, status: Any) -> None:
        try:
            self._span.set_status(status)
        except Exception:  # noqa: BLE001
            pass

    def is_recording(self) -> bool:
        try:
            return bool(self._span.is_recording())
        except Exception:  # noqa: BLE001
            return False


@dataclass
class Tracing:
    """One process-local tracing runtime (provider, tracer and exporter status)."""

    enabled: bool = False
    tracer: Any | None = None
    provider: Any | None = None
    project_name: str = "agentrecx-x"
    service_name: str = "agentrecx-web-demo"
    endpoint: str = ""
    setup_error: str = ""
    _instrumented_apps: list[Any] = field(default_factory=list, repr=False)

    # -- span creation ---------------------------------------------------- #
    @contextmanager
    def span(
        self,
        name: str,
        *,
        attributes: Mapping[str, Any] | None = None,
        openinference_kind: str | None = None,
        kind: Any = None,
    ) -> Iterator[Span | _NoopSpan]:
        """Start a child span and make it current for the duration of the block.

        Exceptions are marked on the span as an error *type only* and are always re-raised.
        The OpenTelemetry context manager itself never suppresses a business exception.
        """
        if not self.enabled or self.tracer is None:
            yield _NoopSpan()
            return
        attrs = clean_attributes(attributes)
        if openinference_kind:
            attrs[_SPAN_KIND] = openinference_kind
        try:
            from opentelemetry.trace import SpanKind  # noqa: PLC0415

            span_kind = kind if kind is not None else SpanKind.INTERNAL
            cm = self.tracer.start_as_current_span(
                name,
                kind=span_kind,
                attributes=attrs,
                record_exception=False,
                set_status_on_exception=False,
            )
        except Exception as exc:  # noqa: BLE001 - configuration/export never breaks business
            LOGGER.debug("could not start span %s: %s", name, type(exc).__name__)
            yield _NoopSpan()
            return
        try:
            raw_span = cm.__enter__()
        except Exception as exc:  # noqa: BLE001 - telemetry setup never blocks the body
            LOGGER.debug("span %s could not start: %s", name, type(exc).__name__)
            yield _NoopSpan()
            return

        facade = Span(raw_span)
        exc_info: tuple[Any, Any, Any] = (None, None, None)
        try:
            yield facade
        except BaseException as exc:
            # Record only the exception class.  Messages can contain user text or
            # credentials and are deliberately excluded.
            exc_info = sys.exc_info()
            facade.set_attribute("error.type", type(exc).__name__)
            try:
                from opentelemetry.trace import Status, StatusCode  # noqa: PLC0415

                facade.set_status(Status(StatusCode.ERROR, type(exc).__name__))
            except Exception:  # noqa: BLE001
                pass
            raise
        finally:
            # The body's exception (if any) has already been re-raised by this point.  A
            # failing exporter/processor exit is contained so it can never replace it.
            try:
                cm.__exit__(*exc_info)
            except Exception as exc:  # noqa: BLE001
                LOGGER.debug("span %s lifecycle failed: %s", name, type(exc).__name__)

    def annotate_current(self, attributes: Mapping[str, Any]) -> None:
        """Attach safe attributes to the active span (for example the FastAPI request span)."""
        if not self.enabled:
            return
        try:
            from opentelemetry import trace  # noqa: PLC0415

            span = trace.get_current_span()
            if span is not None and span.is_recording():
                span.set_attributes(clean_attributes(attributes))
        except Exception:  # noqa: BLE001
            pass

    def inject(self, carrier: dict[str, Any]) -> None:
        """Inject the current W3C trace context into an MCP request ``_meta`` carrier."""
        if not self.enabled:
            return
        try:
            from opentelemetry.propagate import inject  # noqa: PLC0415

            inject(carrier)
        except Exception as exc:  # noqa: BLE001 - a lost parent link is observational only
            LOGGER.debug("trace context injection failed: %s", type(exc).__name__)

    def current_trace_id(self) -> str | None:
        if not self.enabled:
            return None
        try:
            from opentelemetry import trace  # noqa: PLC0415

            context = trace.get_current_span().get_span_context()
            if context.is_valid:
                return format(context.trace_id, "032x")
        except Exception:  # noqa: BLE001
            pass
        return None

    # -- FastAPI / lifecycle ---------------------------------------------- #
    def server_request_hook(self, span: Any, scope: dict[str, Any]) -> None:
        """FastAPI hook: add safe request identity without reading bodies or headers."""
        if span is None:
            return
        try:
            state = scope.setdefault("state", {})
            request_id = state.get("agentrecx_request_id")
            if not request_id:
                request_id = uuid.uuid4().hex
                state["agentrecx_request_id"] = request_id
            span.set_attribute("agentrecx.http_request_id", request_id)
            path = scope.get("path")
            if isinstance(path, str):
                span.set_attribute("agentrecx.request_kind", path)
        except Exception:  # noqa: BLE001
            pass

    def instrument_fastapi(self, app: Any) -> None:
        """Install the official OpenTelemetry FastAPI/ASGI instrumentation."""
        if not self.enabled or self.provider is None:
            return
        try:
            from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor  # noqa: PLC0415

            FastAPIInstrumentor.instrument_app(
                app,
                tracer_provider=self.provider,
                server_request_hook=self.server_request_hook,
                # The demo has no static routes.  Health polling is excluded because it
                # would otherwise dominate every trace project without adding business
                # information; startup readiness is still visible in the BFF logs.
                excluded_urls=r"api/health",
                # The ASGI receive/send spans add no business information here; the server
                # span already carries method, route, status and duration.
                exclude_spans=["receive", "send"],
            )
            self._instrumented_apps.append(app)
        except Exception as exc:  # noqa: BLE001 - tracing is optional
            LOGGER.warning(
                "FastAPI instrumentation unavailable; continuing without request spans: %s",
                type(exc).__name__,
            )

    def shutdown(self) -> None:
        """Flush/stop the provider.  Any exporter failure is logged and contained."""
        if not self.enabled:
            return
        try:
            from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor  # noqa: PLC0415

            for app in self._instrumented_apps:
                try:
                    FastAPIInstrumentor.uninstrument_app(app)
                except Exception:  # noqa: BLE001
                    pass
        except Exception:  # noqa: BLE001
            pass
        self._instrumented_apps.clear()
        provider = self.provider
        if provider is None:
            return
        try:
            provider.force_flush(timeout_millis=2000)
        except Exception as exc:  # noqa: BLE001 - exporter problems are not request failures
            LOGGER.warning("trace flush failed: %s", type(exc).__name__)
        try:
            provider.shutdown()
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("trace provider shutdown failed: %s", type(exc).__name__)
        self.enabled = False
        self.tracer = None
        self.provider = None

    def health(self) -> dict[str, Any]:
        """Non-sensitive status for diagnostics; never exposes the collector URL or headers."""
        return {
            "enabled": bool(self.enabled),
            "project": self.project_name,
            "service": self.service_name,
            "exporter": "otlp-http" if self.enabled else None,
            "error": self.setup_error or None,
        }


def _normalise_endpoint(endpoint: str) -> str:
    text = str(endpoint or "").strip()
    if not text:
        return "http://127.0.0.1:6006/v1/traces"
    if text.rstrip("/").endswith("/v1/traces"):
        return text.rstrip("/")
    return text.rstrip("/") + "/v1/traces"


def _env_endpoint(env: Mapping[str, str]) -> str:
    return (
        str(env.get("AGENTRECX_OTEL_ENDPOINT") or "").strip()
        or str(env.get("PHOENIX_COLLECTOR_ENDPOINT") or "").strip()
        or str(env.get("OTEL_EXPORTER_OTLP_ENDPOINT") or "").strip()
        or "http://127.0.0.1:6006/v1/traces"
    )


def configure_tracing(
    *,
    enabled: bool = False,
    endpoint: str = "http://127.0.0.1:6006/v1/traces",
    project_name: str = "agentrecx-x",
    service_name: str = "agentrecx-web-demo",
    sample_ratio: float = 1.0,
    export_timeout_s: float = 2.0,
    span_exporter: Any | None = None,
    use_batch: bool = True,
    set_global_provider: bool = False,
) -> Tracing:
    """Configure a process-local tracer provider.

    ``span_exporter`` and ``use_batch=False`` exist for deterministic tests; the production
    path uses an OTLP/HTTP batch exporter with a bounded export timeout.
    """
    global _ACTIVE
    if not enabled:
        tracing = Tracing(enabled=False, project_name=project_name, service_name=service_name)
        set_active_tracing(tracing)
        return tracing

    try:
        from opentelemetry import trace  # noqa: PLC0415
        from opentelemetry.sdk.resources import Resource  # noqa: PLC0415
        from opentelemetry.sdk.trace import TracerProvider  # noqa: PLC0415
        from opentelemetry.sdk.trace.export import (  # noqa: PLC0415
            BatchSpanProcessor,
            SimpleSpanProcessor,
        )
        from opentelemetry.sdk.trace.sampling import (  # noqa: PLC0415
            ALWAYS_ON,
            ParentBased,
            TraceIdRatioBased,
        )
        from openinference.semconv.resource import ResourceAttributes  # noqa: PLC0415
    except Exception as exc:  # noqa: BLE001 - optional dependency is genuinely optional
        error = f"{type(exc).__name__}: tracing dependency unavailable"
        LOGGER.warning("%s; tracing stays disabled", error)
        tracing = Tracing(
            enabled=False, project_name=project_name, service_name=service_name,
            setup_error=error,
        )
        set_active_tracing(tracing)
        return tracing

    try:
        ratio = max(0.0, min(1.0, float(sample_ratio)))
    except (TypeError, ValueError):
        ratio = 1.0
    sampler = ParentBased(ALWAYS_ON if ratio >= 1.0 else TraceIdRatioBased(ratio))
    resource = Resource.create({
        "service.name": str(service_name or "agentrecx-web-demo"),
        "service.version": "m3-a",
        "deployment.environment.name": "local-demo",
        ResourceAttributes.PROJECT_NAME: str(project_name or "agentrecx-x"),
    })
    provider = TracerProvider(resource=resource, sampler=sampler)
    try:
        if span_exporter is None:
            from opentelemetry.exporter.otlp.proto.http.trace_exporter import (  # noqa: PLC0415
                OTLPSpanExporter,
            )

            span_exporter = OTLPSpanExporter(
                endpoint=_normalise_endpoint(endpoint),
                timeout=max(0.1, float(export_timeout_s)),
            )
        if use_batch:
            processor = BatchSpanProcessor(
                span_exporter,
                max_queue_size=2048,
                schedule_delay_millis=500,
                max_export_batch_size=256,
                export_timeout_millis=max(1, int(float(export_timeout_s) * 1000)),
            )
        else:
            processor = SimpleSpanProcessor(span_exporter)
        provider.add_span_processor(processor)
        if set_global_provider:
            try:
                trace.set_tracer_provider(provider)
            except Exception as exc:  # noqa: BLE001 - already configured by host process
                LOGGER.warning(
                    "global tracer provider already configured (%s); MCP server spans may use it",
                    type(exc).__name__,
                )
        tracing = Tracing(
            enabled=True,
            tracer=provider.get_tracer("agentrecx", "m3-a"),
            provider=provider,
            project_name=str(project_name or "agentrecx-x"),
            service_name=str(service_name or "agentrecx-web-demo"),
            endpoint=_normalise_endpoint(endpoint),
        )
        set_active_tracing(tracing)
        LOGGER.info(
            "OpenTelemetry tracing enabled project=%s service=%s exporter=otlp-http",
            tracing.project_name, tracing.service_name,
        )
        return tracing
    except Exception as exc:  # noqa: BLE001 - never prevent the Web demo from starting
        error = f"{type(exc).__name__}: tracing setup failed"
        LOGGER.warning("%s; tracing stays disabled", error)
        tracing = Tracing(
            enabled=False, project_name=project_name, service_name=service_name,
            setup_error=error,
        )
        set_active_tracing(tracing)
        return tracing


def configure_tracing_from_env(
    env: Mapping[str, str] | None = None,
    *,
    service_name: str | None = None,
    set_global_provider: bool = False,
) -> Tracing:
    """Read the small allow-listed tracing configuration from the environment."""
    values = os.environ if env is None else env
    enabled = _truthy(values.get("AGENTRECX_OTEL_ENABLED"))
    endpoint = _env_endpoint(values)
    project = str(values.get("AGENTRECX_OTEL_PROJECT") or "agentrecx-x").strip() or "agentrecx-x"
    service = str(
        values.get("AGENTRECX_OTEL_SERVICE_NAME")
        or service_name
        or "agentrecx-web-demo"
    ).strip() or "agentrecx-web-demo"
    ratio = _number(values.get("AGENTRECX_OTEL_SAMPLE_RATIO"))
    timeout = _number(values.get("AGENTRECX_OTEL_EXPORT_TIMEOUT_S"))
    return configure_tracing(
        enabled=enabled,
        endpoint=endpoint,
        project_name=project,
        service_name=service,
        sample_ratio=1.0 if ratio is None else ratio,
        export_timeout_s=2.0 if timeout is None else timeout,
        set_global_provider=set_global_provider,
    )


def set_active_tracing(tracing: Tracing) -> None:
    global _ACTIVE
    with _ACTIVE_LOCK:
        _ACTIVE = tracing


def get_tracing() -> Tracing:
    """The process-local runtime used by core modules that are not given one explicitly."""
    runtime = _ACTIVE
    if runtime is not None:
        return runtime
    return Tracing(enabled=False)


def annotate_current_span(attributes: Mapping[str, Any]) -> None:
    """Attach safe attributes to the active span without requiring a tracing handle."""
    try:
        from opentelemetry import trace  # noqa: PLC0415

        span = trace.get_current_span()
        if span is not None and span.is_recording():
            span.set_attributes(clean_attributes(attributes))
    except Exception:  # noqa: BLE001
        pass


def current_trace_id() -> str | None:
    """Current W3C trace id, or ``None`` when tracing is disabled/unavailable."""
    try:
        from opentelemetry import trace  # noqa: PLC0415

        context = trace.get_current_span().get_span_context()
        if context.is_valid:
            return format(context.trace_id, "032x")
    except Exception:  # noqa: BLE001
        pass
    return None


# A disabled runtime is still usable in all call sites, which keeps business code free of
# ``if tracing`` branches.  ``configure_tracing`` replaces the module-level active runtime.
CURRENT_SPAN_KIND = _SPAN_KIND

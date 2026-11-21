"""AgentRec-X observability adapters.

The public implementation lives in :mod:`observability.tracing`; this package deliberately
contains no business logic.  OpenTelemetry is optional at runtime: importing this package must
remain cheap and must work when the tracing SDK is not installed.
"""

from observability.tracing import (
    OPENINFERENCE_AGENT,
    OPENINFERENCE_CHAIN,
    OPENINFERENCE_RETRIEVER,
    OPENINFERENCE_TOOL,
    Tracing,
    annotate_current_span,
    configure_tracing,
    configure_tracing_from_env,
    current_trace_id,
    get_tracing,
    set_active_tracing,
)

__all__ = [
    "OPENINFERENCE_AGENT",
    "OPENINFERENCE_CHAIN",
    "OPENINFERENCE_RETRIEVER",
    "OPENINFERENCE_TOOL",
    "Tracing",
    "annotate_current_span",
    "configure_tracing",
    "configure_tracing_from_env",
    "current_trace_id",
    "get_tracing",
    "set_active_tracing",
]

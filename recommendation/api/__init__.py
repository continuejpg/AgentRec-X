"""HTTP API for the AgentRec-X recommendation service (Milestone 6).

A thin FastAPI adapter over :mod:`recommendation.inference`.  The API layer owns
request validation and error mapping only; all model, mapping, encoding, masking and
ranking logic lives in the inference package and is shared with future non-HTTP
callers (for example an Agent recommendation tool).

This package imports FastAPI, so it is a runtime dependency of the service rather
than of the core library.  Install ``requirements.txt`` to serve; ``requirements-dev.txt``
adds the test client dependencies.
"""

from __future__ import annotations

__all__ = ["create_app", "ServiceSettings"]

__version__ = "1.0.0"


def __getattr__(name: str):
    """Import the app lazily so ``import recommendation.api`` stays cheap."""
    if name in ("create_app", "ServiceSettings"):
        from .app import ServiceSettings, create_app

        return {"create_app": create_app, "ServiceSettings": ServiceSettings}[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

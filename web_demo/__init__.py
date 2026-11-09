"""AgentRec-X Web Demo BFF (:mod:`web_demo`).

A local FastAPI BFF that owns two long-lived official MCP stdio clients (one per fixed demo
profile) and exposes the HTTP + SSE contract in ``docs/WEB_DEMO_CONTRACT.md``.

Run: ``.venv/bin/python -m web_demo`` (host 127.0.0.1, port 8787 by default).
"""

__all__ = ["__version__"]

__version__ = "1.0.0"

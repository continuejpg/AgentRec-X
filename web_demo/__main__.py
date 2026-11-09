"""Entry point: ``.venv/bin/python -m web_demo``.

Binds 127.0.0.1 only (local demo).  Environment variables are documented in
:mod:`web_demo.config`.
"""

from __future__ import annotations

import logging
import sys

import uvicorn

from web_demo.app import create_app
from web_demo.config import load_settings


def main() -> int:
    logging.basicConfig(
        stream=sys.stderr,
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    settings = load_settings()
    app = create_app(settings)
    logging.getLogger("agentrecx.web_demo.bff").info(
        "listening http://%s:%s (search_mode=%s, memory_writes=%s)",
        settings.host,
        settings.port,
        settings.search_mode,
        settings.allow_memory_writes,
    )
    uvicorn.run(app, host=settings.host, port=settings.port, log_level="info")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

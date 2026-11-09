"""Configuration for the AgentRec-X Web Demo BFF.

Everything the BFF needs to know about paths, fixed demo identities and environment
overrides lives here.  No behaviour lives in this module.

Environment variables (all optional):
    AGENTRECX_WEB_DEMO_PORT                  BFF port, default 8787 (host is always 127.0.0.1)
    AGENTRECX_WEB_DEMO_SEARCH_MODE           "bge" (default) or "bm25" (dev/tests only)
    AGENTRECX_WEB_DEMO_ALLOW_MEMORY_WRITES   "1" enables writes for the 0057 profile; default 0
    AGENTRECX_WEB_DEMO_MCP_TIMEOUT_S         per-tool-call timeout in seconds, default 180
    AGENTRECX_WEB_DEMO_CONSOLE_ONLY_LOG      "1" lets the one-command launcher own bff.log via tee
                                             (the launcher sets it; a direct BFF run keeps the
                                             file handler and writes bff.log itself)
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

REPO = Path(__file__).resolve().parents[1]
RUNS_DIR = REPO / "runs/web_demo"
LOG_DIR = RUNS_DIR / "logs"
REPLAY_PATH = REPO / "runs/p18/qwen_replay.json"
MEMORY_FIXTURE = REPO / "data/p14e2/benchmark/dev.agent.jsonl"
MCP_SERVER_MODULE = "recommendation.mcp.server"

PROTOCOL_TARGET = "2026-07-28"
SEARCH_TOP_K = 10
COMPARE_MIN = 2
COMPARE_MAX = 5
DEFAULT_MCP_TIMEOUT_S = 180.0
DEFAULT_CLARIFICATION_TTL_S = 1800.0
DEFAULT_PORT = 8787

# Frozen P18/P17 catalogue snapshot size (76,263 Electronics products).  Health reports this
# verified value until a real tool result observed on this process returns catalogue_size.
CATALOGUE_SIZE = 76263

# Verified real comparison ids used by the P18 demo (both are inventory ids known to the
# catalogue and to the P16 comparison scenarios).
VERIFIED_COMPARISON_IDS = ("B004K6R43G", "B00008VF63")

NOTICES = [
    "Local CPU demo; catalogue facts only; no live price/stock/ratings.",
    "Memory writes for Demo B are local process memory and reset when the BFF restarts.",
    "Qwen Agent mode is historical replay only; there is no live Qwen call.",
    "Catalogue evidence does not prove real-world quality or user satisfaction.",
]


@dataclass(frozen=True)
class ProfileConfig:
    """One fixed demo identity.  The browser only ever receives ``profile_id``."""

    profile_id: str
    user_key: str
    label: str
    description: str
    memory_writes: bool


PROFILES: tuple[ProfileConfig, ...] = (
    ProfileConfig(
        profile_id="profile_0049",
        user_key="p14e2_lite_0049",
        label="Demo A · Read-only",
        description=(
            "Read-only demo profile (p14e2_lite_0049).  Real catalogue search, recommendation "
            "Gate, evidence and comparison; memory writes are disabled on this MCP server."
        ),
        memory_writes=False,
    ),
    ProfileConfig(
        profile_id="profile_0057",
        user_key="p14e2_lite_0057",
        label="Demo B · Memory write",
        description=(
            "Memory-write demo profile (p14e2_lite_0057).  Clarification replies may persist "
            "real M9 preference memory, but only when the BFF was launched with "
            "AGENTRECX_WEB_DEMO_ALLOW_MEMORY_WRITES=1."
        ),
        memory_writes=True,
    ),
)
PROFILE_BY_ID: dict[str, ProfileConfig] = {profile.profile_id: profile for profile in PROFILES}
DEFAULT_PROFILE_ID = "profile_0049"


def _env_bool(value: str | None) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    host: str = "127.0.0.1"
    port: int = DEFAULT_PORT
    search_mode: str = "bge"
    allow_memory_writes: bool = False
    mcp_timeout_s: float = DEFAULT_MCP_TIMEOUT_S
    clarification_ttl_s: float = DEFAULT_CLARIFICATION_TTL_S
    console_only_log: bool = False
    repo: Path = REPO
    log_dir: Path = LOG_DIR
    replay_path: Path = REPLAY_PATH
    memory_fixture: Path = MEMORY_FIXTURE
    python_executable: str = field(default_factory=lambda: sys.executable)

    def writes_enabled_for(self, profile_id: str) -> bool:
        profile = PROFILE_BY_ID[profile_id]
        return bool(profile.memory_writes and self.allow_memory_writes)

    def profile_public(self, profile_id: str) -> dict[str, Any]:
        profile = PROFILE_BY_ID[profile_id]
        return {
            "profile_id": profile.profile_id,
            "label": profile.label,
            "description": profile.description,
            "user_key": profile.user_key,
            "memory_writes": self.writes_enabled_for(profile.profile_id),
            "search_mode": self.search_mode,
        }


def load_settings(env: Mapping[str, str] | None = None) -> Settings:
    want = os.environ if env is None else env
    port_raw = str(want.get("AGENTRECX_WEB_DEMO_PORT", DEFAULT_PORT)).strip()
    try:
        port = int(port_raw)
    except ValueError as exc:  # surface a real config error instead of silently defaulting
        raise ValueError(f"AGENTRECX_WEB_DEMO_PORT must be an integer, got {port_raw!r}") from exc
    if not 1 <= port <= 65535:
        raise ValueError(f"AGENTRECX_WEB_DEMO_PORT out of range: {port}")
    mode = str(want.get("AGENTRECX_WEB_DEMO_SEARCH_MODE", "bge")).strip().lower() or "bge"
    if mode not in {"bge", "bm25"}:
        raise ValueError(f"AGENTRECX_WEB_DEMO_SEARCH_MODE must be 'bge' or 'bm25', got {mode!r}")
    timeout_raw = str(want.get("AGENTRECX_WEB_DEMO_MCP_TIMEOUT_S", DEFAULT_MCP_TIMEOUT_S)).strip()
    try:
        timeout = float(timeout_raw)
    except ValueError as exc:
        raise ValueError(
            f"AGENTRECX_WEB_DEMO_MCP_TIMEOUT_S must be a number, got {timeout_raw!r}"
        ) from exc
    if timeout <= 0:
        raise ValueError("AGENTRECX_WEB_DEMO_MCP_TIMEOUT_S must be positive")
    return Settings(
        port=port,
        search_mode=mode,
        allow_memory_writes=_env_bool(want.get("AGENTRECX_WEB_DEMO_ALLOW_MEMORY_WRITES")),
        mcp_timeout_s=timeout,
        console_only_log=_env_bool(want.get("AGENTRECX_WEB_DEMO_CONSOLE_ONLY_LOG")),
    )


def modes_public(*, cpu_live_available: bool) -> dict[str, Any]:
    """The three run modes exposed to the UI.  CPU live is the only live mode."""
    return {
        "cpu_live": {
            "available": bool(cpu_live_available),
            "label": "CPU Live Mode",
            "detail": (
                "BGE dense retrieval + BM25 fallback + Rule Planner + V0 Scorer + "
                "P15/P16 + P17 MCP"
            ),
        },
        "qwen_replay": {
            "available": True,
            "label": "Qwen Replay Mode",
            "badge": "Historical Replay — Not a Live Model Call",
        },
        "qwen_live": {
            "available": False,
            "label": "Qwen Live Mode",
            "disabled_reason": "No live Qwen service detected (P18)",
        },
    }


def limits_public(settings: Settings) -> dict[str, Any]:
    return {
        "search_top_k": SEARCH_TOP_K,
        "compare_min": COMPARE_MIN,
        "compare_max": COMPARE_MAX,
        "clarification_ttl_s": settings.clarification_ttl_s,
    }

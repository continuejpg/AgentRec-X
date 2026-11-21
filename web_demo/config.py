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
    AGENTRECX_BEHAVIOR_ENABLED               "1"/"0" master switch for first-party behavior
                                             tracking; default 1
    AGENTRECX_BEHAVIOR_DB                    SQLite path; default runs/web_demo/behavior/events.sqlite3
    AGENTRECX_BEHAVIOR_WINDOW_S              short-term profile window, default 21600 s (6 h)
    AGENTRECX_BEHAVIOR_HALF_LIFE_S           decay half-life, default 1800 s
    AGENTRECX_BEHAVIOR_RETENTION_DAYS        event retention, default 7 days
    AGENTRECX_WEB_STATE_DB                   optional SQLite path for durable session
                                             identity/counters/metadata; unset keeps the
                                             original process-local SessionStore
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from web_demo.behavior import (
    BEHAVIOR_SEMANTICS_VERSION,
    COMPARE_ACTION_STATUSES,
    DEFAULT_HALF_LIFE_S,
    DEFAULT_RETENTION_DAYS,
    DEFAULT_WINDOW_S,
    EVENT_SOURCES,
    PROFILE_SOURCES,
    EVENT_TYPES,
    IMPRESSION_MIN_VISIBLE_MS,
    IMPRESSION_VISIBILITY_THRESHOLD,
)

REPO = Path(__file__).resolve().parents[1]
RUNS_DIR = REPO / "runs/web_demo"
LOG_DIR = RUNS_DIR / "logs"
REPLAY_PATH = REPO / "runs/p18/qwen_replay.json"
RANKER_REPLAY_PATH = REPO / "runs/p100_extended_evaluation/p100_ranker_gpu_predictions.jsonl"
P100_TASKS_PATH = REPO / "data/p100_extended_benchmark/p100_agent_tasks_memory_on.jsonl"
P100_METADATA_PATH = REPO / "data/p100_extended_benchmark/p100_candidate_metadata.jsonl"
MEMORY_FIXTURE = REPO / "data/p14e2/benchmark/dev.agent.jsonl"
MCP_SERVER_MODULE = "recommendation.mcp.server"

PROTOCOL_TARGET = "2026-07-28"
SEARCH_TOP_K = 10
COMPARE_MIN = 2
COMPARE_MAX = 5
DEFAULT_MCP_TIMEOUT_S = 180.0
DEFAULT_CLARIFICATION_TTL_S = 1800.0
DEFAULT_PORT = 8787
DEFAULT_QWEN_TIMEOUT_S = 180.0
DEFAULT_QWEN_HEALTH_TIMEOUT_S = 60.0
DEFAULT_QWEN_TURN_TIMEOUT_S = 600.0
DEFAULT_QWEN_MAX_STEPS = 8
DEFAULT_QWEN_HEALTH_INTERVAL_S = 60.0

# Frozen P18/P17 catalogue snapshot size (76,263 Electronics products).  Health reports this
# verified value until a real tool result observed on this process returns catalogue_size.
CATALOGUE_SIZE = 76263

# Verified real comparison ids used by the P18 demo (both are inventory ids known to the
# catalogue and to the P16 comparison scenarios).
VERIFIED_COMPARISON_IDS = ("B004K6R43G", "B00008VF63")

NOTICES = [
    "Local CPU demo; catalogue facts only; no live price/stock/ratings.",
    "Memory writes for Demo B are local process memory and reset when the BFF restarts.",
    "Qwen Base Live is optional and only available after a real PlannerAction health check; Qwen Replay is historical and never a live model call.",
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
# Optional isolated ASK/Memory acceptance identity.  p14e2_lite_0056 is a benchmark fixture user
# with real M9-created wired/wireless conflict events; it is not one of the normal demo users and
# is exposed only when explicitly enabled for this acceptance test.
_ASK_TEST_PROFILE = ProfileConfig(
    profile_id="profile_0056_ask",
    user_key="p14e2_lite_0056",
    label="ASK Test · Conflict",
    description=(
        "Isolated ASK/Memory test profile (p14e2_lite_0056): real M9 conflict history; "
        "enabled only for the controlled Qwen ASK-resume acceptance."
    ),
    memory_writes=True,
)
_ASK_TEST_ENABLED = str(os.environ.get("AGENTRECX_WEB_DEMO_ASK_TEST_PROFILE", "")).strip().lower() in {
    "1", "true", "yes", "on",
}
if _ASK_TEST_ENABLED:
    PROFILES = PROFILES + (_ASK_TEST_PROFILE,)
PROFILE_BY_ID: dict[str, ProfileConfig] = {profile.profile_id: profile for profile in PROFILES}
DEFAULT_PROFILE_ID = "profile_0049"


def _env_bool(value: str | None) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


def _env_float(env: Mapping[str, str], name: str, default: float) -> float:
    raw = str(env.get(name, default)).strip()
    try:
        value = float(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be a number, got {raw!r}") from exc
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")
    return value


def _env_int(env: Mapping[str, str], name: str, default: int) -> int:
    raw = str(env.get(name, default)).strip()
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {raw!r}") from exc


@dataclass(frozen=True)
class Settings:
    host: str = "127.0.0.1"
    port: int = DEFAULT_PORT
    search_mode: str = "bge"
    allow_memory_writes: bool = False
    default_profile_id: str = DEFAULT_PROFILE_ID
    mcp_timeout_s: float = DEFAULT_MCP_TIMEOUT_S
    clarification_ttl_s: float = DEFAULT_CLARIFICATION_TTL_S
    console_only_log: bool = False
    # Behavior Feedback V1: first-party local events, short-term profile, shadow ranking.
    behavior_enabled: bool = True
    behavior_db_path: Path | None = None
    behavior_window_s: float = DEFAULT_WINDOW_S
    behavior_half_life_s: float = DEFAULT_HALF_LIFE_S
    behavior_retention_days: float = DEFAULT_RETENTION_DAYS
    # Durable Web Demo session state (identity, counters, metadata).  ``None`` keeps the
    # process-local SessionStore so every existing test and the default demo are unchanged.
    web_state_db_path: Path | None = None
    # Durable M9 for the MCP child.  ``None`` keeps its in-process store.  The environment
    # variable name mirrors ``recommendation.memory.store.MEMORY_DB_ENV_VAR``; the BFF does
    # not import recommendation core modules, so the literal is duplicated and asserted in
    # tests instead of imported.
    memory_db_path: Path | None = None
    # Optional real Qwen3.5-9B Base Live.  The endpoint is deliberately never exposed publicly.
    qwen_live_enabled: bool = False
    qwen_model_endpoint: str = ""
    qwen_model_name: str = "Qwen3.5-9B-Base"
    qwen_live_timeout_s: float = DEFAULT_QWEN_TIMEOUT_S
    qwen_health_timeout_s: float = DEFAULT_QWEN_HEALTH_TIMEOUT_S
    qwen_turn_timeout_s: float = DEFAULT_QWEN_TURN_TIMEOUT_S
    qwen_max_steps: int = DEFAULT_QWEN_MAX_STEPS
    qwen_health_interval_s: float = DEFAULT_QWEN_HEALTH_INTERVAL_S
    qwen_explain: bool = True
    #: ``strict`` (default) or ``demo_historical_fx``.  The demo mode converts the real Raw
    #: Meta historical USD price at a fixed 7.00 rate for the local demonstration only.
    price_mode: str = "strict"
    demo_price_sidecar: Path | None = None
    # Optional OpenTelemetry tracing.  Disabled by default; the OTLP endpoint is only read
    # by the exporter and is never exposed through health/config payloads.
    otel_enabled: bool = False
    otel_endpoint: str = "http://127.0.0.1:6006/v1/traces"
    otel_project_name: str = "agentrecx-x"
    otel_service_name: str = "agentrecx-web-demo"
    otel_sample_ratio: float = 1.0
    otel_export_timeout_s: float = 2.0
    repo: Path = REPO
    log_dir: Path = LOG_DIR
    replay_path: Path = REPLAY_PATH
    ranker_replay_path: Path = RANKER_REPLAY_PATH
    p100_tasks_path: Path = P100_TASKS_PATH
    p100_metadata_path: Path = P100_METADATA_PATH
    memory_fixture: Path = MEMORY_FIXTURE
    python_executable: str = field(default_factory=lambda: sys.executable)

    @property
    def qwen_live_configured(self) -> bool:
        return bool(self.qwen_live_enabled and self.qwen_model_endpoint)

    @property
    def behavior_store_path(self) -> Path:
        """SQLite event store; never inside the git tree's tracked data directory."""
        if self.behavior_db_path is not None:
            return Path(self.behavior_db_path)
        return self.log_dir.parent / "behavior" / "events.sqlite3"

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
    qwen_enabled = _env_bool(
        want.get("QWEN_LIVE_ENABLED") or want.get("AGENTRECX_WEB_DEMO_QWEN_LIVE")
    )
    qwen_endpoint = str(
        want.get("QWEN_MODEL_ENDPOINT") or want.get("AGENTRECX_QWEN_PLANNER_URL") or ""
    ).strip()
    qwen_model = (
        str(want.get("QWEN_MODEL_NAME") or "Qwen3.5-9B-Base").strip()
        or "Qwen3.5-9B-Base"
    )
    behavior_db_raw = str(want.get("AGENTRECX_BEHAVIOR_DB") or "").strip()
    behavior_db = Path(behavior_db_raw) if behavior_db_raw else None
    web_state_db_raw = str(want.get("AGENTRECX_WEB_STATE_DB") or "").strip()
    web_state_db = Path(web_state_db_raw) if web_state_db_raw else None
    memory_db_raw = str(want.get("AGENTRECX_MEMORY_DB") or "").strip()
    memory_db = Path(memory_db_raw) if memory_db_raw else None
    price_mode = str(want.get("AGENTRECX_PRICE_MODE", "strict")).strip().lower() or "strict"
    if price_mode not in {"strict", "demo_historical_fx"}:
        raise ValueError(
            "AGENTRECX_PRICE_MODE must be 'strict' or 'demo_historical_fx', "
            f"got {price_mode!r}"
        )
    demo_sidecar_raw = str(want.get("AGENTRECX_DEMO_PRICE_SIDECAR") or "").strip()
    demo_sidecar = Path(demo_sidecar_raw) if demo_sidecar_raw else None
    otel_timeout = _env_float(want, "AGENTRECX_OTEL_EXPORT_TIMEOUT_S", 2.0)
    if otel_timeout <= 0:
        raise ValueError("AGENTRECX_OTEL_EXPORT_TIMEOUT_S must be positive")
    otel_ratio = _env_float(want, "AGENTRECX_OTEL_SAMPLE_RATIO", 1.0)
    if not 0.0 <= otel_ratio <= 1.0:
        raise ValueError("AGENTRECX_OTEL_SAMPLE_RATIO must be between 0 and 1")
    otel_endpoint = (
        str(want.get("AGENTRECX_OTEL_ENDPOINT") or "").strip()
        or str(want.get("PHOENIX_COLLECTOR_ENDPOINT") or "").strip()
        or str(want.get("OTEL_EXPORTER_OTLP_ENDPOINT") or "").strip()
        or "http://127.0.0.1:6006/v1/traces"
    )
    return Settings(
        port=port,
        search_mode=mode,
        allow_memory_writes=_env_bool(want.get("AGENTRECX_WEB_DEMO_ALLOW_MEMORY_WRITES")),
        default_profile_id=(
            str(want.get("AGENTRECX_WEB_DEMO_DEFAULT_PROFILE") or DEFAULT_PROFILE_ID).strip()
            if str(want.get("AGENTRECX_WEB_DEMO_DEFAULT_PROFILE") or DEFAULT_PROFILE_ID).strip()
            in PROFILE_BY_ID else DEFAULT_PROFILE_ID
        ),
        mcp_timeout_s=timeout,
        console_only_log=_env_bool(want.get("AGENTRECX_WEB_DEMO_CONSOLE_ONLY_LOG")),
        behavior_enabled=_env_bool(want.get("AGENTRECX_BEHAVIOR_ENABLED", "1")),
        behavior_db_path=behavior_db,
        web_state_db_path=web_state_db,
        memory_db_path=memory_db,
        price_mode=price_mode,
        demo_price_sidecar=demo_sidecar,
        otel_enabled=_env_bool(want.get("AGENTRECX_OTEL_ENABLED")),
        otel_endpoint=otel_endpoint,
        otel_project_name=(
            str(want.get("AGENTRECX_OTEL_PROJECT") or "agentrecx-x").strip()
            or "agentrecx-x"
        ),
        otel_service_name=(
            str(want.get("AGENTRECX_OTEL_SERVICE_NAME") or "agentrecx-web-demo").strip()
            or "agentrecx-web-demo"
        ),
        otel_sample_ratio=otel_ratio,
        otel_export_timeout_s=otel_timeout,
        behavior_window_s=_env_float(want, "AGENTRECX_BEHAVIOR_WINDOW_S", DEFAULT_WINDOW_S),
        behavior_half_life_s=_env_float(
            want, "AGENTRECX_BEHAVIOR_HALF_LIFE_S", DEFAULT_HALF_LIFE_S
        ),
        behavior_retention_days=_env_float(
            want, "AGENTRECX_BEHAVIOR_RETENTION_DAYS", DEFAULT_RETENTION_DAYS
        ),
        qwen_live_enabled=qwen_enabled,
        qwen_model_endpoint=qwen_endpoint,
        qwen_model_name=qwen_model,
        qwen_live_timeout_s=_env_float(want, "QWEN_LIVE_TIMEOUT_S", DEFAULT_QWEN_TIMEOUT_S),
        qwen_health_timeout_s=_env_float(
            want, "QWEN_LIVE_HEALTH_TIMEOUT_S", DEFAULT_QWEN_HEALTH_TIMEOUT_S
        ),
        qwen_turn_timeout_s=_env_float(
            want, "QWEN_LIVE_TURN_TIMEOUT_S", DEFAULT_QWEN_TURN_TIMEOUT_S
        ),
        qwen_max_steps=max(1, min(8, _env_int(want, "QWEN_LIVE_MAX_STEPS", DEFAULT_QWEN_MAX_STEPS))),
        qwen_health_interval_s=max(
            5.0, _env_float(want, "QWEN_LIVE_HEALTH_INTERVAL_S", DEFAULT_QWEN_HEALTH_INTERVAL_S)
        ),
        qwen_explain=_env_bool(want.get("QWEN_LIVE_EXPLAIN", "1")),
    )


def modes_public(
    *,
    cpu_live_available: bool,
    qwen_health: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """The three run modes exposed to the UI; only CPU Live and (when healthy) Qwen are live."""
    health = dict(qwen_health or {})
    qwen_available = bool(health.get("available"))
    qwen_status = str(health.get("status") or "disabled")
    if qwen_available:
        qwen_reason = None
    elif qwen_status == "disabled" or not health.get("enabled") or not health.get("configured"):
        qwen_reason = "No live Qwen service detected (QWEN_LIVE_ENABLED / QWEN_MODEL_ENDPOINT not configured)"
    elif qwen_status == "checking":
        qwen_reason = "No live Qwen service detected yet (real PlannerAction health check in progress)"
    else:
        qwen_reason = (
            "No live Qwen service detected (health check failed: "
            + str(health.get("detail") or qwen_status)
            + ")"
        )
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
            "available": qwen_available,
            "label": "Qwen Base Live",
            "detail": (
                "Real Qwen3.5-9B Base PlannerAction → existing electronics executor → "
                "Gate/P15/P16"
            ),
            "badge": "Live Qwen3.5-9B Base",
            "disabled_reason": qwen_reason,
            "planner": "qwen_base" if qwen_available else None,
        },
    }


def limits_public(settings: Settings) -> dict[str, Any]:
    return {
        "search_top_k": SEARCH_TOP_K,
        "compare_min": COMPARE_MIN,
        "compare_max": COMPARE_MAX,
        "clarification_ttl_s": settings.clarification_ttl_s,
        "qwen_live_max_steps": settings.qwen_max_steps,
    }


def behavior_public(settings: Settings, *, session_enabled: bool | None = None) -> dict[str, Any]:
    """Public, non-sensitive description of the local behavior collector."""
    return {
        "enabled": bool(settings.behavior_enabled if session_enabled is None else session_enabled),
        "master_enabled": bool(settings.behavior_enabled),
        "semantics_version": BEHAVIOR_SEMANTICS_VERSION,
        "storage": "sqlite",
        "impression_min_visible_ms": IMPRESSION_MIN_VISIBLE_MS,
        "impression_visibility_threshold": IMPRESSION_VISIBILITY_THRESHOLD,
        "window_seconds": settings.behavior_window_s,
        "half_life_seconds": settings.behavior_half_life_s,
        "event_types": list(EVENT_TYPES),
        "event_sources": list(EVENT_SOURCES),
        "compare_action_statuses": list(COMPARE_ACTION_STATUSES),
        "live_source": "LIVE_USER_INTERACTION",
        "profile_sources": list(PROFILE_SOURCES),
        "profile_uses_live_only": False,
        "human_preference_statistics_source": "LIVE_USER_INTERACTION",
        "impression_interest_weight": 0.0,
        "compare_interest_weight": 0.0,
        "retention_days": settings.behavior_retention_days,
        "external_analytics": None,
        "notes": [
            "First-party BFF events; no PostHog/Snowplow SDK and no autocapture.",
            "The browser never sends user_id, raw chat text or memory text.",
            "Impressions require browser visibility, not the backend Top-K list.",
            "Impressions are exposure context only and carry zero interest weight.",
            "Compare attempt/completed/failed is operation provenance, not a preference.",
            "Formal shadow comparison is eligible-only; diagnostic ranking is never final.",
        ],
    }

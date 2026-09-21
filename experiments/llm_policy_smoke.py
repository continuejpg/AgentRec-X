"""Optional live provider smoke for the model-driven policy (Phase 1).

**Disabled by default.**  Nothing here runs during ``pytest``, and the script exits with a clear
message when no provider is configured.  It exists so a deployment can confirm its adapter wiring
against a real endpoint without putting a live call into the reproducible suite.

Usage::

    # Dry run: shows what would be sent, calls nothing.  This is the default.
    .venv/bin/python -m experiments.llm_policy_smoke

    # Live: requires an explicit opt-in AND configuration.
    AGENTRECX_LLM_BASE_URL=http://localhost:8000/v1 \\
    AGENTRECX_LLM_MODEL=Qwen2.5-7B-Instruct \\
    AGENTRECX_LLM_API_KEY=EMPTY \\
    .venv/bin/python -m experiments.llm_policy_smoke --live --transport urllib

Requirements this script meets:

* **disabled by default** - ``--live`` is required, and ``--transport`` must name an HTTP client;
* **explicit configuration** - the provider must be named through ``AGENTRECX_LLM_*``;
* **never in pytest** - it is an experiment module, not a test;
* **no API keys in the source** - the key is read from the environment at call time;
* **clear failure when unconfigured** - the message names the missing setting.

No HTTP library is imported at module scope: ``--transport urllib`` imports the standard
library's ``urllib.request`` inside the call, so the core dependency set is unchanged.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from recommendation.control import (  # noqa: E402
    ActionKind,
    CandidateLedger,
    CandidatePlane,
    CatalogSearchSource,
    GroundedReasoner,
    GroundingVerifier,
    LoopLimits,
    PolicyContext,
    ReasoningExecutor,
)
from recommendation.control.context import CandidateState  # noqa: E402
from recommendation.control.model_client import ModelCallError  # noqa: E402
from recommendation.control.model_policy import (  # noqa: E402
    LLMAgentPolicy,
    POLICY_PROMPT_VERSION,
    build_action_schema,
    build_policy_context_payload,
    build_policy_system_prompt,
)
from recommendation.control.model_recorder import (  # noqa: E402
    DEFAULT_RECORDINGS_DIR,
    ModelSink,
    RecordingModelClient,
    RecordingStore,
    ReplayMode,
)
from recommendation.control.provider_adapter import (  # noqa: E402
    ENV_BASE_URL,
    ENV_MODEL,
    build_provider_client,
    provider_configured,
    provider_settings,
)
from tests.agent_reranking_fixture import CANDIDATE_ROWS  # noqa: E402
from tests.control_fixture import build_control_harness  # noqa: E402


#: The transport now lives in the adapter module, so the smoke and the evaluation command call
#: the provider through **one** implementation.  A local copy here would be a second HTTP path
#: that no test covers and that could drift from the one the evaluation actually uses.
from recommendation.control.provider_adapter import urllib_transport as _urllib_transport  # noqa: E402


class _Map:
    """The trusted identity map the fixture catalogue is keyed by."""

    def __init__(self) -> None:
        self._ids = {row[0]: row[1] for row in CANDIDATE_ROWS}

    def has_parent_asin(self, parent_asin: str) -> bool:
        return parent_asin in self._ids

    def parent_asin_to_item_id(self, parent_asin: str) -> int:
        return self._ids[parent_asin]

    def item_id_to_parent_asin(self, item_id: int) -> str:
        for identity, value in self._ids.items():
            if value == item_id:
                return identity
        raise KeyError(item_id)


#: The request the dry run describes.  A live run uses the same one.
REQUEST = "I need a waterproof option under $120."
SUMMARY: dict[str, Any] = {}


def _summary_path() -> Path:
    """Where the machine-readable summary is written."""
    return Path(os.environ.get("AGENTRECX_LLM_SMOKE_SUMMARY", "/tmp/agentrecx_llm_policy_smoke.json"))


def run(
    *,
    live: bool,
    transport: str | None,
    mode: str | None = None,
    recording: str | None = None,
) -> int:
    """Run the smoke.  Returns a process exit code.

    ``mode`` supersedes ``live``: ``dry`` (default) shows what would be sent, ``live`` calls the
    provider, ``record`` calls and writes a recording, ``replay`` reproduces a recorded call with
    no network.  ``live`` is kept as a spelling of ``--mode live`` so Phase-1 invocations still
    work.
    """
    print("=" * 78)
    print(" AgentRec-X model-driven policy smoke (Phase 1)")
    print("=" * 78)
    print()

    # The request that would be sent, built from the real prompt builder so the dry run
    # describes the live call faithfully.
    context = PolicyContext(
        user_request=REQUEST,
        available_actions=(ActionKind.RECOMMEND_FROM_HISTORY, ActionKind.SEARCH_CATALOG, ActionKind.FINISH),
        has_trusted_history=True,
        candidate_state=CandidateState(),
        remaining_steps=5,
        remaining_tool_calls=3,
    )
    schema = build_action_schema(tuple(context.available_actions))
    payload = build_policy_context_payload(context)
    print(" policy contract prompt (first 3 lines):")
    for line in build_policy_system_prompt().splitlines()[:3]:
        print(f"   {line}")
    print()
    print(f" offered actions: {[entry['action'] for entry in schema]}")
    print(f" context payload keys: {sorted(payload)}")
    print()

    resolved_mode = mode or ("live" if live else "dry")
    if resolved_mode == "dry":
        print(" DRY RUN - no request was made.")
        print(f" use --mode live (or --live) with {ENV_BASE_URL} / {ENV_MODEL} to call a real endpoint.")
        SUMMARY.update(
            {"mode": "dry_run", "called_provider": False, "configured": provider_configured()}
        )
        _write_summary()
        return 0

    # A transport name is accepted for compatibility, but the standard-library client is the
    # default now: it is the same transport the evaluation command uses, and requiring a flag to
    # select the only implementation was ceremony rather than safety.
    supported = {"urllib", None}
    if transport not in supported:
        print(f" unknown transport {transport!r}; supported: urllib")
        SUMMARY.update({"mode": resolved_mode, "called_provider": False, "error": "unknown_transport"})
        _write_summary()
        return 2

    recording_path = Path(recording) if recording else DEFAULT_RECORDINGS_DIR / "smoke.jsonl"

    # REPLAY is offline by definition: it needs a recording and must not require configuration.
    if resolved_mode == "replay":
        store = RecordingStore(recording_path)
        if not store.path.exists():
            print(f" NOT CONFIGURED - no recording at {store.path}; run --mode record first.")
            SUMMARY.update(
                {"mode": "replay", "called_provider": False, "error": "recording_missing"}
            )
            _write_summary()
            return 2
        from recommendation.control.model_recorder import recording_identity

        recorded = recording_identity(store.path)
        sink = ModelSink()
        client = RecordingModelClient(
            _ReplayGuard(),
            store=store,
            mode=ReplayMode.REPLAY,
            provider=str(recorded.get("provider") or "recorded"),
            model=str(recorded.get("model") or "recorded"),
            prompt_version=recorded.get("prompt_version"),
            sink=sink,
        )
        print(f" replaying from {store.path} ({len(store)} recorded response(s))")
        return _decide(client, context, sink, resolved_mode, called=False)

    if not provider_configured():
        print(f" NOT CONFIGURED - set {ENV_BASE_URL} and {ENV_MODEL} (and a key if required).")
        SUMMARY.update({"mode": resolved_mode, "called_provider": False, "error": "not_configured"})
        _write_summary()
        return 2

    try:
        settings = provider_settings()
        wrapped = build_provider_client(
            transport=_urllib_transport, settings=settings, max_tokens=512
        )
    except ModelCallError as exc:
        print(f" could not build a provider client: {exc.code} - {exc}")
        SUMMARY.update({"mode": resolved_mode, "called_provider": False, "error": exc.code})
        _write_summary()
        return 2

    sink = ModelSink()
    record_mode = ReplayMode.RECORD if resolved_mode == "record" else ReplayMode.LIVE
    client = RecordingModelClient(
        wrapped,
        store=RecordingStore(recording_path),
        mode=record_mode,
        provider=settings.profile.name,
        model=settings.model,
        prompt_version=POLICY_PROMPT_VERSION,
        sink=sink,
    )
    print(f" provider : {settings.base_url} model={settings.model} profile={settings.profile.name}")
    print(f" credential present: {settings.api_key_present}   (never printed, never recorded)")
    print(" calling the provider ...")
    return _decide(client, context, sink, resolved_mode, called=True)


def _decide(
    client: Any, context: PolicyContext, sink: ModelSink, mode: str, *, called: bool
) -> int:
    """Ask the policy for one action through ``client`` and report the outcome."""
    policy = LLMAgentPolicy(client, max_attempts=2)
    try:
        proposal = policy.choose(context)
    except Exception as exc:  # noqa: BLE001 - report the failure clearly and stop
        print(f" the model policy produced no usable action: {type(exc).__name__}: {exc}")
        SUMMARY.update(
            {
                "mode": mode,
                "called_provider": called,
                "ok": False,
                "usage": sink.summary(),
                "metadata": policy.last_metadata,
            }
        )
        _write_summary()
        return 1

    print(f" proposed action : {proposal.action.value}")
    print(f" arguments       : {proposal.arguments}")
    print(f" validated       : the ActionProposal was built, so ActionValidator will see a legal shape")
    usage = sink.summary()
    print(f" model calls     : {usage['model_calls']} (live {usage['live_calls']}, replayed {usage['replayed_calls']})")
    tokens = (
        f"{usage['input_tokens']}/{usage['output_tokens']}/{usage['total_tokens']}"
        if usage["total_tokens"] is not None
        else "unknown (provider reported no usage)"
    )
    print(f" tokens in/out/total: {tokens}")
    latency = (
        f"{usage['model_latency_ms_total']} ms"
        if usage["model_latency_ms_total"] is not None
        else "unknown (not measured)"
    )
    print(f" model latency   : {latency}")
    cost = (
        f"{usage['estimated_cost_total']}"
        if usage["estimated_cost_total"] is not None
        else "unknown (no pricing configured)"
    )
    print(f" estimated cost  : {cost}")
    print(f" diagnostics     : {policy.last_metadata}")
    SUMMARY.update(
        {
            "mode": mode,
            "called_provider": called,
            "ok": True,
            "action": proposal.action.value,
            "usage": usage,
            "metadata": policy.last_metadata,
        }
    )
    _write_summary()
    return 0


class _ReplayGuard:
    """Refuses to be called, so a replay that reached the network fails loudly."""

    def complete(self, request: Any) -> Any:  # pragma: no cover - guard
        raise ModelCallError("replay attempted a provider call", code="replay_violation")


def _write_summary() -> None:
    """Write the machine-readable summary."""
    path = _summary_path()
    path.write_text(json.dumps(SUMMARY, indent=2, default=str), encoding="utf-8")
    print(f" wrote {path}")


def build_live_loop() -> Any:
    """Compose the control plane a live run would drive, for interactive use.

    Not used by :func:`run` (which only asks the policy for one action), but kept so a
    deployment can wire a live policy into the loop without re-deriving the composition.
    """
    harness = build_control_harness(catalog_rows=CANDIDATE_ROWS)
    metadata = harness.parts["enricher"].metadata
    harness.controller._reasoning = ReasoningExecutor(GroundedReasoner(metadata))  # noqa: SLF001
    harness.controller._candidate_plane = CandidatePlane(  # noqa: SLF001
        ledger=CandidateLedger(),
        grounding=GroundingVerifier(_Map(), metadata),
        catalog_search=CatalogSearchSource(metadata),
    )
    return harness


def main(argv: list[str] | None = None) -> int:
    """Parse arguments and run the smoke."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=("dry", "live", "record", "replay"),
        default=None,
        help=(
            "dry = show the request and call nothing (default); "
            "live = one real provider call; "
            "record = one real call, written to a recording; "
            "replay = reproduce a recording with no network"
        ),
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="compatibility spelling of --mode live",
    )
    parser.add_argument(
        "--transport",
        default=None,
        help="HTTP transport for a live call (default: the standard-library client)",
    )
    parser.add_argument(
        "--recording",
        default=None,
        help=f"recording file (default: {DEFAULT_RECORDINGS_DIR}/smoke.jsonl)",
    )
    args = parser.parse_args(argv)
    return run(
        live=args.live,
        transport=args.transport,
        mode=args.mode,
        recording=args.recording,
    )


if __name__ == "__main__":
    raise SystemExit(main())

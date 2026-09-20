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
    build_action_schema,
    build_policy_context_payload,
    build_policy_system_prompt,
)
from recommendation.control.provider_adapter import (  # noqa: E402
    ENV_BASE_URL,
    ENV_MODEL,
    build_provider_client,
    provider_configured,
)
from tests.agent_reranking_fixture import CANDIDATE_ROWS  # noqa: E402
from tests.control_fixture import build_control_harness  # noqa: E402


def _urllib_transport(url: str, *, headers: dict[str, str], payload: dict[str, Any], timeout: float):
    """A standard-library transport, imported lazily so the module needs no HTTP dependency."""
    import urllib.error
    import urllib.request

    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(url, data=data, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status, response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:  # a provider error carries a status
        return exc.code, exc.read().decode("utf-8", "replace")


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


def run(*, live: bool, transport: str | None) -> int:
    """Run the smoke.  Returns a process exit code."""
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

    if not live:
        print(" DRY RUN - no request was made.")
        print(f" set --live and configure {ENV_BASE_URL} / {ENV_MODEL} to call a real endpoint.")
        SUMMARY.update({"mode": "dry_run", "called_provider": False, "configured": provider_configured()})
        _write_summary()
        return 0

    if not provider_configured():
        print(f" NOT CONFIGURED - set {ENV_BASE_URL} and {ENV_MODEL} (and a key if required).")
        SUMMARY.update({"mode": "live", "called_provider": False, "error": "not_configured"})
        _write_summary()
        return 2

    if transport is None:
        print(" refuse: --live requires --transport <name> (this repo imports no HTTP client).")
        SUMMARY.update({"mode": "live", "called_provider": False, "error": "no_transport"})
        _write_summary()
        return 2

    transport_fn = {"urllib": _urllib_transport}.get(transport)
    if transport_fn is None:
        print(f" unknown transport {transport!r}; supported: urllib")
        SUMMARY.update({"mode": "live", "called_provider": False, "error": "unknown_transport"})
        _write_summary()
        return 2

    try:
        client = build_provider_client(transport=transport_fn)
    except ModelCallError as exc:
        print(f" could not build a provider client: {exc.code} - {exc}")
        SUMMARY.update({"mode": "live", "called_provider": False, "error": exc.code})
        _write_summary()
        return 2

    policy = LLMAgentPolicy(client, max_attempts=2)
    print(" calling the provider ...")
    try:
        proposal = policy.choose(context)
    except Exception as exc:  # noqa: BLE001 - report the failure clearly and stop
        print(f" the model policy produced no usable action: {type(exc).__name__}: {exc}")
        SUMMARY.update(
            {
                "mode": "live",
                "called_provider": True,
                "ok": False,
                "metadata": policy.last_metadata,
            }
        )
        _write_summary()
        return 1

    print(f" proposed action : {proposal.action.value}")
    print(f" arguments       : {proposal.arguments}")
    print(f" diagnostics     : {policy.last_metadata}")
    SUMMARY.update(
        {
            "mode": "live",
            "called_provider": True,
            "ok": True,
            "action": proposal.action.value,
            "metadata": policy.last_metadata,
        }
    )
    _write_summary()
    return 0


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
    parser.add_argument("--live", action="store_true", help="actually call the provider")
    parser.add_argument(
        "--transport",
        default=None,
        help="HTTP transport to use for a live call (supported: urllib)",
    )
    args = parser.parse_args(argv)
    return run(live=args.live, transport=args.transport)


if __name__ == "__main__":
    raise SystemExit(main())

"""Real-provider agent evaluation with record and replay (Phase 3).

Runs the existing agent case suite through the existing bounded policy seam, with a **real
provider** in place of the deterministic double, and records each model exchange so the run can
be reproduced offline.

::

    # 1. Live: call the provider, record every exchange.
    AGENTRECX_LLM_BASE_URL=https://api.deepseek.com \\
    AGENTRECX_LLM_MODEL=deepseek-flash \\
    AGENTRECX_LLM_API_KEY=... \\
    .venv/bin/python -m experiments.live_agent_evaluation --record

    # 2. Replay: identical policy decisions, zero network, zero cost.
    .venv/bin/python -m experiments.live_agent_evaluation --replay

    # 3. The deterministic baselines, for the same-suite comparison.
    .venv/bin/python -m experiments.agent_evaluation_smoke

What varies between the three
-----------------------------
Only the **policy**.  The runner, the controller, the tools, the catalogue, the candidate
ledger, the constraint kernel, the reference boundary, the completion guard and the case
definitions are the same objects in every run (see ``experiments/_harness.py``).  That is what
makes the comparison an ablation rather than a description of three different systems.

What is recorded
----------------
The provider's *responses*, keyed by a canonical fingerprint of the request - not the policy's
decisions.  A replay therefore reproduces the model's answers and lets the same policy, validator
and guards run again, which is a stronger claim than replaying a trajectory: the trusted runtime
genuinely re-executes, and only the model's text is substituted.

The report this writes contains counts, verdicts, timings and model usage.  It contains no
credential, no authorization header and no provider chain-of-thought - the adapter never reads
the reasoning field, and the repository never requests it.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments._harness import case_controller_factory  # noqa: E402
from recommendation.control.model_client import ModelCallError  # noqa: E402
from recommendation.control.model_recorder import (  # noqa: E402
    DEFAULT_RECORDINGS_DIR,
    ModelSink,
    RecordingModelClient,
    RecordingStore,
    ReplayMode,
    recording_identity,
)
from recommendation.control.model_policy import (  # noqa: E402
    POLICY_PARSE_PROTOCOL_VERSION,
    POLICY_PROMPT_VERSION,
)
from recommendation.control.provider_adapter import (  # noqa: E402
    build_provider_client,
    provider_settings,
)
from recommendation.evaluation.agent import (  # noqa: E402
    ABLATION_LIVE_MODEL,
    CaseRunner,
    load_cases,
)
from recommendation.evaluation.agent.model_policy_adapter import (  # noqa: E402
    build_live_model_policy_factory,
)
from recommendation.evaluation.agent.runner import SuiteReport  # noqa: E402

__all__ = ["main", "run"]

#: Modes the command exposes, mapped onto the recorder's modes.  ``record`` writes a fresh
#: recording; ``replay`` is strictly offline; ``live`` calls without writing, for a smoke.
MODES: dict[str, ReplayMode] = {
    "live": ReplayMode.LIVE,
    "record": ReplayMode.RECORD,
    "replay": ReplayMode.REPLAY,
}


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Real-provider agent evaluation (Phase 3).")
    parser.add_argument(
        "--mode",
        choices=sorted(MODES),
        default="replay",
        help=(
            "live = call the provider, record nothing; "
            "record = call the provider and write a recording; "
            "replay = never call the provider (default, so an accidental run costs nothing)"
        ),
    )
    parser.add_argument(
        "--recording",
        default=None,
        help=f"recording file (default: {DEFAULT_RECORDINGS_DIR}/deepseek_agent_suite.jsonl)",
    )
    parser.add_argument("--output", default=None, help="write the JSON report to this path")
    parser.add_argument("--max-attempts", type=int, default=2, help="model calls per decision")
    parser.add_argument(
        "--confirm-live",
        action="store_true",
        help=(
            "required in addition to --mode live/record: a live run costs money, so it takes two "
            "deliberate flags rather than one, and no configuration alone can trigger it"
        ),
    )
    return parser.parse_args(argv)


def _recording_path(explicit: str | None) -> Path:
    if explicit:
        return Path(explicit)
    return DEFAULT_RECORDINGS_DIR / "deepseek_agent_suite.jsonl"


def _build_client(
    *,
    mode: ReplayMode,
    recording: Path,
    sink: ModelSink,
    max_attempts: int,
    wrapped: Any = None,
) -> tuple[Any, dict[str, Any]]:
    """Build the recording client for the requested mode, or fail with a clear message.

    Returns ``(client, identity)``.  ``identity`` is log-safe: it names the provider and model and
    reports only *whether* a credential is present, never the credential.

    ``wrapped`` is an injection point for tests and for a deployment with its own client.  It is
    only consulted in the provider-calling modes; ``REPLAY`` never needs one.
    """
    if mode is ReplayMode.REPLAY:
        store = RecordingStore(recording)
        if not store.path.exists():
            raise ModelCallError(
                f"--mode replay needs a recording at {store.path}; "
                "run --mode record first, or pass --recording <path>",
                code="recording_missing",
            )
        # The fingerprint includes the provider and model, so the replay must key on the identity
        # the recording was *made* against - not on whatever the environment says now.  Reading
        # it from the file is what makes a replay reproducible on a machine with no provider
        # configured at all, which is the whole point of the mode.
        recorded = recording_identity(store.path)
        client = RecordingModelClient(
            _NoProvider(),
            store=store,
            mode=ReplayMode.REPLAY,
            provider=str(recorded.get("provider") or "recorded"),
            model=str(recorded.get("model") or "recorded"),
            prompt_version=recorded.get("prompt_version"),
            parse_protocol_version=recorded.get("parse_protocol_version"),
            sink=sink,
        )
        return client, {
            "execution_mode": ReplayMode.REPLAY.value,
            "provider": recorded.get("provider"),
            "model": recorded.get("model"),
            "prompt_version_recorded": recorded.get("prompt_version"),
            "parse_protocol_version_recorded": recorded.get("parse_protocol_version"),
            "recording": str(store.path),
            "recorded_entries": len(store),
        }

    if wrapped is not None:
        # An injected client: a test supplies a deterministic provider double, and a deployment
        # may supply its own pooled client.  The recorder wraps it exactly as it wraps the
        # standard-library one, so no path exists that bypasses recording.
        client = RecordingModelClient(
            wrapped,
            store=RecordingStore(recording),
            mode=mode,
            provider=getattr(wrapped, "provider_name", "injected"),
            model=getattr(wrapped, "model", "injected"),
            prompt_version=POLICY_PROMPT_VERSION,
            parse_protocol_version=POLICY_PARSE_PROTOCOL_VERSION,
            sink=sink,
        )
        return client, {
            "execution_mode": mode.value,
            "provider": getattr(wrapped, "provider_name", "injected"),
            "model": getattr(wrapped, "model", "injected"),
            "recording": str(recording),
        }

    settings = provider_settings()
    provider_client = build_provider_client(settings=settings, max_tokens=1024)
    client = RecordingModelClient(
        provider_client,
        store=RecordingStore(recording),
        mode=mode,
        provider=settings.profile.name,
        model=settings.model,
        prompt_version=POLICY_PROMPT_VERSION,
        sink=sink,
    )
    identity = {
        "execution_mode": mode.value,
        "recording": str(recording),
        **settings.describe(),
    }
    return client, identity


class _NoProvider:
    """A wrapped client that must never be called, for the replay path.

    ``RecordingModelClient`` in ``REPLAY`` mode never reaches it; having it raise loudly means a
    future mode-semantics regression fails the run instead of silently reaching the network.
    """

    def complete(self, request: Any) -> Any:  # pragma: no cover - guard
        raise ModelCallError(
            "replay attempted a provider call; this must never happen", code="replay_violation"
        )


def run(
    *,
    mode: ReplayMode,
    recording: Path,
    output: Path | None,
    max_attempts: int = 2,
    on_case: Any = None,
    wrapped: Any = None,
) -> tuple[int, dict[str, Any]]:
    """Run the suite and return ``(exit_code, report)``.

    ``wrapped`` injects the underlying client, which is how an offline test exercises the whole
    record/replay path without a provider.
    """
    sink = ModelSink()
    client, identity = _build_client(
        mode=mode,
        recording=recording,
        sink=sink,
        max_attempts=max_attempts,
        wrapped=wrapped,
    )
    factory = build_live_model_policy_factory(client=client, max_attempts=max_attempts)

    runner = CaseRunner(case_controller_factory, variant=ABLATION_LIVE_MODEL)
    outcomes: list[Any] = []
    started = time.monotonic()
    for case in load_cases():
        outcome = runner.run(case, policy=factory(case))
        outcomes.append(outcome)
        if on_case is not None:
            on_case(case, outcome)
    wall_ms = round(max(0.0, (time.monotonic() - started) * 1000.0), 3)

    report = SuiteReport(ABLATION_LIVE_MODEL, outcomes)
    payload = _report_payload(
        report=report,
        identity=identity,
        sink=sink,
        client=client,
        wall_ms=wall_ms,
    )
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return (0 if report.failed == 0 else 1), payload


def _report_payload(
    *,
    report: SuiteReport,
    identity: dict[str, Any],
    sink: ModelSink,
    client: Any,
    wall_ms: float,
) -> dict[str, Any]:
    """Assemble the experiment report: per-case verdicts plus model efficiency.

    Deliberately **not** collapsed into one score.  Each case reports the dimensions the plane
    already measures, and the suite reports component-level failure counts; the model section is
    efficiency (calls, tokens, latency, cost), which is orthogonal to quality and must not be
    folded into it.
    """
    payload = report.as_dict()
    payload["model"] = {
        **identity,
        "policy_prompt_version": POLICY_PROMPT_VERSION,
        "policy_parse_protocol_version": POLICY_PARSE_PROTOCOL_VERSION,
        "replayed_calls": getattr(client, "replayed_calls", 0),
        "live_calls": getattr(client, "live_calls", 0),
        "usage": sink.summary(),
        "suite_wall_latency_ms": wall_ms,
    }
    # The trajectory-level model facts, per case, so a reader can attribute cost and behaviour
    # to a specific case without re-deriving them from the step list.
    for entry, outcome in zip(payload["cases"], report.outcomes):
        trajectory = outcome.trajectory
        entry["model"] = {
            "execution_mode": trajectory.execution_mode,
            "model_provider": trajectory.model_provider,
            "model_endpoint_id": trajectory.model_endpoint_id,
            "model_calls": trajectory.model_calls,
            "input_tokens": trajectory.input_tokens,
            "output_tokens": trajectory.output_tokens,
            "total_tokens": trajectory.total_tokens,
            "model_latency_ms": trajectory.model_latency_ms,
            "wall_latency_ms": trajectory.wall_latency_ms,
            "estimated_cost": trajectory.estimated_cost,
        }
    return payload


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    mode = MODES[args.mode]
    recording = _recording_path(args.recording)

    print("=" * 78)
    print(" AgentRec-X real-provider agent evaluation (Phase 3)")
    print("=" * 78)
    print(f" mode: {mode.value}   recording: {recording}")
    print(f" prompt version: {POLICY_PROMPT_VERSION}")
    print()

    # Live execution is opt-in twice over: an explicit mode *and* an explicit confirmation.  No
    # environment variable can start a paid run on its own.
    if mode.calls_provider and not args.confirm_live:
        print(f"--mode {mode.value} would call a real provider and cost money.")
        print(" Re-run with --confirm-live to proceed. Nothing was sent.")
        return 2

    try:
        exit_code, payload = run(
            mode=mode,
            recording=recording,
            output=Path(args.output) if args.output else None,
            max_attempts=args.max_attempts,
            on_case=_print_case,
        )
    except ModelCallError as exc:
        # A configuration or recording problem is reported as an actionable message, not a
        # traceback: the point of the command is to be runnable by someone who is not its author.
        print(f"\n cannot run: [{exc.code}] {exc}")
        return 2

    model = payload["model"]
    print()
    print("=" * 78)
    print(f" {payload['passed']}/{payload['total']} passed   variant={payload['variant']}")
    print(f" failures by component: {payload['failures_by_component'] or '{} (none)'}")
    print()
    usage = model["usage"]
    print(f" model: {model.get('model')} via {model.get('provider')} ({model['execution_mode']})")
    print(
        f" calls: {usage['model_calls']} "
        f"(live {usage['live_calls']}, replayed {usage['replayed_calls']})"
    )
    tokens = (
        f"{usage['input_tokens']}/{usage['output_tokens']}/{usage['total_tokens']}"
        if usage["total_tokens"] is not None
        else "unknown (provider reported no usage)"
    )
    print(f" tokens in/out/total: {tokens}")
    latency = (
        f"{usage['model_latency_ms_total']} ms total, "
        f"{usage['model_latency_ms_mean']} ms mean"
        if usage["model_latency_ms_total"] is not None
        else "unknown (not measured)"
    )
    print(f" model latency: {latency}")
    print(f" trajectory wall latency: {model['suite_wall_latency_ms']} ms")
    cost = (
        f"{usage['estimated_cost_total']}"
        if usage["estimated_cost_total"] is not None
        else "unknown (no pricing configured)"
    )
    print(f" estimated cost: {cost}")
    if args.output:
        print(f"\n wrote {args.output}")
    return exit_code


def _print_case(case: Any, outcome: Any) -> None:
    """Stream one case's result as it completes."""
    mark = "PASS" if outcome.passed else "FAIL"
    actions = " -> ".join(outcome.trajectory.action_sequence()) or "(none)"
    print(
        f"   [{mark}] {case.case_id:32s} {outcome.trajectory.terminal.value:19s} "
        f"steps={outcome.trajectory.steps} wall={outcome.trajectory.wall_latency_ms}ms"
    )
    print(f"          {actions}")
    if not outcome.passed:
        for name, owner in outcome.attribution.attributed.items():
            print(f"          ! {name} <- {owner}")


if __name__ == "__main__":  # pragma: no cover - CLI entry
    raise SystemExit(main())

"""M4 provider concurrency probe (non-cohort, <=16 calls).

Answers one question only: which provider concurrency is empirically stable, so the formal run can
use the highest rate that does not produce 429s or transport failures.

Scope discipline
----------------
* **Non-cohort.** The prompt is a synthetic structured-action request. No cohort user, no history,
  no catalogue, no recommendation metric is involved.
* **<=16 calls total**, split 1-way / 2-way / 4-way.
* **No tuning.** Temperature, max_tokens, JSON mode, profile and model all come from the frozen
  configuration via the repository's own ``build_provider_client``. Nothing is adjusted in response
  to a result except the concurrency selection itself, which is this probe's entire purpose.
* The credential is read through ``experiments.m4_benchmark.load_local_credential`` and never
  printed, stored or returned.

Usage
-----
    python -m experiments.m4_provider_probe --out /tmp/m4_probe.json
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import pathlib
import statistics
import sys
import time
from typing import Any, Sequence

REPO = pathlib.Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from experiments.m4_benchmark import load_local_credential  # noqa: E402

#: Calls per concurrency level. 3 + 4 + 8 = 15, within the 16-call budget.
PLAN: tuple[tuple[int, int], ...] = ((1, 3), (2, 4), (4, 8))

#: A minimal structured-action request of the same shape the M4 policy sends, so the probe
#: exercises the real JSON-mode path without touching any cohort content.
SYSTEM_PROMPT = (
    "You are a decision policy. Reply with exactly one JSON object and nothing else, of the form "
    '{"action": "<name>", "rationale": "<short reason>"}.'
)
CONTEXT = {
    "user_request": "probe: choose an action",
    "available_actions": ["select_source", "finish"],
    "budget": {"remaining_tool_calls": 4},
}
ACTION_SCHEMA = (
    {"action": "select_source", "arguments": [{"name": "source"}], "required_arguments": ["source"]},
    {"action": "finish", "arguments": [], "required_arguments": []},
)


def _one_call(client: Any) -> dict[str, Any]:
    """Make one provider call and report its outcome without interpreting the content."""
    from recommendation.control.model_client import ModelRequest

    request = ModelRequest(
        system_prompt=SYSTEM_PROMPT,
        context_payload=CONTEXT,
        action_schema=ACTION_SCHEMA,
    )
    started = time.time()
    try:
        response = client.complete(request)
    except Exception as exc:  # noqa: BLE001 - the probe records failures, it does not raise
        return {
            "ok": False,
            "error": f"{type(exc).__name__}: {str(exc)[:160]}",
            "code": getattr(exc, "code", None),
            "seconds": time.time() - started,
        }
    text = (response.text or "").strip()
    parsed = False
    parsed_action = None
    try:
        decoded = json.loads(text)
        parsed = isinstance(decoded, dict) and isinstance(decoded.get("action"), str)
        parsed_action = decoded.get("action") if isinstance(decoded, dict) else None
    except Exception:  # noqa: BLE001
        parsed = False
    return {
        "ok": True,
        "seconds": time.time() - started,
        "model_id": response.model_id,
        "input_tokens": response.input_tokens,
        "output_tokens": response.output_tokens,
        "schema_ok": parsed,
        "schema_action": parsed_action,
        "text_len": len(text),
    }


def probe_level(client: Any, concurrency: int, calls: int) -> dict[str, Any]:
    """Run ``calls`` calls at ``concurrency`` and summarise latency, throughput and failures."""
    started = time.time()
    results: list[dict[str, Any]] = []
    if concurrency == 1:
        for _ in range(calls):
            results.append(_one_call(client))
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
            futures = [pool.submit(_one_call, client) for _ in range(calls)]
            for future in concurrent.futures.as_completed(futures):
                results.append(future.result())
    elapsed = time.time() - started

    ok = [r for r in results if r["ok"]]
    latencies = [r["seconds"] for r in ok]
    failures = [r for r in results if not r["ok"]]
    rate_limited = sum(
        1 for r in failures if "429" in str(r.get("error", "")) or "rate" in str(r.get("error", "")).lower()
    )
    return {
        "concurrency": concurrency,
        "calls": calls,
        "wall_seconds": round(elapsed, 3),
        "successes": len(ok),
        "failures": len(failures),
        "rate_limited_429": rate_limited,
        "transport_errors": len(failures) - rate_limited,
        "schema_ok": sum(1 for r in ok if r.get("schema_ok")),
        "schema_failures": sum(1 for r in ok if not r.get("schema_ok")),
        "calls_per_second": round(len(results) / elapsed, 3) if elapsed else None,
        "latency_mean_ms": round(1000 * statistics.mean(latencies), 1) if latencies else None,
        "latency_min_ms": round(1000 * min(latencies), 1) if latencies else None,
        "latency_max_ms": round(1000 * max(latencies), 1) if latencies else None,
        "total_input_tokens": sum(int(r.get("input_tokens") or 0) for r in ok),
        "total_output_tokens": sum(int(r.get("output_tokens") or 0) for r in ok),
        "failure_details": [r.get("error") for r in failures][:5],
        "schema_actions": sorted({r.get("schema_action") for r in ok if r.get("schema_action")}),
    }


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="M4 provider concurrency probe")
    parser.add_argument("--out", type=str, default="")
    args = parser.parse_args(argv)

    if not load_local_credential():
        print("PROVIDER BLOCKED: no credential available", flush=True)
        return 2

    from recommendation.control.provider_adapter import build_provider_client, provider_settings

    settings = provider_settings()
    # Frozen configuration: temperature and max_tokens come from the preregistration, not from here.
    client = build_provider_client(temperature=0.0, max_tokens=512)
    print("provider:", settings.describe(), flush=True)

    levels: list[dict[str, Any]] = []
    total = 0
    for concurrency, calls in PLAN:
        if total + calls > 16:
            break
        total += calls
        result = probe_level(client, concurrency, calls)
        levels.append(result)
        print(
            f"  {concurrency}-way: calls={result['calls']} ok={result['successes']} "
            f"fail={result['failures']} 429={result['rate_limited_429']} "
            f"schema_ok={result['schema_ok']}/{result['successes']} "
            f"mean={result['latency_mean_ms']}ms rate={result['calls_per_second']}/s",
            flush=True,
        )

    # Selection rule, fixed before looking: the highest concurrency with zero failures of any kind
    # and full schema compliance. Falling back to 1-way is the safe default, not a guess.
    selected = 1
    for result in levels:
        if result["failures"] == 0 and result["schema_failures"] == 0:
            selected = max(selected, result["concurrency"])
    payload = {
        "provider": settings.describe(),
        "frozen_config": {"temperature": 0.0, "max_tokens": 512, "json_mode": settings.json_mode},
        "total_calls": total,
        "levels": levels,
        "selected_concurrency": selected,
        "selection_rule": (
            "highest concurrency with zero failures and zero schema failures; 1-way otherwise"
        ),
    }
    print(f"\nselected concurrency: {selected}-way  (total calls: {total})", flush=True)
    if args.out:
        pathlib.Path(args.out).write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"wrote {args.out}", flush=True)
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI
    raise SystemExit(main())
